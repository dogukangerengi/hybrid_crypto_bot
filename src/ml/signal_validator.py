# =============================================================================
# STATISTICAL SIGNAL VALIDATOR — İSTATİSTİKSEL SİNYAL DOĞRULAMA
# =============================================================================
# Amaç: LightGBM tahminini istatistiksel testlerle doğrulamak.
#        Model "karlı" dese bile, istatistiksel koşullar uygun değilse
#        sinyali reddetmek veya güven skorunu düşürmek.
#
# Neden Gerekli?
# - LightGBM tek bir olasılık tahmini verir ama BELİRSİZLİK ölçmez
# - Bootstrap CI: "Bu tahmin ne kadar stabil?" sorusunu cevaplar
# - Regime Filter: "Bu piyasa koşullarında model güvenilir mi?" sorusunu cevaplar
# - Anomaly Detection: "Bu sinyal geçmiş verilere göre normal mi?" sorusunu cevaplar
#
# Pipeline Konumu:
#   IC Analysis → Feature Eng → LightGBM Predict → [VALIDATOR] → Execution
#                                                     ↑ BURASI
#
# Validator'ın 4 Kontrolü:
# ┌──────────────────────┬──────────────────────────────────────────────┐
# │ Kontrol              │ Ne Yapar?                                    │
# ├──────────────────────┼──────────────────────────────────────────────┤
# │ 1. Bootstrap CI      │ Tahmin belirsizliğini ölçer (dar CI = stabil)│
# │ 2. Regime Filter     │ Piyasa rejimi uygunluğunu kontrol eder      │
# │ 3. Ensemble Agreement│ IC yönü ile model yönü uyuşuyor mu?        │
# │ 4. Feature Anomaly   │ Feature değerleri normal aralıkta mı?      │
# └──────────────────────┴──────────────────────────────────────────────┘
#
# Kullanım:
#   from ml.signal_validator import SignalValidator
#   validator = SignalValidator()
#   result = validator.validate(features, model, ic_direction, regime)
# =============================================================================

import logging                                 # Yapılandırılmış log mesajları
import numpy as np                             # Sayısal hesaplamalar
import pandas as pd                            # DataFrame işlemleri
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfları
from typing import Dict, List, Optional, Tuple, Any  # Tip belirteçleri
from datetime import datetime, timezone        # UTC zaman damgası

# Proje içi importlar
from .feature_engineer import (
    MLDecision,                                # LONG/SHORT/WAIT enum
    MLDecisionResult,                          # Nihai karar objesi
    MLFeatureVector,                           # Feature vektörü
)
# [SORUN 8 DÜZELTMESİ] — lgbm_model.py kaldırıldı, ensemble_model'dan al
from .ensemble_model import EnsemblePredictor as LGBMSignalModel  # noqa: F401
try:
    import lightgbm  # noqa
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


# =============================================================================
# SABİTLER
# =============================================================================

# Bootstrap parametreleri
BOOTSTRAP_N_ITERATIONS = 500                   # Bootstrap tekrar sayısı (500 = hız/doğruluk dengesi)
BOOTSTRAP_SAMPLE_RATIO = 0.8                   # Her iterasyonda kullanılan sample oranı
BOOTSTRAP_CI_LEVEL = 0.90                      # Güven aralığı seviyesi (%90)

# Regime filtreleme (unknown çıkmasın diye default'u ranging gibi cezalandırıyoruz)
REGIME_PENALTIES = {
    'trending_up': 1.00,
    'trending_down': 1.00,
    'trending': 1.00,
    'ranging': 0.85,
    'volatile': 0.75,
    'transitioning': 0.90,
    'unknown': 0.85, # Bilinmeyen = Yatay varsayımı
}

# Anomaly detection
FEATURE_ZSCORE_THRESHOLD = 3.5                 # |z| > 3.5 → anomali (6σ olayı seviyesinde nadir)
MAX_ANOMALY_RATIO = 0.25                       # Feature'ların %25'inden fazlası anomaliyse → red

# Ensemble agreement
IC_MODEL_AGREEMENT_BONUS = 1.10                # IC ve model aynı yönde → %10 güven bonusu
IC_MODEL_DISAGREE_PENALTY = 0.85               # IC ve model farklı yönde → %15 penaltı

# ── COLD START PARAMETRELERİ ──
# Model henüz yeterli kapalı trade görmemişken bootstrap veto'yu
# devre dışı bırakır. Bu sayede model ilk trade'leri açabilir,
# öğrenecek veri biriktirir. Eşiğe ulaşınca veto otomatik devreye girer.
COLD_START_MIN_TRADES = 20                     # Bu kadar kapalı trade birikene kadar veto devre dışı


# =============================================================================
# DOĞRULAMA SONUÇ DATACLASS
# =============================================================================

@dataclass
class ValidationResult:
    """
    Sinyal doğrulama sonucu.
    """
    is_valid: bool = True                      
    adjusted_confidence: float = 0.0           
    original_confidence: float = 0.0           

    # Bootstrap CI
    bootstrap_mean: float = 0.0                
    bootstrap_std: float = 0.0                 
    bootstrap_ci_lower: float = 0.0            
    bootstrap_ci_upper: float = 1.0            
    bootstrap_ci_width: float = 1.0            
    bootstrap_passed: bool = True              

    # Regime Filter
    regime: str = "ranging" # Varsayılan değer 'unknown' yerine 'ranging' yapıldı             
    regime_penalty: float = 1.0                
    regime_passed: bool = True                 

    # Ensemble Agreement
    ic_direction: str = "NEUTRAL"              
    model_direction: str = "WAIT"              
    directions_agree: bool = True              
    agreement_multiplier: float = 1.0          
    ensemble_passed: bool = True               

    # Feature Anomaly
    n_anomalies: int = 0                       
    anomaly_ratio: float = 0.0                 
    anomaly_features: List[str] = field(default_factory=list)  
    anomaly_passed: bool = True                

    # Veto sebepleri
    veto_reasons: List[str] = field(default_factory=list)

    # Cold start bilgisi (log/debug için)
    cold_start_bypass: bool = False            # True ise veto cold start nedeniyle atlandı

    def summary(self) -> str:
        """Telegram / log için okunabilir özet."""
        status = "✅ Geçerli" if self.is_valid else "❌ Reddedildi"
        parts = [
            f"🔬 Validator: {status}",
            f"  Güven: {self.original_confidence:.0f} → {self.adjusted_confidence:.0f}",
            f"  Bootstrap CI: [{self.bootstrap_ci_lower:.2f}, {self.bootstrap_ci_upper:.2f}] "
            f"(width={self.bootstrap_ci_width:.3f})",
            f"  Rejim: {self.regime} (×{self.regime_penalty:.2f})",
            f"  Uyum: IC={self.ic_direction} Model={self.model_direction} "
            f"({'✓' if self.directions_agree else '✗'})",
        ]
        if self.cold_start_bypass:
            parts.append(f"  ⚡ Cold start: Bootstrap veto atlandı")
        if self.n_anomalies > 0:
            parts.append(f"  Anomali: {self.n_anomalies} feature ({self.anomaly_ratio:.0%})")
        if self.veto_reasons:
            parts.append(f"  Veto: {'; '.join(self.veto_reasons)}")
        return "\n".join(parts)


# =============================================================================
# ANA VALIDATOR SINIFI
# =============================================================================

class SignalValidator:
    """
    İstatistiksel sinyal doğrulayıcı.
    """

    def __init__(
        self,
        n_bootstrap: int = BOOTSTRAP_N_ITERATIONS,
        ci_level: float = BOOTSTRAP_CI_LEVEL,
        regime_penalties: Optional[Dict] = None,
        verbose: bool = True,
    ):
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.regime_penalties = regime_penalties or REGIME_PENALTIES
        self.verbose = verbose

        self._feature_means: Optional[pd.Series] = None
        self._feature_stds: Optional[pd.Series] = None
        self._feature_names: List[str] = []

        # ── Cold Start Sayacı ──
        # main.py her döngüde trade_memory'den kapalı trade sayısını
        # bu attribute'a yazar. Validator bunu okuyarak cold start
        # döneminde bootstrap veto'yu devre dışı bırakır.
        # 0 = henüz hiç kapalı trade yok (cold start aktif)
        self._closed_trade_count: int = 0

    def fit_train_stats(self, X_train: pd.DataFrame) -> None:
        """Eğitim verisinin feature istatistiklerini hesapla."""
        feature_cols = [c for c in X_train.columns if not c.startswith('_')]
        self._feature_means = X_train[feature_cols].mean()
        self._feature_stds = X_train[feature_cols].std()
        self._feature_names = feature_cols
        self._feature_stds = self._feature_stds.replace(0, 1.0)
        if self.verbose:
            logger.info(f"  📏 Train stats hesaplandı: {len(feature_cols)} feature")

    def validate(
        self,
        feature_vector: MLFeatureVector,
        model: Any,
        model_decision: MLDecision,
        model_confidence: float,
        ic_direction: str = "NEUTRAL",
        ic_score: float = 0.0,
        regime: str = "ranging", # Varsayılan ranging
    ) -> ValidationResult:
        """Sinyali 4 istatistiksel kontrolle doğrula."""
        
        # Son bir güvenlik kontrolü: Eğer bir sebepten 'unknown' veya None geldiyse ranging yap
        safe_regime = regime if regime and regime != "unknown" else "ranging"

        result = ValidationResult(
            original_confidence=model_confidence,
            ic_direction=ic_direction,
            model_direction=model_decision.value,
            regime=safe_regime,
        )

        # ── 1. Bootstrap CI ──
        self._check_bootstrap_stability(model, feature_vector, result)

        # ── 2. Regime Filter ──
        self._check_regime(safe_regime, result)

        # ── 3. Ensemble Agreement ──
        self._check_ensemble(model_decision, ic_direction, result)

        # ── 4. Feature Anomaly ──
        self._check_anomaly(feature_vector, result)

        # ── 5. Final Güven Hesaplama ──
        self._compute_final_confidence(model_confidence, result)

        if self.verbose:
            logger.info(f"\n{result.summary()}")

        return result

    def _check_bootstrap_stability(
        self,
        model: Any,
        feature_vector: MLFeatureVector,
        result: ValidationResult
    ) -> None:
        """
        Bootstrap ile tahmin stabilitesini ölçer.
        """
        # ── Guard: Model yoksa veya eğitilmemişse atla ──
        if not HAS_LIGHTGBM or model is None or not model.is_trained:
            result.bootstrap_passed = True     
            return

        try:
            # ── 1. Base feature vektörünü çıkar ──
            base_values = []                   
            for col in model.feature_names:
                val = 0.0                      
                if col in feature_vector.ic_features:
                    val = feature_vector.ic_features[col]
                elif col in feature_vector.market_features:
                    val = feature_vector.market_features[col]
                elif col in feature_vector.cross_tf_features:
                    val = feature_vector.cross_tf_features[col]
                elif col in feature_vector.price_features:
                    val = feature_vector.price_features[col]
                elif col in feature_vector.risk_features:
                    val = feature_vector.risk_features[col]
                elif col in feature_vector.temporal_features:
                    val = feature_vector.temporal_features[col]
                base_values.append(val)

            base_values = np.array(base_values, dtype=np.float64)

            # ── 2. Per-feature noise ölçeği hesapla ──
            MIN_NOISE_FLOOR = 0.05             

            n_features = len(base_values)      
            feature_noise_scales = np.zeros(n_features)  

            for idx, col in enumerate(model.feature_names):
                if (self._feature_stds is not None        
                    and col in self._feature_stds.index):  
                    train_std = self._feature_stds[col]
                    feature_noise_scales[idx] = max(train_std, MIN_NOISE_FLOOR)
                else:
                    feature_noise_scales[idx] = max(abs(base_values[idx]) * 0.20, MIN_NOISE_FLOOR)

            # ── 3. Bootstrap iterasyonları ──
            n_iter = min(self.n_bootstrap, 200)  
            predictions = np.zeros(n_iter)       

            rng = np.random.default_rng()        
            noise_fraction = 0.25

            for i in range(n_iter):
                noise = rng.normal(
                    loc=0.0,                           
                    scale=feature_noise_scales * noise_fraction,  
                    size=n_features                    
                )
                perturbed = base_values + noise        

                X_perturbed = pd.DataFrame(
                    [perturbed],                       
                    columns=model.feature_names        
                )

                if model.calibrator is not None:
                    prob = model.calibrator.predict_proba(X_perturbed)[0][1]
                else:
                    prob = model.model.predict_proba(X_perturbed)[0][1]

                predictions[i] = prob                  

            # ── 4. CI hesapla ──
            lower_bound = float(np.percentile(predictions, 5))    
            upper_bound = float(np.percentile(predictions, 95))   
            ci_width = upper_bound - lower_bound                  

            result.bootstrap_mean = float(np.mean(predictions))   
            result.bootstrap_std = float(np.std(predictions))     
            result.bootstrap_ci_lower = lower_bound
            result.bootstrap_ci_upper = upper_bound
            result.bootstrap_ci_width = ci_width

            # ── 5. Karar: CI kabul edilebilir mi? ──
            contains_50 = (lower_bound <= 0.50 <= upper_bound)

            if ci_width > 0.55:
                result.bootstrap_passed = False
                result.veto_reasons.append(
                    f"Bootstrap CI geniş: {ci_width:.3f} > 0.55 → tahmin belirsiz"
                )
            elif contains_50:
                result.bootstrap_passed = True
                logger.info(
                    f"  ℹ️ Bootstrap CI 0.50'yi kapsıyor [{lower_bound:.2f}, {upper_bound:.2f}] "
                    f"ama width={ci_width:.3f} kabul edilebilir"
                )
            else:
                result.bootstrap_passed = True

        except Exception as e:
            logger.warning(f"⚠️ Bootstrap hatası: {e}. Kontrol atlanıyor.")
            result.bootstrap_passed = True             


    def _check_regime(self, regime: str, result: ValidationResult) -> None:
        """Piyasa rejimine göre güven düzeltmesi."""
        # Regime zaten validator girişinde güvenli hale getirildiği için direkt alıyoruz
        penalty = self.regime_penalties.get(regime, 0.85) # Bilinmeyen bir şey gelirse 0.85 ceza
        result.regime_penalty = penalty

        if regime == 'volatile' and penalty < 0.75:
            result.regime_passed = False
            result.veto_reasons.append(f"Yüksek volatilite rejimi: penaltı ×{penalty:.2f}")
        else:
            result.regime_passed = True

    def _check_ensemble(self, model_decision: MLDecision, ic_direction: str, result: ValidationResult) -> None:
        """IC yönü ile model kararının uyumunu kontrol et."""
        model_dir = model_decision.value       

        if ic_direction == "NEUTRAL" or model_dir == "WAIT":
            result.directions_agree = True
            result.agreement_multiplier = 1.0
            result.ensemble_passed = True
            return

        if ic_direction == model_dir:
            result.directions_agree = True
            result.agreement_multiplier = IC_MODEL_AGREEMENT_BONUS
            result.ensemble_passed = True
        else:
            result.directions_agree = False
            result.agreement_multiplier = IC_MODEL_DISAGREE_PENALTY
            result.ensemble_passed = True      

    def _check_anomaly(self, feature_vector: MLFeatureVector, result: ValidationResult) -> None:
        """Feature değerlerinin eğitim dağılımına göre anomali kontrolü."""
        if self._feature_means is None or self._feature_stds is None:
            result.anomaly_passed = True
            return

        try:
            feature_dict = feature_vector.to_dict()  
            anomalies = []                     

            for fname in self._feature_names:
                value = feature_dict.get(fname)
                if value is None or np.isnan(value):
                    continue                   

                mean = self._feature_means.get(fname, 0)
                std = self._feature_stds.get(fname, 1)

                z = abs(value - mean) / std    
                if z > FEATURE_ZSCORE_THRESHOLD:
                    anomalies.append(fname)    

            n_checked = sum(1 for fn in self._feature_names if feature_dict.get(fn) is not None and not np.isnan(feature_dict.get(fn, float('nan'))))

            result.n_anomalies = len(anomalies)
            result.anomaly_features = anomalies[:5]  
            result.anomaly_ratio = len(anomalies) / max(n_checked, 1)

            if result.anomaly_ratio > MAX_ANOMALY_RATIO:
                result.anomaly_passed = False
                result.veto_reasons.append(f"Anomali oranı yüksek: {result.anomaly_ratio:.0%} > {MAX_ANOMALY_RATIO:.0%} ({len(anomalies)} feature)")
            else:
                result.anomaly_passed = True

        except Exception as e:
            logger.warning(f"⚠️ Anomaly check hatası: {e}. Kontrol atlanıyor.")
            result.anomaly_passed = True

    def _compute_final_confidence(self, base_confidence: float, result: ValidationResult) -> None:
        """Tüm kontrol sonuçlarını birleştirip final güven skoru hesapla."""
        
        if result.bootstrap_ci_width < 0.10:
            bootstrap_factor = 1.05
        elif result.bootstrap_ci_width < 0.20:
            bootstrap_factor = 1.00
        elif result.bootstrap_ci_width < 0.35:
            bootstrap_factor = 0.95
        elif result.bootstrap_ci_width < 0.50:
            bootstrap_factor = 0.90
        else:
            bootstrap_factor = 0.85            

        anomaly_factor = 1.0 - (result.anomaly_ratio * 0.5)  

        adjusted = (
            base_confidence
            * result.regime_penalty            
            * result.agreement_multiplier      
            * bootstrap_factor                 
            * anomaly_factor                   
        )

        result.adjusted_confidence = round(max(0, min(100, adjusted)), 1)

        if result.adjusted_confidence < 40.0:
            result.veto_reasons.append(f"Güvenlik barajı aşılamadı: %{result.adjusted_confidence:.1f} < %40.0")
            
        if result.veto_reasons:
            result.is_valid = False
        else:
            result.is_valid = True

    def get_regime_info(self) -> Dict[str, float]:
        """Aktif rejim penaltılarını döndür."""
        return dict(self.regime_penalties)

    def get_train_stats_summary(self) -> Dict[str, Any]:
        """Eğitim istatistikleri özeti."""
        if self._feature_means is None:
            return {"status": "not_fitted", "n_features": 0}
        return {
            "status": "fitted",
            "n_features": len(self._feature_names),
            "mean_range": f"[{self._feature_means.min():.2f}, {self._feature_means.max():.2f}]",
            "std_range": f"[{self._feature_stds.min():.2f}, {self._feature_stds.max():.2f}]",
        }
