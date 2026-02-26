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
from .lgbm_model import (
    LGBMSignalModel,                           # Eğitilmiş model
    HAS_LIGHTGBM,                              # LightGBM var mı?
)

logger = logging.getLogger(__name__)


# =============================================================================
# SABİTLER
# =============================================================================

# Bootstrap parametreleri
BOOTSTRAP_N_ITERATIONS = 500                   # Bootstrap tekrar sayısı (500 = hız/doğruluk dengesi)
BOOTSTRAP_SAMPLE_RATIO = 0.8                   # Her iterasyonda kullanılan sample oranı
BOOTSTRAP_CI_LEVEL = 0.90                      # Güven aralığı seviyesi (%90)

# Regime filtreleme
REGIME_PENALTIES = {                           # Rejime göre güven çarpanı
    'trending_up': 1.00,                       # Trend yukarı → tam güven
    'trending_down': 1.00,                     # Trend aşağı → tam güven
    'trending': 1.00,                          # Genel trend → tam güven
    'ranging': 0.80,                           # Yatay → %20 penaltı (sahte sinyaller artar)
    'volatile': 0.70,                          # Yüksek volatilite → %30 penaltı (noise artar)
    'transitioning': 0.85,                     # Geçiş → %15 penaltı (belirsiz dönem)
    'unknown': 0.90,                           # Bilinmiyor → hafif penaltı
}

# Anomaly detection
FEATURE_ZSCORE_THRESHOLD = 3.5                 # |z| > 3.5 → anomali (6σ olayı seviyesinde nadir)
MAX_ANOMALY_RATIO = 0.25                       # Feature'ların %25'inden fazlası anomaliyse → red

# Ensemble agreement
IC_MODEL_AGREEMENT_BONUS = 1.10                # IC ve model aynı yönde → %10 güven bonusu
IC_MODEL_DISAGREE_PENALTY = 0.80               # IC ve model farklı yönde → %20 güven penaltisi


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
    regime: str = "unknown"                    
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
        regime: str = "unknown",
    ) -> ValidationResult:
        """Sinyali 4 istatistiksel kontrolle doğrula."""
        result = ValidationResult(
            original_confidence=model_confidence,
            ic_direction=ic_direction,
            model_direction=model_decision.value,
            regime=regime,
        )

        # ── 1. Bootstrap CI ──
        self._check_bootstrap_stability(model, feature_vector, result)

        # ── 2. Regime Filter ──
        self._check_regime(regime, result)

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
        """Bootstrap ile tahmin stabilitesini ölçer."""
        if not HAS_LIGHTGBM or model is None or not model.is_trained:
            result.bootstrap_passed = True
            return

        try:
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

            base_values = np.array(base_values)
            predictions = np.zeros(self.n_bootstrap)
            rng = np.random.default_rng(42)

            noise_scale = np.std(base_values) if np.std(base_values) > 0 else 1e-4

            for i in range(self.n_bootstrap):
                noise = rng.normal(0, noise_scale * 0.1, size=len(base_values))
                perturbed = base_values + noise
                X_perturbed = pd.DataFrame([perturbed], columns=model.feature_names)

                if not HAS_LIGHTGBM and hasattr(model, '_impute_median'):
                    X_perturbed = X_perturbed.fillna(model._impute_median)

                if model.calibrator is not None:
                    prob = model.calibrator.predict_proba(X_perturbed)[0][1]
                else:
                    prob = model.model.predict_proba(X_perturbed)[0][1]

                predictions[i] = prob

            lower_bound = float(np.percentile(predictions, 5))
            upper_bound = float(np.percentile(predictions, 95))
            ci_width = upper_bound - lower_bound

            result.bootstrap_ci_lower = lower_bound
            result.bootstrap_ci_upper = upper_bound
            result.bootstrap_ci_width = ci_width
            
            contains_50 = (lower_bound <= 0.50 <= upper_bound)
            
            if ci_width > 0.25 or contains_50:
                result.bootstrap_passed = False
                if ci_width > 0.25:
                    result.veto_reasons.append(f"Bootstrap CI geniş: {ci_width:.3f} > 0.25 → tahmin belirsiz")
                if contains_50:
                    result.veto_reasons.append(f"Bootstrap CI 0.50'yi kapsıyor: [{lower_bound:.2f}, {upper_bound:.2f}] → kararsız yön")
            else:
                result.bootstrap_passed = True

        except Exception as e:
            logger.warning(f"⚠️ Bootstrap hatası: {e}. Kontrol atlanıyor.")
            result.bootstrap_passed = True     

    def _check_regime(self, regime: str, result: ValidationResult) -> None:
        """Piyasa rejimine göre güven düzeltmesi."""
        penalty = self.regime_penalties.get(regime, 0.90)  
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
        elif result.bootstrap_ci_width < 0.30:
            bootstrap_factor = 0.90            
        else:
            bootstrap_factor = 0.80            

        anomaly_factor = 1.0 - (result.anomaly_ratio * 0.5)  

        adjusted = (
            base_confidence
            * result.regime_penalty            
            * result.agreement_multiplier      
            * bootstrap_factor                 
            * anomaly_factor                   
        )

        result.adjusted_confidence = round(max(0, min(100, adjusted)), 1)

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