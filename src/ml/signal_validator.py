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
    
    Her kontrol (bootstrap, regime, ensemble, anomaly) ayrı bir skor üretir.
    Final skor tüm kontrolların birleşimidir.
    """
    # Sonuç
    is_valid: bool = True                      # Sinyal geçerli mi? (tüm kontroller geçtiyse True)
    adjusted_confidence: float = 0.0           # Validator sonrası düzeltilmiş güven (0-100)
    original_confidence: float = 0.0           # Model'den gelen orijinal güven (0-100)

    # Bootstrap CI
    bootstrap_mean: float = 0.0                # Bootstrap tahmin ortalaması
    bootstrap_std: float = 0.0                 # Bootstrap tahmin standart sapması
    bootstrap_ci_lower: float = 0.0            # CI alt sınır (5. persentil @90%)
    bootstrap_ci_upper: float = 1.0            # CI üst sınır (95. persentil @90%)
    bootstrap_ci_width: float = 1.0            # CI genişliği (dar = stabil tahmin)
    bootstrap_passed: bool = True              # CI kontrolü geçti mi?

    # Regime Filter
    regime: str = "unknown"                    # Tespit edilen piyasa rejimi
    regime_penalty: float = 1.0                # Rejim penaltı çarpanı (0-1)
    regime_passed: bool = True                 # Rejim kontrolü geçti mi?

    # Ensemble Agreement
    ic_direction: str = "NEUTRAL"              # IC'nin önerdiği yön
    model_direction: str = "WAIT"              # Model'in önerdiği yön
    directions_agree: bool = True              # Yönler uyuşuyor mu?
    agreement_multiplier: float = 1.0          # Uyum çarpanı (bonus veya penaltı)
    ensemble_passed: bool = True               # Ensemble kontrolü geçti mi?

    # Feature Anomaly
    n_anomalies: int = 0                       # Anomali tespit edilen feature sayısı
    anomaly_ratio: float = 0.0                 # Anomali oranı (n_anomalies / total_features)
    anomaly_features: List[str] = field(default_factory=list)  # Anomalili feature isimleri
    anomaly_passed: bool = True                # Anomali kontrolü geçti mi?

    # Veto sebepleri (hangi kontrol(ler) başarısız oldu)
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
    
    LightGBM tahminini 4 bağımsız kontrolle doğrular:
    1. Bootstrap CI → tahmin stabilitesi
    2. Regime Filter → piyasa uygunluğu
    3. Ensemble Agreement → IC-model uyumu
    4. Feature Anomaly → girdi kalitesi
    
    Her kontrol bağımsız olarak sinyali "veto" edebilir veya
    güven skorunu düşürebilir / artırabilir.
    """

    def __init__(
        self,
        n_bootstrap: int = BOOTSTRAP_N_ITERATIONS,   # Bootstrap tekrar sayısı
        ci_level: float = BOOTSTRAP_CI_LEVEL,         # Güven aralığı seviyesi
        regime_penalties: Optional[Dict] = None,       # Özel rejim penaltıları
        verbose: bool = True,                          # Detaylı log
    ):
        """
        SignalValidator başlatır.
        
        Parameters:
        ----------
        n_bootstrap : int
            Bootstrap tekrar sayısı. 500 = ~50ms, 1000 = ~100ms.
            Daha fazla = daha dar CI ama daha yavaş.
            
        ci_level : float
            Güven aralığı seviyesi. 0.90 = %90 CI (5.-95. persentil).
            0.95 daha geniş CI → daha muhafazakar.
            
        regime_penalties : Dict, optional
            Rejim bazlı güven çarpanları. None → varsayılan kullanılır.
            
        verbose : bool
            True → kontrol detaylarını logla.
        """
        self.n_bootstrap = n_bootstrap                 # Bootstrap iterasyon sayısı
        self.ci_level = ci_level                       # CI seviyesi
        self.regime_penalties = regime_penalties or REGIME_PENALTIES  # Rejim penaltıları
        self.verbose = verbose                         # Log detay seviyesi

        # Eğitim verisi istatistikleri (anomaly detection için)
        # fit_train_stats() çağrılana kadar None → anomaly check atlanır
        self._feature_means: Optional[pd.Series] = None   # Feature ortalamaları
        self._feature_stds: Optional[pd.Series] = None    # Feature standart sapmaları
        self._feature_names: List[str] = []                # Feature isimleri

    # =========================================================================
    # EĞİTİM VERİSİ İSTATİSTİKLERİ (ANOMALY DETECTION İÇİN)
    # =========================================================================

    def fit_train_stats(self, X_train: pd.DataFrame) -> None:
        """
        Eğitim verisinin feature istatistiklerini hesapla.
        
        Bu istatistikler anomaly detection'da referans olarak kullanılır.
        Yeni bir sinyal geldiğinde feature değerleri eğitim dağılımıyla
        karşılaştırılır → aşırı sapma = anomali.
        
        Neden Gerekli?
        - Model eğitim verisinin dağılımını öğrenir
        - Eğitim dağılımının dışındaki verilerle tahmin güvenilir değildir
        - Örnek: ic_confidence eğitimde [30-90] arasıysa ve yeni sinyal 5 gelirse
          model bu bölgeyi hiç görmemiş → tahmin güvenilmez
        
        Parameters:
        ----------
        X_train : pd.DataFrame
            Eğitim feature matrix'i (meta kolonlar (_) dahil olabilir, filtrelenir)
        """
        # Meta kolonları filtrele
        feature_cols = [c for c in X_train.columns if not c.startswith('_')]

        self._feature_means = X_train[feature_cols].mean()   # Her feature'ın ortalaması
        self._feature_stds = X_train[feature_cols].std()     # Her feature'ın std'si
        self._feature_names = feature_cols                    # Feature isimleri

        # std=0 olan feature'lar (sabit değer) → 1.0 ile değiştir (sıfıra bölme koruması)
        self._feature_stds = self._feature_stds.replace(0, 1.0)

        if self.verbose:
            logger.info(
                f"  📏 Train stats hesaplandı: {len(feature_cols)} feature"
            )

    # =========================================================================
    # ANA DOĞRULAMA FONKSİYONU
    # =========================================================================

    def validate(
        self,
        feature_vector: MLFeatureVector,       # Adım 1'deki feature vektörü
        model: LGBMSignalModel,                # Eğitilmiş LightGBM model
        model_decision: MLDecision,            # Model'in kararı (LONG/SHORT/WAIT)
        model_confidence: float,               # Model'in güven skoru (0-100)
        ic_direction: str = "NEUTRAL",         # IC'nin önerdiği yön
        ic_score: float = 0.0,                 # IC composite skoru
        regime: str = "unknown",               # Tespit edilen piyasa rejimi
    ) -> ValidationResult:
        """
        Sinyali 4 istatistiksel kontrolle doğrula.
        
        Pipeline:
        1. Bootstrap CI → tahmin stabilitesini ölç
        2. Regime Filter → piyasa rejimi uygunluğunu kontrol et
        3. Ensemble Agreement → IC-model yön uyumunu kontrol et
        4. Feature Anomaly → girdi kalitesini kontrol et
        5. Sonuçları birleştir → final adjusted_confidence hesapla
        
        Parameters:
        ----------
        feature_vector : MLFeatureVector
            Yeni sinyalin feature vektörü
            
        model : LGBMSignalModel
            Eğitilmiş model (bootstrap için predict_proba gerekli)
            
        model_decision : MLDecision
            Model'in önerdiği karar (LONG/SHORT/WAIT)
            
        model_confidence : float
            Model'in güven skoru (0-100)
            
        ic_direction : str
            IC analizinin önerdiği yön ('LONG'/'SHORT'/'NEUTRAL')
            
        ic_score : float
            IC composite skoru (0-100)
            
        regime : str
            Piyasa rejimi ('trending_up', 'ranging', 'volatile', vb.)
            
        Returns:
        -------
        ValidationResult
            Doğrulama sonucu (is_valid, adjusted_confidence, kontrol detayları)
        """
        result = ValidationResult(
            original_confidence=model_confidence,
            ic_direction=ic_direction,
            model_direction=model_decision.value,
            regime=regime,
        )

        # ── 1. Bootstrap CI ──
        self._check_bootstrap(feature_vector, model, result)

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

    # =========================================================================
    # 1. BOOTSTRAP CONFIDENCE INTERVAL
    # =========================================================================

    def _check_bootstrap(
        self,
        feature_vector: MLFeatureVector,
        model: LGBMSignalModel,
        result: ValidationResult,
    ) -> None:
        """
        Bootstrap ile tahmin belirsizliğini ölç.
        
        Yöntem:
        - Feature vektörüne küçük Gaussian noise ekle (N iterasyon)
        - Her perturbed versiyon için model tahmini al
        - Tahminlerin dağılımından CI hesapla
        
        Neden Gaussian Noise?
        - Gerçek dünyada feature değerleri noise'lu (ölçüm hatası, gecikme, vb.)
        - Noise'a karşı stabil tahmin = güvenilir sinyal
        - CI dar → model emin, CI geniş → model emin değil
        
        Not: Bu "gerçek" bootstrap değil (resample yerine perturbation).
        Gerçek bootstrap için çok sayıda geçmiş veri gerekir.
        Perturbation bootstrap aynı zamanda model robustness testi de yapar.
        """
        # Model eğitilmemişse bootstrap atlansın
        if not model.is_trained or model.model is None:
            result.bootstrap_passed = True     # Kontrol atlandı → geçti say
            return

        try:
            # Feature'ları numpy array'e çevir
            feature_dict = feature_vector.to_dict()                # Dict: {name: value}
            X_base = pd.DataFrame([feature_dict])[model.feature_names]  # Sıralı DataFrame
            base_values = X_base.values[0]     # 1D numpy array

            # Feature std'leri (noise büyüklüğü için)
            if self._feature_stds is not None:
                noise_scale = self._feature_stds.reindex(model.feature_names).fillna(0.1).values
            else:
                # Eğitim istatistikleri yoksa → feature değerinin %5'i kadar noise
                noise_scale = np.abs(base_values) * 0.05 + 1e-6

            # Bootstrap iterasyonları
            rng = np.random.RandomState(42)    # Tekrarlanabilirlik
            predictions = np.zeros(self.n_bootstrap)  # Tahmin deposu

            for i in range(self.n_bootstrap):
                # Gaussian noise ekle (her feature'a kendi scale'inde)
                noise = rng.normal(0, noise_scale * 0.3, size=len(base_values))

                perturbed = base_values + noise
                X_perturbed = pd.DataFrame([perturbed], columns=model.feature_names)

                # NaN handling (sklearn fallback)
                if not HAS_LIGHTGBM and hasattr(model, '_impute_median'):
                    X_perturbed = X_perturbed.fillna(model._impute_median)

                # Tahmin al
                if model.calibrator is not None:
                    prob = model.calibrator.predict_proba(X_perturbed)[0][1]
                else:
                    prob = model.model.predict_proba(X_perturbed)[0][1]

                predictions[i] = prob

            # CI hesapla
            alpha = (1 - self.ci_level) / 2    # Tek kuyruk alpha (0.05 for 90% CI)
            ci_lower = float(np.percentile(predictions, alpha * 100))        # 5. persentil
            ci_upper = float(np.percentile(predictions, (1 - alpha) * 100))  # 95. persentil
            ci_width = ci_upper - ci_lower     # CI genişliği

            result.bootstrap_mean = float(np.mean(predictions))
            result.bootstrap_std = float(np.std(predictions))
            result.bootstrap_ci_lower = ci_lower
            result.bootstrap_ci_upper = ci_upper
            result.bootstrap_ci_width = ci_width

            # Kontrol: CI genişliği.
            # width > 0.25 → model belirsiz → veto.
            # width <= 0.25 → model kararlı → passed.
            # Not: Dar CI (width~0.00) LightGBM stabilitesini gösterir,
            # penalize edilmemeli.
            if ci_width > 0.25:
                result.bootstrap_passed = False
                result.veto_reasons.append(
                    f"Bootstrap CI geniş: {ci_width:.3f} > 0.25 → tahmin belirsiz"
                )
            else:
                result.bootstrap_passed = True

        except Exception as e:
            logger.warning(f"⚠️ Bootstrap hatası: {e}. Kontrol atlanıyor.")
            result.bootstrap_passed = True     # Hata → kontrol atla

    # =========================================================================
    # 2. REGIME FILTER
    # =========================================================================

    def _check_regime(self, regime: str, result: ValidationResult) -> None:
        """
        Piyasa rejimine göre güven düzeltmesi.
        
        Neden Rejim Önemli?
        - Trend modelleri range piyasasında kötü çalışır (sahte sinyaller)
        - Volatil dönemde SL daha kolay tetiklenir
        - ADX bazlı rejim tespiti mevcut pipeline'da zaten var
        
        Rejim Penaltıları:
        - trending: ×1.00 (tam güven — model trending verilerle eğitildi)
        - ranging:  ×0.80 (düşük güven — mean reversion dominant)
        - volatile: ×0.70 (çok düşük güven — noise seviyesi yüksek)
        - transitioning: ×0.85 (orta güven — rejim değişiyor)
        """
        penalty = self.regime_penalties.get(regime, 0.90)  # Bilinmeyen rejim → 0.90
        result.regime_penalty = penalty

        # Volatile rejimde güven çok düşükse → veto
        if regime == 'volatile' and penalty < 0.75:
            result.regime_passed = False
            result.veto_reasons.append(
                f"Yüksek volatilite rejimi: penaltı ×{penalty:.2f}"
            )
        else:
            result.regime_passed = True

    # =========================================================================
    # 3. ENSEMBLE AGREEMENT (IC - MODEL UYUMU)
    # =========================================================================

    def _check_ensemble(
        self,
        model_decision: MLDecision,
        ic_direction: str,
        result: ValidationResult,
    ) -> None:
        """
        IC yönü ile model kararının uyumunu kontrol et.
        
        Uyum Senaryoları:
        ┌──────────┬───────────┬──────────────────────┐
        │ IC       │ Model     │ Sonuç                │
        ├──────────┼───────────┼──────────────────────┤
        │ LONG     │ LONG      │ ×1.10 bonus (uyumlu) │
        │ SHORT    │ SHORT     │ ×1.10 bonus (uyumlu) │
        │ LONG     │ SHORT     │ ×0.80 penaltı (çelişki)│
        │ SHORT    │ LONG      │ ×0.80 penaltı (çelişki)│
        │ NEUTRAL  │ herhangi  │ ×1.00 (nötr)         │
        │ herhangi │ WAIT      │ ×1.00 (nötr)         │
        └──────────┴───────────┴──────────────────────┘
        """
        model_dir = model_decision.value       # Enum → string

        # Nötr durumlar → bonus/penaltı yok
        if ic_direction == "NEUTRAL" or model_dir == "WAIT":
            result.directions_agree = True
            result.agreement_multiplier = 1.0
            result.ensemble_passed = True
            return

        # Yön karşılaştırması
        if ic_direction == model_dir:
            # Aynı yön → bonus
            result.directions_agree = True
            result.agreement_multiplier = IC_MODEL_AGREEMENT_BONUS
            result.ensemble_passed = True
        else:
            # Farklı yön → penaltı
            result.directions_agree = False
            result.agreement_multiplier = IC_MODEL_DISAGREE_PENALTY
            result.ensemble_passed = True      # Veto etme, sadece penaltı ver

    # =========================================================================
    # 4. FEATURE ANOMALY DETECTION
    # =========================================================================

    def _check_anomaly(
        self,
        feature_vector: MLFeatureVector,
        result: ValidationResult,
    ) -> None:
        """
        Feature değerlerinin eğitim dağılımına göre anomali kontrolü.
        
        Yöntem: Z-Score bazlı outlier detection
        - Her feature için z = (value - mean) / std hesapla
        - |z| > 3.5 → anomali (eğitim verisinde görülmemiş aşırı değer)
        
        Neden Z-Score 3.5?
        - Normal dağılımda P(|z| > 3.5) ≈ 0.00047 (%0.05)
        - Bu seviye "gerçekten nadir" anlamına gelir
        - 3.0 çok agresif (çok fazla false alarm), 4.0 çok gevşek
        
        Anomali Eşiği:
        - Feature'ların %25'inden fazlası anomaliyse → veto
        - %25 altında → uyarı ama geçer
        """
        # Eğitim istatistikleri yoksa → anomaly check atlansın
        if self._feature_means is None or self._feature_stds is None:
            result.anomaly_passed = True
            return

        try:
            feature_dict = feature_vector.to_dict()  # Feature dict
            anomalies = []                     # Anomalili feature listesi

            for fname in self._feature_names:
                value = feature_dict.get(fname)
                if value is None or np.isnan(value):
                    continue                   # NaN → skip (model handle eder)

                mean = self._feature_means.get(fname, 0)
                std = self._feature_stds.get(fname, 1)

                z = abs(value - mean) / std    # Z-score hesapla
                if z > FEATURE_ZSCORE_THRESHOLD:
                    anomalies.append(fname)    # Anomali tespit edildi

            n_checked = sum(
                1 for fn in self._feature_names
                if feature_dict.get(fn) is not None
                and not np.isnan(feature_dict.get(fn, float('nan')))
            )

            result.n_anomalies = len(anomalies)
            result.anomaly_features = anomalies[:5]  # İlk 5'i kaydet (log için)
            result.anomaly_ratio = len(anomalies) / max(n_checked, 1)

            # Eşik kontrolü
            if result.anomaly_ratio > MAX_ANOMALY_RATIO:
                result.anomaly_passed = False
                result.veto_reasons.append(
                    f"Anomali oranı yüksek: {result.anomaly_ratio:.0%} > {MAX_ANOMALY_RATIO:.0%} "
                    f"({len(anomalies)} feature: {', '.join(anomalies[:3])})"
                )
            else:
                result.anomaly_passed = True

        except Exception as e:
            logger.warning(f"⚠️ Anomaly check hatası: {e}. Kontrol atlanıyor.")
            result.anomaly_passed = True

    # =========================================================================
    # 5. FİNAL GÜVEN HESAPLAMA
    # =========================================================================

    def _compute_final_confidence(
        self,
        original_confidence: float,
        result: ValidationResult,
    ) -> None:
        """
        Tüm kontrol sonuçlarını birleştirip final güven skoru hesapla.
        
        Formül:
          adjusted = original × regime_penalty × agreement_multiplier × bootstrap_factor × anomaly_factor
        
        bootstrap_factor: CI genişliğine göre ek düzeltme
        - CI < 0.10 → ×1.05 (çok dar CI = çok stabil tahmin)
        - CI 0.10-0.20 → ×1.00 (normal)
        - CI 0.20-0.30 → ×0.90 (geniş CI = belirsiz)
        - CI > 0.30 → ×0.80 (çok belirsiz)
        
        anomaly_factor: anomali oranı kadar güven düşür
        - %10 anomali → ×0.95
        - %20 anomali → ×0.90
        
        is_valid: TÜM kontroller geçtiyse True
        """
        # Bootstrap factor
        if result.bootstrap_ci_width < 0.10:
            bootstrap_factor = 1.05            # Çok stabil tahmin → küçük bonus
        elif result.bootstrap_ci_width < 0.20:
            bootstrap_factor = 1.00            # Normal stabilite
        elif result.bootstrap_ci_width < 0.30:
            bootstrap_factor = 0.90            # Belirsiz tahmin → penaltı
        else:
            bootstrap_factor = 0.80            # Çok belirsiz → büyük penaltı

        # Anomali factor
        anomaly_factor = 1.0 - (result.anomaly_ratio * 0.5)  # %20 anomali → ×0.90

        # Final güven hesaplama
        adjusted = (
            original_confidence
            * result.regime_penalty            # Rejim düzeltmesi
            * result.agreement_multiplier      # Ensemble uyum düzeltmesi
            * bootstrap_factor                 # Tahmin stabilite düzeltmesi
            * anomaly_factor                   # Girdi kalitesi düzeltmesi
        )

        # Güveni 0-100 aralığında sınırla
        result.adjusted_confidence = round(max(0, min(100, adjusted)), 1)

        # IC eşiğini geçmiş sinyallerde minimum güven garantisi.
        # Regime × bootstrap penaltıları geçerli sinyali sıfırlamasın.
        MIN_CONFIDENCE_FLOOR = 12.0
        result.adjusted_confidence = max(result.adjusted_confidence, MIN_CONFIDENCE_FLOOR)

        # Geçerlilik: tüm kontroller geçtiyse True
        result.is_valid = all([
            result.bootstrap_passed,
            result.regime_passed,
            result.ensemble_passed,
            result.anomaly_passed,
        ])

    # =========================================================================
    # YARDIMCI FONKSIYONLAR
    # =========================================================================

    def get_regime_info(self) -> Dict[str, float]:
        """Aktif rejim penaltılarını döndür (debug/log için)."""
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
