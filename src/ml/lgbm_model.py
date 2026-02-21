# =============================================================================
# LIGHTGBM WALK-FORWARD MODEL — EĞİTİM / TAHMİN / RETRAIN
# =============================================================================
# Amaç: Geçmiş trade sonuçlarından öğrenerek yeni sinyallerin karlılığını
#        tahmin etmek. Gemini'nin "semantik onay" rolünü veriye dayalı
#        olasılıksal tahminle değiştirmek.
#
# Gemini vs LightGBM:
# ┌────────────────────┬──────────────────────┬────────────────────────┐
# │ Özellik            │ Gemini               │ LightGBM               │
# ├────────────────────┼──────────────────────┼────────────────────────┤
# │ Öğrenme            │ Yok (stateless)      │ Geçmiş trade'lerden    │
# │ Karar tipi         │ Semantik prompt       │ Olasılıksal tahmin     │
# │ Feedback loop      │ Yok                   │ Her retrain'de iyileşir│
# │ Deterministik      │ Hayır (temperature)   │ Evet (seed fixed)      │
# │ Maliyet            │ API call (quota)      │ Sıfır (lokal)          │
# │ Cold start         │ Yok (hemen çalışır)   │ Var (min 30 trade)     │
# └────────────────────┴──────────────────────┴────────────────────────┘
#
# Model Detayları:
# - Task: Binary Classification (karlı=1, zararlı=0)
# - Validation: Walk-forward expanding window (temporal order korunur)
# - Purged Gap: Eğitim sonu ile test başı arasında 2 trade boşluk (leakage koruması)
# - Calibration: Platt Scaling ile olasılık kalibrasyonu
# - Cold Start: Min 30 trade altında IC-only fallback (model tahmini yapılmaz)
# - Regularization: L1/L2 + max_depth limit + early stopping
#
# Kullanım:
#   from ml.lgbm_model import LGBMSignalModel
#   model = LGBMSignalModel()
#   model.train(feature_df, target_series)
#   result = model.predict(feature_vector)
# =============================================================================

import json                                    # Model meta verisi JSON olarak saklanır
import logging                                 # Yapılandırılmış log mesajları
import pickle                                  # Model serialization (joblib yerine — daha az bağımlılık)
import numpy as np                             # Sayısal hesaplamalar ve array işlemleri
import pandas as pd                            # DataFrame — feature matrix ve sonuç tabloları
from pathlib import Path                       # Platform-bağımsız dosya yolları
from typing import Dict, List, Optional, Tuple, Any  # Tip belirteçleri
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfları
from datetime import datetime, timezone        # Model versiyonlama zamanı
from copy import deepcopy                      # Model kopyalama (snapshot)

# LightGBM — gradient boosting framework
# Neden LightGBM (XGBoost / CatBoost değil)?
# 1. Histogram-based split → hızlı eğitim (küçük veri setlerinde bile verimli)
# 2. Native NaN handling → imputation gereksiz
# 3. Native categorical support → one-hot encoding ihtiyacı azalır
# 4. Leaf-wise growth → daha derin ağaçlar, daha az overfitting (max_depth ile kontrol)
try:
    import lightgbm as lgb                     # LightGBM ana kütüphanesi
    HAS_LIGHTGBM = True                        # LightGBM kurulu
except ImportError:
    lgb = None                                 # LightGBM kurulu değil
    HAS_LIGHTGBM = False                       # Fallback: sklearn GradientBoosting

# Scikit-learn — calibration, metrikler ve fallback model
from sklearn.calibration import CalibratedClassifierCV  # Platt Scaling
from sklearn.ensemble import GradientBoostingClassifier  # LightGBM yoksa fallback model
from sklearn.metrics import (                  # Değerlendirme metrikleri
    accuracy_score,                            # Doğruluk (baseline referans)
    roc_auc_score,                             # AUC-ROC (ranking kalitesi)
    brier_score_loss,                          # Olasılık kalibrasyonu ölçüsü
    log_loss,                                  # Logaritmik kayıp (olasılık kalitesi)
    precision_score,                           # Precision (yanlış pozitif kontrolü)
    recall_score,                              # Recall (kaçırılan fırsat kontrolü)
    f1_score,                                  # F1 (precision-recall dengesi)
    classification_report,                     # Detaylı sınıflandırma raporu
)

# Proje içi import
from .feature_engineer import (                # Adım 1'de oluşturduğumuz modül
    MLDecision,                                # LONG/SHORT/WAIT enum
    MLDecisionResult,                          # Nihai karar objesi
    MLFeatureVector,                           # Feature vektörü
)

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# SABİTLER
# =============================================================================

MIN_SAMPLES_TRAIN = 30                         # Minimum eğitim sample sayısı (cold start eşiği)
MIN_SAMPLES_POSITIVE = 10                      # Minimum pozitif sınıf sample (class imbalance koruması)
PURGE_GAP = 2                                  # Walk-forward'da eğitim-test arası boşluk (temporal leakage koruması)
DEFAULT_SEED = 42                              # Tekrarlanabilirlik için sabit random seed
MODEL_DIR = Path("models")                     # Model dosyalarının kaydedileceği dizin


# =============================================================================
# MODEL PERFORMANS DATACLASS
# =============================================================================

@dataclass
class ModelMetrics:
    """
    Model eğitim/validasyon metrikleri.
    
    Her retrain sonrası bu obje oluşturulur ve loglanır.
    Zaman içinde metriklerin seyri takip edilir (model degradation kontrolü).
    """
    # Sınıflandırma metrikleri
    accuracy: float = 0.0                      # Doğruluk oranı (baseline)
    auc_roc: float = 0.5                       # AUC-ROC (0.5 = rastgele, 1.0 = mükemmel)
    brier_score: float = 0.5                   # Brier Score (düşük = iyi kalibrasyon)
    log_loss_val: float = 1.0                  # Log Loss (düşük = iyi olasılık tahmini)
    precision: float = 0.0                     # Precision (trade açtığında doğruluk)
    recall: float = 0.0                        # Recall (karlı fırsatları yakalama)
    f1: float = 0.0                            # F1 Score (precision-recall dengesi)

    # Veri bilgisi
    n_train: int = 0                           # Eğitim sample sayısı
    n_val: int = 0                             # Validasyon sample sayısı
    n_positive: int = 0                        # Pozitif sınıf (karlı trade) sayısı
    positive_rate: float = 0.0                 # Pozitif sınıf oranı (base rate)

    # Model bilgisi
    n_features: int = 0                        # Kullanılan feature sayısı
    best_iteration: int = 0                    # Early stopping'de en iyi iterasyon
    feature_importance_top5: List[str] = field(default_factory=list)  # En önemli 5 feature

    # Meta
    model_version: str = ""                    # Eğitim tarihi bazlı versiyon
    trained_at: str = ""                       # Eğitim zamanı (UTC ISO)

    def is_usable(self) -> bool:
        """
        Model kullanılabilir durumda mı?
        
        Kriterler:
        1. AUC > 0.52 — rastgeleden ölçülebilir şekilde iyi (0.5 = rastgele)
        2. Yeterli sample ile eğitilmiş (n_train >= MIN_SAMPLES_TRAIN)
        3. Brier score makul (< 0.35 — kötü kalibrasyon değil)
        
        Not: 0.52 kasıtlı olarak düşük tutuluyor çünkü:
        - Kripto'da edge çok küçük olabilir
        - Başlangıçta az veriyle eğitilecek
        - Zamanla iyileşmesi bekleniyor
        """
        return (
            self.auc_roc > 0.52                # Rastgeleden anlamlı fark
            and self.n_train >= MIN_SAMPLES_TRAIN  # Yeterli veri
            and self.brier_score < 0.35        # Makul kalibrasyon
        )

    def summary(self) -> str:
        """Telegram / log için okunabilir özet."""
        status = "✅ Kullanılabilir" if self.is_usable() else "⚠️ Yetersiz"
        return (
            f"📊 Model Metrikleri ({status})\n"
            f"  AUC: {self.auc_roc:.3f} | F1: {self.f1:.3f} | Brier: {self.brier_score:.3f}\n"
            f"  Precision: {self.precision:.3f} | Recall: {self.recall:.3f}\n"
            f"  Train: {self.n_train} | Val: {self.n_val} | +Rate: {self.positive_rate:.1%}\n"
            f"  Top Features: {', '.join(self.feature_importance_top5[:3])}"
        )


# =============================================================================
# ANA MODEL SINIFI
# =============================================================================

class LGBMSignalModel:
    """
    LightGBM binary classification modeli — trade karlılık tahmini.
    
    Pipeline:
    1. train()   : Geçmiş trade feature'ları + sonuçları ile model eğit
    2. predict() : Yeni sinyal için karlılık olasılığı tahmin et
    3. retrain() : Yeni trade verileri gelince modeli güncelle
    4. save/load : Model persist (disk'e kaydet / diskten yükle)
    
    Walk-Forward Validasyon:
    - Eğitim verisi zamana göre sıralı (temporal order)
    - Son %20 validasyon, ilk %80 eğitim (expanding window)
    - Purge gap: eğitim sonu ile validasyon başı arasında 2 sample boşluk
    - Bu sayede temporal leakage riski minimize edilir
    
    Cold Start Stratejisi:
    - < 30 trade: Model eğitilmez, IC-only fallback kullanılır
    - 30-100 trade: Basit model (az ağaç, yüksek regularization)
    - > 100 trade: Tam model (daha fazla ağaç, daha az regularization)
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,      # Model kayıt dizini
        seed: int = DEFAULT_SEED,              # Random seed (tekrarlanabilirlik)
        verbose: bool = True,                  # Detaylı log mesajları
    ):
        """
        LGBMSignalModel başlatır.
        
        Parameters:
        ----------
        model_dir : Path, optional
            Model dosyalarının kaydedileceği dizin
            None → proje kökünde 'models/' kullanılır
            
        seed : int
            Random seed — aynı veri ile aynı model çıktısı garanti
            
        verbose : bool
            True → eğitim/tahmin detaylarını logla
        """
        self.model_dir = model_dir or MODEL_DIR  # Model kayıt dizini
        self.seed = seed                       # Tekrarlanabilirlik seed'i
        self.verbose = verbose                 # Log detay seviyesi

        # Model state
        self.model: Optional[lgb.LGBMClassifier] = None        # Eğitilmiş LightGBM modeli
        self.calibrator: Optional[CalibratedClassifierCV] = None  # Platt Scaling kalibrasyonu
        self.feature_names: List[str] = []     # Eğitimde kullanılan feature isimleri (sıralı)
        self.metrics: Optional[ModelMetrics] = None  # Son eğitimin metrikleri
        self.is_trained: bool = False          # Model eğitilmiş mi?
        self.model_version: str = ""           # Versiyon string'i (tarih bazlı)

        # Eğitim geçmişi (model degradation takibi)
        self.metrics_history: List[ModelMetrics] = []  # Tüm eğitimlerin metrikleri

        # Model dizinini oluştur
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LIGHTGBM HİPERPARAMETRELER
    # =========================================================================

    def _get_params(self, n_samples: int) -> Dict[str, Any]:
        """
        Veri büyüklüğüne göre adaptif hiperparametreler.
        
        Küçük veri (30-100 sample):
        - Az ağaç (50) + yüksek regularization → overfitting koruması
        - Sığ ağaçlar (max_depth=3) → basit karar kuralları
        - Yüksek min_child_samples → her yaprak yeterli örnekle destekli
        
        Büyük veri (100+ sample):
        - Daha fazla ağaç (200) + orta regularization
        - Daha derin ağaçlar (max_depth=5) → karmaşık etkileşimler
        - Düşük min_child_samples → daha ince granülarite
        
        LightGBM yoksa sklearn GradientBoosting parametreleri döndürülür.
        
        Parameters:
        ----------
        n_samples : int
            Toplam eğitim sample sayısı
            
        Returns:
        -------
        Dict[str, Any]
            Model parametre dict'i (LightGBM veya sklearn formatında)
        """
        if HAS_LIGHTGBM:
            # ── LightGBM Parametreleri ──
            if n_samples < 100:
                return {
                    'objective': 'binary',             # Binary classification (karlı/zararlı)
                    'metric': 'binary_logloss',        # Optimizasyon metriği: log loss
                    'boosting_type': 'gbdt',           # Gradient Boosted Decision Trees
                    'n_estimators': 50,                # Az ağaç — overfitting riski düşük
                    'max_depth': 3,                    # Sığ ağaç — basit karar kuralları
                    'num_leaves': 7,                   # 2^3 - 1 = 7 (max_depth ile tutarlı)
                    'learning_rate': 0.05,             # Yavaş öğrenme — daha stabil
                    'min_child_samples': 10,           # Her yaprakta min 10 sample
                    'reg_alpha': 1.0,                  # L1 regularization (feature selection etkisi)
                    'reg_lambda': 2.0,                 # L2 regularization (ağırlık shrinkage)
                    'subsample': 0.8,                  # Her ağaçta %80 sample kullan (bagging)
                    'colsample_bytree': 0.8,           # Her ağaçta %80 feature kullan
                    'min_split_gain': 0.01,            # Minimum split kazancı
                    'random_state': self.seed,         # Tekrarlanabilirlik
                    'verbose': -1,                     # LightGBM sessiz mod
                    'is_unbalance': True,              # Class imbalance otomatik ağırlıklama
                }
            return {
                'objective': 'binary',                 # Binary classification
                'metric': 'binary_logloss',            # Log loss
                'boosting_type': 'gbdt',               # GBDT
                'n_estimators': 200,                   # Daha fazla ağaç
                'max_depth': 5,                        # Orta derinlik
                'num_leaves': 20,                      # Daha fazla yaprak
                'learning_rate': 0.03,                 # Daha yavaş öğrenme
                'min_child_samples': 5,                # Daha ince granülarite
                'reg_alpha': 0.5,                      # Orta L1
                'reg_lambda': 1.0,                     # Orta L2
                'subsample': 0.85,                     # %85 sample bagging
                'colsample_bytree': 0.85,              # %85 feature bagging
                'min_split_gain': 0.005,               # Daha düşük split eşiği
                'random_state': self.seed,             # Tekrarlanabilirlik
                'verbose': -1,                         # Sessiz mod
                'is_unbalance': True,                  # Class imbalance handling
            }
        else:
            # ── sklearn GradientBoosting Parametreleri (LightGBM yoksa fallback) ──
            if n_samples < 100:
                return {
                    'n_estimators': 50,                # Az ağaç
                    'max_depth': 3,                    # Sığ ağaç
                    'learning_rate': 0.05,             # Yavaş öğrenme
                    'min_samples_leaf': 10,            # Min yaprak sample (lgb: min_child_samples)
                    'subsample': 0.8,                  # Stochastic gradient boosting
                    'max_features': 0.8,               # Feature bagging (lgb: colsample_bytree)
                    'random_state': self.seed,         # Tekrarlanabilirlik
                }
            return {
                'n_estimators': 200,                   # Daha fazla ağaç
                'max_depth': 5,                        # Orta derinlik
                'learning_rate': 0.03,                 # Daha yavaş öğrenme
                'min_samples_leaf': 5,                 # Daha ince granülarite
                'subsample': 0.85,                     # %85 stochastic
                'max_features': 0.85,                  # %85 feature
                'random_state': self.seed,             # Tekrarlanabilirlik
            }

    # =========================================================================
    # EĞİTİM (TRAIN)
    # =========================================================================

    def train(
        self,
        X: pd.DataFrame,                      # Feature matrix (satır=trade, kolon=feature)
        y: pd.Series,                          # Hedef (1=karlı, 0=zararlı)
        val_ratio: float = 0.2,               # Validasyon oranı (son %20)
        purge_gap: int = PURGE_GAP,           # Eğitim-validasyon arası boşluk
    ) -> ModelMetrics:
        """
        Walk-forward expanding window ile model eğit.
        
        Walk-Forward Neden?
        - Zaman serisi verisinde random split → temporal leakage
        - Walk-forward: eğitim DAİMA geçmiş, validasyon DAİMA gelecek
        - Bu sayede gerçek dünya performansına daha yakın tahmin
        
        Purge Gap Neden?
        - Ardışık trade'ler aynı piyasa koşullarında açılmış olabilir
        - Eğitim sonundaki trade ile validasyon başındaki trade korelasyonlu olabilir
        - 2 trade boşluk bırakarak bu temporal sızıntıyı azaltırız
        
        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix — feature_engineer.build_batch_features() çıktısı
            Meta kolonlar (_symbol, _coin, _timestamp) dahil olabilir, filtrelenir
            
        y : pd.Series
            Hedef değişken — 1=karlı trade (net_pnl > 0), 0=zararlı trade
            
        val_ratio : float
            Validasyon seti oranı (varsayılan 0.2 = son %20)
            
        purge_gap : int
            Eğitim-validasyon arası boşluk (temporal leakage koruması)
            
        Returns:
        -------
        ModelMetrics
            Eğitim ve validasyon metrikleri
        """
        # ── 0. Veri Validasyonu ──
        # Meta kolonları (_ile başlayanlar) feature'lardan ayır
        feature_cols = [c for c in X.columns if not c.startswith('_')]
        X_clean = X[feature_cols].copy()       # Sadece feature kolonları
        self.feature_names = feature_cols       # Feature isimlerini kaydet

        n_total = len(X_clean)                 # Toplam sample sayısı
        n_positive = int(y.sum())              # Karlı trade sayısı

        # Minimum sample kontrolü (cold start)
        if n_total < MIN_SAMPLES_TRAIN:
            logger.warning(
                f"⚠️ Yetersiz veri: {n_total} < {MIN_SAMPLES_TRAIN} minimum. "
                f"Model eğitilmeyecek, IC-only fallback kullanılacak."
            )
            return self._empty_metrics(n_total, n_positive)

        # Minimum pozitif sınıf kontrolü (class imbalance)
        if n_positive < MIN_SAMPLES_POSITIVE:
            logger.warning(
                f"⚠️ Yetersiz pozitif sınıf: {n_positive} < {MIN_SAMPLES_POSITIVE}. "
                f"Model dengesiz olabilir."
            )
            # Eğitime devam et ama uyar (is_unbalance=True bunu handle eder)

        # ── 1. Walk-Forward Split ──
        # Temporal order korunuyor — SON val_ratio kadarı validasyon
        val_size = max(int(n_total * val_ratio), 5)  # Min 5 validasyon sample
        train_end = n_total - val_size - purge_gap   # Purge gap bırak

        if train_end < MIN_SAMPLES_TRAIN:
            # Purge gap çıkarınca eğitim seti çok küçük kaldı
            train_end = n_total - val_size     # Purge gap'i kaldır (az veri durumu)
            purge_gap = 0

        X_train = X_clean.iloc[:train_end]             # Eğitim feature'ları
        y_train = y.iloc[:train_end]                    # Eğitim hedefi
        X_val = X_clean.iloc[train_end + purge_gap:]   # Validasyon feature'ları (purge gap sonrası)
        y_val = y.iloc[train_end + purge_gap:]          # Validasyon hedefi

        if self.verbose:
            logger.info(
                f"  📐 Walk-Forward Split: "
                f"Train={len(X_train)} | Purge={purge_gap} | Val={len(X_val)} | "
                f"+Rate: {y_train.mean():.1%} (train) / {y_val.mean():.1%} (val)"
            )

        # ── 2. Model Eğitimi ──
        params = self._get_params(len(X_train))  # Adaptif hiperparametreler

        if HAS_LIGHTGBM:
            # LightGBM — native NaN handling, early stopping callback
            self.model = lgb.LGBMClassifier(**params)

            # Early stopping callback: validasyon loss iyileşmezse dur
            callbacks = [
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            # sklearn GradientBoosting — NaN'ları median ile doldur (native NaN desteği yok)
            self._impute_median = X_train.median()         # Median değerleri sakla (predict'te de lazım)
            X_train_filled = X_train.fillna(self._impute_median)
            X_val_filled = X_val.fillna(self._impute_median)

            self.model = GradientBoostingClassifier(**params)
            self.model.fit(X_train_filled, y_train)

            # sklearn'de early stopping yok → n_estimators sabit kalır
            # Validasyon seti kalibrasyon için kullanılacak

        # ── 3. Olasılık Kalibrasyonu (Platt Scaling) ──
        # LightGBM'in ham olasılıkları genellikle well-calibrated değil
        # Platt Scaling: sigmoid fit ile kalibre et
        try:
            X_val_for_cal = X_val if HAS_LIGHTGBM else X_val.fillna(self._impute_median)

            # sklearn >= 1.6'da 'prefit' kaldırıldı, cv=2 ile çalıştır
            # prefit destekliyorsa onu kullan (daha doğru), yoksa mini cv
            try:
                self.calibrator = CalibratedClassifierCV(
                    self.model, method='sigmoid', cv='prefit',
                )
                self.calibrator.fit(X_val_for_cal, y_val)
            except (ValueError, TypeError):
                # prefit desteklenmiyorsa → 2-fold CV ile kalibre et
                # Not: Bu eğitim setinde yeniden fit yapar ama Platt Scaling
                # sadece sigmoid parametreleri öğreniyor, model ağırlıkları değişmiyor
                X_cal = pd.concat([X_train if HAS_LIGHTGBM else X_train.fillna(self._impute_median),
                                   X_val_for_cal], ignore_index=True)
                y_cal = pd.concat([y_train, y_val], ignore_index=True)
                self.calibrator = CalibratedClassifierCV(
                    estimator=GradientBoostingClassifier(**params) if not HAS_LIGHTGBM else lgb.LGBMClassifier(**params),
                    method='sigmoid', cv=2,
                )
                self.calibrator.fit(X_cal, y_cal)
        except Exception as e:
            logger.warning(f"⚠️ Kalibrasyon başarısız: {e}. Ham olasılıklar kullanılacak.")
            self.calibrator = None

        # ── 4. Metrik Hesaplama ──
        metrics = self._evaluate(X_val, y_val, X_train, y_train)
        self.metrics = metrics                 # Son metrikleri kaydet
        self.metrics_history.append(metrics)   # Geçmişe ekle
        self.is_trained = True                 # Model kullanılabilir

        # Versiyon string'i oluştur (tarih bazlı)
        self.model_version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
        metrics.model_version = self.model_version
        metrics.trained_at = datetime.now(timezone.utc).isoformat()

        if self.verbose:
            logger.info(f"\n{metrics.summary()}")

        return metrics

    # =========================================================================
    # TAHMİN (PREDICT)
    # =========================================================================

    def predict(
        self,
        feature_vector: 'MLFeatureVector',     # Adım 1'deki feature vektörü
        ic_score: float = 0.0,                # IC composite skoru (gate keeper için)
        ic_direction: str = "NEUTRAL",         # IC yönü (fallback için)
        gate_thresholds: Optional[Dict] = None,  # Gate keeper eşikleri
    ) -> MLDecisionResult:
        """
        Yeni sinyal için karlılık tahmini yap ve karar döndür.
        
        Pipeline:
        1. Gate Keeper kontrolü (IC eşikleri — mevcut sistemle aynı)
        2. Feature vektörünü numpy array'e çevir
        3. LightGBM predict_proba() → karlılık olasılığı
        4. Kalibrasyon uygula (varsa)
        5. Olasılık → karar mapping (eşik bazlı)
        6. MLDecisionResult oluştur
        
        Cold Start: Model eğitilmemişse IC-only fallback kullanılır.
        
        Parameters:
        ----------
        feature_vector : MLFeatureVector
            Adım 1'de oluşturulan feature vektörü
            
        ic_score : float
            IC composite skoru (0-100) — gate keeper kontrolü için
            
        ic_direction : str
            IC'nin önerdiği yön ('LONG'/'SHORT'/'NEUTRAL')
            
        gate_thresholds : Dict, optional
            Gate keeper eşikleri {'no_trade': 40, 'full_trade': 70}
            None → varsayılan değerler kullanılır
            
        Returns:
        -------
        MLDecisionResult
            Karar + güven + gerekçe (execution modülüne gönderilir)
        """
        # Varsayılan gate eşikleri
        if gate_thresholds is None:
            gate_thresholds = {'no_trade': 40, 'full_trade': 70}

        # ── 1. Gate Keeper ──
        # IC eşik kontrolü (mevcut sistemle aynı mantık)
        gate_action = self._check_gate(ic_score, gate_thresholds)

        if gate_action == "NO_TRADE":
            return MLDecisionResult(
                decision=MLDecision.WAIT,
                confidence=max(ic_score * 0.5, 10),  # Düşük güven
                reasoning=f"IC skoru ({ic_score:.0f}) gate eşiğinin altında ({gate_thresholds['no_trade']})",
                gate_action="NO_TRADE",
                ic_score=ic_score,
                model_version=self.model_version or "no_model",
            )

        # ── 2. Cold Start Kontrolü ──
        # Model eğitilmemişse veya yetersizse IC-only fallback
        if not self.is_trained or self.model is None:
            return self._ic_fallback(ic_score, ic_direction, gate_action)

        # Model metrikleri yetersizse uyar ama yine de tahmin yap
        if self.metrics and not self.metrics.is_usable():
            logger.warning(
                f"⚠️ Model metrikleri yetersiz (AUC={self.metrics.auc_roc:.3f}). "
                f"Tahmin yapılacak ama güven düşürülecek."
            )

        # ── 3. Feature Array Hazırla ──
        try:
            feature_dict = feature_vector.to_dict()  # Feature dict

            # Eğitimdeki feature sıralaması ile aynı mı kontrol et
            X_pred = pd.DataFrame([feature_dict])[self.feature_names]
        except KeyError as e:
            logger.error(f"❌ Feature uyumsuzluğu: {e}")
            return self._ic_fallback(ic_score, ic_direction, gate_action)

        # ── 4. Tahmin ──
        try:
            # sklearn fallback'te NaN'ları median ile doldur
            if not HAS_LIGHTGBM and hasattr(self, '_impute_median'):
                X_pred = X_pred.fillna(self._impute_median)

            if self.calibrator is not None:
                # Kalibre edilmiş olasılık (daha güvenilir)
                prob = self.calibrator.predict_proba(X_pred)[0][1]
            else:
                # Ham model olasılığı
                prob = self.model.predict_proba(X_pred)[0][1]
        except Exception as e:
            logger.error(f"❌ Tahmin hatası: {e}")
            return self._ic_fallback(ic_score, ic_direction, gate_action)

        # ── 5. Olasılık → Karar Mapping ──
        confidence = prob * 100                # 0-1 → 0-100 skala

        # Model metrikler yetersizse güveni düşür
        if self.metrics and not self.metrics.is_usable():
            confidence *= 0.75                 # %25 penaltı

        # Karar eşikleri:
        # prob >= 0.55 → IC yönünde işlem aç (küçük edge bile değerli)
        # prob < 0.45  → IC yönünün tersi veya WAIT
        # 0.45-0.55    → Belirsiz bölge → WAIT
        if prob >= 0.55:
            decision = MLDecision.from_direction(ic_direction)  # IC yönünde
        elif prob < 0.45:
            decision = MLDecision.WAIT                          # Model onaylamıyor
        else:
            decision = MLDecision.WAIT                          # Belirsiz bölge

        # ── 6. Feature Importance (top 3) ──
        top3_features = self._get_top_features(3)

        # ── 7. Gerekçe Oluştur ──
        reasoning = self._build_reasoning(prob, ic_score, ic_direction, decision, top3_features)

        return MLDecisionResult(
            decision=decision,
            confidence=round(confidence, 1),
            reasoning=reasoning,
            gate_action=gate_action,
            ic_score=ic_score,
            model_version=self.model_version,
            feature_importance_top3=top3_features,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # =========================================================================
    # RETRAIN (YENİ VERİ İLE GÜNCELLEME)
    # =========================================================================

    def retrain(
        self,
        X: pd.DataFrame,                      # Tüm mevcut feature matrix (eski + yeni)
        y: pd.Series,                          # Tüm hedef değişken (eski + yeni)
    ) -> ModelMetrics:
        """
        Yeni trade verileri ile modeli sıfırdan eğit.
        
        Neden Incremental Değil?
        - LightGBM incremental learning destekler ama küçük veri setlerinde
          sıfırdan eğitim daha stabil sonuç verir
        - 30-500 trade arasında sıfırdan eğitim < 1 saniye sürer
        - Incremental'da eski pattern'ler unutulabilir (catastrophic forgetting)
        
        Çağrılma Zamanı:
        - Her N trade kapandığında (varsayılan: 5)
        - Günlük periyodik (cron/scheduler)
        - Manuel tetikleme
        
        Parameters:
        ----------
        X : pd.DataFrame
            TÜM geçmiş trade'lerin feature matrix'i (trade_memory'den gelir)
            
        y : pd.Series
            TÜM geçmiş trade'lerin hedefi (1=karlı, 0=zararlı)
            
        Returns:
        -------
        ModelMetrics
            Yeni eğitimin metrikleri
        """
        if self.verbose:
            logger.info(
                f"🔄 Retrain başlıyor: {len(X)} sample | "
                f"+Rate: {y.mean():.1%}"
            )

        # Eski modeli snapshot al (rollback için)
        old_model = deepcopy(self.model)       # Derin kopya
        old_metrics = deepcopy(self.metrics)

        # Sıfırdan eğit
        new_metrics = self.train(X, y)

        # Model degradation kontrolü
        # Yeni model eskisinden belirgin şekilde kötüyse rollback yap
        if old_metrics and old_metrics.is_usable() and new_metrics.is_usable():
            if new_metrics.auc_roc < old_metrics.auc_roc - 0.05:
                # AUC 0.05'ten fazla düştü → rollback
                logger.warning(
                    f"⚠️ Model degradation! AUC: {old_metrics.auc_roc:.3f} → {new_metrics.auc_roc:.3f}. "
                    f"Eski model korunuyor."
                )
                self.model = old_model
                self.metrics = old_metrics
                return old_metrics

        # Auto-save
        try:
            self.save()
        except Exception as e:
            logger.warning(f"⚠️ Model kayıt hatası: {e}")

        return new_metrics

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Feature importance tablosu döndür.
        
        3 farklı importance tipi:
        - split: Kaç kez split'te kullanılmış (sıklık)
        - gain: Toplam bilgi kazancı (kalite)
        - combined: Normalize split + gain ortalaması (genel önem)
        
        Returns:
        -------
        pd.DataFrame
            Kolonlar: feature, split, gain, combined
            Sıralama: combined (azalan)
        """
        if not self.is_trained or self.model is None:
            return pd.DataFrame()              # Eğitilmemiş → boş

        if HAS_LIGHTGBM and hasattr(self.model, 'booster_'):
            # LightGBM — split ve gain bazlı iki ayrı importance
            split_imp = self.model.booster_.feature_importance(importance_type='split')
            gain_imp = self.model.booster_.feature_importance(importance_type='gain')

            df = pd.DataFrame({
                'feature': self.feature_names,
                'split': split_imp,
                'gain': gain_imp,
            })

            for col in ['split', 'gain']:
                total = df[col].sum()
                df[f'{col}_norm'] = df[col] / total if total > 0 else 0.0

            df['combined'] = (df['split_norm'] + df['gain_norm']) / 2
        else:
            # sklearn GradientBoosting — feature_importances_ (impurity-based)
            imp = self.model.feature_importances_
            df = pd.DataFrame({
                'feature': self.feature_names,
                'split': imp,                  # sklearn'de tek importance var
                'gain': imp,                   # Aynı değeri kopyala (uyumluluk)
            })
            total = imp.sum()
            df['split_norm'] = imp / total if total > 0 else 0.0
            df['gain_norm'] = df['split_norm']
            df['combined'] = df['split_norm']

        # Combined'a göre sırala (en önemli üstte)
        df = df.sort_values('combined', ascending=False).reset_index(drop=True)

        return df

    # =========================================================================
    # MODEL KAYIT / YÜKLEME
    # =========================================================================

    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Eğitilmiş modeli diske kaydet.
        
        Kaydedilenler:
        1. LightGBM model (pickle)
        2. Calibrator (pickle)
        3. Feature names (JSON)
        4. Metrics (JSON)
        5. Versiyon bilgisi (JSON)
        
        Parameters:
        ----------
        filepath : Path, optional
            Kayıt dosya yolu. None → model_dir/lgbm_signal_model.pkl
            
        Returns:
        -------
        Path
            Kaydedilen dosya yolu
        """
        if not self.is_trained:
            raise ValueError("Model eğitilmemiş — kayıt yapılamaz")

        save_path = filepath or self.model_dir / "lgbm_signal_model.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Tek dosyada tüm state'i kaydet
        state = {
            'model': self.model,               # LightGBM veya sklearn model objesi
            'calibrator': self.calibrator,      # Platt Scaling objesi
            'feature_names': self.feature_names,  # Feature isimleri (sıralı)
            'metrics': self.metrics,            # Son metrikler
            'model_version': self.model_version,  # Versiyon string'i
            'has_lightgbm': HAS_LIGHTGBM,      # Hangi engine ile eğitildi
            'impute_median': getattr(self, '_impute_median', None),  # sklearn NaN dolgu değerleri
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }

        with open(save_path, 'wb') as f:
            pickle.dump(state, f)              # Pickle ile serialize et

        # Meta bilgiyi ayrıca JSON olarak da kaydet (okunabilirlik)
        meta_path = save_path.with_suffix('.json')
        meta = {
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'is_usable': self.metrics.is_usable() if self.metrics else False,
            'auc_roc': self.metrics.auc_roc if self.metrics else 0,
            'n_train': self.metrics.n_train if self.metrics else 0,
            'saved_at': state['saved_at'],
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            logger.info(f"💾 Model kaydedildi: {save_path}")

        return save_path

    def load(self, filepath: Optional[Path] = None) -> bool:
        """
        Kaydedilmiş modeli diskten yükle.
        
        Parameters:
        ----------
        filepath : Path, optional
            Yüklenecek dosya yolu. None → model_dir/lgbm_signal_model.pkl
            
        Returns:
        -------
        bool
            True = başarıyla yüklendi, False = dosya yok veya hata
        """
        load_path = filepath or self.model_dir / "lgbm_signal_model.pkl"

        if not load_path.exists():
            logger.info(f"📂 Model dosyası bulunamadı: {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)         # Deserialize et

            self.model = state['model']
            self.calibrator = state.get('calibrator')
            self.feature_names = state['feature_names']
            self.metrics = state.get('metrics')
            self.model_version = state.get('model_version', 'loaded')
            self.is_trained = True
            # sklearn NaN dolgu değerlerini restore et
            if state.get('impute_median') is not None:
                self._impute_median = state['impute_median']

            if self.verbose:
                logger.info(
                    f"📂 Model yüklendi: {load_path} | "
                    f"Version: {self.model_version} | "
                    f"Features: {len(self.feature_names)}"
                )

            return True

        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            return False

    # =========================================================================
    # YARDIMCI METODLAR (PRIVATE)
    # =========================================================================

    def _check_gate(self, ic_score: float, thresholds: Dict) -> str:
        """
        Gate Keeper: IC skoru eşik kontrolü.
        Mevcut sistemdeki GateAction enum'unun string karşılığı.
        """
        if ic_score < thresholds.get('no_trade', 40):
            return "NO_TRADE"                  # IC çok düşük — işlem yapma
        elif ic_score < thresholds.get('full_trade', 70):
            return "REPORT_ONLY"               # IC orta — sadece raporla
        else:
            return "FULL_TRADE"                # IC yüksek — trade açılabilir

    def _ic_fallback(
        self,
        ic_score: float,
        ic_direction: str,
        gate_action: str,
    ) -> MLDecisionResult:
        """
        Model eğitilmemişken IC-only fallback karar.
        Gemini'deki _ic_fallback ile aynı mantık.
        
        IC >= 70 ve net yön → IC yönünde düşük güvenle karar
        Aksi halde → WAIT
        """
        if ic_score >= 70 and ic_direction in ['LONG', 'SHORT']:
            decision = MLDecision.from_direction(ic_direction)
            confidence = min(ic_score * 0.65, 60)  # Max %60 güven (fallback sınırı)
            reasoning = (
                f"⚠️ ML model henüz eğitilmemiş (cold start). "
                f"IC fallback: {ic_direction} (IC={ic_score:.0f}, güven düşürülmüş)"
            )
        else:
            decision = MLDecision.WAIT
            confidence = max(ic_score * 0.3, 10)
            reasoning = (
                f"ML model eğitilmemiş ve IC skoru yetersiz ({ic_score:.0f}). "
                f"İşlem yapılmıyor."
            )

        return MLDecisionResult(
            decision=decision,
            confidence=round(confidence, 1),
            reasoning=reasoning,
            gate_action=gate_action if gate_action != "NO_TRADE" else "REPORT_ONLY",
            ic_score=ic_score,
            model_version="ic_fallback",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> ModelMetrics:
        """
        Model performans metriklerini hesapla.
        
        Validasyon seti üzerinde değerlendirme yapar.
        Eğitim seti metrikleri sadece overfitting kontrolü için.
        """
        metrics = ModelMetrics()

        # Tahmin olasılıkları (NaN handling for sklearn)
        X_val_pred = X_val if HAS_LIGHTGBM else X_val.fillna(X_train.median())
        if self.calibrator is not None:
            y_prob = self.calibrator.predict_proba(X_val_pred)[:, 1]
        else:
            y_prob = self.model.predict_proba(X_val_pred)[:, 1]

        y_pred = (y_prob >= 0.5).astype(int)   # 0.5 eşik ile sınıflandırma

        # Metrikler
        metrics.accuracy = float(accuracy_score(y_val, y_pred))
        metrics.precision = float(precision_score(y_val, y_pred, zero_division=0))
        metrics.recall = float(recall_score(y_val, y_pred, zero_division=0))
        metrics.f1 = float(f1_score(y_val, y_pred, zero_division=0))
        metrics.brier_score = float(brier_score_loss(y_val, y_prob))
        metrics.log_loss_val = float(log_loss(y_val, y_prob, labels=[0, 1]))

        # AUC-ROC (en az 2 sınıf gerekli)
        if len(y_val.unique()) >= 2:
            metrics.auc_roc = float(roc_auc_score(y_val, y_prob))
        else:
            metrics.auc_roc = 0.5              # Tek sınıf → anlamsız

        # Veri bilgisi
        metrics.n_train = len(X_train)
        metrics.n_val = len(X_val)
        metrics.n_positive = int(y_train.sum() + y_val.sum())
        metrics.positive_rate = float(y_train.mean())
        metrics.n_features = len(self.feature_names)

        # Best iteration (early stopping — sadece LightGBM'de var)
        if HAS_LIGHTGBM and hasattr(self.model, 'best_iteration_'):
            metrics.best_iteration = self.model.best_iteration_ or self.model.n_estimators
        else:
            metrics.best_iteration = getattr(self.model, 'n_estimators', 0)

        # Feature importance (top 5)
        fi = self.get_feature_importance()
        if len(fi) > 0:
            metrics.feature_importance_top5 = fi['feature'].head(5).tolist()

        return metrics

    def _get_top_features(self, n: int = 3) -> List[str]:
        """En önemli N feature'ın isimlerini döndür."""
        fi = self.get_feature_importance()
        if len(fi) > 0:
            return fi['feature'].head(n).tolist()
        return []

    def _build_reasoning(
        self,
        prob: float,
        ic_score: float,
        ic_direction: str,
        decision: MLDecision,
        top_features: List[str],
    ) -> str:
        """Türkçe karar gerekçesi oluştur."""
        parts = []

        # Model tahmini
        if prob >= 0.55:
            parts.append(f"Model karlılık olasılığı: %{prob*100:.0f} (pozitif sinyal)")
        elif prob < 0.45:
            parts.append(f"Model karlılık olasılığı: %{prob*100:.0f} (negatif sinyal)")
        else:
            parts.append(f"Model karlılık olasılığı: %{prob*100:.0f} (belirsiz bölge)")

        # IC bilgisi
        parts.append(f"IC: {ic_score:.0f}/100 yön={ic_direction}")

        # Top features
        if top_features:
            parts.append(f"Etken: {', '.join(top_features[:2])}")

        # Model güvenilirliği
        if self.metrics:
            parts.append(f"Model AUC: {self.metrics.auc_roc:.2f}")

        return " | ".join(parts)

    def _empty_metrics(self, n_total: int, n_positive: int) -> ModelMetrics:
        """Eğitim yapılmadığında boş metrik objesi döndür."""
        return ModelMetrics(
            n_train=n_total,
            n_positive=n_positive,
            positive_rate=n_positive / n_total if n_total > 0 else 0,
            model_version="not_trained",
        )


# =============================================================================
# BAĞIMSIZ ÇALIŞTIRMA TESTİ
# =============================================================================

if __name__ == "__main__":
    """
    Modülü tek başına test et:
      cd src && python -m ml.lgbm_model
    
    Sentetik trade verisi ile eğitim, tahmin, save/load test eder.
    """
    import tempfile
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 60)
    print("  🌳 LIGHTGBM MODEL — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Sentetik Feature Matrix Oluştur ──
    np.random.seed(42)
    n_samples = 80                             # 80 sentetik trade
    n_features = 44                            # Adım 1'deki feature sayısı

    from .feature_engineer import FeatureEngineer
    eng = FeatureEngineer(verbose=False)
    feature_names = eng.get_feature_names()

    # Rastgele feature matrix
    X = pd.DataFrame(
        np.random.randn(n_samples, len(feature_names)),
        columns=feature_names,
    )

    # Sentetik hedef: feature'larla korelasyonlu (model öğrenebilsin)
    # ic_confidence ve ctf_direction_agreement yüksekse karlı olma olasılığı artar
    signal = (
        0.3 * X['ic_confidence'] +
        0.2 * X['ctf_direction_agreement'] +
        0.1 * X['px_momentum_5'] +
        np.random.randn(n_samples) * 0.5      # Noise
    )
    y = (signal > signal.median()).astype(int)  # Medyan üstü = karlı

    print(f"\n  Veri: {n_samples} sample × {len(feature_names)} feature")
    print(f"  +Rate: {y.mean():.1%}")

    # ── Model Eğitimi ──
    with tempfile.TemporaryDirectory() as tmpdir:
        model = LGBMSignalModel(model_dir=Path(tmpdir), verbose=True)

        print("\n  🔧 Eğitim başlıyor...")
        metrics = model.train(X, y)

        print(f"\n  is_usable: {metrics.is_usable()}")
        print(f"  AUC: {metrics.auc_roc:.3f}")

        # ── Tahmin Testi ──
        print("\n  🎯 Tahmin testi...")
        dummy_vec = MLFeatureVector()
        dummy_vec.ic_features = {k: v for k, v in zip(feature_names[:12], X.iloc[0, :12])}
        dummy_vec.market_features = {k: v for k, v in zip(feature_names[12:18], X.iloc[0, 12:18])}
        dummy_vec.cross_tf_features = {k: v for k, v in zip(feature_names[18:24], X.iloc[0, 18:24])}
        dummy_vec.price_features = {k: v for k, v in zip(feature_names[24:34], X.iloc[0, 24:34])}
        dummy_vec.risk_features = {k: v for k, v in zip(feature_names[34:39], X.iloc[0, 34:39])}
        dummy_vec.temporal_features = {k: v for k, v in zip(feature_names[39:], X.iloc[0, 39:])}

        result = model.predict(dummy_vec, ic_score=75.0, ic_direction="SHORT")
        print(f"  Karar: {result.decision.value} | Güven: {result.confidence:.1f}")
        print(f"  Gate: {result.gate_action}")
        print(f"  Gerekçe: {result.reasoning}")

        # ── Cold Start Testi ──
        print("\n  ❄️ Cold start testi...")
        cold_model = LGBMSignalModel(model_dir=Path(tmpdir), verbose=False)
        cold_result = cold_model.predict(dummy_vec, ic_score=75.0, ic_direction="LONG")
        assert cold_result.model_version == "ic_fallback", "Cold start fallback hatası"
        print(f"  Karar: {cold_result.decision.value} (fallback)")

        # ── Save / Load Testi ──
        print("\n  💾 Save/Load testi...")
        save_path = model.save()
        print(f"  Kaydedildi: {save_path}")

        model2 = LGBMSignalModel(model_dir=Path(tmpdir), verbose=False)
        loaded = model2.load()
        assert loaded, "Model yüklenemedi"
        assert model2.is_trained, "Yüklenen model trained değil"
        assert len(model2.feature_names) == len(feature_names), "Feature sayısı uyumsuz"
        print(f"  Yüklendi: {model2.model_version} | Features: {len(model2.feature_names)}")

        # Yüklenen model ile tahmin
        result2 = model2.predict(dummy_vec, ic_score=75.0, ic_direction="SHORT")
        print(f"  Yüklenen model tahmini: {result2.decision.value} | Güven: {result2.confidence:.1f}")

        # ── Feature Importance ──
        print("\n  📊 Feature Importance (Top 10):")
        fi = model.get_feature_importance()
        for _, row in fi.head(10).iterrows():
            print(f"    {row['feature']:<30} combined={row['combined']:.4f}")

        # ── Retrain Testi ──
        print("\n  🔄 Retrain testi...")
        # Yeni veri ekle
        X_new = pd.DataFrame(
            np.random.randn(20, len(feature_names)),
            columns=feature_names,
        )
        signal_new = 0.3 * X_new['ic_confidence'] + np.random.randn(20) * 0.5
        y_new = (signal_new > signal_new.median()).astype(int)

        X_all = pd.concat([X, X_new], ignore_index=True)
        y_all = pd.concat([y, y_new], ignore_index=True)

        retrain_metrics = model.retrain(X_all, y_all)
        print(f"  Retrain AUC: {retrain_metrics.auc_roc:.3f}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ LIGHTGBM MODEL TESTİ TAMAMLANDI")
    print(f"{'=' * 60}")
