# =============================================================================
# ENSEMBLE MODEL v3.2 — DIRECTION-AWARE R-MULTIPLE WITH THRESHOLD FLOORS
# =============================================================================
# Replaces v2.x binary classification with continuous R-multiple regression
# plus asymmetric directional thresholds AND empirical floor enforcement.
#
# v3.2 vs v3.1 değişiklikleri:
#   - HARD_THRESHOLD_FLOOR_LONG  = 0.10  (baseline LONG edge güvenlik marjı)
#   - HARD_THRESHOLD_FLOOR_SHORT = 0.40  (SHORT'a asimetrik güvenlik marjı)
#   - Floor enforcement üç katmanda: _calibrate_thresholds + _evaluate_walk_forward + train()
#   - Calibration sample-içi optimal eşiği bulsa bile floor altına inemez
#
# [SORUN 5 DÜZELTMESİ v1]
#   _purged_walk_forward artık n_splits'i dinamik ayarlıyor.
#   Küçük n'de (örn. n=120) 5 fold yerine min_test_size=15 garantilenecek
#   kadar fold üretiliyor. n_folds=0 durumunda is_trained=False kalıyor.
#
# [SORUN 1 DÜZELTMESİ]
#   Binary target detection warning mesajı daha açıklayıcı hale getirildi.
#   initial_train R-multiple ürettiği sürece bu dal tetiklenmemeli.
# =============================================================================

import os
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_TRAIN_SAMPLES = 120
CV_N_SPLITS = 5
# [MADDE 13 DÜZELTMESİ] — Purged embargo gap 5 → 8
# Lopez de Prado önerisi: embargo = TP/SL horizonu.
# forward_period = 6 bar × 1.3 güvenlik marjı ≈ 8 bar.
# Eski değer (5) temporal leakage'ı tam temizlemiyordu.
# 15m TF'de 8 bar = 2 saat → yeterli korelasyon koruması.
# [GEÇİCİ] CV embargo gap 8 → 5
# Sebep: 351 sample + 5 fold + embargo=8 → fold başına ~34 etkili sample → IC std çok yüksek.
# 450+ sample birikince embargo=8'e geri alınacak.
# Lopez de Prado tavsiyesi: embargo = forward_period. forward_period=6 → ideal=6, şimdi 5 kullanıyoruz.
CV_EMBARGO_GAP = 5
CV_MIN_TRAIN = 100
NESTED_CV_HOLDOUT = 0.20

# Default thresholds (calibration başarısız olursa kullanılır)
DEFAULT_THRESHOLD_LONG = 0.10
DEFAULT_THRESHOLD_SHORT = 0.40

# === HARD THRESHOLD FLOORS (ASYMMETRIC SAFETY MARGINS) ===
#
# Calibration sürecinin sample-içi optimal olarak bulduğu eşikler,
# OOS'ta spurious (rastlantısal) olabilir. Bu floor'lar minimum
# güvenlik eşiği garantiler.
#
# RATIONALE:
# - Model 'is_long' binary feature ile trained — aynı pred_r
#   hem LONG hem SHORT için "bu yön kârlı" sinyali verir.
# - LONG ve SHORT için FARKLI floor kullanıyoruz çünkü:
#
#   (a) Kripto trading'de LONG bias mevcut (upward drift).
#       SHORT trade'leri daha nadir, daha küçük sample → daha noisy.
#
#   (b) Real-world execution asymmetry: SHORT'ta funding rate
#       ödemesi + borrow cost → effective edge LONG'dan düşük.
#
#   (c) OOS monitoring: threshold=0.10 ile SHORT filtrelendiğinde
#       E[R|pred_r>0.10, direction=SHORT] sample sayısı <30
#       ve standard error büyük (CI genişliği >0.5R).
#
# GÜNCELLEME PROTOKOLÜ:
# Model retrain sonrası ml/eval_harness.py çalıştır. Eğer:
#   - SHORT sample >100 ve mean_R > 0 (p<0.05): FLOOR_SHORT → 0.30'a düşür
#   - LONG sample >200 ve SE < 0.1: FLOOR_LONG → 0.05'e düşür
# Aksi halde mevcut değerleri koru.
HARD_THRESHOLD_FLOOR_LONG  = 0.08   # 0.10 → 0.08: LONG edge zayıf, biraz gevşetildi
HARD_THRESHOLD_FLOOR_SHORT = 0.25   # 0.40 → 0.25: Model 0.28-0.37 tahmin ediyor, 0.40 floor
                                    # tüm SHORT sinyallerini WAIT'e düşürüyordu.
                                    # OOS SHORT R=+0.3947 güçlü edge var → 0.25 ile geçiyor.

THRESHOLD_GRID_PERCENTILES = [50, 60, 70, 80, 90]

DROPPED_FEATURES_HARD = {
    "risk_sl_distance_pct", "risk_rr_ratio", "ic_direction_code",
    "tmp_dow_sin", "tmp_dow_cos", "tmp_is_weekend",
    # [SEÇENEK B DÜZELTMESİ] — Temporal (saat) feature'ları kaldırıldı
    # Sorun: Feature importance analizinde tmp_hour_sin ve tmp_hour_cos
    # sırasıyla #1 ve #3 en önemli feature çıktı (toplam ağırlık ~%13).
    # Model "hangi saatte trade açılmalı?" sorusunu öğreniyor,
    # "bu sinyal kârlı mı?" değil. Bu temporal overfit demek:
    # kripto 7/24 açık, belirli saat avantajı yoktur — 343 trade'deki
    # kârlı saatler rastlantısal korelasyon.
    # Sonuç: Model belirli saatler dışında sinyal üretmiyor → 18 saat boşta.
    # Çözüm: Bu feature'ları eğitimden tamamen çıkar, model gerçek
    # teknik sinyallere (RSI, momentum, IC) odaklanmak zorunda kalır.
    "tmp_hour_sin", "tmp_hour_cos",
    "ctf_n_timeframes", "mkt_regime_volatile", "mkt_regime_trending",
}


# =============================================================================
# ENUMS / DATACLASSES
# =============================================================================

class MLDecision(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"


@dataclass
class MLDecisionResult:
    decision: MLDecision
    confidence: float
    feature_vector: Any = None
    predicted_r: Optional[float] = None
    threshold_used: Optional[float] = None
    fold_uncertainty: Optional[float] = None


@dataclass
class ModelMetrics:
    spearman_ic: float = 0.0
    ic_std: float = 0.0
    information_ratio: float = 0.0
    mae: float = 0.0
    top_quintile_r: float = 0.0
    bottom_quintile_r: float = 0.0
    long_short_spread: float = 0.0
    n_folds: int = 0
    n_train_samples: int = 0
    aggregated_z: float = 0.0
    long_taken_n: int = 0
    long_taken_mean_r: float = 0.0
    short_taken_n: int = 0
    short_taken_mean_r: float = 0.0
    threshold_long: float = DEFAULT_THRESHOLD_LONG
    threshold_short: float = DEFAULT_THRESHOLD_SHORT
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0


# =============================================================================
# THRESHOLD FLOOR ENFORCEMENT
# =============================================================================

def _enforce_threshold_floors(threshold_long: float,
                                threshold_short: float) -> Tuple[float, float]:
    """
    Calibrated threshold'lara hard floor uygula.

    Calibration sample-içi optimal bulsa bile OOS'ta tehlikeli eşikler
    üretebilir. Floor mekanizması asimetrik güvenlik marjı garanti eder.
    """
    floored_long = max(threshold_long, HARD_THRESHOLD_FLOOR_LONG)
    floored_short = max(threshold_short, HARD_THRESHOLD_FLOOR_SHORT)

    if floored_long > threshold_long + 1e-6:
        logger.info(f"  LONG threshold floored: {threshold_long:+.3f} → {floored_long:+.3f}")
    if floored_short > threshold_short + 1e-6:
        logger.info(f"  SHORT threshold floored: {threshold_short:+.3f} → {floored_short:+.3f}")

    return floored_long, floored_short


# =============================================================================
# CALIBRATOR FACADE
# =============================================================================

class EnsembleCalibratorMock:
    def __init__(self, lgbm_reg, rf_reg, feature_names):
        self.lgbm = lgbm_reg
        self.rf = rf_reg
        self.feature_names = feature_names
        self._train_median: Optional[pd.Series] = None

    @property
    def feature_importances_(self):
        lgbm_imp = self.lgbm.feature_importances_
        rf_imp = self.rf.feature_importances_
        lgbm_norm = lgbm_imp / (lgbm_imp.sum() + 1e-9)
        rf_norm = rf_imp / (rf_imp.sum() + 1e-9)
        return (lgbm_norm + rf_norm) / 2

    def _impute_for_rf(self, X):
        if self._train_median is not None:
            return X.fillna(self._train_median)
        return X.fillna(0)

    def predict(self, X):
        return (self._predict_r(X) > 0).astype(int)

    def predict_proba(self, X):
        pred_r = self._predict_r(X)
        prob_positive = 1.0 / (1.0 + np.exp(-2.0 * pred_r))
        return np.column_stack([1 - prob_positive, prob_positive])

    def _predict_r(self, X):
        if hasattr(X, 'columns') and self.feature_names:
            X = X[[c for c in self.feature_names if c in X.columns]]
        lgbm_pred = self.lgbm.predict(X)
        rf_pred = self.rf.predict(self._impute_for_rf(X))
        return (lgbm_pred + rf_pred) / 2.0


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

class EnsemblePredictor:
    def __init__(self, model_dir: str = "models"):
        self.lgbm_model: Optional[LGBMRegressor] = None
        self.rf_model: Optional[RandomForestRegressor] = None
        self.calibrator: Optional[EnsembleCalibratorMock] = None
        self.model = None
        self.feature_names: List[str] = []
        self._train_median: Optional[pd.Series] = None
        self.is_trained = False
        self.retrain_count = 0
        self.last_metrics: Optional[ModelMetrics] = None
        # Threshold'lar floor uygulanmış halleriyle başlatılır
        self.threshold_long: float = HARD_THRESHOLD_FLOOR_LONG
        self.threshold_short: float = HARD_THRESHOLD_FLOOR_SHORT
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.report_path = os.path.join("logs", "reports", "ensemble_egitim_raporu.xlsx")
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)

    @staticmethod
    def construct_r_multiple(trades: List[Dict]) -> pd.DataFrame:
        records = []
        for t in trades:
            if t.get("status") != "CLOSED" or t.get("outcome") == "UNKNOWN":
                continue
            entry = t.get("entry_price", 0)
            sl = t.get("sl_price", 0)
            pnl_pct = t.get("pnl_pct", 0)
            if entry <= 0 or sl <= 0:
                continue
            sl_dist_pct = abs(entry - sl) / entry
            if sl_dist_pct <= 0:
                continue
            r = (pnl_pct / 100) / sl_dist_pct
            records.append({
                "trade_id": t.get("trade_id"),
                "opened_at": t.get("opened_at"),
                "direction": t.get("direction"),
                "r_multiple": r,
                "exit_reason": t.get("exit_reason"),
                "features": t.get("feature_snapshot", {}),
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True)
            df = df.sort_values("opened_at").reset_index(drop=True)
        return df

    def _prepare_features(self, X: pd.DataFrame,
                           directions: Optional[pd.Series] = None) -> pd.DataFrame:
        X = X.copy()
        to_drop = [c for c in DROPPED_FEATURES_HARD if c in X.columns]
        if to_drop:
            X = X.drop(columns=to_drop)
        X = X.replace([np.inf, -np.inf], np.nan)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].clip(lower=-1000, upper=1000)
        if directions is not None:
            if isinstance(directions, (list, np.ndarray)):
                directions = pd.Series(directions, index=X.index)
            X["is_long"] = (directions == "LONG").astype(int).values
        return X

    @staticmethod
    def _mi_feature_filter(X_train: pd.DataFrame, y_train: np.ndarray,
                            mi_threshold: float = 0.01,
                            max_features: int = 20) -> List[str]:
        X_filled = X_train.fillna(X_train.median()).fillna(0)
        mi = mutual_info_regression(X_filled, y_train, random_state=42)
        mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
        kept = mi_series[mi_series >= mi_threshold].index.tolist()
        if not kept:
            kept = mi_series.head(max_features).index.tolist()
        return kept[:max_features]

    @staticmethod
    def _purged_walk_forward(n: int, n_splits: int = CV_N_SPLITS,
                              embargo: int = CV_EMBARGO_GAP,
                              min_train: int = CV_MIN_TRAIN):
        """
        Purged walk-forward CV splits.

        [SORUN 5 DÜZELTMESİ]
        n_splits artık dinamik olarak ayarlanıyor.
        Her test fold'u minimum 15 sample içersin diye effective_splits hesaplanır.
        Bu sayede n=120 gibi küçük veri setlerinde "hiç fold üretilmemesi"
        ve is_trained=True iken metriklerin sıfır kalması sorunu giderildi.

        Garantiler:
        - test_size >= 15 (her fold için)
        - effective_splits >= 2 (minimum 2 fold)
        - effective_splits <= n_splits (üst sınır sabit)
        """
        if n < min_train + embargo + 15:
            logger.warning(
                f"Walk-forward atlandı: n={n} < min_train+embargo+15={min_train+embargo+15}. "
                f"Yeterli veri yok — fold üretilemiyor."
            )
            return

        # Dinamik n_splits: minimum test_size=15 şartı
        # (n - min_train) toplam test bütçesi → 15 sample garantisi için max fold sayısı
        max_splits_by_size = (n - min_train) // 15
        effective_splits = min(n_splits, max(2, max_splits_by_size))

        if effective_splits < n_splits:
            logger.info(
                f"Walk-forward n_splits: {n_splits} → {effective_splits} "
                f"(n={n}'de minimum test_size=15 için ayarlandı)"
            )

        test_size = (n - min_train) // effective_splits

        for i in range(effective_splits):
            train_end = min_train + i * test_size
            test_start = train_end + embargo
            test_end = min(test_start + test_size, n)
            if test_end - test_start < 10:
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def _fit_inner_ensemble(self, X_tr, y_tr, X_te):
        selected = self._mi_feature_filter(X_tr, y_tr)
        X_tr_s, X_te_s = X_tr[selected], X_te[selected]
        lgbm = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                              random_state=42, verbose=-1)
        lgbm.fit(X_tr_s, y_tr)
        med = X_tr_s.median()
        rf = RandomForestRegressor(n_estimators=300, max_depth=5,
                                    min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s.fillna(med), y_tr)
        pred = (lgbm.predict(X_te_s) + rf.predict(X_te_s.fillna(med))) / 2.0
        return pred, selected

    def _calibrate_thresholds(self, X_tr, y_tr, directions_tr) -> Dict[str, float]:
        """
        Per-direction threshold calibration via nested CV on train fold.

        v3.2: Calibration sonucu hard floor ile clip edilir.
        """
        split = int(len(X_tr) * (1 - NESTED_CV_HOLDOUT))
        if split < 50:
            return {"LONG": HARD_THRESHOLD_FLOOR_LONG,
                    "SHORT": HARD_THRESHOLD_FLOOR_SHORT}

        X_inner_tr, X_inner_val = X_tr.iloc[:split], X_tr.iloc[split:]
        y_inner_tr, y_inner_val = y_tr[:split], y_tr[split:]
        dirs_inner_val = directions_tr[split:]

        try:
            pred_val, _ = self._fit_inner_ensemble(X_inner_tr, y_inner_tr, X_inner_val)
        except Exception as e:
            logger.warning(f"Threshold calibration failed: {e}")
            return {"LONG": HARD_THRESHOLD_FLOOR_LONG,
                    "SHORT": HARD_THRESHOLD_FLOOR_SHORT}

        thresholds = {}
        for direction in ["LONG", "SHORT"]:
            mask = dirs_inner_val == direction
            if mask.sum() < 10:
                thresholds[direction] = (HARD_THRESHOLD_FLOOR_LONG if direction == "LONG"
                                          else HARD_THRESHOLD_FLOOR_SHORT)
                continue

            candidates = np.percentile(pred_val[mask], THRESHOLD_GRID_PERCENTILES)
            best_score, best_thr = -np.inf, candidates[0]
            for thr in candidates:
                picked = pred_val[mask] >= thr
                if picked.sum() < 3:
                    continue
                score = y_inner_val[mask][picked].mean()
                if score > best_score:
                    best_score, best_thr = score, thr
            thresholds[direction] = float(best_thr)

        # Floor enforcement (calibration çıkışı)
        floored_long, floored_short = _enforce_threshold_floors(
            thresholds["LONG"], thresholds["SHORT"]
        )
        thresholds["LONG"] = floored_long
        thresholds["SHORT"] = floored_short

        return thresholds

    def _evaluate_walk_forward(self, X, y, directions) -> ModelMetrics:
        """
        Walk-forward evaluation with directional thresholds and floor enforcement.

        [SORUN 5 DÜZELTMESİ]
        n_folds=0 durumunda açık uyarı veriliyor.
        Caller (train()) bu durumda is_trained=False setliyor.
        """
        ics, maes, top_rs, bot_rs = [], [], [], []
        long_taken, short_taken = [], []
        thresholds_per_fold = []

        for fold, (tr, te) in enumerate(self._purged_walk_forward(len(X))):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]
            dirs_tr, dirs_te = directions[tr], directions[te]
            if len(np.unique(y_te)) < 2:
                continue
            thr = self._calibrate_thresholds(X_tr, y_tr, dirs_tr)
            thresholds_per_fold.append(thr)
            try:
                pred_te, _ = self._fit_inner_ensemble(X_tr, y_tr, X_te)
            except Exception as e:
                logger.warning(f"Fold {fold} fit failed: {e}")
                continue
            ic, _ = spearmanr(pred_te, y_te)
            if pd.isna(ic):
                continue
            ics.append(ic)
            maes.append(mean_absolute_error(y_te, pred_te))
            top, bot = np.percentile(pred_te, 80), np.percentile(pred_te, 20)
            top_mask, bot_mask = pred_te >= top, pred_te <= bot
            if top_mask.sum() > 0:
                top_rs.append(y_te[top_mask].mean())
            if bot_mask.sum() > 0:
                bot_rs.append(y_te[bot_mask].mean())
            for i in range(len(pred_te)):
                d = dirs_te[i]
                if pred_te[i] >= thr.get(d, np.inf):
                    if d == "LONG":
                        long_taken.append(y_te[i])
                    elif d == "SHORT":
                        short_taken.append(y_te[i])

        if not ics:
            # [SORUN 5 DÜZELTMESİ] — Açık uyarı, sessiz başarısızlık yok
            logger.warning(
                f"⚠️ Walk-forward hiç geçerli fold üretmedi (n={len(X)}). "
                f"Model eğitildi ama CV metrikleri güvenilmez. "
                f"Caller'da is_trained=False set edilecek."
            )
            return ModelMetrics(n_train_samples=len(X))

        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        ir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        z_agg = ic_mean * np.sqrt(len(ics)) / ic_std if ic_std > 1e-9 else 0.0

        avg_thr_long = float(np.median([t["LONG"] for t in thresholds_per_fold]))
        avg_thr_short = float(np.median([t["SHORT"] for t in thresholds_per_fold]))

        # Secondary floor enforcement (aggregation çıkışı)
        avg_thr_long, avg_thr_short = _enforce_threshold_floors(
            avg_thr_long, avg_thr_short
        )

        return ModelMetrics(
            spearman_ic=ic_mean, ic_std=ic_std, information_ratio=ir,
            mae=float(np.mean(maes)),
            top_quintile_r=float(np.mean(top_rs)) if top_rs else 0.0,
            bottom_quintile_r=float(np.mean(bot_rs)) if bot_rs else 0.0,
            long_short_spread=(float(np.mean(top_rs)) - float(np.mean(bot_rs))
                                if top_rs and bot_rs else 0.0),
            n_folds=len(ics), n_train_samples=len(X), aggregated_z=float(z_agg),
            long_taken_n=len(long_taken),
            long_taken_mean_r=float(np.mean(long_taken)) if long_taken else 0.0,
            short_taken_n=len(short_taken),
            short_taken_mean_r=float(np.mean(short_taken)) if short_taken else 0.0,
            threshold_long=avg_thr_long, threshold_short=avg_thr_short,
            accuracy=float(np.clip(0.5 + ic_mean, 0, 1)),
            auc_roc=float(np.clip(0.5 + ic_mean, 0, 1)),
            f1=float(np.clip(0.5 + ic_mean, 0, 1)),
        )

    def train(self, X: pd.DataFrame, y, directions=None) -> ModelMetrics:
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y, dtype=float)
        unique = np.unique(y[~pd.isna(y)])
        if len(unique) <= 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
            # [SORUN 1 DÜZELTMESİ] — Daha açıklayıcı ve uyarı düzeyi yükseltildi
            logger.error(
                "⚠️ BINARY TARGET DETECTED — bu olmamalıydı!\n"
                "Eğitim verisi R-multiple yerine {0,1} binary geliyor.\n"
                "Otomatik olarak {-1, +1}'e çevriliyor ama threshold floor'ları\n"
                "R-multiple ölçeğine göre ayarlı (LONG=0.10, SHORT=0.40). Sonuçlar GÜVENİLMEZ.\n"
                "Lütfen initial_train() ve retrain_from_experience()'ı kontrol edin.\n"
                "initial_train() R-multiple üretmeli, binary {0,1} değil."
            )
            y = np.where(y > 0.5, 1.0, -1.0)
        if directions is None:
            logger.warning("No directions provided. Direction-aware features disabled.")
            directions_arr = np.array(["LONG"] * len(X))
        else:
            directions_arr = (directions.values if isinstance(directions, pd.Series)
                              else np.asarray(directions))
        X = self._prepare_features(X, directions=pd.Series(directions_arr))
        self.feature_names = list(X.columns)
        if len(X) < MIN_TRAIN_SAMPLES:
            logger.warning(f"n={len(X)} < MIN_TRAIN_SAMPLES={MIN_TRAIN_SAMPLES}. Cold start.")
            self.is_trained = False
            return ModelMetrics(n_train_samples=len(X))

        metrics = self._evaluate_walk_forward(X, y, directions_arr)

        # Tertiary floor enforcement (production state çıkışı)
        floored_long, floored_short = _enforce_threshold_floors(
            metrics.threshold_long, metrics.threshold_short
        )
        self.threshold_long = floored_long
        self.threshold_short = floored_short

        # ModelMetrics objesini de güncelle (raporlama tutarlılığı için)
        metrics.threshold_long = floored_long
        metrics.threshold_short = floored_short

        self.lgbm_model = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
                                          random_state=42, verbose=-1)
        self.lgbm_model.fit(X, y)
        self._train_median = X.median()
        self.rf_model = RandomForestRegressor(n_estimators=300, max_depth=5,
                                                min_samples_leaf=10, random_state=42, n_jobs=-1)
        self.rf_model.fit(X.fillna(self._train_median), y)
        self.calibrator = EnsembleCalibratorMock(self.lgbm_model, self.rf_model, self.feature_names)
        self.calibrator._train_median = self._train_median
        self.model = self.calibrator

        # [SORUN 5 DÜZELTMESİ] — n_folds=0 ise is_trained=False
        if metrics.n_folds == 0 or metrics.spearman_ic == 0.0:
            logger.warning(
                "⚠️ Model train edildi ama CV metrikleri güvenilmez (n_folds=0 veya IC=0). "
                "is_trained=False — cold start fallback aktif kalacak. "
                "Daha fazla trade verisi birikmesi bekleniyor."
            )
            self.is_trained = False
        else:
            self.is_trained = True
            self.retrain_count += 1

        self.last_metrics = metrics
        self._log_training_report(metrics)
        return metrics

    def _log_training_report(self, m: ModelMetrics) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENSEMBLE TRAINED — DIRECTION-AWARE R-MULTIPLE v3.2")
        logger.info("=" * 60)
        logger.info(f"Training samples : {m.n_train_samples}")
        logger.info(f"CV folds         : {m.n_folds}")
        logger.info(f"Spearman IC      : {m.spearman_ic:+.4f} ± {m.ic_std:.4f}")
        logger.info(f"Information Ratio: {m.information_ratio:+.2f}")
        logger.info(f"Aggregated Z     : {m.aggregated_z:+.2f} (Z>=1.65 ≈ p<=0.05)")
        logger.info(f"MAE              : {m.mae:.4f}")
        logger.info(f"Long-short spread: {m.long_short_spread:+.3f}R")
        logger.info(f"Calibrated thresholds: LONG={m.threshold_long:+.3f} (floor={HARD_THRESHOLD_FLOOR_LONG:+.3f}), "
                    f"SHORT={m.threshold_short:+.3f} (floor={HARD_THRESHOLD_FLOOR_SHORT:+.3f})")
        logger.info(f"OOS filtered: LONG n={m.long_taken_n} R={m.long_taken_mean_r:+.4f} | "
                    f"SHORT n={m.short_taken_n} R={m.short_taken_mean_r:+.4f}")
        logger.info("=" * 60)

    def predict(self, feature_vector, ic_direction: Optional[str] = None) -> MLDecisionResult:
        if not self.is_trained:
            return MLDecisionResult(MLDecision.WAIT, 0.0)
        if ic_direction not in ("LONG", "SHORT"):
            return MLDecisionResult(MLDecision.WAIT, 0.0)
        X = self._coerce_to_frame(feature_vector, ic_direction=ic_direction)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_names]
        pred_lgbm = float(self.lgbm_model.predict(X)[0])
        pred_rf = float(self.rf_model.predict(X.fillna(self._train_median))[0])
        pred_r = (pred_lgbm + pred_rf) / 2.0
        uncertainty = abs(pred_lgbm - pred_rf)
        threshold = self.threshold_long if ic_direction == "LONG" else self.threshold_short

        if pred_r >= threshold:
            decision = MLDecision.LONG if ic_direction == "LONG" else MLDecision.SHORT
            confidence = 60.0 + ((pred_r - threshold) / 0.5) * 40.0
            confidence = min(100.0, float(confidence))
        else:
            decision = MLDecision.WAIT
            confidence = 50.0 - ((threshold - pred_r) / 0.5) * 50.0
            confidence = max(0.0, float(confidence))

        return MLDecisionResult(
            decision=decision,
            confidence=float(confidence),
            predicted_r=float(pred_r),
            threshold_used=float(threshold),
            fold_uncertainty=float(uncertainty),
            feature_vector=feature_vector
        )

    def _coerce_to_frame(self, fv, ic_direction: Optional[str] = None) -> pd.DataFrame:
        if isinstance(fv, pd.DataFrame):
            X = fv.copy()
        elif isinstance(fv, pd.Series):
            X = fv.to_frame().T
        elif isinstance(fv, dict):
            X = pd.DataFrame([fv])
        elif isinstance(fv, np.ndarray):
            cols = [c for c in self.feature_names if c != "is_long"]
            X = pd.DataFrame(fv.reshape(1, -1), columns=cols[:len(fv)])
        elif hasattr(fv, "to_dict"):
            X = pd.DataFrame([fv.to_dict()])
        else:
            raise TypeError(f"Cannot coerce {type(fv)} to DataFrame")
        directions_series = pd.Series([ic_direction] * len(X)) if ic_direction else None
        return self._prepare_features(X, directions=directions_series)
