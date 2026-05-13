# =============================================================================
# ENSEMBLE MODEL v3.3 — CLASSIFICATION (PROBABILITY) BASED DIRECTION-AWARE
# =============================================================================
# Replaces Regression (R-multiple prediction) with Classification (Win Probability).
# Asymmetric thresholds are now based on % probabilities (e.g., 0.52 = 52%).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_TRAIN_SAMPLES = 120
CV_N_SPLITS = 5
CV_EMBARGO_GAP = 5
CV_MIN_TRAIN = 100
NESTED_CV_HOLDOUT = 0.20

# Varsayılan kazanma olasılığı eşikleri (%52 ve %55)
DEFAULT_THRESHOLD_LONG = 0.52
DEFAULT_THRESHOLD_SHORT = 0.55

# === HARD THRESHOLD FLOORS (PROBABILITY) ===
HARD_THRESHOLD_FLOOR_LONG  = 0.50   # Model %50'den emin değilse asla LONG girme
HARD_THRESHOLD_FLOOR_SHORT = 0.52   # Model %52'den emin değilse asla SHORT girme

THRESHOLD_GRID_PERCENTILES = [50, 60, 70, 80, 90]

DROPPED_FEATURES_HARD = {
    "risk_sl_distance_pct", "risk_rr_ratio", "ic_direction_code",
    "tmp_dow_sin", "tmp_dow_cos", "tmp_is_weekend",
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
# THRESHOLD FLOOR AND CEILING ENFORCEMENT
# =============================================================================

def _enforce_threshold_floors(threshold_long: float,
                                threshold_short: float) -> Tuple[float, float]:
    """
    Calibrated olasılık eşiklerine tavan ve taban uygular.
    """
    CEILING_LONG = 0.65   # İhtimal %65'i aşsa bile eşiği makul tut
    CEILING_SHORT = 0.68  

    floored_long = max(threshold_long, HARD_THRESHOLD_FLOOR_LONG)
    floored_short = max(threshold_short, HARD_THRESHOLD_FLOOR_SHORT)

    capped_long = min(floored_long, CEILING_LONG)
    capped_short = min(floored_short, CEILING_SHORT)

    if capped_long != threshold_long:
        logger.info(f"  LONG prob threshold adjusted: {threshold_long:.3f} → {capped_long:.3f}")
    if capped_short != threshold_short:
        logger.info(f"  SHORT prob threshold adjusted: {threshold_short:.3f} → {capped_short:.3f}")

    return capped_long, capped_short


# =============================================================================
# CALIBRATOR FACADE
# =============================================================================

class EnsembleCalibratorMock:
    def __init__(self, lgbm_clf, rf_clf, feature_names):
        self.lgbm = lgbm_clf
        self.rf = rf_clf
        self.feature_names = feature_names

    @property
    def feature_importances_(self):
        lgbm_imp = self.lgbm.feature_importances_
        rf_imp = self.rf.feature_importances_
        lgbm_norm = lgbm_imp / (lgbm_imp.sum() + 1e-9)
        rf_norm = rf_imp / (rf_imp.sum() + 1e-9)
        return (lgbm_norm + rf_norm) / 2

    def _impute_for_rf(self, X):
        return X.fillna(-999)

    def predict(self, X):
        return (self._predict_prob(X) > 0.5).astype(int)

    def predict_proba(self, X):
        prob_positive = self._predict_prob(X)
        return np.column_stack([1 - prob_positive, prob_positive])

    def _predict_prob(self, X):
        if hasattr(X, 'columns') and self.feature_names:
            X = X[[c for c in self.feature_names if c in X.columns]]
        lgbm_pred = self.lgbm.predict_proba(X)[:, 1]
        rf_pred = self.rf.predict_proba(self._impute_for_rf(X))[:, 1]
        return (lgbm_pred + rf_pred) / 2.0


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

class EnsemblePredictor:
    def __init__(self, model_dir: str = "models"):
        self.lgbm_model: Optional[LGBMClassifier] = None
        self.rf_model: Optional[RandomForestClassifier] = None
        self.calibrator: Optional[EnsembleCalibratorMock] = None
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.retrain_count = 0           
        self.experience_retrain_count = 0  
        self.last_metrics: Optional[ModelMetrics] = None
        
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
        X_filled = X_train.fillna(-999)
        mi = mutual_info_classif(X_filled, y_train, random_state=42)
        mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
        kept = mi_series[mi_series >= mi_threshold].index.tolist()
        if not kept:
            kept = mi_series.head(max_features).index.tolist()
        return kept[:max_features]

    @staticmethod
    def _purged_walk_forward(n: int, n_splits: int = CV_N_SPLITS,
                              embargo: int = CV_EMBARGO_GAP,
                              min_train: int = CV_MIN_TRAIN):
        if n < min_train + embargo + 15:
            logger.warning(
                f"Walk-forward atlandı: n={n} < min_train+embargo+15={min_train+embargo+15}. "
                f"Yeterli veri yok — fold üretilemiyor."
            )
            return

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
        # Y_tr (R-multiple) değişkenini Binary (1=Kâr, 0=Zarar) yapıyoruz
        y_tr_bin = (y_tr > 0).astype(int)
        
        selected = self._mi_feature_filter(X_tr, y_tr_bin)
        X_tr_s, X_te_s = X_tr[selected], X_te[selected]
        
        lgbm = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            min_child_samples=15, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbose=-1
        )
        lgbm.fit(X_tr_s, y_tr_bin)
        
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=4, max_features='sqrt',
            min_samples_leaf=15, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr_s.fillna(-999), y_tr_bin)
        
        # Sınıflandırma algoritmalarından "Kazanma" (1) sınıfının olasılığını al
        pred_lgbm = lgbm.predict_proba(X_te_s)[:, 1]
        pred_rf = rf.predict_proba(X_te_s.fillna(-999))[:, 1]
        
        pred = (pred_lgbm + pred_rf) / 2.0
        return pred, selected

    def _calibrate_thresholds(self, X_tr, y_tr, directions_tr) -> Dict[str, float]:
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
                # Gerçek R-multiple'ı kullanarak bu olasılık eşiğinin ortalama getirisini hesaplıyoruz
                score = y_inner_val[mask][picked].mean()
                if score > best_score:
                    best_score, best_thr = score, thr
            thresholds[direction] = float(best_thr)

        floored_long, floored_short = _enforce_threshold_floors(
            thresholds["LONG"], thresholds["SHORT"]
        )
        thresholds["LONG"] = floored_long
        thresholds["SHORT"] = floored_short

        return thresholds

    def _evaluate_walk_forward(self, X, y, directions) -> ModelMetrics:
        ics, maes, top_rs, bot_rs = [], [], [], []
        long_taken, short_taken = [], []
        thresholds_per_fold = []

        # Y'yi (Hedef değişkeni) değerlendirme aşaması için de Binary'e (1 ve 0) çeviriyoruz
        y_bin = (y > 0).astype(int) 

        for fold, (tr, te) in enumerate(self._purged_walk_forward(len(X))):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]               # Gerçek R-multiple (PnL hesaplamak için lazım)
            y_tr_bin, y_te_bin = y_bin[tr], y_bin[te] # 1 ve 0'lar (Korelasyon için lazım)
            dirs_tr, dirs_te = directions[tr], directions[te]
            
            if len(np.unique(y_te_bin)) < 2:
                continue
                
            thr = self._calibrate_thresholds(X_tr, y_tr, dirs_tr)
            thresholds_per_fold.append(thr)
            
            try:
                # pred_te artık 0 ile 1 arasında bir "kazanma olasılığı" (% ihtimal) döndürüyor
                pred_te, _ = self._fit_inner_ensemble(X_tr, y_tr, X_te)
            except Exception as e:
                logger.warning(f"Fold {fold} fit failed: {e}")
                continue
                
            # KRİTİK DÜZELTME: Olasılık tahmini (pred_te) ile KAZANDI/KAYBETTİ durumunu (y_te_bin) kıyasla!
            # Eski hatalı kod: spearmanr(pred_te, y_te) -> Olasılık ile gerçek dolar kazancını kıyaslıyordu.
            ic, _ = spearmanr(pred_te, y_te_bin)
            
            if pd.isna(ic):
                continue
            ics.append(ic)
            
            # MAE'yi de olasılık üzerinden hesaplıyoruz
            maes.append(mean_absolute_error(y_te_bin, pred_te))
            
            top, bot = np.percentile(pred_te, 80), np.percentile(pred_te, 20)
            top_mask, bot_mask = pred_te >= top, pred_te <= bot
            
            # Buralarda gerçek kazanç/kayıp (y_te) kullanmaya devam ediyoruz ki R-Multiple hesabı şaşmasın
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
            logger.warning(
                f"⚠️ Walk-forward hiç geçerli fold üretmedi (n={len(X)}). "
                f"Model eğitildi ama CV metrikleri güvenilmez."
            )
            return ModelMetrics(n_train_samples=len(X))

        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        ir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        z_agg = ic_mean * np.sqrt(len(ics)) / ic_std if ic_std > 1e-9 else 0.0

        avg_thr_long = float(np.median([t["LONG"] for t in thresholds_per_fold]))
        avg_thr_short = float(np.median([t["SHORT"] for t in thresholds_per_fold]))

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

        # Y (Hedef) değişkeni _evaluate_walk_forward'a hala R-Multiple olarak gider
        metrics = self._evaluate_walk_forward(X, y, directions_arr)

        floored_long, floored_short = _enforce_threshold_floors(
            metrics.threshold_long, metrics.threshold_short
        )
        self.threshold_long = floored_long
        self.threshold_short = floored_short

        metrics.threshold_long = floored_long
        metrics.threshold_short = floored_short

        # Sınıflandırma için y'yi 1-0 yapıyoruz
        y_bin = (y > 0).astype(int)

        self.lgbm_model = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.03,
            min_child_samples=15, reg_alpha=0.5, reg_lambda=0.5,
            random_state=42, verbose=-1
        )
        self.lgbm_model.fit(X, y_bin)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=4, max_features='sqrt',
            min_samples_leaf=15, random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X.fillna(-999), y_bin)
        
        self.calibrator = EnsembleCalibratorMock(self.lgbm_model, self.rf_model, self.feature_names)
        self.model = self.calibrator

        if metrics.n_folds == 0 or metrics.spearman_ic == 0.0:
            logger.warning(
                "⚠️ Model train edildi ama CV metrikleri güvenilmez. "
                "is_trained=False — cold start fallback aktif kalacak."
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
        logger.info("📊 ENSEMBLE TRAINED — PROBABILITY-BASED CLASSIFICATION v3.3")
        logger.info("=" * 60)
        logger.info(f"Training samples : {m.n_train_samples}")
        logger.info(f"CV folds         : {m.n_folds}")
        logger.info(f"Spearman IC      : {m.spearman_ic:+.4f} ± {m.ic_std:.4f}")
        logger.info(f"Information Ratio: {m.information_ratio:+.2f}")
        logger.info(f"Long-short spread: {m.long_short_spread:+.3f}R")
        logger.info(f"Calibrated Thresholds (Win Prob): LONG={m.threshold_long:.3f}, SHORT={m.threshold_short:.3f}")
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
        
        # Regressor yerine predict_proba kullanıp kazanma (1) olasılığını alıyoruz
        pred_lgbm = float(self.lgbm_model.predict_proba(X)[0, 1])
        pred_rf = float(self.rf_model.predict_proba(X.fillna(-999))[0, 1])
        
        pred_prob = (pred_lgbm + pred_rf) / 2.0
        uncertainty = abs(pred_lgbm - pred_rf)
        threshold = self.threshold_long if ic_direction == "LONG" else self.threshold_short
        
        # Expected value artık doğrudan % olasılık (Örn: 0.58 = %58 kazanma şansı)
        expected_value = float(pred_prob)

        if expected_value >= threshold:
            decision = MLDecision.LONG if ic_direction == "LONG" else MLDecision.SHORT
            confidence = expected_value * 100.0  
            confidence = min(100.0, max(50.0, float(confidence)))
        else:
            decision = MLDecision.WAIT
            confidence = expected_value * 100.0
            confidence = max(0.0, min(50.0, float(confidence)))

        return MLDecisionResult(
            decision=decision,
            confidence=float(confidence),
            predicted_r=float(pred_prob), # Geriye dönük uyumluluk için değişken ismini koruduk
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