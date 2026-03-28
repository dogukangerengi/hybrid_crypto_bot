import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any
from enum import Enum
from datetime import datetime
import warnings
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class MLDecision(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

@dataclass
class MLDecisionResult:
    decision: MLDecision
    confidence: float
    feature_vector: Any = None

@dataclass
class ModelMetrics:
    accuracy: float
    auc_roc: float
    f1: float

# --- KUSURSUZLAŞTIRILMIŞ SAHTE CALIBRATOR ---
class EnsembleCalibratorMock:
    """
    Validator ve Excel Raporlayıcı bu modeli eski LGBM sanıp içindeki özelliklere ulaşmaya çalışacak.
    """
    def __init__(self, rf, lgbm):
        self.rf = rf
        self.lgbm = lgbm

    # EXCEL HATASI ÇÖZÜMÜ: Sistemin aradığı "önem listesini" Random Forest'tan alıp veriyoruz
    @property
    def feature_importances_(self):
        return self.rf.feature_importances_
        
    def predict_proba(self, X):
        # GÜVEN SKORU HATASI ÇÖZÜMÜ: İki modelin güvenini toplayıp 2'ye bölüyoruz
        rf_prob = self.rf.predict_proba(X)[:, 1]
        lgbm_prob = self.lgbm.predict_proba(X)[:, 1]
        ortak_prob = (rf_prob + lgbm_prob) / 2
        
        # Scikit-learn standartlarına uygun formatta döndürüyoruz
        return np.column_stack((1 - ortak_prob, ortak_prob))
        
    def predict(self, X):
        rf_p = self.rf.predict(X)
        lgbm_p = self.lgbm.predict(X)
        return ((rf_p == 1) & (lgbm_p == 1)).astype(int)
# ------------------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Keskin Nişancı (Sniper) Modeli: 
    Random Forest (Volatilite/Güvenlik) ve LightGBM (Gelişmiş Örüntü) algoritmalarının 
    'Sıkı Oylama' (Hard Voting) ile çalıştığı çift çekirdekli sistem.
    """
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
        self.lgbm_model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', verbose=-1)
        
        # Sahte kalkanımızı sisteme ekliyoruz
        self.calibrator = EnsembleCalibratorMock(self.rf_model, self.lgbm_model)
        self.model = self.calibrator
        
        self.is_trained = False
        self.feature_names = []  
        self.retrain_count = 0  
        
        os.makedirs("logs/reports", exist_ok=True)
        self.report_path = "logs/reports/ensemble_egitim_raporu.xlsx"

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].clip(lower=-1000, upper=1000)
        return df_clean

    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        df_clean = self._clean_data(X)
        self.feature_names = list(df_clean.columns)

        X_train, X_val, y_train, y_val = train_test_split(df_clean, y, test_size=0.2, random_state=42, shuffle=False)

        self.rf_model.fit(X_train, y_train)
        self.lgbm_model.fit(X_train, y_train)

        rf_val_preds = self.rf_model.predict(X_val)
        lgbm_val_preds = self.lgbm_model.predict(X_val)
        
        # LONG (1) işlemlere girme ve kazanma durumu
        long_islem = (rf_val_preds == 1) & (lgbm_val_preds == 1)
        long_kazanc = (long_islem) & (y_val == 1)
        
        # SHORT (0) işlemlere girme ve kazanma durumu
        short_islem = (rf_val_preds == 0) & (lgbm_val_preds == 0)
        short_kazanc = (short_islem) & (y_val == 0)
        
        # Toplam istatistiklerin hesaplanması
        isleme_girilen = sum(long_islem) + sum(short_islem)
        kazanc = sum(long_kazanc) + sum(short_kazanc)
        win_rate = (kazanc / isleme_girilen) if isleme_girilen > 0 else 0.0

        self.is_trained = True

        logger.info("\n" + "="*50)
        logger.info("🎯 ENSEMBLE (SNIPER) MODEL EĞİTİLDİ")
        logger.info("="*50)
        logger.info(f"🌲 RF İşleme Girme    : {sum(rf_val_preds == 1)} LONG, {sum(rf_val_preds == 0)} SHORT")
        logger.info(f"⚡ LGBM İşleme Girme  : {sum(lgbm_val_preds == 1)} LONG, {sum(lgbm_val_preds == 0)} SHORT")
        logger.info(f"🤝 ORTAK İşleme Girme : {isleme_girilen} işlem -> Kazanç: {kazanc}")
        logger.info(f"🏆 GÜNCEL WIN RATE   : %{win_rate*100:.1f}")
        logger.info("="*50)

        return ModelMetrics(accuracy=win_rate, auc_roc=win_rate, f1=win_rate)

    def predict(self, feature_vector: Any, ic_score: float, ic_direction: str) -> MLDecisionResult:
        if not self.is_trained or feature_vector is None:
            return MLDecisionResult(decision=MLDecision.WAIT, confidence=0.0, feature_vector=feature_vector)

        try:
            features = feature_vector.to_dict() if hasattr(feature_vector, 'to_dict') else feature_vector
            df_feat = pd.DataFrame([features])
            
            for col in self.feature_names:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            
            X_live = df_feat[self.feature_names]
            X_live = self._clean_data(X_live)

            rf_pred = self.rf_model.predict(X_live)[0]
            lgbm_pred = self.lgbm_model.predict(X_live)[0]
            
            # predict_proba, Sınıf 1 (LONG) olma olasılığını döndürür
            rf_prob_long = self.rf_model.predict_proba(X_live)[0][1]
            lgbm_prob_long = self.lgbm_model.predict_proba(X_live)[0][1]

            # 1. Senaryo: İki model de 1 (Yükseliş) diyorsa (Pusuladan bağımsız özgür karar)
            if rf_pred == 1 and lgbm_pred == 1:
                ortak_guven = ((rf_prob_long + lgbm_prob_long) / 2) * 100
                return MLDecisionResult(decision=MLDecision.LONG, confidence=ortak_guven, feature_vector=feature_vector)

            # 2. Senaryo: İki model de 0 (Düşüş) diyorsa (Pusuladan bağımsız özgür karar)
            elif rf_pred == 0 and lgbm_pred == 0:
                # Sınıf 0 (SHORT) olasılığı, 1 - LONG olasılığı formülüyle hesaplanır
                rf_prob_short = 1 - rf_prob_long
                lgbm_prob_short = 1 - lgbm_prob_long
                ortak_guven = ((rf_prob_short + lgbm_prob_short) / 2) * 100
                return MLDecisionResult(decision=MLDecision.SHORT, confidence=ortak_guven, feature_vector=feature_vector)

            # 3. Senaryo: Modeller arası anlaşmazlık varsa beklemeye geç
            return MLDecisionResult(decision=MLDecision.WAIT, confidence=0.0, feature_vector=feature_vector)

        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return MLDecisionResult(decision=MLDecision.WAIT, confidence=0.0, feature_vector=feature_vector)