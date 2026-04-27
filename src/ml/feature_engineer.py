# =============================================================================
# FEATURE ENGINEER — IC + MARKET CONTEXT → LIGHTGBM FEATURE MATRİX
# =============================================================================
# Amaç: Mevcut IC analiz pipeline'ından gelen verileri ve ham OHLCV
#        datasını LightGBM'in anlayacağı sayısal feature vektörüne çevirmek.
#
# Gemini'den Farkı:
# - Gemini: "Bu bağlamda sinyal mantıklı mı?" sorusunu semantik cevaplıyordu
# - FeatureEngineer: Aynı soruyu SAYISAL feature'lar + istatistiksel model ile cevaplar
# - Avantaj: Geçmiş trade'lerden öğrenir, feedback loop var, deterministik
#
# Feature Kategorileri (6 grup, ~45 feature):
# ┌──────────────────────────────────────────────────────────────┐
# │ 1. IC-Based Features     : IC skorları, anlamlılık, tutarlılık│
# │ 2. Market Context         : Volatilite, hacim, rejim          │
# │ 3. Cross-Timeframe        : TF uyumu, skor dağılımı           │
# │ 4. Price Action           : Momentum, mean-reversion, trend   │
# │ 5. Risk Metrics           : ATR, RR oranı, SL/TP mesafesi     │
# │ 6. Temporal               : Saat, gün, session bilgisi        │
# └──────────────────────────────────────────────────────────────┘
#
# İstatistiksel Dikkat:
# - Tüm feature'lar SADECE t ve öncesi veriyi kullanır (look-ahead bias yok)
# - Log-transform: Sağa çarpık dağılımlara uygulanır (hacim, ATR)
# - Z-score: Karşılaştırılabilirlik için normalize
# - NaN handling: LightGBM native NaN desteğini kullanır (impute gerekmiyor)
#
# Kullanım:
#   from ml.feature_engineer import FeatureEngineer
#   engineer = FeatureEngineer()
#   feature_vec = engineer.build_features(coin_analysis, ohlcv_df)
#   X = feature_vec.to_array()   # numpy array (LightGBM input)
# =============================================================================

import sys                                     # Path ayarları için
import numpy as np                             # Sayısal hesaplamalar — feature vektörü numpy array olacak
import pandas as pd                            # DataFrame işlemleri — OHLCV ve indikatör verisi
import logging                                 # Yapılandırılmış log mesajları
from pathlib import Path                       # Platform-bağımsız dosya yolu
from typing import Dict, List, Optional, Any, Tuple  # Tip belirteçleri — mypy uyumlu
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfları
from datetime import datetime, timezone        # Zaman damgası ve temporal feature'lar
from enum import Enum                          # Sabit değer enumları (LONG/SHORT/WAIT)

# Logger — bu modülün log mesajları 'ml.feature_engineer' namespace'inde
logger = logging.getLogger(__name__)


# =============================================================================
# ENUM & DATACLASS TANIMLARI
# =============================================================================

class MLDecision(Enum):
    """
    ML modelinin nihai kararı.
    Gemini'deki AIDecision enum'unun ML karşılığı.
    """
    LONG = "LONG"                              # Alış sinyali
    SHORT = "SHORT"                            # Satış sinyali
    WAIT = "WAIT"                              # İşlem yapma

    @classmethod
    def from_direction(cls, direction: str) -> 'MLDecision':
        """
        String yön ifadesini MLDecision enum'una çevir.
        IC analiz sonucundaki 'LONG'/'SHORT'/'NEUTRAL' → enum.
        """
        d = (direction or "").upper()          # None ve küçük harf koruması
        if d in ("LONG", "BUY", "BULLISH"):
            return cls.LONG
        elif d in ("SHORT", "SELL", "BEARISH"):
            return cls.SHORT
        return cls.WAIT


@dataclass
class MLDecisionResult:
    """
    ML pipeline'ının nihai karar objesi.
    Gemini'deki AIDecisionResult'ın ML karşılığı.
    
    Bu obje execution modülüne gönderilir.
    Mevcut main.py'deki `_evaluate_coin` fonksiyonunda
    AIDecisionResult yerine bu kullanılacak.
    """
    # ML kararı
    decision: MLDecision                       # LONG / SHORT / WAIT — nihai yön kararı
    confidence: float                          # Model güven skoru (0-100) — LightGBM probability × 100
    reasoning: str                             # Karar gerekçesi (Türkçe, okunabilir)

    # IC bilgileri (mevcut sistemle uyumluluk)
    gate_action: str = "NO_TRADE"              # NO_TRADE / REPORT_ONLY / FULL_TRADE — gate keeper sonucu
    ic_score: float = 0.0                      # IC composite skoru (0-100)

    # Risk parametreleri (risk_manager'dan gelir, burada saklanır)
    entry_price: float = 0.0                   # Giriş fiyatı ($)
    sl_price: float = 0.0                      # Stop-Loss fiyatı ($)
    tp_price: float = 0.0                      # Take-Profit fiyatı ($)
    risk_reward: float = 0.0                   # Risk/Reward oranı (TP mesafesi / SL mesafesi)
    atr_multiplier: float = 1.5                # ATR çarpanı — SL mesafesi = ATR × multiplier

    # Model meta bilgileri
    model_version: str = ""                    # Eğitilmiş model versiyonu (tarih bazlı)
    feature_importance_top3: List[str] = field(default_factory=list)  # En önemli 3 feature adı
    timestamp: str = ""                        # Karar zamanı (UTC ISO format)

    # Feature vektörü (trade_memory'ye feature_snapshot olarak kaydedilir)
    # predict() sırasında kullanılan feature'ların kopyası.
    # Model retrain'de bu snapshot kullanılarak her trade'in "o andaki"
    # piyasa koşulları yeniden değerlendirilebilir.
    feature_vector: Optional['MLFeatureVector'] = None

    def should_execute(self) -> bool:
        """
        İşlem gönderilmeli mi?
        
        Koşullar (tümü sağlanmalı):
        1. Gate keeper FULL_TRADE demiş (IC >= eşik)
        2. ML kararı LONG veya SHORT (WAIT değil)
        3. Model güveni >= 60 (düşük güvenli sinyalleri filtrele)
        """
        return (
            self.gate_action == "FULL_TRADE"           # IC eşiğini geçmiş
            and self.decision in [MLDecision.LONG, MLDecision.SHORT]  # Net yön var
            and self.confidence >= 53                   # Model yeterince emin
        )

    def summary(self) -> str:
        """
        Telegram mesajı için okunabilir özet.
        Mevcut AIDecisionResult.summary() ile aynı format.
        """
        dec_emoji = {
            MLDecision.LONG: "🟢 LONG",
            MLDecision.SHORT: "🔴 SHORT",
            MLDecision.WAIT: "⏳ BEKLE"
        }
        gate_emoji = {
            "NO_TRADE": "🚫",
            "REPORT_ONLY": "📋",
            "FULL_TRADE": "✅"
        }

        lines = [
            f"🤖 ML Karar: {dec_emoji.get(self.decision, '❓')}",
            f"🎯 Güven: {self.confidence:.0f}/100",
            f"📊 IC Skor: {self.ic_score:.0f}/100",
            f"🚦 Gate: {gate_emoji.get(self.gate_action, '❓')} {self.gate_action}",
            f"",
            f"💬 {self.reasoning}",
        ]

        if self.should_execute():
            lines.extend([
                f"",
                f"📍 Entry: ${self.entry_price:,.2f}",
                f"🛑 SL: ${self.sl_price:,.2f}",
                f"🎯 TP: ${self.tp_price:,.2f}",
                f"⚖️ RR: {self.risk_reward:.1f}",
            ])

        if self.feature_importance_top3:
            lines.append(f"📈 Top Features: {', '.join(self.feature_importance_top3)}")

        return "\n".join(lines)


@dataclass
class MLFeatureVector:
    """
    Tek bir coin + timeframe analizi için feature vektörü.
    
    Her feature grubunu ayrı dict olarak tutar:
    - Debugging kolaylığı: hangi feature hangi değeri alıyor görmek kolay
    - LightGBM'e gönderirken to_array() veya to_dict() ile düzleştirilir
    - Feature isimlerini de taşır (model interpretability için)
    
    İstatistiksel Not:
    - NaN değerler korunur — LightGBM native NaN split desteği var
    - Bu sayede imputation bias'tan kaçınılır
    """
    # Meta bilgiler (feature DEĞİL, sadece takip için)
    symbol: str = ""                           # 'SOL/USDT:USDT'
    coin: str = ""                             # 'SOL'
    timestamp: str = ""                        # Feature oluşturma zamanı

    # Feature grupları (her biri dict — key: feature_adı, value: sayısal değer)
    ic_features: Dict[str, float] = field(default_factory=dict)        # IC bazlı feature'lar
    market_features: Dict[str, float] = field(default_factory=dict)    # Market context
    cross_tf_features: Dict[str, float] = field(default_factory=dict)  # Cross-timeframe
    price_features: Dict[str, float] = field(default_factory=dict)     # Price action
    risk_features: Dict[str, float] = field(default_factory=dict)      # Risk metrikleri
    temporal_features: Dict[str, float] = field(default_factory=dict)  # Zaman bazlı

    # Hedef değişken (eğitim sırasında doldurulur, tahmin sırasında boş)
    target: Optional[float] = None             # 1.0 = karlı trade, 0.0 = zararlı, None = bilinmiyor
    target_direction: Optional[str] = None     # 'LONG' veya 'SHORT' (hangi yönde trade açılmıştı)

    def to_dict(self) -> Dict[str, float]:
        """
        Tüm feature'ları tek bir düz dict'e birleştir.
        LightGBM pd.DataFrame input'u için kullanılır.
        
        Prefix Convention:
        - ic_* : IC bazlı (örn: ic_confidence, ic_top_abs)
        - mkt_* : Market context (örn: mkt_volatility_24h)
        - ctf_* : Cross-timeframe (örn: ctf_tf_agreement)
        - px_* : Price action (örn: px_momentum_5)
        - risk_* : Risk metrikleri (örn: risk_atr_pct)
        - tmp_* : Temporal (örn: tmp_hour_sin)
        """
        flat = {}                              # Düzleştirilmiş feature dict
        flat.update(self.ic_features)          # IC feature'ları ekle
        flat.update(self.market_features)      # Market feature'ları ekle
        flat.update(self.cross_tf_features)    # Cross-TF feature'ları ekle
        flat.update(self.price_features)       # Price action feature'ları ekle
        flat.update(self.risk_features)        # Risk feature'ları ekle
        flat.update(self.temporal_features)    # Temporal feature'ları ekle
        return flat

    def to_array(self) -> np.ndarray:
        """
        Feature vektörünü numpy array'e çevir.
        LightGBM predict() input'u için kullanılır.
        Sıralama to_dict() key order ile aynı (Python 3.7+ dict order garanti).
        """
        return np.array(list(self.to_dict().values()), dtype=np.float64)

    def feature_names(self) -> List[str]:
        """
        Feature isimlerini sıralı liste olarak döndür.
        LightGBM model eğitimi sırasında feature_name parametresi için.
        """
        return list(self.to_dict().keys())

    def feature_count(self) -> int:
        """Toplam feature sayısı."""
        return len(self.to_dict())


# =============================================================================
# ANA FEATURE ENGINEER SINIFI
# =============================================================================

class FeatureEngineer:
    """
    IC analiz sonuçları + OHLCV verisi → LightGBM feature matrix.
    
    Bu sınıf Gemini optimizer'ın yerini almak için tasarlandı:
    - Gemini: prompt + API call → JSON karar
    - FeatureEngineer: sayısal feature'lar → LightGBM tahmin
    
    Pipeline Entegrasyonu:
    main.py'deki _evaluate_coin() fonksiyonunda:
    
    ESKİ (Gemini):
        ai_input = AIAnalysisInput(...)
        ai_result = self.ai_optimizer.get_decision(ai_input)
    
    YENİ (ML):
        features = self.feature_engineer.build_features(analysis, df)
        ml_result = self.lgbm_model.predict(features)
    
    Feature Sayıları:
    - IC-Based      : ~12 feature
    - Market Context: ~7 feature
    - Cross-TF      : ~6 feature
    - Price Action   : ~10 feature
    - Risk Metrics   : ~5 feature
    - Temporal       : ~5 feature
    ─────────────────────────────
    TOPLAM           : ~45 feature
    """

    def __init__(self, verbose: bool = True):
        """
        FeatureEngineer başlatır.
        
        Parameters:
        ----------
        verbose : bool
            True → feature istatistiklerini logla (debug için faydalı)
        """
        self.verbose = verbose                 # Detaylı log mesajları açık/kapalı

    # =========================================================================
    # ANA METOD: TÜM FEATURE'LARI BİRLEŞTİR
    # =========================================================================

    def build_features(
        self,
        analysis: Any,                         # CoinAnalysisResult objesi (main.py'den)
        ohlcv_df: Optional[pd.DataFrame] = None,  # En iyi TF'nin OHLCV verisi
        all_tf_analyses: Optional[List[Dict]] = None  # Tüm TF analizleri (_analyze_coin'den)
    ) -> MLFeatureVector:
        """
        Tek bir coin için tüm feature'ları hesapla ve birleştir.
        
        Bu metod main.py'deki _analyze_coin() sonrasında çağrılır.
        CoinAnalysisResult + OHLCV DataFrame → MLFeatureVector.
        
        Parameters:
        ----------
        analysis : CoinAnalysisResult
            IC analiz sonucu (ic_confidence, ic_direction, category_tops, tf_rankings, vs.)
            
        ohlcv_df : pd.DataFrame, optional
            En iyi timeframe'in OHLCV + indikatör verisi
            None ise price action feature'ları NaN olur (LightGBM handle eder)
            
        all_tf_analyses : List[Dict], optional
            Tüm timeframe analizlerinin listesi (cross-TF feature'lar için)
            Her dict: {'tf', 'composite', 'direction', 'sig_count', 'top_ic', ...}
            None ise cross-TF feature'lar ortalama değer alır
            
        Returns:
        -------
        MLFeatureVector
            ~45 feature içeren vektör, LightGBM'e hazır
        """
        vec = MLFeatureVector(                 # Boş feature vektörü oluştur
            symbol=getattr(analysis, 'symbol', ''),      # Coin sembolü
            coin=getattr(analysis, 'coin', ''),           # Kısa coin adı
            timestamp=datetime.now(timezone.utc).isoformat(),  # Feature oluşturma zamanı
        )

        # Her feature grubunu hesapla ve vektöre ekle
        vec.ic_features = self._build_ic_features(analysis)           # IC bazlı
        vec.market_features = self._build_market_features(analysis)   # Market context
        vec.cross_tf_features = self._build_cross_tf_features(        # Cross-timeframe
            analysis, all_tf_analyses
        )
        vec.price_features = self._build_price_features(ohlcv_df)     # Price action
        vec.risk_features = self._build_risk_features(analysis)       # Risk metrikleri
        vec.temporal_features = self._build_temporal_features(         # Temporal
            bar_dt=getattr(analysis, 'bar_dt', None)  # [EĞİTİM FIX] bar timestamp varsa kullan
        )

        if self.verbose:
            n_features = vec.feature_count()   # Toplam feature sayısı
            n_nan = sum(                       # NaN feature sayısı (LightGBM handle edecek)
                1 for v in vec.to_dict().values()
                if v is None or (isinstance(v, float) and np.isnan(v))
            )
            logger.debug(
                f"  🧬 Feature: {n_features} toplam | "
                f"{n_nan} NaN | Coin: {vec.coin}"
            )

        return vec

    # =========================================================================
    # 1. IC-BASED FEATURES (~12 feature)
    # =========================================================================

    def _build_ic_features(self, analysis: Any) -> Dict[str, float]:
        """
        IC analiz sonuçlarını sayısal feature'lara çevir.
        
        Bu feature'lar mevcut sistemdeki composite_score hesabının
        ayrıştırılmış hali — LightGBM her bileşenin ağırlığını
        kendisi öğrenecek (hardcoded %40/%25/%15/%20 yerine).
        
        Parameters:
        ----------
        analysis : CoinAnalysisResult
            IC analiz sonucu objesi
            
        Returns:
        -------
        Dict[str, float]
            Feature adı → sayısal değer
        """
        features = {}

        # ── Ana IC Metrikleri ──
        # IC Confidence: Composite skor (0-100 arası)
        # Mevcut sistemde: top_norm×0.4 + avg_norm×0.25 + cnt_norm×0.15 + cons_norm×0.20
        # Burada bileşenleri ayrıştırıyoruz → LightGBM optimal ağırlıkları öğrenecek
        features['ic_confidence'] = float(                   # IC composite skoru (0-100)
            getattr(analysis, 'ic_confidence', 0.0)
        )

        # Top IC: En güçlü indikatörün mutlak IC değeri
        # Yüksek = güçlü tekil sinyal var (0.02-0.40 arası tipik)
        features['ic_top_abs'] = float(                      # En iyi |IC| değeri
            abs(getattr(analysis, 'top_ic', 0.0))
        )

        # Anlamlı indikatör sayısı (FDR correction sonrası)
        # Çok sayıda anlamlı indikatör = piyasa okunabilir durumda
        features['ic_sig_count'] = float(                    # FDR-significant indikatör sayısı
            getattr(analysis, 'significant_count', 0)
        )

        # IC Yön kodlaması (numerik):
        # LONG = +1, SHORT = -1, NEUTRAL = 0
        # LightGBM split'leri bu kodlama ile yönü ayırt edebilir
        direction = getattr(analysis, 'ic_direction', 'NEUTRAL')
        features['ic_direction_code'] = (                    # Yön kodu: +1 LONG, -1 SHORT, 0 NEUTRAL
            1.0 if direction == 'LONG'
            else -1.0 if direction == 'SHORT'
            else 0.0
        )

        # ── Kategori Bazlı IC'ler ──
        # Her kategori (trend/momentum/volatility/volume) için en iyi IC
        # Gemini bunu prompt'ta "hangi kategoriler uyumlu" olarak değerlendiriyordu
        # LightGBM'de bunlar ayrı feature → kategori etkileşimlerini öğrenebilir
        category_tops = getattr(analysis, 'category_tops', {})

        for cat in ['trend', 'momentum', 'volatility', 'volume']:
            if cat in category_tops and 'ic' in category_tops[cat]:
                # Ham IC değeri (negatif = SHORT sinyali, pozitif = LONG sinyali)
                features[f'ic_cat_{cat}'] = float(           # Kategori IC değeri (işaretli)
                    category_tops[cat]['ic']
                )
                # Mutlak IC (sinyal gücü, yönden bağımsız)
                features[f'ic_cat_{cat}_abs'] = float(       # Kategori |IC| değeri (güç ölçüsü)
                    abs(category_tops[cat]['ic'])
                )
            else:
                features[f'ic_cat_{cat}'] = np.nan           # Veri yok → NaN (LightGBM handle eder)
                features[f'ic_cat_{cat}_abs'] = np.nan       # Veri yok → NaN

        return features

    # =========================================================================
    # 2. MARKET CONTEXT FEATURES (~7 feature)
    # =========================================================================

    def _build_market_features(self, analysis: Any) -> Dict[str, float]:
        """
        Piyasa bağlamı feature'ları.
        
        Gemini'nin "market regime" ve "bağlamsal değerlendirme" 
        yeteneğinin sayısal karşılığı. Volatilite, hacim, momentum
        gibi piyasa koşullarını kodlar.
        
        Parameters:
        ----------
        analysis : CoinAnalysisResult
            IC analiz sonucu (price, change_24h, volume_24h, vs. içerir)
            
        Returns:
        -------
        Dict[str, float]
            Market context feature'ları
        """
        features = {}

        # 24h fiyat değişimi (%)
        # Momentum göstergesi: büyük pozitif → rally, büyük negatif → sell-off
        features['mkt_change_24h'] = float(                  # 24 saatlik fiyat değişimi (%)
            getattr(analysis, 'change_24h', 0.0) or 0.0
        )

        # 24h mutlak fiyat değişimi
        # Yön fark etmeksizin hareket büyüklüğü (volatilite proxy'si)
        features['mkt_abs_change_24h'] = float(              # |24h değişim| — hareket büyüklüğü
            abs(getattr(analysis, 'change_24h', 0.0) or 0.0)
        )

        # 24h hacim (log-transformed)
        # Log dönüşümü: hacim çok sağa çarpık (BTC milyarlar, small coin milyonlar)
        # Log alınca dağılım normalleşir → LightGBM split'leri daha anlamlı olur
        volume_24h = getattr(analysis, 'volume_24h', 0) or 0
        features['mkt_volume_24h_log'] = float(              # log10(24h USDT hacim + 1)
            np.log10(volume_24h + 1)                         # +1: log(0) undefined koruması
            if volume_24h > 0 else 0.0
        )

        # Market rejimi (one-hot encoding)
        # GÜVENLİK AĞI: 'unknown' gelirse 'ranging' yapıyoruz
        regime = getattr(analysis, 'market_regime', 'ranging')
        if not regime or regime == 'unknown':
            regime = 'ranging'

        features['mkt_regime_trending'] = (                  # 1.0 = trending piyasa (güçlü trend var)
            1.0 if regime == 'trending' else 0.0
        )
        features['mkt_regime_ranging'] = (                   # 1.0 = yatay piyasa (trend yok)
            1.0 if regime == 'ranging' else 0.0
        )
        features['mkt_regime_volatile'] = (                  # 1.0 = aşırı volatil piyasa
            1.0 if regime == 'volatile' else 0.0
        )

        return features

    # =========================================================================
    # 3. CROSS-TIMEFRAME FEATURES (~6 feature)
    # =========================================================================

    def _build_cross_tf_features(
        self,
        analysis: Any,
        all_tf_analyses: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Farklı timeframe'ler arası tutarlılık feature'ları.
        
        Motivasyon: Tek bir TF'de güçlü sinyal olabilir ama diğer TF'ler
        zıt yönde gösterebilir. Cross-TF uyumu yüksek = sinyal güvenilir.
        
        Gemini bunu prompt'ta "timeframe sıralaması" olarak görüyordu.
        Burada sayısal metrikler olarak kodlanıyor.
        
        Parameters:
        ----------
        analysis : CoinAnalysisResult
            tf_rankings: [{'tf': '1h', 'score': 75, 'direction': 'SHORT'}, ...]
            
        all_tf_analyses : List[Dict], optional
            _analyze_coin'den gelen ham TF analizleri (daha fazla detay)
            
        Returns:
        -------
        Dict[str, float]
            Cross-timeframe feature'ları
        """
        features = {}

        # TF rankings'ten feature çıkar
        tf_rankings = getattr(analysis, 'tf_rankings', [])

        if tf_rankings and len(tf_rankings) > 0:
            # TF composite skorları
            scores = [r.get('score', 0) for r in tf_rankings]   # Her TF'nin skoru

            # En iyi TF skoru (pipeline'ın seçtiği timeframe)
            features['ctf_best_score'] = float(max(scores))     # En iyi TF'nin IC×100 skoru (tipik 5-20, yüzde DEĞİL)

            # TF skorları ortalaması (genel sinyal kalitesi)
            features['ctf_avg_score'] = float(np.mean(scores))  # Ortalama TF IC×100 skoru

            # TF skorları standart sapması (TF'ler arası tutarlılık)
            # Düşük std = tüm TF'ler benzer sinyal → güvenilir
            # Yüksek std = TF'ler çelişiyor → riskli
            features['ctf_score_std'] = float(                  # TF IC×100 skorları std (düşük=tutarlı)
                np.std(scores) if len(scores) > 1 else 0.0
            )

            # En iyi vs en kötü TF skoru farkı (spread)
            # Yüksek spread = TF'ler çok farklı → sinyal gürültülü olabilir
            features['ctf_score_spread'] = float(               # Max - Min TF skoru
                max(scores) - min(scores) if len(scores) > 1 else 0.0
            )

            # TF yön uyumu: Kaç TF aynı yönde sinyal veriyor?
            # Oran olarak: 1.0 = tam uyum, 0.5 = yarısı zıt yönde
            main_direction = getattr(analysis, 'ic_direction', 'NEUTRAL')
            if main_direction != 'NEUTRAL':
                agreeing = sum(                                 # Aynı yöndeki TF sayısı
                    1 for r in tf_rankings
                    if r.get('direction', '') == main_direction
                )
                features['ctf_direction_agreement'] = float(    # Yön uyumu oranı (0-1)
                    agreeing / len(tf_rankings)
                )
            else:
                features['ctf_direction_agreement'] = 0.0       # NEUTRAL → uyum yok

            # Analiz edilen TF sayısı
            # Az TF = az veri → düşük güvenilirlik
            features['ctf_n_timeframes'] = float(               # Analiz edilen timeframe sayısı
                len(tf_rankings)
            )

        else:
            # TF verisi yok → NaN (LightGBM handle eder)
            features['ctf_best_score'] = np.nan
            features['ctf_avg_score'] = np.nan
            features['ctf_score_std'] = np.nan
            features['ctf_score_spread'] = np.nan
            features['ctf_direction_agreement'] = np.nan
            features['ctf_n_timeframes'] = 0.0

        return features

    # =========================================================================
    # 4. PRICE ACTION FEATURES (~10 feature)
    # =========================================================================

    def _build_price_features(
        self,
        ohlcv_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        OHLCV verisinden price action feature'ları çıkar.
        
        Bu feature'lar Gemini'nin "piyasa dinamik" değerlendirmesinin
        karşılığı. Son N bar'ın momentum, volatilite ve trend istatistikleri.
        
        İstatistiksel Dikkat:
        - Tüm hesaplamalar sadece t ve öncesi veriyi kullanır (look-ahead bias yok)
        - Rolling window istatistikleri: ilk N-1 bar NaN olur (normal)
        
        Parameters:
        ----------
        ohlcv_df : pd.DataFrame, optional
            OHLCV DataFrame (columns: open, high, low, close, volume)
            İndikatörler de hesaplanmış olabilir (calculator çıktısı)
            None → tüm feature'lar NaN
            
        Returns:
        -------
        Dict[str, float]
            Price action feature'ları
        """
        features = {}

        if ohlcv_df is None or len(ohlcv_df) < 20:
            # OHLCV verisi yok veya çok kısa → tüm feature'lar NaN
            for key in [
                'px_momentum_5', 'px_momentum_10', 'px_momentum_20',
                'px_volatility_10', 'px_volatility_20',
                'px_trend_strength', 'px_mean_reversion',
                'px_body_ratio', 'px_upper_wick', 'px_lower_wick',
            ]:
                features[key] = np.nan
            return features

        close = ohlcv_df['close']              # Kapanış fiyatları serisi

        # ── Momentum Feature'ları ──
        # Son N bar'daki yüzde değişim
        # Kısa vadeli momentum: fiyat hangi yönde hareket ediyor?
        features['px_momentum_5'] = float(                   # Son 5 bar getiri (%)
            (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if len(close) > 6 else np.nan
        )
        features['px_momentum_10'] = float(                  # Son 10 bar getiri (%)
            (close.iloc[-1] / close.iloc[-11] - 1) * 100
            if len(close) > 11 else np.nan
        )
        features['px_momentum_20'] = float(                  # Son 20 bar getiri (%)
            (close.iloc[-1] / close.iloc[-21] - 1) * 100
            if len(close) > 21 else np.nan
        )

        # ── Volatilite Feature'ları ──
        # Log return'lerin standart sapması (annualize edilmemiş)
        # Yüksek volatilite → daha geniş SL gerekir, sinyal gürültülü olabilir
        log_returns = np.log(close / close.shift(1)).dropna()  # Log getiriler

        features['px_volatility_10'] = float(                # Son 10 bar volatilite (log return std)
            log_returns.iloc[-10:].std()
            if len(log_returns) >= 10 else np.nan
        )
        features['px_volatility_20'] = float(                # Son 20 bar volatilite
            log_returns.iloc[-20:].std()
            if len(log_returns) >= 20 else np.nan
        )

        # ── Trend Strength ──
        # Son 20 bar'ın lineer regresyon R² değeri
        # R² → 1: güçlü trend, R² → 0: yatay/düzensiz
        if len(close) >= 20:
            y = close.iloc[-20:].values        # Son 20 bar kapanış
            x = np.arange(20)                  # 0, 1, 2, ..., 19
            # np.polyfit: 1. derece polinom (lineer regresyon)
            coeffs = np.polyfit(x, y, 1)       # [slope, intercept]
            y_pred = np.polyval(coeffs, x)     # Tahmin değerleri
            ss_res = np.sum((y - y_pred) ** 2) # Residual sum of squares
            ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0  # R² hesabı
            features['px_trend_strength'] = float(r_squared) # Trend gücü (0-1, R²)
        else:
            features['px_trend_strength'] = np.nan

        # ── Mean Reversion ──
        # Fiyatın 20-bar SMA'dan sapması (z-score)
        # Yüksek pozitif → overbought, yüksek negatif → oversold
        if len(close) >= 20:
            sma_20 = close.iloc[-20:].mean()   # 20-bar basit hareketli ortalama
            std_20 = close.iloc[-20:].std()    # 20-bar standart sapma
            features['px_mean_reversion'] = float(           # SMA z-score
                (close.iloc[-1] - sma_20) / std_20           # (fiyat - SMA) / std
                if std_20 > 0 else 0.0
            )
        else:
            features['px_mean_reversion'] = np.nan

        # ── Candlestick Özellikleri (Son Bar) ──
        # Mum gövdesi ve fitil oranları → piyasa katılımcı davranışı
        if all(col in ohlcv_df.columns for col in ['open', 'high', 'low', 'close']):
            last = ohlcv_df.iloc[-1]           # Son mum
            high_low = last['high'] - last['low']  # Toplam mum uzunluğu

            if high_low > 0:                   # Sıfıra bölme koruması
                body = abs(last['close'] - last['open'])     # Gövde uzunluğu
                # Body ratio: gövde / toplam uzunluk
                # Yüksek → güçlü yönlü hareket (marubozu)
                # Düşük → kararsızlık (doji)
                features['px_body_ratio'] = float(           # Gövde oranı (0-1)
                    body / high_low
                )

                # Upper wick: yüksek fiyat tepkisi → satış baskısı göstergesi
                upper_wick = last['high'] - max(last['open'], last['close'])
                features['px_upper_wick'] = float(           # Üst fitil oranı (0-1)
                    upper_wick / high_low
                )

                # Lower wick: düşük fiyat tepkisi → alış baskısı göstergesi
                lower_wick = min(last['open'], last['close']) - last['low']
                features['px_lower_wick'] = float(           # Alt fitil oranı (0-1)
                    lower_wick / high_low
                )
            else:
                # Fiyat değişmemiş (çok nadir ama mümkün)
                features['px_body_ratio'] = 0.0
                features['px_upper_wick'] = 0.0
                features['px_lower_wick'] = 0.0
        else:
            features['px_body_ratio'] = np.nan
            features['px_upper_wick'] = np.nan
            features['px_lower_wick'] = np.nan

        return features

    # =========================================================================
    # 5. RISK METRICS FEATURES (~5 feature)
    # =========================================================================

    def _build_risk_features(self, analysis: Any) -> Dict[str, float]:
        """
        Risk hesaplama feature'ları.
        
        ATR, SL/TP mesafeleri ve Risk/Reward oranı.
        Gemini bunu "risk hesaplamaları" bölümünde prompt olarak alıyordu.
        
        Parameters:
        ----------
        analysis : CoinAnalysisResult
            Risk metrikleri (atr, atr_pct, sl_price, tp_price, risk_reward, leverage)
            
        Returns:
        -------
        Dict[str, float]
            Risk feature'ları
        """
        features = {}

        # ATR yüzde (fiyata göre normalize edilmiş volatilite ölçüsü)
        # Yüksek ATR% → geniş SL gerekir → daha düşük kaldıraç
        features['risk_atr_pct'] = float(                    # ATR / Fiyat × 100
            getattr(analysis, 'atr_pct', 0.0) or 0.0
        )

        # Risk/Reward oranı
        # > 1.5 → ödül riski aşıyor (iyi), < 1.0 → risk ödülü aşıyor (kötü)
        features['risk_rr_ratio'] = float(                   # TP mesafesi / SL mesafesi
            getattr(analysis, 'risk_reward', 0.0) or 0.0
        )

        # SL mesafesi yüzde (fiyattan SL'ye olan uzaklık)
        price = getattr(analysis, 'price', 0)
        sl_price = getattr(analysis, 'sl_price', 0)
        if price and price > 0 and sl_price and sl_price > 0:
            features['risk_sl_distance_pct'] = float(        # |fiyat - SL| / fiyat × 100
                abs(price - sl_price) / price * 100
            )
        else:
            features['risk_sl_distance_pct'] = np.nan

        # Kaldıraç seviyesi
        # Yüksek kaldıraç → küçük hareketle büyük kayıp riski
        features['risk_leverage'] = float(                   # Kullanılan kaldıraç (2x-20x)
            getattr(analysis, 'leverage', 0) or 0
        )

        # Pozisyon büyüklüğü (normalize)
        # Log-transform: büyük pozisyon farkları var (0.001 BTC vs 1000 DOGE)
        pos_size = getattr(analysis, 'position_size', 0) or 0
        features['risk_position_size_log'] = float(          # log10(pozisyon + 1)
            np.log10(pos_size + 1) if pos_size > 0 else 0.0
        )

        return features

    # =========================================================================
    # 6. TEMPORAL FEATURES (~5 feature)
    # =========================================================================

    def _build_temporal_features(self, bar_dt=None) -> Dict[str, float]:
        """
        Zaman bazlı feature'lar.
        
        Kripto piyasaları 7/24 açık ama belirli saatlerde hacim/volatilite
        farklı (Asya session, US session, vs.). Bu pattern'leri yakalamak için
        cyclical encoding kullanılır.
        
        Cyclical Encoding Neden?
        - Saat 23 ve saat 0 aslında yakın ama numerik olarak uzak (23 vs 0)
        - sin/cos dönüşümü ile: 23:00 ve 01:00 yakın değerler alır
        - LightGBM böylece "gece yarısı civarı" pattern'ini öğrenebilir

        Parameters:
        ----------
        bar_dt : datetime, optional
            [EĞİTİM FIX] Eğitim sırasında bar'ın gerçek timestamp'i.
            None ise canlı trading zamanı (datetime.now) kullanılır.
            Neden gerekli: initial_train'de tüm barlar aynı anda üretilir,
            datetime.now() her bar için aynı değeri döndürür — tüm eğitim
            verisi tek bir saat/gün değeri alır, model temporal pattern öğrenemez.

        Returns:
        -------
        Dict[str, float]
            Temporal feature'lar
        """
        # [EĞİTİM FIX] bar_dt verilmişse onu kullan, yoksa anlık zamanı al
        now = bar_dt if bar_dt is not None else datetime.now(timezone.utc)
        features = {}

        # Saat — sin/cos cyclical encoding
        # 24 saatlik periyot: sin(2π × hour/24), cos(2π × hour/24)
        hour = now.hour + now.minute / 60.0    # Ondalıklı saat (örn: 14.5 = 14:30)
        features['tmp_hour_sin'] = float(                    # Saat sinüs bileşeni
            np.sin(2 * np.pi * hour / 24.0)
        )
        features['tmp_hour_cos'] = float(                    # Saat kosinüs bileşeni
            np.cos(2 * np.pi * hour / 24.0)
        )

        # Haftanın günü — sin/cos cyclical encoding
        # 7 günlük periyot: Pazartesi=0, Pazar=6
        dow = now.weekday()                    # 0=Pazartesi, 6=Pazar
        features['tmp_dow_sin'] = float(                     # Gün sinüs bileşeni
            np.sin(2 * np.pi * dow / 7.0)
        )
        features['tmp_dow_cos'] = float(                     # Gün kosinüs bileşeni
            np.cos(2 * np.pi * dow / 7.0)
        )

        # Hafta sonu flag
        # Kripto hafta sonu da açık ama hacim genelde düşük
        features['tmp_is_weekend'] = float(                  # 1.0 = Cumartesi veya Pazar
            1.0 if dow >= 5 else 0.0
        )

        return features

    # =========================================================================
    # BATCH FEATURE OLUŞTURMA (BİRDEN FAZLA COİN)
    # =========================================================================

    def build_batch_features(
        self,
        analyses: List[Any],
        ohlcv_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        all_tf_data: Optional[Dict[str, List[Dict]]] = None,
    ) -> pd.DataFrame:
        """
        Birden fazla coin için feature matrix oluştur.
        
        LightGBM eğitimi için: geçmiş trade'lerin feature'larını
        DataFrame olarak döndürür.
        
        Parameters:
        ----------
        analyses : List[CoinAnalysisResult]
            Analiz sonuçları listesi
            
        ohlcv_dfs : Dict[str, pd.DataFrame], optional
            Coin sembolü → OHLCV DataFrame mapping
            
        all_tf_data : Dict[str, List[Dict]], optional
            Coin sembolü → TF analizleri mapping
            
        Returns:
        -------
        pd.DataFrame
            Satır = coin, Kolon = feature
            LightGBM eğitim input'u için hazır
        """
        rows = []                              # Feature vektörleri listesi

        for analysis in analyses:
            symbol = getattr(analysis, 'symbol', '')

            # Bu coin'in OHLCV verisi var mı?
            ohlcv = None
            if ohlcv_dfs and symbol in ohlcv_dfs:
                ohlcv = ohlcv_dfs[symbol]

            # Bu coin'in TF analizleri var mı?
            tf_data = None
            if all_tf_data and symbol in all_tf_data:
                tf_data = all_tf_data[symbol]

            # Feature vektörü oluştur
            vec = self.build_features(analysis, ohlcv, tf_data)
            row = vec.to_dict()                # Dict'e çevir
            row['_symbol'] = symbol            # Meta bilgi (feature DEĞİL, takip için)
            row['_coin'] = vec.coin
            row['_timestamp'] = vec.timestamp
            rows.append(row)

        if not rows:
            return pd.DataFrame()              # Boş DataFrame

        df = pd.DataFrame(rows)                # Feature matrix oluştur

        if self.verbose:
            n_features = len([c for c in df.columns if not c.startswith('_')])
            logger.debug(
                f"  📊 Batch Features: {len(df)} coin × {n_features} feature"
            )

        return df

    # =========================================================================
    # FEATURE İSİMLERİ VE META BİLGİ
    # =========================================================================

    def get_feature_names(self) -> List[str]:
        """
        Tüm feature isimlerini sıralı olarak döndür.
        
        LightGBM eğitimi sırasında feature_name parametresi
        ve feature importance analizi için kullanılır.
        
        Boş bir MLFeatureVector oluşturup isimlerini çeker.
        """
        # Dummy analiz objesi ile boş feature vektörü oluştur
        dummy = type('DummyAnalysis', (), {     # Anonim dummy sınıf
            'symbol': '', 'coin': '', 'price': 0,
            'change_24h': 0, 'volume_24h': 0,
            'ic_confidence': 0, 'top_ic': 0,
            'significant_count': 0, 'ic_direction': 'NEUTRAL',
            'category_tops': {}, 'tf_rankings': [],
            'market_regime': 'ranging',
            'atr': 0, 'atr_pct': 0,
            'sl_price': 0, 'tp_price': 0,
            'risk_reward': 0, 'leverage': 0,
            'position_size': 0,
        })()

        vec = self.build_features(dummy, None, None)
        return vec.feature_names()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Her feature'ın kısa açıklamasını döndür.
        Model interpretability ve dokümantasyon için.
        """
        return {
            # IC Features
            'ic_confidence':          'IC composite skoru (0-100)',
            'ic_top_abs':             'En güçlü |IC| değeri',
            'ic_sig_count':           'FDR-significant indikatör sayısı',
            'ic_direction_code':      'IC yön kodu (+1=LONG, -1=SHORT, 0=NEUTRAL)',
            'ic_cat_trend':           'Trend kategorisi IC değeri (işaretli)',
            'ic_cat_trend_abs':       'Trend kategorisi |IC|',
            'ic_cat_momentum':        'Momentum kategorisi IC değeri',
            'ic_cat_momentum_abs':    'Momentum kategorisi |IC|',
            'ic_cat_volatility':      'Volatilite kategorisi IC değeri',
            'ic_cat_volatility_abs':  'Volatilite kategorisi |IC|',
            'ic_cat_volume':          'Hacim kategorisi IC değeri',
            'ic_cat_volume_abs':      'Hacim kategorisi |IC|',
            # Market Features
            'mkt_change_24h':         '24h fiyat değişimi (%)',
            'mkt_abs_change_24h':     '24h mutlak değişim (%)',
            'mkt_volume_24h_log':     'log10(24h USDT hacim)',
            'mkt_regime_trending':    '1=trending piyasa',
            'mkt_regime_ranging':     '1=yatay piyasa',
            'mkt_regime_volatile':    '1=volatil piyasa',
            # Cross-TF Features
            'ctf_best_score':         'En iyi TF IC×100 skoru (0-40 tipik, yüzde değil)',
            'ctf_avg_score':          'Ortalama TF IC×100 skoru (0-40 tipik)',
            'ctf_score_std':          'TF IC×100 skorları std (düşük=tutarlı TF)  ',
            'ctf_score_spread':       'En iyi vs en kötü TF farkı',
            'ctf_direction_agreement':'TF yön uyumu oranı (0-1)',
            'ctf_n_timeframes':       'Analiz edilen TF sayısı',
            # Price Action Features
            'px_momentum_5':          'Son 5 bar getiri (%)',
            'px_momentum_10':         'Son 10 bar getiri (%)',
            'px_momentum_20':         'Son 20 bar getiri (%)',
            'px_volatility_10':       'Son 10 bar volatilite (log ret std)',
            'px_volatility_20':       'Son 20 bar volatilite',
            'px_trend_strength':      'Lineer regresyon R² (0-1)',
            'px_mean_reversion':      'SMA-20 z-score',
            'px_body_ratio':          'Son mum gövde oranı (0-1)',
            'px_upper_wick':          'Son mum üst fitil oranı',
            'px_lower_wick':          'Son mum alt fitil oranı',
            # Risk Features
            'risk_atr_pct':           'ATR/Fiyat (%)',
            'risk_rr_ratio':          'Risk/Reward oranı',
            'risk_sl_distance_pct':   'SL mesafesi (%)',
            'risk_leverage':          'Kaldıraç seviyesi',
            'risk_position_size_log': 'log10(pozisyon büyüklüğü)',
            # Temporal Features
            'tmp_hour_sin':           'Saat sinüs (cyclical)',
            'tmp_hour_cos':           'Saat kosinüs (cyclical)',
            'tmp_dow_sin':            'Gün sinüs (cyclical)',
            'tmp_dow_cos':            'Gün kosinüs (cyclical)',
            'tmp_is_weekend':         '1=hafta sonu',
        }


# =============================================================================
# BAĞIMSIZ ÇALIŞTIRMA TESTİ
# =============================================================================

if __name__ == "__main__":
    """
    Modülü tek başına test et:
      cd src && python -m ml.feature_engineer
    
    Sentetik CoinAnalysisResult ile feature vektörü oluşturur
    ve feature isimlerini/değerlerini yazdırır.
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 60)
    print("  🧬 FEATURE ENGINEER — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Dummy CoinAnalysisResult oluştur ──
    # (main.py'den gelen gerçek objeymiş gibi simüle et)
    class DummyAnalysis:
        """Test için sahte CoinAnalysisResult."""
        symbol = 'SOL/USDT:USDT'
        coin = 'SOL'
        price = 185.0
        change_24h = -2.3
        volume_24h = 500_000_000
        ic_confidence = 75.0
        ic_direction = 'SHORT'
        top_ic = -0.15
        top_indicator = 'SUPERTREND'
        significant_count = 12
        market_regime = 'trending'
        category_tops = {
            'trend':      {'name': 'SUPERTREND', 'ic': -0.12},
            'momentum':   {'name': 'RSI_14',     'ic': -0.08},
            'volatility': {'name': 'ATR_14',     'ic': 0.05},
            'volume':     {'name': 'CMF_20',     'ic': -0.10},
        }
        tf_rankings = [
            {'tf': '1h',  'score': 75, 'direction': 'SHORT', 'regime': 'trending'},
            {'tf': '30m', 'score': 68, 'direction': 'SHORT', 'regime': 'trending'},
            {'tf': '15m', 'score': 55, 'direction': 'NEUTRAL', 'regime': 'ranging'},
            {'tf': '2h',  'score': 50, 'direction': 'SHORT', 'regime': 'trending'},
        ]
        atr = 3.70
        atr_pct = 2.0
        sl_price = 188.70
        tp_price = 179.45
        risk_reward = 1.5
        position_size = 0.405
        leverage = 4

    # ── 2. Sentetik OHLCV verisi oluştur ──
    np.random.seed(42)
    n_bars = 200
    prices = 185.0 + np.cumsum(np.random.randn(n_bars) * 2)  # Random walk
    ohlcv = pd.DataFrame({
        'open':   prices + np.random.rand(n_bars) * 0.5,
        'high':   prices + np.abs(np.random.randn(n_bars)) * 2,
        'low':    prices - np.abs(np.random.randn(n_bars)) * 2,
        'close':  prices,
        'volume': np.random.randint(1000, 100000, n_bars).astype(float),
    })

    # ── 3. Feature Engineer çalıştır ──
    engineer = FeatureEngineer(verbose=True)
    vec = engineer.build_features(DummyAnalysis(), ohlcv)

    # ── 4. Sonuçları yazdır ──
    print(f"\n{'─' * 60}")
    print(f"📊 Feature Vektörü ({vec.feature_count()} feature)")
    print(f"{'─' * 60}")

    all_features = vec.to_dict()
    descriptions = engineer.get_feature_descriptions()

    for name, value in all_features.items():
        desc = descriptions.get(name, '')
        if isinstance(value, float) and not np.isnan(value):
            print(f"  {name:<30} = {value:>10.4f}  │ {desc}")
        else:
            print(f"  {name:<30} = {'NaN':>10}  │ {desc}")

    # ── 5. Numpy array çıktısı ──
    arr = vec.to_array()
    print(f"\n  numpy shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  NaN count: {np.isnan(arr).sum()}")

    # ── 6. Feature isimleri ──
    names = engineer.get_feature_names()
    print(f"\n  Feature isimleri ({len(names)}): {names[:5]}...")

    # ── 7. MLDecisionResult test ──
    result = MLDecisionResult(
        decision=MLDecision.SHORT,
        confidence=78.5,
        reasoning="IC=75, 3/4 TF SHORT, trending rejim, momentum negatif",
        gate_action="FULL_TRADE",
        ic_score=75.0,
        entry_price=185.0,
        sl_price=188.70,
        tp_price=179.45,
        risk_reward=1.5,
        feature_importance_top3=['ic_confidence', 'px_momentum_5', 'ctf_direction_agreement'],
    )

    print(f"\n{'─' * 60}")
    print(f"📋 MLDecisionResult Test")
    print(f"{'─' * 60}")
    print(f"  should_execute: {result.should_execute()}")
    print(f"\n{result.summary()}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ FEATURE ENGINEER TESTİ TAMAMLANDI")
    print(f"{'=' * 60}")