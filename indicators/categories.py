# =============================================================================
# İNDİKATÖR KATEGORİLERİ VE TANIMLARI
# =============================================================================
# Amaç: 64+ teknik indikatörü kategorize etmek ve parametrelerini tanımlamak
#
# Eski projeden taşındı, değişiklikler:
# - Aynı yapı korundu (IndicatorConfig dataclass)
# - Composite kategorisinden kararsız indikatörler çıkarıldı
# - Futures'a özel not eklendi
#
# Kategoriler:
# 1. TREND       - Fiyat yönü ve trend gücü (MA'lar, ADX, Supertrend)
# 2. MOMENTUM    - Aşırı alım/satım ve momentum (RSI, Stochastic, MACD)
# 3. VOLATILITY  - Piyasa volatilitesi (ATR, Bollinger, Keltner)
# 4. VOLUME      - Hacim analizi (OBV, MFI, CMF)
#
# İstatistiksel Not:
# - Aynı kategorideki indikatörler yüksek korelasyon gösterir (multicollinearity)
# - IC selector kategori başına 1-2 indikatör seçecek → multicollinearity azalır
# - 64 indikatör × 6 TF = 384 IC testi → FDR correction ZORUNLU
#
# Futures Notu:
# - Spot ile aynı OHLCV yapısı (pandas-ta fark görmez)
# - Volume = kontrat adedi (USD değil), ama göreli analiz için sorun yok
# - Funding rate ayrı bir feature olarak eklenebilir (ileride)
# =============================================================================

from typing import Dict, List, Any          # Tip belirteçleri
from dataclasses import dataclass           # Yapılandırılmış veri sınıfı


@dataclass
class IndicatorConfig:
    """
    Tek bir indikatörün yapılandırması.
    
    Attributes:
    ----------
    name : str
        pandas-ta fonksiyon adı (örn: 'rsi', 'macd', 'bbands')
        df.ta.rsi(length=14) şeklinde çağrılır
        
    display_name : str
        Gösterim adı (Telegram mesajı ve loglarda kullanılır)
        
    category : str
        Kategori: 'trend', 'momentum', 'volatility', 'volume'
        IC selector bu kategorilere göre gruplar
        
    params : Dict[str, Any]
        pandas-ta fonksiyonuna geçirilen parametreler
        Örn: {"length": 14, "scalar": 2.0}
        
    output_columns : List[str]
        Fonksiyonun ürettiği DataFrame kolon adları
        IC hesaplamasında bu kolonlar tek tek değerlendirilir
        
    description : str
        İndikatörün kısa açıklaması (Türkçe)
        
    signal_type : str
        Sinyal türü:
        - "level": Sabit eşik değerleri (RSI 30-70 gibi)
        - "crossover": İki çizginin kesişimi (MACD-Signal gibi)
        - "band": Bant kırılımı (Bollinger gibi)
    """
    name: str
    display_name: str
    category: str
    params: Dict[str, Any]
    output_columns: List[str]
    description: str
    signal_type: str = "level"


# =============================================================================
# TREND İNDİKATÖRLERİ (17 adet)
# =============================================================================
# Fiyat yönünü ve trend gücünü ölçer.
# MA'lar: Fiyatın üstü = bullish, altı = bearish
# ADX: Trend gücü (>25 güçlü), yön göstermez
# Supertrend: ATR bazlı trailing stop + yön
# =============================================================================

TREND_INDICATORS: List[IndicatorConfig] = [
    # --- Hareketli Ortalamalar ---
    IndicatorConfig(
        "sma", "SMA_20", "trend", {"length": 20}, ["SMA_20"],
        "Simple MA 20 - Kısa vadeli trend filtresi", "crossover"
    ),
    IndicatorConfig(
        "sma", "SMA_50", "trend", {"length": 50}, ["SMA_50"],
        "Simple MA 50 - Orta vadeli trend filtresi", "crossover"
    ),
    IndicatorConfig(
        "sma", "SMA_200", "trend", {"length": 200}, ["SMA_200"],
        "Simple MA 200 - Uzun vadeli trend (bull/bear market)", "crossover"
    ),
    IndicatorConfig(
        "ema", "EMA_12", "trend", {"length": 12}, ["EMA_12"],
        "EMA 12 - MACD fast bileşeni, hızlı tepki", "crossover"
    ),
    IndicatorConfig(
        "ema", "EMA_20", "trend", {"length": 20}, ["EMA_20"],
        "EMA 20 - Kısa vadeli dinamik destek/direnç", "crossover"
    ),
    IndicatorConfig(
        "ema", "EMA_26", "trend", {"length": 26}, ["EMA_26"],
        "EMA 26 - MACD slow bileşeni", "crossover"
    ),
    IndicatorConfig(
        "ema", "EMA_50", "trend", {"length": 50}, ["EMA_50"],
        "EMA 50 - Orta vadeli dinamik destek/direnç", "crossover"
    ),
    IndicatorConfig(
        "wma", "WMA_20", "trend", {"length": 20}, ["WMA_20"],
        "Weighted MA - Son fiyatlara daha çok ağırlık", "crossover"
    ),
    IndicatorConfig(
        "dema", "DEMA_20", "trend", {"length": 20}, ["DEMA_20"],
        "Double EMA - Düşük gecikme (lag)", "crossover"
    ),
    IndicatorConfig(
        "tema", "TEMA_20", "trend", {"length": 20}, ["TEMA_20"],
        "Triple EMA - Minimum gecikme", "crossover"
    ),
    IndicatorConfig(
        "hma", "HMA_20", "trend", {"length": 20}, ["HMA_20"],
        "Hull MA - Çok düşük lag, overshooting riski var", "crossover"
    ),
    IndicatorConfig(
        "kama", "KAMA", "trend", {"length": 10}, ["KAMA_10_2_30"],
        "Kaufman Adaptive MA - Volatiliteye göre hız ayarlar", "crossover"
    ),
    # --- Trend Gücü Ölçenleri ---
    IndicatorConfig(
        "adx", "ADX", "trend", {"length": 14},
        ["ADX_14", "DMP_14", "DMN_14"],
        "ADX - Trend gücü (>25 güçlü). DMP/DMN yön gösterir", "level"
    ),
    IndicatorConfig(
        "aroon", "AROON", "trend", {"length": 25},
        ["AROONU_25", "AROOND_25", "AROONOSC_25"],
        "Aroon - Trend başlangıcı ve gücü tespiti", "crossover"
    ),
    # --- Trend Takip Sistemleri ---
    IndicatorConfig(
        "psar", "PSAR", "trend",
        {"af0": 0.02, "af": 0.02, "max_af": 0.2},
        ["PSARl_0.02_0.2", "PSARs_0.02_0.2"],
        "Parabolic SAR - Trailing stop, trend dönüşü tespiti", "level"
    ),
    IndicatorConfig(
        "supertrend", "SUPERTREND", "trend",
        {"length": 10, "multiplier": 3.0},
        ["SUPERT_10_3.0", "SUPERTd_10_3.0"],
        "Supertrend - ATR bazlı trend yönü ve seviyesi", "level"
    ),
    IndicatorConfig(
        "vortex", "VORTEX", "trend", {"length": 14},
        ["VTXP_14", "VTXN_14"],
        "Vortex - Trend yönü crossover sistemi", "crossover"
    ),
]


# =============================================================================
# MOMENTUM İNDİKATÖRLERİ (18 adet)
# =============================================================================
# Fiyatın hızını ve aşırı alım/satım durumunu ölçer.
# RSI/Stochastic: 0-100 arası, uçlarda reversal sinyali
# MACD/PPO: Momentum değişim hızı, crossover sinyali
# =============================================================================

MOMENTUM_INDICATORS: List[IndicatorConfig] = [
    # --- RSI Varyasyonları ---
    IndicatorConfig(
        "rsi", "RSI_14", "momentum", {"length": 14}, ["RSI_14"],
        "RSI 14 - Klasik momentum. <30 aşırı satım, >70 aşırı alım", "level"
    ),
    IndicatorConfig(
        "rsi", "RSI_7", "momentum", {"length": 7}, ["RSI_7"],
        "RSI 7 - Kısa periyot, daha hassas (daha çok sinyal)", "level"
    ),
    IndicatorConfig(
        "rsi", "RSI_21", "momentum", {"length": 21}, ["RSI_21"],
        "RSI 21 - Uzun periyot, daha az sinyal (daha güvenilir)", "level"
    ),
    # --- Stochastic ---
    IndicatorConfig(
        "stoch", "STOCH", "momentum",
        {"k": 14, "d": 3, "smooth_k": 3},
        ["STOCHk_14_3_3", "STOCHd_14_3_3"],
        "Stochastic - Range-bound momentum, K-D crossover", "crossover"
    ),
    IndicatorConfig(
        "stochrsi", "STOCHRSI", "momentum",
        {"length": 14, "rsi_length": 14, "k": 3, "d": 3},
        ["STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3"],
        "Stochastic RSI - Çok hassas, momentum uçlarını yakalar", "crossover"
    ),
    # --- Diğer Oscillator'lar ---
    IndicatorConfig(
        "willr", "WILLR", "momentum", {"length": 14}, ["WILLR_14"],
        "Williams %R - Stochastic benzeri, -20/-80 ekstrem bölgeler", "level"
    ),
    IndicatorConfig(
        "cci", "CCI", "momentum", {"length": 20}, ["CCI_20_0.015"],
        "CCI - Mean deviation bazlı. +100/-100 üstü ekstrem", "level"
    ),
    IndicatorConfig(
        "mom", "MOM", "momentum", {"length": 10}, ["MOM_10"],
        "Momentum - Basit fiyat farkı (P_t - P_{t-n})", "level"
    ),
    IndicatorConfig(
        "roc", "ROC", "momentum", {"length": 10}, ["ROC_10"],
        "Rate of Change - Yüzdesel fiyat değişimi", "level"
    ),
    IndicatorConfig(
        "roc", "ROC_20", "momentum", {"length": 20}, ["ROC_20"],
        "ROC 20 - Orta vadeli momentum hızı", "level"
    ),
    IndicatorConfig(
        "ao", "AO", "momentum", {"fast": 5, "slow": 34}, ["AO_5_34"],
        "Awesome Oscillator - Bill Williams, SMA(5)-SMA(34)", "level"
    ),
    # --- MACD Ailesi ---
    IndicatorConfig(
        "macd", "MACD", "momentum",
        {"fast": 12, "slow": 26, "signal": 9},
        ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        "MACD - Trend momentum. Histogram > 0 = bullish", "crossover"
    ),
    IndicatorConfig(
        "ppo", "PPO", "momentum",
        {"fast": 12, "slow": 26, "signal": 9},
        ["PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9"],
        "PPO - Yüzdesel MACD. Farklı fiyatlı coinleri karşılaştırır", "crossover"
    ),
    IndicatorConfig(
        "tsi", "TSI", "momentum",
        {"fast": 13, "slow": 25, "signal": 13},
        ["TSI_13_25_13", "TSIs_13_25_13"],
        "True Strength Index - Çift smoothing momentum", "crossover"
    ),
    IndicatorConfig(
        "uo", "UO", "momentum",
        {"fast": 7, "medium": 14, "slow": 28}, ["UO_7_14_28"],
        "Ultimate Oscillator - Multi-timeframe momentum birleşimi", "level"
    ),
    IndicatorConfig(
        "cmo", "CMO", "momentum", {"length": 14}, ["CMO_14"],
        "Chande Momentum - RSI alternatifi, simetrik ölçek", "level"
    ),
    IndicatorConfig(
        "fisher", "FISHER", "momentum", {"length": 9},
        ["FISHERT_9_1", "FISHERTs_9_1"],
        "Fisher Transform - Gaussian dönüşüm, net dönüş sinyalleri", "crossover"
    ),
    IndicatorConfig(
        "coppock", "COPPOCK", "momentum",
        {"length": 10, "fast": 11, "slow": 14}, ["COPC_11_14_10"],
        "Coppock Curve - Uzun vadeli momentum dip tespiti", "level"
    ),
]


# =============================================================================
# VOLATİLİTE İNDİKATÖRLERİ (12 adet)
# =============================================================================
# Piyasanın ne kadar "hareketli" olduğunu ölçer.
# ATR: Pozisyon sizing ve SL mesafesi için kritik
# Bollinger: Squeeze (sıkışma) → breakout sinyali
# =============================================================================

VOLATILITY_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig(
        "atr", "ATR", "volatility", {"length": 14}, ["ATRr_14"],
        "ATR 14 - Volatilite ölçüsü. SL/TP mesafesi + pozisyon sizing", "level"
    ),
    IndicatorConfig(
        "atr", "ATR_7", "volatility", {"length": 7}, ["ATRr_7"],
        "ATR 7 - Kısa vadeli volatilite (scalping için)", "level"
    ),
    IndicatorConfig(
        "natr", "NATR", "volatility", {"length": 14}, ["NATR_14"],
        "Normalized ATR - Yüzdesel volatilite (farklı coinleri karşılaştır)", "level"
    ),
    IndicatorConfig(
        "bbands", "BBANDS", "volatility",
        {"length": 20, "std": 2.0},
        ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"],
        "Bollinger Bands 2σ - Volatilite bandları, squeeze tespiti", "band"
    ),
    IndicatorConfig(
        "bbands", "BBANDS_1STD", "volatility",
        {"length": 20, "std": 1.0},
        ["BBL_20_1.0", "BBM_20_1.0", "BBU_20_1.0", "BBB_20_1.0", "BBP_20_1.0"],
        "Bollinger 1σ - Dar bant, trend içi geri çekilme", "band"
    ),
    IndicatorConfig(
        "kc", "KC", "volatility",
        {"length": 20, "scalar": 1.5},
        ["KCLe_20_1.5", "KCBe_20_1.5", "KCUe_20_1.5"],
        "Keltner Channel - ATR bazlı bantlar (BB ile squeeze tespiti)", "band"
    ),
    IndicatorConfig(
        "donchian", "DONCHIAN", "volatility",
        {"lower_length": 20, "upper_length": 20},
        ["DCL_20_20", "DCM_20_20", "DCU_20_20"],
        "Donchian Channel - Breakout sistemi (20-bar high/low)", "band"
    ),
    IndicatorConfig(
        "massi", "MASSI", "volatility",
        {"fast": 9, "slow": 25}, ["MASSI_9_25"],
        "Mass Index - Reversal bulge tespiti (genişleme → daralma)", "level"
    ),
    IndicatorConfig(
        "ui", "UI", "volatility", {"length": 14}, ["UI_14"],
        "Ulcer Index - Downside risk ölçüsü (drawdown bazlı)", "level"
    ),
    IndicatorConfig(
        "accbands", "ACCBANDS", "volatility", {"length": 20},
        ["ACCBL_20", "ACCBM_20", "ACCBU_20"],
        "Acceleration Bands - Momentum bazlı volatilite bantları", "band"
    ),
    IndicatorConfig(
        "rvi", "RVI", "volatility", {"length": 14},
        ["RVI_14", "RVIs_14"],
        "Relative Volatility Index - Volatilitenin yönü", "crossover"
    ),
    IndicatorConfig(
        "true_range", "TR", "volatility", {}, ["TRUERANGE_1"],
        "True Range - Tek bar volatilite (ATR'nin bileşeni)", "level"
    ),
]


# =============================================================================
# HACİM İNDİKATÖRLERİ (11 adet)
# =============================================================================
# Fiyat hareketinin arkasındaki hacim gücünü ölçer.
# OBV: Kümülatif hacim (divergence aranır)
# MFI: Volume-weighted RSI (para akışı yönü)
# CMF: Net alım/satım baskısı
#
# Futures Notu:
# - Volume = kontrat adedi (spot'taki gibi coin miktarı değil)
# - Göreli analiz (artış/azalış) hâlâ geçerli
# - Open Interest (OI) ayrı bir feature olarak eklenebilir
# =============================================================================

VOLUME_INDICATORS: List[IndicatorConfig] = [
    IndicatorConfig(
        "obv", "OBV", "volume", {}, ["OBV"],
        "On-Balance Volume - Kümülatif hacim, fiyat-hacim divergence", "level"
    ),
    IndicatorConfig(
        "ad", "AD", "volume", {}, ["AD"],
        "Accumulation/Distribution - Close pozisyonuna göre hacim ağırlığı", "level"
    ),
    IndicatorConfig(
        "adosc", "ADOSC", "volume",
        {"fast": 3, "slow": 10}, ["ADOSC_3_10"],
        "A/D Oscillator - A/D hattının MACD'si", "crossover"
    ),
    IndicatorConfig(
        "cmf", "CMF", "volume", {"length": 20}, ["CMF_20"],
        "Chaikin Money Flow - Net para akışı. >0 alım, <0 satım", "level"
    ),
    IndicatorConfig(
        "mfi", "MFI", "volume", {"length": 14}, ["MFI_14"],
        "Money Flow Index - Volume-weighted RSI. <20 aşırı satım", "level"
    ),
    IndicatorConfig(
        "efi", "EFI", "volume", {"length": 13}, ["EFI_13"],
        "Elder Force Index - Fiyat değişim × hacim", "level"
    ),
    IndicatorConfig(
        "nvi", "NVI", "volume", {"length": 1}, ["NVI_1"],
        "Negative Volume Index - Düşük hacimli günlerin etkisi", "level"
    ),
    IndicatorConfig(
        "pvi", "PVI", "volume", {"length": 1}, ["PVI_1"],
        "Positive Volume Index - Yüksek hacimli günlerin etkisi", "level"
    ),
    IndicatorConfig(
        "pvol", "PVOL", "volume", {}, ["PVOL"],
        "Price-Volume - Fiyat × Hacim çarpımı", "level"
    ),
    IndicatorConfig(
        "pvt", "PVT", "volume", {}, ["PVT"],
        "Price Volume Trend - ROC ağırlıklı kümülatif hacim", "level"
    ),
    IndicatorConfig(
        "vwma", "VWMA", "volume", {"length": 20}, ["VWMA_20"],
        "Volume Weighted MA - Hacim ağırlıklı ortalama", "crossover"
    ),
]


# =============================================================================
# TÜM KATEGORİLER (BİRLEŞİK)
# =============================================================================
# IC analizi bu 4 kategoride çalışır.
# Toplam: 17 + 18 + 12 + 11 = 58 indikatör tanımı
# Her indikatör 1-5 kolon üretir → ~100+ IC testi yapılacak
# =============================================================================

ALL_INDICATORS: Dict[str, List[IndicatorConfig]] = {
    "trend": TREND_INDICATORS,
    "momentum": MOMENTUM_INDICATORS,
    "volatility": VOLATILITY_INDICATORS,
    "volume": VOLUME_INDICATORS,
}


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def get_all_indicators() -> List[IndicatorConfig]:
    """Tüm indikatörlerin düz listesini döndürür (58 adet)."""
    result = []
    for indicators in ALL_INDICATORS.values():
        result.extend(indicators)
    return result


def get_indicators_by_category(category: str) -> List[IndicatorConfig]:
    """Belirli kategorideki indikatörleri döndürür."""
    return ALL_INDICATORS.get(category, [])


def get_category_names() -> List[str]:
    """Kategori isimlerini döndürür: ['trend', 'momentum', 'volatility', 'volume']"""
    return list(ALL_INDICATORS.keys())


def get_indicator_count() -> Dict[str, int]:
    """Her kategorideki indikatör sayısını döndürür."""
    return {cat: len(indicators) for cat, indicators in ALL_INDICATORS.items()}


def get_total_output_columns() -> int:
    """Tüm indikatörlerin toplam çıktı kolon sayısını döndürür."""
    total = 0
    for indicators in ALL_INDICATORS.values():
        for ind in indicators:
            total += len(ind.output_columns)
    return total


# =============================================================================
# TEST KODU
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  İNDİKATÖR KATEGORİLERİ")
    print("=" * 60)

    counts = get_indicator_count()
    total = sum(counts.values())
    total_cols = get_total_output_columns()

    print(f"\n  Toplam indikatör: {total}")
    print(f"  Toplam çıktı kolonu: {total_cols}")
    print(f"  (Her kolon için ayrı IC testi yapılacak)\n")

    for cat, count in counts.items():
        indicators = get_indicators_by_category(cat)
        cols = sum(len(i.output_columns) for i in indicators)
        print(f"  {cat.upper():<12}: {count:>2} indikatör → {cols:>3} kolon")

    print("=" * 60)
