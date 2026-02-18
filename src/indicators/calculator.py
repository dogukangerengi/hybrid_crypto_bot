# =============================================================================
# İNDİKATÖR HESAPLAMA MOTORU (INDICATOR CALCULATOR)
# =============================================================================
# Amaç: pandas-ta kütüphanesi ile 58+ teknik indikatör hesaplamak
#
# Eski projeden taşındı, değişiklikler:
# - categories.py'den import (relative import uyumlu)
# - Daha detaylı hata yakalama (hangi indikatör başarısız oldu)
# - NaN istatistikleri iyileştirildi
# - add_price_features ve add_rolling_stats korundu
# - add_forward_returns korundu (IC hedef değişkeni)
#
# İstatistiksel Dikkat:
# 1. Look-ahead bias: Tüm indikatörler SADECE t ve öncesi veriyi kullanır ✓
# 2. NaN handling: Rolling window başlangıcında NaN oluşur (normal)
# 3. Multicollinearity: Aynı kategorideki indikatörler yüksek korelasyonlu
#    → IC selector bu sorunu çözer (kategori başına max 2 seçer)
#
# Futures Notu:
# - OHLCV yapısı spot ile aynı → pandas-ta fark görmez
# - Volume = kontrat adedi (USD değil), göreli analiz geçerli
# =============================================================================

import pandas as pd                          # Veri yapıları
import pandas_ta as ta                       # 130+ teknik analiz indikatörü
import numpy as np                           # Sayısal hesaplamalar
from typing import Dict, List, Optional, Any # Tip belirteçleri
import warnings                              # Gereksiz uyarıları susturmak için
import logging                               # Loglama

# Aynı klasördeki categories modülünden import
from .categories import (
    ALL_INDICATORS,          # Tüm kategorilerin dictionary'si
    IndicatorConfig,         # Tek indikatör yapılandırması
    get_all_indicators,      # Tüm indikatörlerin düz listesi
    get_indicators_by_category,  # Kategori bazlı liste
    get_category_names,      # Kategori isimleri
)

# Logger
logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """
    Teknik indikatörleri hesaplayan sınıf.
    
    pandas-ta kütüphanesi üzerine wrapper. Her indikatör için:
    1. Parametre validasyonu (IndicatorConfig'den gelir)
    2. pandas-ta fonksiyonu çağrılır
    3. Sonuç DataFrame'e eklenir
    4. Hata durumunda log yazılır, diğerleri devam eder
    
    Kullanım:
    --------
    calc = IndicatorCalculator()
    
    # Tek kategori
    df_mom = calc.calculate_category(df, "momentum")
    
    # Tüm kategoriler
    df_all = calc.calculate_all(df)
    
    # Price features + rolling stats + forward returns
    df_all = calc.add_price_features(df_all)
    df_all = calc.add_rolling_stats(df_all)
    df_all = calc.add_forward_returns(df_all)
    """
    
    def __init__(self, verbose: bool = True):
        """
        IndicatorCalculator başlatır.
        
        Parameters:
        ----------
        verbose : bool
            True → hesaplama detayları loglanır
            False → sadece hatalar loglanır (production modu)
        """
        self.verbose = verbose
        
        # pandas-ta ve numpy uyarılarını sustur
        # Bu uyarılar genellikle NaN'dan kaynaklı ve beklenen davranış
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    # =========================================================================
    # TEK İNDİKATÖR HESAPLAMA
    # =========================================================================
    
    def calculate_single(
        self,
        df: pd.DataFrame,
        indicator: IndicatorConfig
    ) -> pd.DataFrame:
        """
        Tek bir indikatör hesaplar.
        
        pandas-ta fonksiyonunu çağırır ve sonucu DataFrame olarak döndürür.
        Hata durumunda boş DataFrame döner (diğer indikatörler etkilenmez).
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame. Kolonlar: open, high, low, close, volume
            Index: DatetimeIndex (timestamp)
            
        indicator : IndicatorConfig
            categories.py'den gelen indikatör yapılandırması
            
        Returns:
        -------
        pd.DataFrame
            Hesaplanan indikatör kolonları (1-5 kolon arası)
            Başarısız olursa boş DataFrame döner
            
        Örnek:
        ------
        RSI_14 → 1 kolon: RSI_14
        MACD → 3 kolon: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        Bollinger → 5 kolon: BBL, BBM, BBU, BBB, BBP
        """
        
        try:
            # pandas-ta fonksiyonunu dinamik olarak çağır
            # df.ta.rsi(length=14) gibi
            result = df.ta.__getattribute__(indicator.name)(**indicator.params)
            
            # Sonuç None olabilir (yetersiz veri durumunda)
            if result is None:
                if self.verbose:
                    logger.debug(f"  ⚠ {indicator.display_name}: None döndü (yetersiz veri?)")
                return pd.DataFrame(index=df.index)
            
            # Bazı fonksiyonlar tuple döndürür (örn: ichimoku)
            # Bu durumda ilk elementi al
            if isinstance(result, tuple):
                result = result[0] if len(result) > 0 else None
                if result is None:
                    return pd.DataFrame(index=df.index)
            
            # Series ise DataFrame'e çevir (tek kolonlu indikatörler)
            if isinstance(result, pd.Series):
                result = result.to_frame()
            
            # DataFrame değilse boş döndür
            if not isinstance(result, pd.DataFrame):
                if self.verbose:
                    logger.debug(f"  ⚠ {indicator.display_name}: Beklenmeyen tip {type(result)}")
                return pd.DataFrame(index=df.index)
            
            # Boş DataFrame kontrolü
            if result.empty:
                if self.verbose:
                    logger.debug(f"  ⚠ {indicator.display_name}: Boş sonuç")
                return pd.DataFrame(index=df.index)
            
            return result
            
        except Exception as e:
            # Hata yakalama: loglayıp devam et
            if self.verbose:
                logger.warning(f"  ✗ {indicator.display_name}: {str(e)[:80]}")
            return pd.DataFrame(index=df.index)
    
    # =========================================================================
    # KATEGORİ BAZLI HESAPLAMA
    # =========================================================================
    
    def calculate_category(
        self,
        df: pd.DataFrame,
        category: str
    ) -> pd.DataFrame:
        """
        Bir kategorideki TÜM indikatörleri hesaplar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame
            
        category : str
            Kategori: 'trend', 'momentum', 'volatility', 'volume'
            
        Returns:
        -------
        pd.DataFrame
            Orijinal OHLCV + o kategorinin tüm indikatör kolonları
            
        Örnek:
        ------
        calculate_category(df, "momentum") → RSI, MACD, Stochastic, vb. eklenir
        """
        
        # Kategorideki indikatörleri al
        indicators = get_indicators_by_category(category)
        
        if not indicators:
            raise ValueError(
                f"Geçersiz kategori: '{category}'. "
                f"Geçerli: {get_category_names()}"
            )
        
        if self.verbose:
            logger.info(f"  📊 {category.upper()} hesaplanıyor ({len(indicators)} indikatör)...")
        
        result_df = df.copy()              # Orijinali koru
        success_count = 0                  # Başarılı hesaplama sayısı
        
        for ind in indicators:
            # Tek indikatör hesapla
            ind_result = self.calculate_single(df, ind)
            
            if not ind_result.empty:
                # Yeni kolonları ekle (var olanları ezme)
                for col in ind_result.columns:
                    if col not in result_df.columns:
                        result_df[col] = ind_result[col]
                success_count += 1
        
        if self.verbose:
            logger.info(f"  ✓ {category.upper()}: {success_count}/{len(indicators)} başarılı")
        
        return result_df
    
    # =========================================================================
    # TÜM KATEGORİLERİ HESAPLA
    # =========================================================================
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        TÜM kategorilerdeki indikatörleri hesaplar.
        
        Bu fonksiyon sonrası DataFrame ~100+ kolon içerir.
        IC selector bu kolonlardan anlamlı olanları seçecek.
        
        Parameters:
        ----------
        df : pd.DataFrame
            OHLCV DataFrame (min 200 bar önerilir)
            
        categories : List[str], optional
            Hesaplanacak kategoriler
            None → tüm kategoriler: trend, momentum, volatility, volume
            
        Returns:
        -------
        pd.DataFrame
            OHLCV + tüm indikatör kolonları
            
        İstatistiksel Uyarı:
        -------------------
        100+ kolon = yüksek boyutlu veri (curse of dimensionality)
        Her kolon için IC testi yapılacak → multiple testing correction ZORUNLU
        IC selector (selector.py) → Benjamini-Hochberg FDR ile düzeltir
        """
        
        # Varsayılan: tüm kategoriler
        if categories is None:
            categories = get_category_names()
        
        if self.verbose:
            logger.info("=" * 60)
            logger.info("TÜM İNDİKATÖRLER HESAPLANIYOR")
            logger.info(f"  Kategoriler: {categories}")
            logger.info(f"  Veri: {len(df)} bar")
            logger.info("=" * 60)
        
        result_df = df.copy()
        
        for category in categories:
            # Her kategoriyi hesapla
            category_df = self.calculate_category(df, category)
            
            # Yeni kolonları ana DataFrame'e ekle
            new_cols = [c for c in category_df.columns if c not in result_df.columns]
            for col in new_cols:
                result_df[col] = category_df[col]
        
        # NaN istatistikleri
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in result_df.columns if c not in ohlcv_cols]
        
        if self.verbose and indicator_cols:
            nan_pct = result_df[indicator_cols].isnull().mean() * 100
            logger.info(f"\n  HESAPLAMA TAMAMLANDI")
            logger.info(f"  Toplam kolon: {len(result_df.columns)}")
            logger.info(f"  İndikatör kolonu: {len(indicator_cols)}")
            logger.info(f"  Ortalama NaN: {nan_pct.mean():.1f}%")
        
        return result_df
    
    # =========================================================================
    # FİYAT ÖZELLİKLERİ (Price Features)
    # =========================================================================
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temel fiyat özellikleri ekler (pandas-ta dışı, manuel hesaplama).
        
        Bu özellikler mum çubuğu (candlestick) yapısını sayısallaştırır.
        IC analizi ile hangi mum özelliklerinin tahmin gücü olduğu belirlenir.
        
        Eklenen özellikler:
        ------------------
        log_return     : Logaritmik getiri = ln(Close_t / Close_{t-1})
                         Toplamsal, simetrik, ~normal dağılım
                         
        simple_return  : Basit yüzdesel getiri = (P_t - P_{t-1}) / P_{t-1}
        
        range          : High - Low (bar volatilitesi, True Range değil)
        
        body           : Close - Open (mum gövdesi)
                         > 0 → yeşil mum (alıcı baskısı)
                         < 0 → kırmızı mum (satıcı baskısı)
                         
        body_pct       : body / Open × 100 (normalize edilmiş gövde)
                         Farklı fiyatlı coinleri karşılaştırmak için
                         
        upper_wick     : Üst fitil = High - max(Open, Close)
                         Uzun üst fitil → satış baskısı / reddedilme
                         
        lower_wick     : Alt fitil = min(Open, Close) - Low
                         Uzun alt fitil → alış baskısı / destek
                         
        gap            : Open_t - Close_{t-1} (açılış boşluğu)
                         Futures'da 7/24 açık ama hafta sonu gap olabilir
                         
        gap_pct        : gap / Close_{t-1} × 100 (normalize gap)
        
        hl_position    : (Close - Low) / (High - Low) [0-1 arası]
                         0 → close = low (günün dibinde kapanış)
                         1 → close = high (günün zirvesinde kapanış)
                         
        volume_sma_20  : Volume'un 20 barlık ortalaması
        
        volume_ratio   : Volume / volume_sma_20
                         > 1 → ortalamanın üstünde hacim (dikkat!)
                         < 1 → düşük hacim
        """
        
        result_df = df.copy()
        
        # --- Getiriler ---
        result_df['log_return'] = np.log(
            result_df['close'] / result_df['close'].shift(1)
        )
        result_df['simple_return'] = result_df['close'].pct_change()
        
        # --- Mum Yapısı ---
        result_df['range'] = result_df['high'] - result_df['low']
        result_df['body'] = result_df['close'] - result_df['open']
        result_df['body_pct'] = (result_df['body'] / result_df['open']) * 100
        
        # --- Fitiller ---
        result_df['upper_wick'] = (
            result_df['high'] - result_df[['open', 'close']].max(axis=1)
        )
        result_df['lower_wick'] = (
            result_df[['open', 'close']].min(axis=1) - result_df['low']
        )
        
        # --- Gap ---
        result_df['gap'] = result_df['open'] - result_df['close'].shift(1)
        result_df['gap_pct'] = (
            result_df['gap'] / result_df['close'].shift(1)
        ) * 100
        
        # --- Close Pozisyonu (0=Low'da, 1=High'da) ---
        # 1e-10 ekleniyor: range=0 durumunda 0'a bölme önlenir
        result_df['hl_position'] = (
            (result_df['close'] - result_df['low']) /
            (result_df['range'] + 1e-10)
        )
        
        # --- Hacim Özellikleri ---
        result_df['volume_sma_20'] = result_df['volume'].rolling(20).mean()
        result_df['volume_ratio'] = (
            result_df['volume'] / (result_df['volume_sma_20'] + 1e-10)
        )
        
        return result_df
    
    # =========================================================================
    # ROLLING İSTATİSTİKLER
    # =========================================================================
    
    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Rolling (kayan pencere) istatistiksel özellikler ekler.
        
        Her window boyutu için getiri dağılımının 4 momenti hesaplanır.
        Bu özellikler piyasanın "rejimini" sayısallaştırır.
        
        Parameters:
        ----------
        windows : List[int]
            Rolling pencere boyutları (bar sayısı)
            [10, 20, 50] → kısa, orta, uzun vade
            
        Eklenen özellikler (her window için):
        ------------------------------------
        roll{w}_ret_mean  : Ortalama getiri (trend yönü)
                            > 0 → yükselen trend
                            < 0 → düşen trend
                            
        roll{w}_ret_std   : Getiri std sapması = volatilite (σ)
                            Yüksek σ → belirsiz piyasa
                            
        roll{w}_ret_skew  : Asimetri (çarpıklık)
                            < 0 → sol kuyruk uzun (crash riski)
                            > 0 → sağ kuyruk uzun (rally potansiyeli)
                            
        roll{w}_ret_kurt  : Basıklık (tailedness)
                            > 3 → kalın kuyruk (extreme event riski)
                            = 3 → normal dağılım
                            
        roll{w}_zscore    : Z-skor = (Close - MA) / Std
                            |Z| > 2 → fiyat ortalamadan çok uzak
                            Mean-reversion sinyali için kullanışlı
                            
        roll{w}_pct_rank  : Fiyatın window içindeki yüzdelik sırası
                            0 → window'un dibi
                            1 → window'un zirvesi
        """
        
        result_df = df.copy()
        
        # Log return yoksa hesapla
        if 'log_return' not in result_df.columns:
            result_df['log_return'] = np.log(
                result_df['close'] / result_df['close'].shift(1)
            )
        
        returns = result_df['log_return']
        
        for w in windows:
            prefix = f"roll{w}_"
            
            # Getiri dağılımının 4 momenti
            result_df[f'{prefix}ret_mean'] = returns.rolling(w).mean()
            result_df[f'{prefix}ret_std'] = returns.rolling(w).std()
            result_df[f'{prefix}ret_skew'] = returns.rolling(w).skew()
            result_df[f'{prefix}ret_kurt'] = returns.rolling(w).kurt()
            
            # Z-score: Fiyatın rolling dağılımdaki pozisyonu
            roll_mean = result_df['close'].rolling(w).mean()
            roll_std = result_df['close'].rolling(w).std()
            result_df[f'{prefix}zscore'] = (
                (result_df['close'] - roll_mean) / (roll_std + 1e-10)
            )
            
            # Percentile rank (0-1 arası)
            # Son değerin window içindeki sırası
            result_df[f'{prefix}pct_rank'] = result_df['close'].rolling(w).apply(
                lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5,
                raw=False
            )
        
        return result_df
    
    # =========================================================================
    # FORWARD RETURNS (IC HEDEF DEĞİŞKENİ)
    # =========================================================================
    
    def add_forward_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        İleri (forward) getiriler ekler — IC hesaplamasının TARGET'ı.
        
        ⚠️ UYARI: Bu kolonlar SADECE backtest ve IC hesabı için!
        Canlı sistemde bu bilgi mevcut DEĞİLDİR (geleceği bilemeyiz).
        Production'da bu kolonlar olmadan çalışılır.
        
        IC Formülü:
        IC = Spearman(indicator_t, fwd_ret_{t+n})
        
        Yani: "Bu indikatörün t anındaki değeri, n bar sonraki getiriyi
        ne kadar iyi tahmin ediyor?"
        
        Parameters:
        ----------
        periods : List[int]
            İleri periyotlar
            1 = sonraki bar, 5 = 5 bar sonra, vb.
            
        Eklenen kolonlar:
        ----------------
        fwd_ret_N   : N bar sonraki log getiri
                      = ln(Close_{t+N} / Close_t)
                      
        fwd_dir_N   : N bar sonraki yön (binary)
                      1 = fiyat yükseldi
                      0 = fiyat düştü
        """
        
        result_df = df.copy()
        
        for p in periods:
            # İleri log getiri
            # shift(-p) → p bar SONRA'nın close değerini al
            result_df[f'fwd_ret_{p}'] = np.log(
                result_df['close'].shift(-p) / result_df['close']
            )
            
            # İleri yön (binary: 1=up, 0=down)
            # Sınıflandırma modelleri için kullanışlı
            result_df[f'fwd_dir_{p}'] = (
                result_df[f'fwd_ret_{p}'] > 0
            ).astype(int)
        
        return result_df
    
    # =========================================================================
    # TEMİZ VERİ
    # =========================================================================
    
    def get_clean_data(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
        drop_forward: bool = True
    ) -> pd.DataFrame:
        """
        Analiz için temizlenmiş veri döndürür.
        
        NaN'ları temizler ve opsiyonel olarak forward return
        kolonlarını kaldırır (canlı sistem için).
        
        Parameters:
        ----------
        dropna : bool
            True → NaN içeren satırları sil
            Rolling window başlangıcındaki NaN'lar temizlenir
            
        drop_forward : bool
            True → fwd_ret_* ve fwd_dir_* kolonlarını sil
            Canlı sistemde True olmalı (look-ahead bias önleme)
            
        Returns:
        -------
        pd.DataFrame
            Temizlenmiş DataFrame
        """
        
        result_df = df.copy()
        
        # Forward return kolonlarını kaldır (canlı sistem)
        if drop_forward:
            fwd_cols = [c for c in result_df.columns if c.startswith('fwd_')]
            if fwd_cols:
                result_df = result_df.drop(columns=fwd_cols)
                if self.verbose:
                    logger.info(f"  Forward kolonları silindi: {len(fwd_cols)}")
        
        # NaN'ları kaldır
        if dropna:
            before_len = len(result_df)
            result_df = result_df.dropna()
            after_len = len(result_df)
            
            if self.verbose:
                dropped = before_len - after_len
                logger.info(f"  NaN temizleme: {before_len} → {after_len} ({dropped} satır silindi)")
        
        return result_df
