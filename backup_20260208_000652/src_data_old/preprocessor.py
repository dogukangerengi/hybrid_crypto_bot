# =============================================================================
# VERİ ÖN İŞLEME MODÜLÜ (DATA PREPROCESSOR)
# =============================================================================
# Amaç: Ham OHLCV verisini IC analizi ve indikatör hesaplaması için hazırlamak
#
# Pipeline:
# 1. Missing value → ffill (look-ahead bias yok)
# 2. Return hesaplama → log return (toplamsal, ~normal dağılım)
# 3. Outlier winsorization → uç değerleri percentile'a çek
# 4. Volatilite hesaplama → Garman-Klass (OHLC bazlı, verimli)
# 5. Forward return → IC hesaplaması için gelecek getiri (TARGET)
#
# İstatistiksel Önem:
# - Veri kalitesi TÜM downstream analizleri belirler
# - Look-ahead bias → ffill DIŞI yöntemlerden kaçın
# - Outlier'ları silme DEĞİL winsorize et (veri kaybı önle)
# - Forward return = IC'nin bağımlı değişkeni (SADECE backtest'te!)
#
# Eski projeden farklar:
# - Daha hafif (gereksiz yöntemler kaldırıldı)
# - Futures-odaklı (funding rate desteği hazır)
# - Pipeline tek fonksiyonla çalışır
# =============================================================================

import pandas as pd                            # Veri manipülasyonu
import numpy as np                             # Sayısal hesaplamalar
import logging                                 # Log yönetimi
from typing import Optional, Dict, Tuple, List # Tip belirteçleri
from scipy import stats                        # İstatistiksel fonksiyonlar

# Logger
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    OHLCV verisini analiz için ön işleme tabi tutan sınıf.
    
    IC analizi zincirindeki yeri:
    
    Ham OHLCV → [PREPROCESSOR] → Temiz Veri → İndikatörler → IC Analizi
    
    Her method bağımsız çalışır (stateless tasarım).
    Pipeline ile hepsini sırasıyla uygulayabilirsin.
    
    Kullanım:
    --------
    preprocessor = DataPreprocessor()
    
    # Tek adımda tüm ön işleme:
    df_clean = preprocessor.full_pipeline(df_raw)
    
    # Veya adım adım:
    df = preprocessor.handle_missing(df)
    df = preprocessor.add_returns(df)
    df = preprocessor.winsorize_returns(df)
    """
    
    def __init__(self):
        """
        Stateless preprocessor başlatır.
        Hiçbir state tutmaz, her çağrı bağımsız çalışır.
        """
        pass
    
    # =========================================================================
    # 1. EKSİK VERİ İŞLEME
    # =========================================================================
    
    def handle_missing(
        self,
        df: pd.DataFrame,
        method: str = "ffill",
        max_gap: int = 5
    ) -> pd.DataFrame:
        """
        Eksik verileri tespit eder ve doldurur.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            OHLCV DataFrame
            
        method : str
            "ffill" = Forward fill (önceki değerle doldur)
            SADECE ffill kullan! Diğerleri look-ahead bias riski taşır.
            
        max_gap : int
            Ardışık eksik veri sayısı bu değeri aşarsa doldurma yapılmaz.
            Uzun gap'ler genellikle borsa maintenance'ını gösterir.
        
        Döndürür:
        --------
        pd.DataFrame
            Eksik değerleri işlenmiş DataFrame
        
        İstatistiksel Not:
        -----------------
        Forward fill NEDEN güvenli?
        → Sadece t zamanında bildiğin veriyi (t-1) kullanır
        → Look-ahead bias = 0
        
        Backward fill NEDEN tehlikeli?
        → t+1 zamanındaki veriyi t'de kullanır
        → Look-ahead bias = ∞ (tüm analiz geçersiz)
        """
        df_clean = df.copy()
        
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before == 0:
            return df_clean
        
        # Forward fill: Önceki geçerli değerle doldur
        # limit=max_gap: En fazla max_gap ardışık NaN doldur
        df_clean = df_clean.ffill(limit=max_gap)
        
        # Başlangıçtaki NaN'ları da backward fill ile doldur
        # (Sadece ilk birkaç satır - look-ahead bias riski minimal)
        df_clean = df_clean.bfill(limit=2)
        
        missing_after = df_clean.isnull().sum().sum()
        
        if missing_before > 0:
            logger.info(
                f"  Missing: {missing_before} → {missing_after} "
                f"({missing_before - missing_after} dolduruldu)"
            )
        
        return df_clean
    
    # =========================================================================
    # 2. RETURN (GETİRİ) HESAPLAMA
    # =========================================================================
    
    def add_returns(
        self,
        df: pd.DataFrame,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        Fiyat verisinden getiri (return) hesaplar.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            En az 'close' kolonu içermeli
            
        method : str
            "log" = Logaritmik return: ln(P_t / P_{t-1})
            "simple" = Basit return: (P_t - P_{t-1}) / P_{t-1}
        
        Döndürür:
        --------
        pd.DataFrame
            Orijinal + 'log_return' ve 'simple_return' kolonları
        
        İstatistiksel Not:
        -----------------
        Log return avantajları:
        1. Toplamsal: r_total = r_1 + r_2 + ... + r_n
           (Simple return çarpımsal: R_total = (1+r_1)(1+r_2)...-1)
        2. Simetri: +10% ve -10% log return aynı büyüklükte
        3. Normal dağılıma daha yakın (CLT varsayımı için)
        4. Volatilite hesaplamaları için daha uygun
        
        IC hesaplamasında log return tercih edilir çünkü:
        - Spearman korelasyonu rank-based, ama dağılım yakınlığı yine önemli
        - Extreme return'ler log'da daha simetrik → daha güvenilir IC
        """
        result = df.copy()
        
        # Log return: ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})
        result['log_return'] = np.log(
            result['close'] / result['close'].shift(1)
        )
        
        # Simple return: (P_t - P_{t-1}) / P_{t-1}
        result['simple_return'] = result['close'].pct_change()
        
        return result
    
    # =========================================================================
    # 3. OUTLIER İŞLEME (WİNSORİZATİON)
    # =========================================================================
    
    def winsorize_returns(
        self,
        df: pd.DataFrame,
        column: str = "log_return",
        lower_pct: float = 0.5,
        upper_pct: float = 99.5
    ) -> pd.DataFrame:
        """
        Uç değerleri (outlier) percentile değerlerine çeker.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Return kolonu içeren DataFrame
            
        column : str
            Winsorize edilecek kolon
            
        lower_pct : float
            Alt percentile (varsayılan: 0.5 → %0.5 alt uç)
            
        upper_pct : float
            Üst percentile (varsayılan: 99.5 → %0.5 üst uç)
        
        Döndürür:
        --------
        pd.DataFrame
            Winsorize edilmiş DataFrame
        
        Neden Winsorization (silme veya NaN yerine)?
        -------------------------------------------
        1. Veri kaybı yok → sample size korunur → IC istatistiksel gücü korunur
        2. Uç değerlerin etkisi azalır → IC daha stabil
        3. Gerçek crash'ler tamamen silinmez → realistic backtest
        
        Neden %0.5 / %99.5?
        → %1 çok agresif (gerçek hareketleri de kırpar)
        → %0.1 çok gevşek (flash crash'ler bozar)
        → %0.5 optimal trade-off (finans literatürü standardı)
        """
        result = df.copy()
        
        if column not in result.columns:
            return result
        
        # Percentile değerlerini hesapla
        lower_val = result[column].quantile(lower_pct / 100)  # %0.5'lik değer
        upper_val = result[column].quantile(upper_pct / 100)  # %99.5'lik değer
        
        # Clip: Değerleri [lower, upper] aralığına sınırla
        before_outliers = (
            (result[column] < lower_val) | (result[column] > upper_val)
        ).sum()
        
        result[column] = result[column].clip(lower_val, upper_val)
        
        if before_outliers > 0:
            logger.info(
                f"  Winsorize ({column}): {before_outliers} outlier "
                f"[{lower_val:.4f}, {upper_val:.4f}] aralığına çekildi"
            )
        
        return result
    
    # =========================================================================
    # 4. VOLATİLİTE HESAPLAMA
    # =========================================================================
    
    def add_volatility(
        self,
        df: pd.DataFrame,
        window: int = 20,
        method: str = "garman_klass"
    ) -> pd.DataFrame:
        """
        Rolling volatilite hesaplar.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            OHLCV DataFrame
            
        window : int
            Rolling window boyutu (bar sayısı)
            20 = ~1 gün @ 1h TF, ~5 saat @ 15m TF
            
        method : str
            "standard" = Standart sapma (sadece close kullanır)
            "garman_klass" = OHLC bazlı (en verimli estimator)
        
        Döndürür:
        --------
        pd.DataFrame
            Orijinal + 'volatility' kolonu
        
        Garman-Klass Neden Daha İyi?
        ----------------------------
        Standard:    Sadece close fiyatını kullanır → bilgi kaybı
        Parkinson:   High-Low kullanır → ~5x daha verimli
        Garman-Klass: Open-High-Low-Close → ~8x daha verimli
        
        "Verimli" = Aynı doğruluk için daha az veri noktası gerektirir
        → Daha kısa pencerede bile güvenilir volatilite tahmini
        """
        result = df.copy()
        
        if method == "garman_klass":
            # Garman-Klass formülü:
            # GK = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
            # Volatilite = sqrt(rolling_mean(GK))
            
            log_hl = np.log(result['high'] / result['low'])     # ln(High/Low)
            log_co = np.log(result['close'] / result['open'])   # ln(Close/Open)
            
            # GK bileşenleri
            gk = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
            
            # Rolling ortalama alıp karekök → volatilite
            result['volatility'] = np.sqrt(gk.rolling(window=window).mean())
            
        elif method == "standard":
            # Basit standart sapma (sadece close return kullanır)
            if 'log_return' not in result.columns:
                result['log_return'] = np.log(
                    result['close'] / result['close'].shift(1)
                )
            result['volatility'] = result['log_return'].rolling(window=window).std()
        
        return result
    
    # =========================================================================
    # 5. FORWARD RETURN (HEDEF DEĞİŞKEN)
    # =========================================================================
    
    def add_forward_returns(
        self,
        df: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        İleri getiriler ekler (IC hesaplamasının TARGET değişkeni).
        
        ⚠️ KRİTİK UYARI:
        Bu kolonlar SADECE IC hesaplama ve backtest'te kullanılmalı!
        Canlı sistemde bu bilgi mevcut DEĞİL (gelecek bilinmiyor).
        Bu kolonları modele input olarak vermek = LOOK-AHEAD BIAS.
        
        Parametreler:
        ------------
        df : pd.DataFrame
            En az 'close' kolonu
            
        periods : List[int]
            İleri periyotlar
            1 = sonraki bar, 5 = 5 bar sonra, vb.
            
            IC hesaplamasında genellikle target_period=5 kullanıyoruz.
            Bu, "bu indikatör 5 bar sonraki getiriyi tahmin edebiliyor mu?"
            sorusunu cevaplar.
        
        Döndürür:
        --------
        pd.DataFrame
            Orijinal + fwd_ret_N ve fwd_dir_N kolonları
            
            fwd_ret_5 = 5 bar sonraki log return (sürekli değer)
            fwd_dir_5 = 5 bar sonra yön (1=yukarı, 0=aşağı, binary)
        
        IC Formülü Hatırlatma:
        ---------------------
        IC = Spearman(indicator_t, fwd_ret_t)
        IC > 0 → İndikatör yükselince fiyat da yükseliyor
        IC < 0 → İndikatör yükselince fiyat düşüyor
        """
        result = df.copy()
        
        for p in periods:
            # Log return: ln(P_{t+p} / P_t)
            # shift(-p) = gelecek p bar'ın fiyatı
            result[f'fwd_ret_{p}'] = np.log(
                result['close'].shift(-p) / result['close']
            )
            
            # Binary yön: 1 = yukarı (pozitif return), 0 = aşağı (negatif)
            result[f'fwd_dir_{p}'] = (result[f'fwd_ret_{p}'] > 0).astype(int)
        
        return result
    
    # =========================================================================
    # 6. FİYAT ÖZELLİKLERİ (PRICE FEATURES)
    # =========================================================================
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temel fiyat özelliklerini ekler (pandas-ta dışı, ham hesaplamalar).
        
        Bu özellikler indikatör hesaplamasından ÖNCE eklenir.
        Bazıları tek başına IC analizi için kullanılabilir.
        
        Eklenen Özellikler:
        ------------------
        range: High - Low (bar aralığı, volatilite proxy)
        body: Close - Open (mum gövdesi, alım/satım baskısı)
        body_pct: body / open * 100 (normalize gövde)
        upper_wick: Üst fitil (satış baskısı)
        lower_wick: Alt fitil (alım baskısı)
        gap: Open_t - Close_{t-1} (açılış boşluğu)
        hl_position: Close'un High-Low aralığındaki yeri (0=Low, 1=High)
        volume_ratio: Volume / SMA(Volume, 20) (hacim anomalisi)
        
        Döndürür:
        --------
        pd.DataFrame
            Orijinal + yeni özellik kolonları
        """
        result = df.copy()
        
        # Bar aralığı: Tek bar volatilitesi
        result['range'] = result['high'] - result['low']
        
        # Mum gövdesi: Pozitif = bullish (yeşil), Negatif = bearish (kırmızı)
        result['body'] = result['close'] - result['open']
        
        # Normalize gövde: Farklı fiyatlı coinleri karşılaştırmak için
        result['body_pct'] = (result['body'] / result['open']) * 100
        
        # Fitiller: Rejection sinyali
        # Üst fitil uzun = Satış baskısı (fiyat High'a çıkıp geri gelmiş)
        result['upper_wick'] = result['high'] - result[['open', 'close']].max(axis=1)
        # Alt fitil uzun = Alım baskısı (fiyat Low'a inip geri çıkmış)
        result['lower_wick'] = result[['open', 'close']].min(axis=1) - result['low']
        
        # Gap: Bir önceki bar'ın close'u ile bu bar'ın open'ı arasındaki fark
        result['gap'] = result['open'] - result['close'].shift(1)
        result['gap_pct'] = (result['gap'] / result['close'].shift(1)) * 100
        
        # Close'un High-Low aralığındaki pozisyonu (0=Low'da, 1=High'da)
        # 0.5'e yakınsa = doji (kararsızlık)
        # 1'e yakınsa = güçlü close (bullish)
        # 0'a yakınsa = zayıf close (bearish)
        result['hl_position'] = (
            (result['close'] - result['low']) / 
            (result['range'] + 1e-10)  # Sıfıra bölme koruması
        )
        
        # Hacim oranı: Ortalama hacmin kaç katı?
        # >1.5 = anormal hacim, breakout sinyali olabilir
        # <0.5 = düşük hacim, sahte hareket olabilir
        result['volume_sma_20'] = result['volume'].rolling(20).mean()
        result['volume_ratio'] = result['volume'] / (result['volume_sma_20'] + 1e-10)
        
        return result
    
    # =========================================================================
    # 7. ROLLING İSTATİSTİKLER
    # =========================================================================
    
    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Rolling istatistiksel özellikler ekler.
        
        Parametreler:
        ------------
        windows : List[int]
            Rolling pencere boyutları
            [10, 20, 50] = kısa, orta, uzun vade
        
        Her window için eklenen özellikler:
        ----------------------------------
        roll{w}_ret_mean : Ortalama getiri (trend yönü)
        roll{w}_ret_std  : Getiri volatilitesi (σ)
        roll{w}_ret_skew : Asimetri (kuyruk yönü)
        roll{w}_ret_kurt : Basıklık (kuyruk kalınlığı)
        roll{w}_zscore   : (Close - MA) / Std (mean-reversion sinyali)
        
        İstatistiksel Yorum:
        -------------------
        Skewness < 0 → Sol kuyruk uzun → crash riski yüksek
        Kurtosis > 3 → Fat-tail → extreme event riski yüksek
        |Z-score| > 2 → Fiyat ortalamanın 2σ uzağında → mean-reversion beklenir
        """
        result = df.copy()
        
        # Log return yoksa hesapla
        if 'log_return' not in result.columns:
            result['log_return'] = np.log(
                result['close'] / result['close'].shift(1)
            )
        
        returns = result['log_return']
        
        for w in windows:
            prefix = f"roll{w}_"
            
            # 1. Getiri istatistikleri
            result[f'{prefix}ret_mean'] = returns.rolling(w).mean()   # Trend yönü
            result[f'{prefix}ret_std'] = returns.rolling(w).std()     # Volatilite
            result[f'{prefix}ret_skew'] = returns.rolling(w).skew()   # Asimetri
            result[f'{prefix}ret_kurt'] = returns.rolling(w).kurt()   # Basıklık
            
            # 2. Z-score: Fiyatın rolling ortalamasına göre konumu
            # Z > 0: Ortalamanın üstünde (overvalued?)
            # Z < 0: Ortalamanın altında (undervalued?)
            roll_mean = result['close'].rolling(w).mean()
            roll_std = result['close'].rolling(w).std()
            result[f'{prefix}zscore'] = (
                (result['close'] - roll_mean) / (roll_std + 1e-10)
            )
        
        return result
    
    # =========================================================================
    # FULL PIPELINE: TEK FONKSİYONLA TÜM ÖN İŞLEME
    # =========================================================================
    
    def full_pipeline(
        self,
        df: pd.DataFrame,
        forward_periods: List[int] = [1, 5, 10, 20],
        rolling_windows: List[int] = [10, 20, 50],
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Tüm ön işleme adımlarını sırasıyla uygular.
        
        Bu fonksiyon, ham OHLCV → IC-ready veri dönüşümünü tek adımda yapar.
        
        Pipeline Sırası:
        ---------------
        1. Missing values → ffill (look-ahead bias güvenli)
        2. Returns → log + simple return hesaplama
        3. Winsorization → uç değerleri %0.5-%99.5'e çek
        4. Price features → range, body, wick, gap, volume_ratio
        5. Rolling stats → mean, std, skew, kurt, zscore
        6. Volatility → Garman-Klass (OHLC bazlı)
        7. Forward returns → IC'nin hedef değişkeni (TARGET)
        8. NaN temizleme → rolling başlangıcındaki NaN'ları kaldır
        
        Parametreler:
        ------------
        df : pd.DataFrame
            Ham OHLCV DataFrame
            
        forward_periods : List[int]
            Forward return periyotları [1, 5, 10, 20]
            
        rolling_windows : List[int]
            Rolling istatistik pencereleri [10, 20, 50]
            
        drop_na : bool
            True ise NaN satırları kaldır (önerilir)
        
        Döndürür:
        --------
        pd.DataFrame
            Tam işlenmiş, IC analizine hazır DataFrame
            
        Örnek:
        ------
        >>> pp = DataPreprocessor()
        >>> df_raw = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", 500)
        >>> df_clean = pp.full_pipeline(df_raw)
        >>> print(df_clean.shape)
        (450, 25)  # NaN kırpılmış, tüm özellikler ekli
        """
        logger.info("📋 Preprocessing pipeline başlıyor...")
        
        rows_before = len(df)
        
        # 1. Missing values
        result = self.handle_missing(df)
        
        # 2. Returns
        result = self.add_returns(result)
        
        # 3. Winsorization
        result = self.winsorize_returns(result, column='log_return')
        
        # 4. Price features
        result = self.add_price_features(result)
        
        # 5. Rolling stats
        result = self.add_rolling_stats(result, windows=rolling_windows)
        
        # 6. Volatility
        result = self.add_volatility(result)
        
        # 7. Forward returns (IC hedef değişkeni)
        result = self.add_forward_returns(result, periods=forward_periods)
        
        # 8. NaN temizleme
        if drop_na:
            result = result.dropna()
        
        rows_after = len(result)
        new_cols = len(result.columns) - len(df.columns)
        
        logger.info(
            f"  ✓ Pipeline tamamlandı: {rows_before} → {rows_after} satır, "
            f"+{new_cols} yeni kolon"
        )
        
        return result
    
    # =========================================================================
    # VERİ KALİTE RAPORU
    # =========================================================================
    
    def quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Veri kalite özet raporu oluşturur.
        
        Debug ve doğrulama için kullanılır.
        Her yeni veri kaynağında bir kez çalıştırılmalı.
        
        Döndürür:
        --------
        Dict
            Kalite metrikleri
        """
        report = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_total': int(df.isnull().sum().sum()),
            'missing_pct': float(df.isnull().mean().mean() * 100),
        }
        
        # Return istatistikleri (varsa)
        if 'log_return' in df.columns:
            returns = df['log_return'].dropna()
            report['return_stats'] = {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'skew': float(returns.skew()),
                'kurt': float(returns.kurtosis()),
                'min': float(returns.min()),
                'max': float(returns.max()),
            }
        
        # Volatilite istatistikleri (varsa)
        if 'volatility' in df.columns:
            vol = df['volatility'].dropna()
            report['volatility_stats'] = {
                'current': float(vol.iloc[-1]),
                'mean': float(vol.mean()),
                'percentile_25': float(vol.quantile(0.25)),
                'percentile_75': float(vol.quantile(0.75)),
            }
        
        return report


# =============================================================================
# TEST KODU
# =============================================================================
if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 60)
    print("  PREPROCESSOR TEST")
    print("=" * 60)
    
    # Rastgele OHLCV verisi oluştur (test için)
    np.random.seed(42)
    n = 500
    
    # Random walk fiyat serisi
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    df_test = pd.DataFrame({
        'open': price + np.random.randn(n) * 50,
        'high': price + abs(np.random.randn(n) * 200),
        'low': price - abs(np.random.randn(n) * 200),
        'close': price,
        'volume': np.random.exponential(1000, n),
    })
    
    # High/Low düzelt (tutarlılık için)
    df_test['high'] = df_test[['open', 'high', 'close']].max(axis=1) + 10
    df_test['low'] = df_test[['open', 'low', 'close']].min(axis=1) - 10
    
    # Timestamp ekle
    df_test.index = pd.date_range('2025-01-01', periods=n, freq='1h', tz='UTC')
    
    # Birkaç NaN ekle (test)
    df_test.iloc[10:12, 0] = np.nan
    
    print(f"\nTest verisi: {len(df_test)} satır, {len(df_test.columns)} kolon")
    
    # Preprocessor test
    pp = DataPreprocessor()
    
    # 1. Tek tek adımlar
    print("\n[1] Handle missing:")
    df1 = pp.handle_missing(df_test)
    
    print("\n[2] Add returns:")
    df2 = pp.add_returns(df1)
    print(f"   Return kolonları: {[c for c in df2.columns if 'return' in c]}")
    
    print("\n[3] Winsorize:")
    df3 = pp.winsorize_returns(df2)
    
    print("\n[4] Price features:")
    df4 = pp.add_price_features(df3)
    print(f"   Yeni kolonlar: {[c for c in df4.columns if c not in df3.columns]}")
    
    print("\n[5] Rolling stats:")
    df5 = pp.add_rolling_stats(df4, windows=[10, 20])
    print(f"   Roll kolonları: {[c for c in df5.columns if 'roll' in c]}")
    
    print("\n[6] Volatility:")
    df6 = pp.add_volatility(df5)
    print(f"   Son volatilite: {df6['volatility'].iloc[-1]:.6f}")
    
    print("\n[7] Forward returns:")
    df7 = pp.add_forward_returns(df6, periods=[1, 5])
    print(f"   Forward kolonlar: {[c for c in df7.columns if 'fwd' in c]}")
    
    # 2. Full pipeline
    print("\n" + "=" * 60)
    print("[FULL PIPELINE]")
    df_clean = pp.full_pipeline(df_test)
    print(f"   Sonuç: {df_clean.shape[0]} satır, {df_clean.shape[1]} kolon")
    
    # 3. Kalite raporu
    print("\n[KALİTE RAPORU]")
    report = pp.quality_report(df_clean)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"   {k}:")
            for k2, v2 in v.items():
                print(f"     {k2}: {v2:.6f}" if isinstance(v2, float) else f"     {k2}: {v2}")
        else:
            print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("  TÜM TESTLER TAMAMLANDI ✅")
    print("=" * 60)
