# =============================================================================
# BİTGET FUTURES VERİ ÇEKME MODÜLÜ (DATA FETCHER)
# =============================================================================
# Amaç: CCXT ile Bitget USDT-M Perpetual Futures'dan OHLCV verisi çekmek
#
# Eski projeden farklar:
# - Binance → Bitget (swap market)
# - Sembol format: "BTC/USDT:USDT" (Futures perpetual)
# - Dinamik sembol desteği (sadece BTC değil, tüm USDT çiftleri)
# - Contract size ve lot bilgisi çekme
# - config.py entegrasyonu
#
# İstatistiksel Not:
# - Daha fazla veri = daha güvenilir IC analizi (larger sample size)
# - Çok eski veri = rejim değişikliği riski (non-stationarity)
# - Optimal: 3-6 ay veri (trade-off)
# =============================================================================

import ccxt                                    # Borsa unified API'si
import pandas as pd                            # Veri yapıları
import numpy as np                             # Sayısal hesaplamalar
import time                                    # Rate limiting için bekleme
import logging                                 # Log yönetimi
from datetime import datetime, timedelta, timezone  # Zaman işlemleri
from typing import Optional, List, Dict, Tuple      # Tip belirteçleri
from pathlib import Path                       # Dosya yolu işlemleri

# Proje config'ini import et
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg                         # Merkezi yapılandırma

# Logger ayarla
logger = logging.getLogger(__name__)


class BitgetFetcher:
    """
    Bitget USDT-M Perpetual Futures'dan OHLCV verisi çeken sınıf.
    
    Eski projeden (DataFetcher) adapte edildi:
    - Bitget swap market desteği
    - Dinamik sembol (herhangi bir USDT Futures çifti)
    - Pagination ile büyük veri çekme
    - Market info (lot size, precision, max leverage)
    
    Kullanım:
    --------
    fetcher = BitgetFetcher()
    
    # Tek timeframe
    df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=500)
    
    # Maksimum veri
    df = fetcher.fetch_max_ohlcv("ETH/USDT:USDT", "15m", max_bars=2000)
    
    # Tüm timeframe'ler
    data = fetcher.fetch_all_timeframes("SOL/USDT:USDT")
    """
    
    # =========================================================================
    # TİMEFRAME TANIMLARI
    # =========================================================================
    # Her timeframe'in dakika cinsinden karşılığı
    # IC hesaplaması ve volatilite ölçekleme için gerekli
    
    TIMEFRAME_MINUTES: Dict[str, int] = {
        "1m": 1,           # Scalping - çok gürültülü, IC güvenilirliği düşük
        "3m": 3,           # Kısa scalping
        "5m": 5,           # Day trading kısa vade ⭐
        "15m": 15,         # Day trading ana TF ⭐
        "30m": 30,         # Trend konfirmasyonu ⭐
        "1h": 60,          # Intraday trend ⭐
        "2h": 120,         # Swing noktaları ⭐
        "4h": 240,         # Büyük resim, major S/R ⭐
        "6h": 360,         # Pozisyon trading
        "12h": 720,        # Uzun vade
        "1d": 1440,        # Pozisyon / HODLing
        "1w": 10080,       # Haftalık trend
    }
    
    # =========================================================================
    # AKTİF TİMEFRAME'LER (settings.yaml'dan okunur, fallback burada)
    # =========================================================================
    DEFAULT_ACTIVE_TIMEFRAMES: List[str] = [
        "5m", "15m", "30m", "1h", "2h", "4h"
    ]
    
    # =========================================================================
    # ÖNERİLEN BAR SAYILARI (her TF için optimal veri miktarı)
    # =========================================================================
    # Kısa TF → daha fazla bar (gürültü fazla, sample size önemli)
    # Uzun TF → daha az bar (her bar daha bilgi yoğun)
    RECOMMENDED_BARS: Dict[str, int] = {
        "1m": 10000,       # ~7 gün
        "3m": 7000,        # ~14 gün
        "5m": 5000,        # ~17 gün
        "15m": 4000,       # ~42 gün (~6 hafta)
        "30m": 3000,       # ~62 gün (~2 ay)
        "1h": 2000,        # ~83 gün (~3 ay)
        "2h": 1500,        # ~125 gün (~4 ay)
        "4h": 1000,        # ~166 gün (~5.5 ay)
        "6h": 750,         # ~187 gün
        "12h": 500,        # ~250 gün
        "1d": 365,         # 1 yıl
        "1w": 104,         # 2 yıl
    }
    
    # =========================================================================
    # BİTGET API LİMİTLERİ
    # =========================================================================
    MAX_CANDLES_PER_REQUEST = 200             # Bitget'in tek istekte max mum sayısı
    RATE_LIMIT_DELAY = 0.15                   # İstekler arası bekleme (saniye)
    
    def __init__(self, symbol: str = None):
        """
        BitgetFetcher başlatır.
        
        Parametreler:
        ------------
        symbol : str, optional
            Varsayılan sembol (örn: "BTC/USDT:USDT")
            None ise config'deki default_symbol kullanılır
        """
        # Varsayılan sembol
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # CCXT Bitget exchange nesnesi (public API - key gerektirmez)
        # Veri çekmek için API key'e ihtiyaç yok
        self.exchange = ccxt.bitget({
            'options': {
                'defaultType': 'swap',       # swap = USDT-M Perpetual Futures
            },
            'enableRateLimit': True,          # Otomatik rate limiting
        })
        
        # Market bilgilerini yükle (çift listesi, lot boyutu, precision)
        self._markets_loaded = False          # Lazy loading flag
        self._market_cache: Dict = {}         # Market bilgisi cache
    
    # =========================================================================
    # MARKET BİLGİSİ
    # =========================================================================
    
    def _ensure_markets_loaded(self):
        """
        Market bilgilerini lazy-load eder (ilk erişimde bir kez yüklenir).
        
        Neden lazy loading?
        - Her script çalışmasında markets yüklenmek zorunda değil
        - İlk veri çekme isteğinde otomatik yüklenir
        - ~2-3 saniye sürer, gereksiz bekleme önlenir
        """
        if not self._markets_loaded:
            logger.info("Bitget market bilgileri yükleniyor...")
            self.exchange.load_markets()
            self._markets_loaded = True
            
            # USDT Futures çift sayısını logla
            usdt_count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
            logger.info(f"✓ {usdt_count} USDT-M Futures çifti yüklendi")
    
    def get_market_info(self, symbol: str = None) -> Dict:
        """
        Bir sembolün market bilgisini döndürür.
        
        İçerik:
        - contractSize: Kontrat büyüklüğü (örn: BTC = 1 kontrat = 0.001 BTC)
        - precision: Fiyat ve miktar hassasiyeti
        - limits: Min/max sipariş miktarları
        - maxLeverage: Maximum kaldıraç
        
        Bu bilgiler emir gönderirken kritik:
        - Yanlış precision → emir reddedilir
        - Min amount altında → emir reddedilir
        
        Parametreler:
        ------------
        symbol : str
            Sembol (örn: "BTC/USDT:USDT")
        
        Döndürür:
        --------
        Dict
            Market bilgisi
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        if symbol not in self.exchange.markets:
            raise ValueError(
                f"'{symbol}' Bitget Futures'da bulunamadı. "
                f"Doğru format: 'BTC/USDT:USDT'"
            )
        
        market = self.exchange.markets[symbol]
        
        # Kullanışlı bilgileri düzenle
        info = {
            'symbol': symbol,
            'type': market.get('type', 'unknown'),        # swap = futures
            'contract_size': market.get('contractSize', 1), # Kontrat büyüklüğü
            'precision': {
                'price': market.get('precision', {}).get('price', 0.01),
                'amount': market.get('precision', {}).get('amount', 0.001),
            },
            'limits': {
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
                'min_cost': market.get('limits', {}).get('cost', {}).get('min', 5),
                'max_amount': market.get('limits', {}).get('amount', {}).get('max', None),
            },
            'max_leverage': int(market.get('info', {}).get('maxLever', 20)),
        }
        
        return info
    
    def get_all_usdt_futures(self) -> List[str]:
        """
        Bitget'teki TÜM USDT-M Futures çiftlerini döndürür.
        
        Scanner modülü bu listeyi kullanarak hacim filtreleme yapacak.
        
        Döndürür:
        --------
        List[str]
            Sembol listesi (örn: ["BTC/USDT:USDT", "ETH/USDT:USDT", ...])
        """
        self._ensure_markets_loaded()
        
        # ':USDT' ile biten semboller = USDT-M Perpetual Futures
        usdt_futures = [
            symbol for symbol in self.exchange.markets.keys()
            if symbol.endswith(':USDT')
        ]
        
        return sorted(usdt_futures)
    
    # =========================================================================
    # FİYAT BİLGİSİ (TİCKER)
    # =========================================================================
    
    def get_ticker(self, symbol: str = None) -> Dict[str, float]:
        """
        Anlık fiyat bilgisini çeker.
        
        Ticker verisi:
        - last: Son işlem fiyatı
        - bid/ask: Alış/satış fiyatı
        - 24h high/low/volume/change: Günlük istatistikler
        
        Parametreler:
        ------------
        symbol : str
            Sembol (varsayılan: config'deki default)
        
        Döndürür:
        --------
        Dict[str, float]
            Fiyat bilgileri
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'last': ticker.get('last', 0),           # Son fiyat
                'bid': ticker.get('bid', 0),              # En iyi alış
                'ask': ticker.get('ask', 0),              # En iyi satış
                'spread': (ticker.get('ask', 0) - ticker.get('bid', 0)),  # Bid-ask farkı
                'high_24h': ticker.get('high', 0),        # 24h en yüksek
                'low_24h': ticker.get('low', 0),          # 24h en düşük
                'volume_24h': ticker.get('quoteVolume', 0),  # 24h USDT hacim
                'change_24h': ticker.get('percentage', 0),   # 24h % değişim
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            }
            
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Ağ hatası (ticker): {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Borsa hatası (ticker): {e}")
    
    # =========================================================================
    # OHLCV VERİ ÇEKME (TEK İSTEK)
    # =========================================================================
    
    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        limit: int = 200,                     # Bitget max: 200
        since: Optional[int] = None           # Başlangıç timestamp (ms)
    ) -> pd.DataFrame:
        """
        Tek istekte OHLCV (mum) verisi çeker.
        
        OHLCV = Open, High, Low, Close, Volume
        Her satır bir "mum"u (candlestick) temsil eder.
        
        Parametreler:
        ------------
        symbol : str
            İşlem çifti (örn: "BTC/USDT:USDT")
            
        timeframe : str
            Zaman dilimi (5m, 15m, 30m, 1h, 2h, 4h, vb.)
            
        limit : int
            Çekilecek mum sayısı (Bitget max: 200 per request)
            
        since : int, optional
            Başlangıç zamanı (Unix timestamp, milisaniye)
            None ise en son mumlardan geriye doğru çeker
        
        Döndürür:
        --------
        pd.DataFrame
            Index: timestamp (UTC, timezone-aware)
            Kolonlar: open, high, low, close, volume
        
        İstatistiksel Not:
        -----------------
        - Her mum bağımsız bir gözlem (observation) değildir
        - Ardışık mumlar arasında otokorelasyon vardır
        - IC hesaplamasında bu göz önünde bulundurulmalı
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        # Timeframe validasyonu
        if timeframe not in self.TIMEFRAME_MINUTES:
            valid = list(self.TIMEFRAME_MINUTES.keys())
            raise ValueError(f"Geçersiz timeframe: {timeframe}. Geçerli: {valid}")
        
        # Sembol validasyonu
        if symbol not in self.exchange.markets:
            raise ValueError(f"'{symbol}' Bitget Futures'da bulunamadı")
        
        # Bitget limiti
        limit = min(limit, self.MAX_CANDLES_PER_REQUEST)
        
        try:
            # CCXT unified fetch_ohlcv çağrısı
            # Döndürdüğü format: [[timestamp_ms, open, high, low, close, volume], ...]
            ohlcv_raw = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )
            
            if not ohlcv_raw:
                logger.warning(f"{symbol} {timeframe}: Boş veri döndü")
                return pd.DataFrame()
            
            # Ham listeyi pandas DataFrame'e çevir
            df = pd.DataFrame(
                ohlcv_raw,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Unix timestamp (ms) → datetime (UTC, timezone-aware)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)  # Timestamp'i index yap
            df.index.name = None                      # Index adını temizle
            
            # Veri tiplerini float64'e optimize et (tutarlılık için)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype('float64')
            
            return df
            
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Ağ hatası ({symbol} {timeframe}): {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Borsa hatası ({symbol} {timeframe}): {e}")
    
    # =========================================================================
    # MAKSİMUM VERİ ÇEKME (PAGİNATION)
    # =========================================================================
    
    def fetch_max_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        max_bars: Optional[int] = None,       # None = önerilen miktar
        progress: bool = True                  # İlerleme göster
    ) -> pd.DataFrame:
        """
        Pagination ile Bitget'in 200-bar limitini aşarak büyük veri çeker.
        
        Strateji: Geçmişten bugüne doğru chunk'lar halinde çeker.
        Her chunk 200 bar, since parametresi ile ilerler.
        
        Parametreler:
        ------------
        symbol : str
            İşlem çifti
            
        timeframe : str
            Zaman dilimi
            
        max_bars : int, optional
            Hedef mum sayısı. None ise RECOMMENDED_BARS kullanılır.
            
        progress : bool
            İlerleme durumu göster
        
        Döndürür:
        --------
        pd.DataFrame
            Birleştirilmiş OHLCV DataFrame (kronolojik sıralı)
        
        İstatistiksel Önem:
        ------------------
        Daha fazla veri:
        + IC hesaplamasında daha yüksek istatistiksel güç (statistical power)
        + Walk-forward validation için daha büyük out-of-sample pencere
        - Rejim değişikliği riski (eski verinin artık geçerli olmaması)
        Optimal: 3-6 ay veri
        """
        symbol = symbol or self.default_symbol
        
        # Hedef bar sayısını belirle
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 1000)
        
        # Tahmini gün sayısı (loglama için)
        tf_minutes = self.TIMEFRAME_MINUTES[timeframe]
        estimated_days = (max_bars * tf_minutes) / (60 * 24)
        
        if progress:
            logger.info(
                f"📊 {symbol} | {timeframe} | "
                f"Hedef: {max_bars} bar (~{estimated_days:.0f} gün)"
            )
        
        # =====================================================================
        # PAGİNATION: Geçmişten bugüne doğru chunk'lar halinde çek
        # =====================================================================
        # 1. Başlangıç tarihini hesapla (şu an - tahmini süre - %20 buffer)
        # 2. since parametresi ile ileri doğru ilerle
        # 3. Her chunk son bar'ın timestamp'inden devam eder
        # =====================================================================
        
        buffer_factor = 1.3                   # %30 buffer (hafta sonu/gap'ler için)
        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=int(max_bars * tf_minutes * buffer_factor)
        )
        since_ms = int(start_time.timestamp() * 1000)  # Milisaniye cinsinden
        
        all_chunks: List[pd.DataFrame] = []   # Toplanan chunk'lar
        total_fetched = 0                     # Toplam çekilen bar sayısı
        
        while total_fetched < max_bars:
            # Kalan bar sayısı kadar çek (max 200)
            remaining = max_bars - total_fetched
            fetch_limit = min(self.MAX_CANDLES_PER_REQUEST, remaining)
            
            try:
                # Chunk çek
                df_chunk = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=fetch_limit,
                    since=since_ms
                )
                
                # Boş geldiyse veri sonu
                if df_chunk.empty:
                    if progress:
                        logger.info(f"   ⚠ Veri sonu (toplam: {total_fetched})")
                    break
                
                all_chunks.append(df_chunk)
                total_fetched += len(df_chunk)
                
                # Sonraki chunk: son bar'ın timestamp'i + 1ms (overlap önleme)
                last_ts = df_chunk.index[-1]
                since_ms = int(last_ts.timestamp() * 1000) + 1
                
                if progress and total_fetched % 400 == 0:
                    pct = min(100, 100 * total_fetched / max_bars)
                    logger.info(f"   → {total_fetched}/{max_bars} bar ({pct:.0f}%)")
                
                # Beklenen miktardan az geldiyse daha fazla veri yok
                if len(df_chunk) < fetch_limit:
                    if progress:
                        logger.info(f"   ✓ Veri sonuna ulaşıldı")
                    break
                
                # Güncel zamana ulaştıysak dur
                if last_ts >= datetime.now(timezone.utc) - timedelta(minutes=tf_minutes):
                    if progress:
                        logger.info(f"   ✓ Güncel veriye ulaşıldı")
                    break
                
                # Rate limiting (API limiti aşmamak için)
                time.sleep(self.RATE_LIMIT_DELAY)
                
            except Exception as e:
                logger.error(f"   ⚠ Chunk hatası: {e}")
                time.sleep(1)                 # Hata sonrası uzun bekleme
                break
        
        # Hiç veri gelemediyse hata
        if not all_chunks:
            raise ValueError(f"{symbol} {timeframe} için veri çekilemedi")
        
        # Tüm chunk'ları birleştir
        df_combined = pd.concat(all_chunks)
        
        # Duplicate index'leri kaldır (overlap olabilir)
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        
        # Kronolojik sırala (eski → yeni)
        df_combined = df_combined.sort_index()
        
        # Fazla çekileni kırp (son max_bars kadar tut)
        if len(df_combined) > max_bars:
            df_combined = df_combined.tail(max_bars)
        
        if progress:
            actual_days = (df_combined.index[-1] - df_combined.index[0]).days
            logger.info(
                f"   ✓ {len(df_combined)} bar | {actual_days} gün | "
                f"{df_combined.index[0].strftime('%Y-%m-%d')} → "
                f"{df_combined.index[-1].strftime('%Y-%m-%d')}"
            )
        
        return df_combined
    
    # =========================================================================
    # TÜM TİMEFRAME'LER İÇİN VERİ ÇEKME
    # =========================================================================
    
    def fetch_all_timeframes(
        self,
        symbol: str = None,
        timeframes: Optional[List[str]] = None,
        max_bars_override: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Bir sembol için TÜM aktif timeframe'lerde veri çeker.
        
        Ana analiz döngüsünde her coin için bu fonksiyon çağrılır.
        
        Parametreler:
        ------------
        symbol : str
            İşlem çifti
            
        timeframes : List[str], optional
            Hangi TF'ler çekilsin. None ise config'den okunur.
            
        max_bars_override : int, optional
            Her TF için sabit bar sayısı. None ise RECOMMENDED_BARS kullanılır.
        
        Döndürür:
        --------
        Dict[str, pd.DataFrame]
            Anahtar: timeframe string, Değer: OHLCV DataFrame
            
        Örnek:
        ------
        >>> data = fetcher.fetch_all_timeframes("ETH/USDT:USDT")
        >>> data.keys()
        dict_keys(['5m', '15m', '30m', '1h', '2h', '4h'])
        >>> data['1h'].shape
        (2000, 5)
        """
        symbol = symbol or self.default_symbol
        
        # Timeframe listesini belirle
        if timeframes is None:
            # Config'deki timeframes dict'inden key'leri al
            if cfg.timeframes:
                timeframes = list(cfg.timeframes.keys())
            else:
                timeframes = self.DEFAULT_ACTIVE_TIMEFRAMES
        
        logger.info(f"📥 {symbol} → {len(timeframes)} TF çekiliyor...")
        
        data_dict: Dict[str, pd.DataFrame] = {}
        
        for tf in timeframes:
            try:
                # Bar sayısını belirle: override > config > recommended
                if max_bars_override:
                    bars = max_bars_override
                elif cfg.timeframes and tf in cfg.timeframes:
                    bars = cfg.timeframes[tf].get('bars', self.RECOMMENDED_BARS.get(tf, 1000))
                else:
                    bars = self.RECOMMENDED_BARS.get(tf, 1000)
                
                # Veri çek
                df = self.fetch_max_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    max_bars=bars,
                    progress=True
                )
                
                # Minimum bar kontrolü (çok az veri ile IC anlamsız)
                if len(df) >= 100:
                    data_dict[tf] = df
                    logger.info(f"  {tf}: ✓ {len(df)} bar")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz ({len(df)} < 100)")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
            
            # TF'ler arası bekleme
            time.sleep(0.3)
        
        logger.info(f"📊 {symbol}: {len(data_dict)}/{len(timeframes)} TF başarılı")
        return data_dict
    
    # =========================================================================
    # VERİ DOĞRULAMA
    # =========================================================================
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Çekilen verinin kalitesini doğrular.
        
        Kontroller:
        1. Missing values (eksik veri)
        2. OHLC tutarlılığı (High >= max(Open,Close), Low <= min(Open,Close))
        3. Volume anomalileri (negatif veya sıfır hacim)
        4. Zaman boşlukları (gap tespiti)
        
        Bu kontroller downstream analiz kalitesini doğrudan etkiler:
        - Eksik veri → IC hesaplamasında NaN yayılımı
        - OHLC hatalı → indikatörler yanlış hesaplanır
        - Gap'ler → rolling hesaplamalarda atlamalar
        
        Döndürür:
        --------
        Dict
            Doğrulama sonuçları ve istatistikler
        """
        results = {}
        
        # 1. Temel bilgiler
        results['total_rows'] = len(df)
        results['columns'] = list(df.columns)
        
        # 2. Missing value kontrolü
        missing = df.isnull().sum().to_dict()
        results['missing_values'] = missing
        results['has_missing'] = any(v > 0 for v in missing.values())
        
        # 3. OHLC tutarlılık (her mum için High en yüksek, Low en düşük olmalı)
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            high_ok = (df['high'] >= df['open']) & (df['high'] >= df['close'])
            low_ok = (df['low'] <= df['open']) & (df['low'] <= df['close'])
            invalid = (~high_ok | ~low_ok).sum()
            results['ohlc_invalid'] = int(invalid)
        
        # 4. Volume kontrolü
        if 'volume' in df.columns:
            results['zero_volume'] = int((df['volume'] == 0).sum())
            results['negative_volume'] = int((df['volume'] < 0).sum())
        
        # 5. Zaman aralığı
        if len(df) > 0:
            results['start'] = df.index[0].strftime('%Y-%m-%d %H:%M')
            results['end'] = df.index[-1].strftime('%Y-%m-%d %H:%M')
            results['days'] = (df.index[-1] - df.index[0]).days
        
        # 6. Gap tespiti (beklenen aralıktan fazla boşluk)
        if len(df) > 1:
            diffs = df.index.to_series().diff().dropna()
            if len(diffs) > 0:
                median_diff = diffs.median()
                gaps = (diffs > median_diff * 2).sum()  # 2x median'dan uzun boşluklar
                results['gaps'] = int(gaps)
        
        # Genel geçerlilik
        results['is_valid'] = (
            not results.get('has_missing', True) and
            results.get('ohlc_invalid', 1) == 0 and
            results.get('negative_volume', 1) == 0
        )
        
        return results


# =============================================================================
# TEST KODU
# =============================================================================
if __name__ == "__main__":
    
    # Loglama ayarla (test için)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 65)
    print("  BİTGET FETCHER TEST")
    print("=" * 65)
    
    # Fetcher oluştur
    fetcher = BitgetFetcher()
    
    # Test 1: Market bilgisi
    print("\n[1] Market Bilgisi:")
    info = fetcher.get_market_info("BTC/USDT:USDT")
    for k, v in info.items():
        print(f"   {k}: {v}")
    
    # Test 2: Ticker
    print("\n[2] Güncel Fiyat:")
    ticker = fetcher.get_ticker("BTC/USDT:USDT")
    print(f"   BTC: ${ticker['last']:,.2f} ({ticker['change_24h']:+.2f}%)")
    
    # Test 3: Tek OHLCV
    print("\n[3] Tek OHLCV (1h, 100 bar):")
    df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=100)
    print(f"   {len(df)} bar çekildi")
    print(f"   Son close: ${df['close'].iloc[-1]:,.2f}")
    
    # Test 4: Max OHLCV (pagination)
    print("\n[4] Max OHLCV (1h, 500 bar - pagination):")
    df_max = fetcher.fetch_max_ohlcv("BTC/USDT:USDT", "1h", max_bars=500)
    print(f"   {len(df_max)} bar çekildi")
    
    # Test 5: Veri doğrulama
    print("\n[5] Veri Doğrulama:")
    validation = fetcher.validate_data(df_max)
    print(f"   Geçerli: {validation['is_valid']}")
    print(f"   Gap: {validation.get('gaps', 0)}")
    print(f"   Missing: {validation['has_missing']}")
    
    # Test 6: Tüm USDT Futures çiftleri
    print("\n[6] USDT Futures Çiftleri:")
    pairs = fetcher.get_all_usdt_futures()
    print(f"   Toplam: {len(pairs)} çift")
    print(f"   İlk 5: {pairs[:5]}")
    
    # Test 7: Çoklu TF (küçük miktar - hızlı test)
    print("\n[7] Çoklu TF (ETH, 100 bar):")
    data = fetcher.fetch_all_timeframes(
        "ETH/USDT:USDT",
        timeframes=["15m", "1h", "4h"],
        max_bars_override=100
    )
    for tf, df_tf in data.items():
        print(f"   {tf}: {len(df_tf)} bar | "
              f"${df_tf['close'].iloc[-1]:,.2f}")
    
    print("\n" + "=" * 65)
    print("  TÜM TESTLER TAMAMLANDI ✅")
    print("=" * 65)
