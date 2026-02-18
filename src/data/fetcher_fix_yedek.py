# =============================================================================
# HYBRID DATA FETCHER v3.0 — Binance Data + Bitget Execution
# =============================================================================
# SORUN: Bitget API Türkiye'den boş/eksik ticker verisi döndürüyor.
#        Bu yüzden CoinScanner coin bulamıyor → pipeline çalışmıyor.
#
# ÇÖZÜM:
#   - TÜM VERİ (OHLCV + Ticker + Market listesi) → BINANCE
#   - SADECE EMİR GÖNDERME → BITGET (API key ile)
#   - Bitget market info → lazy load, hata toleranslı
#
# SEMBOL DÖNÜŞÜM KURALI:
#   Pipeline boyunca Bitget formatı kullanılır: 'BTC/USDT:USDT'
#   Binance'e giderken ':USDT' kısmı otomatik temizlenir
#   Örnek: 'BTC/USDT:USDT' → 'BTC/USDT' (Binance için)
#
# Kullanım:
#   from data.fetcher import BitgetFetcher
#   f = BitgetFetcher()
#   
#   # Coin listesi (Binance'den, Bitget formatında)
#   symbols = f.get_all_usdt_futures()  # ['BTC/USDT:USDT', ...]
#   
#   # OHLCV verisi (Binance'den)
#   df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=500)
#   
#   # Anlık fiyat (Binance'den, Bitget fallback)
#   ticker = f.get_ticker('BTC/USDT:USDT')
#   
#   # Toplu ticker (Binance'den)
#   tickers = f.fetch_tickers()  # CoinScanner bunu kullanır
# =============================================================================

import ccxt                                    # Borsa unified API kütüphanesi
import pandas as pd                            # DataFrame veri yapısı
import time                                    # Rate limiting / zamanlama
import logging                                 # Log yönetimi
import sys                                     # Path ayarları
from datetime import datetime, timezone        # UTC zaman damgaları
from typing import Optional, List, Dict        # Tip belirteçleri
from pathlib import Path                       # Platform-bağımsız dosya yolları

# ─── Proje config'i import et ───
sys.path.insert(0, str(Path(__file__).parent.parent))  # → src/
from config import cfg                         # Merkezi yapılandırma nesnesi

# ─── Logger ───
logger = logging.getLogger(__name__)           # Bu modülün logger'ı


class BitgetFetcher:
    """
    Hybrid veri çekici: Binance'den analiz verisi, Bitget'ten execution bilgisi.
    
    v3.0 değişiklikler:
    - get_all_usdt_futures() artık Binance'den listeliyor
    - fetch_tickers() Binance'den çekip Bitget formatına dönüştürüyor
    - get_ticker() önce Binance, fallback olarak Bitget
    - Bitget bağlantı hatası tüm sistemi durdurmaz
    """
    
    # =========================================================================
    # SABİTLER
    # =========================================================================
    
    # Timeframe → dakika dönüşüm tablosu (IC hesaplaması için)
    TIMEFRAME_MINUTES: Dict[str, int] = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    
    # Binance tek istekte max 1500 mum verebilir (biz güvenli tarafta 1000)
    MAX_CANDLES_PER_REQUEST: int = 1000
    
    # Timeframe başına önerilen bar sayıları (IC analiz kalitesi için)
    # Daha fazla bar = daha güvenilir IC, ama çok eski veri = regime change riski
    RECOMMENDED_BARS: Dict[str, int] = {
        "5m":  3000,                           # ~10 gün — scalping
        "15m": 2000,                           # ~20 gün — day trading
        "30m": 1500,                           # ~30 gün — kısa swing
        "1h":  1000,                           # ~40 gün — intraday
        "4h":  500,                            # ~80 gün — swing
        "1d":  365,                            # ~1 yıl  — position
    }
    
    # Varsayılan aktif timeframe'ler (config'de yoksa bunlar kullanılır)
    DEFAULT_ACTIVE_TIMEFRAMES: List[str] = ["15m", "1h", "4h"]
    
    # Stablecoin ve sorunlu coinleri filtrele (Binance listesinden)
    EXCLUDE_KEYWORDS: List[str] = [
        'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD',  # Stablecoinler
        'UP', 'DOWN', 'BULL', 'BEAR',              # Leveraged tokenlar
        '3L', '3S', '5L', '5S',                    # Leveraged varyantları
        'BTTC',                                     # Bilinen sorunlu coin
    ]
    
    # =========================================================================
    # CONSTRUCTOR
    # =========================================================================
    
    def __init__(self, symbol: str = None):
        """
        Fetcher'ı başlat: Binance (veri) + Bitget (execution) bağlantıları.
        
        Parametreler:
        ------------
        symbol : str, optional
            Varsayılan işlem sembolü (örn: "BTC/USDT:USDT")
            None ise config'deki default_symbol kullanılır
        """
        # Varsayılan sembol (pipeline genelinde kullanılır)
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # ─── 1. BINANCE — Ana Veri Kaynağı ───
        # Public API: OHLCV, ticker, market listesi — API key gerekmez
        # Neden Binance? Daha kaliteli veri, daha az boşluk, daha hızlı
        self.binance = ccxt.binance({
            'options': {'defaultType': 'future'},  # USDT-M Futures modu
            'enableRateLimit': True,               # Otomatik rate limit koruması
            'timeout': 15000,                      # 15s timeout (yavaş bağlantı için)
        })
        
        # ─── 2. BITGET — Sadece Execution ───
        # API key gerekli: emir gönderme, bakiye sorgulama
        # Market info: contract size, precision, lot size
        self.exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},    # USDT-M Futures (swap)
            'enableRateLimit': True,
            'timeout': 15000,
        })
        
        # ─── Durum Flagleri ───
        self._binance_markets_loaded = False    # Binance marketleri yüklendi mi?
        self._bitget_markets_loaded = False     # Bitget marketleri yüklendi mi?
        self._bitget_available = True           # Bitget erişilebilir mi?
        
        # ─── Cache ───
        self._binance_symbols_cache: List[str] = []   # Binance USDT Futures listesi
        self._symbol_map_cache: Dict[str, str] = {}   # Bitget→Binance sembol eşlemesi
        
        logger.info(
            f"BitgetFetcher v3.0 başlatıldı | "
            f"Veri: Binance | Execution: Bitget | "
            f"Sembol: {self.default_symbol}"
        )
    
    # =========================================================================
    # MARKET BİLGİSİ — BINANCE (Ana Liste)
    # =========================================================================
    
    def _ensure_binance_markets_loaded(self):
        """
        Binance market bilgilerini lazy-load eder (ilk çağrıda bir kez).
        
        Neden Binance'den?
        → Bitget boş veri döndürebiliyor (IP kısıtlaması vs.)
        → Binance daha güvenilir ve hızlı
        → Sembol dönüşümü basit: 'BTC/USDT' → 'BTC/USDT:USDT'
        """
        if self._binance_markets_loaded:
            return                             # Zaten yüklü, tekrar yükleme
        
        try:
            logger.info("📡 Binance Futures market bilgileri yükleniyor...")
            start = time.time()
            self.binance.load_markets()
            elapsed = time.time() - start
            self._binance_markets_loaded = True
            
            # USDT-M Futures çiftlerini cache'le
            # Binance formatı: 'BTC/USDT' (type='future')
            self._binance_symbols_cache = sorted([
                s for s, m in self.binance.markets.items()
                if s.endswith('/USDT')                     # USDT çifti
                and m.get('type') == 'swap'                # Futures (perpetual)
                and m.get('active', True)                  # Aktif market
            ])
            
            # Sembol dönüşüm haritası: Binance→Bitget formatı
            # 'BTC/USDT' → 'BTC/USDT:USDT'
            self._symbol_map_cache = {
                s: f"{s}:USDT" for s in self._binance_symbols_cache
            }
            
            logger.info(
                f"✅ Binance: {len(self._binance_symbols_cache)} USDT-M Futures "
                f"çifti yüklendi ({elapsed:.1f}s)"
            )
            
        except Exception as e:
            logger.error(f"❌ Binance market yükleme hatası: {e}")
            # Boş liste döndür ama crash etme
            self._binance_symbols_cache = []
    
    def _ensure_bitget_markets_loaded(self):
        """
        Bitget market bilgilerini lazy-load eder.
        
        Bu bilgi SADECE emir gönderirken gerekli:
        - contract_size, precision, limits
        - Veri çekmek için gerekmez
        
        Bitget'e erişim yoksa sistemi durdurmaz.
        """
        if self._bitget_markets_loaded:
            return
        
        try:
            logger.info("📡 Bitget market bilgileri yükleniyor (execution için)...")
            self.exchange.load_markets()
            self._bitget_markets_loaded = True
            
            usdt_count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
            logger.info(f"✅ Bitget: {usdt_count} USDT-M çifti yüklendi")
            
        except Exception as e:
            logger.warning(
                f"⚠️ Bitget market yüklenemedi: {e}\n"
                f"   Emir göndermek için Bitget gerekli olacak!\n"
                f"   Veri çekme etkilenmez (Binance kullanılıyor)."
            )
            self._bitget_available = False
    
    # =========================================================================
    # SEMBOL DÖNÜŞÜM
    # =========================================================================
    
    def _to_binance_symbol(self, symbol: str) -> str:
        """
        Bitget sembolünü Binance formatına dönüştürür.
        
        Örnek: 'BTC/USDT:USDT' → 'BTC/USDT'
        Eğer zaten Binance formatındaysa dokunmaz.
        """
        return symbol.split(':')[0]            # ':USDT' kısmını çıkar
    
    def _to_bitget_symbol(self, symbol: str) -> str:
        """
        Binance sembolünü Bitget formatına dönüştürür.
        
        Örnek: 'BTC/USDT' → 'BTC/USDT:USDT'
        Eğer zaten Bitget formatındaysa dokunmaz.
        """
        if ':' not in symbol:
            return f"{symbol}:USDT"            # ':USDT' ekle
        return symbol                          # Zaten doğru format
    
    # =========================================================================
    # COİN LİSTESİ — BINANCE'DEN (ESKİDEN BİTGET'TENDİ)
    # =========================================================================
    
    def get_all_usdt_futures(self) -> List[str]:
        """
        TÜM USDT-M Futures çiftlerini Binance'den çekip Bitget formatında döndürür.
        
        ESKİ DAVRANIŞI: Bitget'ten çekiyordu → Boş liste geliyordu
        YENİ DAVRANIŞI: Binance'den çeker → Bitget formatına dönüştürür
        
        CoinScanner bu listeyi kullanıyor:
          all_symbols = self.fetcher.get_all_usdt_futures()
        
        Döndürür:
        --------
        List[str]
            Bitget formatında semboller: ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]
        """
        self._ensure_binance_markets_loaded()
        
        # Binance sembollerini Bitget formatına dönüştür
        # 'BTC/USDT' → 'BTC/USDT:USDT'
        bitget_format_symbols = [
            self._to_bitget_symbol(s) for s in self._binance_symbols_cache
        ]
        
        # Stablecoin ve sorunlu coinleri filtrele
        filtered = []
        for sym in bitget_format_symbols:
            coin = sym.split('/')[0].upper()   # 'BTC/USDT:USDT' → 'BTC'
            
            # Blacklist kontrolü
            if not any(kw in coin for kw in self.EXCLUDE_KEYWORDS):
                filtered.append(sym)
        
        logger.info(
            f"📋 {len(filtered)} USDT-M çifti listelendi "
            f"(Binance kaynaklı, Bitget formatında)"
        )
        
        return sorted(filtered)
    
    # =========================================================================
    # TOPLU TİCKER — BINANCE'DEN (CoinScanner BUNU KULLANIYOR)
    # =========================================================================
    
    def fetch_tickers(self, symbols: List[str] = None) -> Dict:
        """
        Toplu ticker verisi çeker — Binance'den, Bitget formatıyla döndürür.
        
        CoinScanner._fetch_all_tickers() bu fonksiyonu dolaylı kullanıyor:
          all_tickers = self.fetcher.exchange.fetch_tickers()
        
        ARTIK:
          all_tickers = self.fetcher.fetch_tickers()
        
        Parametreler:
        ------------
        symbols : List[str], optional
            İstenen semboller (Bitget formatı). None ise tümü döner.
        
        Döndürür:
        --------
        Dict
            {Bitget_sembol: ticker_data} formatında
            Örn: {'BTC/USDT:USDT': {'last': 96000, 'quoteVolume': 5e9, ...}}
        """
        try:
            # 1. Binance'den tüm ticker'ları çek (tek API çağrısı)
            raw_tickers = self.binance.fetch_tickers()
            
            # 2. Bitget formatına dönüştür
            # 'BTC/USDT' → 'BTC/USDT:USDT' key olarak
            converted = {}
            for bn_sym, ticker_data in raw_tickers.items():
                # Sadece USDT Futures çiftlerini al
                if bn_sym.endswith('/USDT') and ':' not in bn_sym:
                    bg_sym = self._to_bitget_symbol(bn_sym)  # 'BTC/USDT:USDT'
                    converted[bg_sym] = ticker_data
            
            # 3. Eğer belirli semboller istendiyse filtrele
            if symbols:
                converted = {s: converted[s] for s in symbols if s in converted}
            
            logger.info(f"📊 {len(converted)} ticker çekildi (Binance)")
            return converted
            
        except Exception as e:
            logger.error(f"❌ Binance ticker hatası: {e}")
            
            # Fallback: Bitget'i dene
            try:
                logger.info("↩️ Bitget ticker'a fallback ediliyor...")
                return self.exchange.fetch_tickers()
            except Exception as e2:
                logger.error(f"❌ Bitget fallback da başarısız: {e2}")
                return {}
    
    # =========================================================================
    # TEK TİCKER — ANLIK FİYAT
    # =========================================================================
    
    def get_ticker(self, symbol: str = None) -> Dict:
        """
        Tek sembolün anlık fiyat bilgisini çeker.
        
        Öncelik: Binance → Bitget (fallback)
        
        NOT: İşlemi Bitget fiyatıyla açacağız ama Bitget erişim sorunu varsa
        Binance fiyatı da yeterli (fark genelde <0.01%).
        
        Parametreler:
        ------------
        symbol : str
            Bitget formatında sembol (örn: 'BTC/USDT:USDT')
        
        Döndürür:
        --------
        Dict
            last, bid, ask, volume_24h, percentage alanları
        """
        symbol = symbol or self.default_symbol
        bn_symbol = self._to_binance_symbol(symbol)  # 'BTC/USDT'
        
        try:
            # Binance'den ticker al
            ticker = self.binance.fetch_ticker(bn_symbol)
            
            return {
                'last':       ticker.get('last', 0) or 0,
                'bid':        ticker.get('bid', 0) or 0,
                'ask':        ticker.get('ask', 0) or 0,
                'volume_24h': ticker.get('quoteVolume', 0) or 0,
                'percentage': ticker.get('percentage', 0) or 0,
            }
            
        except Exception as e:
            logger.warning(f"Binance ticker hatası ({bn_symbol}): {e}")
            
            # Fallback: Bitget'i dene
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return {
                    'last':       ticker.get('last', 0) or 0,
                    'bid':        ticker.get('bid', 0) or 0,
                    'ask':        ticker.get('ask', 0) or 0,
                    'volume_24h': ticker.get('quoteVolume', 0) or 0,
                    'percentage': ticker.get('percentage', 0) or 0,
                }
            except Exception as e2:
                logger.error(f"Bitget ticker da başarısız ({symbol}): {e2}")
                return {
                    'last': 0, 'bid': 0, 'ask': 0, 
                    'volume_24h': 0, 'percentage': 0,
                }
    
    # =========================================================================
    # MARKET BİLGİSİ — BITGET (Execution için gerekli)
    # =========================================================================
    
    def get_market_info(self, symbol: str = None) -> Dict:
        """
        Emir göndermek için gereken market bilgisini Bitget'ten alır.
        
        Bu bilgi SADECE canlı emir gönderirken gerekli:
        - contract_size: Kontrat büyüklüğü
        - precision: Fiyat/miktar hassasiyeti
        - limits: Min/max sipariş miktarları
        - max_leverage: Maksimum kaldıraç
        
        Parametreler:
        ------------
        symbol : str
            Bitget formatında sembol (örn: 'BTC/USDT:USDT')
        """
        symbol = symbol or self.default_symbol
        self._ensure_bitget_markets_loaded()
        
        if not self._bitget_available:
            # Bitget erişim yoksa varsayılan değerler döndür
            logger.warning(f"Bitget erişimi yok, varsayılan market info döndürülüyor")
            return {
                'symbol': symbol,
                'contract_size': 1,
                'precision': {'price': 0.01, 'amount': 0.001},
                'limits': {'min_amount': 0, 'min_cost': 5, 'max_amount': None},
                'max_leverage': 20,
            }
        
        if symbol not in self.exchange.markets:
            raise ValueError(f"'{symbol}' Bitget Futures'da bulunamadı")
        
        market = self.exchange.markets[symbol]
        return {
            'symbol': symbol,
            'contract_size': market.get('contractSize', 1),
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
    
    # =========================================================================
    # OHLCV VERİSİ — BINANCE
    # =========================================================================
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 200,
        since=None
    ) -> pd.DataFrame:
        """
        OHLCV (mum) verisi çeker — Binance'den.
        
        Girdi sembolü Bitget formatında olabilir, otomatik dönüştürülür.
        Örnek: 'BTC/USDT:USDT' → Binance'de 'BTC/USDT' olarak çekilir
        
        Parametreler:
        ------------
        symbol : str
            Bitget veya Binance formatında sembol
        timeframe : str
            Mum zaman dilimi ('1m', '5m', '15m', '1h', '4h', '1d')
        limit : int
            Çekilecek mum sayısı (max 1000)
        since : int, optional
            Başlangıç timestamp (milisaniye). None ise en son mumlardan çeker.
        
        Döndürür:
        --------
        pd.DataFrame
            Sütunlar: open, high, low, close, volume
            Index: datetime (UTC, timezone-aware)
            Boş ise pd.DataFrame() döner
        """
        # 1. Sembol Dönüşümü: 'BTC/USDT:USDT' → 'BTC/USDT'
        clean_symbol = self._to_binance_symbol(symbol)
        
        # 2. Limit kontrolü (Binance max 1500, biz 1000 ile sınırlıyoruz)
        req_limit = min(limit, self.MAX_CANDLES_PER_REQUEST)
        
        try:
            # 3. Binance'den OHLCV çek
            ohlcv = self.binance.fetch_ohlcv(
                clean_symbol, timeframe, limit=req_limit, since=since
            )
            
            # 4. Boş kontrol
            if not ohlcv:
                logger.debug(f"{clean_symbol} {timeframe}: Binance boş veri döndü")
                return pd.DataFrame()
            
            # 5. DataFrame oluştur
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 6. Timestamp dönüşümü: Unix ms → datetime UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = None               # Temiz index adı
            
            # 7. Veri tipi optimizasyonu (tutarlılık için float64)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype('float64')
            
            return df
            
        except Exception as e:
            logger.debug(f"Binance OHLCV hatası ({clean_symbol} {timeframe}): {e}")
            return pd.DataFrame()
    
    # =========================================================================
    # MAKSİMUM VERİ ÇEKME (PAGİNATION)
    # =========================================================================
    
    def fetch_max_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        max_bars: Optional[int] = None,
        progress: bool = False
    ) -> pd.DataFrame:
        """
        Geriye dönük geniş veri seti çeker — Binance üzerinden.
        
        Binance tek istekte 1000 mum verebildiği için çoğu durumda
        pagination gerekmez. Ama max_bars > 1000 ise pagination yapar.
        
        Parametreler:
        ------------
        symbol : str
            İşlem çifti (Bitget formatı)
        timeframe : str
            Zaman dilimi
        max_bars : int, optional
            Hedef mum sayısı. None ise RECOMMENDED_BARS kullanılır.
        progress : bool
            True ise ilerleme loglanır
        """
        symbol = symbol or self.default_symbol
        
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 1000)
        
        # Binance tek seferde 1000 mum verebilir
        if max_bars <= self.MAX_CANDLES_PER_REQUEST:
            # Tek çağrı yeterli
            return self.fetch_ohlcv(symbol, timeframe, limit=max_bars)
        
        # Pagination gerekli (max_bars > 1000)
        from datetime import timedelta
        
        tf_minutes = self.TIMEFRAME_MINUTES.get(timeframe, 60)
        
        # Geriye doğru başlangıç noktasını hesapla (buffer ile)
        buffer_factor = 1.3                    # %30 buffer (hafta sonu boşlukları için)
        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=int(max_bars * tf_minutes * buffer_factor)
        )
        since_ms = int(start_time.timestamp() * 1000)
        
        all_chunks: List[pd.DataFrame] = []
        total_fetched = 0
        
        while total_fetched < max_bars:
            remaining = max_bars - total_fetched
            fetch_limit = min(self.MAX_CANDLES_PER_REQUEST, remaining)
            
            try:
                df_chunk = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=fetch_limit,
                    since=since_ms
                )
                
                if df_chunk.empty:
                    if progress:
                        logger.info(f"  ⚠ Veri sonu (toplam: {total_fetched})")
                    break
                
                all_chunks.append(df_chunk)
                total_fetched += len(df_chunk)
                
                # Sonraki chunk: son mumun timestamp'i + 1ms
                since_ms = int(df_chunk.index[-1].timestamp() * 1000) + 1
                
                if progress and total_fetched % 500 == 0:
                    pct = min(100, 100 * total_fetched / max_bars)
                    logger.info(f"  → {total_fetched}/{max_bars} bar ({pct:.0f}%)")
                
                # Beklenen miktardan az geldiyse veri bitmiş
                if len(df_chunk) < fetch_limit:
                    break
                
                time.sleep(0.1)                # Rate limit koruması
                
            except Exception as e:
                logger.warning(f"  ⚠ Pagination chunk hatası: {e}")
                break
        
        if not all_chunks:
            return pd.DataFrame()
        
        # Chunk'ları birleştir
        df_combined = pd.concat(all_chunks)
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()
        
        # Fazlayı kırp
        if len(df_combined) > max_bars:
            df_combined = df_combined.tail(max_bars)
        
        if progress:
            actual_days = (df_combined.index[-1] - df_combined.index[0]).days
            logger.info(
                f"  ✓ {len(df_combined)} bar | {actual_days} gün | "
                f"{df_combined.index[0].strftime('%Y-%m-%d')} → "
                f"{df_combined.index[-1].strftime('%Y-%m-%d')}"
            )
        
        return df_combined
    
    # =========================================================================
    # ÇOKLU TİMEFRAME VERİ ÇEKME
    # =========================================================================
    
    def fetch_all_timeframes(
        self,
        symbol: str = None,
        timeframes: Optional[List[str]] = None,
        max_bars_override: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Bir sembol için TÜM aktif timeframe'lerde veri çeker.
        
        Her coin analizi için bu fonksiyon çağrılır.
        
        Parametreler:
        ------------
        symbol : str
            Bitget formatında sembol
        timeframes : List[str], optional
            Hangi TF'ler çekilsin. None ise config'den okunur.
        max_bars_override : int, optional
            Her TF için sabit bar sayısı.
        
        Döndürür:
        --------
        Dict[str, pd.DataFrame]
            Anahtar: timeframe string, Değer: OHLCV DataFrame
        """
        symbol = symbol or self.default_symbol
        
        # Timeframe listesini belirle (öncelik: parametre → config → varsayılan)
        if timeframes is None:
            if cfg.timeframes:
                timeframes = list(cfg.timeframes.keys())
            else:
                timeframes = self.DEFAULT_ACTIVE_TIMEFRAMES
        
        logger.info(f"📥 {symbol} → {len(timeframes)} TF çekiliyor (Binance)...")
        
        data_dict: Dict[str, pd.DataFrame] = {}
        
        for tf in timeframes:
            try:
                # Bar sayısı: override > config > recommended
                if max_bars_override:
                    bars = max_bars_override
                elif cfg.timeframes and tf in cfg.timeframes:
                    bars = cfg.timeframes[tf].get('bars', self.RECOMMENDED_BARS.get(tf, 500))
                else:
                    bars = self.RECOMMENDED_BARS.get(tf, 500)
                
                # Veri çek (Binance'den)
                df = self.fetch_max_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    max_bars=bars,
                    progress=False
                )
                
                # Minimum bar kontrolü (IC hesabı için en az 100 bar gerekli)
                if len(df) >= 100:
                    data_dict[tf] = df
                    logger.debug(f"  {tf}: ✓ {len(df)} bar")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz ({len(df)} < 100)")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata - {e}")
            
            time.sleep(0.1)                    # Rate limit koruması
        
        logger.info(
            f"📊 {self._to_binance_symbol(symbol)}: "
            f"{len(data_dict)}/{len(timeframes)} TF başarılı"
        )
        return data_dict
    
    # =========================================================================
    # VERİ DOĞRULAMA
    # =========================================================================
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Çekilen verinin kalitesini kontrol eder.
        
        Kontroller:
        1. Boş mu?
        2. Eksik değer (NaN) var mı?
        3. Son veri ne kadar güncel?
        """
        if df.empty:
            return {'is_valid': False, 'rows': 0, 'last_date': None}
        
        return {
            'is_valid': df.isnull().sum().sum() == 0,  # NaN yoksa geçerli
            'rows': len(df),
            'last_date': df.index[-1] if not df.empty else None,
            'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
        }


# =============================================================================
# BAĞIMSIZ TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("\n" + "="*60)
    print("  🔬 BitgetFetcher v3.0 TEST")
    print("="*60)
    
    f = BitgetFetcher()
    
    # Test 1: Coin listesi
    print("\n[1] Coin listesi (Binance'den):")
    symbols = f.get_all_usdt_futures()
    print(f"  {len(symbols)} çift bulundu")
    print(f"  İlk 5: {symbols[:5]}")
    
    # Test 2: Toplu ticker
    print("\n[2] Toplu ticker (Binance'den):")
    tickers = f.fetch_tickers()
    print(f"  {len(tickers)} ticker çekildi")
    
    btc = tickers.get('BTC/USDT:USDT', {})
    if btc:
        print(f"  BTC: ${btc.get('last', 0):,.2f} | Vol: ${btc.get('quoteVolume', 0):,.0f}")
    
    # Test 3: Tek ticker
    print("\n[3] Tek ticker:")
    t = f.get_ticker('BTC/USDT:USDT')
    print(f"  BTC: ${t['last']:,.2f} | Bid: ${t['bid']:,.2f} | Ask: ${t['ask']:,.2f}")
    
    # Test 4: OHLCV
    print("\n[4] OHLCV verisi:")
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=100)
    print(f"  {len(df)} bar çekildi")
    if not df.empty:
        print(f"  Son: ${df['close'].iloc[-1]:,.2f} @ {df.index[-1]}")
    
    # Test 5: Çoklu TF
    print("\n[5] Çoklu timeframe:")
    data = f.fetch_all_timeframes('ETH/USDT:USDT', timeframes=['15m', '1h', '4h'])
    for tf, d in data.items():
        print(f"  {tf}: {len(d)} bar")
    
    print("\n✅ Tüm testler başarılı!\n")
