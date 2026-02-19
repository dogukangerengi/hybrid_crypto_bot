#!/usr/bin/env python3
# =============================================================================
# KESİN ÇÖZÜM — TÜM VERİ AKIŞI BİTGET'TEN
# =============================================================================
# Bu script:
# 1. fetcher.py'yi sıfırdan yazar (tamamen Bitget, pagination destekli)
# 2. coin_scanner.py uyumluluğunu doğrular
# 3. main.py'deki OHLCV/ticker çağrılarını kontrol eder
# 4. Canlı doğrulama testleri çalıştırır
#
# Çalıştır: cd ~/hybrid_crypto_bot && python final_fix.py
# =============================================================================

import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

# ─── Renk kodları ───
G = "\033[92m"    # Yeşil
R = "\033[91m"    # Kırmızı
Y = "\033[93m"    # Sarı
C = "\033[96m"    # Cyan
B = "\033[1m"     # Bold
X = "\033[0m"     # Reset

def ok(m):   print(f"  {G}✅ {m}{X}")
def fail(m): print(f"  {R}❌ {m}{X}")
def warn(m): print(f"  {Y}⚠️  {m}{X}")
def info(m): print(f"  {C}ℹ️  {m}{X}")

# ─── src/ dizinini bul ───
cwd = Path.cwd()
src = cwd / 'src' if (cwd / 'src').exists() else (cwd if cwd.name == 'src' else None)
if not src:
    fail("src/ klasörü bulunamadı! cd ~/hybrid_crypto_bot yapıp tekrar dene.")
    sys.exit(1)

print(f"\n{B}{'='*60}")
print(f"  🔧 KESİN ÇÖZÜM — TAMAMEN BİTGET MİMARİSİ")
print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}{X}")


# =============================================================================
# ADIM 1: YEDEK AL
# =============================================================================
print(f"\n{B}[1/4] Yedekleniyor...{X}")

backup_dir = cwd / 'backups' / f"pre_final_fix_{datetime.now().strftime('%H%M%S')}"
backup_dir.mkdir(parents=True, exist_ok=True)

files_to_backup = [
    src / 'data' / 'fetcher.py',
    src / 'main.py',
    src / 'scanner' / 'coin_scanner.py',
]

for f in files_to_backup:
    if f.exists():
        dest = backup_dir / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

ok(f"Yedek alındı → {backup_dir.relative_to(cwd)}/")


# =============================================================================
# ADIM 2: fetcher.py — TAMAMEN BİTGET
# =============================================================================
print(f"\n{B}[2/4] fetcher.py yazılıyor (tamamen Bitget)...{X}")

FETCHER_CODE = r'''# =============================================================================
# BİTGET FUTURES VERİ ÇEKME MODÜLÜ v4.0 — TAM BİTGET
# =============================================================================
# Tüm veri Bitget'ten geliyor:
# - Coin listesi      → Bitget USDT-M Futures markets
# - Ticker (fiyat)    → Bitget fetch_ticker / fetch_tickers
# - OHLCV (mum veri)  → Bitget fetch_ohlcv + pagination
# - Market bilgisi    → Bitget markets (contract size, precision vb.)
#
# Binance bağımlılığı KALDIRILDI. Tek exchange = daha az hata noktası.
#
# Bitget OHLCV limiti: 200 mum/istek → pagination ile 1000+ mum çekebiliriz
# =============================================================================

import ccxt                                    # Borsa unified API'si
import pandas as pd                            # Veri yapıları
import time                                    # Rate limiting
import logging                                 # Log yönetimi
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path

# Config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

logger = logging.getLogger(__name__)


class BitgetFetcher:
    """
    Bitget USDT-M Perpetual Futures veri çekme sınıfı.
    
    Tüm veri tek borsadan (Bitget) gelir:
    - get_all_usdt_futures()  → coin listesi
    - fetch_tickers()         → toplu ticker (tek API çağrısı)
    - get_ticker()            → tek coin fiyatı
    - fetch_ohlcv()           → OHLCV mum verisi (pagination destekli)
    - fetch_max_ohlcv()       → büyük veri seti (otomatik pagination)
    - fetch_all_timeframes()  → çoklu TF verisi
    - get_market_info()       → contract size, precision, leverage
    """
    
    # =========================================================================
    # SABİTLER
    # =========================================================================
    
    # Timeframe → dakika eşleştirmesi (pagination hesabı için)
    TIMEFRAME_MINUTES: Dict[str, int] = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    
    # Bitget tek istekte max 200 mum verir
    MAX_CANDLES_PER_REQUEST = 200
    
    # IC analizi için önerilen minimum bar sayıları
    RECOMMENDED_BARS: Dict[str, int] = {
        "5m": 2000,    # ~7 gün     | IC güvenilir olması için yeterli sample
        "15m": 1500,   # ~15 gün    | Trend + momentum analizi
        "30m": 1000,   # ~20 gün    | Swing analizi
        "1h": 750,     # ~31 gün    | Günlük döngü analizi
        "4h": 500,     # ~83 gün    | Haftalık trend
        "1d": 365,     # ~1 yıl     | Uzun vadeli rejim
    }
    
    # Varsayılan aktif timeframe'ler
    DEFAULT_ACTIVE_TIMEFRAMES = ["15m", "1h", "4h"]

    # =========================================================================
    # BAŞLATMA (CONSTRUCTOR)
    # =========================================================================
    
    def __init__(self, symbol: str = None):
        """
        BitgetFetcher başlat.
        
        Parameters:
        ----------
        symbol : str, optional
            Varsayılan sembol. None ise config'den okunur.
        """
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # CCXT Bitget bağlantısı — USDT-M Perpetual Futures
        self.exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},  # swap = USDT-M Futures
            'enableRateLimit': True,              # Otomatik rate limiting
        })
        
        self._markets_loaded = False             # Lazy loading flag

    # =========================================================================
    # MARKET BİLGİSİ
    # =========================================================================
    
    def _ensure_markets_loaded(self):
        """Bitget market bilgilerini lazy-load et (ilk çağrıda yüklenir)."""
        if not self._markets_loaded:
            try:
                logger.info("Bitget market bilgileri yükleniyor...")
                self.exchange.load_markets()
                self._markets_loaded = True
                
                # İstatistik log
                count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
                logger.info(f"✓ {count} Bitget USDT-M çifti aktif")
            except Exception as e:
                logger.error(f"Market yükleme hatası: {e}")
                raise

    def get_all_usdt_futures(self) -> List[str]:
        """
        Bitget'teki TÜM USDT-M Perpetual Futures çiftlerini döndürür.
        
        CoinScanner bu listeyi alıp hacim/spread filtresi uygular.
        
        Returns:
        -------
        List[str]
            Sembol listesi — örn: ["BTC/USDT:USDT", "ETH/USDT:USDT", ...]
        """
        self._ensure_markets_loaded()
        return sorted([s for s in self.exchange.markets if s.endswith(':USDT')])

    def get_market_info(self, symbol: str = None) -> Dict:
        """
        Emir gönderirken gereken market kurallarını döndürür.
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü (örn: 'BTC/USDT:USDT')
            
        Returns:
        -------
        Dict
            contract_size, precision, limits, max_leverage bilgileri
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        if symbol not in self.exchange.markets:
            raise ValueError(f"{symbol} Bitget'te bulunamadı!")
        
        market = self.exchange.markets[symbol]
        return {
            'symbol': symbol,
            'type': market.get('type', 'unknown'),
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
    # TICKER (FİYAT BİLGİSİ)
    # =========================================================================
    
    def get_ticker(self, symbol: str = None) -> Dict:
        """
        Tek coin için anlık fiyat bilgisi (Bitget).
        
        İşlem açarken Bitget fiyatı kullanılmalı (execution price).
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü (örn: 'BTC/USDT:USDT')
            
        Returns:
        -------
        Dict
            last, bid, ask, volume_24h, quoteVolume, percentage, high_24h, low_24h
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last': ticker.get('last', 0),               # Son işlem fiyatı
                'bid': ticker.get('bid', 0),                  # En iyi alış
                'ask': ticker.get('ask', 0),                  # En iyi satış
                'spread': (ticker.get('ask', 0) or 0) - (ticker.get('bid', 0) or 0),
                'high_24h': ticker.get('high', 0),            # 24s en yüksek
                'low_24h': ticker.get('low', 0),              # 24s en düşük
                'volume_24h': ticker.get('quoteVolume', 0),   # 24s USDT hacim
                'quoteVolume': ticker.get('quoteVolume', 0),  # Alias (geriye uyumluluk)
                'percentage': ticker.get('percentage', 0),     # 24s % değişim
                'change_24h': ticker.get('percentage', 0),     # Alias
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            }
        except ccxt.NetworkError as e:
            raise ConnectionError(f"Ağ hatası (ticker): {e}")
        except ccxt.ExchangeError as e:
            raise ValueError(f"Borsa hatası (ticker): {e}")

    def fetch_tickers(self, symbols: List[str] = None) -> Dict:
        """
        Toplu ticker verisi — tek API çağrısı ile tüm marketleri çeker.
        
        CoinScanner._fetch_all_tickers() bu metodu çağırır.
        
        Parameters:
        ----------
        symbols : List[str], optional
            Filtrelenecek semboller. None ise tüm USDT-M ticker'lar döner.
            
        Returns:
        -------
        Dict
            {symbol: ticker_data} formatında
        """
        all_tickers = self.exchange.fetch_tickers()
        
        if symbols:
            return {s: all_tickers[s] for s in symbols if s in all_tickers}
        
        # symbols verilmediyse sadece USDT-M olanları döndür
        return {k: v for k, v in all_tickers.items() if k.endswith(':USDT')}

    # =========================================================================
    # OHLCV VERİ ÇEKME — PAGİNATİON DESTEKLİ
    # =========================================================================
    
    def fetch_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        limit: int = 200,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Bitget'ten OHLCV (mum) verisi çeker.
        
        200'den fazla mum istenirse otomatik pagination yapar.
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü (örn: 'BTC/USDT:USDT')
        timeframe : str
            Zaman dilimi (1m, 5m, 15m, 1h, 4h, 1d vb.)
        limit : int
            İstenen mum sayısı. >200 ise pagination yapılır.
        since : int, optional
            Başlangıç timestamp (ms). None ise en son mumlardan geriye.
            
        Returns:
        -------
        pd.DataFrame
            Index: timestamp (UTC), Columns: open, high, low, close, volume
        """
        symbol = symbol or self.default_symbol
        
        try:
            # 200'den az isteniyorsa tek istek yeterli
            if limit <= self.MAX_CANDLES_PER_REQUEST:
                return self._fetch_ohlcv_single(symbol, timeframe, limit, since)
            
            # 200'den fazla → pagination
            return self._fetch_ohlcv_paginated(symbol, timeframe, limit)
            
        except Exception as e:
            logger.warning(f"OHLCV hatası ({symbol} {timeframe}): {e}")
            return pd.DataFrame()

    def _fetch_ohlcv_single(
        self, symbol: str, timeframe: str, limit: int, since=None
    ) -> pd.DataFrame:
        """Tek istekte OHLCV çeker (≤200 mum)."""
        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe, limit=min(limit, self.MAX_CANDLES_PER_REQUEST), since=since
        )
        return self._ohlcv_to_dataframe(ohlcv)

    def _fetch_ohlcv_paginated(
        self, symbol: str, timeframe: str, total_limit: int
    ) -> pd.DataFrame:
        """
        Pagination ile büyük OHLCV verisi çeker.
        
        Strateji: En eski mumdan başla, ileriye doğru git.
        Her istekte 200 mum çek, timestamp'i ilerlet.
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü
        timeframe : str
            Zaman dilimi
        total_limit : int
            Toplam istenen mum sayısı
        """
        # Geriye dönük başlangıç zamanını hesapla
        tf_minutes = self.TIMEFRAME_MINUTES.get(timeframe, 60)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (total_limit * tf_minutes * 60 * 1000)
        
        all_data = []          # Tüm mumlar buraya toplanır
        current_since = start_ms
        remaining = total_limit
        max_retries = 3        # API hatası durumunda tekrar deneme
        
        while remaining > 0:
            batch_size = min(remaining, self.MAX_CANDLES_PER_REQUEST)
            
            for retry in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, timeframe,
                        limit=batch_size,
                        since=current_since,
                    )
                    break                      # Başarılı → döngüden çık
                except Exception as e:
                    if retry == max_retries - 1:
                        logger.warning(
                            f"Pagination hatası ({symbol} {timeframe}, "
                            f"sayfa {len(all_data)//200 + 1}): {e}"
                        )
                        # Toplanan veriyi döndür (kısmi veri > hiç veri)
                        break
                    time.sleep(0.5 * (retry + 1))  # Exponential backoff
            else:
                break  # max_retries aşıldı
            
            if not ohlcv:
                break                          # Veri bitti
            
            all_data.extend(ohlcv)
            remaining -= len(ohlcv)
            
            # Sonraki sayfa için timestamp'i ilerlet
            # Son mumun zamanı + 1 timeframe kadar ilerle
            last_ts = ohlcv[-1][0]
            current_since = last_ts + (tf_minutes * 60 * 1000)
            
            # Rate limit koruması (Bitget: 20 req/s genel, 10 req/s per IP)
            time.sleep(0.15)
            
            # Aynı mumları tekrar çekiyorsak dur (veri sonu)
            if len(ohlcv) < batch_size:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Duplicate temizliği (pagination sınırlarında olabilir)
        df = self._ohlcv_to_dataframe(all_data)
        df = df[~df.index.duplicated(keep='last')]
        
        return df

    def _ohlcv_to_dataframe(self, ohlcv: list) -> pd.DataFrame:
        """Ham OHLCV listesini pandas DataFrame'e çevirir."""
        if not ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Unix timestamp (ms) → UTC datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.index.name = None
        
        # Float dönüşümü (bazen string gelebilir)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df

    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================
    
    def fetch_max_ohlcv(
        self,
        symbol: str = None,
        timeframe: str = "1h",
        max_bars: int = None,
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        İC analizi için geriye dönük geniş veri seti çeker.
        
        Otomatik pagination ile Bitget'in 200 mum limitini aşar.
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü
        timeframe : str
            Zaman dilimi
        max_bars : int, optional
            İstenen bar sayısı. None ise RECOMMENDED_BARS'tan okunur.
        progress : bool
            True ise her sayfa loglanır
        """
        symbol = symbol or self.default_symbol
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 500)
        
        return self.fetch_ohlcv(symbol, timeframe, limit=max_bars)

    def fetch_all_timeframes(
        self,
        symbol: str = None,
        timeframes: List[str] = None,
        max_bars_override: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Birden fazla timeframe için OHLCV verisi çeker.
        
        Ana analiz döngüsünde her coin için çağrılır.
        
        Parameters:
        ----------
        symbol : str
            Bitget sembolü
        timeframes : List[str]
            Hangi TF'ler çekilsin. None ise config'den okunur.
        max_bars_override : int
            Her TF için sabit bar sayısı. None ise RECOMMENDED_BARS.
            
        Returns:
        -------
        Dict[str, pd.DataFrame]
            {timeframe: OHLCV DataFrame} — örn: {"1h": df_1h, "4h": df_4h}
        """
        symbol = symbol or self.default_symbol
        
        # TF listesini belirle
        if timeframes is None:
            if cfg.timeframes:
                timeframes = list(cfg.timeframes.keys())
            else:
                timeframes = self.DEFAULT_ACTIVE_TIMEFRAMES
        
        logger.info(f"📥 {symbol} → {len(timeframes)} TF çekiliyor...")
        
        data = {}
        for tf in timeframes:
            try:
                # Bar sayısını belirle
                if max_bars_override:
                    bars = max_bars_override
                elif cfg.timeframes and tf in cfg.timeframes:
                    bars = cfg.timeframes[tf].get('bars', self.RECOMMENDED_BARS.get(tf, 500))
                else:
                    bars = self.RECOMMENDED_BARS.get(tf, 500)
                
                df = self.fetch_ohlcv(symbol, tf, limit=bars)
                
                # Minimum bar kontrolü (IC analizi için en az 50 bar)
                if len(df) >= 50:
                    data[tf] = df
                    logger.info(f"  {tf}: ✓ {len(df)} bar")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz ({len(df)} < 50)")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata — {e}")
            
            # TF'ler arası bekleme (rate limit koruması)
            time.sleep(0.2)
        
        logger.info(f"📊 {symbol}: {len(data)}/{len(timeframes)} TF başarılı")
        return data

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Çekilen verinin kalitesini doğrular.
        
        Returns:
        -------
        Dict
            is_valid, rows, missing_count, last_date bilgileri
        """
        if df.empty:
            return {'is_valid': False, 'rows': 0, 'missing_count': 0, 'last_date': None}
        
        missing = df.isnull().sum().sum()
        return {
            'is_valid': missing == 0 and len(df) > 0,
            'rows': len(df),
            'missing_count': int(missing),
            'last_date': df.index[-1] if not df.empty else None,
        }


# =============================================================================
# TEST — Doğrudan çalıştırıldığında
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("\n🔧 BitgetFetcher v4.0 — Tam Bitget Test")
    print("=" * 50)
    
    f = BitgetFetcher()
    
    # 1. Coin listesi
    symbols = f.get_all_usdt_futures()
    print(f"\n✅ {len(symbols)} USDT-M çifti")
    
    # 2. Toplu ticker
    tickers = f.fetch_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'])
    for sym, t in tickers.items():
        print(f"  {sym.split('/')[0]}: ${t.get('last', 0):,.2f}")
    
    # 3. Tek ticker
    t = f.get_ticker('BTC/USDT:USDT')
    print(f"\n✅ BTC Ticker: ${t['last']:,.2f} | Vol: ${t['volume_24h']:,.0f}")
    
    # 4. OHLCV — tek istek (≤200)
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=100)
    print(f"\n✅ BTC 1h (100 bar): {len(df)} bar | Son: ${df['close'].iloc[-1]:,.2f}")
    
    # 5. OHLCV — pagination (>200)
    df500 = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=500)
    print(f"✅ BTC 1h (500 bar): {len(df500)} bar | İlk: {df500.index[0].strftime('%m/%d')}")
    
    # 6. Multi-TF
    data = f.fetch_all_timeframes('ETH/USDT:USDT', timeframes=['15m', '1h', '4h'])
    for tf, df in data.items():
        print(f"  {tf}: {len(df)} bar")
    
    print(f"\n🎉 Tüm testler başarılı — Bitget tam çalışıyor!")
'''

fetcher_path = src / 'data' / 'fetcher.py'
fetcher_path.write_text(FETCHER_CODE, encoding='utf-8')
ok("fetcher.py v4.0 yazıldı (tamamen Bitget, pagination destekli)")


# =============================================================================
# ADIM 3: coin_scanner.py ve main.py KONTROL
# =============================================================================
print(f"\n{B}[3/4] Diğer dosyalar kontrol ediliyor...{X}")

# ─── coin_scanner.py ───
scanner_path = src / 'scanner' / 'coin_scanner.py'
if scanner_path.exists():
    sc = scanner_path.read_text(encoding='utf-8')
    
    # coin_scanner'da 3 olası pattern var:
    # A) self.fetcher.exchange.fetch_tickers() → Doğru (direkt Bitget)
    # B) self.fetcher.fetch_tickers(symbols)   → Doğru (wrapper metod)
    # C) self.fetcher.fetch_tickers()          → Doğru (parametresiz)
    
    if 'self.fetcher.exchange.fetch_tickers()' in sc:
        ok("coin_scanner.py: exchange.fetch_tickers() → Bitget direkt ✓")
    elif 'self.fetcher.fetch_tickers(' in sc:
        ok("coin_scanner.py: fetcher.fetch_tickers() → wrapper metod ✓")
    else:
        warn("coin_scanner.py: Ticker çağrısı tespit edilemedi")
        info("  Manuel kontrol: grep -n 'fetch_ticker' src/scanner/coin_scanner.py")

# ─── main.py ───
main_path = src / 'main.py'
if main_path.exists():
    mc = main_path.read_text(encoding='utf-8')
    changes = 0
    
    # 1. volume_24h → quoteVolume (Bitget raw ticker'da quoteVolume var)
    if "ticker.get('volume_24h'" in mc:
        mc = mc.replace(
            "ticker.get('volume_24h', 0)",
            "ticker.get('quoteVolume', 0)"
        )
        changes += 1
        ok("main.py: volume_24h → quoteVolume düzeltildi")
    elif "ticker.get('quoteVolume'" in mc:
        ok("main.py: quoteVolume zaten doğru ✓")
    
    # 2. Ticker çağrısı kontrol
    #    self.fetcher.exchange.fetch_ticker() → Doğru (Bitget direkt)
    #    self.fetcher.get_ticker()            → Doğru (wrapper)
    if 'self.fetcher.exchange.fetch_ticker(' in mc:
        ok("main.py: exchange.fetch_ticker() → Bitget direkt ✓")
    if 'self.fetcher.get_ticker(' in mc:
        ok("main.py: get_ticker() wrapper kullanılıyor ✓")
    
    # 3. OHLCV çağrısı kontrol
    if 'self.fetcher.fetch_ohlcv(' in mc:
        ok("main.py: fetch_ohlcv() kullanılıyor → artık Bitget'e gidecek ✓")
    
    # 4. _analyze_coin sembol formatı kontrol
    if 'full_symbol = f"{clean_coin}/USDT:USDT"' in mc:
        ok("main.py: _analyze_coin sembol formatı doğru ✓")
    
    if changes > 0:
        main_path.write_text(mc, encoding='utf-8')
        ok(f"main.py: {changes} değişiklik uygulandı")
    else:
        ok("main.py: Değişiklik gerekmedi")

# ─── data/__init__.py ───
init_path = src / 'data' / '__init__.py'
if init_path.exists():
    ic = init_path.read_text(encoding='utf-8')
    if 'DataFetcher = BitgetFetcher' not in ic:
        # Alias ekle (geriye uyumluluk)
        ic = ic.replace(
            "from .fetcher import BitgetFetcher",
            "from .fetcher import BitgetFetcher\n\n# Geriye uyumluluk alias'ı\nDataFetcher = BitgetFetcher"
        )
        init_path.write_text(ic, encoding='utf-8')
        ok("data/__init__.py: DataFetcher alias eklendi")
    else:
        ok("data/__init__.py: Alias mevcut ✓")


# =============================================================================
# ADIM 4: CANLI DOĞRULAMA
# =============================================================================
print(f"\n{B}[4/4] Canlı doğrulama testleri...{X}")

# Modül cache'ini temizle
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['data', 'fetcher', 'config', 'scanner']):
        del sys.modules[mod]

sys.path.insert(0, str(src))

try:
    from data.fetcher import BitgetFetcher
    f = BitgetFetcher()
    
    # Test 1: Coin listesi
    symbols = f.get_all_usdt_futures()
    assert len(symbols) > 100, f"Sadece {len(symbols)} çift!"
    ok(f"[1/6] Coin listesi: {len(symbols)} çift")
    
    # Test 2: Toplu ticker
    tickers = f.fetch_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT'])
    btc = tickers.get('BTC/USDT:USDT', {})
    assert btc.get('last', 0) > 1000, "BTC fiyat yok!"
    ok(f"[2/6] Toplu ticker: BTC ${btc['last']:,.2f}")
    
    # Test 3: Tek ticker
    t = f.get_ticker('BTC/USDT:USDT')
    assert t['last'] > 1000
    ok(f"[3/6] Tek ticker: BTC ${t['last']:,.2f}")
    
    # Test 4: OHLCV tek istek (≤200)
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=100)
    assert len(df) >= 50, f"OHLCV yetersiz: {len(df)}"
    ok(f"[4/6] OHLCV (100 bar): {len(df)} bar çekildi")
    
    # Test 5: OHLCV pagination (>200)
    df_big = f.fetch_ohlcv('SOL/USDT:USDT', '1h', limit=400)
    assert len(df_big) >= 200, f"Pagination başarısız: {len(df_big)}"
    ok(f"[5/6] OHLCV pagination (400 bar): {len(df_big)} bar çekildi")
    
    # Test 6: fetch_ohlcv BTC ile doğrula (pipeline'daki asıl çağrı)
    df_verify = f.fetch_ohlcv('BTC/USDT:USDT', timeframe='4h', limit=500)
    assert len(df_verify) >= 100, f"4h verisi yetersiz: {len(df_verify)}"
    ok(f"[6/6] BTC 4h (500 bar): {len(df_verify)} bar çekildi")

except Exception as e:
    fail(f"Doğrulama hatası: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n{R}Düzeltme başarısız! Yedek geri yükleniyor...{X}")
    # Yedekten geri yükle
    for f in files_to_backup:
        backup_f = backup_dir / f.relative_to(src)
        if backup_f.exists():
            shutil.copy2(backup_f, f)
    print(f"Yedek geri yüklendi → {backup_dir}")
    sys.exit(1)


# =============================================================================
# SONUÇ
# =============================================================================
print(f"\n{B}{'='*60}")
print(f"  ✅ KESİN ÇÖZÜM TAMAMLANDI!")
print(f"{'='*60}{X}")
print(f"""
  Mimari (v4.0 — Tam Bitget):
  ┌─────────────────────────────────────────┐
  │  Coin Listesi  → Bitget USDT-M Markets  │
  │  Ticker/Fiyat  → Bitget fetch_tickers   │
  │  OHLCV Verisi  → Bitget + Pagination    │
  │  Emir/İşlem    → Bitget Executor        │
  │  Binance       → KALDIRILDI ❌           │
  └─────────────────────────────────────────┘

  Yedek: {backup_dir.relative_to(cwd)}/

  Şimdi pipeline'ı test et:
  {C}cd src && python main.py --dry-run{X}

  Veya sadece fetcher'ı test et:
  {C}cd src && python data/fetcher.py{X}
""")
