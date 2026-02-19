#!/usr/bin/env python3
# =============================================================================
# DOĞRU DÜZELTME SCRİPTİ — fetcher_v3 hatalarını geri alır
# =============================================================================
# fetcher_v3.py'nin bozduğu 2 şeyi düzeltir:
#
# 1. get_all_usdt_futures() → Binance (BOZUK: 0 sembol) → Bitget (DOĞRU: 536)
# 2. fetch_tickers() metodu → Binance filtre bozuk → SİL (gereksiz)
# 3. main.py volume_24h → quoteVolume geri döndür
#
# Çalıştır: python fix_now.py
# =============================================================================

import sys
import shutil
from pathlib import Path
from datetime import datetime

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")

# ─── Proje kökünü bul ───
cwd = Path.cwd()
if (cwd / 'src').exists():
    src = cwd / 'src'
elif cwd.name == 'src':
    src = cwd
else:
    fail("src/ klasörü bulunamadı! Proje kökünden çalıştır.")
    sys.exit(1)

print(f"\n{BOLD}{'='*60}")
print(f"  🔧 DÜZELTME — {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*60}{RESET}")

# =============================================================================
# DÜZELTME 1: fetcher.py — get_all_usdt_futures() Bitget'e geri döndür
# =============================================================================
print(f"\n{BOLD}[1/3] fetcher.py düzeltiliyor...{RESET}")

fetcher_path = src / 'data' / 'fetcher.py'
backup_path = src / 'data' / 'fetcher_YEDEK.py'

if not fetcher_path.exists():
    fail("fetcher.py bulunamadı!")
    sys.exit(1)

# Yedek al (eğer yoksa)
if not (src / 'data' / 'fetcher_fix_yedek.py').exists():
    shutil.copy2(fetcher_path, src / 'data' / 'fetcher_fix_yedek.py')
    info("Mevcut fetcher.py yedeği alındı → fetcher_fix_yedek.py")

content = fetcher_path.read_text(encoding='utf-8')

# ─── Kontrol: fetcher_v3 mi yüklü? ───
has_binance_markets = '_binance_markets_loaded' in content or '_ensure_binance_markets_loaded' in content
has_fetch_tickers_method = 'def fetch_tickers(' in content

if has_binance_markets or has_fetch_tickers_method:
    info("fetcher_v3 tespit edildi — Bitget + Binance hybrid yapıya geri dönülüyor")
    
    # Yedekten geri yükle (varsa)
    if backup_path.exists():
        shutil.copy2(backup_path, fetcher_path)
        ok("fetcher.py: Yedekten geri yüklendi (hybrid Bitget+Binance)")
    else:
        # Yedek yoksa manuel düzeltme yap
        warn("Yedek dosya bulunamadı — manuel düzeltme yapılıyor")
        
        # Tam çalışan fetcher'ı yaz
        new_fetcher = '''# =============================================================================
# HYBRID DATA FETCHER — Bitget Market + Binance OHLCV
# =============================================================================
# Strateji:
# 1. Coin Listesi & Fiyatlar → BITGET (İşlem burada yapılacak)
# 2. Tarihsel Veri (OHLCV)   → BINANCE (Veri kalitesi daha iyi)
# =============================================================================

import ccxt
import pandas as pd
import time
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

logger = logging.getLogger(__name__)


class BitgetFetcher:
    """
    Coin listesi ve fiyatlar Bitget'ten, OHLCV verisi Binance'den.
    """
    
    TIMEFRAME_MINUTES = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    
    MAX_CANDLES_PER_REQUEST = 1000              # Binance tek istekte max
    
    RECOMMENDED_BARS = {
        "5m": 3000, "15m": 2000, "30m": 1500,
        "1h": 1000, "4h": 500, "1d": 365,
    }

    def __init__(self, symbol: str = None):
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # 1. BITGET — Coin listesi, ticker, market info, execution
        self.exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
        
        # 2. BINANCE — OHLCV verisi (API key gerekmez)
        self.binance = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
        
        self._markets_loaded = False

    # =========================================================================
    # MARKET BİLGİSİ — BITGET
    # =========================================================================
    
    def _ensure_markets_loaded(self):
        """Bitget marketlerini lazy-load et."""
        if not self._markets_loaded:
            try:
                logger.info("Bitget market bilgileri yükleniyor...")
                self.exchange.load_markets()
                self._markets_loaded = True
                count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
                logger.info(f"✓ {count} Bitget USDT-M çifti aktif")
            except Exception as e:
                logger.error(f"Market yükleme hatası: {e}")

    def get_all_usdt_futures(self) -> List[str]:
        """Bitget'teki TÜM USDT-M Futures çiftlerini döndürür."""
        self._ensure_markets_loaded()
        return sorted([s for s in self.exchange.markets if s.endswith(':USDT')])

    def get_market_info(self, symbol: str = None) -> Dict:
        """Emir gönderirken gereken market bilgisini Bitget'ten alır."""
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        if symbol not in self.exchange.markets:
            raise ValueError(f"{symbol} Bitget'te bulunamadı!")
        
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
            },
            'max_leverage': int(market.get('info', {}).get('maxLever', 20)),
        }

    # =========================================================================
    # FİYAT BİLGİSİ — BITGET (execution fiyatı)
    # =========================================================================
    
    def get_ticker(self, symbol: str = None) -> Dict:
        """Anlık fiyatı Bitget'ten alır (execution fiyatı)."""
        symbol = symbol or self.default_symbol
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'last': ticker.get('last', 0),
            'bid': ticker.get('bid', 0),
            'ask': ticker.get('ask', 0),
            'volume_24h': ticker.get('quoteVolume', 0),
            'quoteVolume': ticker.get('quoteVolume', 0),  # Geriye uyumluluk
            'percentage': ticker.get('percentage', 0),
        }

    # =========================================================================
    # OHLCV VERİSİ — BINANCE
    # =========================================================================
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", 
                    limit: int = 200, since=None) -> pd.DataFrame:
        """
        Bitget sembolünü alır, Binance'den OHLCV çeker.
        'BTC/USDT:USDT' → 'BTC/USDT' dönüşümü otomatik.
        """
        clean_symbol = symbol.split(':')[0]     # Bitget → Binance format
        
        try:
            req_limit = min(limit, self.MAX_CANDLES_PER_REQUEST)
            ohlcv = self.binance.fetch_ohlcv(
                clean_symbol, timeframe, limit=req_limit, since=since
            )
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = None
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.debug(f"Binance OHLCV hatası ({clean_symbol} {timeframe}): {e}")
            return pd.DataFrame()

    def fetch_max_ohlcv(self, symbol: str = None, timeframe: str = "1h",
                        max_bars=None, progress=False) -> pd.DataFrame:
        """Geriye dönük geniş veri seti çeker (Binance)."""
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 1000)
        return self.fetch_ohlcv(symbol or self.default_symbol, timeframe, limit=max_bars)

    def fetch_all_timeframes(self, symbol=None, timeframes=None,
                             max_bars_override=None) -> Dict[str, pd.DataFrame]:
        """Çoklu timeframe verisi çeker (Binance)."""
        symbol = symbol or self.default_symbol
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]
        
        data = {}
        for tf in timeframes:
            try:
                bars = max_bars_override or self.RECOMMENDED_BARS.get(tf, 500)
                df = self.fetch_ohlcv(symbol, tf, limit=bars)
                if len(df) > 50:
                    data[tf] = df
            except:
                pass
            time.sleep(0.1)                    # Binance rate limit
        
        return data

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Veri kalitesini kontrol et."""
        return {
            'is_valid': not df.empty and df.isnull().sum().sum() == 0,
            'rows': len(df),
            'last_date': df.index[-1] if not df.empty else None,
        }


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    f = BitgetFetcher()
    
    symbol = "BTC/USDT:USDT"
    print(f"\\nTest: {symbol}")
    
    # Coin listesi (Bitget)
    symbols = f.get_all_usdt_futures()
    print(f"Bitget çiftleri: {len(symbols)}")
    
    # Ticker (Bitget)
    t = f.get_ticker(symbol)
    print(f"Bitget Fiyat: ${t['last']:,.2f}")
    
    # OHLCV (Binance)
    df = f.fetch_ohlcv(symbol, "1h", limit=100)
    print(f"Binance Verisi: {len(df)} bar")
    print(f"Son Kapanış: ${df['close'].iloc[-1]:,.2f}")
'''
        
        fetcher_path.write_text(new_fetcher, encoding='utf-8')
        ok("fetcher.py: Sıfırdan yazıldı (Bitget market + Binance OHLCV)")

else:
    info("fetcher.py zaten doğru versiyonda görünüyor")
    
    # Ama get_all_usdt_futures kontrol et
    if 'self.exchange.markets' in content and 'get_all_usdt_futures' in content:
        ok("get_all_usdt_futures() → Bitget kullanıyor")
    else:
        warn("get_all_usdt_futures() kontrol edilemedi — manuel bak")


# =============================================================================
# DÜZELTME 2: main.py — volume_24h → quoteVolume
# =============================================================================
print(f"\n{BOLD}[2/3] main.py düzeltiliyor...{RESET}")

main_path = src / 'main.py'
if main_path.exists():
    content = main_path.read_text(encoding='utf-8')
    changes = 0
    
    # volume_24h → quoteVolume (Bitget ticker 'quoteVolume' döndürür)
    old_vol = "ticker.get('volume_24h', 0)"
    new_vol = "ticker.get('quoteVolume', 0)"
    
    if old_vol in content:
        content = content.replace(old_vol, new_vol)
        changes += 1
        ok("volume_24h → quoteVolume düzeltildi")
    
    # self.fetcher.get_ticker → self.fetcher.exchange.fetch_ticker varsa düzelt
    # (eski kodda self.fetcher.exchange.fetch_ticker kullanılıyor ama
    #  yeni kodda self.fetcher.get_ticker() da çalışır çünkü get_ticker var)
    
    if changes > 0:
        main_path.write_text(content, encoding='utf-8')
        ok(f"main.py: {changes} değişiklik uygulandı")
    else:
        # quoteVolume zaten doğruysa kontrol et
        if "ticker.get('quoteVolume'" in content:
            ok("main.py: quoteVolume zaten doğru")
        else:
            info("main.py: Beklenen pattern bulunamadı — muhtemelen eski formatta")
else:
    fail("main.py bulunamadı!")


# =============================================================================
# DÜZELTME 3: coin_scanner.py — kontrol
# =============================================================================
print(f"\n{BOLD}[3/3] coin_scanner.py kontrol ediliyor...{RESET}")

scanner_path = src / 'scanner' / 'coin_scanner.py'
if scanner_path.exists():
    content = scanner_path.read_text(encoding='utf-8')
    
    if 'self.fetcher.exchange.fetch_tickers()' in content:
        ok("coin_scanner.py: Bitget ticker kullanıyor (DOĞRU)")
    elif 'self.fetcher.fetch_tickers(' in content:
        warn("coin_scanner.py: fetcher.fetch_tickers() kullanıyor — düzeltiliyor")
        content = content.replace(
            'self.fetcher.fetch_tickers(symbols)',
            'self.fetcher.exchange.fetch_tickers()'
        )
        content = content.replace(
            'self.fetcher.fetch_tickers()',
            'self.fetcher.exchange.fetch_tickers()'
        )
        # Return satırını da düzelt
        if 'return all_tickers\n' in content:
            content = content.replace(
                'return all_tickers\n',
                'return {s: all_tickers[s] for s in symbols if s in all_tickers}\n'
            )
        scanner_path.write_text(content, encoding='utf-8')
        ok("coin_scanner.py: Bitget ticker'a geri döndürüldü")
    else:
        info("coin_scanner.py: Ticker pattern farklı — manuel kontrol gerekli")
else:
    fail("coin_scanner.py bulunamadı!")


# =============================================================================
# DOĞRULAMA
# =============================================================================
print(f"\n{BOLD}{'='*60}")
print(f"  🔬 DOĞRULAMA TESTLERİ")
print(f"{'='*60}{RESET}")

sys.path.insert(0, str(src))

try:
    # Import kontrolü (cache'i temizle)
    if 'data.fetcher' in sys.modules:
        del sys.modules['data.fetcher']
    if 'data' in sys.modules:
        del sys.modules['data']
    
    from data.fetcher import BitgetFetcher
    f = BitgetFetcher()
    
    # Test 1: Coin listesi
    symbols = f.get_all_usdt_futures()
    if len(symbols) > 100:
        ok(f"get_all_usdt_futures(): {len(symbols)} çift ✓")
    else:
        fail(f"get_all_usdt_futures(): {len(symbols)} çift ✗")
        fail("fetcher.py düzgün yazılamadı — 'e' ile apply_fix.py --undo çalıştır")
        sys.exit(1)
    
    # Test 2: Ticker
    ticker = f.get_ticker('BTC/USDT:USDT')
    price = ticker.get('last', 0)
    if price > 1000:
        ok(f"get_ticker() BTC: ${price:,.2f} ✓")
    else:
        fail(f"get_ticker() BTC fiyatı bozuk: {price}")
    
    # Test 3: OHLCV
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=20)
    if len(df) > 10:
        ok(f"fetch_ohlcv() BTC 1h: {len(df)} bar ✓")
    else:
        fail(f"fetch_ohlcv() boş: {len(df)} bar")
    
    # Test 4: Batch ticker (Bitget exchange)
    import time
    start = time.time()
    all_tickers = f.exchange.fetch_tickers()
    elapsed = time.time() - start
    usdt = {k: v for k, v in all_tickers.items() if k.endswith(':USDT')}
    filled = sum(1 for t in usdt.values() if t.get('last', 0) > 0)
    ok(f"exchange.fetch_tickers(): {filled}/{len(usdt)} dolu ({elapsed:.1f}s) ✓")
    
except Exception as e:
    fail(f"Doğrulama hatası: {e}")
    import traceback
    traceback.print_exc()


# =============================================================================
# SONUÇ
# =============================================================================
print(f"\n{BOLD}{'='*60}")
print(f"  ✅ DÜZELTME TAMAMLANDI!")
print(f"{'='*60}{RESET}")
print(f"""
  Şimdi pipeline'ı test et:

  {CYAN}cd src{RESET}
  {CYAN}python main.py --dry-run{RESET}

  Veya debug_pipeline.py ile tekrar kontrol et:
  {CYAN}cd ~/hybrid_crypto_bot{RESET}
  {CYAN}python debug_pipeline.py{RESET}
""")
