# =============================================================================
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
    
    def fetch_tickers(self, symbols=None):
        """
        Toplu ticker verisi — Bitget exchange'e delege eder.
        CoinScanner._fetch_all_tickers() bu metodu çağırır.
        """
        all_tickers = self.exchange.fetch_tickers()
        if symbols:
            return {s: all_tickers[s] for s in symbols if s in all_tickers}
        return all_tickers

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
    print(f"\nTest: {symbol}")
    
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
