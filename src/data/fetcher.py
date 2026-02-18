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

# =============================================================================
# BINANCE DATA & BITGET EXECUTION MODÜLÜ
# =============================================================================
# Strateji:
# 1. Coin Listesi & Fiyatlar -> BITGET (Çünkü işlem burada yapılacak)
# 2. Tarihsel Veri (OHLCV)   -> BINANCE (Çünkü veri orada daha kaliteli/eksiksiz)
# =============================================================================

import ccxt
import pandas as pd
import time
import logging
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
    Veriyi Binance'den analiz edip, Bitget uyumlu formatta sunan sınıf.
    """
    
    # Binance ve Bitget uyumlu timeframe haritası
    TIMEFRAME_MINUTES = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    
    # Binance tek seferde 1500 mum verebilir (Bitget 200'dü)
    MAX_CANDLES_PER_REQUEST = 1000 
    
    # Analiz için önerilen bar sayıları
    RECOMMENDED_BARS = {
        "5m": 3000, "15m": 2000, "30m": 1500, 
        "1h": 1000, "4h": 500, "1d": 365
    }

    def __init__(self, symbol: str = None):
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # 1. BITGET (Execution & Market Info)
        # İşlem açacağımız borsa. Semboller ve anlık fiyat buradan gelmeli.
        self.exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'}, # USDT-M Futures
            'enableRateLimit': True,
        })
        
        # 2. BINANCE (Data Source)
        # Analiz verisi buradan gelecek. API Key gerekmez (Public data).
        self.binance = ccxt.binance({
            'options': {'defaultType': 'future'}, # USDT-M Futures
            'enableRateLimit': True,
        })
        
        self._markets_loaded = False

    def _ensure_markets_loaded(self):
        """Bitget marketlerini yükle (Binance'e gerek yok)."""
        if not self._markets_loaded:
            try:
                logger.info("Bitget market bilgileri yükleniyor...")
                self.exchange.load_markets()
                self._markets_loaded = True
                
                # İstatistik
                count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
                logger.info(f"✓ {count} Bitget USDT-M çifti aktif")
            except Exception as e:
                logger.error(f"Market yükleme hatası: {e}")

    def get_all_usdt_futures(self) -> List[str]:
        """Sadece Bitget'te işlem gören coinleri listele."""
        self._ensure_markets_loaded()
        return [s for s in self.exchange.markets if s.endswith(':USDT')]

    def get_market_info(self, symbol: str = None) -> Dict:
        """Emir gönderirken gereken kuralları Bitget'ten al."""
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

    def get_ticker(self, symbol: str = None) -> Dict:
        """
        Anlık fiyatı BITGET'ten al.
        ÖNEMLİ: Binance verisiyle analiz yapsak da, işlemi Bitget fiyatıyla açacağız.
        Spread/Slippage hesabı için Bitget fiyatı şart.
        """
        symbol = symbol or self.default_symbol
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'last': ticker.get('last', 0),
            'bid': ticker.get('bid', 0),
            'ask': ticker.get('ask', 0),
            'volume_24h': ticker.get('quoteVolume', 0),
            'percentage': ticker.get('percentage', 0),
        }

    # =========================================================================
    # VERİ ÇEKME MOTORU (BINANCE GÜCÜYLE)
    # =========================================================================
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200, since=None) -> pd.DataFrame:
        """
        Bitget sembolünü alır, Binance'den verisini çeker.
        Girdi: 'BTC/USDT:USDT' (Bitget formatı)
        İşlem: 'BTC/USDT' (Binance formatı) ile veri çek
        """
        # 1. Sembol Dönüşümü
        # Bitget: "BTC/USDT:USDT" -> Binance: "BTC/USDT"
        clean_symbol = symbol.split(':')[0] 
        
        try:
            # 2. Binance'den Veri Çek
            # Limit kontrolü: Binance max 1500, biz güvenli tarafta 1000 diyelim
            req_limit = min(limit, 1000)
            
            ohlcv = self.binance.fetch_ohlcv(clean_symbol, timeframe, limit=req_limit, since=since)
            
            if not ohlcv:
                # logger.warning(f"{clean_symbol} Binance'de veri yok.")
                return pd.DataFrame()
            
            # 3. DataFrame Formatlama
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = None
            
            # Float dönüşümü
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            # logger.debug(f"Binance veri hatası ({clean_symbol}): {e}")
            return pd.DataFrame()

    def fetch_max_ohlcv(self, symbol: str, timeframe: str = "1h", max_bars=None, progress=False) -> pd.DataFrame:
        """Geriye dönük geniş veri seti çeker (Binance üzerinden)."""
        if max_bars is None:
            max_bars = self.RECOMMENDED_BARS.get(timeframe, 1000)
            
        # Binance tek istekte 1000+ mum verebildiği için pagination'a pek gerek kalmaz
        # ama yine de fetch_ohlcv fonksiyonunu çağırarak standart yapıyı koruyoruz.
        return self.fetch_ohlcv(symbol, timeframe, limit=max_bars)

    def fetch_all_timeframes(self, symbol, timeframes=None, max_bars_override=None) -> Dict[str, pd.DataFrame]:
        """Çoklu timeframe verisi çeker."""
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]
            
        data = {}
        # Binance hızlı olduğu için bekleme süresini (sleep) kısabiliriz
        for tf in timeframes:
            try:
                bars = max_bars_override or self.RECOMMENDED_BARS.get(tf, 500)
                df = self.fetch_ohlcv(symbol, tf, limit=bars)
                if len(df) > 50:
                    data[tf] = df
            except:
                pass
            time.sleep(0.1) # Binance API rate limit koruması
            
        return data

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Veri kalitesini kontrol et."""
        return {
            'is_valid': not df.empty and df.isnull().sum().sum() == 0,
            'rows': len(df),
            'last_date': df.index[-1] if not df.empty else None
        }

# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    f = BitgetFetcher()
    
    # Test: Bitget sembolü verip Binance verisi alıyor muyuz?
    symbol = "BTC/USDT:USDT" 
    print(f"\nTest ediliyor: {symbol}")
    
    # Ticker (Bitget'ten gelmeli)
    t = f.get_ticker(symbol)
    print(f"Bitget Fiyat: ${t['last']}")
    
    # OHLCV (Binance'den gelmeli)
    df = f.fetch_ohlcv(symbol, "1h", limit=100)
    print(f"Binance Verisi: {len(df)} bar çekildi")
    print(f"Son Kapanış: ${df['close'].iloc[-1]}")
