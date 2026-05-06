# =============================================================================
# BINANCE FUTURES VERİ ÇEKME MODÜLÜ v5.0 — TAM BINANCE
# =============================================================================
# Tüm veri Binance'ten geliyor:
# - Coin listesi      → Binance USDT-M Futures markets
# - Ticker (fiyat)    → Binance fetch_ticker / fetch_tickers
# - OHLCV (mum veri)  → Binance fetch_ohlcv + pagination
# - Market bilgisi    → Binance markets (contract size, precision vb.)
#
# Binance OHLCV limiti: 1000 mum/istek → Binance'e göre 5 kat daha hızlı ve stabil!
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


class BinanceFetcher:
    """
    Binance USDT-M Perpetual Futures veri çekme sınıfı.
    
    Tüm veri tek borsadan (Binance) gelir:
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
    
    # Binance tek istekte max 1000 mum verir (Binance'in 200 sınırından çok daha iyi)
    MAX_CANDLES_PER_REQUEST = 1000
    
    # IC analizi için önerilen minimum bar sayıları
    RECOMMENDED_BARS: Dict[str, int] = {
        "5m": 2000,    # ~7 gün     | IC güvenilir olması için yeterli sample
        "15m": 1500,   # ~15 gün    | Trend + momentum analizi
        "30m": 1000,   # ~20 gün    | Swing analizi
        "1h": 750,     # ~31 gün    | Günlük döngü analizi
        "2h": 1500,    # ~125 gün (~4 ay)
        "4h": 500,     # ~83 gün    | Haftalık trend
        "1d": 365,     # ~1 yıl     | Uzun vadeli rejim
    }
    
    # Varsayılan aktif timeframe'ler
    DEFAULT_ACTIVE_TIMEFRAMES = [
        "15m", "30m", "1h", "2h", "4h"            # settings.yaml ile senkron
    ]

    # =========================================================================
    # BAŞLATMA (CONSTRUCTOR)
    # =========================================================================
    
    def __init__(self, symbol: str = None):
        """
        BinanceFetcher başlat.
        
        Parameters:
        ----------
        symbol : str, optional
            Varsayılan sembol. None ise config'den okunur.
        """
        self.default_symbol = symbol or cfg.exchange.default_symbol
        
        # CCXT Binance bağlantısı — USDT-M Perpetual Futures
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'}, # future = USDT-M Futures (Binance için)
            'enableRateLimit': True,              # Otomatik rate limiting
        })
        
        self._markets_loaded = False             # Lazy loading flag

    # =========================================================================
    # MARKET BİLGİSİ
    # =========================================================================
    
    def _ensure_markets_loaded(self):
        """Binance market bilgilerini lazy-load et (ilk çağrıda yüklenir)."""
        if not self._markets_loaded:
            try:
                logger.info("Binance market bilgileri yükleniyor...")
                self.exchange.load_markets()
                self._markets_loaded = True
                
                # İstatistik log
                count = sum(1 for s in self.exchange.markets if s.endswith(':USDT'))
                logger.info(f"✓ {count} Binance USDT-M çifti aktif")
            except Exception as e:
                logger.error(f"Market yükleme hatası: {e}")
                raise

    def get_all_usdt_futures(self) -> List[str]:
        """
        Binance'teki TÜM USDT-M Perpetual Futures çiftlerini döndürür.
        
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
            Binance sembolü (örn: 'BTC/USDT:USDT')
            
        Returns:
        -------
        Dict
            contract_size, precision, limits, max_leverage bilgileri
        """
        symbol = symbol or self.default_symbol
        self._ensure_markets_loaded()
        
        if symbol not in self.exchange.markets:
            raise ValueError(f"{symbol} Binance'te bulunamadı!")
        
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
            # Binance'de maxLeverage genellikle info içinde veya limits altındadır
            'max_leverage': int(market.get('limits', {}).get('leverage', {}).get('max', 20)),
        }

    # =========================================================================
    # TICKER (FİYAT BİLGİSİ)
    # =========================================================================
    
    def get_ticker(self, symbol: str = None) -> Dict:
        """
        Tek coin için anlık fiyat bilgisi (Binance).
        
        İşlem açarken Binance fiyatı kullanılmalı (execution price).
        
        Parameters:
        ----------
        symbol : str
            Binance sembolü (örn: 'BTC/USDT:USDT')
            
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
        
        CoinScanner bu metodu çağırır.
        
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
        limit: int = 1000,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Binance'ten OHLCV (mum) verisi çeker.
        
        1000'den fazla mum istenirse otomatik pagination yapar.
        
        Parameters:
        ----------
        symbol : str
            Binance sembolü (örn: 'BTC/USDT:USDT')
        timeframe : str
            Zaman dilimi (1m, 5m, 15m, 1h, 4h, 1d vb.)
        limit : int
            İstenen mum sayısı. >1000 ise pagination yapılır.
        since : int, optional
            Başlangıç timestamp (ms). None ise en son mumlardan geriye.
            
        Returns:
        -------
        pd.DataFrame
            Index: timestamp (UTC), Columns: open, high, low, close, volume
        """
        symbol = symbol or self.default_symbol
        
        try:
            # 1000'den az isteniyorsa tek istek yeterli
            if limit <= self.MAX_CANDLES_PER_REQUEST:
                return self._fetch_ohlcv_single(symbol, timeframe, limit, since)
            
            # 1000'den fazla → pagination
            return self._fetch_ohlcv_paginated(symbol, timeframe, limit)
            
        except Exception as e:
            logger.warning(f"OHLCV hatası ({symbol} {timeframe}): {e}")
            return pd.DataFrame()

    def _fetch_ohlcv_single(
        self, symbol: str, timeframe: str, limit: int, since=None
    ) -> pd.DataFrame:
        """Tek istekte OHLCV çeker (≤1000 mum)."""
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
        Her istekte 1000 mum çek, timestamp'i ilerlet.
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
                            f"sayfa {len(all_data)//1000 + 1}): {e}"
                        )
                        break
                    time.sleep(0.5 * (retry + 1))  # Exponential backoff
            else:
                break  # max_retries aşıldı
            
            if not ohlcv:
                break                          # Veri bitti
            
            all_data.extend(ohlcv)
            remaining -= len(ohlcv)
            
            last_ts = ohlcv[-1][0]
            current_since = last_ts + (tf_minutes * 60 * 1000)
            
            time.sleep(0.15)
            
            if len(ohlcv) < batch_size:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Duplicate temizliği
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
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.index.name = None
        
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
        """
        symbol = symbol or self.default_symbol
        
        if timeframes is None:
            if hasattr(cfg, 'timeframes') and cfg.timeframes:
                timeframes = list(cfg.timeframes.keys())
            else:
                timeframes = self.DEFAULT_ACTIVE_TIMEFRAMES
        
        logger.info(f"📥 {symbol} → {len(timeframes)} TF çekiliyor...")
        
        data = {}
        for tf in timeframes:
            try:
                if max_bars_override:
                    bars = max_bars_override
                elif hasattr(cfg, 'timeframes') and cfg.timeframes and tf in cfg.timeframes:
                    tf_cfg = cfg.timeframes[tf]
                    bars = tf_cfg.get("bars", self.RECOMMENDED_BARS.get(tf, 500)) if isinstance(tf_cfg, dict) else int(tf_cfg)
                else:
                    bars = self.RECOMMENDED_BARS.get(tf, 500)
                
                df = self.fetch_ohlcv(symbol, tf, limit=bars)
                
                if len(df) >= 50:
                    data[tf] = df
                    logger.info(f"  {tf}: ✓ {len(df)} bar")
                else:
                    logger.warning(f"  {tf}: ✗ Yetersiz ({len(df)} < 50)")
                    
            except Exception as e:
                logger.error(f"  {tf}: ✗ Hata — {e}")
            
            time.sleep(0.1) # Binance daha esnek, süreyi kısalttık
        
        logger.info(f"📊 {symbol}: {len(data)}/{len(timeframes)} TF başarılı")
        return data

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Çekilen verinin kalitesini doğrular.
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
    
    print("\n🔧 BinanceFetcher v5.0 — Tam Binance Test")
    print("=" * 50)
    
    f = BinanceFetcher()
    
    # 1. Coin listesi
    symbols = f.get_all_usdt_futures()
    print(f"\n✅ {len(symbols)} USDT-M çifti bulundu")
    
    # 2. Toplu ticker
    tickers = f.fetch_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'])
    for sym, t in tickers.items():
        print(f"  {sym.split('/')[0]}: ${t.get('last', 0):,.2f}")
    
    # 3. Tek ticker
    t = f.get_ticker('BTC/USDT:USDT')
    print(f"\n✅ BTC Ticker: ${t['last']:,.2f} | Vol: ${t['volume_24h']:,.0f}")
    
    # 4. OHLCV — tek istek
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=100)
    print(f"\n✅ BTC 1h (100 bar): {len(df)} bar | Son: ${df['close'].iloc[-1]:,.2f}")
    
    # 5. Multi-TF
    data = f.fetch_all_timeframes('ETH/USDT:USDT', timeframes=['15m', '1h'])
    for tf, df in data.items():
        print(f"  {tf}: {len(df)} bar")
    
    print(f"\n🎉 Tüm testler başarılı — Binance tam çalışıyor!")