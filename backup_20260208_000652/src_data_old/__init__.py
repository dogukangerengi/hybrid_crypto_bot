# =============================================================================
# VERİ MODÜLÜ (DATA MODULE)
# =============================================================================
# Bitget Futures veri çekme ve ön işleme.
#
# Kullanım:
# from data import BitgetFetcher, DataPreprocessor
#
# fetcher = BitgetFetcher()
# df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=200)
#
# pp = DataPreprocessor()
# df_clean = pp.full_pipeline(df)
# =============================================================================

from .fetcher import BitgetFetcher
from .preprocessor import DataPreprocessor

__all__ = [
    'BitgetFetcher',
    'DataPreprocessor',
]

__version__ = '2.0.0'
