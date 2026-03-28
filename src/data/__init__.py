# =============================================================================
# VERİ MODÜLÜ (DATA MODULE) — Binance Futures
# =============================================================================
# Binance USDT-M Perpetual Futures veri çekme ve ön işleme.
#
# Kullanım:
#   from data import BinanceFetcher, DataPreprocessor
#   # veya geriye uyumluluk için:
#   from data import DataFetcher  # → BinanceFetcher alias'ı
# =============================================================================

from .fetcher import BinanceFetcher
from .preprocessor import DataPreprocessor

# Geriye uyumluluk alias'ı — eski modüller DataFetcher bekliyor
# main.py, telegram_bot.py, app.py hepsi DataFetcher import eder
DataFetcher = BinanceFetcher

__all__ = [
    'BinanceFetcher',      # Yeni isim (Binance Futures)
    'DataFetcher',         # Eski isim (alias, geriye uyumlu)
    'DataPreprocessor',    # Veri ön işleme
]

__version__ = '2.1.0'     # v2.1: alias eklendi