# =============================================================================
# VERİ MODÜLÜ (DATA MODULE) — Bitget Futures
# =============================================================================
# Bitget USDT-M Perpetual Futures veri çekme ve ön işleme.
#
# Kullanım:
#   from data import BitgetFetcher, DataPreprocessor
#   # veya geriye uyumluluk için:
#   from data import DataFetcher  # → BitgetFetcher alias'ı
# =============================================================================

from .fetcher import BitgetFetcher
from .preprocessor import DataPreprocessor

# Geriye uyumluluk alias'ı — eski modüller DataFetcher bekliyor
# main.py, telegram_bot.py, app.py hepsi DataFetcher import eder
DataFetcher = BitgetFetcher

__all__ = [
    'BitgetFetcher',       # Yeni isim (Bitget Futures)
    'DataFetcher',         # Eski isim (alias, geriye uyumlu)
    'DataPreprocessor',    # Veri ön işleme
]

__version__ = '2.1.0'     # v2.1: alias eklendi
