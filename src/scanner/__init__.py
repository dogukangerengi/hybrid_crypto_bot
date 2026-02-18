# =============================================================================
# DİNAMİK TARAYICI MODÜLÜ (SCANNER)
# =============================================================================
# Bitget USDT-M Futures çiftlerini tarayıp IC analiz için en uygun
# coin'leri otomatik seçer.
#
# Kullanım:
#   from scanner import CoinScanner, CoinScanResult
#   scanner = CoinScanner()
#   top_coins = scanner.scan(top_n=20)
# =============================================================================

from .coin_scanner import CoinScanner, CoinScanResult

__all__ = [
    'CoinScanner',         # Ana tarayıcı sınıf
    'CoinScanResult',      # Tarama sonucu dataclass
]

__version__ = '1.0.0'
