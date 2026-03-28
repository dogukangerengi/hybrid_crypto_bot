# =============================================================================
# EMİR YÖNETİMİ MODÜLÜ (EXECUTION) - BINANCE FUTURES
# =============================================================================
# Risk yönetimi, pozisyon sizing, emir gönderme ve izleme.
#
# Kullanım:
#   from execution import RiskManager, BinanceExecutor
#   rm = RiskManager(balance=75.0)
#   trade = rm.calculate_trade(entry=185, direction='SHORT', atr=3.7)
#   executor = BinanceExecutor(dry_run=True)
#   result = executor.execute_trade(trade)
# =============================================================================

from .risk_manager import (
    RiskManager,
    TradeCalculation,
    StopLossResult,
    TakeProfitResult,
    PositionSizeResult,
    RiskCheckStatus,
    TradeDirection,
)

from .binance_executor import (
    BinanceExecutor,
    ExecutionResult,
    OrderResult,
)

__all__ = [
    # Risk yönetimi
    'RiskManager',
    'TradeCalculation',
    'StopLossResult',
    'TakeProfitResult',
    'PositionSizeResult',
    'RiskCheckStatus',
    'TradeDirection',
    # Emir yönetimi
    'BinanceExecutor',
    'ExecutionResult',
    'OrderResult',
]

__version__ = '2.0.0'