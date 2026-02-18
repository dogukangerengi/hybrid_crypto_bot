# =============================================================================
# EMİR YÖNETİMİ MODÜLÜ (EXECUTION)
# =============================================================================
# Risk yönetimi, pozisyon sizing, emir gönderme ve izleme.
#
# Kullanım:
#   from execution import RiskManager, BitgetExecutor
#   rm = RiskManager(balance=75.0)
#   trade = rm.calculate_trade(entry=185, direction='SHORT', atr=3.7)
#   executor = BitgetExecutor(dry_run=True)
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

from .bitget_executor import (
    BitgetExecutor,
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
    'BitgetExecutor',
    'ExecutionResult',
    'OrderResult',
]

__version__ = '1.1.0'
