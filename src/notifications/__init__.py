# =============================================================================
# BİLDİRİM MODÜLÜ (NOTIFICATIONS) — v3.0
# =============================================================================
# Telegram bildirim sistemi — IC analiz, AI karar, trade execution, risk uyarı.
#
# Kullanım:
#   from notifications import TelegramNotifier, AnalysisReport
#   notifier = TelegramNotifier()
#   notifier.send_trade_sync(execution_result)
#   notifier.send_ai_decision_sync(ai_decision)
#   notifier.send_risk_alert_sync("kill_switch", "DD %15 aşıldı!", "critical")
# =============================================================================

from .telegram_notifier import (
    TelegramNotifier,
    AnalysisReport,
    create_notifier_from_env,
)

__all__ = [
    'TelegramNotifier',
    'AnalysisReport',
    'create_notifier_from_env',
]

__version__ = '3.0.0'
