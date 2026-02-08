# =============================================================================
# AI KARAR MODÜLÜ (AI OPTIMIZER)
# =============================================================================
# Gemini API ile IC analiz sonuçlarını değerlendirip
# nihai LONG/SHORT/WAIT kararı verir.
#
# Kullanım:
#   from ai import GeminiOptimizer, AIAnalysisInput
#   optimizer = GeminiOptimizer()
#   decision = optimizer.get_decision(input_data)
# =============================================================================

from .gemini_optimizer import (
    GeminiOptimizer,
    AIAnalysisInput,
    AIDecisionResult,
    AIDecision,
    GateAction,
)

__all__ = [
    'GeminiOptimizer',         # Ana AI optimizer sınıfı
    'AIAnalysisInput',         # Giriş verisi dataclass
    'AIDecisionResult',        # Karar sonucu dataclass
    'AIDecision',              # LONG/SHORT/WAIT enum
    'GateAction',              # NO_TRADE/REPORT_ONLY/FULL_TRADE enum
]

__version__ = '1.0.0'
