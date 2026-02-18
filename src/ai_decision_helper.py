# =============================================================================
# AI DECISION HELPER — Direction → AIDecisionType Çevirici
# =============================================================================
# Bu dosya gemini_optimizer.py'ye eklenecek veya ayrı kullanılacak helper.
#
# Mevcut gemini_optimizer.py'de AIDecisionType enum'una şu metodu ekle:
# =============================================================================

from enum import Enum


class AIDecisionType(Enum):
    """AI karar türleri."""
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    
    @classmethod
    def from_direction(cls, direction: str) -> 'AIDecisionType':
        """
        IC direction'dan AIDecisionType'a çevir.
        
        Parameters:
        ----------
        direction : str
            IC analiz sonucu yön ('LONG', 'SHORT', 'NEUTRAL', etc.)
            
        Returns:
        -------
        AIDecisionType
            Karşılık gelen decision type
        """
        direction_upper = direction.upper() if direction else "WAIT"
        
        if direction_upper in ("LONG", "BUY", "BULLISH"):
            return cls.LONG
        elif direction_upper in ("SHORT", "SELL", "BEARISH"):
            return cls.SHORT
        else:
            return cls.WAIT


# =============================================================================
# KURULUM TALİMATI
# =============================================================================
# 
# Mevcut gemini_optimizer.py dosyandaki AIDecisionType enum'una
# from_direction metodunu ekle:
#
# class AIDecisionType(Enum):
#     LONG = "LONG"
#     SHORT = "SHORT"
#     WAIT = "WAIT"
#     
#     @classmethod
#     def from_direction(cls, direction: str) -> 'AIDecisionType':
#         direction_upper = direction.upper() if direction else "WAIT"
#         if direction_upper in ("LONG", "BUY", "BULLISH"):
#             return cls.LONG
#         elif direction_upper in ("SHORT", "SELL", "BEARISH"):
#             return cls.SHORT
#         else:
#             return cls.WAIT
#
# =============================================================================
