# =============================================================================
# RİSK YÖNETİMİ MOTORU (RISK MANAGER)
# =============================================================================
# Amaç: Her işlem için pozisyon büyüklüğü, kaldıraç, SL/TP hesaplaması
#        ve risk limitlerinin kontrolünü sağlamak.
#
# Matematiksel Temel:
# ------------------
# 1. Position Size = Risk($) / SL_Distance($)
#    → Kaybedersek tam olarak Risk($) kadar kaybet
#
# 2. SL Distance = ATR × multiplier
#    → Volatiliteye adaptif stop loss
#    → Yüksek ATR = geniş SL = küçük pozisyon
#    → Düşük ATR = dar SL = büyük pozisyon
#
# 3. Leverage = Position_Value / Available_Margin
#    → Otomatik hesaplanır, config limitleri içinde kalır
#
# 4. Risk/Reward = TP_Distance / SL_Distance ≥ 1.5
#    → Pozitif beklenen değer için minimum eşik
# =============================================================================

import sys                                     
import logging                                 
import math                                    
from pathlib import Path                       
from typing import Dict, List, Optional, Tuple 
from dataclasses import dataclass, field       
from datetime import datetime, timezone, date  
from enum import Enum                          

# Proje config import
sys.path.insert(0, str(Path(__file__).parent.parent))  
from config import cfg                         

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUM & DATACLASS TANIMLARI
# =============================================================================

class TradeDirection(Enum):
    LONG = "LONG"                              
    SHORT = "SHORT"                            


class RiskCheckStatus(Enum):
    APPROVED = "APPROVED"                      
    REJECTED = "REJECTED"                      
    WARNING = "WARNING"                        


@dataclass
class StopLossResult:
    price: float                               
    distance: float                            
    distance_pct: float                        
    atr_multiplier: float                      


@dataclass
class TakeProfitResult:
    price: float                               
    distance: float                            
    distance_pct: float                        
    risk_reward: float                         


@dataclass
class PositionSizeResult:
    size: float                                
    value: float                               
    risk_amount: float                         
    margin_required: float                     
    leverage: int                              


@dataclass
class TradeCalculation:
    symbol: str                                
    direction: str                             
    entry_price: float                         
    
    stop_loss: StopLossResult                  
    take_profit: TakeProfitResult              
    position: PositionSizeResult               
    
    status: RiskCheckStatus                    
    checks: Dict[str, bool] = field(default_factory=dict)  
    rejection_reasons: List[str] = field(default_factory=list)  
    warnings: List[str] = field(default_factory=list)          
    
    def is_approved(self) -> bool:
        return self.status == RiskCheckStatus.APPROVED
    
    def summary(self) -> str:
        sl = self.stop_loss
        tp = self.take_profit
        pos = self.position
        
        emoji = "🟢" if self.direction == "LONG" else "🔴"
        status_emoji = "✅" if self.is_approved() else "❌"
        
        lines = [
            f"{status_emoji} {emoji} {self.symbol} {self.direction}",
            f"Entry: ${self.entry_price:,.2f}",
            f"SL: ${sl.price:,.2f} ({sl.distance_pct:+.2f}%, {sl.atr_multiplier}x ATR)",
            f"TP: ${tp.price:,.2f} ({tp.distance_pct:+.2f}%, RR={tp.risk_reward:.1f})",
            f"Size: {pos.size:.4f} (${pos.value:,.2f})",
            f"Leverage: {pos.leverage}x | Margin: ${pos.margin_required:,.2f}",
            f"Risk: ${pos.risk_amount:,.2f}",
        ]
        
        if self.rejection_reasons:
            lines.append(f"❌ Red: {', '.join(self.rejection_reasons)}")
        if self.warnings:
            lines.append(f"⚠️ Uyarı: {', '.join(self.warnings)}")
        
        return "\n".join(lines)


# =============================================================================
# ANA RİSK YÖNETİCİSİ SINIFI
# =============================================================================

class RiskManager:
    # SL mesafesi = ATR × multiplier
    # Geniş stop ile iğnelerden korunmak için 3.0x kullanıyoruz (Swing Trade)
    DEFAULT_ATR_MULTIPLIER = 3.0               
    MIN_ATR_MULTIPLIER = 2.0                   
    MAX_ATR_MULTIPLIER = 5.0                   
    
    def __init__(
        self,
        balance: float = 0.0,
        used_margin: float = 0.0,
        open_positions: int = 0,
        daily_pnl: float = 0.0,
        initial_balance: Optional[float] = None
    ):
        self.balance = balance
        self.used_margin = used_margin
        self.open_positions = open_positions
        self.daily_pnl = daily_pnl
        self.initial_balance = initial_balance or balance
        self.risk_cfg = cfg.risk
        
        logger.info(
            f"RiskManager başlatıldı | "
            f"Bakiye: ${balance:,.2f} | "
            f"Açık poz: {open_positions} | "
            f"Günlük PnL: ${daily_pnl:+,.2f}"
        )
    
    def update_state(
        self,
        balance: Optional[float] = None,
        used_margin: Optional[float] = None,
        open_positions: Optional[int] = None,
        daily_pnl: Optional[float] = None
    ) -> None:
        if balance is not None:
            self.balance = balance
        if used_margin is not None:
            self.used_margin = used_margin
        if open_positions is not None:
            self.open_positions = open_positions
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = None
    ) -> StopLossResult:
        multiplier = atr_multiplier or self.DEFAULT_ATR_MULTIPLIER
        multiplier = max(self.MIN_ATR_MULTIPLIER, min(multiplier, self.MAX_ATR_MULTIPLIER))
        
        sl_distance = atr * multiplier
        
        if direction.upper() == 'LONG':
            sl_price = entry_price - sl_distance   
        else:  
            sl_price = entry_price + sl_distance   
        
        max_sl_pct = getattr(self.risk_cfg, 'max_sl_pct', 8.0)
        sl_pct_raw = (sl_distance / entry_price) * 100
        if sl_pct_raw > max_sl_pct:
            logger.warning(f"   ⚠️ SL mesafesi daraltıldı: {sl_pct_raw:.1f}% → {max_sl_pct}%")
            sl_distance = entry_price * max_sl_pct / 100
            if direction == "LONG":
                sl_price = entry_price - sl_distance
            else:
                sl_price = entry_price + sl_distance
        
        sl_distance_pct = (sl_distance / entry_price) * 100
        
        return StopLossResult(
            price=round(sl_price, 6),
            distance=round(sl_distance, 6),
            distance_pct=round(sl_distance_pct, 4),
            atr_multiplier=multiplier
        )
    
    def calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        sl_distance: float,
        risk_reward: float = None
    ) -> TakeProfitResult:
        
        # A YOLU DÜZELTMESİ: RR Kesinlikle 1.5 olacak. Esnetme iptal edildi.
        config_rr = 1.5 
        rr = risk_reward or config_rr
        rr = max(rr, 1.5) # Asla 1.5'in altına inme!
        
        tp_distance = sl_distance * rr
        
        if direction.upper() == 'LONG':
            tp_price = entry_price + tp_distance   
        else:  
            tp_price = entry_price - tp_distance   
        
        tp_distance_pct = (tp_distance / entry_price) * 100
        
        return TakeProfitResult(
            price=round(tp_price, 6),
            distance=round(tp_distance, 6),
            distance_pct=round(tp_distance_pct, 4),
            risk_reward=rr
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        sl_distance: float,
        min_amount: float = 0.001,
        amount_precision: int = 3,
        contract_size: float = 1.0
    ):
        import math
        
        RISK_PERCENT = 0.02          
        MAX_TOTAL_MARGIN = 0.65      
        MAX_TRADES = 10               
        MAX_LEVERAGE = 20.0          
        
        max_margin_per_trade = (self.balance * MAX_TOTAL_MARGIN) / MAX_TRADES
        risk_amount = self.balance * RISK_PERCENT
        
        if sl_distance <= 0:
            sl_distance = entry_price * 0.01  
            
        ideal_size = risk_amount / sl_distance
        ideal_pos_value = ideal_size * entry_price
        calculated_lev = ideal_pos_value / max_margin_per_trade if max_margin_per_trade > 0 else 1
        
        if calculated_lev > MAX_LEVERAGE:
            leverage = int(MAX_LEVERAGE)
            max_allowed_value = max_margin_per_trade * leverage
            final_size = max_allowed_value / entry_price
            margin_used = max_margin_per_trade
        else:
            leverage = int(max(1, math.ceil(calculated_lev)))
            final_size = ideal_size
            margin_used = ideal_pos_value / leverage
            
        if contract_size == 1.0 and ideal_size < 1.0:
            final_size = ideal_size
        else:
            final_size = math.floor(final_size / contract_size) * contract_size
            
        final_size = round(final_size, amount_precision)
        if final_size < min_amount:
            final_size = min_amount

        return PositionSizeResult(
            size=final_size,
            value=final_size * entry_price,
            risk_amount=risk_amount,
            margin_required=margin_used,
            leverage=leverage
        )
    
    def check_position_limit(self) -> Tuple[bool, str]:
        max_pos = self.risk_cfg.max_open_positions
        if self.open_positions >= max_pos:
            return False, f"Max açık pozisyon ({max_pos}) aşıldı ({self.open_positions})"
        return True, "OK"
    
    def check_margin_available(self, margin_required: float) -> Tuple[bool, str]:
        max_per_trade = self.balance * (self.risk_cfg.max_margin_per_trade_pct / 100)
        if margin_required > max_per_trade:
            return False, f"Margin > max/işlem"
        
        total_margin = self.used_margin + margin_required
        max_total = self.balance * (self.risk_cfg.max_total_margin_pct / 100)
        if total_margin > max_total:
            return False, f"Toplam margin > max toplam"
        
        return True, "OK"
    
    def check_daily_loss_limit(self) -> Tuple[bool, str]:
        if self.daily_pnl >= 0:
            return True, "OK"                  
        
        daily_loss = abs(self.daily_pnl)
        max_daily_loss = self.balance * (self.risk_cfg.daily_max_loss_pct / 100)
        
        if daily_loss >= max_daily_loss:
            return False, f"Günlük kayıp limiti aşıldı"
        
        if daily_loss >= max_daily_loss * 0.80:
            return True, f"⚠️ Günlük kayıp limite yaklaşıyor"
        
        return True, "OK"
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        if self.initial_balance <= 0:
            return True, "OK"
        
        current_equity = self.balance + self.daily_pnl
        drawdown_pct = ((self.initial_balance - current_equity) / self.initial_balance) * 100
        
        if drawdown_pct >= self.risk_cfg.kill_switch_drawdown_pct:
            return False, f"🚨 KILL SWITCH! Drawdown %{drawdown_pct:.1f}"
        
        warning_threshold = self.risk_cfg.kill_switch_drawdown_pct * 0.70
        if drawdown_pct >= warning_threshold:
            return True, f"⚠️ Drawdown %{drawdown_pct:.1f} (Kill switch yaklaşıyor)"
        
        return True, "OK"
    
    def update_balance(self, new_balance: float):
        self.balance = float(new_balance)
    
    def check_risk_reward(self, sl_distance: float, tp_distance: float) -> Tuple[bool, str]:
        if sl_distance <= 0:
            return False, "SL distance <= 0"
        
        rr = tp_distance / sl_distance
        min_allowed_rr = 1.5  # A YOLU: Minimum RR artık kesinlikle 1.5
        
        if rr < min_allowed_rr - 0.001:  
            return False, f"RR ({rr:.2f}) < min ({min_allowed_rr})"
        
        return True, "OK"
    
    def calculate_trade(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        symbol: str = "UNKNOWN",
        atr_multiplier: float = None,
        risk_reward: float = None,
        min_amount: float = 0.001,
        amount_precision: int = 3,
        contract_size: float = 1.0
    ) -> TradeCalculation:
        direction = direction.upper()
        checks = {}
        rejection_reasons = []
        warnings = []
        
        logger.info(f"📊 {symbol} {direction} trade hesaplanıyor...")
        logger.info(f"   Entry: ${entry_price:,.2f} | ATR: ${atr:,.4f} | Bakiye: ${self.balance:,.2f}")
        
        sl = self.calculate_stop_loss(entry_price, direction, atr, atr_multiplier)
        tp = self.calculate_take_profit(entry_price, direction, sl.distance, risk_reward)
        pos = self.calculate_position_size(entry_price, sl.distance, min_amount, amount_precision, contract_size)
        
        passed, msg = self.check_position_limit()
        checks['position_limit'] = passed
        if not passed: rejection_reasons.append(msg)
        
        passed, msg = self.check_margin_available(pos.margin_required)
        checks['margin_available'] = passed
        if not passed: rejection_reasons.append(msg)
        elif msg != "OK": warnings.append(msg)
        
        passed, msg = self.check_daily_loss_limit()
        checks['daily_loss'] = passed
        if not passed: rejection_reasons.append(msg)
        elif msg != "OK": warnings.append(msg)
        
        passed, msg = self.check_kill_switch()
        checks['kill_switch'] = passed
        if not passed: rejection_reasons.append(msg)
        elif msg != "OK": warnings.append(msg)
        
        passed, msg = self.check_risk_reward(sl.distance, tp.distance)
        checks['risk_reward'] = passed
        if not passed: rejection_reasons.append(msg)
        
        checks['position_size'] = pos.size > 0
        if pos.size <= 0:
            rejection_reasons.append("Pozisyon büyüklüğü 0 (bakiye yetersiz)")
        
        if rejection_reasons:
            status = RiskCheckStatus.REJECTED
        elif warnings:
            status = RiskCheckStatus.WARNING
        else:
            status = RiskCheckStatus.APPROVED
        
        result = TradeCalculation(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            position=pos,
            status=status,
            checks=checks,
            rejection_reasons=rejection_reasons,
            warnings=warnings
        )
        
        if result.is_approved():
            logger.info(f"   ✅ ONAYLANDI: {pos.size:.4f} @ ${entry_price:,.2f}")
        elif status == RiskCheckStatus.WARNING:
            logger.warning(f"   ⚠️ UYARI İLE ONAY: {warnings}")
        else:
            logger.warning(f"   ❌ REDDEDİLDİ: {rejection_reasons}")
        
        return result