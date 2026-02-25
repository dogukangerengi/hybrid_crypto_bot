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
#
# Hard Limitler (config.py → RiskConfig):
# - Max risk/işlem: %2
# - Max açık pozisyon: 2
# - Max margin/işlem: bakiyenin %25'i
# - Max toplam margin: bakiyenin %60'ı
# - Min RR: 1.5
# - Günlük max kayıp: %6
# - Kill switch: %15 toplam DD
#
# Kullanım:
# --------
# from execution.risk_manager import RiskManager
# rm = RiskManager(balance=75.0)
# result = rm.calculate_trade(
#     entry=185.00, direction='SHORT',
#     atr=3.70, current_price=185.00
# )
# =============================================================================

import sys                                     # Path ayarları
import logging                                 # Log yönetimi
import math                                    # Yuvarlama fonksiyonları
from pathlib import Path                       # Platform-bağımsız dosya yolları
from typing import Dict, List, Optional, Tuple # Tip belirteçleri
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfı
from datetime import datetime, timezone, date  # Zaman damgası
from enum import Enum                          # Sabit değer enumları

# Proje config import
sys.path.insert(0, str(Path(__file__).parent.parent))  # → src/
from config import cfg                         # Merkezi config (RiskConfig dahil)

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUM & DATACLASS TANIMLARI
# =============================================================================

class TradeDirection(Enum):
    """İşlem yönü."""
    LONG = "LONG"                              # Uzun pozisyon (fiyat artışı beklenir)
    SHORT = "SHORT"                            # Kısa pozisyon (fiyat düşüşü beklenir)


class RiskCheckStatus(Enum):
    """Risk kontrol sonucu."""
    APPROVED = "APPROVED"                      # ✅ İşlem onaylandı
    REJECTED = "REJECTED"                      # ❌ İşlem reddedildi
    WARNING = "WARNING"                        # ⚠️ Uyarı ile onay


@dataclass
class StopLossResult:
    """
    Stop-Loss hesaplama sonucu.
    
    ATR bazlı SL mesafesi:
    - LONG:  SL = Entry - (ATR × multiplier)
    - SHORT: SL = Entry + (ATR × multiplier)
    """
    price: float                               # SL fiyatı ($)
    distance: float                            # Entry'den uzaklık ($)
    distance_pct: float                        # Entry'den uzaklık (%)
    atr_multiplier: float                      # Kullanılan ATR çarpanı


@dataclass
class TakeProfitResult:
    """
    Take-Profit hesaplama sonucu.
    
    RR bazlı TP:
    - TP_distance = SL_distance × RR_ratio
    - LONG:  TP = Entry + TP_distance
    - SHORT: TP = Entry - TP_distance
    """
    price: float                               # TP fiyatı ($)
    distance: float                            # Entry'den uzaklık ($)
    distance_pct: float                        # Entry'den uzaklık (%)
    risk_reward: float                         # Kullanılan RR oranı


@dataclass
class PositionSizeResult:
    """
    Pozisyon büyüklüğü hesaplama sonucu.
    
    Formül: size = risk_amount / sl_distance
    Sonra lot büyüklüğüne yuvarlanır.
    """
    size: float                                # Pozisyon büyüklüğü (coin adedi)
    value: float                               # Pozisyon değeri ($)
    risk_amount: float                         # Risk edilen miktar ($)
    margin_required: float                     # Gereken margin ($)
    leverage: int                              # Hesaplanan kaldıraç


@dataclass
class TradeCalculation:
    """
    Tek bir işlem için tüm risk hesaplamalarının sonucu.
    
    validate_trade() fonksiyonu bu objeyi döndürür.
    Execution modülü bunu kullanarak emir gönderir.
    """
    # İşlem parametreleri
    symbol: str                                # İşlem çifti
    direction: str                             # LONG veya SHORT
    entry_price: float                         # Giriş fiyatı ($)
    
    # SL/TP
    stop_loss: StopLossResult                  # Stop-Loss detayları
    take_profit: TakeProfitResult              # Take-Profit detayları
    
    # Pozisyon
    position: PositionSizeResult               # Pozisyon büyüklüğü detayları
    
    # Risk kontrol
    status: RiskCheckStatus                    # Onay durumu
    checks: Dict[str, bool] = field(default_factory=dict)  # Kontrol sonuçları
    rejection_reasons: List[str] = field(default_factory=list)  # Red nedenleri
    warnings: List[str] = field(default_factory=list)          # Uyarılar
    
    def is_approved(self) -> bool:
        """İşlem onaylı mı?"""
        return self.status == RiskCheckStatus.APPROVED
    
    def summary(self) -> str:
        """İşlem özetini döndürür (Telegram mesajı için)."""
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
    """
    Pozisyon büyüklüğü, kaldıraç ve risk limitleri yönetimi.
    
    Config.py'deki RiskConfig parametrelerini kullanır.
    Her işlem öncesi validate_trade() çağrılarak onay alınır.
    
    İstatistiksel Gerekçe:
    --------------------
    - ATR bazlı SL: Volatiliteye adaptif → farklı market rejimlerde
      pozisyon boyutu otomatik ayarlanır (Kelly criterion benzeri)
    - Fixed fractional risk (%2): Geometric growth optimal'e yakın
      (Kelly'nin yarısı — conservative approach)
    - RR ≥ 1.5: Win rate %40'ta bile pozitif beklenen değer sağlar
      E[R] = WR × TP - (1-WR) × SL > 0
      0.40 × 1.5 - 0.60 × 1.0 = 0.0 (breakeven)
      0.45 × 1.5 - 0.55 × 1.0 = +0.125 (profitable)
    """
    
    # =========================================================================
    # ATR ÇARPANLARI
    # =========================================================================
    # SL mesafesi = ATR × multiplier
    # Conservative: 2.0x (daha az whipsaw ama daha büyük kayıp)
    # Aggressive: 1.0x (sıkı stop ama daha çok false trigger)
    # Default: 1.5x (denge noktası)
    
    DEFAULT_ATR_MULTIPLIER = 1.5               # Varsayılan SL ATR çarpanı
    MIN_ATR_MULTIPLIER = 1.0                   # Minimum ATR çarpanı
    MAX_ATR_MULTIPLIER = 3.0                   # Maksimum ATR çarpanı
    
    def __init__(
        self,
        balance: float = 0.0,
        used_margin: float = 0.0,
        open_positions: int = 0,
        daily_pnl: float = 0.0,
        initial_balance: Optional[float] = None
    ):
        """
        RiskManager başlatır.
        
        Parameters:
        ----------
        balance : float
            Mevcut kullanılabilir USDT bakiye.
            Bitget API'den fetch_balance() ile alınır.
            
        used_margin : float
            Halihazırda kullanılan margin ($).
            Açık pozisyonlar tarafından bloke edilen miktar.
            
        open_positions : int
            Şu an açık olan pozisyon sayısı.
            
        daily_pnl : float
            Bugünkü toplam PnL ($). Negatif = kayıp.
            Günlük kayıp limitini kontrol etmek için.
            
        initial_balance : float, optional
            Başlangıç bakiyesi ($). Kill switch DD hesabı için.
            None ise balance kullanılır.
        """
        self.balance = balance
        self.used_margin = used_margin
        self.open_positions = open_positions
        self.daily_pnl = daily_pnl
        self.initial_balance = initial_balance or balance
        
        # Config'den risk parametreleri
        self.risk_cfg = cfg.risk
        
        logger.info(
            f"RiskManager başlatıldı | "
            f"Bakiye: ${balance:,.2f} | "
            f"Açık poz: {open_positions} | "
            f"Günlük PnL: ${daily_pnl:+,.2f}"
        )
    
    # =========================================================================
    # BAKİYE GÜNCELLEME
    # =========================================================================
    
    def update_state(
        self,
        balance: Optional[float] = None,
        used_margin: Optional[float] = None,
        open_positions: Optional[int] = None,
        daily_pnl: Optional[float] = None
    ) -> None:
        """
        Risk yöneticisinin durumunu günceller.
        
        Her trade cycle'da çağrılarak bakiye, margin ve PnL
        güncel tutulur. Bitget API'den alınan verilerle beslenir.
        """
        if balance is not None:
            self.balance = balance
        if used_margin is not None:
            self.used_margin = used_margin
        if open_positions is not None:
            self.open_positions = open_positions
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
    
    # =========================================================================
    # STOP-LOSS HESAPLAMA (ATR BAZLI)
    # =========================================================================
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = None
    ) -> StopLossResult:
        """
        ATR bazlı Stop-Loss hesaplar.
        
        Formül:
        ------
        SL_distance = ATR × multiplier
        LONG:  SL = Entry - SL_distance
        SHORT: SL = Entry + SL_distance
        
        Neden ATR bazlı?
        → Volatiliteye adaptif: yüksek vol → geniş SL → küçük pozisyon
        → Market rejimi değiştiğinde otomatik ayarlanır
        → Fixed pip SL'ye göre çok daha robust
        
        Parameters:
        ----------
        entry_price : float
            Giriş fiyatı ($)
        direction : str
            'LONG' veya 'SHORT'
        atr : float
            ATR değeri ($) — mum bazlı volatilite ölçüsü
        atr_multiplier : float, optional
            ATR çarpanı (1.0-3.0). None ise DEFAULT kullanılır.
            
        Returns:
        -------
        StopLossResult
            SL fiyatı, mesafe ($), mesafe (%), çarpan
        """
        # ATR çarpanı sınırla
        multiplier = atr_multiplier or self.DEFAULT_ATR_MULTIPLIER
        multiplier = max(self.MIN_ATR_MULTIPLIER,
                         min(multiplier, self.MAX_ATR_MULTIPLIER))
        
        # SL mesafesi ($)
        sl_distance = atr * multiplier
        
        # Yön bazlı SL fiyatı
        if direction.upper() == 'LONG':
            sl_price = entry_price - sl_distance   # Altına düşerse kaybet
        else:  # SHORT
            sl_price = entry_price + sl_distance   # Üstüne çıkarsa kaybet
        
        # ── Max SL% Cap ── 
        # NAORIS gibi düşük fiyatlı coinlerde ATR-bazlı SL %20 olabiliyor
        # %2 risk formülü pozisyonu küçültse de, SL çok geniş olunca
        # tek bir spike tüm riski realize eder. Cap ile sınırla.
        max_sl_pct = getattr(self.risk_cfg, 'max_sl_pct', 8.0)
        sl_pct_raw = (sl_distance / entry_price) * 100
        if sl_pct_raw > max_sl_pct:
            logger.warning(
                f"   ⚠️ SL mesafesi daraltıldı: {sl_pct_raw:.1f}% → {max_sl_pct}%"
            )
            sl_distance = entry_price * max_sl_pct / 100
            if direction == "LONG":
                sl_price = entry_price - sl_distance
            else:
                sl_price = entry_price + sl_distance
        
        # SL mesafesi yüzde
        sl_distance_pct = (sl_distance / entry_price) * 100
        
        return StopLossResult(
            price=round(sl_price, 6),
            distance=round(sl_distance, 6),
            distance_pct=round(sl_distance_pct, 4),
            atr_multiplier=multiplier
        )
    
    # =========================================================================
    # TAKE-PROFIT HESAPLAMA (RR BAZLI)
    # =========================================================================
    
    def calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        sl_distance: float,
        risk_reward: float = None
    ) -> TakeProfitResult:
        """
        Risk/Reward oranına göre Take-Profit hesaplar.
        
        Formül:
        ------
        TP_distance = SL_distance × RR_ratio
        LONG:  TP = Entry + TP_distance
        SHORT: TP = Entry - TP_distance
        
        Beklenen Değer:
        E[R] = WR × RR - (1 - WR)
        RR = 1.5, WR = 0.45 → E[R] = 0.45 × 1.5 - 0.55 = +0.125 (kârlı)
        RR = 1.5, WR = 0.40 → E[R] = 0.40 × 1.5 - 0.60 = 0.0 (breakeven)
        
        Parameters:
        ----------
        entry_price : float
            Giriş fiyatı ($)
        direction : str
            'LONG' veya 'SHORT'
        sl_distance : float
            Stop-Loss mesafesi ($) — SL hesaplamasından gelir
        risk_reward : float, optional
            RR oranı. None ise config'den okunur (min 1.5)
            
        Returns:
        -------
        TakeProfitResult
            TP fiyatı, mesafe ($), mesafe (%), RR oranı
        """
        # RR oranı (config minimum'dan düşük olamaz)
        rr = risk_reward or self.risk_cfg.min_risk_reward_ratio
        rr = max(rr, self.risk_cfg.min_risk_reward_ratio)
        
        # TP mesafesi ($)
        tp_distance = sl_distance * rr
        
        # Yön bazlı TP fiyatı
        if direction.upper() == 'LONG':
            tp_price = entry_price + tp_distance   # Üstüne çıkarsa kazan
        else:  # SHORT
            tp_price = entry_price - tp_distance   # Altına düşerse kazan
        
        # TP mesafesi yüzde
        tp_distance_pct = (tp_distance / entry_price) * 100
        
        return TakeProfitResult(
            price=round(tp_price, 6),
            distance=round(tp_distance, 6),
            distance_pct=round(tp_distance_pct, 4),
            risk_reward=rr
        )
    
    # =========================================================================
    # POZİSYON BÜYÜKLÜĞÜ HESAPLAMA
    # =========================================================================
    
    def calculate_position_size(
        self,
        entry_price: float,
        sl_distance: float,
        min_amount: float = 0.001,
        amount_precision: int = 3,
        contract_size: float = 1.0
    ):
        """
        Kasa Yönetimi: %2 Risk, %65 Kullanım, %35 Nakit (Max 5 İşlem, Max 20x Kaldıraç)
        """
        import math
        
        # --- STRATEJİ KURALLARI ---
        RISK_PERCENT = 0.02          # Stop olduğunda kasanın en fazla %2'si gitsin
        MAX_TOTAL_MARGIN = 0.65      # Kasanın sadece %65'i işlemlerde kullanılabilir
        MAX_TRADES = 5               # Maksimum 5 işlem hedefleniyor
        MAX_LEVERAGE = 20.0          # İşlem başına maksimum 20x kaldıraç
        
        # 1. İşlem Başına Düşen Maksimum Marjin Limiti
        # Örn: 1000$ kasanın %65'i = 650$. 5 işleme bölersek işlem başı nakit = 130$
        max_margin_per_trade = (self.balance * MAX_TOTAL_MARGIN) / MAX_TRADES
        
        # 2. Riske Edilecek Miktar
        # Örn: 1000$ * 0.02 = 20$
        risk_amount = self.balance * RISK_PERCENT
        
        # 3. İdeal Lot (Coin Adedi) Hesaplama
        if sl_distance <= 0:
            sl_distance = entry_price * 0.01  # Güvenlik (Sıfıra bölmeyi önler)
            
        ideal_size = risk_amount / sl_distance
        ideal_pos_value = ideal_size * entry_price
        
        # 4. İhtiyaç Duyulan Kaldıraç (Bu işlemi kendi bütçemizle açmak için gereken X)
        calculated_lev = ideal_pos_value / max_margin_per_trade if max_margin_per_trade > 0 else 1
        
        # 5. Sınırları Uygula ve Optimize Et
        if calculated_lev > MAX_LEVERAGE:
            # Eğer stop çok darsa ve 20x'i aşıyorsa, limiti 20x'e çakıp lotu küçült (Risk %2'nin de altına düşer, çok güvenli)
            leverage = int(MAX_LEVERAGE)
            max_allowed_value = max_margin_per_trade * leverage
            final_size = max_allowed_value / entry_price
            margin_used = max_margin_per_trade
        else:
            # Kaldıraç 20x'in altındaysa, tam %2 riskle ideale göre gir ve kaldıracı tavana yuvarla
            leverage = int(max(1, math.ceil(calculated_lev)))
            final_size = ideal_size
            margin_used = ideal_pos_value / leverage
            
        # 6. Borsa Küsurat Kurallarına Göre Yuvarla
        final_size = round(final_size / contract_size) * contract_size
        final_size = round(final_size, amount_precision)
        
        if final_size < min_amount:
            final_size = min_amount

        # Sonucu var olan objeyle döndür
        from execution.risk_manager import PositionCalculation
        
        return PositionCalculation(
            size=final_size,
            margin_required=margin_used,
            leverage=leverage
        )
    
    # =========================================================================
    # RİSK KONTROLLERİ
    # =========================================================================
    
    def check_position_limit(self) -> Tuple[bool, str]:
        """
        Max açık pozisyon limitini kontrol eder.
        
        Returns: (geçti_mi, mesaj)
        """
        max_pos = self.risk_cfg.max_open_positions
        if self.open_positions >= max_pos:
            return False, f"Max açık pozisyon ({max_pos}) aşıldı ({self.open_positions})"
        return True, "OK"
    
    def check_margin_available(self, margin_required: float) -> Tuple[bool, str]:
        """
        Yeni işlem için yeterli margin var mı kontrol eder.
        
        Kontroller:
        1. İşlem margin'i ≤ bakiyenin max_margin_per_trade_pct'si
        2. Toplam margin (mevcut + yeni) ≤ bakiyenin max_total_margin_pct'si
        """
        # İşlem başına max margin
        max_per_trade = self.balance * (self.risk_cfg.max_margin_per_trade_pct / 100)
        if margin_required > max_per_trade:
            return False, (
                f"Margin (${margin_required:,.2f}) > "
                f"max/işlem (${max_per_trade:,.2f}, %{self.risk_cfg.max_margin_per_trade_pct})"
            )
        
        # Toplam margin kontrolü
        total_margin = self.used_margin + margin_required
        max_total = self.balance * (self.risk_cfg.max_total_margin_pct / 100)
        if total_margin > max_total:
            return False, (
                f"Toplam margin (${total_margin:,.2f}) > "
                f"max toplam (${max_total:,.2f}, %{self.risk_cfg.max_total_margin_pct})"
            )
        
        return True, "OK"
    
    def check_daily_loss_limit(self) -> Tuple[bool, str]:
        """
        Günlük kayıp limitini kontrol eder.
        
        Günlük kayıp = |daily_pnl| (negatif ise)
        Limit = bakiye × daily_max_loss_pct / 100
        """
        if self.daily_pnl >= 0:
            return True, "OK"                  # Kârdayız, sorun yok
        
        daily_loss = abs(self.daily_pnl)
        max_daily_loss = self.balance * (self.risk_cfg.daily_max_loss_pct / 100)
        
        if daily_loss >= max_daily_loss:
            return False, (
                f"Günlük kayıp (${daily_loss:,.2f}) ≥ "
                f"limit (${max_daily_loss:,.2f}, %{self.risk_cfg.daily_max_loss_pct})"
            )
        
        # Uyarı: %80'ine yaklaştıysa
        if daily_loss >= max_daily_loss * 0.80:
            return True, (
                f"⚠️ Günlük kayıp limite yaklaşıyor: "
                f"${daily_loss:,.2f} / ${max_daily_loss:,.2f}"
            )
        
        return True, "OK"
    
    def check_kill_switch(self) -> Tuple[bool, str]:
        """
        Kill switch kontrolü — toplam drawdown limiti.
        
        DD = (initial_balance - current_balance) / initial_balance × 100
        DD ≥ kill_switch_pct → SİSTEMİ DURDUR
        """
        if self.initial_balance <= 0:
            return True, "OK"
        
        current_equity = self.balance + self.daily_pnl
        drawdown_pct = ((self.initial_balance - current_equity) / self.initial_balance) * 100
        
        if drawdown_pct >= self.risk_cfg.kill_switch_drawdown_pct:
            return False, (
                f"🚨 KILL SWITCH! Drawdown %{drawdown_pct:.1f} ≥ "
                f"limit %{self.risk_cfg.kill_switch_drawdown_pct}"
            )
        
        # Uyarı: %70'ine yaklaştıysa
        warning_threshold = self.risk_cfg.kill_switch_drawdown_pct * 0.70
        if drawdown_pct >= warning_threshold:
            return True, (
                f"⚠️ Drawdown %{drawdown_pct:.1f} — "
                f"kill switch %{self.risk_cfg.kill_switch_drawdown_pct}'de"
            )
        
        return True, "OK"
    
    def update_balance(self, new_balance: float):
        """
        Dış kaynaklardan (PaperTrader veya API) gelen güncel bakiyeyi işler.
        Pozisyon büyüklüğü hesaplamalarının güncel sermaye ile yapılmasını sağlar.
        """
        self.balance = float(new_balance)
    
    def check_risk_reward(self, sl_distance: float, tp_distance: float) -> Tuple[bool, str]:
        """
        Risk/Reward oranını kontrol eder.
        
        RR = TP_distance / SL_distance
        RR < min_rr → işlem reddedilir
        """
        if sl_distance <= 0:
            return False, "SL distance <= 0"
        
        rr = tp_distance / sl_distance
        
        if rr < self.risk_cfg.min_risk_reward_ratio - 0.001:  # Float tolerance
            return False, (
                f"RR ({rr:.2f}) < min ({self.risk_cfg.min_risk_reward_ratio})"
            )
        
        return True, "OK"
    
    # =========================================================================
    # ANA İŞLEM DOĞRULAMA FONKSİYONU
    # =========================================================================
    
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
        """
        Tam işlem hesaplaması + tüm risk kontrollerini çalıştırır.
        
        Pipeline:
        1. Stop-Loss hesapla (ATR bazlı)
        2. Take-Profit hesapla (RR bazlı)
        3. Pozisyon büyüklüğü hesapla
        4. Risk kontrolleri çalıştır
        5. Onay/Red kararı ver
        
        Parameters:
        ----------
        entry_price : float
            Giriş fiyatı ($)
        direction : str
            'LONG' veya 'SHORT'
        atr : float
            ATR değeri ($) — indikatör katmanından gelir
        symbol : str
            İşlem çifti (log ve Telegram için)
        atr_multiplier : float, optional
            SL için ATR çarpanı (1.0-3.0)
        risk_reward : float, optional
            TP için RR oranı (min 1.5)
        min_amount : float
            Borsa minimum sipariş miktarı
        amount_precision : int
            Borsa miktar hassasiyeti
        contract_size : float
            Kontrat büyüklüğü
            
        Returns:
        -------
        TradeCalculation
            Tüm hesaplamalar + risk kontrol sonuçları.
            .is_approved() ile onay kontrol edilir.
        """
        direction = direction.upper()
        checks = {}
        rejection_reasons = []
        warnings = []
        
        logger.info(f"📊 {symbol} {direction} trade hesaplanıyor...")
        logger.info(f"   Entry: ${entry_price:,.2f} | ATR: ${atr:,.4f} | "
                     f"Bakiye: ${self.balance:,.2f}")
        
        # ---- 1. STOP-LOSS ----
        sl = self.calculate_stop_loss(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            atr_multiplier=atr_multiplier
        )
        
        # ---- 2. TAKE-PROFIT ----
        tp = self.calculate_take_profit(
            entry_price=entry_price,
            direction=direction,
            sl_distance=sl.distance,
            risk_reward=risk_reward
        )
        
        # ---- 3. POZİSYON BÜYÜKLÜĞÜ ----
        pos = self.calculate_position_size(
            entry_price=entry_price,
            sl_distance=sl.distance,
            min_amount=min_amount,
            amount_precision=amount_precision,
            contract_size=contract_size
        )
        
        # ---- 4. RİSK KONTROLLERİ ----
        
        # 4a. Pozisyon limiti
        passed, msg = self.check_position_limit()
        checks['position_limit'] = passed
        if not passed:
            rejection_reasons.append(msg)
        
        # 4b. Margin yeterliliği
        passed, msg = self.check_margin_available(pos.margin_required)
        checks['margin_available'] = passed
        if not passed:
            rejection_reasons.append(msg)
        elif msg != "OK":
            warnings.append(msg)
        
        # 4c. Günlük kayıp limiti
        passed, msg = self.check_daily_loss_limit()
        checks['daily_loss'] = passed
        if not passed:
            rejection_reasons.append(msg)
        elif msg != "OK":
            warnings.append(msg)
        
        # 4d. Kill switch
        passed, msg = self.check_kill_switch()
        checks['kill_switch'] = passed
        if not passed:
            rejection_reasons.append(msg)
        elif msg != "OK":
            warnings.append(msg)
        
        # 4e. Risk/Reward
        passed, msg = self.check_risk_reward(sl.distance, tp.distance)
        checks['risk_reward'] = passed
        if not passed:
            rejection_reasons.append(msg)
        
        # 4f. Pozisyon büyüklüğü > 0 mı?
        checks['position_size'] = pos.size > 0
        if pos.size <= 0:
            rejection_reasons.append("Pozisyon büyüklüğü 0 (bakiye yetersiz)")
        
        # ---- 5. KARAR ----
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
        
        # Log
        if result.is_approved():
            logger.info(f"   ✅ ONAYLANDI: {pos.size:.4f} @ ${entry_price:,.2f}")
        elif status == RiskCheckStatus.WARNING:
            logger.warning(f"   ⚠️ UYARI İLE ONAY: {warnings}")
        else:
            logger.warning(f"   ❌ REDDEDİLDİ: {rejection_reasons}")
        
        return result


# =============================================================================
# BAĞIMSIZ ÇALIŞTIRMA TESTİ
# =============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 65)
    print("  ⚖️ RİSK YÖNETİMİ MOTORU — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # $75 bakiye senaryosu (Roadmap'teki örnek)
    rm = RiskManager(balance=75.0, initial_balance=75.0)
    
    # Senaryo 1: SOL SHORT (Roadmap'teki hesap)
    print("\n[1] SOL/USDT SHORT — Roadmap Senaryosu:")
    trade = rm.calculate_trade(
        entry_price=185.00,
        direction='SHORT',
        atr=3.70,                              # ATR = $3.70
        symbol='SOL/USDT:USDT',
        atr_multiplier=1.0,                    # 1x ATR → SL @ $188.70
        risk_reward=1.5
    )
    print(trade.summary())
    
    # Senaryo 2: BTC LONG
    print("\n[2] BTC/USDT LONG:")
    trade2 = rm.calculate_trade(
        entry_price=97000.00,
        direction='LONG',
        atr=1500.0,                            # ATR = $1500
        symbol='BTC/USDT:USDT',
        atr_multiplier=1.5,
        risk_reward=2.0
    )
    print(trade2.summary())
    
    # Senaryo 3: Günlük kayıp limiti test
    print("\n[3] Günlük kayıp limiti testi:")
    rm_loss = RiskManager(
        balance=75.0, 
        daily_pnl=-4.0,                        # $4 kayıp (bakiyenin %5.3'ü)
        initial_balance=75.0
    )
    trade3 = rm_loss.calculate_trade(
        entry_price=185.00, direction='LONG',
        atr=3.70, symbol='SOL/USDT:USDT'
    )
    print(f"   Günlük kayıp: $4.00 | Limit: ${75 * 0.06:.2f}")
    print(f"   Durum: {trade3.status.value}")
    
    # Senaryo 4: Kill switch test
    print("\n[4] Kill switch testi:")
    rm_dd = RiskManager(
        balance=60.0,                          # $75 → $60 (%20 DD)
        initial_balance=75.0
    )
    trade4 = rm_dd.calculate_trade(
        entry_price=185.00, direction='SHORT',
        atr=3.70, symbol='SOL/USDT:USDT'
    )
    print(f"   DD: %{((75-60)/75)*100:.1f} | Kill switch: %15")
    print(f"   Durum: {trade4.status.value}")
    if trade4.rejection_reasons:
        print(f"   Neden: {trade4.rejection_reasons[0]}")
    
    print(f"\n{'=' * 65}")
    print(f"  ✅ TEST TAMAMLANDI")
    print(f"{'=' * 65}")
