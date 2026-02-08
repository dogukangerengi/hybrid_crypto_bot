# =============================================================================
# ADIM 5: RİSK YÖNETİMİ MOTORU TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_risk_manager.py
#
# API GEREKTİRMEZ — Tüm testler sentetik bakiye/fiyat verileriyle çalışır.
# Tüm testler geçerse Adım 5 tamamdır.
#
# Test Listesi:
# 1.  SL: ATR bazlı Stop-Loss hesaplama (LONG + SHORT)
# 2.  TP: RR bazlı Take-Profit hesaplama (LONG + SHORT)
# 3.  Position Size: Fixed fractional pozisyon büyüklüğü
# 4.  Leverage: Otomatik kaldıraç hesaplama + config limitleri
# 5.  Risk Checks: Pozisyon limiti, margin, günlük kayıp
# 6.  Kill Switch: Drawdown bazlı sistem durdurma
# 7.  Full Trade: Tam pipeline (SL → TP → Size → Checks → Karar)
# 8.  Roadmap Scenario: $75 bakiye SOL SHORT (PROJECT_ROADMAP.md örneği)
# 9.  Edge Cases: Sıfır bakiye, çok yüksek ATR, min amount
# 10. State Update: Bakiye güncelleme ve tekrar kontrol
# =============================================================================

import sys
import time
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# Path ayarı (src/ altından çalışır)
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Loglama
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
warnings.filterwarnings('ignore')


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_test(test_num: int, test_name: str, test_func) -> bool:
    """Tek testi çalıştır, süre ölç, hata yakala."""
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num}: {test_name}")
    print(f"{'─' * 55}")
    
    start = time.time()
    try:
        test_func()
        elapsed = time.time() - start
        print(f"\n  ✅ BAŞARILI ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ BAŞARISIZ ({elapsed:.2f}s)")
        print(f"     Hata: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TEST 1: ATR BAZLI STOP-LOSS
# =============================================================================

def test_01_stop_loss():
    """ATR bazlı SL doğru hesaplanıyor mu? (LONG + SHORT)"""
    from execution.risk_manager import RiskManager
    
    rm = RiskManager(balance=100.0)
    
    # LONG: SL = Entry - (ATR × multiplier) = 100 - (5 × 1.5) = 92.5
    sl_long = rm.calculate_stop_loss(
        entry_price=100.0, direction='LONG',
        atr=5.0, atr_multiplier=1.5
    )
    assert abs(sl_long.price - 92.5) < 0.01, \
        f"LONG SL hatalı: {sl_long.price} != 92.5"
    assert abs(sl_long.distance - 7.5) < 0.01, \
        f"LONG SL distance hatalı: {sl_long.distance} != 7.5"
    assert abs(sl_long.distance_pct - 7.5) < 0.01, \
        f"LONG SL pct hatalı: {sl_long.distance_pct} != 7.5"
    
    print(f"  LONG:  Entry=$100 | ATR=$5 × 1.5 | SL=${sl_long.price} ✓")
    
    # SHORT: SL = Entry + (ATR × multiplier) = 100 + (5 × 1.5) = 107.5
    sl_short = rm.calculate_stop_loss(
        entry_price=100.0, direction='SHORT',
        atr=5.0, atr_multiplier=1.5
    )
    assert abs(sl_short.price - 107.5) < 0.01, \
        f"SHORT SL hatalı: {sl_short.price} != 107.5"
    
    print(f"  SHORT: Entry=$100 | ATR=$5 × 1.5 | SL=${sl_short.price} ✓")
    
    # ATR multiplier sınırlama (min=1.0, max=3.0)
    sl_clamped = rm.calculate_stop_loss(
        entry_price=100.0, direction='LONG',
        atr=5.0, atr_multiplier=5.0            # 5.0 > max 3.0 → 3.0'a clamp
    )
    assert sl_clamped.atr_multiplier == 3.0, \
        f"ATR multiplier clamp hatalı: {sl_clamped.atr_multiplier} != 3.0"
    
    print(f"  ATR multiplier clamp (5.0 → 3.0): ✓")
    print(f"  ✓ ATR bazlı Stop-Loss doğru")


# =============================================================================
# TEST 2: RR BAZLI TAKE-PROFIT
# =============================================================================

def test_02_take_profit():
    """RR bazlı TP doğru hesaplanıyor mu?"""
    from execution.risk_manager import RiskManager
    
    rm = RiskManager(balance=100.0)
    
    sl_distance = 7.5                          # Önceki testten
    
    # LONG: TP = Entry + (SL_dist × RR) = 100 + (7.5 × 1.5) = 111.25
    tp_long = rm.calculate_take_profit(
        entry_price=100.0, direction='LONG',
        sl_distance=sl_distance, risk_reward=1.5
    )
    assert abs(tp_long.price - 111.25) < 0.01, \
        f"LONG TP hatalı: {tp_long.price} != 111.25"
    assert abs(tp_long.risk_reward - 1.5) < 0.01, \
        f"RR hatalı: {tp_long.risk_reward} != 1.5"
    
    print(f"  LONG:  Entry=$100 | SL_dist=$7.5 | RR=1.5 | TP=${tp_long.price} ✓")
    
    # SHORT: TP = Entry - (SL_dist × RR) = 100 - (7.5 × 2.0) = 85.0
    tp_short = rm.calculate_take_profit(
        entry_price=100.0, direction='SHORT',
        sl_distance=sl_distance, risk_reward=2.0
    )
    assert abs(tp_short.price - 85.0) < 0.01, \
        f"SHORT TP hatalı: {tp_short.price} != 85.0"
    
    print(f"  SHORT: Entry=$100 | SL_dist=$7.5 | RR=2.0 | TP=${tp_short.price} ✓")
    
    # Min RR enforcement: 1.0 verilse bile config min (1.5) uygulanmalı
    tp_min = rm.calculate_take_profit(
        entry_price=100.0, direction='LONG',
        sl_distance=sl_distance, risk_reward=1.0  # < min 1.5
    )
    assert tp_min.risk_reward >= 1.5, \
        f"Min RR enforcement hatalı: {tp_min.risk_reward} < 1.5"
    
    print(f"  Min RR enforcement (1.0 → 1.5): ✓")
    print(f"  ✓ RR bazlı Take-Profit doğru")


# =============================================================================
# TEST 3: POZİSYON BÜYÜKLÜĞÜ (FIXED FRACTIONAL)
# =============================================================================

def test_03_position_size():
    """Fixed fractional pozisyon büyüklüğü doğru hesaplanıyor mu?"""
    from execution.risk_manager import RiskManager
    
    # $100 bakiye, %2 risk = $2 risk/işlem
    rm = RiskManager(balance=100.0)
    
    # Entry=$50, SL_distance=$2 → Size = $2 / $2 = 1.0 coin
    pos = rm.calculate_position_size(
        entry_price=50.0,
        sl_distance=2.0,
        min_amount=0.001,
        amount_precision=3
    )
    
    expected_risk = 100.0 * 0.02               # $2
    assert abs(pos.risk_amount - expected_risk) < 0.01, \
        f"Risk amount hatalı: {pos.risk_amount} != {expected_risk}"
    
    expected_size = expected_risk / 2.0         # 1.0 coin
    assert abs(pos.size - expected_size) < 0.01, \
        f"Position size hatalı: {pos.size} != {expected_size}"
    
    expected_value = expected_size * 50.0       # $50
    assert abs(pos.value - expected_value) < 0.5, \
        f"Position value hatalı: {pos.value} != {expected_value}"
    
    print(f"  Bakiye: $100 | Risk: %2 = ${pos.risk_amount}")
    print(f"  Entry: $50 | SL dist: $2")
    print(f"  Size: {pos.size} coin = ${pos.value} ✓")
    print(f"  Leverage: {pos.leverage}x | Margin: ${pos.margin_required}")
    print(f"  ✓ Position sizing doğru")


# =============================================================================
# TEST 4: KALDIRAC HESAPLAMA
# =============================================================================

def test_04_leverage():
    """Kaldıraç config limitleri içinde kalıyor mu?"""
    from execution.risk_manager import RiskManager
    
    # $75 bakiye
    rm = RiskManager(balance=75.0)
    
    # Küçük pozisyon → düşük kaldıraç
    pos_small = rm.calculate_position_size(
        entry_price=10.0,
        sl_distance=0.5,                       # Risk=$1.5, Size=3 coin, Value=$30
        min_amount=0.01
    )
    # max_margin_per_trade = 75 × 25% = $18.75
    # raw_leverage = 30 / 18.75 = 1.6 → ceil → 2 (min leverage)
    assert pos_small.leverage >= 2, f"Min leverage kontrolü: {pos_small.leverage}"
    assert pos_small.leverage <= 20, f"Max leverage aşıldı: {pos_small.leverage}"
    
    print(f"  Küçük pozisyon: {pos_small.leverage}x (range: 2-20) ✓")
    
    # Büyük pozisyon → yüksek kaldıraç
    pos_big = rm.calculate_position_size(
        entry_price=97000.0,
        sl_distance=1000.0,                    # Risk=$1.5, Size=0.0015 BTC, Value=$145.5
        min_amount=0.0001,
        amount_precision=4
    )
    assert pos_big.leverage >= 2, f"Min leverage: {pos_big.leverage}"
    assert pos_big.leverage <= 20, f"Max leverage: {pos_big.leverage}"
    
    print(f"  Büyük pozisyon: {pos_big.leverage}x (range: 2-20) ✓")
    
    # Margin kontrolü: margin ≤ max_per_trade
    max_margin = 75 * 0.25                     # $18.75
    assert pos_small.margin_required <= max_margin + 1, \
        f"Margin aşıldı: {pos_small.margin_required} > {max_margin}"
    
    print(f"  Margin kontrol: ${pos_small.margin_required:.2f} ≤ ${max_margin:.2f} ✓")
    print(f"  ✓ Kaldıraç hesaplama doğru")


# =============================================================================
# TEST 5: RİSK KONTROLLERİ
# =============================================================================

def test_05_risk_checks():
    """Pozisyon limiti, margin ve günlük kayıp kontrolleri."""
    from execution.risk_manager import RiskManager
    
    # Test 5a: Pozisyon limiti (max 2)
    rm_full = RiskManager(balance=100.0, open_positions=2)
    passed, msg = rm_full.check_position_limit()
    assert not passed, "2/2 pozisyon açıkken yeni açılmamalı"
    print(f"  Pozisyon limiti (2/2): Reddedildi ✓")
    
    # Test 5b: Margin yeterliliği
    rm_margin = RiskManager(balance=100.0, used_margin=55.0)
    # Toplam margin: 55 + 10 = 65 > max_total (100 × 60% = 60) → red
    passed, msg = rm_margin.check_margin_available(10.0)
    assert not passed, f"Toplam margin aşıldı ama geçti: {msg}"
    print(f"  Margin toplam limiti ($55+$10 > $60): Reddedildi ✓")
    
    # Test 5c: Günlük kayıp limiti
    rm_loss = RiskManager(balance=100.0, daily_pnl=-6.0)
    # Günlük kayıp: $6 = %6 (tam limitte) → red
    passed, msg = rm_loss.check_daily_loss_limit()
    assert not passed, f"Günlük kayıp limitinde ama geçti: {msg}"
    print(f"  Günlük kayıp limiti ($6 = %6): Reddedildi ✓")
    
    # Test 5d: Normal durum — tüm kontroller geçmeli
    rm_ok = RiskManager(balance=100.0, open_positions=0, daily_pnl=0.0)
    p1, _ = rm_ok.check_position_limit()
    p2, _ = rm_ok.check_margin_available(10.0)
    p3, _ = rm_ok.check_daily_loss_limit()
    assert p1 and p2 and p3, "Normal durumda tüm kontroller geçmeli"
    print(f"  Normal durum: Tüm kontroller geçti ✓")
    
    print(f"  ✓ Risk kontrolleri doğru")


# =============================================================================
# TEST 6: KILL SWITCH
# =============================================================================

def test_06_kill_switch():
    """Drawdown bazlı kill switch çalışıyor mu?"""
    from execution.risk_manager import RiskManager, RiskCheckStatus
    
    # DD = (75 - 60) / 75 = %20 ≥ %15 → KILL SWITCH
    rm_dd = RiskManager(balance=60.0, initial_balance=75.0)
    passed, msg = rm_dd.check_kill_switch()
    assert not passed, f"DD %20 ama kill switch tetiklenmedi: {msg}"
    assert "KILL SWITCH" in msg, f"Kill switch mesajı eksik: {msg}"
    print(f"  DD %20 (limit %15): 🚨 KILL SWITCH tetiklendi ✓")
    
    # DD = (75 - 70) / 75 = %6.7 < %15 → geçmeli
    rm_ok = RiskManager(balance=70.0, initial_balance=75.0)
    passed, msg = rm_ok.check_kill_switch()
    assert passed, f"DD %6.7 ama kill switch tetiklendi: {msg}"
    print(f"  DD %6.7 (limit %15): Geçti ✓")
    
    # DD = (75 - 65) / 75 = %13.3 → uyarı (>%10.5 = %15'in %70'i)
    rm_warn = RiskManager(balance=65.0, initial_balance=75.0)
    passed, msg = rm_warn.check_kill_switch()
    assert passed, "DD %13.3 kill switch olmamalı"
    assert "⚠️" in msg, f"DD %13.3 uyarı vermeli: {msg}"
    print(f"  DD %13.3 (limit %15): ⚠️ Uyarı verildi ✓")
    
    # Full trade ile kill switch — trade reddedilmeli
    trade = rm_dd.calculate_trade(
        entry_price=185.0, direction='SHORT',
        atr=3.7, symbol='TEST/USDT:USDT'
    )
    assert trade.status == RiskCheckStatus.REJECTED, \
        f"Kill switch'te trade onaylanmamalı: {trade.status}"
    assert not trade.checks.get('kill_switch', True), "Kill switch check false olmalı"
    print(f"  Full trade + kill switch: Reddedildi ✓")
    
    print(f"  ✓ Kill switch doğru")


# =============================================================================
# TEST 7: TAM TRADE PİPELİNE
# =============================================================================

def test_07_full_trade():
    """Tam pipeline: SL → TP → Size → Checks → Karar"""
    from execution.risk_manager import RiskManager, RiskCheckStatus
    
    rm = RiskManager(balance=100.0, initial_balance=100.0)
    
    trade = rm.calculate_trade(
        entry_price=50.0,
        direction='LONG',
        atr=2.0,                               # ATR = $2
        symbol='TEST/USDT:USDT',
        atr_multiplier=1.5,                    # SL dist = $3
        risk_reward=2.0                        # TP dist = $6
    )
    
    # SL kontrolü: LONG → SL = 50 - 3 = 47
    assert abs(trade.stop_loss.price - 47.0) < 0.01, \
        f"SL: {trade.stop_loss.price} != 47"
    
    # TP kontrolü: LONG → TP = 50 + 6 = 56
    assert abs(trade.take_profit.price - 56.0) < 0.01, \
        f"TP: {trade.take_profit.price} != 56"
    
    # Position size: risk = $2, sl_dist = $3 → size = 0.667 coin
    expected_size = round(2.0 / 3.0, 3)        # 0.667
    assert abs(trade.position.size - expected_size) < 0.01, \
        f"Size: {trade.position.size} != {expected_size}"
    
    # Onay kontrolü
    assert trade.status in [RiskCheckStatus.APPROVED, RiskCheckStatus.WARNING], \
        f"Trade onaylanmalı: {trade.status}"
    assert trade.is_approved() or trade.status == RiskCheckStatus.WARNING
    
    # summary() çağrılabilir mi?
    summary = trade.summary()
    assert len(summary) > 0, "Summary boş olmamalı"
    
    print(f"  {trade.symbol} {trade.direction}")
    print(f"  SL: ${trade.stop_loss.price} | TP: ${trade.take_profit.price}")
    print(f"  Size: {trade.position.size} | Leverage: {trade.position.leverage}x")
    print(f"  Status: {trade.status.value}")
    print(f"  ✓ Tam pipeline çalışıyor")


# =============================================================================
# TEST 8: ROADMAP SENARYOSU ($75 SOL SHORT)
# =============================================================================

def test_08_roadmap_scenario():
    """PROJECT_ROADMAP.md'deki $75 bakiye SOL SHORT senaryosu."""
    from execution.risk_manager import RiskManager
    
    rm = RiskManager(balance=75.0, initial_balance=75.0)
    
    trade = rm.calculate_trade(
        entry_price=185.00,
        direction='SHORT',
        atr=3.70,                              # ATR = $3.70
        symbol='SOL/USDT:USDT',
        atr_multiplier=1.0,                    # 1x ATR → SL dist = $3.70
        risk_reward=1.5,                       # RR = 1.5
        min_amount=0.01,
        amount_precision=3
    )
    
    # Risk amount = $75 × 2% = $1.50
    assert abs(trade.position.risk_amount - 1.50) < 0.01, \
        f"Risk amount: {trade.position.risk_amount} != 1.50"
    
    # SL: Entry + ATR × 1.0 = 185 + 3.70 = 188.70 (SHORT)
    assert abs(trade.stop_loss.price - 188.70) < 0.01, \
        f"SL: {trade.stop_loss.price} != 188.70"
    
    # TP: Entry - (SL_dist × 1.5) = 185 - 5.55 = 179.45 (SHORT)
    assert abs(trade.take_profit.price - 179.45) < 0.01, \
        f"TP: {trade.take_profit.price} != 179.45"
    
    # Size = 1.50 / 3.70 = 0.405
    expected_size = round(1.50 / 3.70, 3)      # 0.405
    assert abs(trade.position.size - expected_size) < 0.01, \
        f"Size: {trade.position.size} != {expected_size}"
    
    # Position value ≈ 0.405 × 185 ≈ $74.93
    expected_value = expected_size * 185.0
    assert abs(trade.position.value - expected_value) < 1.0, \
        f"Value: {trade.position.value} != ~{expected_value}"
    
    print(f"  💰 Bakiye: $75.00")
    print(f"  📊 Risk: ${trade.position.risk_amount} (%2)")
    print(f"  📍 Entry: ${trade.entry_price} (SHORT)")
    print(f"  🛑 SL: ${trade.stop_loss.price} (+{trade.stop_loss.distance_pct:.2f}%)")
    print(f"  🎯 TP: ${trade.take_profit.price} (-{trade.take_profit.distance_pct:.2f}%)")
    print(f"  📦 Size: {trade.position.size} SOL (${trade.position.value:,.2f})")
    print(f"  ⚡ Leverage: {trade.position.leverage}x")
    print(f"  💵 Margin: ${trade.position.margin_required:,.2f}")
    print(f"  Status: {trade.status.value}")
    
    # Kazanç/Kayıp hesabı
    win = trade.take_profit.distance * trade.position.size
    loss = trade.stop_loss.distance * trade.position.size
    print(f"\n  Kazanırsa: +${win:,.2f} | Kaybederse: -${loss:,.2f}")
    
    print(f"  ✓ Roadmap senaryosu eşleşiyor")


# =============================================================================
# TEST 9: EDGE CASES
# =============================================================================

def test_09_edge_cases():
    """Sıfır bakiye, çok yüksek ATR, minimum miktar altı."""
    from execution.risk_manager import RiskManager, RiskCheckStatus
    
    # Edge 1: Sıfır bakiye → pozisyon 0 olmalı, reddedilmeli
    rm_zero = RiskManager(balance=0.0)
    trade = rm_zero.calculate_trade(
        entry_price=100.0, direction='LONG', atr=5.0
    )
    assert trade.position.size == 0, "Sıfır bakiyede pozisyon 0 olmalı"
    assert trade.status == RiskCheckStatus.REJECTED, "Sıfır bakiyede red olmalı"
    print(f"  Sıfır bakiye: Reddedildi ✓")
    
    # Edge 2: Çok yüksek ATR → risk/SL_dist çok küçük → min_amount altı
    rm_small = RiskManager(balance=10.0)       # $10 bakiye
    trade2 = rm_small.calculate_trade(
        entry_price=97000.0,                   # BTC
        direction='LONG',
        atr=5000.0,                            # Çok yüksek ATR
        min_amount=0.001                       # Min 0.001 BTC = ~$97
    )
    # risk = $10 × 2% = $0.20, size = $0.20 / $5000 = 0.00004 < min 0.001
    assert trade2.position.size == 0, \
        f"Min amount altında size != 0: {trade2.position.size}"
    print(f"  Yüksek ATR + düşük bakiye: Size=0 ✓")
    
    # Edge 3: RR check — SL distance 0
    rm_rr = RiskManager(balance=100.0)
    passed, msg = rm_rr.check_risk_reward(sl_distance=0.0, tp_distance=10.0)
    assert not passed, "SL distance 0 RR check geçmemeli"
    print(f"  SL distance = 0: Reddedildi ✓")
    
    print(f"  ✓ Edge case'ler doğru işleniyor")


# =============================================================================
# TEST 10: STATE GÜNCELLEME
# =============================================================================

def test_10_state_update():
    """update_state() ile bakiye güncelleme ve tekrar kontrol."""
    from execution.risk_manager import RiskManager, RiskCheckStatus
    
    rm = RiskManager(balance=100.0, open_positions=0, initial_balance=100.0)
    
    # İlk trade → onaylanmalı
    trade1 = rm.calculate_trade(
        entry_price=50.0, direction='LONG',
        atr=2.0, symbol='TEST/USDT:USDT'
    )
    assert trade1.is_approved() or trade1.status == RiskCheckStatus.WARNING
    print(f"  İlk trade: {trade1.status.value} ✓")
    
    # Pozisyon açıldı: state güncelle
    rm.update_state(
        open_positions=1,
        used_margin=trade1.position.margin_required
    )
    
    # İkinci trade → hâlâ onaylanmalı (max 2 pozisyon)
    trade2 = rm.calculate_trade(
        entry_price=50.0, direction='SHORT',
        atr=2.0, symbol='TEST2/USDT:USDT'
    )
    assert trade2.is_approved() or trade2.status == RiskCheckStatus.WARNING
    print(f"  İkinci trade: {trade2.status.value} ✓")
    
    # State güncelle: 2 pozisyon açık
    rm.update_state(open_positions=2)
    
    # Üçüncü trade → reddedilmeli (max 2)
    trade3 = rm.calculate_trade(
        entry_price=50.0, direction='LONG',
        atr=2.0, symbol='TEST3/USDT:USDT'
    )
    assert trade3.status == RiskCheckStatus.REJECTED
    assert not trade3.checks.get('position_limit', True)
    print(f"  Üçüncü trade (2/2 açık): Reddedildi ✓")
    
    # Günlük kayıp güncelle
    rm.update_state(daily_pnl=-5.5, open_positions=0)  # %5.5 kayıp
    passed, msg = rm.check_daily_loss_limit()
    # %5.5 < %6 ama > %80 × %6 = %4.8 → uyarı
    assert passed, "Günlük kayıp %5.5 limiti (%6) geçmemiş"
    print(f"  Günlük kayıp güncelleme: PnL=-$5.5 → {msg}")
    
    print(f"  ✓ State güncelleme doğru")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    """Tüm testleri sırasıyla çalıştırır."""
    
    print("=" * 55)
    print("  ADIM 5: RİSK YÖNETİMİ MOTORU TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    tests = [
        (1,  "SL: ATR bazlı Stop-Loss (LONG+SHORT)",       test_01_stop_loss),
        (2,  "TP: RR bazlı Take-Profit",                    test_02_take_profit),
        (3,  "Position Size: Fixed fractional",              test_03_position_size),
        (4,  "Leverage: Otomatik + config limitleri",        test_04_leverage),
        (5,  "Risk Checks: Limit/margin/günlük kayıp",      test_05_risk_checks),
        (6,  "Kill Switch: Drawdown %15",                    test_06_kill_switch),
        (7,  "Full Trade: Tam pipeline",                     test_07_full_trade),
        (8,  "Roadmap: $75 SOL SHORT senaryosu",             test_08_roadmap_scenario),
        (9,  "Edge Cases: Sıfır bakiye, yüksek ATR",        test_09_edge_cases),
        (10, "State Update: Bakiye güncelleme",              test_10_state_update),
    ]
    
    results = []
    total_start = time.time()
    
    for num, name, func in tests:
        success = run_test(num, name, func)
        results.append((num, name, success))
    
    total_time = time.time() - total_start
    
    # Özet
    print("\n" + "=" * 55)
    print("  TEST SONUÇLARI")
    print("=" * 55)
    
    passed = 0
    failed = 0
    for num, name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} Test {num:>2}: {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {passed + failed} | Başarılı: {passed} | Başarısız: {failed}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        print(f"\n  🎉 ADIM 5 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 6 → AI Entry Optimizer (Gemini)")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları kontrol edin.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
