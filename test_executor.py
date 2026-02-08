# =============================================================================
# ADIM 7: BİTGET EXECUTION ENGİNE TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_executor.py
#
# Test 1-7: DRY RUN (API key gerekmez — simülasyon)
# Test 8-10: ONLINE (BITGET_API_KEY gerekli — yoksa SKIP)
#
# Test Listesi:
# 1.  DRY RUN: Bakiye sorgusu (simülasyon)
# 2.  DRY RUN: Pozisyon sorgusu (simülasyon)
# 3.  DRY RUN: Market emir (simülasyon)
# 4.  DRY RUN: SL/TP trigger emirleri (simülasyon)
# 5.  DRY RUN: Full trade pipeline (RiskManager entegrasyon)
# 6.  DRY RUN: Pozisyon kapatma (simülasyon)
# 7.  DRY RUN: Roadmap senaryosu ($75 SOL SHORT)
# 8.  API: Bakiye sorgulama (ONLINE)
# 9.  API: Market info (ONLINE)
# 10. API: Pozisyon sorgulama (ONLINE)
# =============================================================================

import sys
import time
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# Path ayarı
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

def run_test(test_num, test_name, test_func, skip_reason=None):
    """Tek testi çalıştır."""
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num:>2}: {test_name}")
    print(f"{'─' * 55}")
    
    if skip_reason:
        print(f"  ⏭️  ATLANILDI: {skip_reason}")
        return None
    
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
# TEST 1: DRY RUN BAKİYE
# =============================================================================

def test_01_dry_balance():
    """DRY RUN modda bakiye sorgusu çalışıyor mu?"""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=True)
    balance = executor.fetch_balance()
    
    assert isinstance(balance, dict), "Bakiye dict olmalı"
    assert 'total' in balance, "total key olmalı"
    assert 'free' in balance, "free key olmalı"
    assert 'used' in balance, "used key olmalı"
    assert balance['total'] > 0, "DRY RUN bakiye > 0 olmalı"
    
    print(f"  💰 Total: ${balance['total']:,.2f}")
    print(f"  💵 Free:  ${balance['free']:,.2f}")
    print(f"  🔒 Used:  ${balance['used']:,.2f}")
    print(f"  ✓ DRY RUN bakiye sorgusu doğru")


# =============================================================================
# TEST 2: DRY RUN POZİSYON
# =============================================================================

def test_02_dry_positions():
    """DRY RUN modda pozisyon sorgusu."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=True)
    positions = executor.fetch_positions()
    
    assert isinstance(positions, list), "Pozisyonlar list olmalı"
    assert len(positions) == 0, "DRY RUN'da açık pozisyon yok"
    
    print(f"  📊 Açık pozisyon: {len(positions)}")
    print(f"  ✓ DRY RUN pozisyon sorgusu doğru")


# =============================================================================
# TEST 3: DRY RUN MARKET EMİR
# =============================================================================

def test_03_dry_market_order():
    """DRY RUN modda market emir simülasyonu."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=True)
    
    # BUY emri (LONG açmak için)
    order_buy = executor.place_market_order(
        symbol='SOL/USDT:USDT',
        side='buy',
        amount=0.405
    )
    assert order_buy.success, f"Buy emri başarısız: {order_buy.error}"
    assert order_buy.order_id.startswith('DRY_'), "DRY RUN order ID olmalı"
    # SOL precision amount=1 → 0.405 truncate → 0.4 (doğru davranış)
    assert order_buy.filled <= 0.405, f"Filled miktar orijinalden büyük olamaz: {order_buy.filled}"
    assert order_buy.filled > 0, f"Filled miktar 0 olamaz"
    assert order_buy.status == 'closed', f"Status hatalı: {order_buy.status}"
    
    print(f"  BUY:  {order_buy.side} {order_buy.amount} (0.405→{order_buy.filled} truncate) → {order_buy.status} ✓")
    
    # SELL emri (SHORT açmak için)
    order_sell = executor.place_market_order(
        symbol='BTC/USDT:USDT',
        side='sell',
        amount=0.001
    )
    assert order_sell.success, f"Sell emri başarısız: {order_sell.error}"
    
    print(f"  SELL: {order_sell.side} {order_sell.amount} → {order_sell.status} ✓")
    print(f"  ✓ DRY RUN market emir doğru (precision truncate çalışıyor)")


# =============================================================================
# TEST 4: DRY RUN SL/TP
# =============================================================================

def test_04_dry_sl_tp():
    """DRY RUN modda SL ve TP trigger emirleri."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=True)
    
    # SL emri (SHORT pozisyon için → buy ile kapat)
    sl = executor.place_stop_loss(
        symbol='SOL/USDT:USDT',
        side='buy',                            # SHORT kapatma
        amount=0.405,
        trigger_price=188.70
    )
    assert sl.success, f"SL emri başarısız: {sl.error}"
    assert sl.order_id.startswith('DRY_SL_'), "DRY SL ID olmalı"
    assert sl.price == 188.70, f"SL fiyat hatalı: {sl.price}"
    
    print(f"  🛑 SL: {sl.side} @ ${sl.price:,.2f} → {sl.status} ✓")
    
    # TP emri (SHORT pozisyon için → buy ile kapat)
    tp = executor.place_take_profit(
        symbol='SOL/USDT:USDT',
        side='buy',
        amount=0.405,
        trigger_price=179.45
    )
    assert tp.success, f"TP emri başarısız: {tp.error}"
    assert tp.order_id.startswith('DRY_TP_'), "DRY TP ID olmalı"
    
    print(f"  🎯 TP: {tp.side} @ ${tp.price:,.2f} → {tp.status} ✓")
    print(f"  ✓ DRY RUN SL/TP emirleri doğru")


# =============================================================================
# TEST 5: DRY RUN FULL TRADE PİPELİNE
# =============================================================================

def test_05_dry_full_trade():
    """RiskManager + Executor entegrasyonu (DRY RUN)."""
    from execution.risk_manager import RiskManager
    from execution.bitget_executor import BitgetExecutor
    
    # Risk hesapla
    rm = RiskManager(balance=100.0, initial_balance=100.0)
    trade = rm.calculate_trade(
        entry_price=50.0,
        direction='LONG',
        atr=2.0,
        symbol='TEST/USDT:USDT',
        atr_multiplier=1.5,
        risk_reward=2.0
    )
    
    assert trade.is_approved(), f"Trade onaylı değil: {trade.rejection_reasons}"
    
    # Execute (DRY RUN)
    executor = BitgetExecutor(dry_run=True)
    result = executor.execute_trade(trade)
    
    assert result.success, f"Execution başarısız: {result.error}"
    assert result.dry_run, "DRY RUN modda olmalı"
    assert result.direction == 'LONG', f"Yön hatalı: {result.direction}"
    assert result.main_order is not None, "Ana emir olmalı"
    assert result.main_order.success, "Ana emir başarılı olmalı"
    assert result.sl_order is not None, "SL emri olmalı"
    assert result.sl_order.success, "SL emri başarılı olmalı"
    assert result.tp_order is not None, "TP emri olmalı"
    assert result.tp_order.success, "TP emri başarılı olmalı"
    
    print(f"  📋 Trade: {result.direction} {result.symbol}")
    print(f"  📍 Entry: ${result.actual_entry:,.2f} ({result.main_order.status})")
    print(f"  🛑 SL: ${result.sl_order.price:,.2f} ({result.sl_order.status})")
    print(f"  🎯 TP: ${result.tp_order.price:,.2f} ({result.tp_order.status})")
    print(f"  🧪 Dry Run: {result.dry_run}")
    print(f"  ✓ Full trade pipeline doğru")


# =============================================================================
# TEST 6: DRY RUN POZİSYON KAPATMA
# =============================================================================

def test_06_dry_close():
    """DRY RUN pozisyon kapatma."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=True)
    
    # LONG pozisyon kapatma (sell ile)
    result = executor.close_position(
        symbol='SOL/USDT:USDT',
        side='long',
        amount=0.405
    )
    assert result.success, f"Kapatma başarısız: {result.error}"
    assert result.side == 'sell', "LONG kapatma → sell olmalı"
    
    print(f"  LONG kapatma: {result.side} {result.amount} → {result.status} ✓")
    
    # SHORT pozisyon kapatma (buy ile)
    result2 = executor.close_position(
        symbol='BTC/USDT:USDT',
        side='short',
        amount=0.001
    )
    assert result2.success, f"Kapatma başarısız: {result2.error}"
    assert result2.side == 'buy', "SHORT kapatma → buy olmalı"
    
    print(f"  SHORT kapatma: {result2.side} {result2.amount} → {result2.status} ✓")
    print(f"  ✓ DRY RUN pozisyon kapatma doğru")


# =============================================================================
# TEST 7: ROADMAP SENARYOSU ($75 SOL SHORT)
# =============================================================================

def test_07_roadmap_scenario():
    """$75 bakiye ile SOL SHORT — tam pipeline DRY RUN."""
    from execution.risk_manager import RiskManager
    from execution.bitget_executor import BitgetExecutor
    
    # Risk hesapla (Adım 5'teki senaryo)
    rm = RiskManager(balance=75.0, initial_balance=75.0)
    trade = rm.calculate_trade(
        entry_price=185.00,
        direction='SHORT',
        atr=3.70,
        symbol='SOL/USDT:USDT',
        atr_multiplier=1.0,
        risk_reward=1.5,
        min_amount=0.01,
        amount_precision=3
    )
    
    # Execute
    executor = BitgetExecutor(dry_run=True)
    result = executor.execute_trade(trade)
    
    assert result.success, f"Execution başarısız: {result.error}"
    assert result.direction == 'SHORT'
    assert result.main_order.side == 'sell'            # SHORT = sell ile aç
    
    # SL/TP yönleri doğru mu?
    assert result.sl_order.side == 'buy', "SHORT SL → buy ile kapat"
    assert result.tp_order.side == 'buy', "SHORT TP → buy ile kapat"
    
    # Fiyatlar
    assert result.sl_order.price == 188.70, f"SL fiyat: {result.sl_order.price}"
    assert result.tp_order.price == 179.45, f"TP fiyat: {result.tp_order.price}"
    
    print(f"  💰 Bakiye: $75.00")
    print(f"  📊 {result.direction} {result.symbol}")
    print(f"  📦 Size: {result.main_order.amount} SOL (0.405→truncate)")
    print(f"  📍 Entry: market order ({result.main_order.side})")
    print(f"  🛑 SL: ${result.sl_order.price:,.2f} ({result.sl_order.side})")
    print(f"  🎯 TP: ${result.tp_order.price:,.2f} ({result.tp_order.side})")
    print(f"  🧪 Mod: DRY RUN")
    
    # Summary test
    summary = result.summary()
    assert len(summary) > 30, "Summary boş olmamalı"
    assert 'SHORT' in summary
    
    print(f"\n  📋 Summary:\n  {summary.replace(chr(10), chr(10) + '  ')}")
    print(f"  ✓ Roadmap senaryosu doğru")


# =============================================================================
# TEST 8: API BAKİYE (ONLINE)
# =============================================================================

def test_08_api_balance():
    """Gerçek Bitget API ile bakiye sorgulama."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=False)       # CANLI mod (sadece okuma)
    balance = executor.fetch_balance()
    
    assert isinstance(balance, dict)
    assert 'total' in balance
    assert balance['total'] >= 0, "Bakiye negatif olamaz"
    
    print(f"  💰 Total: ${balance['total']:,.2f}")
    print(f"  💵 Free:  ${balance['free']:,.2f}")
    print(f"  🔒 Used:  ${balance['used']:,.2f}")
    print(f"  ✓ API bakiye sorgusu çalışıyor")


# =============================================================================
# TEST 9: API MARKET INFO (ONLINE)
# =============================================================================

def test_09_api_market_info():
    """Gerçek API ile market info çekme."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=False)
    
    # BTC market info
    info = executor.get_market_info('BTC/USDT:USDT')
    
    assert 'precision' in info
    assert 'limits' in info
    assert info['max_leverage'] >= 50, f"BTC max leverage >= 50 olmalı: {info['max_leverage']}"
    
    print(f"  📊 BTC/USDT:USDT Market Info:")
    print(f"     Price precision: {info['precision']['price']}")
    print(f"     Amount precision: {info['precision']['amount']}")
    print(f"     Min amount: {info['limits']['min_amount']}")
    print(f"     Min cost: ${info['limits']['min_cost']}")
    print(f"     Max leverage: {info['max_leverage']}x")
    
    # SOL market info
    sol_info = executor.get_market_info('SOL/USDT:USDT')
    print(f"\n  📊 SOL/USDT:USDT:")
    print(f"     Amount precision: {sol_info['precision']['amount']}")
    print(f"     Min amount: {sol_info['limits']['min_amount']}")
    
    print(f"  ✓ Market info çalışıyor")


# =============================================================================
# TEST 10: API POZİSYON (ONLINE)
# =============================================================================

def test_10_api_positions():
    """Gerçek API ile açık pozisyon sorgulama."""
    from execution.bitget_executor import BitgetExecutor
    
    executor = BitgetExecutor(dry_run=False)
    positions = executor.fetch_positions()
    
    assert isinstance(positions, list)
    
    print(f"  📊 Açık pozisyon: {len(positions)}")
    
    for pos in positions:
        dir_emoji = "🟢" if pos['side'] == 'long' else "🔴"
        print(f"     {dir_emoji} {pos['symbol']}: {pos['amount']} "
              f"@ ${pos['entry_price']:,.2f} | "
              f"PnL: ${pos['unrealized_pnl']:+,.2f}")
    
    if not positions:
        print(f"     (pozisyon yok)")
    
    print(f"  ✓ Pozisyon sorgusu çalışıyor")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    print("=" * 55)
    print("  ADIM 7: BİTGET EXECUTION ENGİNE TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    # API durumu
    from config import cfg
    has_api = cfg.exchange.is_configured()
    
    print(f"\n  Bitget API: {'✅ Yapılandırılmış' if has_api else '❌ Key eksik (Test 8-10 atlanacak)'}")
    
    api_skip = None if has_api else "BITGET_API_KEY yok"
    
    tests = [
        (1,  "DRY RUN: Bakiye sorgusu",               test_01_dry_balance,      None),
        (2,  "DRY RUN: Pozisyon sorgusu",              test_02_dry_positions,    None),
        (3,  "DRY RUN: Market emir",                   test_03_dry_market_order, None),
        (4,  "DRY RUN: SL/TP trigger emirleri",        test_04_dry_sl_tp,       None),
        (5,  "DRY RUN: Full trade pipeline",           test_05_dry_full_trade,  None),
        (6,  "DRY RUN: Pozisyon kapatma",              test_06_dry_close,       None),
        (7,  "DRY RUN: Roadmap $75 SOL SHORT",         test_07_roadmap_scenario, None),
        (8,  "API: Bakiye sorgulama (ONLINE)",          test_08_api_balance,     api_skip),
        (9,  "API: Market info (ONLINE)",               test_09_api_market_info, api_skip),
        (10, "API: Pozisyon sorgulama (ONLINE)",        test_10_api_positions,   api_skip),
    ]
    
    results = []
    total_start = time.time()
    
    for num, name, func, skip in tests:
        success = run_test(num, name, func, skip_reason=skip)
        results.append((num, name, success))
    
    total_time = time.time() - total_start
    
    # Özet
    print("\n" + "=" * 55)
    print("  TEST SONUÇLARI")
    print("=" * 55)
    
    passed = failed = skipped = 0
    for num, name, success in results:
        if success is None:
            status = "⏭️"
            skipped += 1
        elif success:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        print(f"  {status} Test {num:>2}: {name}")
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {len(results)} | ✅ {passed} | ❌ {failed} | ⏭️ {skipped}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        if skipped > 0:
            print(f"\n  ✅ DRY RUN testleri geçti. API testleri için Bitget key gerekli.")
        else:
            print(f"\n  🎉 ADIM 7 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 8 → Telegram Bildirim Entegrasyonu")
    else:
        print(f"\n  ⚠️  {failed} test başarısız.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
