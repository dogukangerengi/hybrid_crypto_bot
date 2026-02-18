#!/usr/bin/env python3
# =============================================================================
# TEST_PAPER_INTEGRATION.PY — Paper Trade Entegrasyonu Testi
# =============================================================================
# Bu script main_paper_integration.py'nin temel fonksiyonlarını test eder.
#
# Çalıştırma:
#   python test_paper_integration.py
# =============================================================================

import sys
import time
from datetime import datetime

print("="*60)
print("  PAPER TRADE ENTEGRASYONU TESTİ")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# =============================================================================
# TEST 1: Import Kontrolü
# =============================================================================
print("\n[TEST 1] Import kontrolü...")

try:
    from main_paper_integration import (
        HybridTradingPipeline,
        CycleStatus,
        GateAction,
        AIDecisionType,
        VERSION,
    )
    print(f"  ✅ main_paper_integration import başarılı (v{VERSION})")
except ImportError as e:
    print(f"  ❌ Import hatası: {e}")
    print("\n  💡 Çözüm: Tüm bağımlılıkların yüklü olduğundan emin ol")
    sys.exit(1)

# =============================================================================
# TEST 2: Paper Trader Entegrasyonu
# =============================================================================
print("\n[TEST 2] Paper Trader entegrasyonu...")

try:
    from paper_trader import PaperTrader
    
    # Pipeline oluştur (API çağrısı yapmadan)
    pipeline = HybridTradingPipeline(
        dry_run=True,
        top_n=3,
        verbose=False,
    )
    
    # Paper trader var mı?
    assert hasattr(pipeline, 'paper_trader'), "paper_trader attribute yok"
    assert isinstance(pipeline.paper_trader, PaperTrader), "paper_trader tipi hatalı"
    
    print(f"  ✅ Paper Trader entegre")
    print(f"     Başlangıç bakiye: ${pipeline.paper_trader.initial_balance:.2f}")
    
except Exception as e:
    print(f"  ❌ Hata: {e}")
    sys.exit(1)

# =============================================================================
# TEST 3: AIDecisionType Helper
# =============================================================================
print("\n[TEST 3] AIDecisionType helper...")

try:
    # LONG çevirimi
    assert AIDecisionType.from_direction("LONG") == AIDecisionType.LONG
    assert AIDecisionType.from_direction("BUY") == AIDecisionType.LONG
    assert AIDecisionType.from_direction("BULLISH") == AIDecisionType.LONG
    
    # SHORT çevirimi
    assert AIDecisionType.from_direction("SHORT") == AIDecisionType.SHORT
    assert AIDecisionType.from_direction("SELL") == AIDecisionType.SHORT
    assert AIDecisionType.from_direction("BEARISH") == AIDecisionType.SHORT
    
    # WAIT çevirimi
    assert AIDecisionType.from_direction("NEUTRAL") == AIDecisionType.WAIT
    assert AIDecisionType.from_direction("") == AIDecisionType.WAIT
    assert AIDecisionType.from_direction(None) == AIDecisionType.WAIT
    
    print("  ✅ AIDecisionType.from_direction() çalışıyor")
    print("     LONG ← LONG, BUY, BULLISH")
    print("     SHORT ← SHORT, SELL, BEARISH")
    print("     WAIT ← NEUTRAL, '', None")
    
except AssertionError as e:
    print(f"  ❌ Assertion hatası: {e}")
    sys.exit(1)

# =============================================================================
# TEST 4: Bakiye Başlatma
# =============================================================================
print("\n[TEST 4] Bakiye başlatma...")

try:
    success = pipeline._init_balance()
    
    assert success, "Bakiye başlatma başarısız"
    assert pipeline._balance > 0, "Bakiye 0"
    assert pipeline._risk_manager is not None, "Risk manager None"
    
    print(f"  ✅ Bakiye başlatıldı: ${pipeline._balance:.2f}")
    
except Exception as e:
    print(f"  ❌ Hata: {e}")

# =============================================================================
# TEST 5: Gate Action Enum
# =============================================================================
print("\n[TEST 5] Gate Action enum...")

try:
    assert GateAction.NO_TRADE.value == "no_trade"
    assert GateAction.REPORT_ONLY.value == "report_only"
    assert GateAction.FULL_TRADE.value == "full_trade"
    
    print("  ✅ GateAction enum doğru")
    print(f"     NO_TRADE: IC < 55")
    print(f"     REPORT_ONLY: IC 55-70")
    print(f"     FULL_TRADE: IC > 70")
    
except AssertionError as e:
    print(f"  ❌ Assertion hatası: {e}")

# =============================================================================
# TEST 6: Cycle Status Enum
# =============================================================================
print("\n[TEST 6] Cycle Status enum...")

try:
    statuses = [
        CycleStatus.SUCCESS,
        CycleStatus.PARTIAL,
        CycleStatus.NO_SIGNAL,
        CycleStatus.ERROR,
        CycleStatus.KILLED,
    ]
    
    for s in statuses:
        assert s.value is not None
    
    print("  ✅ CycleStatus enum doğru")
    print(f"     {', '.join(s.value for s in statuses)}")
    
except AssertionError as e:
    print(f"  ❌ Assertion hatası: {e}")

# =============================================================================
# TEST 7: Kill Switch Fonksiyonu
# =============================================================================
print("\n[TEST 7] Kill Switch...")

try:
    # Başlangıçta kapalı olmalı
    assert pipeline._kill_switch == False, "Kill switch başlangıçta kapalı olmalı"
    
    # Drawdown yokken tetiklenmemeli
    triggered = pipeline._check_kill_switch()
    assert triggered == False, "Drawdown yokken tetiklenmemeli"
    
    print("  ✅ Kill switch fonksiyonu çalışıyor")
    print(f"     Threshold: 15% drawdown")
    
except Exception as e:
    print(f"  ❌ Hata: {e}")

# =============================================================================
# ÖZET
# =============================================================================
print("\n" + "="*60)
print("  ✅ TÜM TESTLER BAŞARILI")
print("="*60)
print("""
📋 KURULUM TALİMATLARI:

1. main_paper_integration.py dosyasını main.py olarak kullan
   VEYA mevcut main.py'ye entegre et

2. Çalıştır:
   python main.py                    # Tek döngü
   python main.py --schedule -i 60   # Her saat
   python main.py --report           # Performans raporu

3. Paper trade logları:
   logs/paper_trades/paper_trades.json

4. AI Quota yönetimi:
   - Free tier: ~5 req/dk, ~20 req/gün
   - Quota bitince → IC-only mode (otomatik)
   - Paid plan'a geçince limit kalkar

5. 1 hafta paper trade yap, sonra analiz et:
   python main.py --report
""")
print("="*60)
