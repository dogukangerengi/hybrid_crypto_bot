# =============================================================================
# ADIM 8: TELEGRAM BİLDİRİM SİSTEMİ TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_telegram.py
#
# Test 1-6: OFFLİNE (Telegram token gerekmez — format testleri)
# Test 7-8: ONLINE (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID gerekli)
# =============================================================================

import sys
import os
import time
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime

# === ÖNCELİKLİ: .env yükle (tüm import'lardan önce) ===
CURRENT_DIR = Path(__file__).parent            # → src/
PROJECT_ROOT = CURRENT_DIR.parent              # → hybrid_crypto_bot/
ENV_FILE = PROJECT_ROOT / '.env'               # → hybrid_crypto_bot/.env

from dotenv import load_dotenv
load_dotenv(ENV_FILE)                          # .env'deki key'leri os.environ'a yükle

# Path
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
# TEST 1: KONFİGÜRASYON
# =============================================================================

def test_01_config():
    """TelegramNotifier oluşturma ve config kontrolü."""
    from notifications.telegram_notifier import TelegramNotifier
    
    notifier = TelegramNotifier(token=None, chat_id=None)
    
    notifier2 = TelegramNotifier(token="test_token", chat_id="test_chat")
    assert notifier2.is_configured(), "Manuel token ile configured olmalı"
    assert notifier2.token == "test_token"
    assert notifier2.chat_id == "test_chat"
    
    print(f"  Env configured: {notifier.is_configured()}")
    print(f"  Manuel configured: {notifier2.is_configured()} ✓")
    print(f"  ✓ Konfigürasyon doğru")


# =============================================================================
# TEST 2: IC ANALİZ RAPORU FORMATLAMA
# =============================================================================

def test_02_analysis_format():
    """AnalysisReport → Telegram mesajı."""
    from notifications.telegram_notifier import TelegramNotifier, AnalysisReport
    
    notifier = TelegramNotifier(token="test", chat_id="test")
    
    report = AnalysisReport(
        symbol="SOL/USDT:USDT",
        price=185.50,
        recommended_timeframe="15m",
        market_regime="trending_up",
        direction="LONG",
        confidence_score=78.5,
        active_indicators={'trend': ['EMA_20', 'SuperT'], 'momentum': ['RSI_14']},
        category_tops={
            'trend': {'name': 'ADX_14', 'ic': 0.085, 'direction': 'LONG'},
            'momentum': {'name': 'RSI_14', 'ic': 0.072, 'direction': 'LONG'},
        },
        tf_rankings=[
            {'tf': '15m', 'score': 78.5, 'direction': 'LONG'},
            {'tf': '1h', 'score': 65.0, 'direction': 'LONG'},
            {'tf': '4h', 'score': 52.0, 'direction': 'SHORT'},
        ],
        change_24h=3.45,
    )
    
    msg = notifier.format_analysis_report(report)
    
    assert len(msg) > 100, f"Mesaj çok kısa: {len(msg)}"
    assert len(msg) < 4096, f"Mesaj çok uzun: {len(msg)}"
    assert "SOL/USDT" in msg
    assert "185.50" in msg
    assert "LONG" in msg
    assert "78" in msg
    assert "Trend" in msg
    assert "15m" in msg
    
    print(f"  Mesaj uzunluğu: {len(msg)} char")
    print(f"  İçerik: sembol, fiyat, yön, güven, kategori, TF ✓")
    print(f"  ✓ IC analiz formatı doğru")


# =============================================================================
# TEST 3: AI KARAR BİLDİRİMİ FORMATLAMA
# =============================================================================

def test_03_ai_decision_format():
    """AIDecisionResult mock → Telegram mesajı."""
    from notifications.telegram_notifier import TelegramNotifier
    from dataclasses import dataclass
    
    @dataclass
    class MockAIDecision:
        decision: str = "LONG"
        confidence: float = 75.0
        gate_action: str = "FULL_TRADE"
        ic_score: float = 78.0
        atr_multiplier: float = 1.5
        reasoning: str = "IC skoru yüksek, trend yönü LONG ile uyumlu."
        def should_execute(self): return True
    
    notifier = TelegramNotifier(token="test", chat_id="test")
    msg = notifier.format_ai_decision(MockAIDecision())
    
    assert "LONG" in msg
    assert "75" in msg
    assert "TRADE" in msg
    assert "İŞLEM YAP" in msg
    
    @dataclass
    class MockWaitDecision:
        decision: str = "WAIT"
        confidence: float = 30.0
        gate_action: str = "NO_TRADE"
        ic_score: float = 40.0
        atr_multiplier: float = 1.0
        reasoning: str = "IC skoru düşük."
        def should_execute(self): return False
    
    msg_wait = notifier.format_ai_decision(MockWaitDecision())
    assert "WAIT" in msg_wait
    assert "İŞLEM YAPMA" in msg_wait
    
    print(f"  LONG mesaj: {len(msg)} char ✓")
    print(f"  WAIT mesaj: {len(msg_wait)} char ✓")
    print(f"  ✓ AI karar formatı doğru")


# =============================================================================
# TEST 4: TRADE EXECUTION BİLDİRİMİ FORMATLAMA
# =============================================================================

def test_04_trade_format():
    """ExecutionResult mock → Telegram mesajı."""
    from notifications.telegram_notifier import TelegramNotifier
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class MockOrder:
        success: bool = True
        price: float = 0.0
        amount: float = 0.0
    
    @dataclass
    class MockExecution:
        success: bool = True
        symbol: str = "SOL/USDT:USDT"
        direction: str = "SHORT"
        main_order: Optional[MockOrder] = None
        sl_order: Optional[MockOrder] = None
        tp_order: Optional[MockOrder] = None
        actual_entry: float = 185.00
        actual_amount: float = 0.4
        actual_cost: float = 74.0
        dry_run: bool = True
        error: str = ""
        timestamp: str = "2026-02-08 16:45:00 UTC"
    
    notifier = TelegramNotifier(token="test", chat_id="test")
    
    exec_result = MockExecution(
        main_order=MockOrder(success=True, price=185.00, amount=0.4),
        sl_order=MockOrder(success=True, price=188.70),
        tp_order=MockOrder(success=True, price=179.45),
    )
    
    msg = notifier.format_trade_execution(exec_result)
    assert "SHORT" in msg
    assert "SOL" in msg
    assert "185" in msg
    assert "188.70" in msg
    assert "179.45" in msg
    assert "DRY RUN" in msg
    assert "R:R" in msg
    
    err_result = MockExecution(
        success=False, error="Yetersiz bakiye",
        main_order=None, sl_order=None, tp_order=None
    )
    msg_err = notifier.format_trade_execution(err_result)
    assert "BAŞARISIZ" in msg_err
    
    print(f"  Başarılı trade: {len(msg)} char ✓")
    print(f"  Hatalı trade: {len(msg_err)} char ✓")
    print(f"  R:R hesaplaması: var ✓")
    print(f"  ✓ Trade execution formatı doğru")


# =============================================================================
# TEST 5: RİSK UYARISI FORMATLAMA
# =============================================================================

def test_05_risk_alert_format():
    """Risk uyarısı mesaj formatları."""
    from notifications.telegram_notifier import TelegramNotifier
    
    notifier = TelegramNotifier(token="test", chat_id="test")
    
    msg_kill = notifier.format_risk_alert(
        'kill_switch', 'Drawdown %15.3 eşiği aşıldı!', 'critical')
    assert "KİLL SWİTCH" in msg_kill
    assert "🚨" in msg_kill
    
    msg_daily = notifier.format_risk_alert(
        'daily_loss', 'Günlük kayıp limiti %5.8', 'warning')
    assert "GÜNLÜK KAYIP" in msg_daily
    
    msg_reject = notifier.format_risk_alert(
        'trade_rejected', 'SOL SHORT: Margin limiti aşıldı', 'warning')
    assert "REDDEDİLDİ" in msg_reject
    
    print(f"  Kill switch: {len(msg_kill)} char ✓")
    print(f"  Daily loss: {len(msg_daily)} char ✓")
    print(f"  Trade rejected: {len(msg_reject)} char ✓")
    print(f"  ✓ Risk uyarı formatı doğru")


# =============================================================================
# TEST 6: SİSTEM DURUMU FORMATLAMA
# =============================================================================

def test_06_system_status_format():
    """Sistem durum mesajları."""
    from notifications.telegram_notifier import TelegramNotifier
    
    notifier = TelegramNotifier(token="test", chat_id="test")
    
    msg_start = notifier.format_system_status('startup', {
        'balance': 75.0, 'mode': '🧪 DRY RUN'})
    assert "BAŞLATILDI" in msg_start
    
    msg_hb = notifier.format_system_status('heartbeat', {
        'balance': 73.50, 'positions': 1, 'daily_pnl': -1.50, 'uptime': '2h 15m'})
    assert "DURUM" in msg_hb
    
    msg_scan = notifier.format_system_status('scan_complete', {
        'scanned': 523, 'passed': 12, 'best_coin': 'SOL', 'best_score': 87.5})
    assert "TARAMA" in msg_scan
    
    msg_err = notifier.format_system_status('error', {
        'error': 'Connection timeout', 'component': 'Bitget API'})
    assert "HATA" in msg_err
    
    msg_stop = notifier.format_system_status('shutdown', {
        'reason': 'Kill switch tetiklendi'})
    assert "DURDURULDU" in msg_stop
    
    print(f"  Startup: {len(msg_start)} char ✓")
    print(f"  Heartbeat: {len(msg_hb)} char ✓")
    print(f"  Scan complete: {len(msg_scan)} char ✓")
    print(f"  Error: {len(msg_err)} char ✓")
    print(f"  Shutdown: {len(msg_stop)} char ✓")
    print(f"  ✓ Sistem durumu formatları doğru")


# =============================================================================
# TEST 7: BOT BAĞLANTI TESTİ (ONLINE)
# =============================================================================

def test_07_bot_connection():
    """Gerçek Telegram bot bağlantısı."""
    from notifications.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    success = notifier.test_connection_sync()
    assert success, "Bot bağlantısı başarısız!"
    print(f"  ✓ Bot bağlantısı başarılı")


# =============================================================================
# TEST 8: GERÇEK MESAJ GÖNDERME (ONLINE)
# =============================================================================

def test_08_send_messages():
    """Telegram'a gerçek test mesajları gönder."""
    from notifications.telegram_notifier import TelegramNotifier
    
    notifier = TelegramNotifier()
    
    success1 = notifier.send_system_status_sync('startup', {
        'balance': 75.0, 'mode': '🧪 TEST MODU'})
    assert success1, "Startup mesajı gönderilemedi"
    print(f"  📤 Startup mesajı: gönderildi ✓")
    
    time.sleep(1)
    
    from dataclasses import dataclass
    from typing import Optional
    
    @dataclass
    class MockOrder:
        success: bool = True
        price: float = 0.0
        amount: float = 0.0
    
    @dataclass
    class MockExec:
        success: bool = True
        symbol: str = "SOL/USDT:USDT"
        direction: str = "SHORT"
        main_order: Optional[MockOrder] = None
        sl_order: Optional[MockOrder] = None
        tp_order: Optional[MockOrder] = None
        actual_entry: float = 185.00
        actual_amount: float = 0.4
        actual_cost: float = 74.0
        dry_run: bool = True
        error: str = ""
        timestamp: str = "2026-02-08 TEST"
    
    exec_result = MockExec(
        main_order=MockOrder(success=True, price=185.00, amount=0.4),
        sl_order=MockOrder(success=True, price=188.70),
        tp_order=MockOrder(success=True, price=179.45),
    )
    
    success2 = notifier.send_trade_sync(exec_result)
    assert success2, "Trade mesajı gönderilemedi"
    print(f"  📤 Trade mesajı: gönderildi ✓")
    
    print(f"  ✓ Telegram'a 2 mesaj gönderildi — telefonunuzu kontrol edin!")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    print("=" * 55)
    print("  ADIM 8: TELEGRAM BİLDİRİM SİSTEMİ TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    # .env'den okunan değerler
    has_token = bool(os.getenv('TELEGRAM_BOT_TOKEN'))
    has_chat = bool(os.getenv('TELEGRAM_CHAT_ID'))
    has_telegram = has_token and has_chat
    
    print(f"\n  TELEGRAM_BOT_TOKEN: {'✅' if has_token else '❌ Eksik'}")
    print(f"  TELEGRAM_CHAT_ID:  {'✅' if has_chat else '❌ Eksik'}")
    
    # DEBUG: Eğer key bulunamadıysa .env içeriğini göster
    if not has_telegram:
        print(f"\n  🔍 DEBUG — .env yolu: {ENV_FILE}")
        print(f"  🔍 .env mevcut: {ENV_FILE.exists()}")
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r') as f:
                lines = f.readlines()
            print(f"  🔍 .env satır sayısı: {len(lines)}")
            for i, line in enumerate(lines, 1):
                line_s = line.strip()
                if line_s and not line_s.startswith('#'):
                    key = line_s.split('=')[0] if '=' in line_s else line_s
                    print(f"      Satır {i}: {key}=*** ({len(line_s)} char)")
    
    tg_skip = None if has_telegram else "TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID eksik"
    
    tests = [
        (1, "Konfigürasyon kontrolü",           test_01_config,           None),
        (2, "IC Analiz raporu formatlama",       test_02_analysis_format,  None),
        (3, "AI karar bildirimi formatlama",     test_03_ai_decision_format, None),
        (4, "Trade execution formatlama",        test_04_trade_format,     None),
        (5, "Risk uyarısı formatlama",           test_05_risk_alert_format, None),
        (6, "Sistem durumu formatlama",          test_06_system_status_format, None),
        (7, "Bot bağlantı testi (ONLINE)",       test_07_bot_connection,   tg_skip),
        (8, "Gerçek mesaj gönderme (ONLINE)",    test_08_send_messages,    tg_skip),
    ]
    
    results = []
    total_start = time.time()
    
    for num, name, func, skip in tests:
        success = run_test(num, name, func, skip_reason=skip)
        results.append((num, name, success))
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 55)
    print("  TEST SONUÇLARI")
    print("=" * 55)
    
    passed = failed = skipped = 0
    for num, name, success in results:
        if success is None:
            status = "⏭️"; skipped += 1
        elif success:
            status = "✅"; passed += 1
        else:
            status = "❌"; failed += 1
        print(f"  {status} Test {num:>2}: {name}")
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {len(results)} | ✅ {passed} | ❌ {failed} | ⏭️ {skipped}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        if skipped > 0:
            print(f"\n  ✅ Format testleri geçti. Online testler için Telegram key gerekli.")
        else:
            print(f"\n  🎉 ADIM 8 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 9 → Main Pipeline Entegrasyonu")
    else:
        print(f"\n  ⚠️  {failed} test başarısız.")
    
    print("=" * 55)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
