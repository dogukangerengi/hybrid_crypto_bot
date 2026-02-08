# =============================================================================
# ADIM 6: AI ENTRY OPTİMİZER TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_ai_optimizer.py
#
# Test 1-6: OFFLİNE (API key gerekmez)
# Test 7-8: ONLİNE (GEMINI_API_KEY gerekli — yoksa SKIP edilir)
#
# Test Listesi:
# 1.  Gate Keeper: IC eşik kontrolleri (NO_TRADE/REPORT/FULL)
# 2.  Prompt Builder: Structured prompt oluşturma
# 3.  Response Parser: JSON parse (clean + code block + broken)
# 4.  Validator: Parsed JSON doğrulama (aralık, type)
# 5.  IC Fallback: Gemini başarısız olursa IC bazlı karar
# 6.  Decision Result: should_execute() ve summary() kontrolleri
# 7.  Gemini API: Gerçek API çağrısı (GEMINI_API_KEY gerekli)
# 8.  Full Pipeline: Input → Gate → Gemini → Decision (API gerekli)
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
# YARDIMCI: TEST INPUT OLUŞTURUCU
# =============================================================================

def make_test_input(
    ic_confidence=75.0,
    ic_direction='SHORT',
    price=185.0,
    coin='SOL'
):
    """Standart test input oluşturur."""
    from ai.gemini_optimizer import AIAnalysisInput
    
    return AIAnalysisInput(
        symbol=f'{coin}/USDT:USDT',
        coin=coin,
        price=price,
        change_24h=-2.3,
        best_timeframe='1h',
        ic_confidence=ic_confidence,
        ic_direction=ic_direction,
        category_tops={
            'trend': {'name': 'SUPERTREND', 'ic': -0.12},
            'momentum': {'name': 'RSI_14', 'ic': -0.08},
            'volatility': {'name': 'ATR_14', 'ic': 0.05},
            'volume': {'name': 'CMF_20', 'ic': -0.10},
        },
        tf_rankings=[
            {'tf': '1h', 'score': 75, 'direction': 'SHORT'},
            {'tf': '30m', 'score': 68, 'direction': 'SHORT'},
            {'tf': '4h', 'score': 55, 'direction': 'NEUTRAL'},
        ],
        atr=3.70,
        atr_pct=2.0,
        sl_price=188.70,
        tp_price=179.45,
        risk_reward=1.5,
        position_size=0.405,
        leverage=4,
        market_regime='trending',
        volume_24h=500_000_000,
        volatility=4.0
    )


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_test(test_num, test_name, test_func, skip_reason=None):
    """Tek testi çalıştır."""
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num}: {test_name}")
    print(f"{'─' * 55}")
    
    if skip_reason:
        print(f"  ⏭️  ATLANILDI: {skip_reason}")
        return None  # None = skipped
    
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
# TEST 1: GATE KEEPER
# =============================================================================

def test_01_gate_keeper():
    """IC eşik kontrolleri doğru çalışıyor mu?"""
    from ai.gemini_optimizer import GeminiOptimizer, GateAction
    
    optimizer = GeminiOptimizer()
    
    # IC < 55 → NO_TRADE
    assert optimizer._check_gate(40) == GateAction.NO_TRADE, "IC=40 → NO_TRADE olmalı"
    assert optimizer._check_gate(54.9) == GateAction.NO_TRADE, "IC=54.9 → NO_TRADE olmalı"
    print(f"  IC < 55: NO_TRADE ✓")
    
    # 55 ≤ IC < 70 → REPORT_ONLY
    assert optimizer._check_gate(55) == GateAction.REPORT_ONLY, "IC=55 → REPORT_ONLY olmalı"
    assert optimizer._check_gate(69.9) == GateAction.REPORT_ONLY, "IC=69.9 → REPORT_ONLY olmalı"
    print(f"  55 ≤ IC < 70: REPORT_ONLY ✓")
    
    # IC ≥ 70 → FULL_TRADE
    assert optimizer._check_gate(70) == GateAction.FULL_TRADE, "IC=70 → FULL_TRADE olmalı"
    assert optimizer._check_gate(95) == GateAction.FULL_TRADE, "IC=95 → FULL_TRADE olmalı"
    print(f"  IC ≥ 70: FULL_TRADE ✓")
    
    # Edge: 0 ve 100
    assert optimizer._check_gate(0) == GateAction.NO_TRADE
    assert optimizer._check_gate(100) == GateAction.FULL_TRADE
    print(f"  Edge (0, 100): ✓")
    
    print(f"  ✓ Gate Keeper eşikleri doğru")


# =============================================================================
# TEST 2: PROMPT BUILDER
# =============================================================================

def test_02_prompt_builder():
    """Structured prompt doğru oluşturuluyor mu?"""
    from ai.gemini_optimizer import GeminiOptimizer
    
    optimizer = GeminiOptimizer()
    test_input = make_test_input()
    
    prompt = optimizer._build_prompt(test_input)
    
    # Prompt boş olmamalı
    assert len(prompt) > 200, f"Prompt çok kısa: {len(prompt)} char"
    
    # Temel bilgiler prompt'ta olmalı
    assert 'SOL' in prompt, "Coin adı prompt'ta olmalı"
    assert '185' in prompt, "Fiyat prompt'ta olmalı"
    assert 'SHORT' in prompt, "IC yönü prompt'ta olmalı"
    assert 'SUPERTREND' in prompt, "Top indikatör prompt'ta olmalı"
    assert 'json' in prompt.lower(), "JSON format talimatı olmalı"
    assert 'LONG' in prompt and 'WAIT' in prompt, "Karar seçenekleri olmalı"
    assert 'trending' in prompt, "Market rejimi olmalı"
    assert 'ATR' in prompt, "ATR bilgisi olmalı"
    
    print(f"  Prompt uzunluğu: {len(prompt)} char")
    print(f"  Coin/fiyat/yön/indikatör: var ✓")
    print(f"  JSON format talimatı: var ✓")
    print(f"  Risk bilgileri: var ✓")
    print(f"  ✓ Prompt builder doğru")


# =============================================================================
# TEST 3: RESPONSE PARSER
# =============================================================================

def test_03_response_parser():
    """JSON parse — clean, code block ve broken formatlar."""
    from ai.gemini_optimizer import GeminiOptimizer
    
    optimizer = GeminiOptimizer()
    
    # Clean JSON
    clean = '{"decision": "SHORT", "confidence": 78, "reasoning": "IC güçlü SHORT", "atr_multiplier": 1.5}'
    result = optimizer._parse_response(clean)
    assert result['decision'] == 'SHORT', f"Clean parse hatalı: {result}"
    assert result['confidence'] == 78
    print(f"  Clean JSON: ✓")
    
    # Markdown code block
    code_block = '```json\n{"decision": "LONG", "confidence": 82, "reasoning": "Test", "atr_multiplier": 2.0}\n```'
    result2 = optimizer._parse_response(code_block)
    assert result2['decision'] == 'LONG'
    assert result2['confidence'] == 82
    print(f"  Code block: ✓")
    
    # Ön/arka text ile JSON
    messy = 'İşte analiz sonuçum: {"decision": "WAIT", "confidence": 45, "reasoning": "Belirsiz", "atr_multiplier": 1.5} Bu kadar.'
    result3 = optimizer._parse_response(messy)
    assert result3['decision'] == 'WAIT'
    assert result3['confidence'] == 45
    print(f"  Messy text + JSON: ✓")
    
    # Tamamen geçersiz → default response
    broken = 'Bu bir JSON değil, sadece düz text.'
    result4 = optimizer._parse_response(broken)
    assert result4['decision'] == 'WAIT', "Broken → WAIT olmalı"
    assert result4['confidence'] <= 50, "Broken → düşük güven olmalı"
    print(f"  Broken text → WAIT fallback: ✓")
    
    print(f"  ✓ Response parser doğru")


# =============================================================================
# TEST 4: VALIDATOR
# =============================================================================

def test_04_validator():
    """Parse edilen JSON doğrulama (aralık, type)."""
    from ai.gemini_optimizer import GeminiOptimizer
    
    optimizer = GeminiOptimizer()
    
    # Geçersiz decision → WAIT'e çevrilmeli
    result = optimizer._validate_parsed({'decision': 'INVALID', 'confidence': 50})
    assert result['decision'] == 'WAIT'
    print(f"  Invalid decision → WAIT: ✓")
    
    # Confidence aralık dışı → clamp
    result2 = optimizer._validate_parsed({'decision': 'LONG', 'confidence': 150})
    assert result2['confidence'] == 100
    print(f"  Confidence > 100 → 100: ✓")
    
    result3 = optimizer._validate_parsed({'decision': 'SHORT', 'confidence': -20})
    assert result3['confidence'] == 0
    print(f"  Confidence < 0 → 0: ✓")
    
    # ATR multiplier aralık dışı → clamp
    result4 = optimizer._validate_parsed({
        'decision': 'LONG', 'confidence': 50, 'atr_multiplier': 5.0
    })
    assert result4['atr_multiplier'] == 3.0
    print(f"  ATR mult > 3.0 → 3.0: ✓")
    
    result5 = optimizer._validate_parsed({
        'decision': 'SHORT', 'confidence': 50, 'atr_multiplier': 0.5
    })
    assert result5['atr_multiplier'] == 1.0
    print(f"  ATR mult < 1.0 → 1.0: ✓")
    
    print(f"  ✓ Validator doğru")


# =============================================================================
# TEST 5: IC FALLBACK
# =============================================================================

def test_05_ic_fallback():
    """Gemini başarısız olursa IC bazlı fallback karar."""
    from ai.gemini_optimizer import GeminiOptimizer, AIDecision, GateAction
    
    optimizer = GeminiOptimizer()
    
    # Yüksek IC + net yön → IC yönünü takip et
    strong_input = make_test_input(ic_confidence=80, ic_direction='SHORT')
    fb1 = optimizer._ic_fallback(strong_input)
    assert fb1.decision == AIDecision.SHORT, "IC=80 SHORT → SHORT olmalı"
    assert fb1.confidence <= 65, "Fallback max %65 güven olmalı"
    assert fb1.gate_action == GateAction.REPORT_ONLY, "Fallback hiç FULL_TRADE vermemeli"
    print(f"  IC=80 SHORT: {fb1.decision.value}, güven={fb1.confidence:.0f} ✓")
    
    # Düşük IC → WAIT
    weak_input = make_test_input(ic_confidence=50, ic_direction='NEUTRAL')
    fb2 = optimizer._ic_fallback(weak_input)
    assert fb2.decision == AIDecision.WAIT, "IC=50 NEUTRAL → WAIT olmalı"
    print(f"  IC=50 NEUTRAL: {fb2.decision.value} ✓")
    
    # Yüksek IC ama NEUTRAL yön → WAIT
    neutral_input = make_test_input(ic_confidence=85, ic_direction='NEUTRAL')
    fb3 = optimizer._ic_fallback(neutral_input)
    assert fb3.decision == AIDecision.WAIT, "NEUTRAL yön → WAIT olmalı"
    print(f"  IC=85 NEUTRAL: {fb3.decision.value} ✓")
    
    print(f"  ✓ IC Fallback doğru")


# =============================================================================
# TEST 6: DECISION RESULT
# =============================================================================

def test_06_decision_result():
    """should_execute() ve summary() çalışıyor mu?"""
    from ai.gemini_optimizer import AIDecisionResult, AIDecision, GateAction
    
    # FULL_TRADE + LONG + yüksek güven → execute
    exec_yes = AIDecisionResult(
        decision=AIDecision.LONG,
        confidence=75,
        reasoning="Güçlü sinyal",
        gate_action=GateAction.FULL_TRADE,
        ic_score=80,
        entry_price=185.0,
        sl_price=180.0,
        tp_price=195.0,
        risk_reward=2.0
    )
    assert exec_yes.should_execute(), "FULL_TRADE + LONG + 75% → execute olmalı"
    print(f"  FULL_TRADE + LONG + 75%: should_execute=True ✓")
    
    # REPORT_ONLY → execute etmemeli
    exec_no_report = AIDecisionResult(
        decision=AIDecision.SHORT,
        confidence=80,
        reasoning="Sinyal var",
        gate_action=GateAction.REPORT_ONLY,
        ic_score=60
    )
    assert not exec_no_report.should_execute(), "REPORT_ONLY → execute olmamalı"
    print(f"  REPORT_ONLY: should_execute=False ✓")
    
    # WAIT kararı → execute etmemeli
    exec_no_wait = AIDecisionResult(
        decision=AIDecision.WAIT,
        confidence=40,
        reasoning="Belirsiz",
        gate_action=GateAction.FULL_TRADE,
        ic_score=75
    )
    assert not exec_no_wait.should_execute(), "WAIT → execute olmamalı"
    print(f"  WAIT: should_execute=False ✓")
    
    # Düşük güven → execute etmemeli
    exec_no_conf = AIDecisionResult(
        decision=AIDecision.LONG,
        confidence=50,                         # < 60 eşiği
        reasoning="Zayıf sinyal",
        gate_action=GateAction.FULL_TRADE,
        ic_score=80
    )
    assert not exec_no_conf.should_execute(), "Güven < 60 → execute olmamalı"
    print(f"  Güven < 60: should_execute=False ✓")
    
    # summary() çağrılabilir mi?
    summary = exec_yes.summary()
    assert len(summary) > 50, "Summary boş olmamalı"
    assert "LONG" in summary, "Summary'de karar olmalı"
    assert "Entry" in summary, "Execute durumda SL/TP bilgisi olmalı"
    print(f"  summary(): {len(summary)} char, LONG + Entry var ✓")
    
    print(f"  ✓ Decision Result doğru")


# =============================================================================
# TEST 7: GEMİNİ API ÇAĞRISI (ONLINE)
# =============================================================================

def test_07_gemini_api():
    """Gerçek Gemini API çağrısı."""
    from ai.gemini_optimizer import GeminiOptimizer
    
    optimizer = GeminiOptimizer()
    
    # Basit prompt ile test
    prompt = """Sen bir kripto analistisin. Aşağıdaki bilgilere göre karar ver.
Coin: BTC, Fiyat: $97000, IC Yönü: LONG, IC Skoru: 75/100.
Yanıtını SADECE JSON ver:
```json
{"decision": "LONG", "confidence": 75, "reasoning": "Test yanıtı", "atr_multiplier": 1.5}
```"""
    
    response = optimizer._call_gemini(prompt)
    
    assert len(response) > 10, f"Yanıt çok kısa: {response}"
    
    # Parse edilebilmeli
    parsed = optimizer._parse_response(response)
    assert parsed['decision'] in ['LONG', 'SHORT', 'WAIT'], \
        f"Geçersiz karar: {parsed['decision']}"
    assert 0 <= parsed['confidence'] <= 100
    
    print(f"  Model: {optimizer.get_model_name()}")
    print(f"  Yanıt uzunluğu: {len(response)} char")
    print(f"  Karar: {parsed['decision']} | Güven: {parsed['confidence']}")
    print(f"  Gerekçe: {parsed['reasoning'][:50]}...")
    print(f"  ✓ Gemini API çalışıyor")


# =============================================================================
# TEST 8: FULL PİPELİNE (ONLINE)
# =============================================================================

def test_08_full_pipeline():
    """Tam pipeline: Input → Gate → Gemini → Decision."""
    from ai.gemini_optimizer import GeminiOptimizer, GateAction
    
    optimizer = GeminiOptimizer()
    
    # Senaryo A: Yüksek IC (≥70) → FULL_TRADE gate, Gemini'ye gidecek
    input_high = make_test_input(ic_confidence=78, ic_direction='SHORT')
    decision_a = optimizer.get_decision(input_high)
    
    assert decision_a.gate_action == GateAction.FULL_TRADE, \
        f"IC=78 → FULL_TRADE olmalı, {decision_a.gate_action}"
    assert decision_a.ic_score == 78
    assert decision_a.model_used != "gate_keeper"         # Gemini kullanılmış
    
    print(f"  Senaryo A (IC=78):")
    print(f"    Gate: {decision_a.gate_action.value}")
    print(f"    Karar: {decision_a.decision.value}")
    print(f"    Güven: {decision_a.confidence:.0f}")
    print(f"    Execute: {decision_a.should_execute()}")
    
    # Senaryo B: Düşük IC (< 55) → NO_TRADE gate, Gemini'ye gitmeyecek
    input_low = make_test_input(ic_confidence=40, ic_direction='NEUTRAL')
    decision_b = optimizer.get_decision(input_low)
    
    assert decision_b.gate_action == GateAction.NO_TRADE, \
        f"IC=40 → NO_TRADE olmalı, {decision_b.gate_action}"
    assert decision_b.decision.value == 'WAIT'
    assert decision_b.model_used == "gate_keeper"          # Gemini çağrılmamış
    assert not decision_b.should_execute()
    
    print(f"\n  Senaryo B (IC=40):")
    print(f"    Gate: {decision_b.gate_action.value}")
    print(f"    Karar: WAIT (Gate tarafından durduruldu)")
    print(f"    Execute: False")
    
    print(f"  ✓ Full pipeline çalışıyor")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    """Tüm testleri çalıştırır."""
    
    print("=" * 55)
    print("  ADIM 6: AI ENTRY OPTİMİZER TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    # Gemini API durumu
    from ai.gemini_optimizer import GeminiOptimizer
    optimizer = GeminiOptimizer()
    has_api = optimizer.is_available()
    
    print(f"\n  Gemini API: {'✅ Yapılandırılmış' if has_api else '❌ Key eksik (Test 7-8 atlanacak)'}")
    
    api_skip = None if has_api else "GEMINI_API_KEY yok"
    
    tests = [
        (1, "Gate Keeper: IC eşikleri",              test_01_gate_keeper,    None),
        (2, "Prompt Builder: Structured prompt",      test_02_prompt_builder, None),
        (3, "Response Parser: JSON parse",            test_03_response_parser, None),
        (4, "Validator: Aralık/type kontrolü",        test_04_validator,      None),
        (5, "IC Fallback: Gemini yoksa IC karar",     test_05_ic_fallback,    None),
        (6, "Decision Result: Execute/summary",       test_06_decision_result, None),
        (7, "Gemini API: Gerçek çağrı (ONLINE)",      test_07_gemini_api,     api_skip),
        (8, "Full Pipeline: Input→Gate→AI (ONLINE)",   test_08_full_pipeline,  api_skip),
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
    
    passed = 0
    failed = 0
    skipped = 0
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
        print(f"  {status} Test {num}: {name}")
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {len(results)} | ✅ {passed} | ❌ {failed} | ⏭️ {skipped}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        if skipped > 0:
            print(f"\n  ✅ Offline testler geçti. API testleri için GEMINI_API_KEY gerekli.")
            print(f"  .env dosyasına ekleyin: GEMINI_API_KEY=your_key_here")
        else:
            print(f"\n  🎉 ADIM 6 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 7 → Bitget Execution Engine")
    else:
        print(f"\n  ⚠️  {failed} test başarısız.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
