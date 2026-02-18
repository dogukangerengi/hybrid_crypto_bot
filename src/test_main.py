# =============================================================================
# ADIM 9: ANA ORKESTRASYON + SCHEDULER TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_main.py
#
# Test 1-7:  OFFLİNE (API key gerekmez — import, config, mock testleri)
# Test 8-10: ONLİNE (BITGET + GEMINI + TELEGRAM API key'leri gerekli)
#
# Test Listesi:
# 1.  İmport: Tüm modüller doğru import ediliyor mu?
# 2.  Config: AppConfig tüm bileşenleri yüklüyor mu?
# 3.  Pipeline Init: HybridTradingPipeline başlatılıyor mu?
# 4.  Balance Init: DRY RUN bakiye doğru atanıyor mu?
# 5.  Kill Switch: Drawdown kontrolü çalışıyor mu?
# 6.  Regime Detection: ADX bazlı rejim doğru mu?
# 7.  CLI Parser: Argümanlar doğru parse ediliyor mu?
# 8.  Scanner: CoinScanner market taraması (ONLINE)
# 9.  Single Coin: Tek coin analiz pipeline (ONLINE)
# 10. Full Cycle: Tam pipeline döngüsü (ONLINE)
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

# Loglama — WARNING seviyesi (test çıktısı temiz kalsın)
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
    """Tek testi çalıştır, süre ölç, hata yakala."""
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
# TEST 1: MODÜL İMPORTLARI
# =============================================================================

def test_01_imports():
    """Tüm modüller doğru import edilebiliyor mu?"""

    # Config
    from config import cfg, AppConfig
    assert isinstance(cfg, AppConfig), "cfg bir AppConfig instance'ı olmalı"
    print(f"  ✓ config: AppConfig yüklendi")

    # Scanner
    from scanner import CoinScanner, CoinScanResult
    print(f"  ✓ scanner: CoinScanner, CoinScanResult")

    # Data
    from data import BitgetFetcher, DataPreprocessor
    print(f"  ✓ data: BitgetFetcher, DataPreprocessor")

    # Indicators
    from indicators import IndicatorCalculator, IndicatorSelector, IndicatorScore
    print(f"  ✓ indicators: Calculator, Selector, Score")

    # AI
    from ai import GeminiOptimizer, AIAnalysisInput, AIDecisionResult, AIDecision, GateAction
    print(f"  ✓ ai: GeminiOptimizer, AIAnalysisInput, AIDecisionResult")

    # Execution
    from execution import RiskManager, BitgetExecutor, TradeCalculation, ExecutionResult
    print(f"  ✓ execution: RiskManager, BitgetExecutor")

    # Notifications
    from notifications import TelegramNotifier, AnalysisReport
    print(f"  ✓ notifications: TelegramNotifier, AnalysisReport")

    # Main orchestration
    from main import (
        HybridTradingPipeline,
        CoinAnalysisResult,
        CycleReport,
        CycleStatus,
        VERSION,
    )
    print(f"  ✓ main: HybridTradingPipeline v{VERSION}")

    print(f"\n  🎯 Toplam 7 modül + main başarıyla import edildi")


# =============================================================================
# TEST 2: CONFIG KONTROLÜ
# =============================================================================

def test_02_config():
    """AppConfig tüm bileşenleri doğru yüklüyor mu?"""
    from config import cfg

    # Exchange config
    assert hasattr(cfg, 'exchange'), "exchange config eksik"
    assert cfg.exchange.id == 'bitget', f"Exchange ID: {cfg.exchange.id} (beklenen: bitget)"
    assert cfg.exchange.market_type == 'swap', "Market type: swap olmalı"
    print(f"  ✓ Exchange: {cfg.exchange.id} ({cfg.exchange.market_type})")

    # Risk config
    assert hasattr(cfg, 'risk'), "risk config eksik"
    assert 0 < cfg.risk.risk_per_trade_pct <= 5, f"Risk/trade: {cfg.risk.risk_per_trade_pct}%"
    assert 1 <= cfg.risk.min_leverage <= cfg.risk.max_leverage, "Leverage aralığı hatalı"
    print(f"  ✓ Risk: {cfg.risk.risk_per_trade_pct}%/trade | Lev: {cfg.risk.min_leverage}-{cfg.risk.max_leverage}x")

    # Gate keeper config
    assert hasattr(cfg, 'gate'), "gate config eksik"
    assert cfg.gate.no_trade <= cfg.gate.full_trade, "Gate eşikleri tutarsız"
    print(f"  ✓ Gate: <{cfg.gate.no_trade} NO | >{cfg.gate.full_trade} FULL")

    # AI config
    assert hasattr(cfg, 'ai'), "ai config eksik"
    print(f"  ✓ AI: {cfg.ai.model} (configured: {cfg.ai.is_configured()})")

    # Telegram config
    assert hasattr(cfg, 'telegram'), "telegram config eksik"
    print(f"  ✓ Telegram: configured={cfg.telegram.is_configured()}")


# =============================================================================
# TEST 3: PİPELİNE İNİTİALİZASYON
# =============================================================================

def test_03_pipeline_init():
    """HybridTradingPipeline doğru başlatılıyor mu?"""
    from main import HybridTradingPipeline

    # DRY RUN modu
    pipeline = HybridTradingPipeline(
        dry_run=True,
        top_n=5,
        verbose=False,
    )

    # Tüm bileşenler oluşturulmuş mu?
    assert pipeline.scanner is not None, "Scanner None"
    assert pipeline.fetcher is not None, "Fetcher None"
    assert pipeline.preprocessor is not None, "Preprocessor None"
    assert pipeline.calculator is not None, "Calculator None"
    assert pipeline.selector is not None, "Selector None"
    assert pipeline.ai_optimizer is not None, "AI Optimizer None"
    assert pipeline.executor is not None, "Executor None"
    assert pipeline.notifier is not None, "Notifier None"

    # Dry run flag doğru mu?
    assert pipeline.dry_run == True, "Dry run flag hatalı"
    assert pipeline.top_n == 5, f"Top N: {pipeline.top_n} (beklenen: 5)"
    assert pipeline._kill_switch == False, "Kill switch başlangıçta kapalı olmalı"
    assert pipeline._cycle_count == 0, "Cycle count başlangıçta 0 olmalı"

    print(f"  ✓ Pipeline başlatıldı (DRY RUN)")
    print(f"  ✓ 8 modül initialize edildi")
    print(f"  ✓ Durum değişkenleri doğru")


# =============================================================================
# TEST 4: BAKİYE BAŞLATMA (DRY RUN)
# =============================================================================

def test_04_balance_init():
    """DRY RUN modda bakiye doğru atanıyor mu?"""
    from main import HybridTradingPipeline

    pipeline = HybridTradingPipeline(dry_run=True, verbose=False)
    success = pipeline._init_balance()

    assert success, "Bakiye başlatma başarısız"
    assert pipeline._balance == 75.0, f"Bakiye: {pipeline._balance} (beklenen: 75.0)"
    assert pipeline._initial_balance == 75.0, f"Initial: {pipeline._initial_balance}"
    assert pipeline._risk_manager is not None, "RiskManager oluşturulmamış"

    print(f"  ✓ DRY RUN bakiye: ${pipeline._balance}")
    print(f"  ✓ Initial bakiye: ${pipeline._initial_balance}")
    print(f"  ✓ RiskManager başlatıldı")


# =============================================================================
# TEST 5: KILL SWITCH
# =============================================================================

def test_05_kill_switch():
    """Drawdown bazlı kill switch doğru çalışıyor mu?"""
    from main import HybridTradingPipeline

    pipeline = HybridTradingPipeline(dry_run=True, verbose=False)
    pipeline._init_balance()  # $75 bakiye

    # --- Normal durum: drawdown yok ---
    assert pipeline._check_kill_switch() == False, "Normal durumda kill switch tetiklenmemeli"
    print(f"  ✓ Normal durum: Kill switch kapalı")

    # --- Küçük drawdown: %10 ---
    pipeline._balance = 67.50  # $75 → $67.50 = %10 DD
    assert pipeline._check_kill_switch() == False, "%10 DD'de kill switch tetiklenmemeli"
    print(f"  ✓ %10 drawdown: Kill switch kapalı")

    # --- Kritik drawdown: %16 (eşik %15) ---
    pipeline._balance = 63.00  # $75 → $63 = %16 DD
    # Notifier'ı devre dışı bırak (Telegram göndermemesin)
    pipeline.notifier = type('MockNotifier', (), {
        'is_configured': lambda self: False,
        'send_risk_alert_sync': lambda self, **kw: None,
    })()

    result = pipeline._check_kill_switch()
    assert result == True, "%16 DD'de kill switch tetiklenmeli"
    assert pipeline._kill_switch == True, "Kill switch flag True olmalı"
    print(f"  ✓ %16 drawdown: Kill switch AKTİF ✅")

    # --- Kill switch sonrası 2. kontrol (zaten aktif) ---
    result2 = pipeline._check_kill_switch()
    assert result2 == True, "Kill switch aktifken True dönmeli"
    print(f"  ✓ Tekrarlanan kontrol: Hâlâ aktif")


# =============================================================================
# TEST 6: REJİM TESPİTİ
# =============================================================================

def test_06_regime_detection():
    """ADX bazlı piyasa rejimi doğru tespit ediliyor mu?"""
    import pandas as pd
    from main import HybridTradingPipeline

    pipeline = HybridTradingPipeline(dry_run=True, verbose=False)

    # Trending UP: ADX=30, DI+ > DI-
    df_trend_up = pd.DataFrame({
        'ADX_14': [30.0],
        'DMP_14': [35.0],  # DI+ (bullish)
        'DMN_14': [15.0],  # DI- (bearish)
    })
    assert pipeline._detect_regime(df_trend_up) == 'trending_up'
    print(f"  ✓ ADX=30, DI+>DI- → trending_up")

    # Trending DOWN: ADX=28, DI- > DI+
    df_trend_down = pd.DataFrame({
        'ADX_14': [28.0],
        'DMP_14': [12.0],
        'DMN_14': [30.0],
    })
    assert pipeline._detect_regime(df_trend_down) == 'trending_down'
    print(f"  ✓ ADX=28, DI->DI+ → trending_down")

    # Ranging: ADX=15
    df_ranging = pd.DataFrame({'ADX_14': [15.0]})
    assert pipeline._detect_regime(df_ranging) == 'ranging'
    print(f"  ✓ ADX=15 → ranging")

    # Transitioning: ADX=22
    df_trans = pd.DataFrame({
        'ADX_14': [22.0],
        'DMP_14': [20.0],
        'DMN_14': [20.0],
    })
    assert pipeline._detect_regime(df_trans) == 'transitioning'
    print(f"  ✓ ADX=22 → transitioning")

    # Unknown: ADX yok
    df_empty = pd.DataFrame({'close': [100.0]})
    assert pipeline._detect_regime(df_empty) == 'unknown'
    print(f"  ✓ ADX yok → unknown")


# =============================================================================
# TEST 7: CLI PARSER
# =============================================================================

def test_07_cli_parser():
    """Argparse doğru parse ediyor mu?"""
    from main import parse_args

    # Varsayılan argümanlar (boş sys.argv simülasyonu)
    original_argv = sys.argv
    try:
        # Test 1: Varsayılanlar
        sys.argv = ['main.py']
        args = parse_args()
        assert args.dry_run == True, "Varsayılan dry_run=True olmalı"
        assert args.schedule == False, "Varsayılan schedule=False olmalı"
        assert args.interval == 60, f"Varsayılan interval: {args.interval}"
        assert args.symbol is None, "Varsayılan symbol=None olmalı"
        print(f"  ✓ Varsayılanlar: dry_run=True, interval=60, symbol=None")

        # Test 2: Schedule modu
        sys.argv = ['main.py', '--schedule', '-i', '15']
        args = parse_args()
        assert args.schedule == True, "Schedule flag aktif olmalı"
        assert args.interval == 15, f"Interval: {args.interval}"
        print(f"  ✓ Schedule: True, interval=15")

        # Test 3: Tek coin
        sys.argv = ['main.py', '--symbol', 'SOL']
        args = parse_args()
        assert args.symbol == 'SOL', f"Symbol: {args.symbol}"
        print(f"  ✓ Symbol: SOL")

        # Test 4: Canlı mod
        sys.argv = ['main.py', '--live', '--top', '10']
        args = parse_args()
        assert args.live == True, "Live flag aktif olmalı"
        assert args.top == 10, f"Top: {args.top}"
        print(f"  ✓ Live: True, top=10")

    finally:
        sys.argv = original_argv  # Orijinal argv'yi geri yükle


# =============================================================================
# TEST 8: SCANNER (ONLINE)
# =============================================================================

def test_08_scanner_online():
    """CoinScanner market taraması çalışıyor mu? (API gerekli)"""
    from main import HybridTradingPipeline

    pipeline = HybridTradingPipeline(dry_run=True, top_n=5, verbose=False)
    top_coins = pipeline._scan_market()

    assert len(top_coins) > 0, "Tarama sonucu boş"
    assert len(top_coins) <= 5, f"Top N aşıldı: {len(top_coins)}"

    # İlk coin'in alanları dolu mu?
    first = top_coins[0]
    assert first.symbol, "Symbol boş"
    assert first.volume_24h > 0, "Volume 0"
    assert first.composite_score > 0, "Score 0"

    print(f"  ✓ {len(top_coins)} coin tarandı")
    for i, c in enumerate(top_coins, 1):
        print(f"    #{i} {c.symbol}: Vol=${c.volume_24h:,.0f} | Score={c.composite_score:.1f}")


# =============================================================================
# TEST 9: TEK COİN ANALİZ (ONLINE)
# =============================================================================

def test_09_single_coin():
    """Tek coin analiz pipeline çalışıyor mu? (API gerekli)"""
    from main import HybridTradingPipeline

    pipeline = HybridTradingPipeline(dry_run=True, verbose=False)
    pipeline._init_balance()

    # BTC analiz et (en likit coin, her zaman veri var)
    analysis = pipeline._analyze_coin('BTC/USDT:USDT', 'BTC')

    assert analysis is not None, "Analiz None döndü"
    assert analysis.status in ('analyzed', 'skipped'), f"Status: {analysis.status}"

    if analysis.status == 'analyzed':
        assert analysis.price > 0, f"Fiyat: {analysis.price}"
        assert analysis.best_timeframe != "", "TF boş"
        assert 0 <= analysis.ic_confidence <= 100, f"IC: {analysis.ic_confidence}"
        assert analysis.ic_direction in ('LONG', 'SHORT', 'NEUTRAL'), f"Yön: {analysis.ic_direction}"
        assert analysis.atr > 0, f"ATR: {analysis.atr}"
        assert len(analysis.tf_rankings) > 0, "TF rankings boş"

        print(f"  ✓ BTC analiz tamamlandı")
        print(f"    Fiyat: ${analysis.price:,.2f}")
        print(f"    TF: {analysis.best_timeframe} | IC: {analysis.ic_confidence:.0f}")
        print(f"    Yön: {analysis.ic_direction} | Rejim: {analysis.market_regime}")
        print(f"    ATR: ${analysis.atr:.2f} ({analysis.atr_pct:.1f}%)")
        print(f"    Anlamlı: {analysis.significant_count} indikatör")
        print(f"    Süre: {analysis.elapsed:.1f}s")
    else:
        print(f"  ⚠️ BTC analizde sinyal bulunamadı (normal olabilir): {analysis.error}")


# =============================================================================
# TEST 10: TAM DÖNGÜ (ONLINE)
# =============================================================================

def test_10_full_cycle():
    """Tam pipeline döngüsü çalışıyor mu? (API gerekli, DRY RUN)"""
    from main import HybridTradingPipeline, CycleStatus

    pipeline = HybridTradingPipeline(
        dry_run=True,          # Paper trade — emir göndermez
        top_n=3,               # Sadece 3 coin (hız için)
        verbose=False,
    )

    # Bakiye başlat
    assert pipeline._init_balance(), "Bakiye başlatma başarısız"

    # Telegram'ı devre dışı bırak (test ortamında mesaj gönderme)
    pipeline.notifier = type('MockNotifier', (), {
        'is_configured': lambda self: False,
        'send_alert_sync': lambda self, **kw: None,
        'send_risk_alert_sync': lambda self, **kw: None,
    })()

    # Tam döngü çalıştır
    report = pipeline.run_cycle()

    assert report is not None, "Rapor None"
    assert report.status in (
        CycleStatus.SUCCESS,
        CycleStatus.PARTIAL,
        CycleStatus.NO_SIGNAL,
    ), f"Beklenmeyen status: {report.status}"

    assert report.elapsed > 0, "Süre 0 olamaz"
    assert report.balance > 0, "Bakiye 0 olamaz"

    print(f"  ✓ Tam döngü tamamlandı")
    print(f"    Status: {report.status.value}")
    print(f"    Taranan: {report.total_scanned}")
    print(f"    Analiz: {report.total_analyzed}")
    print(f"    Gate+: {report.total_above_gate}")
    print(f"    İşlem: {report.total_traded}")
    print(f"    Bakiye: ${report.balance:,.2f}")
    print(f"    Süre: {report.elapsed:.0f}s")

    if report.errors:
        print(f"    Hatalar ({len(report.errors)}):")
        for err in report.errors[:3]:
            print(f"      • {err[:60]}")


# =============================================================================
# ANA TEST RUNNER
# =============================================================================

def main():
    """Tüm testleri çalıştır."""
    print("=" * 55)
    print("  ADIM 9: ANA ORKESTRASYON + SCHEDULER TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # API durumu kontrol — online testler için
    from config import cfg
    has_bitget = cfg.exchange.is_configured()
    has_gemini = cfg.ai.is_configured()
    has_telegram = cfg.telegram.is_configured()

    print(f"\n  Bitget API : {'✅' if has_bitget else '❌ Key eksik'}")
    print(f"  Gemini API : {'✅' if has_gemini else '❌ Key eksik'}")
    print(f"  Telegram   : {'✅' if has_telegram else '❌ Key eksik'}")

    # Online testler için atlama sebebi
    online_skip = None if has_bitget else "BITGET_API_KEY yok (online testler atlanıyor)"

    tests = [
        (1,  "İmport: Tüm modüller",                test_01_imports,        None),
        (2,  "Config: AppConfig bileşenleri",        test_02_config,         None),
        (3,  "Pipeline Init: Modül başlatma",        test_03_pipeline_init,  None),
        (4,  "Balance: DRY RUN bakiye",              test_04_balance_init,   None),
        (5,  "Kill Switch: Drawdown kontrolü",       test_05_kill_switch,    None),
        (6,  "Regime: ADX bazlı rejim tespiti",      test_06_regime_detection, None),
        (7,  "CLI: Argüman parse",                   test_07_cli_parser,     None),
        (8,  "Scanner: Market tarama (ONLINE)",      test_08_scanner_online, online_skip),
        (9,  "Single Coin: BTC analiz (ONLINE)",     test_09_single_coin,    online_skip),
        (10, "Full Cycle: Tam döngü (ONLINE)",       test_10_full_cycle,     online_skip),
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
        print(f"  {status} Test {num:>2}: {name}")

    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {len(results)} | ✅ {passed} | ❌ {failed} | ⏭️ {skipped}")
    print(f"  Süre: {total_time:.1f}s")

    if failed == 0:
        if skipped > 0:
            print(f"\n  ✅ Offline testler geçti!")
            print(f"  API testleri için .env'de key'lerin olması gerekiyor.")
        else:
            print(f"\n  🎉 ADIM 9 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 10 → Paper Trading + Optimizasyon")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları kontrol edin.")

    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
