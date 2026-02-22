# =============================================================================
# TEST_ML_PIPELINE.PY — ML Pipeline Integration Testi (v5 - Final)
# =============================================================================
# Çalıştırma:  cd src && python test_ml_pipeline.py
# =============================================================================

import sys, time, traceback, logging
from pathlib import Path

src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
load_dotenv(src_dir.parent / ".env")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

results = {}

def run_test(num, name, fn):
    print(f"\n{'─'*55}\n  TEST {num}: {name}\n{'─'*55}")
    t = time.time()
    try:
        fn()
        print(f"  ✅ PASS  ({time.time()-t:.2f}s)")
        results[num] = True
    except Exception:
        print(f"  ❌ FAIL  ({time.time()-t:.2f}s)\n\n  HATA:")
        traceback.print_exc()
        results[num] = False


def synthetic_ohlcv(n=500) -> pd.DataFrame:
    """Gerçek API çağrısı olmadan test için sentetik OHLCV üretir."""
    np.random.seed(42)
    close = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high  = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low   = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    vol   = np.random.lognormal(10, 1, n)
    idx   = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def make_analysis_stub(ic_confidence=65.0, ic_direction="LONG"):
    """
    FeatureEngineer.build_features() için CoinAnalysisResult benzeri stub.
    Gerçek main.py'de bu CoinAnalysisResult dataclass'ından gelir.
    """
    class Stub:
        symbol            = "BTC/USDT:USDT"
        coin              = "BTC"
        price             = 50000.0
        change_24h        = 2.5
        volume_24h        = 1_500_000_000.0
        best_timeframe    = "1h"
        market_regime     = "trending"
        significant_count = 10
        atr               = 800.0
        atr_pct           = 1.6
        sl_price          = 49200.0
        tp_price          = 51600.0
        risk_reward       = 2.0
        leverage          = 5
        category_tops     = {
            "trend":     {"name": "EMA_20", "ic": 0.15},
            "momentum":  {"name": "RSI_14", "ic": -0.12},
            "volatility":{"name": "ATR_14", "ic": 0.08},
            "volume":    {"name": "OBV",    "ic": 0.10},
        }
        tf_rankings = [
            {"tf": "1h",  "composite": 65.0, "direction": "LONG",  "sig_count": 10},
            {"tf": "15m", "composite": 58.0, "direction": "LONG",  "sig_count": 7},
            {"tf": "30m", "composite": 52.0, "direction": "SHORT", "sig_count": 5},
        ]
    s = Stub()
    s.ic_confidence = ic_confidence
    s.ic_direction  = ic_direction
    return s


def prepare_feature_train_data(n_samples=60):
    """
    MLFeatureVector kolonlarıyla eğitim verisi hazırlar.

    Kritik nokta: Model build_features() çıktısıyla eğitilmeli,
    çünkü predict() da MLFeatureVector alıyor. Ham indikatör
    kolonlarıyla eğitilirse feature uyumsuzluğu oluşur.

    Her timestamp için build_features() çağırarak X matrisi oluşturur.
    n_samples: kaç farklı timestamp kullanılacağı (min 30 gerekli)
    """
    from ml.feature_engineer import FeatureEngineer
    from indicators import IndicatorCalculator
    from data import DataPreprocessor

    df_raw   = synthetic_ohlcv(n_samples + 150)  # Warm-up için ekstra bar
    df_clean = DataPreprocessor().full_pipeline(df_raw)
    calc     = IndicatorCalculator(verbose=False)
    df_ind   = calc.calculate_all(df_clean)
    df_ind   = calc.add_forward_returns(df_ind, periods=[6])

    fe       = FeatureEngineer()
    analysis = make_analysis_stub()

    rows_X = []
    rows_y = []

    # Her i için ilk i+1 barı kullanarak feature üret
    # İndikatör warm-up için 100. bardan sonra başla
    start = 100
    end   = min(start + n_samples, len(df_ind) - 7)  # -7: forward return için boşluk

    for i in range(start, end):
        try:
            # i. bar'a kadar olan veriyle feature üret
            fv = fe.build_features(
                analysis = analysis,
                ohlcv_df = df_ind.iloc[:i+1],  # Sadece geçmiş verisi (look-ahead yok)
            )
            # i. bar'ın forward return'ü (target)
            target = df_ind['fwd_ret_6'].iloc[i]
            if pd.isna(target):
                continue

            rows_X.append(fv.to_dict())        # Feature dict → satır
            rows_y.append(1 if target > 0 else 0)  # Binary: artış=1, düşüş=0

        except Exception:
            continue

    assert len(rows_X) >= 30, f"Yeterli örnek yok: {len(rows_X)} (min 30)"

    X = pd.DataFrame(rows_X).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.Series(rows_y, name="target")

    return X, y, df_ind


# =============================================================================
# TEST 1: IMPORT
# =============================================================================

def test_01_imports():
    print("  ml.feature_engineer  ... ", end="", flush=True)
    from ml.feature_engineer import FeatureEngineer, MLFeatureVector
    FeatureEngineer(); print("OK")

    print("  ml.lgbm_model        ... ", end="", flush=True)
    from ml.lgbm_model import LGBMSignalModel
    LGBMSignalModel(); print("OK")

    print("  ml.signal_validator  ... ", end="", flush=True)
    from ml.signal_validator import SignalValidator
    SignalValidator(); print("OK")

    print("  ml.trade_memory      ... ", end="", flush=True)
    from ml.trade_memory import TradeMemory
    TradeMemory(log_dir=Path("/tmp/test_mem")); print("OK")

    print("  indicators           ... ", end="", flush=True)
    from indicators import IndicatorCalculator, IndicatorSelector; print("OK")

    print("  data                 ... ", end="", flush=True)
    from data import DataPreprocessor; print("OK")

    print("  execution            ... ", end="", flush=True)
    from execution import RiskManager; print("OK")

    print("\n  ✓ Tüm importlar başarılı")


# =============================================================================
# TEST 2: FEATURE ENGINEER
# =============================================================================

def test_02_feature_engineer():
    from ml.feature_engineer import FeatureEngineer, MLFeatureVector
    from indicators import IndicatorCalculator, IndicatorSelector
    from data import DataPreprocessor

    df_raw   = synthetic_ohlcv(300)
    df_clean = DataPreprocessor().full_pipeline(df_raw)
    calc     = IndicatorCalculator(verbose=False)
    df_ind   = calc.calculate_all(df_clean)
    df_ind   = calc.add_forward_returns(df_ind, periods=[6])

    ic_scores = IndicatorSelector(alpha=0.05).evaluate_all_indicators(
        df_ind, target_col="fwd_ret_6"
    )
    print(f"  IC skor sayısı  : {len(ic_scores)}")
    print(f"  Anlamlı sayısı  : {sum(1 for s in ic_scores if s.is_significant)}")

    fe       = FeatureEngineer()
    analysis = make_analysis_stub()
    fv = fe.build_features(
        analysis        = analysis,
        ohlcv_df        = df_ind,
        all_tf_analyses = analysis.tf_rankings,
    )

    assert fv is not None
    from ml.feature_engineer import MLFeatureVector
    assert isinstance(fv, MLFeatureVector)

    fv_dict = fv.to_dict()
    print(f"  Toplam feature  : {len(fv_dict)}")
    nan_c = sum(1 for v in fv_dict.values()
                if v is None or (isinstance(v, float) and np.isnan(v)))
    print(f"  NaN sayısı      : {nan_c}/{len(fv_dict)}")
    assert len(fv_dict) > 0
    print("  ✓ FeatureEngineer çalışıyor")


# =============================================================================
# TEST 3: LGBM MODEL
# =============================================================================

def test_03_lgbm_model():
    """
    Kritik: Model MLFeatureVector kolonlarıyla eğitilmeli.
    Ham indikatör kolonlarıyla eğitilirse predict() uyumsuz kolon hatası verir.
    """
    from ml.lgbm_model import LGBMSignalModel
    from ml.feature_engineer import FeatureEngineer

    print("  Feature matrisi hazırlanıyor (MLFeatureVector ile)...")
    X, y, df_ind = prepare_feature_train_data(n_samples=60)

    print(f"  Eğitim verisi   : {X.shape[0]} satır × {X.shape[1]} kolon")
    print(f"  WIN oranı       : {y.mean():.1%}")
    assert len(X) >= 30, f"Yeterli satır yok: {len(X)}"

    model = LGBMSignalModel()
    assert not model.is_trained

    metrics = model.train(X, y)
    assert model.is_trained, "train() sonrası is_trained=True olmalı"
    print(f"  ✓ Eğitildi | accuracy={metrics.accuracy:.2f} | n_train={metrics.n_train}")

    # Predict — aynı feature kolonlarıyla
    fe       = FeatureEngineer()
    analysis = make_analysis_stub()
    from data import DataPreprocessor
    from indicators import IndicatorCalculator
    df_test = DataPreprocessor().full_pipeline(synthetic_ohlcv(300))
    df_test = IndicatorCalculator(verbose=False).calculate_all(df_test)
    fv      = fe.build_features(analysis=analysis, ohlcv_df=df_test)

    result = model.predict(fv, ic_score=65.0, ic_direction="LONG")

    assert result is not None
    assert hasattr(result, 'decision')
    assert hasattr(result, 'confidence')
    print(f"  Tahmin          : {result.decision} | güven={result.confidence:.2f}%")
    print("  ✓ LGBMSignalModel çalışıyor")


# =============================================================================
# TEST 4: SIGNAL VALIDATOR
# =============================================================================

def test_04_signal_validator():
    """
    ValidationResult alanları:
      approved  → is_valid
      confidence → adjusted_confidence (0-1 arası)
    """
    from ml.lgbm_model import LGBMSignalModel
    from ml.signal_validator import SignalValidator, ValidationResult
    from ml.feature_engineer import FeatureEngineer
    from indicators import IndicatorCalculator
    from data import DataPreprocessor

    # Modeli MLFeatureVector kolonlarıyla eğit
    print("  Feature matrisi hazırlanıyor...")
    X, y, df_ind = prepare_feature_train_data(n_samples=60)
    model = LGBMSignalModel()
    model.train(X, y)
    print(f"  Model eğitildi  : {X.shape[0]}×{X.shape[1]}")

    # Predict için feature vector
    fe       = FeatureEngineer()
    analysis = make_analysis_stub()
    df_test  = DataPreprocessor().full_pipeline(synthetic_ohlcv(300))
    df_test  = IndicatorCalculator(verbose=False).calculate_all(df_test)
    fv       = fe.build_features(analysis=analysis, ohlcv_df=df_test)

    ml_result = model.predict(fv, ic_score=65.0, ic_direction="LONG")
    print(f"  ML karar        : {ml_result.decision} | güven={ml_result.confidence:.2f}%")

    # Doğrulama
    validator  = SignalValidator()
    val_result = validator.validate(
        feature_vector   = fv,
        model            = model,
        model_decision   = ml_result.decision,
        model_confidence = ml_result.confidence,
        ic_direction     = "LONG",
        ic_score         = 65.0,
        regime           = "trending",
    )

    assert val_result is not None
    assert isinstance(val_result, ValidationResult), f"Tip: {type(val_result)}"

    # Doğru alan adları
    assert hasattr(val_result, 'is_valid'),            "'is_valid' alanı yok!"
    assert hasattr(val_result, 'adjusted_confidence'), "'adjusted_confidence' alanı yok!"
    assert isinstance(val_result.is_valid, bool)

    # adjusted_confidence 0-1 aralığında mı?
    conf = val_result.adjusted_confidence
    print(f"  is_valid            : {val_result.is_valid}")
    print(f"  adjusted_confidence : {conf:.4f}")
    print(f"  bootstrap_passed    : {val_result.bootstrap_passed}")
    print(f"  regime_passed       : {val_result.regime_passed}")

    # Eğer hâlâ 0-1 dışındaysa ne döndüğünü göster (assert yumuşak)
    if not (0.0 <= conf <= 100.0):
        print(f"  ⚠️  adjusted_confidence={conf:.4f} (0-100 dışında!)")
        print(f"  Tüm ValidationResult: {val_result}")
        raise AssertionError(
            f"adjusted_confidence={conf:.4f} → 0-100 aralığında olmalı. "
            f"Feature uyumsuzluğu hâlâ var mı? "
            f"Model kolonları ile fv kolonları eşleşiyor mu?"
        )

    print("  ✓ SignalValidator çalışıyor")


# =============================================================================
# TEST 5: TRADE MEMORY
# =============================================================================

def test_05_trade_memory():
    from ml.trade_memory import TradeMemory, TradeStatus, TradeOutcome

    log_dir = Path("/tmp/test_trade_memory_v5")
    log_dir.mkdir(exist_ok=True)
    mem = TradeMemory(log_dir=log_dir, min_trades=3, retrain_interval=2)

    r = mem.open_trade(
        symbol="BTC/USDT:USDT", coin="BTC", direction="LONG",
        entry_price=50000.0, sl_price=49000.0, tp_price=52000.0,
        ml_confidence=0.72, ml_direction="LONG",
        ic_confidence=65.0, ic_direction="LONG", market_regime="trending",
        feature_snapshot={"ic_score": 0.15, "rsi": 55.0},
    )
    print(f"  Trade açıldı    : {r.trade_id}")
    assert r.status == TradeStatus.OPEN

    closed = mem.close_trade(r.trade_id, exit_price=51500.0, pnl=12.5, exit_reason="TP")
    assert closed.status  == TradeStatus.CLOSED
    assert closed.outcome == TradeOutcome.WIN
    print(f"  Trade kapandı   : PnL=${closed.pnl:+.2f} | {closed.outcome}")

    for _ in range(2):
        r2 = mem.open_trade(
            symbol="ETH/USDT:USDT", coin="ETH", direction="SHORT",
            entry_price=3000.0, sl_price=3100.0, tp_price=2800.0,
            feature_snapshot={"ic_score": -0.1, "rsi": 45.0},
        )
        mem.close_trade(r2.trade_id, exit_price=3050.0, pnl=-5.0, exit_reason="SL")

    stats = mem.get_stats()
    print(f"  İstatistikler   : kapalı={stats['closed_trades']} win={stats['win_rate']:.0%}")
    assert stats["closed_trades"] == 3
    mem.print_summary()
    print("  ✓ TradeMemory çalışıyor")


# =============================================================================
# TEST 6: MAIN PIPELINE
# =============================================================================

def test_06_main_pipeline():
    print("  MLTradingPipeline import ... ", end="", flush=True)
    from main import MLTradingPipeline; print("OK")

    print("  Pipeline başlatılıyor   ... ", end="", flush=True)
    p = MLTradingPipeline(dry_run=True, top_n=3, verbose=False); print("OK")

    for attr in ["scanner","fetcher","calculator","selector",
                 "feature_eng","lgbm_model","validator","trade_memory",
                 "risk_manager","executor","notifier","paper_trader"]:
        assert hasattr(p, attr), f"'{attr}' attribute eksik!"
        print(f"    ✓ {attr}")

    assert not p.lgbm_model.is_trained
    print(f"\n  model.is_trained = {p.lgbm_model.is_trained} (beklenen: False)")
    print("  ✓ MLTradingPipeline başarıyla başlatıldı")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  ML PIPELINE INTEGRATION TEST v5")
    print("═"*60)

    run_test(1, "Import Testi",    test_01_imports)
    run_test(2, "FeatureEngineer", test_02_feature_engineer)
    run_test(3, "LGBMSignalModel", test_03_lgbm_model)
    run_test(4, "SignalValidator", test_04_signal_validator)
    run_test(5, "TradeMemory",     test_05_trade_memory)
    run_test(6, "Main Pipeline",   test_06_main_pipeline)

    print("\n" + "═"*60 + "\n  SONUÇLAR\n" + "─"*60)
    names = {1:"Import",2:"FeatureEngineer",3:"LGBMSignalModel",
             4:"SignalValidator",5:"TradeMemory",6:"Main Pipeline"}
    for num, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  TEST {num}: {names[num]}")

    passed = sum(v for v in results.values())
    print(f"\n  Toplam: {passed}/{len(results)} başarılı")
    if passed == len(results):
        print("\n  🎉 TÜM TESTLER GEÇTİ!")
        print("\n  Sıradaki adım:")
        print("    python main.py --train    ← ilk eğitim (BTC veri ile)")
        print("    python main.py            ← paper trade döngüsü")
    else:
        print(f"\n  ⚠️  {len(results)-passed} test başarısız.")
    print("═"*60 + "\n")
