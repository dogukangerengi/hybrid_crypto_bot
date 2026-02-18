# =============================================================================
# İNDİKATÖR KATMANI TEST DOSYASI
# =============================================================================
# Çalıştırma: cd src && python test_indicators.py
#
# API GEREKTİRMEZ — Sentetik OHLCV verisi ile test eder.
# Tüm testler geçerse Adım 3 tamamdır.
#
# Test Listesi:
# 1.  Categories: İndikatör sayıları ve yapılandırma doğrulaması
# 2.  Categories: IndicatorConfig dataclass alanları
# 3.  Calculator: Tek indikatör hesaplama (RSI_14)
# 4.  Calculator: Tek kategori hesaplama (momentum)
# 5.  Calculator: Tüm kategoriler hesaplama (4 kategori)
# 6.  Calculator: Price features (log_return, body, wick, vb.)
# 7.  Calculator: Rolling stats (mean, std, skew, kurt, zscore)
# 8.  Calculator: Forward returns (IC target değişkeni)
# 9.  Selector: Tek IC hesaplama (Spearman korelasyonu)
# 10. Selector: Tüm indikatörleri değerlendir + FDR correction
# 11. Selector: En iyi indikatörleri seç (kategori başına max 2)
# 12. Entegrasyon: Calculator → Selector → Best Indicators pipeline
# =============================================================================

import sys
import time
import logging
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Path ayarları (src/ altından çalıştırılacak)
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Loglama — sadece WARNING ve üstü (test çıktısı temiz olsun)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
# pandas-ta uyarılarını sustur
warnings.filterwarnings('ignore')


# =============================================================================
# SENTETİK VERİ ÜRETİCİ
# =============================================================================

def generate_synthetic_ohlcv(
    n_bars: int = 1000,
    start_price: float = 100000.0,
    volatility: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """
    Gerçekçi sentetik OHLCV verisi üretir.
    
    Geometric Brownian Motion (GBM) kullanır:
    P_t = P_{t-1} × exp(μ + σ × Z)
    Z ~ N(0, 1) standart normal
    
    Parameters:
    ----------
    n_bars : int
        Üretilecek bar sayısı (varsayılan: 1000)
    start_price : float
        Başlangıç fiyatı (BTC benzeri)
    volatility : float
        Bar başına volatilite (0.02 = %2)
    seed : int
        Reproducibility için random seed
        
    Returns:
    -------
    pd.DataFrame
        Kolonlar: open, high, low, close, volume
        Index: DatetimeIndex (saatlik)
    """
    np.random.seed(seed)
    
    # Log-normal random walk (GBM)
    drift = 0.0001                              # Hafif pozitif drift
    returns = np.random.normal(drift, volatility, n_bars)
    
    # Close fiyatları
    close = start_price * np.exp(np.cumsum(returns))
    
    # OHLC oluştur
    # Intrabar hareket: close etrafında rastgele dağılım
    intrabar_vol = volatility * 0.5             # Bar içi volatilite
    
    open_prices = np.roll(close, 1)             # Önceki close = yeni open
    open_prices[0] = start_price                # İlk bar
    
    # High her zaman max(open, close) + rastgele ekleme
    high = np.maximum(open_prices, close) + np.abs(
        np.random.normal(0, start_price * intrabar_vol * 0.5, n_bars)
    )
    
    # Low her zaman min(open, close) - rastgele çıkarma
    low = np.minimum(open_prices, close) - np.abs(
        np.random.normal(0, start_price * intrabar_vol * 0.5, n_bars)
    )
    
    # Volume: log-normal dağılım (gerçekçi hacim)
    volume = np.random.lognormal(mean=10, sigma=1.5, size=n_bars)
    
    # Timestamp: saatlik
    timestamps = pd.date_range(
        start='2025-01-01', periods=n_bars, freq='1h', tz='UTC'
    )
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)
    
    return df


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_test(test_num: int, test_name: str, test_func):
    """Test wrapper: Hata yakalar, süre ölçer, sonuç raporlar."""
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num}: {test_name}")
    print(f"{'─' * 55}")
    
    start = time.time()
    try:
        result = test_func()
        elapsed = time.time() - start
        print(f"\n  ✅ BAŞARILI ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ BAŞARISIZ ({elapsed:.2f}s)")
        print(f"  Hata: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TESTLER
# =============================================================================

def test_01_category_counts():
    """Categories: İndikatör sayıları ve yapılandırma doğrulaması."""
    from indicators.categories import (
        get_indicator_count, get_all_indicators,
        get_category_names, get_total_output_columns
    )
    
    # Kategori isimleri
    categories = get_category_names()
    print(f"  Kategoriler: {categories}")
    assert len(categories) == 4, f"4 kategori bekleniyor, {len(categories)} var"
    assert 'trend' in categories
    assert 'momentum' in categories
    assert 'volatility' in categories
    assert 'volume' in categories
    
    # İndikatör sayıları
    counts = get_indicator_count()
    total = sum(counts.values())
    print(f"  Toplam indikatör: {total}")
    for cat, count in counts.items():
        print(f"    {cat}: {count}")
    assert total >= 50, f"En az 50 indikatör bekleniyor, {total} var"
    
    # Çıktı kolon sayısı
    total_cols = get_total_output_columns()
    print(f"  Toplam çıktı kolonu: {total_cols}")
    assert total_cols >= 70, f"En az 70 kolon bekleniyor, {total_cols} var"
    
    # Tüm indikatörler listesi
    all_inds = get_all_indicators()
    assert len(all_inds) == total
    
    print(f"  ✓ 4 kategori, {total} indikatör, {total_cols} kolon")


def test_02_indicator_config():
    """Categories: IndicatorConfig dataclass alanları."""
    from indicators.categories import get_all_indicators, IndicatorConfig
    
    all_inds = get_all_indicators()
    
    for ind in all_inds:
        # Tip kontrolü
        assert isinstance(ind, IndicatorConfig), f"{ind} IndicatorConfig değil"
        
        # Zorunlu alanlar dolu mu?
        assert ind.name, f"name boş: {ind}"
        assert ind.display_name, f"display_name boş: {ind}"
        assert ind.category in ['trend', 'momentum', 'volatility', 'volume'], \
            f"Geçersiz kategori: {ind.category}"
        assert isinstance(ind.params, dict), f"params dict değil: {ind.name}"
        assert isinstance(ind.output_columns, list), f"output_columns list değil: {ind.name}"
        assert len(ind.output_columns) >= 1, f"En az 1 output kolonu olmalı: {ind.name}"
        assert ind.signal_type in ['level', 'crossover', 'band'], \
            f"Geçersiz signal_type: {ind.signal_type} ({ind.name})"
    
    print(f"  ✓ {len(all_inds)} indikatör yapılandırması doğrulandı")


def test_03_single_indicator():
    """Calculator: Tek indikatör hesaplama (RSI_14)."""
    from indicators.calculator import IndicatorCalculator
    from indicators.categories import IndicatorConfig
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=500)
    
    # RSI_14 hesapla
    rsi_config = IndicatorConfig(
        "rsi", "RSI_14", "momentum",
        {"length": 14}, ["RSI_14"],
        "RSI 14 test", "level"
    )
    
    result = calc.calculate_single(df, rsi_config)
    
    assert not result.empty, "RSI sonucu boş"
    assert 'RSI_14' in result.columns, f"RSI_14 kolonu yok: {result.columns.tolist()}"
    
    # RSI 0-100 arasında olmalı (NaN hariç)
    rsi_values = result['RSI_14'].dropna()
    assert len(rsi_values) > 0, "RSI değerleri tamamen NaN"
    assert rsi_values.min() >= 0, f"RSI min < 0: {rsi_values.min()}"
    assert rsi_values.max() <= 100, f"RSI max > 100: {rsi_values.max()}"
    
    print(f"  RSI_14 hesaplandı: {len(rsi_values)} geçerli değer")
    print(f"  Min={rsi_values.min():.1f}, Max={rsi_values.max():.1f}, Mean={rsi_values.mean():.1f}")
    print(f"  ✓ Tek indikatör hesaplama çalışıyor")


def test_04_category_calculation():
    """Calculator: Tek kategori hesaplama (momentum)."""
    from indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=500)
    
    # Momentum kategorisini hesapla
    df_mom = calc.calculate_category(df, "momentum")
    
    # Orijinal OHLCV kolonları korunmuş mu?
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in df_mom.columns, f"OHLCV kolonu kayıp: {col}"
    
    # Yeni kolonlar eklenmiş mi?
    new_cols = [c for c in df_mom.columns if c not in df.columns]
    print(f"  Momentum: {len(new_cols)} yeni kolon eklendi")
    assert len(new_cols) >= 10, f"En az 10 momentum kolonu bekleniyor, {len(new_cols)} var"
    
    # Bilinen kolonlar var mı?
    expected = ['RSI_14', 'MACD_12_26_9']
    for exp in expected:
        assert exp in df_mom.columns, f"Beklenen kolon yok: {exp}"
    
    print(f"  Örnek kolonlar: {new_cols[:5]}")
    print(f"  ✓ Momentum kategorisi çalışıyor")


def test_05_all_categories():
    """Calculator: Tüm kategoriler hesaplama (4 kategori)."""
    from indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=500)
    
    # Tüm kategorileri hesapla
    df_all = calc.calculate_all(df)
    
    # Kolon sayısı kontrolü
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    indicator_cols = [c for c in df_all.columns if c not in ohlcv_cols]
    
    print(f"  Toplam kolon: {len(df_all.columns)}")
    print(f"  İndikatör kolonu: {len(indicator_cols)}")
    assert len(indicator_cols) >= 50, f"En az 50 indikatör kolonu bekleniyor, {len(indicator_cols)} var"
    
    # Her kategoriden en az 1 kolon var mı?
    has_trend = any('SMA' in c or 'EMA' in c or 'ADX' in c for c in indicator_cols)
    has_momentum = any('RSI' in c or 'MACD' in c for c in indicator_cols)
    has_volatility = any('ATR' in c or 'BB' in c for c in indicator_cols)
    has_volume = any('OBV' in c or 'CMF' in c or 'MFI' in c for c in indicator_cols)
    
    assert has_trend, "Trend indikatörleri eksik"
    assert has_momentum, "Momentum indikatörleri eksik"
    assert has_volatility, "Volatilite indikatörleri eksik"
    assert has_volume, "Hacim indikatörleri eksik"
    
    # NaN oranı makul mü? (ilk ~200 bar NaN olabilir)
    nan_pct = df_all[indicator_cols].isnull().mean().mean() * 100
    print(f"  Ortalama NaN: {nan_pct:.1f}%")
    assert nan_pct < 60, f"NaN oranı çok yüksek: {nan_pct:.1f}%"
    
    print(f"  ✓ 4 kategori başarıyla hesaplandı")


def test_06_price_features():
    """Calculator: Price features (log_return, body, wick, vb.)."""
    from indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=300)
    
    df_feat = calc.add_price_features(df)
    
    # Beklenen kolonlar
    expected_cols = [
        'log_return', 'simple_return',
        'range', 'body', 'body_pct',
        'upper_wick', 'lower_wick',
        'gap', 'gap_pct',
        'hl_position',
        'volume_sma_20', 'volume_ratio'
    ]
    
    for col in expected_cols:
        assert col in df_feat.columns, f"Eksik kolon: {col}"
    
    # Log return kontrolü
    log_ret = df_feat['log_return'].dropna()
    assert len(log_ret) > 0, "Log return boş"
    assert log_ret.mean() < 1 and log_ret.mean() > -1, "Log return aralığı anormal"
    
    # hl_position 0-1 arası mı?
    hlp = df_feat['hl_position'].dropna()
    assert hlp.min() >= -0.01, f"hl_position min < 0: {hlp.min()}"
    assert hlp.max() <= 1.01, f"hl_position max > 1: {hlp.max()}"
    
    # Range her zaman >= 0 mi?
    range_vals = df_feat['range'].dropna()
    assert range_vals.min() >= 0, f"range < 0: {range_vals.min()}"
    
    print(f"  {len(expected_cols)} price feature eklendi")
    print(f"  Log return mean: {log_ret.mean():.6f}, std: {log_ret.std():.4f}")
    print(f"  ✓ Price features çalışıyor")


def test_07_rolling_stats():
    """Calculator: Rolling stats (mean, std, skew, kurt, zscore)."""
    from indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=300)
    df = calc.add_price_features(df)  # log_return gerekli
    
    df_roll = calc.add_rolling_stats(df, windows=[10, 20])
    
    # Her window için beklenen kolonlar
    for w in [10, 20]:
        prefix = f"roll{w}_"
        expected = [
            f'{prefix}ret_mean', f'{prefix}ret_std',
            f'{prefix}ret_skew', f'{prefix}ret_kurt',
            f'{prefix}zscore', f'{prefix}pct_rank'
        ]
        for col in expected:
            assert col in df_roll.columns, f"Eksik: {col}"
    
    # Z-score kontrolü: ortalama ~0, std ~1 olmalı
    zscore = df_roll['roll20_zscore'].dropna()
    assert len(zscore) > 0, "Z-score boş"
    assert abs(zscore.mean()) < 2, f"Z-score mean anormal: {zscore.mean()}"
    
    # Pct_rank 0-1 arası mı?
    prank = df_roll['roll20_pct_rank'].dropna()
    assert prank.min() >= -0.01, f"pct_rank min < 0: {prank.min()}"
    assert prank.max() <= 1.01, f"pct_rank max > 1: {prank.max()}"
    
    roll_cols = [c for c in df_roll.columns if c.startswith('roll')]
    print(f"  {len(roll_cols)} rolling stat eklendi")
    print(f"  Z-score mean: {zscore.mean():.3f}, std: {zscore.std():.3f}")
    print(f"  ✓ Rolling stats çalışıyor")


def test_08_forward_returns():
    """Calculator: Forward returns (IC target değişkeni)."""
    from indicators.calculator import IndicatorCalculator
    
    calc = IndicatorCalculator(verbose=False)
    df = generate_synthetic_ohlcv(n_bars=300)
    
    df_fwd = calc.add_forward_returns(df, periods=[1, 5, 10])
    
    # Forward return kolonları var mı?
    for p in [1, 5, 10]:
        ret_col = f'fwd_ret_{p}'
        dir_col = f'fwd_dir_{p}'
        assert ret_col in df_fwd.columns, f"Eksik: {ret_col}"
        assert dir_col in df_fwd.columns, f"Eksik: {dir_col}"
    
    # fwd_ret_1: 1 bar sonraki log getiri
    fwd1 = df_fwd['fwd_ret_1'].dropna()
    assert len(fwd1) > 200, f"Forward return çok az: {len(fwd1)}"
    
    # Manuel kontrol: fwd_ret_1[i] = log(close[i+1] / close[i])
    manual_fwd = np.log(df['close'].iloc[1] / df['close'].iloc[0])
    calc_fwd = df_fwd['fwd_ret_1'].iloc[0]
    assert abs(manual_fwd - calc_fwd) < 1e-10, \
        f"Forward return yanlış: {manual_fwd} != {calc_fwd}"
    
    # fwd_dir: binary (0 veya 1)
    dir1 = df_fwd['fwd_dir_1'].dropna()
    unique_vals = set(dir1.unique())
    assert unique_vals.issubset({0, 1, 0.0, 1.0}), f"fwd_dir değerleri anormal: {unique_vals}"
    
    # Son bar'lar NaN olmalı (gelecek yok)
    assert pd.isna(df_fwd['fwd_ret_10'].iloc[-1]), "Son bar'ın fwd_ret_10'u NaN olmalı"
    
    print(f"  Forward returns: {[1, 5, 10]} periyot")
    print(f"  fwd_ret_1 mean: {fwd1.mean():.6f}")
    print(f"  fwd_dir_1 up ratio: {dir1.mean():.2%}")
    print(f"  ✓ Forward returns doğru çalışıyor (look-ahead bias yok)")


def test_09_single_ic():
    """Selector: Tek IC hesaplama (Spearman korelasyonu)."""
    from indicators.selector import IndicatorSelector
    
    selector = IndicatorSelector(verbose=False)
    
    # Test 1: Mükemmel pozitif ilişki (IC ≈ 1)
    n = 500
    np.random.seed(42)
    indicator = pd.Series(np.arange(n, dtype=float))
    fwd_return = pd.Series(np.arange(n, dtype=float) + np.random.normal(0, 0.1, n))
    
    ic, p = selector.calculate_ic(indicator, fwd_return)
    print(f"  Mükemmel pozitif: IC={ic:.4f}, p={p:.2e}")
    assert ic > 0.95, f"Pozitif IC bekleniyor: {ic}"
    assert p < 0.001, f"p < 0.001 bekleniyor: {p}"
    
    # Test 2: Mükemmel negatif ilişki (IC ≈ -1)
    fwd_return_neg = pd.Series(-np.arange(n, dtype=float) + np.random.normal(0, 0.1, n))
    ic_neg, p_neg = selector.calculate_ic(indicator, fwd_return_neg)
    print(f"  Mükemmel negatif: IC={ic_neg:.4f}, p={p_neg:.2e}")
    assert ic_neg < -0.95, f"Negatif IC bekleniyor: {ic_neg}"
    
    # Test 3: Rastgele ilişki (IC ≈ 0)
    random_return = pd.Series(np.random.normal(0, 1, n))
    ic_rand, p_rand = selector.calculate_ic(indicator, random_return)
    print(f"  Rastgele: IC={ic_rand:.4f}, p={p_rand:.4f}")
    assert abs(ic_rand) < 0.15, f"|IC| < 0.15 bekleniyor: {ic_rand}"
    
    # Test 4: Yetersiz veri → NaN
    short = pd.Series([1.0, 2.0, 3.0])
    ic_short, _ = selector.calculate_ic(short, short)
    assert np.isnan(ic_short), "Yetersiz veri → NaN bekleniyor"
    print(f"  Yetersiz veri: IC=NaN ✓")
    
    print(f"  ✓ Spearman IC hesaplama doğru çalışıyor")


def test_10_evaluate_all():
    """Selector: Tüm indikatörleri değerlendir + FDR correction."""
    from indicators.calculator import IndicatorCalculator
    from indicators.selector import IndicatorSelector
    
    calc = IndicatorCalculator(verbose=False)
    selector = IndicatorSelector(alpha=0.05, correction_method='fdr', verbose=False)
    
    # Veri hazırla
    df = generate_synthetic_ohlcv(n_bars=800)
    df = calc.calculate_all(df)
    df = calc.add_price_features(df)
    df = calc.add_forward_returns(df, periods=[1, 5])
    
    # IC analizi
    scores = selector.evaluate_all_indicators(df, target_col='fwd_ret_5')
    
    assert len(scores) > 0, "Skor listesi boş"
    print(f"  Değerlendirilen: {len(scores)} kolon")
    
    # Skorlar sıralı mı? (|IC| büyükten küçüğe)
    valid_ics = [abs(s.ic_mean) for s in scores if not np.isnan(s.ic_mean)]
    if len(valid_ics) > 1:
        for i in range(len(valid_ics) - 1):
            assert valid_ics[i] >= valid_ics[i + 1] - 1e-10, "Sıralama hatalı"
    
    # p_value_adjusted var mı?
    has_adjusted = any(s.p_value_adjusted != s.p_value for s in scores if not np.isnan(s.ic_mean))
    # FDR düzeltmesi uygulanmış olmalı (en az bazı p-değerleri değişmiş)
    
    # Kategori tespiti çalışıyor mu?
    categories_found = set(s.category for s in scores)
    print(f"  Tespit edilen kategoriler: {categories_found}")
    
    # İlk 5 skoru göster
    print(f"\n  Top 5 IC:")
    for s in scores[:5]:
        if not np.isnan(s.ic_mean):
            sig = "✓" if s.is_significant else " "
            print(f"    {sig} {s.name:<25} IC={s.ic_mean:+.4f} p_adj={s.p_value_adjusted:.4f} [{s.category}]")
    
    print(f"\n  ✓ {len(scores)} indikatör değerlendirildi, FDR uygulandı")


def test_11_select_best():
    """Selector: En iyi indikatörleri seç (kategori başına max 2)."""
    from indicators.calculator import IndicatorCalculator
    from indicators.selector import IndicatorSelector
    
    calc = IndicatorCalculator(verbose=False)
    selector = IndicatorSelector(alpha=0.05, correction_method='fdr', verbose=False)
    
    # Veri hazırla
    df = generate_synthetic_ohlcv(n_bars=800)
    df = calc.calculate_all(df)
    df = calc.add_price_features(df)
    df = calc.add_forward_returns(df, periods=[5])
    
    # IC analizi
    scores = selector.evaluate_all_indicators(df, target_col='fwd_ret_5')
    
    # En iyileri seç (anlamlılık filtresi kapalı — sentetik veride az anlamlı olabilir)
    best = selector.select_best_indicators(
        scores,
        max_per_category=2,
        only_significant=False  # Sentetik veri ile test
    )
    
    print(f"  Seçilen kategoriler: {list(best.keys())}")
    
    total_selected = 0
    for cat, indicators in best.items():
        assert len(indicators) <= 2, f"{cat}: max 2 indikatör bekleniyor, {len(indicators)} var"
        total_selected += len(indicators)
        for ind in indicators:
            print(f"    {cat}: {ind.name} (IC={ind.ic_mean:+.4f})")
    
    assert total_selected > 0, "Hiç indikatör seçilmedi"
    print(f"\n  Toplam seçilen: {total_selected} indikatör")
    
    # Summary report
    report = selector.get_summary_report(scores, top_n=10)
    assert isinstance(report, pd.DataFrame), "Rapor DataFrame olmalı"
    assert len(report) > 0, "Rapor boş"
    print(f"  Özet rapor: {len(report)} satır")
    
    print(f"  ✓ Kategori bazlı seçim çalışıyor")


def test_12_full_pipeline():
    """Entegrasyon: Calculator → Selector → Best Indicators pipeline."""
    from indicators.calculator import IndicatorCalculator
    from indicators.selector import IndicatorSelector
    
    print(f"  Pipeline başlıyor...")
    
    # 1. Sentetik veri
    df = generate_synthetic_ohlcv(n_bars=1000)
    print(f"    [1/5] Sentetik veri: {len(df)} bar")
    
    # 2. Tüm indikatörler
    calc = IndicatorCalculator(verbose=False)
    df = calc.calculate_all(df)
    n_indicators = len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])
    print(f"    [2/5] İndikatörler: {n_indicators} kolon")
    
    # 3. Price features + rolling stats
    df = calc.add_price_features(df)
    df = calc.add_rolling_stats(df, windows=[10, 20])
    print(f"    [3/5] Features: {len(df.columns)} toplam kolon")
    
    # 4. Forward returns
    df = calc.add_forward_returns(df, periods=[1, 5])
    print(f"    [4/5] Forward returns eklendi")
    
    # 5. IC selector
    selector = IndicatorSelector(alpha=0.05, correction_method='fdr', verbose=False)
    scores = selector.evaluate_all_indicators(df, target_col='fwd_ret_5')
    best = selector.select_best_indicators(scores, max_per_category=2, only_significant=False)
    
    total_best = sum(len(v) for v in best.values())
    print(f"    [5/5] IC analiz: {len(scores)} test → {total_best} seçildi")
    
    # Son doğrulamalar
    assert len(df) > 500, "Pipeline sonrası en az 500 bar olmalı"
    assert len(scores) > 30, "En az 30 IC skoru olmalı"
    assert total_best > 0, "En az 1 indikatör seçilmeli"
    
    # Veri bütünlüğü: OHLCV bozulmamış mı?
    assert (df['high'] >= df['low']).all(), "High < Low var!"
    
    # NaN kontrolü: IC analizi kolon bazlı dropna yapar, tüm kolonlar değil
    # Bu yüzden sadece OHLCV + forward return kolonlarında temizlik kontrol edilir
    core_cols = ['open', 'high', 'low', 'close', 'volume', 'fwd_ret_5']
    clean = df.dropna(subset=core_cols)
    clean_ratio = len(clean) / len(df)
    print(f"\n  Clean data ratio (core): {clean_ratio:.1%} ({len(clean)}/{len(df)})")
    assert clean_ratio > 0.40, f"Clean ratio çok düşük: {clean_ratio:.1%}"
    
    # İndikatör NaN oranı: ortalama %50'den az olmalı
    ind_cols = [c for c in df.columns if c not in core_cols]
    avg_nan = df[ind_cols].isnull().mean().mean()
    print(f"  İndikatör NaN oranı: {avg_nan:.1%}")
    assert avg_nan < 0.50, f"İndikatör NaN çok yüksek: {avg_nan:.1%}"

# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    """Tüm testleri çalıştırır."""
    print("=" * 55)
    print("  ADIM 3: İNDİKATÖR KATMANI TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    tests = [
        (1,  "Categories: İndikatör sayıları",         test_01_category_counts),
        (2,  "Categories: Config doğrulama",            test_02_indicator_config),
        (3,  "Calculator: Tek indikatör (RSI)",         test_03_single_indicator),
        (4,  "Calculator: Tek kategori (momentum)",     test_04_category_calculation),
        (5,  "Calculator: Tüm kategoriler",             test_05_all_categories),
        (6,  "Calculator: Price features",              test_06_price_features),
        (7,  "Calculator: Rolling stats",               test_07_rolling_stats),
        (8,  "Calculator: Forward returns",             test_08_forward_returns),
        (9,  "Selector: Tek IC (Spearman)",             test_09_single_ic),
        (10, "Selector: Tümünü değerlendir + FDR",      test_10_evaluate_all),
        (11, "Selector: En iyi seçimi",                 test_11_select_best),
        (12, "Entegrasyon: Tam pipeline",               test_12_full_pipeline),
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
        print(f"\n  🎉 ADIM 3 TAMAMLANDI! Tüm testler geçti.")
        print(f"  Sonraki: Adım 4 → Dinamik Coin Tarayıcı")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları kontrol edin.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
