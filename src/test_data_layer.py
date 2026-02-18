# =============================================================================
# VERİ KATMANI TEST DOSYASI
# =============================================================================
# Çalıştırma: cd src && python test_data_layer.py
#
# Bu test Bitget Public API'yi kullanır (API key gerektirmez).
# Tüm testler geçerse Adım 2 tamamdır.
#
# Test Listesi:
# 1. Fetcher: Market bilgisi (BTC contract size, precision)
# 2. Fetcher: Ticker (anlık fiyat)
# 3. Fetcher: Tek OHLCV çekme (200 bar)
# 4. Fetcher: Pagination (500+ bar)
# 5. Fetcher: Çoklu TF (3 farklı timeframe)
# 6. Fetcher: USDT Futures listesi
# 7. Fetcher: Veri doğrulama
# 8. Preprocessor: Full pipeline (returns, winsorize, features)
# 9. Preprocessor: Forward returns (IC target)
# 10. Preprocessor: Kalite raporu
# 11. Entegrasyon: Fetcher → Preprocessor → IC-ready veri
# =============================================================================

import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime

# Path ayarları (src/ altından çalıştırılacak)
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / 'data'))

# Loglama
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_test(test_num: int, test_name: str, test_func):
    """Test wrapper: Hata yakalar, süre ölçer, sonuç raporlar."""
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num}: {test_name}")
    print(f"{'─' * 55}")
    
    start = time.time()
    try:
        result = test_func()
        elapsed = time.time() - start
        print(f"\n  ✅ BAŞARILI ({elapsed:.1f}s)")
        return True, result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ BAŞARISIZ ({elapsed:.1f}s)")
        print(f"     Hata: {e}")
        traceback.print_exc()
        return False, None


def main():
    print("=" * 60)
    print("  🧪 ADIM 2: VERİ KATMANI TESTİ")
    print(f"  ⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Import'lar
    from fetcher import BitgetFetcher
    from preprocessor import DataPreprocessor
    
    fetcher = BitgetFetcher()
    pp = DataPreprocessor()
    
    results = {}    # test_num → (passed, result)
    
    # =========================================================================
    # TEST 1: Market Bilgisi
    # =========================================================================
    def test_market_info():
        info = fetcher.get_market_info("BTC/USDT:USDT")
        
        assert info['type'] == 'swap', f"Tip swap olmalı, {info['type']} geldi"
        assert info['max_leverage'] >= 50, f"Max leverage >= 50 olmalı, {info['max_leverage']} geldi"
        assert info['precision']['price'] > 0, "Price precision > 0 olmalı"
        assert info['limits']['min_cost'] > 0, "Min cost > 0 olmalı"
        
        print(f"   Kontrat büyüklüğü: {info['contract_size']}")
        print(f"   Max kaldıraç: {info['max_leverage']}x")
        print(f"   Min order cost: ${info['limits']['min_cost']}")
        print(f"   Price precision: {info['precision']['price']}")
        print(f"   Amount precision: {info['precision']['amount']}")
        
        return info
    
    results[1] = run_test(1, "Market Bilgisi (BTC)", test_market_info)
    
    # =========================================================================
    # TEST 2: Ticker (Anlık Fiyat)
    # =========================================================================
    def test_ticker():
        ticker = fetcher.get_ticker("BTC/USDT:USDT")
        
        assert ticker['last'] > 0, "Fiyat > 0 olmalı"
        assert ticker['volume_24h'] > 0, "24h hacim > 0 olmalı"
        assert -100 < ticker['change_24h'] < 100, "24h değişim mantıklı aralıkta olmalı"
        
        print(f"   BTC: ${ticker['last']:,.2f}")
        print(f"   24h değişim: {ticker['change_24h']:+.2f}%")
        print(f"   24h hacim: ${ticker['volume_24h']:,.0f}")
        print(f"   Spread: ${ticker['spread']:.2f}")
        
        return ticker
    
    results[2] = run_test(2, "Ticker (Anlık Fiyat)", test_ticker)
    
    # =========================================================================
    # TEST 3: Tek OHLCV Çekme
    # =========================================================================
    def test_single_ohlcv():
        df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=100)
        
        assert len(df) > 50, f"En az 50 bar olmalı, {len(df)} geldi"
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume'], \
            f"Kolonlar OHLCV olmalı, {list(df.columns)} geldi"
        assert df.index.tz is not None, "Index timezone-aware olmalı"
        assert df['close'].dtype == 'float64', "Close float64 olmalı"
        assert (df['high'] >= df['low']).all(), "High >= Low olmalı"
        
        print(f"   {len(df)} bar çekildi")
        print(f"   İlk: {df.index[0]}")
        print(f"   Son: {df.index[-1]}")
        print(f"   Son close: ${df['close'].iloc[-1]:,.2f}")
        
        return df
    
    results[3] = run_test(3, "Tek OHLCV (1h, 100 bar)", test_single_ohlcv)
    
    # =========================================================================
    # TEST 4: Pagination (500+ bar)
    # =========================================================================
    def test_pagination():
        df = fetcher.fetch_max_ohlcv(
            "BTC/USDT:USDT", "1h", 
            max_bars=500, 
            progress=True
        )
        
        assert len(df) >= 400, f"En az 400 bar olmalı, {len(df)} geldi"
        assert df.index.is_monotonic_increasing, "Index kronolojik olmalı"
        assert not df.index.duplicated().any(), "Duplicate index olmamalı"
        
        days = (df.index[-1] - df.index[0]).days
        print(f"   {len(df)} bar | {days} gün")
        print(f"   Tarih aralığı: {df.index[0].strftime('%m-%d')} → {df.index[-1].strftime('%m-%d')}")
        
        return df
    
    results[4] = run_test(4, "Pagination (1h, 500 bar)", test_pagination)
    
    # =========================================================================
    # TEST 5: Çoklu Timeframe
    # =========================================================================
    def test_multi_tf():
        data = fetcher.fetch_all_timeframes(
            "ETH/USDT:USDT",
            timeframes=["15m", "1h", "4h"],
            max_bars_override=100
        )
        
        assert len(data) >= 2, f"En az 2 TF olmalı, {len(data)} geldi"
        
        for tf, df in data.items():
            assert len(df) >= 50, f"{tf}: En az 50 bar olmalı"
            print(f"   {tf}: {len(df)} bar | ${df['close'].iloc[-1]:,.2f}")
        
        return data
    
    results[5] = run_test(5, "Çoklu TF (ETH, 3 TF)", test_multi_tf)
    
    # =========================================================================
    # TEST 6: USDT Futures Listesi
    # =========================================================================
    def test_futures_list():
        pairs = fetcher.get_all_usdt_futures()
        
        assert len(pairs) > 100, f"En az 100 çift olmalı, {len(pairs)} geldi"
        assert "BTC/USDT:USDT" in pairs, "BTC/USDT:USDT listede olmalı"
        assert "ETH/USDT:USDT" in pairs, "ETH/USDT:USDT listede olmalı"
        
        print(f"   Toplam: {len(pairs)} USDT-M Futures çifti")
        print(f"   İlk 8: {pairs[:8]}")
        
        return pairs
    
    results[6] = run_test(6, "USDT Futures Listesi", test_futures_list)
    
    # =========================================================================
    # TEST 7: Veri Doğrulama
    # =========================================================================
    def test_validation():
        # Test 4'ten gelen veriyi kullan
        if results[4][0] and results[4][1] is not None:
            df = results[4][1]
        else:
            df = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=100)
        
        validation = fetcher.validate_data(df)
        
        assert validation['total_rows'] > 0, "Satır sayısı > 0 olmalı"
        assert not validation['has_missing'], "Missing value olmamalı"
        
        print(f"   Satır: {validation['total_rows']}")
        print(f"   Geçerli: {'✅' if validation['is_valid'] else '❌'}")
        print(f"   Missing: {'var ⚠️' if validation['has_missing'] else 'yok ✅'}")
        print(f"   OHLC hatalı: {validation.get('ohlc_invalid', 0)}")
        print(f"   Gap: {validation.get('gaps', 0)}")
        
        return validation
    
    results[7] = run_test(7, "Veri Doğrulama", test_validation)
    
    # =========================================================================
    # TEST 8: Preprocessor Full Pipeline
    # =========================================================================
    def test_pipeline():
        # Gerçek OHLCV verisi (Test 4'ten)
        if results[4][0] and results[4][1] is not None:
            df_raw = results[4][1]
        else:
            df_raw = fetcher.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=200)
        
        df_clean = pp.full_pipeline(
            df_raw,
            forward_periods=[1, 5, 10],
            rolling_windows=[10, 20],
            drop_na=True
        )
        
        # Temel kontroller
        assert len(df_clean) > 0, "Pipeline sonucu boş olmamalı"
        assert 'log_return' in df_clean.columns, "log_return kolonu olmalı"
        assert 'volatility' in df_clean.columns, "volatility kolonu olmalı"
        assert 'fwd_ret_5' in df_clean.columns, "fwd_ret_5 kolonu olmalı"
        assert 'body_pct' in df_clean.columns, "body_pct kolonu olmalı"
        assert 'volume_ratio' in df_clean.columns, "volume_ratio kolonu olmalı"
        assert df_clean.isnull().sum().sum() == 0, "NaN kalmamış olmalı"
        
        print(f"   Girdi: {len(df_raw)} satır, {len(df_raw.columns)} kolon")
        print(f"   Çıktı: {len(df_clean)} satır, {len(df_clean.columns)} kolon")
        print(f"   Yeni kolonlar: +{len(df_clean.columns) - len(df_raw.columns)}")
        print(f"   NaN: {df_clean.isnull().sum().sum()} (0 olmalı)")
        
        # Kolon grupları
        fwd_cols = [c for c in df_clean.columns if 'fwd_' in c]
        roll_cols = [c for c in df_clean.columns if 'roll' in c]
        print(f"   Forward kolonlar: {fwd_cols}")
        print(f"   Rolling kolonlar: {len(roll_cols)} adet")
        
        return df_clean
    
    results[8] = run_test(8, "Preprocessor Pipeline", test_pipeline)
    
    # =========================================================================
    # TEST 9: Forward Return Doğrulama
    # =========================================================================
    def test_forward_returns():
        if results[8][0] and results[8][1] is not None:
            df = results[8][1]
        else:
            import numpy as np
            return None
        
        # fwd_ret_5 = 5 bar sonraki log return
        # Manuel hesapla ve karşılaştır
        import numpy as np
        
        # fwd_ret_1 kontrol (shift(-1) ile)
        manual_fwd_1 = np.log(df['close'].shift(-1) / df['close'])
        
        # NaN olmayan yerlerde karşılaştır
        valid = df['fwd_ret_1'].notna() & manual_fwd_1.notna()
        if valid.sum() > 0:
            diff = (df.loc[valid, 'fwd_ret_1'] - manual_fwd_1[valid]).abs().max()
            assert diff < 1e-10, f"Forward return hesabı hatalı! Max fark: {diff}"
            print(f"   fwd_ret_1 doğrulaması: ✅ (max fark: {diff:.2e})")
        
        # fwd_dir kontrol (binary)
        dir_check = (df['fwd_dir_5'] == (df['fwd_ret_5'] > 0).astype(int))
        valid_dir = df['fwd_ret_5'].notna()
        accuracy = dir_check[valid_dir].mean() * 100
        assert accuracy == 100, f"fwd_dir hesabı hatalı! Doğruluk: {accuracy}%"
        print(f"   fwd_dir_5 doğrulaması: ✅ (%{accuracy:.0f} tutarlı)")
        
        # Return dağılım istatistikleri
        ret = df['log_return']
        print(f"   Return mean: {ret.mean():.6f}")
        print(f"   Return std: {ret.std():.6f}")
        print(f"   Return skew: {ret.skew():.4f}")
        print(f"   Return kurt: {ret.kurtosis():.4f}")
        
        return True
    
    results[9] = run_test(9, "Forward Return Doğrulama", test_forward_returns)
    
    # =========================================================================
    # TEST 10: Kalite Raporu
    # =========================================================================
    def test_quality_report():
        if results[8][0] and results[8][1] is not None:
            df = results[8][1]
        else:
            return None
        
        report = pp.quality_report(df)
        
        assert report['rows'] > 0, "Satır sayısı > 0"
        assert report['missing_total'] == 0, "Missing 0 olmalı (pipeline sonrası)"
        
        print(f"   Satır: {report['rows']}")
        print(f"   Kolon: {report['columns']}")
        print(f"   Missing: {report['missing_total']}")
        
        if 'return_stats' in report:
            rs = report['return_stats']
            print(f"   Return μ: {rs['mean']:.6f}")
            print(f"   Return σ: {rs['std']:.6f}")
            print(f"   Skewness: {rs['skew']:.4f}")
            print(f"   Kurtosis: {rs['kurt']:.4f}")
        
        if 'volatility_stats' in report:
            vs = report['volatility_stats']
            print(f"   Volatilite (şu an): {vs['current']:.6f}")
            print(f"   Volatilite (ort.): {vs['mean']:.6f}")
        
        return report
    
    results[10] = run_test(10, "Kalite Raporu", test_quality_report)
    
    # =========================================================================
    # TEST 11: Entegrasyon (Fetcher → Preprocessor → IC-Ready)
    # =========================================================================
    def test_integration():
        """Tam entegrasyon testi: Fetcher + Preprocessor birlikte."""
        
        # 1. Farklı bir coin dene (sadece BTC değil)
        symbol = "SOL/USDT:USDT"
        print(f"   Sembol: {symbol}")
        
        # 2. Veri çek
        df_raw = fetcher.fetch_max_ohlcv(symbol, "1h", max_bars=300, progress=False)
        print(f"   Ham veri: {len(df_raw)} bar")
        
        # 3. Pipeline uygula
        df_clean = pp.full_pipeline(df_raw, forward_periods=[1, 5])
        print(f"   İşlenmiş: {len(df_clean)} bar, {len(df_clean.columns)} kolon")
        
        # 4. IC analizi için hazır mı?
        required_cols = ['log_return', 'volatility', 'fwd_ret_5', 
                        'body_pct', 'volume_ratio', 'hl_position']
        for col in required_cols:
            assert col in df_clean.columns, f"'{col}' kolonu eksik!"
        
        assert df_clean.isnull().sum().sum() == 0, "NaN var!"
        assert len(df_clean) >= 200, f"Yeterli veri yok ({len(df_clean)} < 200)"
        
        print(f"   Gerekli kolonlar: ✅ ({len(required_cols)}/{len(required_cols)})")
        print(f"   NaN: 0 ✅")
        print(f"   IC analizi için HAZIR ✅")
        
        return df_clean
    
    results[11] = run_test(11, "Entegrasyon (SOL → IC-Ready)", test_integration)
    
    # =========================================================================
    # SONUÇ TABLOSU
    # =========================================================================
    print("\n" + "=" * 60)
    print("  📊 TEST SONUÇLARI")
    print("=" * 60)
    
    test_names = {
        1: "Market Bilgisi",
        2: "Ticker",
        3: "Tek OHLCV",
        4: "Pagination",
        5: "Çoklu TF",
        6: "Futures Listesi",
        7: "Veri Doğrulama",
        8: "Pipeline",
        9: "Forward Return",
        10: "Kalite Raporu",
        11: "Entegrasyon",
    }
    
    passed = 0
    failed = 0
    
    for num, name in test_names.items():
        if num in results:
            status = "✅" if results[num][0] else "❌"
            if results[num][0]:
                passed += 1
            else:
                failed += 1
        else:
            status = "⏭️"
        print(f"  {status} Test {num:>2}: {name}")
    
    print(f"\n  {'=' * 40}")
    print(f"  Toplam: {passed + failed} | ✅ {passed} | ❌ {failed}")
    
    if failed == 0:
        print(f"\n  🎉 ADIM 2 TAMAMLANDI! Veri katmanı hazır.")
        print(f"  → Sonraki: Adım 3 (İndikatör hesaplama)")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları düzelt.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
