# =============================================================================
# ADIM 4: DİNAMİK COİN TARAYICI TESTLERİ
# =============================================================================
# Çalıştırma: cd src && python test_scanner.py
#
# İNTERNET GEREKTİRİR — Bitget API'den gerçek ticker verisi çeker.
# Tüm testler geçerse Adım 4 tamamdır.
#
# Test Listesi:
# 1.  Blacklist: Stablecoin ve leveraged token eleme
# 2.  Ticker: Batch ticker çekme (tek API çağrısı)
# 3.  Scan Results: Metrik hesaplama (spread, volatilite)
# 4.  Filters: Hacim ve spread filtreleri
# 5.  Scoring: Percentile rank composite skor
# 6.  Full Scan: Tam pipeline (scan → filter → score → top N)
# 7.  Cache: 5dk TTL cache mekanizması
# 8.  Helpers: get_symbols(), get_coins(), get_report()
# =============================================================================

import sys
import time
import logging
import traceback
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Path ayarı (src/ altından çalışır)
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# Loglama — test çıktısı temiz olsun
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
    """
    Tek bir testi çalıştırır, süre ölçer, hata yakalar.
    
    Returns:
    -------
    bool
        Test başarılı mı?
    """
    print(f"\n{'─' * 55}")
    print(f"  TEST {test_num}: {test_name}")
    print(f"{'─' * 55}")
    
    start = time.time()
    try:
        test_func()
        elapsed = time.time() - start
        print(f"\n  ✅ BAŞARILI ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ BAŞARISIZ ({elapsed:.1f}s)")
        print(f"     Hata: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# TEST 1: BLACKLIST FİLTRESİ
# =============================================================================

def test_01_blacklist():
    """Stablecoin ve leveraged token'lar eleniyor mu?"""
    from scanner.coin_scanner import CoinScanner
    
    scanner = CoinScanner(verbose=False)
    
    # Test verileri: elenmesi gereken + kalması gereken semboller
    test_symbols = [
        'BTC/USDT:USDT',       # ✓ Kalmalı
        'ETH/USDT:USDT',       # ✓ Kalmalı
        'SOL/USDT:USDT',       # ✓ Kalmalı
        'USDC/USDT:USDT',      # ✗ Stablecoin → elen
        'DAI/USDT:USDT',       # ✗ Stablecoin → elen
        'BTTC/USDT:USDT',      # ✗ Blacklist → elen
    ]
    
    filtered = scanner._apply_blacklist(test_symbols)
    
    # Kalması gerekenler
    assert 'BTC/USDT:USDT' in filtered, "BTC elenmemeli"
    assert 'ETH/USDT:USDT' in filtered, "ETH elenmemeli"
    assert 'SOL/USDT:USDT' in filtered, "SOL elenmemeli"
    
    # Elenmesi gerekenler
    assert 'USDC/USDT:USDT' not in filtered, "USDC elenmeli (stablecoin)"
    assert 'DAI/USDT:USDT' not in filtered, "DAI elenmeli (stablecoin)"
    assert 'BTTC/USDT:USDT' not in filtered, "BTTC elenmeli (blacklist)"
    
    print(f"  Gelen: {len(test_symbols)} → Kalan: {len(filtered)}")
    print(f"  Elenen: {len(test_symbols) - len(filtered)} (beklenen: 3)")
    print(f"  ✓ Blacklist filtresi doğru çalışıyor")


# =============================================================================
# TEST 2: BATCH TİCKER ÇEKİM
# =============================================================================

def test_02_batch_ticker():
    """Bitget'ten batch ticker çekiliyor mu? (API gerekli)"""
    from scanner.coin_scanner import CoinScanner
    
    scanner = CoinScanner(verbose=False)
    
    # Bilinen sembollerin ticker'ını çek
    test_symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    tickers = scanner._fetch_all_tickers(test_symbols)
    
    # En az 2 ticker dönmeli (bazı coinler geçici olarak unavailable olabilir)
    assert len(tickers) >= 2, f"En az 2 ticker bekleniyor, {len(tickers)} geldi"
    
    # BTC ticker'ı kontrol et
    assert 'BTC/USDT:USDT' in tickers, "BTC ticker bulunamadı"
    
    btc = tickers['BTC/USDT:USDT']
    assert btc.get('last', 0) > 1000, f"BTC fiyatı mantıksız: {btc.get('last')}"
    assert btc.get('quoteVolume', 0) > 0, "BTC hacim > 0 olmalı"
    
    print(f"  {len(tickers)} ticker çekildi (tek API çağrısı)")
    print(f"  BTC: ${btc['last']:,.2f} | Vol: ${btc['quoteVolume']:,.0f}")
    print(f"  ✓ Batch ticker çekme çalışıyor")


# =============================================================================
# TEST 3: SCAN RESULT OLUŞTURMA (METRİK HESAPLAMA)
# =============================================================================

def test_03_build_results():
    """Ticker verisinden metrikler doğru hesaplanıyor mu?"""
    from scanner.coin_scanner import CoinScanner
    
    scanner = CoinScanner(verbose=False)
    
    # Sahte ticker verisi — kontrollü test
    fake_tickers = {
        'BTC/USDT:USDT': {
            'last': 100000.0,
            'bid': 99990.0,
            'ask': 100010.0,
            'high': 102000.0,
            'low': 98000.0,
            'quoteVolume': 5_000_000_000,    # $5B
            'percentage': 2.5,
        },
        'LOWVOL/USDT:USDT': {
            'last': 0.0,                      # Geçersiz fiyat → atlanmalı
            'bid': 0.0,
            'ask': 0.0,
            'high': 0.0,
            'low': 0.0,
            'quoteVolume': 0,
            'percentage': 0,
        }
    }
    
    results = scanner._build_scan_results(fake_tickers)
    
    # Geçersiz coin (fiyat=0) atlanmış mı?
    assert len(results) == 1, f"1 sonuç bekleniyor, {len(results)} geldi"
    
    btc = results[0]
    
    # Fiyat doğru mu?
    assert btc.price == 100000.0, f"Fiyat hatalı: {btc.price}"
    assert btc.coin == 'BTC', f"Coin adı hatalı: {btc.coin}"
    
    # Spread hesabı: (100010 - 99990) / 100000 × 100 = 0.02%
    expected_spread = (100010 - 99990) / 100000 * 100
    assert abs(btc.spread_pct - expected_spread) < 0.001, \
        f"Spread hatalı: {btc.spread_pct} != {expected_spread}"
    
    # Volatilite: (102000 - 98000) / 100000 × 100 = 4%
    expected_vol = (102000 - 98000) / 100000 * 100
    assert abs(btc.volatility - expected_vol) < 0.01, \
        f"Volatilite hatalı: {btc.volatility} != {expected_vol}"
    
    print(f"  Spread: {btc.spread_pct:.4f}% (beklenen: {expected_spread:.4f}%)")
    print(f"  Volatilite: {btc.volatility:.2f}% (beklenen: {expected_vol:.2f}%)")
    print(f"  Geçersiz coin (fiyat=0): atlandı ✓")
    print(f"  ✓ Metrik hesaplama doğru")


# =============================================================================
# TEST 4: HARD FİLTRELER
# =============================================================================

def test_04_filters():
    """Hacim ve spread filtreleri doğru çalışıyor mu?"""
    from scanner.coin_scanner import CoinScanner, CoinScanResult
    
    scanner = CoinScanner(verbose=False)
    
    # Test coin'leri: bazıları geçmeli, bazıları elenmeli
    test_results = [
        CoinScanResult(                        # ✓ Geçmeli (yüksek hacim, düşük spread)
            symbol='GOOD/USDT:USDT', coin='GOOD',
            price=100.0, volume_24h=50_000_000,
            change_24h=1.0, spread_pct=0.01, volatility=3.0
        ),
        CoinScanResult(                        # ✗ Düşük hacim
            symbol='LOWVOL/USDT:USDT', coin='LOWVOL',
            price=1.0, volume_24h=100_000,     # $100K < $5M
            change_24h=0.5, spread_pct=0.05, volatility=2.0
        ),
        CoinScanResult(                        # ✗ Yüksek spread
            symbol='WIDESPRD/USDT:USDT', coin='WIDESPRD',
            price=0.5, volume_24h=10_000_000,
            change_24h=-2.0, spread_pct=0.50,  # 0.50% > 0.10%
            volatility=5.0
        ),
        CoinScanResult(                        # ✗ Çok düşük fiyat
            symbol='DUST/USDT:USDT', coin='DUST',
            price=0.00001, volume_24h=10_000_000,  # Fiyat < $0.0001
            change_24h=0.1, spread_pct=0.02, volatility=1.0
        ),
    ]
    
    scanner._apply_filters(test_results)
    
    passed = [r for r in test_results if r.passed_filters]
    failed = [r for r in test_results if not r.passed_filters]
    
    # Sadece GOOD geçmeli
    assert len(passed) == 1, f"1 coin geçmeli, {len(passed)} geçti"
    assert passed[0].coin == 'GOOD', f"GOOD geçmeli, {passed[0].coin} geçti"
    
    # Elenme nedenleri kontrolü
    for r in failed:
        assert r.filter_reason != "", f"{r.coin} elendi ama neden yazılmamış"
        print(f"  ✗ {r.coin:<10} → {r.filter_reason}")
    
    print(f"  ✓ {test_results[0].coin:<10} → GEÇTİ")
    print(f"\n  Geçen: {len(passed)} | Elenen: {len(failed)}")
    print(f"  ✓ Filtreler doğru çalışıyor")


# =============================================================================
# TEST 5: PERCENTİLE RANK COMPOSİTE SKORLAMA
# =============================================================================

def test_05_scoring():
    """Percentile rank ve composite skor doğru hesaplanıyor mu?"""
    from scanner.coin_scanner import CoinScanner, CoinScanResult
    
    scanner = CoinScanner(verbose=False)
    
    # 4 coin: farklı profiller
    test_results = [
        CoinScanResult(                        # En iyi: yüksek hacim, düşük spread, orta vol
            symbol='BEST/USDT:USDT', coin='BEST',
            price=100.0, volume_24h=10_000_000_000,  # $10B
            change_24h=2.0, spread_pct=0.001,        # Çok düşük spread
            volatility=5.0                            # Orta volatilite
        ),
        CoinScanResult(                        # Orta: orta her şey
            symbol='MID/USDT:USDT', coin='MID',
            price=50.0, volume_24h=500_000_000,      # $500M
            change_24h=1.0, spread_pct=0.02,
            volatility=3.0
        ),
        CoinScanResult(                        # Kötü: düşük hacim, yüksek spread
            symbol='LOW/USDT:USDT', coin='LOW',
            price=1.0, volume_24h=10_000_000,        # $10M
            change_24h=-1.0, spread_pct=0.08,
            volatility=1.0
        ),
        CoinScanResult(                        # Volatilite şampiyonu
            symbol='VOLAT/USDT:USDT', coin='VOLAT',
            price=10.0, volume_24h=100_000_000,      # $100M
            change_24h=5.0, spread_pct=0.03,
            volatility=15.0                           # Çok yüksek volatilite
        ),
    ]
    
    scored = scanner._calculate_scores(test_results)
    
    # Her coin'in skoru 0-100 arası mı?
    for r in scored:
        assert 0 <= r.composite_score <= 100, \
            f"{r.coin} skor aralık dışı: {r.composite_score}"
        print(f"  {r.coin:<8} → Skor: {r.composite_score:>5.1f}")
    
    # BEST en yüksek skora sahip olmalı (hacim + likidite avantajı)
    scores = {r.coin: r.composite_score for r in scored}
    assert scores['BEST'] > scores['LOW'], \
        f"BEST ({scores['BEST']}) > LOW ({scores['LOW']}) olmalı"
    
    # Percentile rank testi (birim test)
    arr = np.array([10, 20, 30, 40, 50])
    ranks = CoinScanner._percentile_rank(arr)
    assert ranks[0] == 0.0, f"Min rank 0 olmalı, {ranks[0]} geldi"
    assert ranks[-1] == 100.0, f"Max rank 100 olmalı, {ranks[-1]} geldi"
    assert ranks[2] == 50.0, f"Median rank 50 olmalı, {ranks[2]} geldi"
    
    print(f"\n  Percentile rank: [0, 25, 50, 75, 100] ✓")
    print(f"  BEST > LOW sıralama: ✓")
    print(f"  ✓ Composite skorlama doğru")


# =============================================================================
# TEST 6: TAM TARAMA PİPELİNE (API GEREKLİ)
# =============================================================================

def test_06_full_scan():
    """Tam scan pipeline çalışıyor mu? (Bitget API gerekli)"""
    from scanner.coin_scanner import CoinScanner
    
    scanner = CoinScanner(verbose=False)
    
    # Top 10 coin tara
    top_coins = scanner.scan(top_n=10, force_refresh=True)
    
    assert len(top_coins) > 0, "En az 1 coin dönmeli"
    assert len(top_coins) <= 10, f"Max 10 coin bekleniyor, {len(top_coins)} döndü"
    
    # BTC genellikle top 10'da olmalı (en yüksek hacim)
    coins = [c.coin for c in top_coins]
    assert 'BTC' in coins, f"BTC top 10'da olmalı! Gelen: {coins}"
    
    # İlk coin'in skoru en yüksek olmalı (sıralama kontrolü)
    scores = [c.composite_score for c in top_coins]
    assert scores == sorted(scores, reverse=True), "Skor azalan sırada olmalı"
    
    # Her coin'in temel metrikleri geçerli mi?
    for c in top_coins:
        assert c.price > 0, f"{c.coin} fiyat <= 0"
        assert c.volume_24h > 0, f"{c.coin} hacim <= 0"
        assert c.spread_pct >= 0, f"{c.coin} spread < 0"
        assert c.passed_filters, f"{c.coin} filtreden geçmemiş ama listede!"
    
    print(f"  Top {len(top_coins)} coin tarandı")
    for i, c in enumerate(top_coins[:5], 1):
        print(f"   {i}. {c.coin:<8} ${c.price:>10,.2f} | "
              f"Vol: ${c.volume_24h/1e6:>6,.0f}M | "
              f"Skor: {c.composite_score:>5.1f}")
    if len(top_coins) > 5:
        print(f"   ... ve {len(top_coins)-5} coin daha")
    
    print(f"  ✓ Tam pipeline çalışıyor")


# =============================================================================
# TEST 7: CACHE MEKANİZMASI
# =============================================================================

def test_07_cache():
    """5dk cache mekanizması çalışıyor mu?"""
    from scanner.coin_scanner import CoinScanner
    
    scanner = CoinScanner(verbose=False)
    
    # İlk tarama (API çağrısı yapacak)
    t1_start = time.time()
    result1 = scanner.scan(top_n=10, force_refresh=True)
    t1_elapsed = time.time() - t1_start
    
    # İkinci tarama (cache'den gelmeli)
    t2_start = time.time()
    result2 = scanner.scan(top_n=10)
    t2_elapsed = time.time() - t2_start
    
    # Cache çok daha hızlı olmalı
    assert t2_elapsed < 0.1, f"Cache süresi > 0.1s: {t2_elapsed:.3f}s"
    
    # Aynı sonuçları döndürmeli
    coins1 = [c.coin for c in result1]
    coins2 = [c.coin for c in result2]
    assert coins1 == coins2, f"Cache sonuçları farklı!\n{coins1}\n{coins2}"
    
    # Cache geçerlilik kontrolü
    assert scanner._is_cache_valid(), "Cache geçerli olmalı"
    
    # force_refresh cache'i bypass etmeli
    t3_start = time.time()
    _ = scanner.scan(top_n=10, force_refresh=True)
    t3_elapsed = time.time() - t3_start
    assert t3_elapsed > t2_elapsed, "force_refresh cache'den yavaş olmalı"
    
    print(f"  İlk tarama: {t1_elapsed:.2f}s (API)")
    print(f"  Cache'den:  {t2_elapsed:.4f}s")
    print(f"  Hızlanma:   {t1_elapsed/max(t2_elapsed, 0.001):.0f}x")
    print(f"  force_refresh: {t3_elapsed:.2f}s (API)")
    print(f"  ✓ Cache mekanizması çalışıyor")


# =============================================================================
# TEST 8: YARDIMCI FONKSİYONLAR
# =============================================================================

def test_08_helpers():
    """get_symbols(), get_coins(), get_report() çalışıyor mu?"""
    from scanner.coin_scanner import CoinScanner
    import pandas as pd
    
    scanner = CoinScanner(verbose=False)
    
    # get_symbols() — tam Bitget sembol formatı
    symbols = scanner.get_symbols(top_n=5)
    assert len(symbols) > 0, "Sembol listesi boş"
    assert all(':USDT' in s for s in symbols), "Tüm semboller :USDT içermeli"
    print(f"  get_symbols(5): {symbols}")
    
    # get_coins() — kısa isim
    coins = scanner.get_coins(top_n=5)
    assert len(coins) > 0, "Coin listesi boş"
    assert 'BTC' in coins, "BTC listede olmalı"
    print(f"  get_coins(5):   {coins}")
    
    # get_report() — DataFrame
    report = scanner.get_report(top_n=10)
    assert isinstance(report, pd.DataFrame), "Rapor DataFrame olmalı"
    assert len(report) > 0, "Rapor boş olmamalı"
    
    # Beklenen kolonlar
    expected_cols = ['Coin', 'Symbol', 'Fiyat ($)', '24h Hacim ($)', 'Skor']
    for col in expected_cols:
        assert col in report.columns, f"Raporda '{col}' kolonu eksik"
    
    # Skor sıralaması doğru mu?
    scores = report['Skor'].tolist()
    assert scores == sorted(scores, reverse=True), "Rapor skora göre sıralı olmalı"
    
    print(f"  get_report(10): {len(report)} satır × {len(report.columns)} kolon")
    print(f"  Kolonlar: {list(report.columns)}")
    print(f"  ✓ Yardımcı fonksiyonlar çalışıyor")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================

def main():
    """Tüm testleri sırasıyla çalıştırır."""
    
    print("=" * 55)
    print("  ADIM 4: DİNAMİK COİN TARAYICI TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    tests = [
        (1, "Blacklist: Stablecoin/leveraged eleme",    test_01_blacklist),
        (2, "Ticker: Batch ticker çekme (API)",         test_02_batch_ticker),
        (3, "Metrikler: Spread/volatilite hesaplama",   test_03_build_results),
        (4, "Filtreler: Hacim/spread/fiyat",            test_04_filters),
        (5, "Skorlama: Percentile rank composite",      test_05_scoring),
        (6, "Tam Scan: Pipeline (API)",                 test_06_full_scan),
        (7, "Cache: 5dk TTL mekanizması",               test_07_cache),
        (8, "Helpers: symbols/coins/report",            test_08_helpers),
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
        print(f"  {status} Test {num}: {name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {passed + failed} | Başarılı: {passed} | Başarısız: {failed}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        print(f"\n  🎉 ADIM 4 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Adım 5 → Risk Yönetimi Motoru")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları kontrol edin.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
