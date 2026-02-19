#!/usr/bin/env python3
# =============================================================================
# DURUM KONTROL + PİPELİNE DEBUG SCRİPTİ
# =============================================================================
# Bu script:
# 1. Mevcut fetcher.py/coin_scanner.py/main.py durumunu kontrol eder
# 2. Pipeline'ın hangi aşamada durduğunu tespit eder
# 3. Sorunu düzeltir
#
# Çalıştır: cd hybrid_crypto_bot && python debug_pipeline.py
# =============================================================================

import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime

# Renkli çıktı
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")


def find_src():
    """src/ klasörünü bul ve sys.path'e ekle."""
    cwd = Path.cwd()
    
    # hybrid_crypto_bot/ kökündeyiz → src/ alt klasör
    if (cwd / 'src').exists():
        src = cwd / 'src'
    # src/ içindeyiz
    elif cwd.name == 'src':
        src = cwd
    else:
        print(f"{RED}src/ klasörü bulunamadı! Proje kökünden çalıştır.{RESET}")
        sys.exit(1)
    
    # src'yi path'e ekle
    sys.path.insert(0, str(src))
    return src


# =============================================================================
# BÖLÜM 1: DOSYA DURUM KONTROLÜ
# =============================================================================
def check_file_state(src: Path):
    """Mevcut dosyaların durumunu analiz et."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  📋 BÖLÜM 1: DOSYA DURUM KONTROLÜ")
    print(f"{'='*60}{RESET}")
    
    issues = []
    
    # --- fetcher.py kontrolü ---
    fetcher_path = src / 'data' / 'fetcher.py'
    if fetcher_path.exists():
        content = fetcher_path.read_text(encoding='utf-8')
        
        # Hangi versiyon yüklü?
        has_binance_attr = 'self.binance = ccxt.binance' in content
        has_fetch_tickers_method = 'def fetch_tickers(' in content
        has_v3_marker = 'BINANCE DATA & BITGET EXECUTION' in content or 'v3.0' in content
        has_old_bitget = 'self.exchange = ccxt.bitget' in content
        
        if has_fetch_tickers_method:
            warn("fetcher.py: fetch_tickers() metodu VAR → fetcher_v3 yüklü olabilir")
            info("  Bu metod Binance ticker filtresi bozuk, 0 coin döndürüyor!")
            issues.append('fetcher_v3_loaded')
        elif has_binance_attr and has_old_bitget:
            ok("fetcher.py: Hybrid versiyon (Bitget+Binance) — DOĞRU")
        elif has_old_bitget and not has_binance_attr:
            info("fetcher.py: Sadece Bitget versiyonu")
        
        # get_ticker hangi exchange'i kullanıyor?
        if 'self.exchange.fetch_ticker(' in content and 'def get_ticker' in content:
            ok("get_ticker() → Bitget kullanıyor (DOĞRU)")
        
    else:
        fail("fetcher.py bulunamadı!")
        issues.append('no_fetcher')
    
    # --- coin_scanner.py kontrolü ---
    scanner_path = src / 'scanner' / 'coin_scanner.py'
    if scanner_path.exists():
        content = scanner_path.read_text(encoding='utf-8')
        
        if 'self.fetcher.exchange.fetch_tickers()' in content:
            ok("coin_scanner.py: Bitget ticker kullanıyor (DOĞRU)")
        elif 'self.fetcher.fetch_tickers(' in content:
            warn("coin_scanner.py: fetcher.fetch_tickers() kullanıyor")
            warn("  Bu metod Binance filtresi bozuk → 0 coin döndürür!")
            issues.append('scanner_binance_ticker')
        else:
            warn("coin_scanner.py: Beklenen ticker pattern bulunamadı")
    else:
        fail("coin_scanner.py bulunamadı!")
    
    # --- main.py kontrolü ---
    main_path = src / 'main.py'
    if main_path.exists():
        content = main_path.read_text(encoding='utf-8')
        
        # Ticker çağrıları
        bitget_ticker_count = content.count('self.fetcher.exchange.fetch_ticker(')
        get_ticker_count = content.count('self.fetcher.get_ticker(')
        
        if bitget_ticker_count > 0:
            ok(f"main.py: {bitget_ticker_count}x Bitget ticker çağrısı (DOĞRU)")
        if get_ticker_count > 0:
            info(f"main.py: {get_ticker_count}x get_ticker() çağrısı")
            # Bu da OK eğer get_ticker() Bitget'e gidiyorsa
        
        # quoteVolume vs volume_24h
        if "ticker.get('quoteVolume'" in content:
            ok("main.py: quoteVolume kullanıyor (DOĞRU)")
        elif "ticker.get('volume_24h'" in content:
            warn("main.py: volume_24h kullanıyor (fetcher_v3 formatı)")
            issues.append('main_volume_24h')
    
    # --- Yedek dosyalar var mı? ---
    backup_files = list(src.glob('**/*_YEDEK*'))
    if backup_files:
        info(f"{len(backup_files)} yedek dosya bulundu:")
        for bf in backup_files:
            info(f"  {bf.relative_to(src)}")
    
    return issues


# =============================================================================
# BÖLÜM 2: CANLI API TESTİ
# =============================================================================
def test_live_api(src: Path):
    """Mevcut fetcher ile API bağlantısını test et."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  🔬 BÖLÜM 2: CANLI API TESTİ")
    print(f"{'='*60}{RESET}")
    
    results = {}
    
    try:
        from data.fetcher import BitgetFetcher
        fetcher = BitgetFetcher()
        ok("BitgetFetcher oluşturuldu")
    except Exception as e:
        fail(f"BitgetFetcher oluşturulamadı: {e}")
        traceback.print_exc()
        return results
    
    # Test 1: Coin listesi
    print(f"\n  {CYAN}[1] Coin listesi:{RESET}")
    try:
        symbols = fetcher.get_all_usdt_futures()
        if len(symbols) > 100:
            ok(f"{len(symbols)} USDT-M çifti bulundu")
            results['symbols'] = True
        else:
            fail(f"Sadece {len(symbols)} çift — çok az!")
            results['symbols'] = False
    except Exception as e:
        fail(f"Hata: {e}")
        results['symbols'] = False
    
    # Test 2: Bitget ticker (exchange.fetch_tickers)
    print(f"\n  {CYAN}[2] Bitget batch ticker (exchange.fetch_tickers):{RESET}")
    try:
        start = time.time()
        all_tickers = fetcher.exchange.fetch_tickers()
        elapsed = time.time() - start
        
        usdt_tickers = {k: v for k, v in all_tickers.items() if k.endswith(':USDT')}
        
        # Dolu ticker yüzdesi
        filled = sum(1 for t in usdt_tickers.values() 
                    if t.get('last') and t.get('last', 0) > 0)
        
        ok(f"{len(usdt_tickers)} USDT ticker, {filled} fiyatı dolu ({elapsed:.1f}s)")
        
        btc = usdt_tickers.get('BTC/USDT:USDT', {})
        if btc:
            ok(f"BTC: ${btc.get('last', 0):,.2f} | Vol: ${btc.get('quoteVolume', 0):,.0f}")
        
        results['bitget_tickers'] = filled > 100
    except Exception as e:
        fail(f"Hata: {e}")
        results['bitget_tickers'] = False
    
    # Test 3: fetcher.fetch_tickers() (eğer varsa — v3 metodu)
    print(f"\n  {CYAN}[3] fetcher.fetch_tickers() (varsa):{RESET}")
    if hasattr(fetcher, 'fetch_tickers'):
        try:
            tickers = fetcher.fetch_tickers()
            ok(f"fetch_tickers() → {len(tickers)} ticker")
            
            # BTC var mı?
            btc = tickers.get('BTC/USDT:USDT', {})
            if btc and btc.get('last', 0) > 0:
                ok(f"BTC fiyatı: ${btc['last']:,.2f}")
                results['fetch_tickers'] = True
            else:
                fail("BTC fiyatı YOK — filtreleme bozuk!")
                results['fetch_tickers'] = False
                
        except Exception as e:
            fail(f"Hata: {e}")
            results['fetch_tickers'] = False
    else:
        info("fetch_tickers() metodu yok (eski fetcher versiyonu)")
        results['fetch_tickers'] = None  # Yok ama sorun değil
    
    # Test 4: Tek ticker
    print(f"\n  {CYAN}[4] get_ticker() tek sembol:{RESET}")
    try:
        ticker = fetcher.get_ticker('BTC/USDT:USDT')
        price = ticker.get('last', 0) or ticker.get('close', 0) or 0
        
        if price > 1000:
            ok(f"BTC: ${price:,.2f}")
            results['single_ticker'] = True
        else:
            fail(f"BTC fiyatı bozuk: {price}")
            results['single_ticker'] = False
    except Exception as e:
        fail(f"Hata: {e}")
        results['single_ticker'] = False
    
    # Test 5: OHLCV (Binance)
    print(f"\n  {CYAN}[5] OHLCV verisi (Binance):{RESET}")
    try:
        df = fetcher.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=50)
        
        if not df.empty and len(df) > 10:
            ok(f"BTC 1h: {len(df)} bar, son: ${df['close'].iloc[-1]:,.2f}")
            results['ohlcv'] = True
        else:
            fail(f"OHLCV boş veya yetersiz: {len(df)} bar")
            results['ohlcv'] = False
    except Exception as e:
        fail(f"Hata: {e}")
        results['ohlcv'] = False
    
    return results


# =============================================================================
# BÖLÜM 3: PİPELİNE ADIM ADIM TEST
# =============================================================================
def test_pipeline_steps(src: Path):
    """Pipeline'ın her adımını tek tek test et."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  🔄 BÖLÜM 3: PİPELİNE ADIM ADIM TEST")
    print(f"{'='*60}{RESET}")
    
    # --- Adım 1: CoinScanner ---
    print(f"\n  {CYAN}[Adım 1] CoinScanner.scan():{RESET}")
    try:
        from scanner.coin_scanner import CoinScanner
        scanner = CoinScanner(verbose=False)
        
        top_coins = scanner.scan(top_n=5, force_refresh=True)
        
        if top_coins and len(top_coins) > 0:
            ok(f"{len(top_coins)} coin bulundu")
            for i, c in enumerate(top_coins[:3], 1):
                print(f"    {i}. {c.coin:<8} ${c.price:>10,.2f} | "
                      f"Vol: ${c.volume_24h/1e6:>6,.0f}M | "
                      f"Skor: {c.composite_score:>5.1f}")
        else:
            fail("CoinScanner 0 coin döndürdü!")
            fail("← SORUN BURADA! Pipeline burada duruyor.")
            
            # Neden 0 coin? Debug edelim
            print(f"\n    {YELLOW}Debug: Scanner adımlarını tek tek kontrol ediyorum...{RESET}")
            
            # Sembol listesi
            symbols = scanner.fetcher.get_all_usdt_futures()
            print(f"    get_all_usdt_futures(): {len(symbols)} sembol")
            
            # Blacklist sonrası
            filtered = scanner._apply_blacklist(symbols)
            print(f"    Blacklist sonrası: {len(filtered)} sembol")
            
            # Ticker
            tickers = scanner._fetch_all_tickers(filtered[:5])  # Sadece 5 test
            print(f"    _fetch_all_tickers(5): {len(tickers)} ticker")
            
            if tickers:
                # İlk ticker'ın içeriğini göster
                first_sym = list(tickers.keys())[0]
                first_t = tickers[first_sym]
                print(f"    Örnek ticker ({first_sym}):")
                for key in ['last', 'bid', 'ask', 'quoteVolume']:
                    val = first_t.get(key)
                    status = GREEN if val and val != 0 else RED
                    print(f"      {status}{key}: {val}{RESET}")
            else:
                fail("    _fetch_all_tickers() BOŞ döndü!")
                fail("    ← Kök neden: Ticker çekme başarısız")
            
            return False
            
    except Exception as e:
        fail(f"CoinScanner hatası: {e}")
        traceback.print_exc()
        return False
    
    # --- Adım 2: IC Analiz ---
    print(f"\n  {CYAN}[Adım 2] IC Analiz (BTC):{RESET}")
    try:
        from data.fetcher import BitgetFetcher
        from data.preprocessor import DataPreprocessor
        from analysis.indicator_calculator import IndicatorCalculator
        from analysis.indicator_selector import IndicatorSelector
        
        fetcher = BitgetFetcher()
        pp = DataPreprocessor()
        calc = IndicatorCalculator()
        selector = IndicatorSelector()
        
        # BTC 1h verisi çek
        df = fetcher.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=300)
        
        if df.empty or len(df) < 100:
            fail(f"BTC 1h verisi yetersiz: {len(df)} bar")
            return False
        
        ok(f"BTC 1h: {len(df)} bar çekildi")
        
        # Preprocess
        df = pp.prepare(df) if hasattr(pp, 'prepare') else pp.full_pipeline(df)
        ok(f"Preprocess tamamlandı: {len(df)} bar")
        
        # İndikatörler
        df = calc.add_all_indicators(df)
        indicator_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        ok(f"{len(indicator_cols)} indikatör hesaplandı")
        
        # IC analiz
        ic_result = selector.analyze(df, forward_period=5)
        
        if ic_result:
            sig_count = ic_result.get('significant_count', 0) if isinstance(ic_result, dict) else getattr(ic_result, 'significant_count', 0)
            ok(f"IC analiz tamamlandı, {sig_count} anlamlı indikatör")
        else:
            warn("IC analiz sonucu boş (sinyal yok — bu normal olabilir)")
            
    except ImportError as e:
        warn(f"Import hatası (modül eksik): {e}")
        info("Bu kritik değil — fetcher ve scanner çalışıyorsa sorun yok")
    except Exception as e:
        warn(f"IC analiz hatası: {e}")
        info("Bu aşamada hata olabilir — asıl soru scanner'ın coin bulup bulmadığı")
    
    # --- Adım 3: Pipeline tek döngü ---
    print(f"\n  {CYAN}[Adım 3] Pipeline tek döngü (dry-run):{RESET}")
    try:
        from main import HybridTradingPipeline
        
        pipeline = HybridTradingPipeline(dry_run=True, top_n=3, verbose=False)
        
        if not pipeline._init_balance():
            warn("Bakiye başlatılamadı (paper trade'de sorun olmaz)")
        
        report = pipeline.run_cycle()
        
        print(f"    Status: {report.status}")
        print(f"    Taranan: {report.total_scanned}")
        print(f"    Analiz: {report.total_analyzed}")
        print(f"    Gate+: {report.total_above_gate}")
        print(f"    İşlem: {report.total_traded}")
        print(f"    Süre: {report.elapsed:.1f}s")
        
        if report.errors:
            warn(f"Hatalar ({len(report.errors)}):")
            for err in report.errors[:3]:
                print(f"      {RED}{err[:100]}{RESET}")
        
        if report.total_scanned == 0:
            fail("Pipeline 0 coin taradı!")
            fail("← CoinScanner sorunu — ticker çekme başarısız")
        elif report.total_analyzed == 0:
            warn("Taranan var ama analiz edilemiyor")
            info("← OHLCV veya IC analiz sorunu")
        elif report.total_above_gate == 0:
            info("Analiz yapıldı ama hiç coin gate eşiğini geçemedi")
            info("← Normal olabilir veya threshold çok yüksek")
        elif report.total_traded == 0:
            info("Gate geçen var ama işlem yapılmadı")
            info("← AI WAIT dedi veya execution sorunu")
        else:
            ok(f"{report.total_traded} işlem yapıldı!")
            
    except Exception as e:
        warn(f"Pipeline hatası: {e}")
        traceback.print_exc()
    
    return True


# =============================================================================
# BÖLÜM 4: OTOMATİK DÜZELTME
# =============================================================================
def auto_fix(src: Path, issues: list):
    """Tespit edilen sorunları otomatik düzelt."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  🔧 BÖLÜM 4: OTOMATİK DÜZELTME")
    print(f"{'='*60}{RESET}")
    
    if not issues:
        ok("Düzeltilecek sorun yok!")
        return
    
    for issue in issues:
        
        # SORUN: coin_scanner Binance ticker kullanıyor (bozuk filtre)
        if issue == 'scanner_binance_ticker':
            scanner_path = src / 'scanner' / 'coin_scanner.py'
            content = scanner_path.read_text(encoding='utf-8')
            
            # Eski (bozuk): self.fetcher.fetch_tickers(symbols)
            # Doğru: self.fetcher.exchange.fetch_tickers() + filtre
            old = 'self.fetcher.fetch_tickers(symbols)'
            new = 'self.fetcher.exchange.fetch_tickers()'
            
            if old in content:
                # Ayrıca return satırını da düzelt
                content = content.replace(old, new)
                
                # all_tickers return'ünü de düzelt
                old_return = 'return all_tickers'
                new_return = 'return {s: all_tickers[s] for s in symbols if s in all_tickers}'
                content = content.replace(old_return, new_return)
                
                scanner_path.write_text(content, encoding='utf-8')
                ok("coin_scanner.py: Bitget ticker'a geri döndürüldü")
            else:
                warn("coin_scanner.py: Beklenen pattern bulunamadı")
        
        # SORUN: main.py'de volume_24h kullanılıyor (fetcher_v3 formatı)
        if issue == 'main_volume_24h':
            main_path = src / 'main.py'
            content = main_path.read_text(encoding='utf-8')
            
            old = "ticker.get('volume_24h', 0)"
            new = "ticker.get('quoteVolume', 0)"
            
            if old in content:
                content = content.replace(old, new)
                main_path.write_text(content, encoding='utf-8')
                ok("main.py: volume_24h → quoteVolume düzeltildi")
        
        # SORUN: fetcher_v3 yüklü (Binance ticker filtresi bozuk)
        if issue == 'fetcher_v3_loaded':
            fetcher_path = src / 'data' / 'fetcher.py'
            backup_path = src / 'data' / 'fetcher_YEDEK.py'
            
            if backup_path.exists():
                # Yedekten geri yükle
                import shutil
                shutil.copy2(backup_path, fetcher_path)
                ok("fetcher.py: Yedekten geri yüklendi (hybrid versiyon)")
            else:
                warn("fetcher.py: Yedek bulunamadı — MANUEL düzeltme gerekli")
                warn("  apply_fix.py --undo komutunu dene")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}")
    print(f"  🔍 PİPELİNE DEBUG & DÜZELTME ARACI")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{RESET}")
    
    src = find_src()
    info(f"src dizini: {src}")
    
    # Bölüm 1: Dosya durumu
    issues = check_file_state(src)
    
    # Bölüm 2: API testi
    api_results = test_live_api(src)
    
    # Bölüm 3: Pipeline adım adım
    test_pipeline_steps(src)
    
    # Bölüm 4: Otomatik düzeltme önerisi
    if issues:
        print(f"\n{BOLD}{'='*60}")
        print(f"  ⚠️  {len(issues)} SORUN TESPİT EDİLDİ")
        print(f"{'='*60}{RESET}")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        answer = input(f"\n  Otomatik düzelt? (e/h): ").strip().lower()
        if answer in ('e', 'evet', 'y', 'yes'):
            auto_fix(src, issues)
            print(f"\n  {GREEN}✅ Düzeltmeler uygulandı. Tekrar test et:{RESET}")
            print(f"  python debug_pipeline.py")
        else:
            info("Düzeltme atlandı")
    else:
        print(f"\n{GREEN}  ✅ Dosya durumu temiz{RESET}")
    
    # Son durum
    print(f"\n{BOLD}{'='*60}")
    print(f"  📊 SONUÇ")
    print(f"{'='*60}{RESET}")
    
    if api_results.get('symbols') and api_results.get('bitget_tickers') and api_results.get('ohlcv'):
        print(f"  {GREEN}API'ler çalışıyor. Sorun pipeline mantığında olabilir.{RESET}")
        print(f"  Çıktıyı Claude'a yapıştır — birlikte düzeltelim.")
    else:
        failed = [k for k, v in api_results.items() if v == False]
        print(f"  {RED}Başarısız testler: {', '.join(failed)}{RESET}")
        print(f"  Çıktıyı Claude'a yapıştır.")
    
    print()
