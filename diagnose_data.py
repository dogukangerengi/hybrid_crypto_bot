#!/usr/bin/env python3
# =============================================================================
# TANI SCRIPTI — Boş Veri Sorunu Teşhisi
# =============================================================================
# Bu script Bitget ve Binance API'lerini ayrı ayrı test eder.
# Çalıştır: python diagnose_data.py
# NOT: Proje klasöründen çalıştırmana gerek yok, bağımsız çalışır.
# =============================================================================

import ccxt                                    # Borsa API kütüphanesi
import time                                    # Zamanlama
import json                                    # JSON formatlama
from datetime import datetime                  # Tarih/saat

# Renkli çıktı (terminal'de okunabilirlik)
GREEN  = "\033[92m"                            # Başarılı
RED    = "\033[91m"                            # Hata
YELLOW = "\033[93m"                            # Uyarı
CYAN   = "\033[96m"                            # Bilgi
RESET  = "\033[0m"                             # Renk sıfırla
BOLD   = "\033[1m"                             # Kalın

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")

# =============================================================================
# TEST 1: BITGET BAĞLANTISI
# =============================================================================
def test_bitget_connection():
    """Bitget API'ye bağlanıp market yüklenebiliyor mu?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 1: BITGET BAĞLANTISI")
    print(f"{'='*60}{RESET}")
    
    try:
        # Bitget exchange objesi oluştur (API key gerekmez - public data)
        exchange = ccxt.bitget({
            'options': {'defaultType': 'swap'},  # USDT-M Futures modu
            'enableRateLimit': True,             # Rate limit koruması
            'timeout': 15000,                    # 15 saniye timeout
        })
        
        # Market bilgilerini yükle
        start = time.time()
        exchange.load_markets()
        elapsed = time.time() - start
        
        # USDT-M çiftlerini say
        usdt_futures = [s for s in exchange.markets if s.endswith(':USDT')]
        ok(f"Bitget bağlantısı başarılı ({elapsed:.1f}s)")
        ok(f"{len(usdt_futures)} USDT-M Futures çifti bulundu")
        
        return exchange, usdt_futures
        
    except ccxt.NetworkError as e:
        fail(f"AĞ HATASI: {e}")
        warn("Bitget Türkiye'den erişilemez olabilir!")
        warn("VPN kullanıyor musun? VPN ile tekrar dene.")
        return None, []
    except ccxt.ExchangeError as e:
        fail(f"BORSA HATASI: {e}")
        return None, []
    except Exception as e:
        fail(f"BİLİNMEYEN HATA: {type(e).__name__}: {e}")
        return None, []


# =============================================================================
# TEST 2: BITGET TICKER VERİSİ
# =============================================================================
def test_bitget_tickers(exchange):
    """Bitget'ten ticker verisi geliyor mu?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 2: BITGET TICKER VERİSİ")
    print(f"{'='*60}{RESET}")
    
    if not exchange:
        fail("Bitget bağlantısı yok, ticker test edilemiyor")
        return {}
    
    # Yöntem A: Tek sembol ticker
    print(f"\n  {CYAN}A) Tek sembol ticker (BTC/USDT:USDT):{RESET}")
    try:
        ticker = exchange.fetch_ticker('BTC/USDT:USDT')
        
        last_price = ticker.get('last')        # Son fiyat
        bid = ticker.get('bid')                # En iyi alış
        ask = ticker.get('ask')                # En iyi satış
        volume = ticker.get('quoteVolume')     # 24h USDT hacim
        change = ticker.get('percentage')      # 24h % değişim
        
        # Veri kalitesini kontrol et
        has_price = last_price and last_price > 0
        has_bid_ask = bid and ask and bid > 0 and ask > 0
        has_volume = volume and volume > 0
        
        if has_price:
            ok(f"BTC Fiyat: ${last_price:,.2f}")
        else:
            fail(f"BTC Fiyat BOŞ! Değer: {last_price}")
        
        if has_bid_ask:
            spread = (ask - bid) / bid * 100
            ok(f"Bid: ${bid:,.2f} | Ask: ${ask:,.2f} | Spread: {spread:.4f}%")
        else:
            fail(f"Bid/Ask BOŞ! Bid: {bid}, Ask: {ask}")
        
        if has_volume:
            ok(f"24h Hacim: ${volume:,.0f}")
        else:
            fail(f"Hacim BOŞ! Değer: {volume}")
        
        info(f"24h Değişim: {change}%")
        
        # Ham veriyi göster (debug)
        print(f"\n  {CYAN}Ham ticker alanları:{RESET}")
        important_keys = ['last', 'bid', 'ask', 'high', 'low', 'quoteVolume', 
                         'baseVolume', 'percentage', 'close']
        for key in important_keys:
            val = ticker.get(key)
            status = GREEN if val and val != 0 else RED
            print(f"    {status}{key}: {val}{RESET}")
            
    except Exception as e:
        fail(f"Tek ticker hatası: {e}")
    
    # Yöntem B: Toplu ticker (batch)
    print(f"\n  {CYAN}B) Toplu ticker (fetch_tickers):{RESET}")
    try:
        start = time.time()
        all_tickers = exchange.fetch_tickers()
        elapsed = time.time() - start
        
        # USDT futures ticker'larını filtrele
        usdt_tickers = {k: v for k, v in all_tickers.items() if k.endswith(':USDT')}
        
        ok(f"{len(all_tickers)} toplam ticker ({elapsed:.1f}s)")
        ok(f"{len(usdt_tickers)} USDT-M ticker")
        
        # Boş/dolu ticker istatistiği
        empty_price = 0
        empty_volume = 0
        empty_bidask = 0
        
        for sym, t in usdt_tickers.items():
            if not t.get('last') or t.get('last', 0) == 0:
                empty_price += 1
            if not t.get('quoteVolume') or t.get('quoteVolume', 0) == 0:
                empty_volume += 1
            if not t.get('bid') or not t.get('ask'):
                empty_bidask += 1
        
        total = len(usdt_tickers)
        if total > 0:
            if empty_price > total * 0.5:
                fail(f"Fiyatı BOŞ: {empty_price}/{total} ({empty_price/total*100:.0f}%)")
            else:
                ok(f"Fiyatı dolu: {total - empty_price}/{total}")
            
            if empty_volume > total * 0.5:
                fail(f"Hacmi BOŞ: {empty_volume}/{total} ({empty_volume/total*100:.0f}%)")
            else:
                ok(f"Hacmi dolu: {total - empty_volume}/{total}")
            
            if empty_bidask > total * 0.5:
                fail(f"Bid/Ask BOŞ: {empty_bidask}/{total} ({empty_bidask/total*100:.0f}%)")
            else:
                ok(f"Bid/Ask dolu: {total - empty_bidask}/{total}")
        
        # BTC örneğini göster
        btc_ticker = usdt_tickers.get('BTC/USDT:USDT', {})
        if btc_ticker:
            print(f"\n  {CYAN}BTC/USDT:USDT batch ticker:{RESET}")
            for key in ['last', 'bid', 'ask', 'quoteVolume', 'percentage']:
                val = btc_ticker.get(key)
                status = GREEN if val and val != 0 else RED
                print(f"    {status}{key}: {val}{RESET}")
        
        return usdt_tickers
        
    except Exception as e:
        fail(f"Toplu ticker hatası: {e}")
        return {}


# =============================================================================
# TEST 3: BINANCE BAĞLANTISI
# =============================================================================
def test_binance_connection():
    """Binance API'ye bağlanıp veri çekilebiliyor mu?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 3: BINANCE BAĞLANTISI")
    print(f"{'='*60}{RESET}")
    
    try:
        # Binance Futures (API key gerekmez - public data)
        binance = ccxt.binance({
            'options': {'defaultType': 'future'},  # USDT-M Futures
            'enableRateLimit': True,
            'timeout': 15000,
        })
        
        ok("Binance exchange objesi oluşturuldu")
        return binance
        
    except Exception as e:
        fail(f"Binance bağlantı hatası: {e}")
        return None


# =============================================================================
# TEST 4: BINANCE OHLCV VERİSİ
# =============================================================================
def test_binance_ohlcv(binance):
    """Binance'den OHLCV verisi çekilebiliyor mu?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 4: BINANCE OHLCV VERİSİ")
    print(f"{'='*60}{RESET}")
    
    if not binance:
        fail("Binance bağlantısı yok")
        return
    
    test_cases = [
        ('BTC/USDT', '1h', 100),               # BTC saatlik
        ('ETH/USDT', '4h', 50),                 # ETH 4 saatlik
        ('SOL/USDT', '15m', 200),               # SOL 15dk
    ]
    
    for symbol, tf, limit in test_cases:
        try:
            start = time.time()
            ohlcv = binance.fetch_ohlcv(symbol, tf, limit=limit)
            elapsed = time.time() - start
            
            if ohlcv and len(ohlcv) > 0:
                # İlk ve son mumun zamanını göster
                first_ts = datetime.fromtimestamp(ohlcv[0][0] / 1000)
                last_ts = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
                last_close = ohlcv[-1][4]       # Close fiyat
                
                ok(f"{symbol} {tf}: {len(ohlcv)} mum ({elapsed:.1f}s)")
                info(f"  Aralık: {first_ts.strftime('%m/%d %H:%M')} → {last_ts.strftime('%m/%d %H:%M')}")
                info(f"  Son Kapanış: ${last_close:,.2f}")
            else:
                fail(f"{symbol} {tf}: BOŞ VERİ döndü!")
                
        except Exception as e:
            fail(f"{symbol} {tf}: {e}")
        
        time.sleep(0.2)                        # Rate limit koruması


# =============================================================================
# TEST 5: BINANCE TICKER VERİSİ (ALTERNATİF)
# =============================================================================
def test_binance_tickers(binance):
    """Binance ticker'ları CoinScanner alternatifi olarak kullanılabilir mi?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 5: BINANCE TICKER (ALTERNATİF SCANNER)")
    print(f"{'='*60}{RESET}")
    
    if not binance:
        fail("Binance bağlantısı yok")
        return {}
    
    try:
        start = time.time()
        all_tickers = binance.fetch_tickers()
        elapsed = time.time() - start
        
        # USDT futures ticker'larını filtrele
        usdt_tickers = {k: v for k, v in all_tickers.items() 
                       if k.endswith('/USDT') and ':' not in k}
        
        ok(f"{len(all_tickers)} toplam ticker ({elapsed:.1f}s)")
        ok(f"{len(usdt_tickers)} USDT çifti")
        
        # Hacme göre sırala
        sorted_tickers = sorted(
            usdt_tickers.items(),
            key=lambda x: x[1].get('quoteVolume', 0) or 0,
            reverse=True
        )
        
        # Top 10 göster
        print(f"\n  {CYAN}Top 10 (Binance hacim sırası):{RESET}")
        for i, (sym, t) in enumerate(sorted_tickers[:10], 1):
            last = t.get('last', 0) or 0
            vol = t.get('quoteVolume', 0) or 0
            chg = t.get('percentage', 0) or 0
            bid = t.get('bid', 0) or 0
            ask = t.get('ask', 0) or 0
            
            print(f"    {i:>2}. {sym:<15} ${last:>12,.2f} | "
                  f"Vol: ${vol/1e6:>8,.0f}M | "
                  f"Chg: {chg:>+6.1f}% | "
                  f"Spread: {((ask-bid)/bid*100 if bid > 0 else 0):>.4f}%")
        
        # Boş veri kontrolü
        empty_count = sum(1 for _, t in usdt_tickers.items() 
                         if not t.get('last') or t.get('last', 0) == 0)
        
        if empty_count > len(usdt_tickers) * 0.3:
            warn(f"Fiyatı boş: {empty_count}/{len(usdt_tickers)}")
        else:
            ok(f"Fiyatı dolu: {len(usdt_tickers) - empty_count}/{len(usdt_tickers)}")
        
        return usdt_tickers
        
    except Exception as e:
        fail(f"Binance ticker hatası: {e}")
        return {}


# =============================================================================
# TEST 6: SEMBOL UYUMLULUĞU (BITGET vs BINANCE)
# =============================================================================
def test_symbol_mapping(bitget_exchange, binance):
    """Bitget sembollerinin Binance karşılığı var mı?"""
    print(f"\n{BOLD}{'='*60}")
    print(f"  TEST 6: SEMBOL UYUMLULUĞU (Bitget → Binance)")
    print(f"{'='*60}{RESET}")
    
    if not bitget_exchange or not binance:
        warn("Bir veya iki borsa bağlantısı yok, uyumluluk test edilemiyor")
        return
    
    # Binance marketlerini yükle
    try:
        binance.load_markets()
    except:
        fail("Binance marketleri yüklenemedi")
        return
    
    # Bitget USDT futures
    bitget_symbols = [s for s in bitget_exchange.markets if s.endswith(':USDT')]
    
    matched = 0
    unmatched = []
    
    for bg_sym in bitget_symbols[:50]:          # İlk 50'yi test et (hız için)
        # Bitget: 'BTC/USDT:USDT' → Binance: 'BTC/USDT'
        bn_sym = bg_sym.split(':')[0]           # 'BTC/USDT'
        
        if bn_sym in binance.markets:
            matched += 1
        else:
            unmatched.append(bg_sym)
    
    ok(f"Eşleşen: {matched}/50")
    
    if unmatched:
        warn(f"Eşleşmeyen ({len(unmatched)}): {', '.join(unmatched[:5])}")
    else:
        ok("Tüm test edilen semboller Binance'de mevcut")


# =============================================================================
# SONUÇ VE ÖNERİ
# =============================================================================
def print_diagnosis(bitget_ok, bitget_tickers_ok, binance_ok, binance_ohlcv_ok, binance_tickers_ok):
    """Test sonuçlarına göre teşhis ve çözüm öner."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  📋 TEŞHİS SONUCU")
    print(f"{'='*60}{RESET}")
    
    print(f"""
  Bitget Bağlantı:    {'✅' if bitget_ok else '❌'}
  Bitget Ticker:      {'✅' if bitget_tickers_ok else '❌'}
  Binance Bağlantı:   {'✅' if binance_ok else '❌'}
  Binance OHLCV:      {'✅' if binance_ohlcv_ok else '❌'}
  Binance Ticker:     {'✅' if binance_tickers_ok else '❌'}
    """)
    
    if not bitget_ok:
        print(f"  {RED}SORUN: Bitget API'ye erişim yok!{RESET}")
        print(f"  {YELLOW}ÇÖZÜM: VPN kullan veya Bitget IP whitelist kontrol et{RESET}")
        print(f"  {YELLOW}ALTERNATİF: Tüm veriyi Binance'den çek, sadece emir için Bitget API key kullan{RESET}")
    
    elif not bitget_tickers_ok:
        print(f"  {RED}SORUN: Bitget ticker verisi boş geliyor!{RESET}")
        print(f"  {YELLOW}ÇÖZÜM: CoinScanner'ı Binance ticker kullanacak şekilde güncelle{RESET}")
    
    if binance_ok and binance_ohlcv_ok and binance_tickers_ok:
        print(f"\n  {GREEN}✅ Binance tam çalışıyor! Çözüm:{RESET}")
        print(f"  {GREEN}   → Tüm veri (OHLCV + Ticker) Binance'den gelsin{RESET}")
        print(f"  {GREEN}   → Bitget sadece emir göndermek için kullanılsın{RESET}")
        print(f"  {GREEN}   → fetcher_v2.py dosyasını kullan{RESET}")
    
    print(f"\n{'='*60}\n")


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}")
    print(f"  🔍 VERİ SORUNU TEŞHİS ARACI")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}{RESET}")
    
    # Test 1: Bitget bağlantı
    bitget_exchange, usdt_futures = test_bitget_connection()
    bitget_ok = bitget_exchange is not None
    
    # Test 2: Bitget ticker
    bitget_tickers = {}
    bitget_tickers_ok = False
    if bitget_ok:
        bitget_tickers = test_bitget_tickers(bitget_exchange)
        # En az %50'si dolu ise OK say
        if bitget_tickers:
            total = len(bitget_tickers)
            filled = sum(1 for t in bitget_tickers.values() 
                        if t.get('last') and t.get('last', 0) > 0)
            bitget_tickers_ok = filled > total * 0.5
    
    # Test 3: Binance bağlantı
    binance = test_binance_connection()
    binance_ok = binance is not None
    
    # Test 4: Binance OHLCV
    binance_ohlcv_ok = False
    if binance_ok:
        try:
            ohlcv = binance.fetch_ohlcv('BTC/USDT', '1h', limit=10)
            binance_ohlcv_ok = len(ohlcv) > 0
            test_binance_ohlcv(binance)
        except:
            pass
    
    # Test 5: Binance ticker
    binance_tickers = {}
    binance_tickers_ok = False
    if binance_ok:
        binance_tickers = test_binance_tickers(binance)
        binance_tickers_ok = len(binance_tickers) > 10
    
    # Test 6: Sembol uyumluluğu
    if bitget_ok and binance_ok:
        test_symbol_mapping(bitget_exchange, binance)
    
    # Sonuç
    print_diagnosis(
        bitget_ok, 
        bitget_tickers_ok, 
        binance_ok, 
        binance_ohlcv_ok, 
        binance_tickers_ok
    )
