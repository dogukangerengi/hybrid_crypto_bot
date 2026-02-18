# =============================================================================
# BİTGET BAĞLANTI TEST SCRİPTİ
# =============================================================================
# Amaç: Bitget API bağlantısını adım adım test etmek
# 
# Çalıştırma:
#   cd hybrid_crypto_bot/src
#   python test_bitget_connection.py
#
# Bu script şunları test eder:
# 1. CCXT ile Bitget'e bağlanma
# 2. Market listesi çekme (kaç USDT Futures çifti var?)
# 3. Güncel BTC fiyatı çekme (ticker)
# 4. OHLCV (mum) verisi çekme
# 5. Hesap bakiyesi sorgulama (API key gerektirir)
# =============================================================================

import sys                                   # Sistem çıkış kodları
import time                                  # Zaman ölçümü
from datetime import datetime, timezone      # Zaman damgaları
from pathlib import Path                     # Dosya yolları

# src/ dizininden çalıştığımızı varsayıyoruz
# config.py'yi import edebilmek için path ayarı
sys.path.insert(0, str(Path(__file__).parent))

import ccxt                                  # Kripto borsa unified API


def print_header(text: str):
    """Bölüm başlığı yazdırır."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(success: bool, message: str):
    """Test sonucu yazdırır."""
    icon = "✅" if success else "❌"
    print(f"  {icon} {message}")


def test_basic_connection():
    """
    TEST 1: Temel bağlantı (API key gerektirmez)
    
    CCXT ile Bitget'e bağlanıp market listesini çeker.
    Bu test sadece internet bağlantısı ve borsa erişilebilirliğini doğrular.
    """
    print_header("TEST 1: TEMEL BAĞLANTI (Public API)")
    
    try:
        # CCXT ile Bitget exchange nesnesi oluştur
        # 'swap' = USDT-M Perpetual Futures market'i
        exchange = ccxt.bitget({
            'options': {
                'defaultType': 'swap',       # Futures market
            }
        })
        
        # Market bilgilerini yükle
        # Bu çağrı tüm işlem çiftlerini, lot büyüklüklerini, kaldıraç limitlerini çeker
        start = time.time()
        exchange.load_markets()
        elapsed = time.time() - start
        
        # Sonuçları göster
        all_markets = list(exchange.markets.keys())
        
        # Sadece USDT-M Futures çiftlerini filtrele
        # Bitget'te futures semboller "BTC/USDT:USDT" formatındadır
        usdt_futures = [s for s in all_markets if s.endswith(':USDT')]
        
        print_result(True, f"Bitget bağlantısı başarılı ({elapsed:.1f}s)")
        print_result(True, f"Toplam market: {len(all_markets)}")
        print_result(True, f"USDT-M Futures: {len(usdt_futures)} çift")
        
        # İlk 10 USDT futures çiftini göster
        print(f"\n  📋 Örnek USDT-M Futures çiftleri:")
        for s in sorted(usdt_futures)[:10]:
            print(f"     {s}")
        print(f"     ... ve {len(usdt_futures)-10} çift daha")
        
        return exchange, True
        
    except ccxt.NetworkError as e:
        print_result(False, f"Ağ hatası: {e}")
        return None, False
    except ccxt.ExchangeError as e:
        print_result(False, f"Borsa hatası: {e}")
        return None, False
    except Exception as e:
        print_result(False, f"Beklenmeyen hata: {e}")
        return None, False


def test_ticker(exchange):
    """
    TEST 2: Güncel fiyat çekme (Ticker)
    
    BTC/USDT:USDT (Bitget Futures) için anlık fiyat bilgisi çeker.
    API key gerektirmez (public endpoint).
    """
    print_header("TEST 2: GÜNCEL FİYAT (Ticker)")
    
    symbol = "BTC/USDT:USDT"                # Bitget Futures BTC sembolü
    
    try:
        # fetch_ticker: Anlık fiyat, 24h hacim, 24h değişim
        ticker = exchange.fetch_ticker(symbol)
        
        print_result(True, f"Ticker çekildi: {symbol}")
        print(f"\n  💰 Fiyat Bilgileri:")
        print(f"     Son Fiyat : ${ticker['last']:,.2f}")
        print(f"     Bid/Ask   : ${ticker['bid']:,.2f} / ${ticker['ask']:,.2f}")
        print(f"     24h Yüksek: ${ticker['high']:,.2f}")
        print(f"     24h Düşük : ${ticker['low']:,.2f}")
        print(f"     24h Hacim : ${ticker.get('quoteVolume', 0):,.0f} USDT")
        print(f"     24h Değişim: {ticker.get('percentage', 0):+.2f}%")
        
        return True
        
    except Exception as e:
        print_result(False, f"Ticker hatası: {e}")
        return False


def test_ohlcv(exchange):
    """
    TEST 3: OHLCV (mum) verisi çekme
    
    BTC/USDT:USDT için son 100 mum verisini çeker.
    API key gerektirmez (public endpoint).
    
    OHLCV = Open, High, Low, Close, Volume
    Her mum bir zaman dilimindeki fiyat hareketini temsil eder.
    """
    print_header("TEST 3: OHLCV VERİSİ (Mum Çekme)")
    
    symbol = "BTC/USDT:USDT"
    timeframe = "1h"                         # 1 saatlik mumlar
    limit = 100                              # Son 100 mum
    
    try:
        import pandas as pd                  # Veri yapısı için
        
        # fetch_ohlcv: Geçmiş mum verilerini çeker
        # Döndürdüğü format: [[timestamp, open, high, low, close, volume], ...]
        start = time.time()
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        elapsed = time.time() - start
        
        if not ohlcv:
            print_result(False, "Boş OHLCV verisi döndü")
            return False
        
        # DataFrame'e çevir (pandas ile daha okunur)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Unix timestamp → okunabilir tarih (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        print_result(True, f"{len(df)} mum çekildi ({elapsed:.1f}s)")
        print(f"\n  📊 Veri Özeti ({symbol} {timeframe}):")
        print(f"     Başlangıç : {df.index[0].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"     Bitiş     : {df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"     Fiyat Min : ${df['low'].min():,.2f}")
        print(f"     Fiyat Max : ${df['high'].max():,.2f}")
        print(f"     Son Close : ${df['close'].iloc[-1]:,.2f}")
        
        # Son 3 mum
        print(f"\n  📈 Son 3 Mum:")
        for _, row in df.tail(3).iterrows():
            change = ((row['close'] - row['open']) / row['open']) * 100
            direction = "🟢" if change >= 0 else "🔴"
            print(f"     {direction} O:{row['open']:,.0f} H:{row['high']:,.0f} "
                  f"L:{row['low']:,.0f} C:{row['close']:,.0f} ({change:+.2f}%)")
        
        return True
        
    except Exception as e:
        print_result(False, f"OHLCV hatası: {e}")
        return False


def test_multi_timeframe(exchange):
    """
    TEST 4: Çoklu timeframe veri çekme
    
    Tüm aktif timeframe'ler (5m → 4h) için kısa veri çekerek
    hepsinin çalıştığını doğrular.
    """
    print_header("TEST 4: ÇOKLU TİMEFRAME")
    
    symbol = "BTC/USDT:USDT"
    timeframes = ['5m', '15m', '30m', '1h', '2h', '4h']
    
    success_count = 0
    
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=10)
            
            if ohlcv and len(ohlcv) > 0:
                print_result(True, f"{tf:<4} → {len(ohlcv)} mum OK")
                success_count += 1
            else:
                print_result(False, f"{tf:<4} → Boş veri")
                
        except Exception as e:
            print_result(False, f"{tf:<4} → {str(e)[:50]}")
        
        time.sleep(0.2)                      # Rate limiting (API limiti aşmamak için)
    
    print(f"\n  Sonuç: {success_count}/{len(timeframes)} timeframe başarılı")
    return success_count == len(timeframes)


def test_balance(exchange):
    """
    TEST 5: Hesap bakiyesi sorgulama (API key GEREKTİRİR)
    
    Bu test sadece .env dosyasında geçerli Bitget API key'ler varsa çalışır.
    API key yoksa atlar (bağlantı testi zaten tamamlandı).
    """
    print_header("TEST 5: HESAP BAKİYESİ (Private API)")
    
    try:
        # config.py'den ayarları oku
        from config import cfg
        
        if not cfg.exchange.is_configured():
            print("  ⚠️  API key bulunamadı (.env dosyasını kontrol et)")
            print("  ℹ️  Bu test opsiyonel, public API testleri yeterli")
            return True                      # API key yoksa bile test başarılı say
        
        # API key'li exchange oluştur
        exchange_private = ccxt.bitget({
            'apiKey': cfg.exchange.api_key,
            'secret': cfg.exchange.api_secret,
            'password': cfg.exchange.passphrase,  # Bitget'e özel passphrase
            'options': {
                'defaultType': 'swap',       # Futures
            }
        })
        
        # Futures bakiyesini çek
        # fetch_balance(): Tüm asset bakiyelerini döndürür
        balance = exchange_private.fetch_balance()
        
        # USDT bakiyesini bul
        usdt_total = float(balance.get('USDT', {}).get('total', 0))
        usdt_free = float(balance.get('USDT', {}).get('free', 0))
        usdt_used = float(balance.get('USDT', {}).get('used', 0))
        
        print_result(True, f"Bakiye sorgulandı")
        print(f"\n  💰 USDT Futures Bakiye:")
        print(f"     Toplam   : ${usdt_total:,.2f}")
        print(f"     Kullanılabilir: ${usdt_free:,.2f}")
        print(f"     Kullanımda   : ${usdt_used:,.2f}")
        
        # Risk hesabı göster
        risk_pct = cfg.risk.risk_per_trade_pct
        risk_amount = usdt_total * (risk_pct / 100)
        print(f"\n  ⚖️ Risk Hesabı (%{risk_pct}):")
        print(f"     İşlem başına risk: ${risk_amount:,.2f}")
        print(f"     Max açık pozisyon: {cfg.risk.max_open_positions}")
        print(f"     Max toplam risk  : ${risk_amount * cfg.risk.max_open_positions:,.2f}")
        
        return True
        
    except ccxt.AuthenticationError:
        print_result(False, "API key hatalı! Bitget API yönetimini kontrol et")
        return False
    except ccxt.PermissionDenied:
        print_result(False, "API key izinleri yetersiz! Trade + Read izni gerekli")
        return False
    except Exception as e:
        print_result(False, f"Bakiye hatası: {e}")
        return False


def test_market_info(exchange):
    """
    TEST 6: Market bilgisi detayı
    
    BTC/USDT:USDT için lot büyüklüğü, min sipariş, kaldıraç limiti gibi
    teknik bilgileri çeker. Emir göndermeden önce bu bilgiler gerekli.
    """
    print_header("TEST 6: MARKET BİLGİSİ (Lot, Kaldıraç)")
    
    symbol = "BTC/USDT:USDT"
    
    try:
        market = exchange.market(symbol)     # Market detayını çek
        
        print_result(True, f"Market bilgisi alındı: {symbol}")
        print(f"\n  📋 Kontrat Bilgileri:")
        print(f"     Tip        : {market.get('type', 'N/A')}")
        print(f"     Kontrat Boy: {market.get('contractSize', 'N/A')}")
        print(f"     Min Miktar : {market.get('limits', {}).get('amount', {}).get('min', 'N/A')}")
        print(f"     Min Tutar  : {market.get('limits', {}).get('cost', {}).get('min', 'N/A')} USDT")
        print(f"     Precision  : Fiyat={market.get('precision', {}).get('price', 'N/A')}, "
              f"Miktar={market.get('precision', {}).get('amount', 'N/A')}")
        
        # Bazı ekstra bilgiler (varsa)
        info = market.get('info', {})
        if 'maxLever' in info:
            print(f"     Max Kaldıraç: {info['maxLever']}x")
        
        return True
        
    except Exception as e:
        print_result(False, f"Market bilgi hatası: {e}")
        return False


# =============================================================================
# ANA ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    
    print("\n" + "🚀" * 20)
    print("  BİTGET BAĞLANTI TESTİ")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("🚀" * 20)
    
    results = {}
    
    # Test 1: Temel bağlantı
    exchange, success = test_basic_connection()
    results['Temel Bağlantı'] = success
    
    if exchange is None:
        print("\n❌ Temel bağlantı başarısız, diğer testler atlanıyor.")
        sys.exit(1)
    
    # Test 2: Ticker
    results['Ticker'] = test_ticker(exchange)
    
    # Test 3: OHLCV
    results['OHLCV'] = test_ohlcv(exchange)
    
    # Test 4: Multi-timeframe
    results['Multi-TF'] = test_multi_timeframe(exchange)
    
    # Test 5: Bakiye (opsiyonel - API key gerektirir)
    results['Bakiye'] = test_balance(exchange)
    
    # Test 6: Market bilgisi
    results['Market Info'] = test_market_info(exchange)
    
    # === SONUÇ ÖZETİ ===
    print_header("SONUÇ ÖZETİ")
    
    all_passed = True
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test_name}")
        if not passed:
            all_passed = False
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\n  📊 Sonuç: {passed}/{total} test başarılı")
    
    if all_passed:
        print("\n  🎉 TÜM TESTLER BAŞARILI! Bitget bağlantısı hazır.")
        print("  → Sonraki adım: Veri katmanı (fetcher.py)")
    else:
        print("\n  ⚠️  Bazı testler başarısız. Yukarıdaki hataları kontrol et.")
    
    print()
