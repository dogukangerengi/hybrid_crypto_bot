#!/usr/bin/env python3
# =============================================================================
# TEK SATIRLIK FIX — fetcher.py'ye fetch_tickers() ekle
# =============================================================================
# Sorun: coin_scanner.py → self.fetcher.fetch_tickers() çağırıyor
#        ama yeni fetcher.py'de bu metod yok
# Çözüm: Bitget exchange.fetch_tickers()'a delege eden wrapper ekle
#
# Çalıştır: cd ~/hybrid_crypto_bot && python fix_ticker.py
# =============================================================================

from pathlib import Path
import sys

GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"

# src dizinini bul
cwd = Path.cwd()
src = cwd / 'src' if (cwd / 'src').exists() else cwd

fetcher_path = src / 'data' / 'fetcher.py'
scanner_path = src / 'scanner' / 'coin_scanner.py'

if not fetcher_path.exists():
    print(f"{RED}❌ fetcher.py bulunamadı!{RESET}")
    sys.exit(1)

content = fetcher_path.read_text(encoding='utf-8')

# =============================================================================
# 1. fetcher.py'ye fetch_tickers() metodu ekle (yoksa)
# =============================================================================
if 'def fetch_tickers(' not in content:
    # get_ticker metodundan ÖNCE ekle
    marker = "    def get_ticker("
    
    new_method = '''    def fetch_tickers(self, symbols=None):
        """
        Toplu ticker verisi — Bitget exchange'e delege eder.
        CoinScanner._fetch_all_tickers() bu metodu çağırır.
        """
        all_tickers = self.exchange.fetch_tickers()
        if symbols:
            return {s: all_tickers[s] for s in symbols if s in all_tickers}
        return all_tickers

'''
    
    if marker in content:
        content = content.replace(marker, new_method + marker)
        fetcher_path.write_text(content, encoding='utf-8')
        print(f"{GREEN}✅ fetcher.py: fetch_tickers() metodu eklendi{RESET}")
    else:
        # Alternatif: validate_data'dan önce ekle
        marker2 = "    def validate_data("
        if marker2 in content:
            content = content.replace(marker2, new_method + marker2)
            fetcher_path.write_text(content, encoding='utf-8')
            print(f"{GREEN}✅ fetcher.py: fetch_tickers() metodu eklendi (validate_data öncesi){RESET}")
        else:
            print(f"{RED}❌ Uygun konum bulunamadı — manuel ekle{RESET}")
            sys.exit(1)
else:
    print(f"{GREEN}✅ fetcher.py: fetch_tickers() zaten mevcut{RESET}")

# =============================================================================
# 2. coin_scanner.py kontrol — hangi çağrı kullanılıyor?
# =============================================================================
if scanner_path.exists():
    sc_content = scanner_path.read_text(encoding='utf-8')
    
    if 'self.fetcher.fetch_tickers(' in sc_content:
        print(f"  ℹ️  coin_scanner: self.fetcher.fetch_tickers() kullanıyor")
    elif 'self.fetcher.exchange.fetch_tickers()' in sc_content:
        print(f"  ℹ️  coin_scanner: self.fetcher.exchange.fetch_tickers() kullanıyor")
    else:
        print(f"  ⚠️  coin_scanner: ticker çağrısı bulunamadı")

# =============================================================================
# 3. Hızlı doğrulama
# =============================================================================
print(f"\n🔬 Doğrulama:")

sys.path.insert(0, str(src))
# Cache temizle
for mod in list(sys.modules.keys()):
    if 'data' in mod or 'fetcher' in mod:
        del sys.modules[mod]

try:
    from data.fetcher import BitgetFetcher
    f = BitgetFetcher()
    
    # Coin listesi
    symbols = f.get_all_usdt_futures()
    print(f"  {GREEN}✅ get_all_usdt_futures(): {len(symbols)} çift{RESET}")
    
    # fetch_tickers (yeni metod)
    if hasattr(f, 'fetch_tickers'):
        tickers = f.fetch_tickers(['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        btc = tickers.get('BTC/USDT:USDT', {})
        price = btc.get('last', 0)
        if price > 1000:
            print(f"  {GREEN}✅ fetch_tickers(): BTC ${price:,.2f}{RESET}")
        else:
            print(f"  {RED}❌ fetch_tickers(): BTC fiyat bozuk{RESET}")
    
    # OHLCV
    df = f.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=10)
    print(f"  {GREEN}✅ fetch_ohlcv(): {len(df)} bar{RESET}")
    
    print(f"\n{GREEN}🎉 Artık çalışır! Şimdi:{RESET}")
    print(f"  cd src && python main.py --dry-run")
    
except Exception as e:
    print(f"  {RED}❌ Hata: {e}{RESET}")
    import traceback
    traceback.print_exc()
