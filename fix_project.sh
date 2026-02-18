#!/bin/bash
# =============================================================================
# HYBRID CRYPTO BOT — PROJE DÜZELTME SCRİPTİ
# =============================================================================
# Çalıştırma:
#   cd hybrid_crypto_bot
#   chmod +x fix_project.sh
#   ./fix_project.sh
#
# Bu script:
# 1. Eski (Binance) dosyalarını yedekler
# 2. Yeni (Bitget v2) dosyalarını src/ altına taşır
# 3. Import uyumsuzluklarını düzeltir
# 4. Eksik dosyaları oluşturur (settings.yaml, .env.example)
# =============================================================================

set -e  # Hata durumunda dur

echo "=============================================="
echo "  🔧 PROJE DÜZELTME BAŞLIYOR"
echo "=============================================="

# Proje kökünde miyiz kontrol et
if [ ! -f "requirements.txt" ]; then
    echo "❌ HATA: Bu scripti proje kök dizininde çalıştırın!"
    echo "   cd hybrid_crypto_bot && ./fix_project.sh"
    exit 1
fi

# =============================================================================
# ADIM 1: YEDEK AL
# =============================================================================
echo ""
echo "📦 [1/6] Eski dosyalar yedekleniyor..."

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# src/ altındaki eski dosyaları yedekle
if [ -d "src/data" ]; then
    cp -r src/data "$BACKUP_DIR/src_data_old"
    echo "   ✓ src/data/ yedeklendi"
fi

if [ -d "src/indicators" ]; then
    cp -r src/indicators "$BACKUP_DIR/src_indicators_old"
    echo "   ✓ src/indicators/ yedeklendi"
fi

# Kök dizindeki v2 dosyalarını da yedekle
if [ -d "data" ] && [ ! -L "data" ]; then
    cp -r data "$BACKUP_DIR/root_data_v2"
    echo "   ✓ data/ (v2) yedeklendi"
fi

if [ -d "indicators" ] && [ ! -L "indicators" ]; then
    cp -r indicators "$BACKUP_DIR/root_indicators_v2"
    echo "   ✓ indicators/ (v2) yedeklendi"
fi

echo "   📁 Yedek dizini: $BACKUP_DIR/"

# =============================================================================
# ADIM 2: V2 (BİTGET) DOSYALARINI src/ ALTINA TAŞI
# =============================================================================
echo ""
echo "📂 [2/6] Bitget (v2) dosyaları src/ altına taşınıyor..."

# --- DATA MODÜLÜ ---
# Kök dizinde data/ klasörü varsa (v2 Bitget versiyonu), src/data/ ile değiştir
if [ -d "data" ] && [ -f "data/fetcher.py" ]; then
    # data/__init__.py'de BitgetFetcher varsa bu v2
    if grep -q "BitgetFetcher" data/__init__.py 2>/dev/null; then
        echo "   → Bitget v2 data/ modülü tespit edildi"
        rm -rf src/data
        cp -r data src/data
        echo "   ✓ src/data/ → Bitget v2 ile güncellendi"
    else
        echo "   ⚠️  data/ v2 olarak doğrulanamadı, atlanıyor"
    fi
else
    echo "   ℹ️  Kök dizinde data/ yok, src/data/ korunuyor"
fi

# --- İNDİKATÖR MODÜLÜ ---
# Kök dizinde indicators/ klasörü varsa (v2), src/indicators/ ile değiştir
if [ -d "indicators" ] && [ -f "indicators/categories.py" ]; then
    # v2'de get_total_output_columns fonksiyonu var
    if grep -q "get_total_output_columns" indicators/categories.py 2>/dev/null; then
        echo "   → Bitget v2 indicators/ modülü tespit edildi"
        rm -rf src/indicators
        cp -r indicators src/indicators
        echo "   ✓ src/indicators/ → Bitget v2 ile güncellendi"
    else
        echo "   ⚠️  indicators/ v2 olarak doğrulanamadı, atlanıyor"
    fi
else
    echo "   ℹ️  Kök dizinde indicators/ yok, src/indicators/ korunuyor"
fi

# =============================================================================
# ADIM 3: __init__.py DOSYALARINI DÜZELT (DataFetcher ALIAS)
# =============================================================================
echo ""
echo "🔗 [3/6] Import alias'ları düzeltiliyor..."

# src/data/__init__.py — BitgetFetcher'ı DataFetcher olarak da export et
# Bu sayede eski modüller (main.py, telegram_bot.py) bozulmaz
cat > src/data/__init__.py << 'PYEOF'
# =============================================================================
# VERİ MODÜLÜ (DATA MODULE) — Bitget Futures
# =============================================================================
# Bitget USDT-M Perpetual Futures veri çekme ve ön işleme.
#
# Kullanım:
#   from data import BitgetFetcher, DataPreprocessor
#   # veya geriye uyumluluk için:
#   from data import DataFetcher  # → BitgetFetcher alias'ı
# =============================================================================

from .fetcher import BitgetFetcher
from .preprocessor import DataPreprocessor

# Geriye uyumluluk alias'ı — eski modüller DataFetcher bekliyor
# main.py, telegram_bot.py, app.py hepsi DataFetcher import eder
DataFetcher = BitgetFetcher

__all__ = [
    'BitgetFetcher',       # Yeni isim (Bitget Futures)
    'DataFetcher',         # Eski isim (alias, geriye uyumlu)
    'DataPreprocessor',    # Veri ön işleme
]

__version__ = '2.1.0'     # v2.1: alias eklendi
PYEOF

echo "   ✓ src/data/__init__.py güncellendi (DataFetcher alias eklendi)"

# =============================================================================
# ADIM 4: TELEGRAM_BOT.PY'Yİ DÜZELT
# =============================================================================
echo ""
echo "🤖 [4/6] telegram_bot.py düzeltiliyor..."

if [ -f "src/telegram_bot.py" ]; then
    # 1. Binance referanslarını Bitget'e çevir
    # get_supported_coins() fonksiyonundaki ccxt.binance() → ccxt.bitget()
    sed -i.bak 's/ccxt\.binance()/ccxt.bitget({"options": {"defaultType": "swap"}})/g' src/telegram_bot.py
    
    # 2. Spot filtresi → Futures filtresi
    # Eski: if symbol.endswith('/USDT') and ':' not in symbol
    # Yeni: if symbol.endswith(':USDT')
    sed -i.bak "s/if symbol.endswith('\/USDT') and ':' not in symbol:/if symbol.endswith(':USDT'):/g" src/telegram_bot.py
    
    # 3. Coin çıkarma formatını düzelt
    # Eski: coin = symbol.replace('/USDT', '')
    # Yeni: coin = symbol.split('/')[0]
    sed -i.bak "s/coin = symbol.replace('\/USDT', '')/coin = symbol.split('\/')[0]/g" src/telegram_bot.py
    
    # 4. Spot only yorumunu kaldır
    sed -i.bak 's/# Spot only//g' src/telegram_bot.py
    
    # Yedek dosyaları temizle
    rm -f src/telegram_bot.py.bak
    
    echo "   ✓ telegram_bot.py: Binance → Bitget referansları güncellendi"
else
    echo "   ⚠️  src/telegram_bot.py bulunamadı"
fi

# =============================================================================
# ADIM 5: EKSİK DOSYALARI OLUŞTUR
# =============================================================================
echo ""
echo "📝 [5/6] Eksik dosyalar oluşturuluyor..."

# --- .env.example ---
if [ ! -f ".env.example" ]; then
    cat > .env.example << 'ENVEOF'
# =============================================================================
# HYBRID CRYPTO BOT — ORTAM DEĞİŞKENLERİ
# =============================================================================
# Bu dosyayı ".env" olarak kopyalayın ve kendi değerlerinizi girin:
#   cp .env.example .env
#
# ⚠️  .env dosyası GİT'E GİRMEMELİDİR!
# =============================================================================

# --- BİTGET API (Futures) ---
BITGET_API_KEY=your_api_key_here
BITGET_API_SECRET=your_api_secret_here
BITGET_PASSPHRASE=your_passphrase_here

# --- GOOGLE GEMİNİ AI ---
GEMINI_API_KEY=your_gemini_key_here

# --- TELEGRAM BOT ---
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
ENVEOF
    echo "   ✓ .env.example oluşturuldu"
fi

# --- .gitignore ---
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'GITEOF'
# Ortam değişkenleri (API KEY'LER!)
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
venv/
.venv/

# IDE
.vscode/
.idea/

# Log ve veri
logs/
data/cache/
*.log

# macOS
.DS_Store

# Yedek dosyalar
backup_*/
GITEOF
    echo "   ✓ .gitignore oluşturuldu"
fi

# --- config/ dizini ve settings.yaml ---
mkdir -p config
# settings.yaml ayrı dosya olarak oluşturulacak (fix_settings.yaml → config/settings.yaml)
echo "   ℹ️  config/settings.yaml → ayrı dosya olarak sağlanacak"

# --- Eksik dizinler ---
mkdir -p src/scanner
mkdir -p src/ai
mkdir -p src/execution
mkdir -p src/utils
mkdir -p logs
mkdir -p tests

# Scanner __init__.py
if [ ! -f "src/scanner/__init__.py" ]; then
    echo '# Dinamik coin tarayıcı modülü (Adım 4)' > src/scanner/__init__.py
fi

# AI __init__.py
if [ ! -f "src/ai/__init__.py" ]; then
    echo '# AI optimizasyon modülü - Gemini (Adım 6)' > src/ai/__init__.py
fi

# Execution __init__.py
if [ ! -f "src/execution/__init__.py" ]; then
    echo '# Bitget emir yönetimi modülü (Adım 7)' > src/execution/__init__.py
fi

# Utils __init__.py
if [ ! -f "src/utils/__init__.py" ]; then
    echo '# Yardımcı araçlar' > src/utils/__init__.py
fi

echo "   ✓ Eksik dizinler ve __init__.py dosyaları oluşturuldu"

# =============================================================================
# ADIM 6: KÖK DİZİNDEKİ DUPLICATE DOSYALARI TEMİZLE
# =============================================================================
echo ""
echo "🧹 [6/6] Kök dizindeki duplicate v2 dosyaları temizleniyor..."

# Artık src/ altında v2 var, kök dizindeki kopyalar gereksiz
# Ama silinmeden önce yedek alındığını doğrula
if [ -d "$BACKUP_DIR" ]; then
    # Kök dizindeki data/ ve indicators/'ı sil (src/ altında zaten var)
    if [ -d "data" ] && [ -d "src/data" ]; then
        rm -rf data
        echo "   ✓ Kök data/ silindi (src/data/ aktif)"
    fi
    
    if [ -d "indicators" ] && [ -d "src/indicators" ]; then
        rm -rf indicators
        echo "   ✓ Kök indicators/ silindi (src/indicators/ aktif)"
    fi
else
    echo "   ⚠️  Yedek bulunamadı, kök dizin dosyaları korunuyor"
fi

# =============================================================================
# SONUÇ
# =============================================================================
echo ""
echo "=============================================="
echo "  ✅ DÜZELTME TAMAMLANDI!"
echo "=============================================="
echo ""
echo "  📁 Proje yapısı (Roadmap uyumlu):"
echo "  hybrid_crypto_bot/"
echo "  ├── src/"
echo "  │   ├── config.py          ← Merkezi config (Bitget)"
echo "  │   ├── main.py            ← Ana orkestrasyon"
echo "  │   ├── telegram_bot.py    ← Telegram bot"
echo "  │   ├── app.py             ← Streamlit dashboard"
echo "  │   ├── data/"
echo "  │   │   ├── fetcher.py     ← BitgetFetcher (v2)"
echo "  │   │   └── preprocessor.py"
echo "  │   ├── indicators/"
echo "  │   │   ├── categories.py  ← 4 kategori, 58 indikatör"
echo "  │   │   ├── calculator.py  ← pandas-ta motor"
echo "  │   │   └── selector.py    ← IC bazlı seçim"
echo "  │   ├── notifications/"
echo "  │   ├── scanner/           ← (Adım 4)"
echo "  │   ├── ai/                ← (Adım 6)"
echo "  │   └── execution/         ← (Adım 7)"
echo "  ├── config/"
echo "  │   └── settings.yaml"
echo "  └── tests/"
echo ""
echo "  🧪 Test etmek için:"
echo "  cd src && python test_indicators.py"
echo ""
echo "  📦 Yedek: $BACKUP_DIR/"
echo "=============================================="
