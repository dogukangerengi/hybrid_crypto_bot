#!/bin/bash
# =============================================================================
# HİBRİT KRİPTO BOT - M4 MACBOOK KURULUM SCRİPTİ
# =============================================================================
# Çalıştırma:
#   chmod +x setup_project.sh    # Çalıştırma izni ver (ilk seferlik)
#   ./setup_project.sh           # Kurulumu başlat
#
# Bu script şunları yapar:
# 1. Python versiyonunu kontrol eder (3.11+ gerekli)
# 2. Sanal ortam (venv) oluşturur
# 3. Bağımlılıkları yükler
# 4. .env şablonunu kopyalar
# 5. Git repository başlatır
# 6. Bağlantı testini çalıştırır
# =============================================================================

# Renkli çıktı tanımları
RED='\033[0;31m'                             # Hata rengi
GREEN='\033[0;32m'                           # Başarı rengi
YELLOW='\033[1;33m'                          # Uyarı rengi
BLUE='\033[0;34m'                            # Bilgi rengi
NC='\033[0m'                                 # Renk sıfırlama (No Color)

# Proje dizini (bu script'in bulunduğu dizin)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo -e "${BLUE}🚀 =========================================${NC}"
echo -e "${BLUE}   HİBRİT KRİPTO BOT - KURULUM${NC}"
echo -e "${BLUE}   M4 Macbook Air Optimize${NC}"
echo -e "${BLUE}🚀 =========================================${NC}"
echo ""

# =========================================================================
# ADIM 1: PYTHON KONTROLÜ
# =========================================================================
echo -e "${YELLOW}[1/6] Python kontrol ediliyor...${NC}"

# python3 veya python komutunu bul
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"                  # Python 3.12 varsa tercih et
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"                     # Genel python3
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"                      # Fallback
else
    echo -e "${RED}❌ Python bulunamadı!${NC}"
    echo "   Yükleme: brew install python@3.12"
    exit 1
fi

# Versiyon kontrolü (3.11 minimum)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}❌ Python $PYTHON_VERSION çok eski! 3.11+ gerekli${NC}"
    echo "   Güncelle: brew install python@3.12"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION ($PYTHON_CMD)${NC}"

# =========================================================================
# ADIM 2: SANAL ORTAM (VENV)
# =========================================================================
echo ""
echo -e "${YELLOW}[2/6] Sanal ortam oluşturuluyor...${NC}"

cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    echo -e "${YELLOW}   ⚠️  venv zaten mevcut, atlanıyor${NC}"
else
    # venv oluştur: İzole Python ortamı (sistem Python'unu bozmaz)
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ venv oluşturuldu${NC}"
fi

# Sanal ortamı aktif et
source venv/bin/activate
echo -e "${GREEN}   → Aktif Python: $(which python)${NC}"

# pip güncelle (eski pip sorun çıkarabilir)
echo "   pip güncelleniyor..."
pip install --upgrade pip --quiet

# =========================================================================
# ADIM 3: BAĞIMLILIKLARI YÜKLE
# =========================================================================
echo ""
echo -e "${YELLOW}[3/6] Bağımlılıklar yükleniyor (bu 1-2 dk sürebilir)...${NC}"

pip install -r requirements.txt --quiet 2>&1 | tail -5

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tüm bağımlılıklar yüklendi${NC}"
    
    # Kritik paketleri doğrula
    echo "   Paket kontrolü:"
    python -c "import ccxt; print(f'   ✓ ccxt {ccxt.__version__}')" 2>/dev/null || echo "   ✗ ccxt HATA"
    python -c "import pandas; print(f'   ✓ pandas {pandas.__version__}')" 2>/dev/null || echo "   ✗ pandas HATA"
    python -c "import pandas_ta; print(f'   ✓ pandas-ta OK')" 2>/dev/null || echo "   ✗ pandas-ta HATA"
    python -c "import google.generativeai; print(f'   ✓ google-generativeai OK')" 2>/dev/null || echo "   ✗ google-generativeai HATA"
    python -c "import telegram; print(f'   ✓ python-telegram-bot OK')" 2>/dev/null || echo "   ✗ telegram HATA"
else
    echo -e "${RED}❌ Bazı paketler yüklenemedi${NC}"
    echo "   Manuel dene: pip install -r requirements.txt"
fi

# =========================================================================
# ADIM 4: .ENV DOSYASI
# =========================================================================
echo ""
echo -e "${YELLOW}[4/6] API yapılandırması kontrol ediliyor...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env dosyası mevcut${NC}"
    
    # Key'lerin doldurulup doldurulmadığını kontrol et
    if grep -q "your_binance" .env 2>/dev/null; then
        echo -e "${YELLOW}   ⚠️  Binance API key'leri henüz girilmemiş!${NC}"
        echo "   → .env dosyasını düzenle: code .env"
    else
        echo "   → Binance key'leri girilmiş ✓"
    fi
else
    # .env.example'dan kopyala
    cp .env.example .env
    echo -e "${YELLOW}⚠️  .env dosyası oluşturuldu (şablondan)${NC}"
    echo -e "${YELLOW}   → ŞİMDİ .env dosyasını düzenleyip gerçek key'lerini gir!${NC}"
    echo "   → Komut: code .env"
fi

# =========================================================================
# ADIM 5: GİT BAŞLAT
# =========================================================================
echo ""
echo -e "${YELLOW}[5/6] Git repository kontrol ediliyor...${NC}"

if [ -d ".git" ]; then
    echo -e "${GREEN}✅ Git zaten başlatılmış${NC}"
else
    git init --quiet
    git add .
    git commit -m "🚀 Initial commit: Hybrid Crypto Bot project skeleton" --quiet
    echo -e "${GREEN}✅ Git başlatıldı ve ilk commit yapıldı${NC}"
fi

# =========================================================================
# ADIM 6: BAĞLANTI TESTİ
# =========================================================================
echo ""
echo -e "${YELLOW}[6/6] Binance bağlantı testi...${NC}"
echo "   (API key yoksa sadece public testler çalışır)"
echo ""

cd src
python test_binance_connection.py
cd ..

# =========================================================================
# ÖZET
# =========================================================================
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   KURULUM TAMAMLANDI!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "📁 Proje: ${GREEN}$PROJECT_DIR${NC}"
echo -e "🐍 Python: ${GREEN}$PYTHON_VERSION${NC}"
echo -e "📦 venv: ${GREEN}$PROJECT_DIR/venv${NC}"
echo ""
echo "Sonraki adımlar:"
echo "  1. .env dosyasını düzenle (API key'lerini gir)"
echo "     → code .env"
echo ""
echo "  2. Sanal ortamı aktif et (her terminal açılışında):"
echo "     → source venv/bin/activate"
echo ""
echo "  3. Bağlantıyı tekrar test et:"
echo "     → cd src && python test_binance_connection.py"
echo ""
echo "  4. Geliştirmeye başla! 🚀"
echo ""
