#!/bin/bash
# =============================================================================
# HYBRID CRYPTO BOT — SCHEDULER KURULUM VE YÖNETİM SCRİPTİ
# =============================================================================
# Kullanım:
#   chmod +x setup_scheduler.sh
#   ./setup_scheduler.sh install    # LaunchAgent kur
#   ./setup_scheduler.sh uninstall  # LaunchAgent kaldır
#   ./setup_scheduler.sh status     # Durum kontrol
#   ./setup_scheduler.sh logs       # Son logları göster
#   ./setup_scheduler.sh run        # Manuel tek çalıştırma
#   ./setup_scheduler.sh test       # Telegram bağlantı testi
#   ./setup_scheduler.sh cycle      # Tek döngü (scheduler olmadan)
# =============================================================================

# Renkler
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Proje dizinleri
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"   # Bu script'in bulunduğu dizin
SRC_DIR="$PROJECT_DIR/src"
LOG_DIR="$PROJECT_DIR/logs"
VENV_DIR="$PROJECT_DIR/venv"

# LaunchAgent dosyaları
PLIST_SRC="$PROJECT_DIR/com.hybrid.crypto.bot.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.hybrid.crypto.bot.plist"
LABEL="com.hybrid.crypto.bot"

# Log dizinini oluştur
mkdir -p "$LOG_DIR"

# ---- BANNER ----
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════╗"
echo "║   🚀 Hybrid Crypto Bot — Scheduler      ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${NC}"

case "$1" in
    install)
        echo -e "${GREEN}📦 LaunchAgent kuruluyor...${NC}"
        echo ""
        
        # Plist dosyası var mı?
        if [ ! -f "$PLIST_SRC" ]; then
            echo -e "${RED}❌ Hata: $PLIST_SRC bulunamadı${NC}"
            echo "   Önce plist dosyasını proje dizinine kopyalayın."
            exit 1
        fi
        
        # venv var mı?
        if [ ! -d "$VENV_DIR" ]; then
            echo -e "${RED}❌ Hata: venv bulunamadı ($VENV_DIR)${NC}"
            echo "   Önce: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
            exit 1
        fi
        
        # Eski job'ı durdur (varsa)
        launchctl unload "$PLIST_DST" 2>/dev/null
        
        # Plist'i kopyala
        cp "$PLIST_SRC" "$PLIST_DST"
        echo -e "${GREEN}  ✓ Plist kopyalandı → ~/Library/LaunchAgents/${NC}"
        
        # LaunchAgent'ı yükle
        launchctl load "$PLIST_DST"
        echo -e "${GREEN}  ✓ LaunchAgent yüklendi${NC}"
        
        echo ""
        echo -e "${GREEN}✅ Kurulum tamamlandı!${NC}"
        echo "   Bot her 90 dakikada bir çalışacak"
        echo "   Mode: DRY RUN (paper trade)"
        echo ""
        echo "   Durum kontrolü: $0 status"
        echo "   Loglar: $0 logs"
        ;;
        
    uninstall)
        echo -e "${YELLOW}🗑️  LaunchAgent kaldırılıyor...${NC}"
        
        launchctl unload "$PLIST_DST" 2>/dev/null
        rm -f "$PLIST_DST"
        
        echo -e "${GREEN}✅ Scheduler kaldırıldı${NC}"
        ;;
        
    status)
        echo -e "${GREEN}📊 Scheduler Durumu${NC}"
        echo "═══════════════════════"
        
        if launchctl list 2>/dev/null | grep -q "$LABEL"; then
            echo -e "${GREEN}  ✓ Scheduler AKTİF${NC}"
            launchctl list 2>/dev/null | grep "$LABEL"
        else
            echo -e "${YELLOW}  ○ Scheduler PASİF${NC}"
        fi
        
        echo ""
        echo "📋 Yapılandırma:"
        echo "   Proje: $PROJECT_DIR"
        echo "   venv:  $VENV_DIR"
        echo "   Logs:  $LOG_DIR"
        
        # API durumu
        echo ""
        echo "🔑 API Durumu:"
        cd "$SRC_DIR"
        source "$VENV_DIR/bin/activate" 2>/dev/null
        python -c "
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('$PROJECT_DIR/.env'))
from config import cfg
print(f'   Binance:   {\"✅\" if cfg.exchange.is_configured() else \"❌ Key eksik\"}')
print(f'   ML Modeli:   {\"✅\" if cfg.ai.is_configured() else \"❌ Key eksik\"}')
print(f'   Telegram: {\"✅\" if cfg.telegram.is_configured() else \"❌ Key eksik\"}')
" 2>/dev/null || echo "   ⚠️ Config kontrol edilemedi"
        
        echo ""
        echo "📜 Son 10 log satırı:"
        if [ -f "$LOG_DIR/stdout.log" ]; then
            tail -10 "$LOG_DIR/stdout.log"
        else
            echo "   (henüz log yok)"
        fi
        ;;
        
    logs)
        echo -e "${GREEN}📜 Son Loglar${NC}"
        echo "═══════════════"
        
        echo -e "\n${BLUE}--- stdout.log ---${NC}"
        if [ -f "$LOG_DIR/stdout.log" ]; then
            tail -50 "$LOG_DIR/stdout.log"
        else
            echo "  (boş)"
        fi
        
        echo -e "\n${RED}--- stderr.log ---${NC}"
        if [ -f "$LOG_DIR/stderr.log" ]; then
            tail -20 "$LOG_DIR/stderr.log"
        else
            echo "  (boş)"
        fi
        
        # Günlük bot logu
        TODAY_LOG="$LOG_DIR/bot_$(date +%Y%m%d).log"
        if [ -f "$TODAY_LOG" ]; then
            echo -e "\n${GREEN}--- bot_$(date +%Y%m%d).log ---${NC}"
            tail -50 "$TODAY_LOG"
        fi
        ;;
        
    run)
        echo -e "${GREEN}🚀 Manuel Çalıştırma (tek döngü)${NC}"
        echo "═══════════════════════════════════"
        
        cd "$SRC_DIR"
        source "$VENV_DIR/bin/activate"
        python main.py --dry-run --top 15
        ;;
    
    cycle)
        echo -e "${GREEN}🔄 Tek Döngü (schedule olmadan)${NC}"
        echo "═══════════════════════════════════"
        
        cd "$SRC_DIR"
        source "$VENV_DIR/bin/activate"
        python main.py --dry-run --top 15
        ;;
        
    test)
        echo -e "${GREEN}🧪 Bağlantı Testi${NC}"
        echo "═══════════════════"
        
        cd "$SRC_DIR"
        source "$VENV_DIR/bin/activate"
        
        python -c "
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('$PROJECT_DIR/.env'))

print('1. Config kontrolü...')
from config import cfg
cfg.print_status()

print('\n2. Telegram testi...')
from notifications import TelegramNotifier
notifier = TelegramNotifier()
if notifier.is_configured():
    if notifier.test_connection_sync():
        notifier.send_alert_sync('🧪 Test', 'Hybrid Crypto Bot bağlantı testi başarılı!', 'success')
        print('   ✅ Telegram mesajı gönderildi')
    else:
        print('   ❌ Telegram bağlantı hatası')
else:
    print('   ⚠️ Telegram yapılandırılmamış')

print('\n3. Binance bağlantı testi...')
from execution import BinanceExecutor
executor = BinanceExecutor(dry_run=True)
print('   ✅ Executor başlatıldı (DRY RUN)')
"
        ;;
        
    *)
        echo "Kullanım: $0 {install|uninstall|status|logs|run|cycle|test}"
        echo ""
        echo "  install    LaunchAgent'ı kur (her saat başı çalışır)"
        echo "  uninstall  LaunchAgent'ı kaldır"
        echo "  status     Scheduler durumunu göster"
        echo "  logs       Son logları göster"
        echo "  run        Manuel tek çalıştırma (DRY RUN, top 15)"
        echo "  cycle      Kısa tek döngü (DRY RUN, top 15)"
        echo "  test       Bağlantı testi (Config + Telegram + Binance)"
        exit 1
        ;;
esac
