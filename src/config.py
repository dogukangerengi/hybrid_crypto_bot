# =============================================================================
# MERKEZİ YAPILANDIRMA MODÜLÜ (CONFIG)
# =============================================================================
# Amaç: .env (gizli API key'ler) ve settings.yaml (parametreler) dosyalarını
# tek bir yerden yönetmek. Tüm modüller bu dosyayı import eder.
#
# Güvenlik: API key'ler ASLA kod içinde yazılmaz, .env'den okunur.
# Esneklik: Parametreler settings.yaml'dan okunur, restart gerekmez.
# =============================================================================

import os                                    # İşletim sistemi ortam değişkenleri
import yaml                                  # YAML dosya okuma
from pathlib import Path                     # Platform-bağımsız dosya yolları
from dotenv import load_dotenv               # .env dosyasından değişken yükleme
from dataclasses import dataclass, field     # Yapılandırılmış veri sınıfları
from typing import Dict, List, Optional, Any # Tip belirteçleri


# =============================================================================
# PROJE KÖK DİZİNİ TESPİTİ
# =============================================================================
# Bu dosya src/ altında olacak, proje kökü bir üst dizin
CURRENT_FILE = Path(__file__).resolve()      # Bu dosyanın tam yolu
SRC_DIR = CURRENT_FILE.parent                # src/ dizini
PROJECT_ROOT = SRC_DIR.parent                # Proje kök dizini (hybrid_crypto_bot/)


# =============================================================================
# .ENV DOSYASINI YÜKLE (API KEY'LER)
# =============================================================================
# .env dosyası proje kökünde olmalı: hybrid_crypto_bot/.env
# Bu fonksiyon .env'deki key=value çiftlerini os.environ'a yükler
ENV_FILE = PROJECT_ROOT / '.env'
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)                    # .env → os.environ
else:
    # Geliştirme sırasında src/ altında da arayalım
    alt_env = SRC_DIR / '.env'
    if alt_env.exists():
        load_dotenv(alt_env)


# =============================================================================
# SETTINGS.YAML YÜKLE (PARAMETRELER)
# =============================================================================
def _load_yaml() -> Dict:
    """
    settings.yaml dosyasını okur ve Python dict olarak döndürür.
    
    Dosya bulunamazsa boş dict döndürür (varsayılan değerler kullanılır).
    """
    yaml_path = PROJECT_ROOT / 'config' / 'settings.yaml'
    
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}   # None dönerse boş dict yap
    
    return {}                                # Dosya yoksa boş dict


# Global settings dict (bir kez yükle, her yerde kullan)
_SETTINGS = _load_yaml()


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Nokta-ayrımlı anahtar yolu ile settings.yaml'dan değer okur.
    
    Örnek:
        get_setting('risk.max_leverage', 20)
        → settings.yaml'daki risk: max_leverage: 20 değerini döndürür
    
    Parametreler:
    ------------
    key_path : str
        Nokta ile ayrılmış anahtar yolu (örn: 'exchange.id')
    default : Any
        Anahtar bulunamazsa döndürülecek varsayılan değer
    
    Döndürür:
    --------
    Any
        Bulunan değer veya varsayılan
    """
    keys = key_path.split('.')               # 'risk.max_leverage' → ['risk', 'max_leverage']
    value = _SETTINGS                        # En üst seviyeden başla
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]               # Bir seviye daha derine in
        else:
            return default                   # Anahtar bulunamazsa varsayılan döndür
    
    return value


def get_env(key: str, default: str = None) -> Optional[str]:
    """
    Ortam değişkeninden değer okur (.env dosyasından yüklenmiş).
    
    Parametreler:
    ------------
    key : str
        Ortam değişkeni adı (örn: 'BITGET_API_KEY')
    default : str
        Bulunamazsa varsayılan değer
    
    Döndürür:
    --------
    str veya None
    """
    return os.getenv(key, default)


# =============================================================================
# YAPILANDIRMA SINIFLARI (TİP GÜVENLİ ERİŞİM)
# =============================================================================

@dataclass
class ExchangeConfig:
    """Borsa bağlantı yapılandırması."""
    id: str = "bitget"                       # CCXT borsa ID'si
    market_type: str = "swap"                # swap = USDT-M Perpetual
    default_symbol: str = "BTC/USDT:USDT"    # Bitget futures format
    sandbox: bool = False                     # Demo hesap modu
    
    # API key'ler (.env'den okunur)
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""                     # Bitget'e özel ek güvenlik
    
    def __post_init__(self):
        """Dataclass oluşturulduktan sonra .env'den key'leri yükle."""
        self.api_key = get_env('BITGET_API_KEY', '')
        self.api_secret = get_env('BITGET_API_SECRET', '')
        self.passphrase = get_env('BITGET_PASSPHRASE', '')
        
        # settings.yaml'dan borsa ayarlarını oku
        self.id = get_setting('exchange.id', self.id)
        self.market_type = get_setting('exchange.market_type', self.market_type)
        self.sandbox = get_setting('exchange.sandbox', self.sandbox)
    
    def is_configured(self) -> bool:
        """API key'lerin doldurulup doldurulmadığını kontrol eder."""
        return bool(self.api_key and self.api_secret and self.passphrase)


@dataclass
class RiskConfig:
    """Risk yönetimi yapılandırması."""
    risk_per_trade_pct: float = 2.0          # İşlem başına max risk yüzdesi
    max_open_positions: int = 3              # Max açık pozisyon sayısı
    max_margin_per_trade_pct: float = 25.0   # İşlem başına max margin yüzdesi
    max_total_margin_pct: float = 60.0       # Toplam max margin yüzdesi
    min_leverage: int = 2                    # Min kaldıraç
    max_leverage: int = 20                   # Max kaldıraç
    min_risk_reward_ratio: float = 1.5       # Min RR oranı
    daily_max_loss_pct: float = 6.0          # Günlük max kayıp yüzdesi
    kill_switch_drawdown_pct: float = 15.0   # Kill switch DD yüzdesi
    max_sl_pct: float = 8.0                  # Max SL mesafesi (%)
    
    def __post_init__(self):
        """settings.yaml'dan risk parametrelerini yükle."""
        self.risk_per_trade_pct = get_setting('risk.risk_per_trade_pct', self.risk_per_trade_pct)
        self.max_open_positions = get_setting('risk.max_open_positions', self.max_open_positions)
        self.max_margin_per_trade_pct = get_setting('risk.max_margin_per_trade_pct', self.max_margin_per_trade_pct)
        self.max_total_margin_pct = get_setting('risk.max_total_margin_pct', self.max_total_margin_pct)
        self.min_leverage = get_setting('risk.min_leverage', self.min_leverage)
        self.max_leverage = get_setting('risk.max_leverage', self.max_leverage)
        self.min_risk_reward_ratio = get_setting('risk.min_risk_reward_ratio', self.min_risk_reward_ratio)
        self.daily_max_loss_pct = get_setting('risk.daily_max_loss_pct', self.daily_max_loss_pct)
        self.kill_switch_drawdown_pct = get_setting('risk.kill_switch_drawdown_pct', self.kill_switch_drawdown_pct)
        self.max_sl_pct = get_setting('risk.max_sl_pct', self.max_sl_pct)


@dataclass
class GateKeeperConfig:
    """IC kapı bekçisi eşikleri."""
    no_trade: float = 60.0                   # IC < 40 → İşlem yapma
    report_only: float = 70.0                # IC 40-55 → Sadece rapor
    full_trade: float = 75.0                 # IC > 55 → AI + Trade
    
    def __post_init__(self):
        self.no_trade = get_setting('gate_keeper.no_trade_threshold', self.no_trade)
        self.report_only = get_setting('gate_keeper.report_only_threshold', self.report_only)
        self.full_trade = get_setting('gate_keeper.full_trade_threshold', self.full_trade)




@dataclass
class TelegramConfig:
    """Telegram bildirim yapılandırması."""
    enabled: bool = True
    token: str = ""
    chat_id: str = ""
    
    def __post_init__(self):
        self.token = get_env('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = get_env('TELEGRAM_CHAT_ID', '')
        self.enabled = get_setting('telegram.enabled', self.enabled)
    
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)


# =============================================================================
# ANA CONFIG SINIFI (TEK GİRİŞ NOKTASI)
# =============================================================================

class AppConfig:
    """
    Tüm yapılandırmayı tek yerden erişilebilir yapan ana sınıf.
    
    Kullanım:
    --------
    from config import AppConfig
    
    cfg = AppConfig()
    print(cfg.exchange.api_key)      # Bitget API key
    print(cfg.risk.max_leverage)     # Max kaldıraç
    print(cfg.gate.full_trade)       # IC eşiği
    """
    
    def __init__(self):
        self.exchange = ExchangeConfig()     # Borsa ayarları + API key'ler
        self.risk = RiskConfig()             # Risk yönetimi parametreleri
        self.gate = GateKeeperConfig()       # IC karar eşikleri
        self.telegram = TelegramConfig()     # Telegram yapılandırması
        
        # Analiz ayarları (dict olarak)
        self.timeframes = get_setting('analysis.timeframes', {})
        self.ic_weights = get_setting('analysis.ic_weights', {
            'top_ic': 0.40, 'avg_ic': 0.25, 'count': 0.15, 'consistency': 0.20
        })
        self.target_period = get_setting('analysis.target_period', 5)
    
    def print_status(self):
        """Yapılandırma durumunu yazdırır (debug için)."""
        print("=" * 60)
        print("⚙️  YAPILANDIRMA DURUMU")
        print("=" * 60)
        print(f"  Bitget API : {'✅ Yapılandırılmış' if self.exchange.is_configured() else '❌ Eksik'}")
        print(f"  Telegram   : {'✅ Yapılandırılmış' if self.telegram.is_configured() else '❌ Eksik'}")
        print(f"  Sandbox    : {'⚠️  AÇIK (Demo)' if self.exchange.sandbox else '🔴 KAPALI (Canlı)'}")
        print(f"  Borsa      : {self.exchange.id.upper()}")
        print(f"  Max Kaldıraç: {self.risk.max_leverage}x")
        print(f"  Risk/İşlem : %{self.risk.risk_per_trade_pct}")
        print(f"  IC Eşiği   : {self.gate.full_trade}")
        print("=" * 60)


# =============================================================================
# MODÜL SEVİYESİNDE TEK ÖRNEK (SINGLETON PATTERN)
# =============================================================================
# Her yerde `from config import cfg` ile kullanılır
# Tek bir örnek oluşturulur, tüm modüller aynı config'i paylaşır

cfg = AppConfig()


# =============================================================================
# TEST KODU
# =============================================================================
if __name__ == "__main__":
    cfg.print_status()
    
    print("\n📊 Timeframe'ler:")
    for tf, params in cfg.timeframes.items():
        print(f"  {tf}: {params}")
    
    print(f"\n🎯 IC Ağırlıkları: {cfg.ic_weights}")
    print(f"🎯 Target Period: {cfg.target_period}")
