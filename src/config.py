# =============================================================================
# MERKEZİ YAPILANDIRMA MODÜLÜ (CONFIG)
# =============================================================================
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

ENV_FILE = PROJECT_ROOT / '.env'
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    alt_env = SRC_DIR / '.env'
    if alt_env.exists():
        load_dotenv(alt_env)

def _load_yaml() -> Dict:
    yaml_path = PROJECT_ROOT / 'config' / 'settings.yaml'
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}

_SETTINGS = _load_yaml()

def get_setting(key_path: str, default: Any = None) -> Any:
    keys = key_path.split('.')
    value = _SETTINGS
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

def get_env(key: str, default: str = None) -> Optional[str]:
    return os.getenv(key, default)

# =============================================================================
# YAPILANDIRMA SINIFLARI
# =============================================================================

@dataclass
class ExchangeConfig:
    """Borsa bağlantı yapılandırması (Binance Futures)"""
    id: str = "binance"                      # CCXT borsa ID'si
    market_type: str = "future"              # future = USDT-M Perpetual (Binance)
    default_symbol: str = "BTC/USDT:USDT"    # Binance futures formatı
    sandbox: bool = False                    
    
    api_key: str = ""
    api_secret: str = ""
    
    def __post_init__(self):
        # YENİ: Binance ortam değişkenleri
        self.api_key = get_env('BINANCE_API_KEY', '')
        self.api_secret = get_env('BINANCE_API_SECRET', '')
        
        self.id = get_setting('exchange.id', self.id)
        self.market_type = get_setting('exchange.market_type', self.market_type)
        self.sandbox = get_setting('exchange.sandbox', self.sandbox)
    
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 2.0
    max_open_positions: int = 3
    max_margin_per_trade_pct: float = 25.0
    max_total_margin_pct: float = 60.0
    min_leverage: int = 2
    max_leverage: int = 20
    min_risk_reward_ratio: float = 1.5
    daily_max_loss_pct: float = 6.0
    kill_switch_drawdown_pct: float = 15.0
    max_sl_pct: float = 8.0
    
    def __post_init__(self):
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
    no_trade: float = 60.0
    report_only: float = 70.0
    full_trade: float = 75.0
    
    def __post_init__(self):
        self.no_trade = get_setting('gate_keeper.no_trade_threshold', self.no_trade)
        self.report_only = get_setting('gate_keeper.report_only_threshold', self.report_only)
        self.full_trade = get_setting('gate_keeper.full_trade_threshold', self.full_trade)

@dataclass
class TelegramConfig:
    enabled: bool = True
    token: str = ""
    chat_id: str = ""
    
    def __post_init__(self):
        self.token = get_env('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = get_env('TELEGRAM_CHAT_ID', '')
        self.enabled = get_setting('telegram.enabled', self.enabled)
    
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

class AppConfig:
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.risk = RiskConfig()
        self.gate = GateKeeperConfig()
        self.telegram = TelegramConfig()
        
        self.timeframes = get_setting('analysis.timeframes', {})
        self.ic_weights = get_setting('analysis.ic_weights', {
            'top_ic': 0.40, 'avg_ic': 0.25, 'count': 0.15, 'consistency': 0.20
        })
        self.target_period = get_setting('analysis.target_period', 5)
    
    def print_status(self):
        print("=" * 60)
        print("⚙️  YAPILANDIRMA DURUMU")
        print("=" * 60)
        print(f"  Binance API: {'✅ Yapılandırılmış' if self.exchange.is_configured() else '❌ Eksik'}")
        print(f"  Telegram   : {'✅ Yapılandırılmış' if self.telegram.is_configured() else '❌ Eksik'}")
        print(f"  Sandbox    : {'⚠️  AÇIK (Demo)' if self.exchange.sandbox else '🔴 KAPALI (Canlı)'}")
        print(f"  Borsa      : {self.exchange.id.upper()}")
        print(f"  Max Kaldıraç: {self.risk.max_leverage}x")
        print(f"  Risk/İşlem : %{self.risk.risk_per_trade_pct}")
        print(f"  IC Eşiği   : {self.gate.full_trade}")
        print("=" * 60)

cfg = AppConfig()