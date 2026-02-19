# =============================================================================
# TELEGRAM BİLDİRİM MODÜLÜ - v3.0 (FULL ENTEGRASYONs)
# =============================================================================
# v3.0 Değişiklikler:
# - ExecutionResult → trade açma/kapama bildirimi
# - AIDecisionResult → AI karar bildirimi
# - Risk uyarıları (kill switch, daily loss, margin)
# - Sistem durumu (bot start, error, heartbeat)
# - Mevcut AnalysisReport formatı korundu
#
# Entegrasyonlar:
# - execution.bitget_executor → ExecutionResult, OrderResult
# - execution.risk_manager → TradeCalculation, RiskCheckStatus
# - ai.gemini_optimizer → AIDecisionResult, GateAction
# - config → TelegramConfig (token, chat_id)
#
# Kullanım:
# ---------
# from notifications.telegram_notifier import TelegramNotifier
# notifier = TelegramNotifier()
# notifier.send_trade_sync(execution_result)
# notifier.send_ai_decision_sync(ai_decision)
# notifier.send_risk_alert_sync("Kill Switch", "DD %15 aşıldı!")
# =============================================================================

import asyncio                                 # Async event loop
import logging                                 # Log yönetimi
import os                                      # Ortam değişkenleri
from datetime import datetime                  # Zaman damgası
from typing import Dict, List, Optional        # Tip belirteçleri
from dataclasses import dataclass, field       # Yapılandırılmış veri

# Telegram kütüphanesi
from telegram import Bot                       # Bot nesnesi
from telegram.constants import ParseMode       # HTML parse mode
from telegram.error import TelegramError, RetryAfter  # Hata tipleri

# Logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# =============================================================================
# ANALİZ RAPORU DATACLASS (MEVCUT — GERİYE UYUMLU)
# =============================================================================

@dataclass
class AnalysisReport:
    """
    IC analiz raporu — Telegram'a gönderilecek.
    
    Mevcut v2.0 ile aynı yapı, geriye uyumlu.
    """
    symbol: str                                # İşlem çifti (BTC/USDT:USDT)
    price: float                               # Güncel fiyat
    recommended_timeframe: str                 # En iyi TF
    market_regime: str                         # Piyasa rejimi
    direction: str                             # LONG / SHORT / NEUTRAL
    confidence_score: float                    # IC güven skoru (0-100)
    active_indicators: Dict[str, List[str]]    # Aktif indikatörler
    indicator_details: Dict[str, float] = field(default_factory=dict)
    category_tops: Dict[str, dict] = field(default_factory=dict)
    tf_rankings: List[dict] = field(default_factory=list)
    timestamp: datetime = None
    notes: str = ""
    change_24h: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# ANA BİLDİRİM SINIFI
# =============================================================================

class TelegramNotifier:
    """
    Telegram bildirim sistemi — v3.0.
    
    Mesaj tipleri:
    1. IC Analiz Raporu (mevcut)
    2. AI Karar Bildirimi (yeni)
    3. Trade Execution Bildirimi (yeni)
    4. Risk Uyarısı (yeni)
    5. Sistem Durumu (yeni)
    """
    
    MAX_MESSAGE_LENGTH = 4096                  # Telegram karakter limiti
    RATE_LIMIT_DELAY = 1.0                     # İstekler arası bekleme (saniye)
    MAX_RETRIES = 3                            # Max tekrar deneme
    
    # ---- İNDİKATÖR KISALTMALARI ----
    INDICATOR_SHORTCUTS = {
        'AROONU_25': 'Aroon↑', 'AROOND_25': 'Aroon↓', 'AROONOSC_25': 'AroonOsc',
        'SUPERTs_10_3.0': 'SuperT', 'SUPERTl_10_3.0': 'SuperT',
        'PSARs_0.02_0.2': 'PSAR', 'PSARl_0.02_0.2': 'PSAR',
        'ADX_14': 'ADX', 'DMP_14': 'DI+', 'DMN_14': 'DI-',
        'STOCHRSIk_14_14_3_3': 'StochRSI', 'STOCHRSId_14_14_3_3': 'StochRSI',
        'MACD_12_26_9': 'MACD', 'MACDh_12_26_9': 'MACDh',
        'RSI_14': 'RSI14', 'RSI_7': 'RSI7',
        'CCI_20_0.015': 'CCI', 'WILLR_14': 'WillR',
        'BBU_20_2.0': 'BB↑', 'BBL_20_2.0': 'BB↓', 'BBB_20_2.0': 'BB%B',
        'ATRr_14': 'ATR', 'NATR_14': 'NATR',
        'CMF_20': 'CMF', 'MFI_14': 'MFI', 'OBV': 'OBV',
    }
    
    # ---- KATEGORİ BİLGİLERİ ----
    CATEGORY_INFO = {
        'trend': ('📊', 'Trend'),
        'momentum': ('⚡', 'Momentum'),
        'volatility': ('📉', 'Volatilite'),
        'volume': ('📶', 'Hacim'),
    }
    
    # ---- REJİM MAPPING ----
    REGIME_MAP = {
        'trending_up': '📈 Trend↑',
        'trending_down': '📉 Trend↓',
        'ranging': '↔️ Yatay',
        'volatile': '⚡ Volatil',
        'transitioning': '🔄 Geçiş',
        'unknown': '❓',
    }
    
    def __init__(self, token: str = None, chat_id: str = None):
        """
        TelegramNotifier başlatır.
        
        Parameters:
        ----------
        token : str
            Bot token. None ise TELEGRAM_BOT_TOKEN env var'dan okunur.
        chat_id : str
            Chat ID. None ise TELEGRAM_CHAT_ID env var'dan okunur.
        """
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self._bot = None                       # Lazy initialization
    
    @property
    def bot(self) -> Bot:
        """Telegram Bot nesnesi — lazy init."""
        if self._bot is None and self.token:
            self._bot = Bot(token=self.token)
        return self._bot
    
    def is_configured(self) -> bool:
        """Token ve chat_id ayarlı mı?"""
        return bool(self.token and self.chat_id)
    
    # =========================================================================
    # MESAJ GÖNDERME (CORE)
    # =========================================================================
    
    async def send_message(self, text: str, disable_notification: bool = False) -> bool:
        """
        Telegram mesajı gönderir (async).
        Retry logic + rate limit handling dahil.
        """
        if not self.is_configured():
            logger.error("Telegram yapılandırılmamış! TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID gerekli.")
            return False
        
        # Mesaj uzunluk kontrolü — Telegram limiti 4096 karakter
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n<i>... (kısaltıldı)</i>"
        
        # EVENT LOOP HATASINI ÇÖZEN KISIM (async with Bot...)
        async with Bot(token=self.token) as bot:
            for attempt in range(self.MAX_RETRIES):
                try:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        disable_notification=disable_notification
                    )
                    logger.info("✅ Telegram mesajı gönderildi")
                    return True
                    
                except RetryAfter as e:
                    # Telegram rate limit — belirtilen süre kadar bekle
                    logger.warning(f"⏳ Rate limit, {e.retry_after}s bekleniyor...")
                    await asyncio.sleep(e.retry_after + 1)
                    
                except TelegramError as e:
                    logger.error(f"❌ Telegram hatası (deneme {attempt+1}/{self.MAX_RETRIES}): {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self.RATE_LIMIT_DELAY * (attempt + 1))
        
        return False
    
    # =========================================================================
    # 1. IC ANALİZ RAPORU FORMATLAMA (MEVCUT — GERİYE UYUMLU)
    # =========================================================================
    
    def _shorten_indicator(self, name: str) -> str:
        """İndikatör adını kısaltır."""
        if name in self.INDICATOR_SHORTCUTS:
            return self.INDICATOR_SHORTCUTS[name]
        # Prefix kuralları
        for prefix in ['EMA_', 'SMA_', 'RSI_', 'CCI_', 'ROC_', 'MOM_']:
            if name.startswith(prefix):
                parts = name.split('_')
                return f"{parts[0]}{parts[1]}" if len(parts) > 1 else parts[0]
        return name[:8]
    
    def format_analysis_report(self, report: AnalysisReport) -> str:
        """IC analiz raporunu Telegram formatına çevirir."""
        score = report.confidence_score
        
        # Güven barı
        filled = int(score / 10)
        conf_bar = '█' * filled + '░' * (10 - filled)
        
        # Yön emojisi
        dir_map = {'LONG': '🟢 LONG', 'SHORT': '🔴 SHORT', 'NEUTRAL': '⚪ NEUTRAL'}
        dir_text = dir_map.get(report.direction, f'❓ {report.direction}')
        
        # 24h değişim
        change_emoji = '📈' if report.change_24h >= 0 else '📉'
        
        # Kategori sinyalleri
        category_lines = ""
        for cat, info in report.category_tops.items():
            if cat in self.CATEGORY_INFO:
                emoji, name = self.CATEGORY_INFO[cat]
                ind_name = self._shorten_indicator(info.get('name', ''))
                ic_val = info.get('ic', 0)
                direction = info.get('direction', '')
                d_emoji = '↑' if direction == 'LONG' else '↓' if direction == 'SHORT' else '→'
                category_lines += f"\n{emoji} {name}: {ind_name} (IC:{ic_val:.2f}) {d_emoji}"
        
        # TF sıralaması (top 3)
        tf_lines = ""
        for r in report.tf_rankings[:3]:
            marker = "→ " if r.get('tf') == report.recommended_timeframe else "  "
            d_emoji = '↑' if r.get('direction') == 'LONG' else '↓' if r.get('direction') == 'SHORT' else '→'
            tf_lines += f"\n{marker}{r['tf']}: {r['score']:.0f} {d_emoji}"
        
        msg = f"""🔔 <b>{report.symbol} ANALİZ</b>
━━━━━━━━━━━━━━━━━━━━━

💰 Fiyat: ${report.price:,.2f} ({change_emoji}{report.change_24h:+.1f}%)

📊 TF: <b>{report.recommended_timeframe}</b> | {dir_text}
🎯 Güven: {score:.0f}/100 {conf_bar}
📍 Rejim: {self.REGIME_MAP.get(report.market_regime, '❓')}"""
        
        if category_lines:
            msg += f"\n\n⭐ <b>Kategori Sinyalleri:</b>{category_lines}"
        if tf_lines:
            msg += f"\n\n📋 <b>TF Sıralaması:</b>{tf_lines}"
        if report.notes:
            msg += f"\n\n📝 {report.notes}"
        
        msg += f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⏰ {report.timestamp.strftime('%Y-%m-%d %H:%M')}"
        return msg.strip()
    
    # =========================================================================
    # 2. AI KARAR BİLDİRİMİ (YENİ — v3.0)
    # =========================================================================
    
    def format_ai_decision(self, decision_result) -> str:
        """
        AIDecisionResult → Telegram mesajı.
        
        Parameters:
        ----------
        decision_result : AIDecisionResult
            ai.gemini_optimizer'dan gelen karar.
        """
        d = decision_result
        
        # Karar emojisi
        decision_map = {
            'LONG': '🟢 LONG', 'SHORT': '🔴 SHORT', 'WAIT': '⏸️ WAIT'
        }
        decision_text = decision_map.get(d.decision, f'❓ {d.decision}')
        
        # Gate action emojisi
        gate_map = {
            'NO_TRADE': '🚫 NO TRADE', 
            'REPORT_ONLY': '📋 RAPOR',
            'FULL_TRADE': '✅ TRADE'
        }
        gate_text = gate_map.get(d.gate_action, f'❓ {d.gate_action}')
        
        # Güven barı
        conf = d.confidence
        filled = int(conf / 10)
        conf_bar = '█' * filled + '░' * (10 - filled)
        
        # Execute durumu
        exec_emoji = "🚀" if d.should_execute() else "⛔"
        exec_text = "İŞLEM YAP" if d.should_execute() else "İŞLEM YAPMA"
        
        msg = f"""🤖 <b>AI KARAR</b>
━━━━━━━━━━━━━━━━━━━━━

{decision_text} | Güven: {conf:.0f}% {conf_bar}
🚦 Gate: {gate_text}
{exec_emoji} <b>{exec_text}</b>

📊 IC Skoru: {d.ic_score:.0f}/100
🔄 ATR Çarpanı: {d.atr_multiplier:.1f}x"""
        
        if d.reasoning:
            # Gerekçeyi kısalt (max 200 karakter)
            reason = d.reasoning[:200] + '...' if len(d.reasoning) > 200 else d.reasoning
            msg += f"\n\n💡 <b>Gerekçe:</b>\n{reason}"
        
        msg += f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return msg.strip()
    
    # =========================================================================
    # 3. TRADE EXECUTION BİLDİRİMİ (YENİ — v3.0)
    # =========================================================================
    
    def format_trade_execution(self, exec_result) -> str:
        """
        ExecutionResult → Telegram mesajı.
        
        Parameters:
        ----------
        exec_result : ExecutionResult
            execution.bitget_executor'dan gelen sonuç.
        """
        e = exec_result
        
        mode = "🧪 DRY RUN" if e.dry_run else "🔴 CANLI"
        status = "✅ BAŞARILI" if e.success else "❌ BAŞARISIZ"
        dir_emoji = "🟢" if e.direction == "LONG" else "🔴"
        
        msg = f"""{status} <b>TRADE {mode}</b>
━━━━━━━━━━━━━━━━━━━━━

{dir_emoji} <b>{e.symbol} {e.direction}</b>"""
        
        if e.main_order and e.main_order.success:
            msg += f"\n\n📍 <b>Entry:</b> ${e.actual_entry:,.2f}"
            msg += f"\n📦 <b>Miktar:</b> {e.actual_amount:.4f}"
            
            if e.actual_cost > 0:
                msg += f"\n💰 <b>Maliyet:</b> ${e.actual_cost:,.2f}"
        
        if e.sl_order and e.sl_order.success:
            msg += f"\n🛑 <b>SL:</b> ${e.sl_order.price:,.2f}"
        
        if e.tp_order and e.tp_order.success:
            msg += f"\n🎯 <b>TP:</b> ${e.tp_order.price:,.2f}"
        
        # SL-TP arası RR hesapla (varsa)
        if (e.sl_order and e.tp_order and e.sl_order.success 
            and e.tp_order.success and e.actual_entry > 0):
            sl_dist = abs(e.actual_entry - e.sl_order.price)
            tp_dist = abs(e.tp_order.price - e.actual_entry)
            if sl_dist > 0:
                rr = tp_dist / sl_dist
                msg += f"\n📐 <b>R:R:</b> 1:{rr:.1f}"
        
        if e.error:
            msg += f"\n\n❌ <b>Hata:</b> {e.error[:150]}"
        
        msg += f"\n\n━━━━━━━━━━━━━━━━━━━━━\n⏰ {e.timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        return msg.strip()
    
    # =========================================================================
    # 4. RİSK UYARISI (YENİ — v3.0)
    # =========================================================================
    
    def format_risk_alert(self, alert_type: str, details: str, 
                          severity: str = "warning") -> str:
        """
        Risk uyarısı mesajı formatlar.
        
        Parameters:
        ----------
        alert_type : str
            Uyarı tipi: 'kill_switch', 'daily_loss', 'margin_limit', 
            'position_limit', 'trade_rejected'
        details : str
            Detay açıklama
        severity : str
            'info', 'warning', 'critical'
        """
        # Severity emojisi
        sev_map = {
            'info': 'ℹ️',
            'warning': '⚠️', 
            'critical': '🚨'
        }
        sev_emoji = sev_map.get(severity, '⚠️')
        
        # Alert tipi başlığı
        type_map = {
            'kill_switch': '🔴 KİLL SWİTCH AKTİF',
            'daily_loss': '⚠️ GÜNLÜK KAYIP LİMİTİ',
            'margin_limit': '⚠️ MARJİN LİMİTİ',
            'position_limit': '⚠️ POZİSYON LİMİTİ',
            'trade_rejected': '🚫 TRADE REDDEDİLDİ',
            'connection_error': '🔌 BAĞLANTI HATASI',
            'api_error': '🔑 API HATASI',
        }
        title = type_map.get(alert_type, f'{sev_emoji} UYARI')
        
        msg = f"""{sev_emoji} <b>{title}</b>
━━━━━━━━━━━━━━━━━━━━━

{details}

━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        return msg.strip()
    
    # =========================================================================
    # 5. SİSTEM DURUMU (YENİ — v3.0)
    # =========================================================================
    
    def format_system_status(self, status_type: str, details: Dict = None) -> str:
        """
        Sistem durum mesajı formatlar.
        
        Parameters:
        ----------
        status_type : str
            'startup', 'shutdown', 'heartbeat', 'error', 'scan_complete'
        details : Dict
            Ek bilgiler
        """
        details = details or {}
        
        if status_type == 'startup':
            balance = details.get('balance', 0)
            mode = details.get('mode', 'DRY RUN')
            msg = f"""🚀 <b>BOT BAŞLATILDI</b>
━━━━━━━━━━━━━━━━━━━━━

💰 Bakiye: ${balance:,.2f}
🔧 Mod: {mode}
📊 Tarama: Aktif
🤖 AI: Gemini aktif

━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        elif status_type == 'shutdown':
            reason = details.get('reason', 'Manuel')
            msg = f"""🔴 <b>BOT DURDURULDU</b>
━━━━━━━━━━━━━━━━━━━━━

📝 Sebep: {reason}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        elif status_type == 'heartbeat':
            balance = details.get('balance', 0)
            positions = details.get('positions', 0)
            daily_pnl = details.get('daily_pnl', 0)
            uptime = details.get('uptime', 'N/A')
            pnl_emoji = '📈' if daily_pnl >= 0 else '📉'
            
            msg = f"""💓 <b>DURUM RAPORU</b>
━━━━━━━━━━━━━━━━━━━━━

💰 Bakiye: ${balance:,.2f}
📊 Açık pozisyon: {positions}
{pnl_emoji} Günlük PnL: ${daily_pnl:+,.2f}
⏱️ Uptime: {uptime}

━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        elif status_type == 'scan_complete':
            scanned = details.get('scanned', 0)
            passed = details.get('passed', 0)
            best = details.get('best_coin', 'N/A')
            best_score = details.get('best_score', 0)
            
            msg = f"""🔍 <b>TARAMA TAMAMLANDI</b>
━━━━━━━━━━━━━━━━━━━━━

📊 Taranan: {scanned} çift
✅ Geçen: {passed} çift
🏆 En iyi: {best} ({best_score:.0f} puan)

━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        elif status_type == 'error':
            error = details.get('error', 'Bilinmeyen hata')
            component = details.get('component', 'Sistem')
            msg = f"""❌ <b>HATA — {component}</b>
━━━━━━━━━━━━━━━━━━━━━

{error[:500]}

━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        
        else:
            msg = f"ℹ️ {status_type}: {details}"
        
        return msg.strip()
    
    # =========================================================================
    # ASYNC GÖNDERME FONKSİYONLARI
    # =========================================================================
    
    async def send_analysis_report(self, report: AnalysisReport, 
                                    silent: bool = False) -> bool:
        """IC analiz raporu gönder."""
        message = self.format_analysis_report(report)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_ai_decision(self, decision_result, 
                                silent: bool = False) -> bool:
        """AI karar bildirimi gönder."""
        message = self.format_ai_decision(decision_result)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_trade_execution(self, exec_result, 
                                    silent: bool = False) -> bool:
        """Trade execution bildirimi gönder."""
        message = self.format_trade_execution(exec_result)
        return await self.send_message(message, disable_notification=silent)
    
    async def send_risk_alert(self, alert_type: str, details: str,
                               severity: str = "warning") -> bool:
        """Risk uyarısı gönder — her zaman sesli."""
        message = self.format_risk_alert(alert_type, details, severity)
        return await self.send_message(message, disable_notification=False)
    
    async def send_system_status(self, status_type: str, 
                                  details: Dict = None) -> bool:
        """Sistem durumu gönder."""
        message = self.format_system_status(status_type, details)
        silent = status_type == 'heartbeat'    # Heartbeat sessiz
        return await self.send_message(message, disable_notification=silent)
    
    # =========================================================================
    # SYNC WRAPPER'LAR (main.py gibi sync kodlardan çağrılabilir)
    # =========================================================================
    
    def send_message_sync(self, text: str, disable_notification: bool = False) -> bool:
        """Sync mesaj gönder."""
        return asyncio.run(self.send_message(text, disable_notification))
    
    def send_report_sync(self, report: AnalysisReport, silent: bool = False) -> bool:
        """Sync IC analiz raporu gönder."""
        return asyncio.run(self.send_analysis_report(report, silent))
    
    def send_ai_decision_sync(self, decision_result, silent: bool = False) -> bool:
        """Sync AI karar bildirimi gönder."""
        return asyncio.run(self.send_ai_decision(decision_result, silent))
    
    def send_trade_sync(self, exec_result, silent: bool = False) -> bool:
        """Sync trade execution bildirimi gönder."""
        return asyncio.run(self.send_trade_execution(exec_result, silent))
    
    def send_risk_alert_sync(self, alert_type: str, details: str,
                              severity: str = "warning") -> bool:
        """Sync risk uyarısı gönder."""
        return asyncio.run(self.send_risk_alert(alert_type, details, severity))
    
    def send_system_status_sync(self, status_type: str, details: Dict = None) -> bool:
        """Sync sistem durumu gönder."""
        return asyncio.run(self.send_system_status(status_type, details))
    
    # =========================================================================
    # BAĞLANTI TESTİ
    # =========================================================================
    
    async def test_connection(self) -> bool:
        """Bot bağlantısını test eder."""
        if not self.is_configured():
            logger.error("Telegram yapılandırılmamış!")
            return False
        try:
            # EVENT LOOP HATASINI ÇÖZEN KISIM (async with Bot...)
            async with Bot(token=self.token) as bot:
                me = await bot.get_me()
                logger.info(f"✅ Bot bağlantısı: @{me.username}")
                return True
        except TelegramError as e:
            logger.error(f"❌ Bağlantı hatası: {e}")
            return False


# =============================================================================
# FACTORY FONKSİYONU
# =============================================================================

def create_notifier_from_env() -> TelegramNotifier:
    """Ortam değişkenlerinden TelegramNotifier oluşturur."""
    return TelegramNotifier()
