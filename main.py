# =============================================================================
# MAIN.PY PAPER TRADER ENTEGRASYONU (ADIM 10 GÜNCELLEME)
# =============================================================================
# Bu dosya mevcut main.py'ye eklenecek değişiklikleri içerir.
#
# Yapılacaklar:
# 1. PaperTrader import et
# 2. Pipeline'a paper_trader ekle
# 3. Her trade'de paper_trader.open_trade() çağır
# 4. Her cycle başında açık pozisyonları kontrol et
# 5. AI quota hatası → IC-only mode
# 6. Günlük rapor fonksiyonu
#
# KURULUM:
# Bu dosyadaki kodları main.py'ye entegre et veya
# bu dosyayı direkt çalıştır (standalone mode).
# =============================================================================

import asyncio
import sys
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# ---- Mevcut Modüller (Proje Yapısına Göre) ----
from config import cfg

# Scanner modülü (scanner/ klasöründen)
from scanner import CoinScanner

# Data modülü (data/ klasöründen)
from data import BitgetFetcher, DataPreprocessor

# Indicators modülü (indicators/ klasöründen)
from indicators import IndicatorCalculator, IndicatorSelector

# AI modülü (ai/ klasöründen)
from ai import GeminiOptimizer, AIDecision, GateAction

# Execution modülü (execution/ klasöründen)
from execution import RiskManager, BitgetExecutor

# Notifications modülü (notifications/ klasöründen)
from notifications import TelegramNotifier

# AIDecisionType için helper enum
class AIDecisionType(Enum):
    """AI karar türleri - IC direction'dan dönüşüm için."""
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    
    @classmethod
    def from_direction(cls, direction: str) -> 'AIDecisionType':
        """IC direction'dan AIDecisionType'a çevir."""
        d = (direction or "").upper()
        if d in ("LONG", "BUY", "BULLISH"):
            return cls.LONG
        elif d in ("SHORT", "SELL", "BEARISH"):
            return cls.SHORT
        return cls.WAIT

# ---- YENİ: Paper Trading Modülleri ----
from paper_trader import PaperTrader, TradeStatus
from performance_analyzer import PerformanceAnalyzer

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# VERSİYON VE SABİTLER
# =============================================================================

VERSION = "1.1.0"                              # Paper trading eklendi
MAX_COINS_PER_CYCLE = 20                       # Maksimum analiz edilecek coin
DEFAULT_FWD_PERIOD = 6                         # IC forward period (bar)

# Varsayılan timeframe'ler
DEFAULT_TIMEFRAMES = {
    '15m': 200,                                # Scalping
    '1h': 250,                                 # Kısa vade
    '4h': 150,                                 # Orta vade
    '1d': 100,                                 # Uzun vade
}

# AI Quota tracking
AI_QUOTA_EXHAUSTED = False                     # Global flag
AI_ERRORS_TODAY = 0                            # Bugünkü hata sayısı
AI_ERROR_THRESHOLD = 3                         # Bu kadar hatadan sonra AI devre dışı


# =============================================================================
# ENUM'LAR VE DATACLASS'LAR
# =============================================================================

class CycleStatus(Enum):
    """Döngü durumu."""
    SUCCESS = "success"                        # En az 1 işlem yapıldı
    PARTIAL = "partial"                        # Bazı işlemler başarısız
    NO_SIGNAL = "no_signal"                    # Sinyal yok
    ERROR = "error"                            # Kritik hata
    KILLED = "killed"                          # Kill switch aktif


# GateAction ai modülünden import ediliyor (satır 43)


@dataclass
class CoinAnalysisResult:
    """Tek bir coin'in analiz sonucu."""
    coin: str = ""
    full_symbol: str = ""
    price: float = 0.0
    change_24h: float = 0.0
    volume_24h: float = 0.0
    
    # IC Analiz
    best_timeframe: str = ""
    ic_confidence: float = 0.0
    ic_direction: str = ""
    significant_count: int = 0
    market_regime: str = ""
    
    # Risk
    atr: float = 0.0
    atr_pct: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    position_size: float = 0.0
    leverage: int = 1
    risk_reward: float = 0.0
    
    # Kararlar
    gate_action: GateAction = GateAction.NO_TRADE
    ai_decision: Optional[AIDecision] = None
    ai_skipped: bool = False                   # AI quota nedeniyle atlandı mı?
    
    # Sonuç
    status: str = "pending"
    error: str = ""
    execution_result: Any = None
    paper_trade_id: str = ""                   # Paper trade ID


@dataclass
class CycleReport:
    """Döngü özet raporu."""
    timestamp: str = ""
    status: CycleStatus = CycleStatus.NO_SIGNAL
    total_scanned: int = 0
    total_analyzed: int = 0
    total_above_gate: int = 0
    total_traded: int = 0
    coins: List[CoinAnalysisResult] = field(default_factory=list)
    balance: float = 0.0
    paper_balance: float = 0.0                 # Paper trade bakiyesi
    errors: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    ai_mode: str = "normal"                    # "normal" veya "ic_only"


# =============================================================================
# ANA PIPELINE SINIFI (PAPER TRADE ENTEGRELİ)
# =============================================================================

class HybridTradingPipeline:
    """
    Paper Trading entegreli Hybrid Trading Pipeline.
    
    Yenilikler:
    - PaperTrader ile trade kayıtları
    - AI quota yönetimi (free tier için)
    - Otomatik SL/TP takibi
    - Performans raporlama
    """

    def __init__(
        self,
        dry_run: bool = True,
        top_n: int = MAX_COINS_PER_CYCLE,
        timeframes: Dict[str, int] = None,
        fwd_period: int = DEFAULT_FWD_PERIOD,
        verbose: bool = True,
    ):
        self.dry_run = dry_run
        self.top_n = min(top_n, MAX_COINS_PER_CYCLE)
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.fwd_period = fwd_period
        self.verbose = verbose
        
        # ---- Modüller ----
        self.scanner = CoinScanner()
        self.fetcher = BitgetFetcher()
        self.preprocessor = DataPreprocessor()
        self.calculator = IndicatorCalculator()
        self.selector = IndicatorSelector(alpha=0.05)
        self.ai_optimizer = GeminiOptimizer()
        self.executor = BitgetExecutor(dry_run=dry_run)
        self.notifier = TelegramNotifier()
        
        # ---- YENİ: Paper Trader ----
        self.paper_trader = PaperTrader(
            initial_balance=75.0,              # Başlangıç bakiyesi
            log_dir=Path(__file__).parent.parent / "logs" / "paper_trades",
            auto_save=True,
        )
        
        # ---- Durum Değişkenleri ----
        self._balance: float = 0.0
        self._initial_balance: float = 0.0
        self._risk_manager: Optional[RiskManager] = None
        self._is_running: bool = False
        self._kill_switch: bool = False
        self._cycle_count: int = 0
        
        # ---- AI Quota Tracking ----
        self._ai_available: bool = True
        self._ai_errors: int = 0
        
        logger.info(
            f"🚀 HybridTradingPipeline v{VERSION} başlatıldı | "
            f"Mode: {'🧪 DRY RUN' if dry_run else '🔴 CANLI'} | "
            f"Paper Trading: ✅"
        )

    # =========================================================================
    # BAKİYE YÖNETİMİ
    # =========================================================================

    def _init_balance(self) -> bool:
        """Bakiyeyi başlat."""
        try:
            if self.dry_run:
                self._balance = self.paper_trader.balance
                self._initial_balance = self.paper_trader.initial_balance
                logger.info(f"💰 Paper bakiye: ${self._balance:.2f}")
            else:
                balance_info = self.executor.fetch_balance()
                self._balance = balance_info.get('free', 0.0)
                self._initial_balance = balance_info.get('total', self._balance)
                logger.info(f"💰 Canlı bakiye: ${self._balance:.2f}")
            
            self._risk_manager = RiskManager(
                balance=self._balance,
                initial_balance=self._initial_balance
            )
            return self._balance > 0
            
        except Exception as e:
            logger.error(f"❌ Bakiye hatası: {e}")
            return False

    def _refresh_balance(self) -> None:
        """Bakiyeyi güncelle."""
        if self.dry_run:
            self._balance = self.paper_trader.balance
        else:
            try:
                balance_info = self.executor.fetch_balance()
                self._balance = balance_info.get('free', 0.0)
            except Exception as e:
                logger.warning(f"⚠️ Bakiye güncelleme hatası: {e}")
        
        if self._risk_manager:
            self._risk_manager.update_balance(self._balance)

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def _check_kill_switch(self) -> bool:
        """Drawdown bazlı kill switch kontrolü."""
        if self._initial_balance <= 0:
            return False
        
        # Paper trader'dan drawdown al
        if self.dry_run:
            drawdown_pct = self.paper_trader.max_drawdown
        else:
            drawdown_pct = (self._initial_balance - self._balance) / self._initial_balance * 100
        
        threshold = cfg.risk.kill_switch_pct if hasattr(cfg.risk, 'kill_switch_pct') else 15.0
        
        if drawdown_pct >= threshold:
            self._kill_switch = True
            logger.warning(
                f"🛑 KILL SWITCH AKTİF! "
                f"Drawdown: {drawdown_pct:.1f}% >= Threshold: {threshold:.1f}%"
            )
            
            # Tüm açık pozisyonları kapat
            if self.dry_run and self.paper_trader.open_trades:
                prices = self._get_current_prices()
                self.paper_trader.close_all_trades(prices, "Kill switch triggered")
            
            return True
        
        return False

    def _get_current_prices(self) -> Dict[str, float]:
        """Açık pozisyonlar için güncel fiyatları al."""
        prices = {}
        for trade_id, trade in self.paper_trader.open_trades.items():
            try:
                ticker = self.fetcher.exchange.fetch_ticker(trade.full_symbol)
                prices[trade.symbol] = ticker['last']
            except:
                prices[trade.symbol] = trade.entry_price
        return prices

    # =========================================================================
    # MARKET TARAMASI
    # =========================================================================

    def _scan_market(self) -> List:
        """Market taraması yap."""
        try:
            logger.info("🔍 Market taraması başlıyor...")
            top_coins = self.scanner.scan(top_n=self.top_n)
            logger.info(f"✅ {len(top_coins)} coin bulundu")
            return top_coins
        except Exception as e:
            logger.error(f"❌ Tarama hatası: {e}")
            return []

    # =========================================================================
    # IC ANALİZ
    # =========================================================================

    def _analyze_coin(self, symbol: str) -> Optional[CoinAnalysisResult]:
        """Tek bir coin için IC analizi yap."""
        result = CoinAnalysisResult(coin=symbol)
        
        try:
            # Sembol formatı
            full_symbol = f"{symbol}/USDT:USDT"
            result.full_symbol = full_symbol
            
            # Ticker bilgisi
            ticker = self.fetcher.exchange.fetch_ticker(full_symbol)
            result.price = ticker.get('last', 0)
            result.change_24h = ticker.get('percentage', 0) or 0
            result.volume_24h = ticker.get('quoteVolume', 0) or 0
            
            # Her timeframe için analiz
            tf_results = []
            
            for tf, limit in self.timeframes.items():
                try:
                    # OHLCV çek
                    df = self.fetcher.fetch_ohlcv(full_symbol, timeframe=tf, limit=limit)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Preprocess
                    df = self.preprocessor.prepare(df)
                    
                    # İndikatörler hesapla
                    df = self.calculator.add_all_indicators(df)
                    
                    # IC analiz
                    ic_result = self.selector.analyze(df, forward_period=self.fwd_period)
                    
                    if ic_result and ic_result.get('composite_score', 0) > 0:
                        tf_results.append({
                            'tf': tf,
                            'score': ic_result['composite_score'],
                            'direction': ic_result.get('direction', 'NEUTRAL'),
                            'regime': ic_result.get('regime', 'unknown'),
                            'significant': len(ic_result.get('significant_indicators', [])),
                            'atr': ic_result.get('atr', 0),
                            'atr_pct': ic_result.get('atr_pct', 0),
                        })
                        
                except Exception as e:
                    logger.debug(f"TF {tf} hatası: {e}")
                    continue
            
            if not tf_results:
                result.status = "no_data"
                return result
            
            # En iyi TF'yi seç
            best = max(tf_results, key=lambda x: x['score'])
            
            result.best_timeframe = best['tf']
            result.ic_confidence = best['score']
            result.ic_direction = best['direction']
            result.market_regime = best['regime']
            result.significant_count = best['significant']
            result.atr = best['atr']
            result.atr_pct = best['atr_pct']
            
            # Gate keeper kararı
            no_trade_threshold = cfg.gate.no_trade if hasattr(cfg, 'gate') else 55
            full_trade_threshold = cfg.gate.full_trade if hasattr(cfg, 'gate') else 70
            
            if result.ic_confidence < no_trade_threshold:
                result.gate_action = GateAction.NO_TRADE
            elif result.ic_confidence < full_trade_threshold:
                result.gate_action = GateAction.REPORT_ONLY
            else:
                result.gate_action = GateAction.FULL_TRADE
            
            result.status = "analyzed"
            return result
            
        except Exception as e:
            result.status = "error"
            result.error = str(e)
            return result

    # =========================================================================
    # AI OPTİMİZASYON (QUOTA YÖNETİMLİ)
    # =========================================================================

    def _get_ai_decision(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        """
        AI kararı al (quota yönetimli).
        
        Free tier için:
        - 5 request/dakika
        - ~20 request/gün
        
        Quota biterse IC-only mode'a geç.
        """
        global AI_QUOTA_EXHAUSTED, AI_ERRORS_TODAY
        
        # AI devre dışı mı?
        if AI_QUOTA_EXHAUSTED or not self._ai_available:
            result.ai_skipped = True
            result.ai_decision = AIDecision(
                decision=AIDecisionType.from_direction(result.ic_direction),
                confidence=result.ic_confidence * 0.8,  # IC'den %20 düşük güven
                reasoning="AI quota aşıldı - IC skoru ile karar verildi",
            )
            logger.info(f"⚡ {result.coin}: AI atlandı (IC-only mode)")
            return result
        
        try:
            # Rate limiting: 12 saniye bekle (5 req/dk = 12s/req)
            time.sleep(12)
            
            # AI'ya gönder
            ai_decision = self.ai_optimizer.optimize(
                symbol=result.coin,
                ic_score=result.ic_confidence,
                ic_direction=result.ic_direction,
                regime=result.market_regime,
                timeframe=result.best_timeframe,
                price=result.price,
                atr=result.atr,
            )
            
            result.ai_decision = ai_decision
            result.ai_skipped = False
            
            logger.info(
                f"🤖 {result.coin}: AI → {ai_decision.decision.value} "
                f"(Güven: {ai_decision.confidence:.0f})"
            )
            
            # Başarılı istek - hata sayacını sıfırla
            AI_ERRORS_TODAY = 0
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Quota hatası mı?
            if 'quota' in error_msg or '429' in error_msg or 'rate' in error_msg:
                AI_ERRORS_TODAY += 1
                logger.warning(f"⚠️ AI quota hatası ({AI_ERRORS_TODAY}/{AI_ERROR_THRESHOLD}): {e}")
                
                if AI_ERRORS_TODAY >= AI_ERROR_THRESHOLD:
                    AI_QUOTA_EXHAUSTED = True
                    logger.warning("🚫 AI quota tükendi! IC-only mode aktif.")
            
            # IC bazlı fallback
            result.ai_skipped = True
            result.ai_decision = AIDecision(
                decision=AIDecisionType.from_direction(result.ic_direction),
                confidence=result.ic_confidence * 0.8,
                reasoning=f"AI hatası - IC fallback: {str(e)[:50]}",
            )
        
        return result

    # =========================================================================
    # RİSK HESAPLAMA
    # =========================================================================

    def _calculate_risk(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        """Risk parametrelerini hesapla."""
        if not self._risk_manager:
            return result
        
        try:
            direction = result.ai_decision.decision.value if result.ai_decision else result.ic_direction
            
            if direction == "WAIT":
                return result
            
            risk_params = self._risk_manager.calculate(
                entry_price=result.price,
                atr=result.atr,
                direction=direction,
                confidence=result.ic_confidence,
            )
            
            result.sl_price = risk_params['stop_loss']
            result.tp_price = risk_params['take_profit']
            result.position_size = risk_params['position_size']
            result.leverage = risk_params['leverage']
            result.risk_reward = risk_params['risk_reward']
            
        except Exception as e:
            logger.error(f"❌ Risk hesaplama hatası: {e}")
            result.error = str(e)
        
        return result

    # =========================================================================
    # PAPER TRADE AÇMA
    # =========================================================================

    def _execute_paper_trade(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        """Paper trade aç."""
        try:
            direction = result.ai_decision.decision.value if result.ai_decision else result.ic_direction
            
            if direction == "WAIT":
                result.status = "skipped_wait"
                return result
            
            # Paper trade aç
            trade = self.paper_trader.open_trade(
                symbol=result.coin,
                full_symbol=result.full_symbol,
                direction=direction,
                entry_price=result.price,
                position_size=result.position_size,
                stop_loss=result.sl_price,
                take_profit=result.tp_price,
                leverage=result.leverage,
                ic_confidence=result.ic_confidence,
                ic_direction=result.ic_direction,
                best_timeframe=result.best_timeframe,
                market_regime=result.market_regime,
                ai_decision=result.ai_decision.decision.value if result.ai_decision else None,
                ai_confidence=result.ai_decision.confidence if result.ai_decision else None,
            )
            
            result.paper_trade_id = trade.trade_id
            result.status = "executed"
            
            logger.info(
                f"📝 Paper Trade: {trade.trade_id} | "
                f"{result.coin} {direction} @ ${result.price:,.2f} | "
                f"SL: ${result.sl_price:,.2f} | TP: ${result.tp_price:,.2f}"
            )
            
        except Exception as e:
            result.status = "execution_error"
            result.error = str(e)
            logger.error(f"❌ Paper trade hatası: {e}")
        
        return result

    # =========================================================================
    # AÇIK POZİSYON KONTROLÜ
    # =========================================================================

    def _check_open_positions(self) -> List:
        """Açık pozisyonların SL/TP kontrolü."""
        if not self.paper_trader.open_trades:
            return []
        
        # Güncel fiyatları al
        prices = self._get_current_prices()
        
        # SL/TP kontrol et
        closed = self.paper_trader.check_exits(prices)
        
        for trade in closed:
            emoji = "✅" if trade.net_pnl > 0 else "❌"
            logger.info(
                f"{emoji} Trade kapandı: {trade.trade_id} | "
                f"{trade.symbol} {trade.direction} | "
                f"PnL: ${trade.net_pnl:+.2f} ({trade.pnl_percent:+.1f}%)"
            )
        
        return closed

    # =========================================================================
    # ANA DÖNGÜ
    # =========================================================================

    def run_cycle(self) -> CycleReport:
        """Tek bir pipeline döngüsü çalıştır."""
        self._cycle_count += 1
        cycle_start = time.time()
        
        report = CycleReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ai_mode="ic_only" if AI_QUOTA_EXHAUSTED else "normal",
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 CYCLE #{self._cycle_count} | {report.timestamp}")
        logger.info(f"🔧 Mode: {'🧪 PAPER' if self.dry_run else '🔴 CANLI'}")
        logger.info(f"🤖 AI: {'⚡ IC-ONLY' if AI_QUOTA_EXHAUSTED else '✅ AKTIF'}")
        logger.info(f"{'='*60}")
        
        try:
            # 0. Bakiye + Kill Switch
            self._refresh_balance()
            report.balance = self._balance
            report.paper_balance = self.paper_trader.balance
            
            if self._check_kill_switch():
                report.status = CycleStatus.KILLED
                report.elapsed = time.time() - cycle_start
                return report
            
            # 1. Açık pozisyonları kontrol et
            closed_trades = self._check_open_positions()
            if closed_trades:
                logger.info(f"📊 {len(closed_trades)} pozisyon kapandı")
            
            # 2. Market taraması
            top_coins = self._scan_market()
            report.total_scanned = len(top_coins)
            
            if not top_coins:
                report.status = CycleStatus.NO_SIGNAL
                report.elapsed = time.time() - cycle_start
                return report
            
            # 3. Her coin için analiz
            for coin_data in top_coins:
                symbol = coin_data.symbol if hasattr(coin_data, 'symbol') else coin_data.get('symbol', '')
                
                # IC Analiz
                result = self._analyze_coin(symbol)
                if not result or result.status == "error":
                    continue
                
                report.total_analyzed += 1
                
                # Gate kontrolü
                if result.gate_action == GateAction.NO_TRADE:
                    result.status = "below_gate"
                    report.coins.append(result)
                    continue
                
                report.total_above_gate += 1
                
                # AI kararı (FULL_TRADE için)
                if result.gate_action == GateAction.FULL_TRADE:
                    result = self._get_ai_decision(result)
                    
                    # WAIT kararı?
                    if result.ai_decision and result.ai_decision.decision == AIDecisionType.WAIT:
                        result.status = "ai_wait"
                        report.coins.append(result)
                        continue
                    
                    # Risk hesapla
                    result = self._calculate_risk(result)
                    
                    # Trade aç
                    if result.sl_price > 0:
                        result = self._execute_paper_trade(result)
                        if result.status == "executed":
                            report.total_traded += 1
                
                report.coins.append(result)
            
            # Sonuç durumu
            if report.total_traded > 0:
                report.status = CycleStatus.SUCCESS
            elif report.total_above_gate > 0:
                report.status = CycleStatus.PARTIAL
            else:
                report.status = CycleStatus.NO_SIGNAL
            
        except Exception as e:
            report.status = CycleStatus.ERROR
            report.errors.append(str(e))
            logger.error(f"❌ Cycle hatası: {e}")
        
        report.elapsed = time.time() - cycle_start
        
        # Özet
        self._print_cycle_summary(report)
        
        return report

    def _print_cycle_summary(self, report: CycleReport) -> None:
        """Döngü özetini yazdır."""
        print(f"\n{'─'*50}")
        print(f"📊 CYCLE #{self._cycle_count} ÖZET")
        print(f"{'─'*50}")
        print(f"  Status: {report.status.value}")
        print(f"  Taranan: {report.total_scanned} | Analiz: {report.total_analyzed}")
        print(f"  Gate+: {report.total_above_gate} | Trade: {report.total_traded}")
        print(f"  Paper Bakiye: ${report.paper_balance:.2f}")
        print(f"  Açık Pozisyon: {len(self.paper_trader.open_trades)}")
        print(f"  Süre: {report.elapsed:.1f}s")
        print(f"{'─'*50}\n")

    # =========================================================================
    # PERFORMANS RAPORU
    # =========================================================================

    def print_performance(self) -> None:
        """Performans raporunu yazdır."""
        analyzer = PerformanceAnalyzer(self.paper_trader)
        report = analyzer.full_analysis()
        analyzer.print_report(report)

    def get_summary(self) -> Dict:
        """Kısa özet döndür."""
        return self.paper_trader.get_summary()


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(pipeline: HybridTradingPipeline, interval_minutes: int = 60):
    """Pipeline'ı periyodik çalıştır."""
    pipeline._is_running = True
    
    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Durdurma sinyali alındı...")
        pipeline._is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"⏰ Scheduler başladı | Interval: {interval_minutes} dakika")
    logger.info("   Durdurmak için Ctrl+C\n")
    
    # İlk bakiye başlatma
    if not pipeline._init_balance():
        logger.error("❌ Bakiye başlatılamadı")
        return
    
    cycle_count = 0
    
    while pipeline._is_running:
        cycle_count += 1
        
        # Döngü çalıştır
        report = pipeline.run_cycle()
        
        # Kill switch kontrolü
        if report.status == CycleStatus.KILLED:
            logger.warning("🛑 Kill switch - scheduler durduruluyor")
            break
        
        if not pipeline._is_running:
            break
        
        # Sonraki döngüye kadar bekle
        logger.info(f"⏳ Sonraki döngü: {interval_minutes} dakika sonra...")
        
        # 1 dakikalık parçalar halinde bekle (graceful shutdown için)
        for _ in range(interval_minutes):
            if not pipeline._is_running:
                break
            time.sleep(60)
    
    # Bitiş özeti
    logger.info("\n" + "="*50)
    logger.info("📊 SCHEDULER BİTİŞ RAPORU")
    logger.info("="*50)
    pipeline.print_performance()


# =============================================================================
# CLI PARSER
# =============================================================================

def parse_args():
    """Komut satırı argümanlarını parse et."""
    parser = argparse.ArgumentParser(
        description="Hybrid Crypto Trading Bot v" + VERSION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        default=True,
        help='Paper trade modu (varsayılan: True)'
    )
    
    parser.add_argument(
        '--live', '-L',
        action='store_true',
        help='Canlı işlem modu (DİKKAT!)'
    )
    
    parser.add_argument(
        '--schedule', '-s',
        action='store_true',
        help='Sürekli çalışma modu'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Çalışma aralığı (dakika, varsayılan: 60)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Tek coin analizi (örn: BTC)'
    )
    
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=10,
        help='Analiz edilecek coin sayısı (varsayılan: 10)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Sessiz mod (az çıktı)'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Performans raporu göster'
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ana giriş noktası."""
    args = parse_args()
    
    dry_run = not args.live
    
    # Banner
    print(f"\n{'='*60}")
    print(f"  🚀 HYBRID CRYPTO BOT v{VERSION}")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🔧 Mode: {'🧪 PAPER TRADE' if dry_run else '🔴 CANLI'}")
    print(f"  🤖 AI: {'⚡ Free Tier (quota yönetimli)' if not AI_QUOTA_EXHAUSTED else '🚫 IC-Only'}")
    print(f"{'='*60}\n")
    
    # Canlı mod uyarısı
    if not dry_run:
        print("⚠️  CANLI MOD! Gerçek para riski var!")
        confirm = input("Devam etmek için 'EVET' yazın: ").strip()
        if confirm != "EVET":
            print("❌ İptal edildi.")
            sys.exit(0)
    
    # Pipeline başlat
    pipeline = HybridTradingPipeline(
        dry_run=dry_run,
        top_n=args.top,
        verbose=not args.quiet,
    )
    
    # Sadece rapor modu
    if args.report:
        pipeline.print_performance()
        sys.exit(0)
    
    # Çalışma modu
    if args.schedule:
        run_scheduler(pipeline, interval_minutes=args.interval)
    else:
        # Tek döngü
        if not pipeline._init_balance():
            logger.error("❌ Bakiye başlatılamadı")
            sys.exit(1)
        
        report = pipeline.run_cycle()
        
        # Performans özeti
        print("\n" + "─"*40)
        summary = pipeline.get_summary()
        print(f"📊 Paper Trading Özeti:")
        print(f"   Bakiye: ${summary['current_balance']:.2f}")
        print(f"   Toplam Trade: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate_pct']:.1f}%")
        print(f"   Return: {summary['total_return_pct']:+.2f}%")
        print("─"*40 + "\n")
        
        sys.exit(0 if report.status != CycleStatus.ERROR else 1)


if __name__ == "__main__":
    main()
