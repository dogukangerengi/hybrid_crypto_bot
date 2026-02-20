# =============================================================================
# MAIN.PY — HYBRID TRADING PIPELINE v1.3.0
# =============================================================================
import numpy as np
import pandas as pd
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
from ai import GeminiOptimizer, AIDecision, GateAction, AIAnalysisInput

from config import cfg
from scanner import CoinScanner
from data import BitgetFetcher, DataPreprocessor
from indicators import IndicatorCalculator, IndicatorSelector
from ai import GeminiOptimizer, AIDecision, AIDecisionResult, GateAction
from execution import RiskManager, BitgetExecutor
from notifications import TelegramNotifier

class AIDecisionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

    @classmethod
    def from_direction(cls, direction: str) -> 'AIDecisionType':
        d = (direction or "").upper()
        if d in ("LONG", "BUY", "BULLISH"):
            return cls.LONG
        elif d in ("SHORT", "SELL", "BEARISH"):
            return cls.SHORT
        return cls.WAIT

from paper_trader import PaperTrader, TradeStatus
from performance_analyzer import PerformanceAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

VERSION = "1.3.0"
MAX_COINS_PER_CYCLE = 20
DEFAULT_FWD_PERIOD = 6
MAX_OPEN_POSITIONS = 5

DEFAULT_TIMEFRAMES = {
    '15m': 400,
    '30m': 300,
    '1h':  250,
    '2h':  200,
}

AI_QUOTA_EXHAUSTED = False
AI_ERRORS_TODAY = 0
AI_ERROR_THRESHOLD = 3


class CycleStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_SIGNAL = "no_signal"
    ERROR = "error"
    KILLED = "killed"


@dataclass
class CoinAnalysisResult:
    coin: str = ""
    full_symbol: str = ""
    price: float = 0.0
    change_24h: float = 0.0
    volume_24h: float = 0.0
    best_timeframe: str = ""
    ic_confidence: float = 0.0
    ic_direction: str = ""
    significant_count: int = 0
    market_regime: str = ""
    atr: float = 0.0
    atr_pct: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    position_size: float = 0.0
    leverage: int = 1
    risk_reward: float = 0.0
    gate_action: GateAction = GateAction.NO_TRADE
    ai_decision: Optional[AIDecision] = None
    ai_skipped: bool = False
    status: str = "pending"
    error: str = ""
    execution_result: Any = None
    paper_trade_id: str = ""


@dataclass
class CycleReport:
    timestamp: str = ""
    status: CycleStatus = CycleStatus.NO_SIGNAL
    total_scanned: int = 0
    total_analyzed: int = 0
    total_above_gate: int = 0
    total_traded: int = 0
    coins: List[CoinAnalysisResult] = field(default_factory=list)
    balance: float = 0.0
    paper_balance: float = 0.0
    errors: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    ai_mode: str = "normal"


# =============================================================================
# ANA PIPELINE
# =============================================================================

class HybridTradingPipeline:

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

        self.scanner = CoinScanner()
        self.fetcher = BitgetFetcher()
        self.preprocessor = DataPreprocessor()
        self.calculator = IndicatorCalculator()
        self.selector = IndicatorSelector(alpha=0.05)
        self.ai_optimizer = GeminiOptimizer()
        self.executor = BitgetExecutor(dry_run=dry_run)

        self.notifier: Optional[TelegramNotifier] = None
        try:
            if getattr(cfg, "telegram", None) and cfg.telegram.enabled and cfg.telegram.is_configured():
                self.notifier = TelegramNotifier(
                    token=cfg.telegram.token,
                    chat_id=cfg.telegram.chat_id,
                )
                logger.info("📨 Telegram bildirimi: AKTİF")
            else:
                logger.info("📨 Telegram bildirimi: PASİF veya yapılandırılmamış")
        except Exception as e:
            logger.warning(f"📨 Telegram yapılandırma hatası: {e}")
            self.notifier = None

        self.paper_trader = PaperTrader(
            initial_balance=75.0,
            log_dir=Path(__file__).parent.parent / "logs" / "paper_trades",
            auto_save=True,
        )

        self._balance: float = 0.0
        self._initial_balance: float = 0.0
        self._risk_manager: Optional[RiskManager] = None
        self._is_running: bool = False
        self._kill_switch: bool = False
        self._cycle_count: int = 0
        self._ai_available: bool = True
        self._ai_errors: int = 0

        logger.info(
            f"🚀 HybridTradingPipeline v{VERSION} başlatıldı | "
            f"Mode: {'🧪 DRY RUN' if dry_run else '🔴 CANLI'} | "
            f"Paper Trading: ✅"
        )

    # =========================================================================
    # TELEGRAM
    # =========================================================================

    def _notify_system(self, status_type: str, details: Dict[str, Any] | None = None) -> None:
        if not self.notifier:
            return
        try:
            self.notifier.send_system_status_sync(status_type, details or {})
        except Exception as e:
            logger.warning(f"Telegram sistem bildirimi hatası: {e}")

    def _notify_trade_open(self, result: CoinAnalysisResult) -> None:
        if not self.notifier:
            return
        try:
            direction = (result.ai_decision.decision.value
                         if result.ai_decision else result.ic_direction or "UNKNOWN")
            text = (
                f"📈 Yeni trade açıldı\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🪙 Coin: {result.coin}\n"
                f"⏱ TF: {result.best_timeframe}\n"
                f"📊 IC Güven: {result.ic_confidence:.0f}/100\n"
                f"📍 Yön: {direction}\n"
                f"💰 Fiyat: ${result.price:,.6f}\n"
                f"🛑 SL: ${result.sl_price:,.6f}\n"
                f"🎯 TP: ${result.tp_price:,.6f}\n"
                f"⚙️ Kaldıraç: x{result.leverage}\n"
                f"━━━━━━━━━━━━━━━━━━━━━"
            )
            self.notifier.send_message_sync(text)
        except Exception as e:
            logger.warning(f"Telegram trade bildirimi hatası: {e}")

    def _notify_trades_closed(self, closed_trades: List) -> None:
        if not self.notifier or not closed_trades:
            return
        try:
            lines = ["📊 Trade(ler) kapandı", "━━━━━━━━━━━━━━━━━━━━━"]
            for trade in closed_trades:
                emoji = "✅" if getattr(trade, "net_pnl", 0) > 0 else "❌"
                lines.append(
                    f"{emoji} {trade.symbol} {trade.direction} | "
                    f"PnL: ${trade.net_pnl:+.2f} ({trade.pnl_percent:+.1f}%)"
                )
            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            self.notifier.send_message_sync("\n".join(lines))
        except Exception as e:
            logger.warning(f"Telegram kapanış bildirimi hatası: {e}")

    # =========================================================================
    # BAKİYE
    # =========================================================================

    def _init_balance(self) -> bool:
        try:
            if self.dry_run:
                self._balance = self.paper_trader.balance
                self._initial_balance = self.paper_trader.initial_balance
                logger.info(f"💰 Paper bakiye: ${self._balance:.2f}")
            else:
                balance_info = self.executor.fetch_balance()
                # total = free + margin'de kilitli → gerçek bakiyeyi yansıtır
                self._balance = balance_info.get('total', 0.0)
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
        if self.dry_run:
            self._balance = self.paper_trader.balance
        else:
            try:
                balance_info = self.executor.fetch_balance()
                # total kullan — free değil (pozisyon margin'i hesaba kat)
                self._balance = balance_info.get('total', 0.0)
            except Exception as e:
                logger.warning(f"⚠️ Bakiye güncelleme hatası: {e}")

        if self._risk_manager:
            self._risk_manager.update_balance(self._balance)

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def _check_kill_switch(self) -> bool:
        if self._initial_balance <= 0:
            return False

        if self.dry_run:
            drawdown_pct = self.paper_trader.max_drawdown
        else:
            # total bakiye ile hesapla — gerçek drawdown
            drawdown_pct = max(0, (self._initial_balance - self._balance) / self._initial_balance * 100)

        threshold = cfg.risk.kill_switch_pct if hasattr(cfg.risk, 'kill_switch_pct') else 15.0

        if drawdown_pct >= threshold:
            self._kill_switch = True
            logger.warning(f"🛑 KILL SWITCH AKTİF! DD: {drawdown_pct:.1f}% >= {threshold:.1f}%")
            try:
                if self.notifier:
                    self.notifier.send_risk_alert_sync(
                        alert_type="kill_switch",
                        details=f"DD: {drawdown_pct:.1f}% — Eşik: {threshold:.1f}%",
                        severity="critical",
                    )
            except Exception:
                pass

            if self.dry_run and self.paper_trader.open_trades:
                prices = self._get_current_prices()
                self.paper_trader.close_all_trades(prices, "Kill switch")

            return True

        return False

    def _get_current_prices(self) -> Dict[str, float]:
        prices = {}
        for trade_id, trade in self.paper_trader.open_trades.items():
            try:
                ticker = self.fetcher.get_ticker(trade.full_symbol)
                prices[trade.symbol] = ticker['last']
            except:
                prices[trade.symbol] = trade.entry_price
        return prices

    # =========================================================================
    # MARKET TARAMA
    # =========================================================================

    def _scan_market(self) -> List:
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
        clean_coin = symbol.split('/')[0] if '/' in symbol else symbol
        result = CoinAnalysisResult(coin=clean_coin)

        try:
            full_symbol = f"{clean_coin}/USDT:USDT"
            result.full_symbol = full_symbol

            ticker = self.fetcher.get_ticker(full_symbol)
            result.price = ticker.get('last', 0)
            result.change_24h = ticker.get('percentage', 0) or 0
            result.volume_24h = ticker.get('quoteVolume', 0) or 0

            tf_results = []

            for tf, limit in self.timeframes.items():
                try:
                    df = self.fetcher.fetch_ohlcv(full_symbol, timeframe=tf, limit=limit)
                    if df is None or len(df) < 50:
                        continue

                    df = self.calculator.calculate_all(df)
                    df = self.calculator.add_price_features(df)
                    df = self.calculator.add_forward_returns(df, periods=[1, self.fwd_period])

                    target_col = f'fwd_ret_{self.fwd_period}'
                    scores = self.selector.evaluate_all_indicators(df, target_col=target_col)

                    valid_categories = ['trend', 'momentum', 'volatility', 'volume']
                    sig_scores = [
                        s for s in scores
                        if not np.isnan(s.ic_mean)
                        and abs(s.ic_mean) > 0.02
                        and s.category in valid_categories
                    ]

                    if not sig_scores:
                        continue

                    top_score = max(sig_scores, key=lambda x: abs(x.ic_mean))
                    top_ic_val = abs(top_score.ic_mean)
                    avg_ic = np.mean([abs(s.ic_mean) for s in sig_scores])

                    pos_count = sum(1 for s in sig_scores if s.ic_mean > 0)
                    neg_count = sum(1 for s in sig_scores if s.ic_mean < 0)
                    consistency = max(pos_count, neg_count) / len(sig_scores)

                    if neg_count > pos_count * 1.5:
                        direction = 'SHORT'
                    elif pos_count > neg_count * 1.5:
                        direction = 'LONG'
                    else:
                        direction = 'NEUTRAL'

                    regime = 'unknown'
                    if 'ADX_14' in df.columns:
                        adx_series = df['ADX_14'].dropna()
                        if not adx_series.empty:
                            adx_val = float(adx_series.iloc[-1])
                            regime = 'trending' if adx_val > 25 else ('ranging' if adx_val < 15 else 'transitioning')

                    top_norm  = min((top_ic_val - 0.02) / 0.38 * 100, 100)
                    avg_norm  = min((avg_ic - 0.02) / 0.13 * 100, 100)
                    cnt_norm  = min(len(sig_scores) / 50 * 100, 100)
                    cons_norm = max(0, min((consistency - 0.5) / 0.5 * 100, 100))

                    composite = (
                        top_norm  * 0.40 +
                        avg_norm  * 0.25 +
                        cnt_norm  * 0.15 +
                        cons_norm * 0.20
                    )

                    regime_mult = {'ranging': 0.85, 'volatile': 0.80, 'transitioning': 0.90}
                    composite *= regime_mult.get(regime, 1.0)

                    atr_val = 0.0
                    if 'ATRr_14' in df.columns and not df['ATRr_14'].dropna().empty:
                        atr_val = float(df['ATRr_14'].dropna().iloc[-1])
                    elif 'NATR_14' in df.columns and not df['NATR_14'].dropna().empty:
                        natr = float(df['NATR_14'].dropna().iloc[-1])
                        last_price = float(df['close'].iloc[-1])
                        atr_val = last_price * natr / 100
                    else:
                        high = df['high']
                        low = df['low']
                        close = df['close']
                        tr = pd.concat([
                            high - low,
                            (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs()
                        ], axis=1).max(axis=1)
                        atr_series = tr.rolling(14).mean()
                        if not atr_series.dropna().empty:
                            atr_val = float(atr_series.iloc[-1])

                    last_close = float(df['close'].iloc[-1])
                    atr_pct = (atr_val / last_close * 100) if last_close > 0 else 0

                    if composite > 0:
                        tf_results.append({
                            'tf': tf, 'score': composite, 'direction': direction,
                            'regime': regime, 'significant': len(sig_scores),
                            'atr': atr_val, 'atr_pct': atr_pct,
                        })

                except Exception as e:
                    logger.debug(f"  {full_symbol} {tf}: Analiz hatası — {e}")
                    continue

            if not tf_results:
                result.status = "no_data"
                return result

            best = max(tf_results, key=lambda x: x['score'])
            result.best_timeframe = best['tf']
            result.ic_confidence = best['score']
            result.ic_direction = best['direction']
            result.market_regime = best['regime']
            result.significant_count = best['significant']
            result.atr = best['atr']
            result.atr_pct = best['atr_pct']

            no_trade_threshold = cfg.gate.no_trade if hasattr(cfg.gate, 'no_trade') else 40
            full_trade_threshold = cfg.gate.full_trade if hasattr(cfg.gate, 'full_trade') else 55

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
    # AI OPTİMİZASYON
    # =========================================================================

    def _get_ai_decision(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        global AI_QUOTA_EXHAUSTED, AI_ERRORS_TODAY

        ai_input = AIAnalysisInput(
            symbol=result.full_symbol,
            coin=result.coin,
            price=result.price,
            change_24h=result.change_24h,
            best_timeframe=result.best_timeframe,
            ic_confidence=result.ic_confidence,
            ic_direction=result.ic_direction,
            category_tops={},
            tf_rankings=[],
            atr=result.atr,
            atr_pct=result.atr_pct,
            sl_price=result.sl_price,
            tp_price=result.tp_price,
            market_regime=result.market_regime,
            volume_24h=result.volume_24h,
            volatility=0.0
        )

        if AI_QUOTA_EXHAUSTED or not self._ai_available:
            result.ai_skipped = True
            from ai import AIDecisionResult, AIDecision as AI_Decision_Enum
            decision_enum = AIDecisionType.from_direction(result.ic_direction)
            decision_val = AI_Decision_Enum[decision_enum.name]
            result.ai_decision = AIDecisionResult(
                decision=decision_val,
                confidence=result.ic_confidence * 0.8,
                reasoning="AI quota aşıldı — IC fallback",
                gate_action=result.gate_action,
                ic_score=result.ic_confidence,
                model_used="ic_only_mode",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            logger.info(f"⚡ {result.coin}: AI atlandı (IC-only mode)")
            return result

        try:
            time.sleep(12)  # Gemini free tier rate limit (5 req/min)
            ai_decision_result = self.ai_optimizer.get_decision(ai_input)
            result.ai_decision = ai_decision_result
            result.ai_skipped = False

            logger.info(
                f"🤖 {result.coin}: AI → {ai_decision_result.decision.value} "
                f"(Güven: {ai_decision_result.confidence:.0f})"
            )

            AI_ERRORS_TODAY = 0

        except Exception as e:
            error_msg = str(e).lower()
            if 'quota' in error_msg or '429' in error_msg or 'rate' in error_msg:
                AI_ERRORS_TODAY += 1
                logger.warning(f"⚠️ AI quota hatası ({AI_ERRORS_TODAY}/{AI_ERROR_THRESHOLD}): {e}")
                if AI_ERRORS_TODAY >= AI_ERROR_THRESHOLD:
                    AI_QUOTA_EXHAUSTED = True

            from ai import AIDecisionResult, AIDecision as AI_Decision_Enum
            decision_enum = AIDecisionType.from_direction(result.ic_direction)
            decision_val = AI_Decision_Enum[decision_enum.name]
            result.ai_skipped = True
            result.ai_decision = AIDecisionResult(
                decision=decision_val,
                confidence=result.ic_confidence * 0.8,
                reasoning=f"AI hatası — IC fallback",
                gate_action=result.gate_action,
                ic_score=result.ic_confidence,
                model_used="error_fallback",
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )

        return result

    # =========================================================================
    # RİSK HESAPLAMA
    # =========================================================================

    def _calculate_risk(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        if not self._risk_manager:
            return result

        try:
            direction = result.ai_decision.decision.value if result.ai_decision else result.ic_direction
            if direction == "WAIT":
                return result

            atr_mult = 1.5
            if result.ai_decision and hasattr(result.ai_decision, 'atr_multiplier'):
                atr_mult = float(result.ai_decision.atr_multiplier)

            trade_calc = self._risk_manager.calculate_trade(
                entry_price=result.price,
                direction=direction,
                atr=result.atr,
                symbol=result.full_symbol,
                atr_multiplier=atr_mult
            )

            result.sl_price = trade_calc.stop_loss.price
            result.tp_price = trade_calc.take_profit.price
            result.position_size = trade_calc.position.size
            result.leverage = trade_calc.position.leverage
            result.risk_reward = trade_calc.take_profit.risk_reward

        except Exception as e:
            logger.error(f"❌ Risk hesaplama hatası: {e}")
            result.error = str(e)

        return result

    # =========================================================================
    # PAPER TRADE AÇMA
    # =========================================================================

    def _execute_paper_trade(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        try:
            direction = result.ai_decision.decision.value if result.ai_decision else result.ic_direction
            if direction == "WAIT":
                result.status = "skipped_wait"
                return result

            # ✅ Max 5 pozisyon kontrolü
            open_count = len(self.paper_trader.open_trades)
            max_pos = MAX_OPEN_POSITIONS

            if open_count >= max_pos:
                logger.warning(f"⚠️ {result.coin}: Max pozisyon ({max_pos}) dolu — atlanıyor")
                result.status = "position_limit"
                return result

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
                f"📝 Paper Trade: {result.coin} {direction} @ ${result.price:,.6f} | "
                f"SL: ${result.sl_price:,.6f} | TP: ${result.tp_price:,.6f}"
            )
            self._notify_trade_open(result)

        except Exception as e:
            result.status = "execution_error"
            result.error = str(e)
            logger.error(f"❌ Paper trade hatası: {e}")

        return result

    # =========================================================================
    # CANLI TRADE AÇMA
    # =========================================================================

    def _execute_live_trade(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        try:
            direction = result.ai_decision.decision.value if result.ai_decision else result.ic_direction
            if direction == "WAIT":
                result.status = "skipped_wait"
                return result

            # ✅ Max 5 pozisyon kontrolü (canlı pozisyonlar Bitget'ten çekilir)
            try:
                live_positions = self.executor.fetch_positions()
                open_count = len(live_positions)
            except:
                open_count = 0

            max_pos = 5  # Sabit limit
            if open_count >= max_pos:
                logger.warning(f"⚠️ {result.coin}: Max canlı pozisyon ({max_pos}) dolu — atlanıyor")
                result.status = "position_limit"
                return result

            class DynamicTradeCalc:
                def __init__(self, r, d):
                    self.symbol = r.full_symbol
                    self.direction = d
                    self.entry_price = r.price
                    self.rejection_reasons = []

                    class Pos:
                        size = r.position_size
                        leverage = r.leverage
                    self.position = Pos()

                    class Target:
                        def __init__(self, p):
                            self.price = p
                    self.stop_loss = Target(r.sl_price)
                    self.take_profit = Target(r.tp_price)

                def is_approved(self):
                    return True

            trade_calc_obj = DynamicTradeCalc(result, direction)
            exec_result = self.executor.execute_trade(trade_calc_obj)

            result.execution_result = exec_result

            if exec_result.success:
                result.status = "executed"
                logger.info(f"🔴 CANLI TRADE AÇILDI: {result.coin} {direction} @ ${result.price:.6f}")
                self._notify_trade_open(result)
            else:
                result.status = "execution_error"
                result.error = exec_result.error
                logger.error(f"❌ Canlı trade reddedildi: {exec_result.error}")

        except Exception as e:
            result.status = "execution_error"
            result.error = str(e)
            logger.error(f"❌ Canlı trade kritik hata: {e}")

        return result

    # =========================================================================
    # AÇIK POZİSYON KONTROLÜ
    # =========================================================================

    def _check_open_positions(self) -> List:
        if not self.paper_trader.open_trades:
            return []

        prices = self._get_current_prices()
        closed = self.paper_trader.check_exits(prices)

        for trade in closed:
            emoji = "✅" if trade.net_pnl > 0 else "❌"
            logger.info(
                f"{emoji} Trade kapandı: {trade.trade_id} | "
                f"{trade.symbol} {trade.direction} | "
                f"PnL: ${trade.net_pnl:+.2f} ({trade.pnl_percent:+.1f}%)"
            )

        self._notify_trades_closed(closed)
        return closed

    # =========================================================================
    # ANA DÖNGÜ
    # =========================================================================

    def run_cycle(self) -> CycleReport:
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
            self._refresh_balance()
            report.balance = self._balance
            report.paper_balance = self.paper_trader.balance

            if self._check_kill_switch():
                report.status = CycleStatus.KILLED
                report.elapsed = time.time() - cycle_start
                return report

            closed_trades = self._check_open_positions()
            if closed_trades:
                logger.info(f"📊 {len(closed_trades)} pozisyon kapandı")

            top_coins = self._scan_market()
            report.total_scanned = len(top_coins)

            if not top_coins:
                report.status = CycleStatus.NO_SIGNAL
                report.elapsed = time.time() - cycle_start
                return report

            for coin_data in top_coins:
                symbol = coin_data.symbol if hasattr(coin_data, 'symbol') else coin_data.get('symbol', '')

                result = self._analyze_coin(symbol)

                if not result or result.status == "error":
                    logger.warning(f"⚠️ {symbol}: Analiz hatası")
                    continue

                if result.status == "no_data":
                    logger.warning(f"⚠️ {symbol}: Yetersiz veri")
                    continue

                report.total_analyzed += 1

                if result.gate_action == GateAction.NO_TRADE:
                    logger.info(f"🚫 {symbol}: Reddedildi (IC: {result.ic_confidence:.1f})")
                    result.status = "below_gate"
                    report.coins.append(result)
                    continue

                report.total_above_gate += 1
                logger.info(f"✨ {symbol}: Gate GEÇTİ! (IC: {result.ic_confidence:.1f}) -> AI'ya gidiyor...")

                if result.gate_action == GateAction.FULL_TRADE:
                    result = self._get_ai_decision(result)

                    # WAIT kararı → atla
                    if result.ai_decision and result.ai_decision.decision == AIDecision.WAIT:
                        result.status = "ai_wait"
                        report.coins.append(result)
                        continue

                    # ✅ Minimum güven kontrolü (60 altı = trade yok)
                    if result.ai_decision:
                        min_confidence = cfg.ai.__dict__.get('min_confidence', 60)
                        if result.ai_decision.confidence < min_confidence:
                            logger.warning(
                                f"⚠️ {result.coin}: AI güveni düşük "
                                f"({result.ai_decision.confidence:.0f} < {min_confidence}) → SKIP"
                            )
                            result.status = "low_confidence"
                            report.coins.append(result)
                            continue

                    # Risk hesapla
                    result = self._calculate_risk(result)

                    # ✅ SL/TP zorunlu — ikisi de sıfırsa trade açma
                    if result.sl_price <= 0 or result.tp_price <= 0:
                        logger.warning(f"⚠️ {result.coin}: SL/TP hesaplanamadı → trade açılmıyor")
                        result.status = "no_sltp"
                        report.coins.append(result)
                        continue

                    # Trade aç
                    if self.dry_run:
                        result = self._execute_paper_trade(result)
                    else:
                        result = self._execute_live_trade(result)

                    # ✅ SL/TP gönderilemezse trade başarısız say
                    if result.execution_result and not result.execution_result.success:
                        logger.error(f"❌ {result.coin}: Trade execution başarısız")
                    elif result.status == "executed":
                        report.total_traded += 1

                report.coins.append(result)

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
        self._print_cycle_summary(report)
        return report

    def _print_cycle_summary(self, report: CycleReport) -> None:
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

    def print_performance(self) -> None:
        analyzer = PerformanceAnalyzer(self.paper_trader)
        report = analyzer.full_analysis()
        analyzer.print_report(report)

    def get_summary(self) -> Dict:
        return self.paper_trader.get_summary()


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(pipeline: HybridTradingPipeline, interval_minutes: int = 60):
    pipeline._is_running = True

    def signal_handler(signum, frame):
        logger.info("\n🛑 Durdurma sinyali alındı...")
        pipeline._is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"⏰ Scheduler başladı | Interval: {interval_minutes} dakika")

    if not pipeline._init_balance():
        logger.error("❌ Bakiye başlatılamadı")
        return

    pipeline._notify_system('startup', {
        'balance': pipeline._balance,
        'mode': '🧪 DRY RUN' if pipeline.dry_run else '🔴 CANLI',
    })

    while pipeline._is_running:
        report = pipeline.run_cycle()

        if report.status == CycleStatus.KILLED:
            logger.warning("🛑 Kill switch — scheduler durduruluyor")
            break

        if not pipeline._is_running:
            break

        logger.info(f"⏳ Sonraki döngü: {interval_minutes} dakika sonra...")
        for _ in range(interval_minutes):
            if not pipeline._is_running:
                break
            time.sleep(60)

    pipeline.print_performance()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Crypto Trading Bot v" + VERSION)
    parser.add_argument('--dry-run', '-d', action='store_true', default=True)
    parser.add_argument('--live', '-L', action='store_true')
    parser.add_argument('--schedule', '-s', action='store_true')
    parser.add_argument('--interval', '-i', type=int, default=75)
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--top', '-n', type=int, default=15)
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--report', '-r', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    dry_run = not args.live

    print(f"\n{'='*60}")
    print(f"  🚀 HYBRID CRYPTO BOT v{VERSION}")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🔧 Mode: {'🧪 PAPER TRADE' if dry_run else '🔴 CANLI'}")
    print(f"  🤖 AI: {'⚡ Free Tier (quota yönetimli)' if not AI_QUOTA_EXHAUSTED else '🚫 IC-Only'}")
    print(f"{'='*60}\n")

    if not dry_run:
        print("⚠️  CANLI MOD! Gerçek para riski var!")
        if sys.stdin.isatty():
            confirm = input("Devam etmek için 'EVET' yazın: ")
            if confirm != 'EVET':
                print("İptal edildi.")
                sys.exit(0)
        else:
            logger.info("⚡ Arka plan modu — onay atlandı, CANLI başlıyor")

    pipeline = HybridTradingPipeline(
        dry_run=dry_run,
        top_n=args.top,
        verbose=not args.quiet,
    )

    if args.report:
        pipeline.print_performance()
        sys.exit(0)

    if args.schedule:
        run_scheduler(pipeline, interval_minutes=args.interval)
    else:
        if not pipeline._init_balance():
            logger.error("❌ Bakiye başlatılamadı")
            sys.exit(1)

        pipeline._notify_system('startup', {
            'balance': pipeline._balance,
            'mode': '🧪 DRY RUN' if dry_run else '🔴 CANLI',
        })

        report = pipeline.run_cycle()

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
