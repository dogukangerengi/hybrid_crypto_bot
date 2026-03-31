# =============================================================================
# MAIN.PY — ML-DRIVEN TRADING PIPELINE v2.1.3 (BINANCE VADELİ İŞLEMLER)
# =============================================================================
# v2.1.3 Düzeltmeler:
#   ✅ Tarama süresi 10 dakikaya düşürüldü.
#   ✅ Cooldown için UTC saat dilimi hataları düzeltildi.
#   ✅ Scheduler uyku modundayken 30 saniyede bir açık emir takibi eklendi.
# =============================================================================

import sys
import os
import time
import signal
import argparse
import logging
import traceback
import schedule

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from execution.risk_manager import RiskManager
import numpy as np
import pandas as pd

# ── .env yükle ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
_src_dir  = Path(__file__).parent
_root_dir = _src_dir.parent
load_dotenv(_root_dir / ".env")
sys.path.insert(0, str(_src_dir))

# ── Mevcut modüller ───────────────────────────────────────────────────────────
from config import cfg
from scanner import CoinScanner
from data import BinanceFetcher, DataPreprocessor
from indicators import IndicatorCalculator, IndicatorSelector
from execution import RiskManager, BinanceExecutor
from notifications import TelegramNotifier
from paper_trader import PaperTrader
from performance_analyzer import PerformanceAnalyzer

# ── ML modülleri (v2.0) ───────────────────────────────────────────────────────
from ml.feature_engineer import FeatureEngineer, MLFeatureVector
from ml.ensemble_model import EnsemblePredictor as LGBMSignalModel, MLDecisionResult
from ml.signal_validator import SignalValidator, ValidationResult
from ml.trade_memory import TradeMemory, TradeOutcome

# =============================================================================
# LOGLAMA
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# =============================================================================
# SABİTLER
# =============================================================================

VERSION                = "2.1.3"
MAX_COINS_PER_CYCLE    = 30
DEFAULT_FWD_PERIOD     = 6
MAX_OPEN_POSITIONS     = 5
MAX_CONSECUTIVE_ERRORS = 5
ERROR_COOLDOWN_SECONDS = 300

DEFAULT_TIMEFRAMES = {
    '15m': 400,
    '30m': 300,
    '1h' : 250,
    '2h' : 200,
}

IC_NO_TRADE = 12.0
IC_TRADE    = 16.0

# =============================================================================
# ENUM'LAR
# =============================================================================

class CycleStatus(Enum):
    SUCCESS   = "success"
    PARTIAL   = "partial"
    NO_SIGNAL = "no_signal"
    ERROR     = "error"
    KILLED    = "killed"

# =============================================================================
# DATACLASS'LAR
# =============================================================================

@dataclass
class CoinAnalysisResult:
    coin:             str   = ""               
    full_symbol:      str   = ""               
    price:            float = 0.0              
    change_24h:       float = 0.0              
    volume_24h:       float = 0.0              

    best_timeframe:   str   = ""               
    ic_confidence:    float = 0.0              
    ic_direction:     str   = ""               
    significant_count: int  = 0                
    market_regime:    str   = ""               

    category_tops:    Dict  = field(default_factory=dict)
    tf_rankings:      List  = field(default_factory=list)

    atr:              float = 0.0
    atr_pct:          float = 0.0
    sl_price:         float = 0.0
    tp_price:         float = 0.0
    position_size:    float = 0.0
    leverage:         int   = 1
    risk_reward:      float = 0.0

    ml_result:        Optional[MLDecisionResult] = None
    val_result:       Optional[ValidationResult] = None
    ml_skipped:       bool  = False            

    trade_executed:   bool  = False
    status:           str   = "pending"
    error:            str   = ""
    execution_result: Any   = None
    paper_trade_id:   str   = ""


@dataclass
class CycleReport:
    timestamp:        str         = ""
    status:           CycleStatus = CycleStatus.NO_SIGNAL
    total_scanned:    int         = 0
    total_analyzed:   int         = 0
    total_above_gate: int         = 0
    total_traded:     int         = 0
    coins:            List[CoinAnalysisResult] = field(default_factory=list)
    balance:          float       = 0.0
    paper_balance:    float       = 0.0
    errors:           List[str]   = field(default_factory=list)
    elapsed:          float       = 0.0
    ml_stats:         Dict        = field(default_factory=dict)


# =============================================================================
# ANA PIPELINE
# =============================================================================

class MLTradingPipeline:
    def __init__(
        self,
        dry_run:    bool = True,
        top_n:      int  = MAX_COINS_PER_CYCLE,
        timeframes: Dict = None,
        fwd_period: int  = DEFAULT_FWD_PERIOD,
        verbose:    bool = True,
    ):
        self.dry_run    = dry_run
        self.top_n      = min(top_n, MAX_COINS_PER_CYCLE)
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.fwd_period = fwd_period
        self.verbose    = verbose

        self.scanner      = CoinScanner()
        self.fetcher      = BinanceFetcher()
        self.preprocessor = DataPreprocessor()
        self.calculator   = IndicatorCalculator()
        self.selector     = IndicatorSelector(alpha=0.05)
        self.risk_manager = RiskManager()
        self.executor     = BinanceExecutor(dry_run=dry_run)
        self.notifier     = TelegramNotifier()
        self.paper_trader = PaperTrader()

        self.feature_eng  = FeatureEngineer()
        self.lgbm_model   = LGBMSignalModel()
        self.validator    = SignalValidator()
        self.trade_memory = TradeMemory(log_dir = _root_dir / "logs")

        self._balance          = 0.0
        self._initial_balance  = 0.0
        self._kill_switch      = False
        self._is_running       = False
        self._consecutive_errors = 0
        self.cooldowns         = {}

        self._restore_cooldowns()

        logger.info(f"🚀 ML Trading Pipeline v{VERSION} (dry_run={dry_run})")

    def _restore_cooldowns(self):
        try:
            now = datetime.now(timezone.utc) # UTC olarak güncellendi
            
            if not self.dry_run:
                memory_file = self.trade_memory.log_dir / "ml_trade_memory.json"
                if memory_file.exists():
                    import json
                    try:
                        with open(memory_file, 'r', encoding='utf-8') as f:
                            trades = json.load(f)
                        
                        for trade in trades:
                            exit_reason = trade.get('exit_reason', '')
                            closed_at_str = trade.get('closed_at', '')
                            coin = trade.get('coin', '')
                            symbol = trade.get('symbol', '')
                            
                            if exit_reason == "SL Hit" and closed_at_str and coin:
                                try:
                                    closed_time = datetime.fromisoformat(closed_at_str)
                                    # Zaman dilimi yoksa UTC ekle
                                    if closed_time.tzinfo is None:
                                        closed_time = closed_time.replace(tzinfo=timezone.utc)
                                    
                                    if (now - closed_time).total_seconds() < 7200:
                                        kalan_sure = closed_time + timedelta(hours=2)
                                        self.cooldowns[coin] = kalan_sure
                                        if symbol:
                                            self.cooldowns[symbol] = kalan_sure
                                        logger.info(f"   ❄️ HAFIZA GERİ YÜKLENDİ: {coin} (Ceza bitişi: {kalan_sure.strftime('%H:%M:%S')})")
                                except (ValueError, TypeError):
                                    continue
                    except Exception as e:
                        logger.warning(f"Trade memory cooldown yükleme hatası: {e}")
            else:
                if hasattr(self.paper_trader, 'closed_trades'):
                    for trade in self.paper_trader.closed_trades:
                        exit_reason = getattr(trade, 'exit_reason', None) or (trade.get('exit_reason') if isinstance(trade, dict) else None)
                        closed_at = getattr(trade, 'closed_at', None) or (trade.get('closed_at') if isinstance(trade, dict) else None)
                        symbol = getattr(trade, 'symbol', None) or (trade.get('symbol') if isinstance(trade, dict) else None)
                        full_symbol = getattr(trade, 'full_symbol', None) or (trade.get('full_symbol') if isinstance(trade, dict) else None)

                        if exit_reason == "SL Hit" and closed_at and symbol:
                            if isinstance(closed_at, str):
                                try:
                                    closed_time = datetime.fromisoformat(closed_at)
                                    if closed_time.tzinfo is None:
                                        closed_time = closed_time.replace(tzinfo=timezone.utc)
                                except ValueError:
                                    continue
                            else:
                                closed_time = closed_at
                                if closed_time.tzinfo is None:
                                    closed_time = closed_time.replace(tzinfo=timezone.utc)
                            
                            if (now - closed_time).total_seconds() < 7200:
                                kalan_sure = closed_time + timedelta(hours=2)
                                self.cooldowns[symbol] = kalan_sure
                                if full_symbol:
                                    self.cooldowns[full_symbol] = kalan_sure
                                logger.info(f"   ❄️ HAFIZA GERİ YÜKLENDİ: {symbol} (Ceza bitişi: {kalan_sure.strftime('%H:%M:%S')})")
                            
        except Exception as e:
            logger.error(f"Cooldown hafıza yükleme hatası: {e}")

    def _init_balance(self) -> bool:
        try:
            if self.dry_run:
                self._balance = self._initial_balance = 1000.0
                logger.info(f"💰 Paper bakiye: ${self._balance:.2f}")
            else:
                b = self.executor.fetch_balance()
                if isinstance(b, dict):
                    self._balance = self._initial_balance = b.get('total', 0.0)
                else:
                    self._balance = self._initial_balance = float(b)
                
                logger.info(f"💰 Canlı bakiye: ${self._balance:.2f}")
            return True
        except Exception as e:
            logger.error(f"❌ Bakiye hatası: {e}")
            return False

    def _check_kill_switch(self) -> bool:
        if self._kill_switch:
            return True
        if self._initial_balance <= 0:
            return False
        dd = (self._initial_balance - self._balance) / self._initial_balance * 100
        if dd >= cfg.risk.kill_switch_drawdown_pct:
            self._kill_switch = True
            logger.critical(f"🚨 KILL SWITCH! DD={dd:.1f}%")
            if self.notifier.is_configured():
                self.notifier.send_risk_alert_sync(
                    alert_type="KILL_SWITCH",
                    message=f"⛔ Kill switch! DD={dd:.1f}%",
                    balance=self._balance, drawdown=dd,
                )
            return True
        return False

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """ADX bazlı piyasa rejimi: 'trending' / 'ranging' / 'volatile'"""
        try:
            # Sütun adlarında içinde 'ADX' geçen ilk kolonu bulur
            adx_col = next((col for col in df.columns if 'ADX' in col.upper()), None)
            
            if adx_col:
                # Sütundaki NaN (boş) olan değerleri atlayarak sadece geçerli rakamları filtreler
                adx_series = df[adx_col].dropna()
                
                # Eğer seri tamamen boş değilse son geçerli rakamı alır
                if not adx_series.empty:
                    adx = float(adx_series.iloc[-1]) # Son geçerli rakamı ondalıklı sayıya çevir
                    
                    if adx > 25: return 'trending'
                    if adx > 15: return 'ranging'
                    return 'volatile'
        except Exception:
            pass
        return 'unknown'

    def _analyze_coin(self, symbol: str, coin: str) -> CoinAnalysisResult:
        result = CoinAnalysisResult(coin=coin, full_symbol=symbol)

        try:
            all_data = {}
            for tf, limit in self.timeframes.items():
                df_raw = self.fetcher.fetch_ohlcv(symbol, tf, limit=limit)
                if df_raw is None or len(df_raw) < 50:
                    continue
                df_clean = self.preprocessor.full_pipeline(df_raw)
                if df_clean is not None and len(df_clean) > 50:
                    all_data[tf] = df_clean

            if not all_data:
                result.status = "no_data"; return result

            result.price = float(next(iter(all_data.values()))['close'].iloc[-1])

            indicator_data = {}
            for tf, df in all_data.items():
                df_ind = self.calculator.calculate_all(df)
                df_ind = self.calculator.add_forward_returns(
                    df_ind, periods=[self.fwd_period]
                )
                if df_ind is not None and len(df_ind) > 50:
                    indicator_data[tf] = df_ind

            if not indicator_data:
                result.status = "indicator_error"; return result

            target_col = f'fwd_ret_{self.fwd_period}'
            best_tf    = None
            best_ic    = -1.0
            best_scores= []
            tf_rankings= []

            for tf, df in indicator_data.items():
                if target_col not in df.columns:
                    continue
                scores = self.selector.evaluate_all_indicators(df, target_col=target_col)
                if not scores:
                    continue
                avg_ic = sum(abs(s.ic_mean) for s in scores) / len(scores)
                significant = [s for s in scores if s.is_significant]
                if len(significant) == 0:
                    significant = scores[:2]

                tf_rankings.append({
                    'tf': tf,
                    'avg_ic': avg_ic,
                    'sig_count': len([s for s in scores if s.is_significant]),
                })

                if avg_ic > best_ic:
                    best_ic = avg_ic
                    best_tf = tf
                    best_scores = significant

            if not best_tf:
                result.status = "no_ic"; return result

            result.best_timeframe = best_tf
            result.tf_rankings    = sorted(tf_rankings, key=lambda x: x['avg_ic'], reverse=True)

            ic_scores = [s.ic_mean for s in best_scores]
            ic_signal_dir = "NEUTRAL"
            if ic_scores:
                ic_avg = sum(ic_scores) / len(ic_scores)
                if ic_avg > 0:
                    ic_signal_dir = "LONG"
                elif ic_avg < 0:
                    ic_signal_dir = "SHORT"
                result.ic_confidence   = abs(ic_avg) * 100
                result.ic_direction    = ic_signal_dir
                result.significant_count = len(best_scores)

            if result.ic_confidence < IC_NO_TRADE:
                result.status = "ic_too_low"; return result
            if result.ic_confidence < IC_TRADE:
                result.status = "below_gate"; return result

            result.market_regime = self._detect_regime(indicator_data[best_tf])

            CATEGORIES = {
                'volume':  ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI'],
                'momentum':['RSI', 'Stoch', 'MFI', 'UO', 'MACD', 'PPO', 'ROC',
                            'TSI', 'CCI', 'Williams'],
                'trend':   ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST',
                            'Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'ATR', 'Keltner', 'Donchian', 'STD'],
            }
            dynamic_cat_tops = {}
            for cat, cat_indicators in CATEGORIES.items():
                cat_scores = [s for s in best_scores if any(
                    s.name.lower().startswith(ci.lower()) or ci.lower() in s.name.lower()
                    for ci in cat_indicators
                )]
                if not cat_scores:
                    cat_ind_lower = [x.lower() for x in cat_indicators]
                    matched = []
                    for s in best_scores:
                        s_lower = s.name.lower()
                        if not cat_scores:
                            for ci in cat_ind_lower:
                                if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                    if len(ci) >= 3:
                                        matched.append(s)
                                        break
                    cat_scores = matched

                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    dynamic_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}

            result.category_tops = dynamic_cat_tops

            df_best = indicator_data[best_tf]
            if 'ATR_14' in df_best.columns:
                result.atr     = float(df_best['ATR_14'].iloc[-1])
                result.atr_pct = (result.atr / result.price) * 100

            if not self.lgbm_model.is_trained:
                result.ml_skipped = True
                result.status = "model_not_trained"
                return result

            fv = self.feature_eng.build_features(
                analysis   = result,
                ohlcv_df   = df_best
            )

            ml_result = self.lgbm_model.predict(
                fv, result.ic_confidence, result.ic_direction
            )
            result.ml_result = ml_result

            if ml_result.decision.value == "WAIT":
                result.status = "wait"; return result

            val_result = self.validator.validate(
                fv, self.lgbm_model, ml_result.decision, ml_result.confidence,
                result.ic_direction, result.market_regime
            )
            result.val_result = val_result

            # Validator sonucunun is_valid veya is_approved özelliğini güvenli şekilde alıyoruz
            is_ok = getattr(val_result, 'is_valid', False) or getattr(val_result, 'is_approved', False)
            
            # Eğer onay çıkmadıysa işlemi reddet
            if not is_ok:
                result.status = "rejected_by_validator"
                return result

            result.status = "ready"

        except Exception as e:
            result.status = "error"
            result.error  = str(e)
            logger.error(f"   ❌ {coin} analiz hatası: {e}", exc_info=True)

        return result

    def _execute_trade(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        # UTC kontrolü eklendi
        if hasattr(self, 'cooldowns'):
            if result.coin in self.cooldowns or result.full_symbol in self.cooldowns:
                cooldown_key = result.coin if result.coin in self.cooldowns else result.full_symbol
                if datetime.now(timezone.utc) < self.cooldowns[cooldown_key]:
                    kalan_dk = int((self.cooldowns[cooldown_key] - datetime.now(timezone.utc)).total_seconds() / 60)
                    logger.info(f"   ❄️ {result.coin} işlemi REDDEDİLDİ: Soğuma süresinde (Kalan: {kalan_dk} dk)")
                    result.status = "cooldown"
                    return result
                else:
                    if result.coin in self.cooldowns:
                        del self.cooldowns[result.coin]
                    if result.full_symbol in self.cooldowns:
                        del self.cooldowns[result.full_symbol]

        if result.status in ["api_error", "execution_error", "live_price_error"]:
            logger.info(f"   🚫 {result.coin} API tarafından daha önce reddedildi. 1 saat soğumaya alınıyor.")
            if not hasattr(self, 'cooldowns'):
                 self.cooldowns = {}
            self.cooldowns[result.coin] = datetime.now(timezone.utc) + timedelta(hours=1)
            self.cooldowns[result.full_symbol] = datetime.now(timezone.utc) + timedelta(hours=1)
            return result

        if result.status != "ready" or result.ml_result is None:
            return result

        direction = result.ml_result.decision.value
        if direction == "WAIT":
            return result

        try:
            try:
                exchange = self.executor._get_exchange() 
                ticker = exchange.fetch_ticker(result.full_symbol)
                live_price = ticker['last']
                
                if live_price is None or live_price <= 0:
                    raise ValueError(f"Geçersiz canlı fiyat: {live_price}")
                    
                current_balance = self.paper_trader.balance if self.dry_run else self._balance
                self.risk_manager.update_state(balance=current_balance)
                
                trade_calc = self.risk_manager.calculate_trade(
                    symbol      = result.full_symbol,
                    direction   = direction,
                    entry_price = live_price,
                    atr         = result.atr,
                )
                
                if not trade_calc.is_approved():
                    logger.warning(f"   ⚠️ {result.coin} yeni canlı fiyatla riskten geçemedi.")
                    result.status = "risk_rejected"
                    return result
                    
                result.price         = live_price
                result.sl_price      = trade_calc.stop_loss.price
                result.tp_price      = trade_calc.take_profit.price
                result.position_size = trade_calc.position_size
                result.leverage      = trade_calc.leverage
                result.risk_reward   = trade_calc.risk_reward
                
            except Exception as live_err:
                logger.error(f"   ❌ {result.coin} canlı fiyat çekme hatası: {live_err}")
                result.status = "live_price_error"
                result.error  = str(live_err)
                return result

            logger.info(
                f"\n💹 İŞLEM HAZIR: {result.coin} | {direction} @ ${result.price:,.4f} | "
                f"SL: ${result.sl_price:,.4f} | TP: ${result.tp_price:,.4f} | "
                f"Size: {result.position_size:.4f} | Lev: {result.leverage}x"
            )

            if self.dry_run:
                t = self.paper_trader.open_trade(
                    symbol        = result.coin,
                    full_symbol   = result.full_symbol,
                    direction     = direction,
                    entry_price   = result.price,          
                    stop_loss     = result.sl_price,        
                    take_profit   = result.tp_price,        
                    position_size = result.position_size,  
                    leverage      = result.leverage,
                    ic_confidence = result.ic_confidence,
                    ic_direction  = result.ic_direction,
                    best_timeframe= result.best_timeframe,
                    market_regime = result.market_regime,
                )
                result.trade_executed = True
                result.status = "executed"
                
                if self.notifier.is_configured():
                    msg = (f"🧪 <b>SANAL İŞLEM AÇILDI</b>\n"
                           f"━━━━━━━━━━━━━━━━━━━━━\n"
                           f"🪙 <b>Coin:</b> {result.coin}\n"
                           f"📈 <b>Yön:</b> {direction}\n"
                           f"💲 <b>Giriş:</b> ${result.price:,.4f}\n"
                           f"🛑 <b>SL:</b> ${result.sl_price:,.4f}\n"
                           f"🎯 <b>TP:</b> ${result.tp_price:,.4f}\n"
                           f"📊 <b>Kaldıraç:</b> {result.leverage}x")
                    self.notifier.send_message_sync(msg)

            else:
                try:
                    current_positions = self.executor.fetch_positions()
                    open_count = len(current_positions)
                    open_symbols = {p['symbol'] for p in current_positions if p.get('symbol')}
                except Exception as e:
                    result.status = "api_error"
                    logger.warning(f"   ⚠️ Binance API'ye ulaşılamadı ({e}). Risk almamak için işlem atlanıyor.")
                    return result

                if open_count >= MAX_OPEN_POSITIONS:
                    result.status = "position_limit"
                    logger.info(f"   ⚠️ {result.coin} atlandı: Canlı borsada max pozisyona ({MAX_OPEN_POSITIONS}) ulaşıldı.")
                    return result

                candidate_symbols = {
                    result.coin,
                    f"{result.coin}USDT",
                    f"{result.coin}/USDT",
                    result.full_symbol
                }

                if open_symbols & candidate_symbols:
                    result.status = "already_open"
                    logger.info(
                        f"   ⏭️  {result.coin} atlandı: Borsada zaten açık "
                        f"pozisyon mevcut → {open_symbols & candidate_symbols}"
                    )
                    return result

                try:
                    symbol_for_check  = result.full_symbol
                    has_existing_orders = self.executor.has_tp_sl_orders(
                        symbol_for_check
                    )
                except Exception:
                    has_existing_orders = False  

                exec_res = self.executor.execute_trade(
                    trade_calc,
                    skip_sl=has_existing_orders,  
                    skip_tp=has_existing_orders,  
                )

                if exec_res.success:
                    result.trade_executed = True
                    result.status = "executed"
                    
                    if self.notifier.is_configured():
                        msg = (f"🔴 <b>CANLI İŞLEM AÇILDI</b>\n"
                               f"━━━━━━━━━━━━━━━━━━━━━\n"
                               f"🪙 <b>Coin:</b> {result.coin}\n"
                               f"📈 <b>Yön:</b> {direction}\n"
                               f"💲 <b>Giriş:</b> ${exec_res.actual_entry:,.4f}\n"
                               f"🛑 <b>SL:</b> ${result.sl_price:,.4f}\n"
                               f"🎯 <b>TP:</b> ${result.tp_price:,.4f}\n"
                               f"📊 <b>Kaldıraç:</b> {result.leverage}x")
                        self.notifier.send_message_sync(msg)

                    try:
                        import pandas as pd
                        from pathlib import Path
                        from datetime import datetime
                        import uuid
                        
                        log_dir = Path("logs")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        excel_path = log_dir / "live_trades.xlsx"
                        
                        islem_hacmi = exec_res.actual_entry * result.position_size
                        
                        yeni_islem = {
                            "ID": str(uuid.uuid4())[:8],
                            "Tarih (Açılış)": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                            "Tarih (Kapanış)": "",
                            "Coin": result.coin,
                            "Yön": direction,
                            "Giriş ($)": exec_res.actual_entry,
                            "Çıkış ($)": "",
                            "Lot (Coin)": result.position_size,
                            "Hacim ($)": round(islem_hacmi, 2),
                            "Kaldıraç": result.leverage,
                            "SL ($)": result.sl_price,
                            "TP ($)": result.tp_price,
                            "R:R": getattr(result, 'risk_reward', 1.5),
                            "PnL ($)": 0.0,
                            "PnL (%)": 0.0,
                            "Fee ($)": 0.0,
                            "Net PnL ($)": 0.0,
                            "IC Güven": getattr(result, 'ic_confidence', 0.0),
                            "IC Yön": getattr(result, 'ic_direction', ''),
                            "TF": getattr(result, 'best_timeframe', ''),
                            "Rejim": getattr(result, 'market_regime', ''),
                            "AI Karar": direction,
                            "Durum": "open",
                            "Çıkış Nedeni": "",
                            "Süre (dk)": ""
                        }
                        
                        df_yeni = pd.DataFrame([yeni_islem])
                        
                        if excel_path.exists():
                            df_mevcut = pd.read_excel(excel_path)
                            df_son = pd.concat([df_mevcut, df_yeni], ignore_index=True)
                        else:
                            df_son = df_yeni
                            
                        df_son.to_excel(excel_path, index=False)
                        logger.info(f"📊 Canlı işlem Paper formatında Excel'e eklendi: {result.coin}")
                    except Exception as e:
                        logger.error(f"❌ Excel kayıt hatası: {e}")
                else:
                    result.status = "execution_error"
                    result.error  = exec_res.error
                    logger.error(f"   ❌ Canlı işlem açılamadı: {exec_res.error}")
                    if not hasattr(self, 'cooldowns'):
                        self.cooldowns = {}
                    self.cooldowns[result.coin] = datetime.now(timezone.utc) + timedelta(hours=1)
                    self.cooldowns[result.full_symbol] = datetime.now(timezone.utc) + timedelta(hours=1)
                    logger.info(f"   🚫 {result.coin} API tarafından reddedildiği için 1 saat soğumaya alındı.")

            if result.trade_executed:
                fv_dict = {}
                if result.ml_result and hasattr(result.ml_result, 'feature_vector'):
                    try:
                        fv_dict = result.ml_result.feature_vector.to_dict()
                    except Exception:
                        pass

                mem_rec = self.trade_memory.open_trade(
                    symbol           = result.full_symbol,
                    coin             = result.coin,
                    direction        = direction,
                    entry_price      = result.price,
                    sl_price         = result.sl_price,
                    tp_price         = result.tp_price,
                    timeframe        = result.best_timeframe,
                    ml_confidence    = getattr(result.ml_result, 'confidence', 0.0),
                    ml_direction     = direction,
                    ic_confidence    = result.ic_confidence,
                    ic_direction     = result.ic_direction,
                    market_regime    = result.market_regime,
                    validated_conf   = getattr(result.val_result, 'confidence', 0.0),
                    feature_snapshot = fv_dict,
                    position_size    = result.position_size,
                    leverage         = result.leverage,
                    risk_reward      = result.risk_reward,
                    atr              = result.atr,
                )
                if not result.paper_trade_id:
                    result.paper_trade_id = mem_rec.trade_id

        except Exception as e:
            result.status = "execution_error"
            result.error  = str(e)
            logger.error(f"❌ {result.coin} execution hatası: {e}", exc_info=True)

        return result

    def _check_open_positions(self):
        # Arka planda 30 saniyede bir kontrol yapılacağı için terminal kalabalığı yaratmaması adına log seviyesi debug yapıldı
        logger.debug("\n🔍 Açık pozisyonlar kontrol ediliyor...")
        
        if self.dry_run:
            from paper_trader import TradeStatus  
            open_trades_dict = self.paper_trader.open_trades 
            
            if not open_trades_dict:
                return

            closed_count = 0
            trades_to_check = list(open_trades_dict.values())
            
            for trade in trades_to_check:
                try:
                    fetch_symbol = trade.symbol if "USDT" in trade.symbol else f"{trade.symbol}/USDT"
                    ohlcv = self.fetcher.fetch_ohlcv(fetch_symbol, timeframe="15m", limit=3)
                    
                    if ohlcv is not None and not ohlcv.empty:
                        max_high = float(ohlcv['high'].max())
                        min_low = float(ohlcv['low'].min())
                        current_price = float(ohlcv['close'].iloc[-1])
                        
                        close_reason = None
                        exit_price = None
                        status = None
                        
                        if trade.direction == "LONG":
                            if min_low <= trade.stop_loss:
                                close_reason = "SL Hit"
                                exit_price = trade.stop_loss
                                status = TradeStatus.CLOSED_SL
                            elif max_high >= trade.take_profit:
                                close_reason = "TP Hit"
                                exit_price = trade.take_profit
                                status = TradeStatus.CLOSED_TP
                        else:
                            if max_high >= trade.stop_loss:
                                close_reason = "SL Hit"
                                exit_price = trade.stop_loss
                                status = TradeStatus.CLOSED_SL
                            elif min_low <= trade.take_profit:
                                close_reason = "TP Hit"
                                exit_price = trade.take_profit
                                status = TradeStatus.CLOSED_TP
                        
                        if close_reason:
                            if close_reason == "SL Hit":
                                self.cooldowns[trade.symbol] = datetime.now(timezone.utc) + timedelta(hours=2)
                                self.cooldowns[trade.full_symbol] = datetime.now(timezone.utc) + timedelta(hours=2)
                                logger.info(f"   ❄️ {trade.symbol} SL oldu! 2 saat soğuma süresine alındı.")
                            else:
                                logger.info(f"   🎯 {trade.symbol} {close_reason}! Hedefe ulaşıldı.")
                            
                            try:
                                if hasattr(self.executor, 'cancel_all_orders'):
                                    self.executor.cancel_all_orders(trade.full_symbol)
                                    logger.info(f"   🧹 {trade.symbol} için kalan emirler iptal edildi")
                            except Exception as cancel_err:
                                logger.warning(f"   ⚠️ {trade.symbol} emir iptal hatası: {cancel_err}")
                            
                            self.paper_trader._close_trade(trade, exit_price, status, close_reason)
                            closed_count += 1
                            logger.info(f"   ✅ {trade.symbol} işlemi kapandı! Neden: {close_reason} | Fiyat: ${exit_price:,.4f}")

                            if self.notifier.is_configured():
                                pnl_emoji = "✅ KÂR" if trade.net_pnl > 0 else "❌ ZARAR"
                                
                                msg = (f"{pnl_emoji} <b>({close_reason})</b>\n"
                                       f"━━━━━━━━━━━━━━━━━━━━━\n"
                                       f"🪙 <b>Coin:</b> {trade.symbol}\n"
                                       f"💲 <b>Çıkış:</b> ${exit_price:,.4f}\n"
                                       f"💰 <b>Net PnL:</b> ${trade.net_pnl:+.2f} ({trade.pnl_percent:+.2f}%)\n"
                                       f"🏦 <b>Güncel Kasa:</b> ${self.paper_trader.balance:,.2f}")
                                self.notifier.send_message_sync(msg)
                            
                            try:
                                matched = False
                                for mem_id, mem_trade in list(self.trade_memory.open_trades.items()):
                                    if mem_trade.symbol == trade.full_symbol or mem_trade.coin == trade.symbol:
                                        actual_pnl = trade.pnl_absolute if trade.pnl_absolute else 0.0
                                        self.trade_memory.close_trade(
                                            trade_id=mem_id,             
                                            exit_price=exit_price,       
                                            pnl=actual_pnl,              
                                            exit_reason=close_reason,    
                                        )
                                        matched = True
                                        logger.info(f"   🧠 Memory güncellendi: {trade.symbol} → {close_reason} | PnL: ${actual_pnl:+.2f}")
                                        break
                                if not matched:
                                    logger.warning(f"   ⚠️ Memory'de eşleşen trade bulunamadı: {trade.symbol} (full={trade.full_symbol})")
                            except Exception as em:
                                logger.warning(f"   ⚠️ Memory update BAŞARISIZ: {trade.symbol} → {em}")

                except Exception as e:
                    logger.error(f"   ❌ {trade.symbol} pozisyon kontrolünde hata: {e}")
            
            if closed_count > 0:
                logger.info(f"   Mevcut Bakiye: ${self.paper_trader.balance:.2f}")

        else:
            try:
                closed_count = 0
                
                if not hasattr(self.trade_memory, 'open_trades'):
                    return
                if not self.trade_memory.open_trades:
                    return
                
                try:
                    real_positions = self.executor.fetch_positions()
                except Exception as e:
                    logger.error(f"   ❌ Binance pozisyon çekme hatası: {e}")
                    return
                
                active_symbols = {p['symbol'] for p in real_positions if p.get('contracts', 0) != 0}
                
                for mem_id, mem_trade in list(self.trade_memory.open_trades.items()):
                    if mem_trade.symbol not in active_symbols:
                        logger.info(f"   🔍 {mem_trade.coin} pozisyonu kapalı bulundu → kapanış nedeni araştırılıyor...")
                        
                        close_reason = "Manuel / API"  
                        exit_price = mem_trade.entry_price  
                        
                        try:
                            ohlcv = self.fetcher.fetch_ohlcv(mem_trade.symbol, timeframe="15m", limit=5)
                            
                            if ohlcv is not None and not ohlcv.empty:
                                max_high = float(ohlcv['high'].max())
                                min_low = float(ohlcv['low'].min())
                                current_price = float(ohlcv['close'].iloc[-1])
                                
                                if mem_trade.direction == "LONG":
                                    if min_low <= mem_trade.sl_price:
                                        close_reason = "SL Hit"
                                        exit_price = mem_trade.sl_price
                                    elif max_high >= mem_trade.tp_price:
                                        close_reason = "TP Hit"
                                        exit_price = mem_trade.tp_price
                                    else:
                                        exit_price = current_price
                                
                                elif mem_trade.direction == "SHORT":
                                    if max_high >= mem_trade.sl_price:
                                        close_reason = "SL Hit"
                                        exit_price = mem_trade.sl_price
                                    elif min_low <= mem_trade.tp_price:
                                        close_reason = "TP Hit"
                                        exit_price = mem_trade.tp_price
                                    else:
                                        exit_price = current_price
                        
                        except Exception as ohlcv_err:
                            logger.warning(f"   ⚠️ {mem_trade.coin} OHLCV kontrol hatası: {ohlcv_err}")
                        
                        safe_entry = float(mem_trade.entry_price or 0)
                        safe_exit = float(exit_price or safe_entry)
                        safe_size = float(mem_trade.position_size or 0)
                        safe_leverage = int(mem_trade.leverage or 1)
                        
                        if mem_trade.direction == "LONG":
                            pnl_val = (safe_exit - safe_entry) * safe_size
                        else:
                            pnl_val = (safe_entry - safe_exit) * safe_size
                        
                        pnl_pct = (pnl_val / (safe_entry * safe_size)) * 100 * safe_leverage if (safe_entry * safe_size) > 0 else 0.0
                        
                        if close_reason == "SL Hit":
                            self.cooldowns[mem_trade.coin] = datetime.now(timezone.utc) + timedelta(hours=2)
                            self.cooldowns[mem_trade.symbol] = datetime.now(timezone.utc) + timedelta(hours=2)
                            logger.info(f"   ❄️ {mem_trade.coin} SL oldu! 2 saat soğuma süresine alındı.")
                        else:
                            logger.info(f"   🎯 {mem_trade.coin} {close_reason}! Hedefe ulaşıldı.")
                        
                        try:
                            if hasattr(self.executor, 'cancel_all_orders'):
                                self.executor.cancel_all_orders(mem_trade.symbol)
                                logger.info(f"   🧹 {mem_trade.coin} için kalan emirler iptal edildi")
                        except Exception as cancel_err:
                            logger.warning(f"   ⚠️ {mem_trade.coin} emir temizleme hatası: {cancel_err}")
                        
                        try:
                            self.trade_memory.close_trade(
                                trade_id    = mem_id,
                                exit_price  = safe_exit,
                                pnl         = pnl_val,
                                exit_reason = close_reason,
                            )
                            logger.info(f"   🧠 Memory güncellendi: {mem_trade.coin} → {close_reason} | PnL: ${pnl_val:+.2f}")
                        except Exception as mem_err:
                            logger.error(f"   ❌ Memory güncelleme hatası: {mem_trade.coin} → {mem_err}")

                        try:
                            if self.notifier.is_configured():
                                pnl_emoji = "✅ KÂR" if pnl_val > 0 else "❌ ZARAR"
                                msg = (f"{pnl_emoji} <b>({close_reason})</b>\n"
                                       f"━━━━━━━━━━━━━━━━━━━━━\n"
                                       f"🪙 <b>Coin:</b> {mem_trade.coin}\n"
                                       f"💲 <b>Çıkış:</b> ${safe_exit:,.4f}\n"
                                       f"💰 <b>PnL:</b> ${pnl_val:+.2f} ({pnl_pct:+.2f}%)\n"
                                       f"🏦 <b>Güncel Kasa:</b> ${self._balance:,.2f}")
                                self.notifier.send_message_sync(msg)
                        except Exception as tg_err:
                            logger.warning(f"   ⚠️ Telegram bildirimi gönderilemedi: {tg_err}")
                        
                        try:
                            from pathlib import Path
                            import pandas as pd
                            
                            log_dir = Path("logs")
                            excel_path = log_dir / "live_trades.xlsx"
                            
                            if excel_path.exists():
                                df = pd.read_excel(excel_path)
                                
                                for col in ['Tarih (Kapanış)', 'Çıkış Nedeni', 'Durum']:
                                    if col in df.columns:
                                        df[col] = df[col].astype(object)
                                        
                                for col in ['Çıkış ($)', 'PnL ($)', 'PnL (%)']:
                                    if col in df.columns:
                                        df[col] = df[col].astype(float)
                                
                                mask = (df['Coin'] == mem_trade.coin) & (df['Durum'] == 'open')
                                
                                if mask.any():
                                    idx = df[mask].index[-1]
                                    
                                    df.at[idx, 'Tarih (Kapanış)'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    df.at[idx, 'Çıkış ($)'] = safe_exit
                                    df.at[idx, 'Çıkış Nedeni'] = close_reason
                                    df.at[idx, 'PnL ($)'] = round(pnl_val, 4)
                                    df.at[idx, 'PnL (%)'] = round(pnl_pct, 2)
                                    df.at[idx, 'Durum'] = "closed"
                                    
                                    df.to_excel(excel_path, index=False)
                                    logger.info(f"   📊 Excel güncellendi: {mem_trade.coin} ({close_reason}) | PnL: ${pnl_val:+.2f}")
                                else:
                                    logger.warning(f"   ⚠️ Excel'de 'open' statüsünde {mem_trade.coin} kaydı bulunamadı!")
                        except Exception as e:
                            logger.error(f"   ❌ Excel güncelleme hatası: {e}", exc_info=True)
                                
                        closed_count += 1
                            
                if closed_count > 0:
                    logger.info(f"   🧹 {closed_count} işlemin kontrolü ve temizliği tamamlandı.")
                
                try:
                    exchange = self.executor._get_exchange()
                    all_open_orders = exchange.fetch_open_orders()
                    
                    if all_open_orders:
                        orphan_count = 0
                        for order in all_open_orders:
                            order_symbol = order.get('symbol', '')
                            order_id     = order.get('id', '?')
                            order_type   = order.get('type', '?')
                            
                            has_position = False
                            for act_sym in active_symbols:
                                if order_symbol == act_sym:
                                    has_position = True
                                    break
                            
                            if not has_position:
                                try:
                                    exchange.cancel_order(order_id, order_symbol)
                                    orphan_count += 1
                                    logger.info(f"   🗑️ Orphan emir silindi: {order_symbol} | {order_type} | ID: {order_id}")
                                except Exception as cancel_err:
                                    if 'Unknown order' in str(cancel_err) or 'UNKNOWN_ORDER' in str(cancel_err):
                                        logger.debug(f"   ℹ️ Orphan emir zaten yok: {order_id}")
                                    else:
                                        logger.warning(f"   ⚠️ Orphan emir silinemedi: {order_symbol} {order_id}: {cancel_err}")
                        
                        if orphan_count > 0:
                            logger.info(f"   🧹 {orphan_count} sahipsiz (orphan) emir temizlendi.")
                            
                except Exception as orphan_err:
                    logger.warning(f"   ⚠️ Orphan emir kontrolü başarısız (kritik değil): {orphan_err}")
                    
            except Exception as e:
                logger.error(f"   ❌ Canlı pozisyon kontrol/temizlik hatası: {e}", exc_info=True)

    def run_cycle(self) -> CycleReport:
        start  = time.time()
        report = CycleReport(timestamp=datetime.now(timezone.utc).isoformat())

        logger.info(f"\n{'═'*60}")
        logger.info(f"🔄 YENİ DÖNGÜ — {datetime.now().strftime('%H:%M:%S')} "
                    f"| v{VERSION} | {'PAPER' if self.dry_run else 'CANLI'}")
        logger.info(f"{'═'*60}")

        if self._check_kill_switch():
            report.status = CycleStatus.KILLED
            return report

        try:
            self._check_open_positions()

            logger.info(f"\n📡 Coin taraması (top {self.top_n})...")
            coins = self.scanner.scan(top_n=self.top_n)
            if not coins:
                report.status = CycleStatus.ERROR
                return report
            report.total_scanned = len(coins)

            self.validator._closed_trade_count = self.trade_memory._count_closed()
            self.lgbm_model._closed_trade_count = self.trade_memory._count_closed()

            logger.info(f"\n🔬 ML analizi ({len(coins)} coin)...")
            results = []
            
            if self.dry_run:
                open_coins = [trade.symbol for trade in self.paper_trader.open_trades.values()]
            else:
                open_coins = [mem_trade.coin for mem_trade in self.trade_memory.open_trades.values()]

            for c in coins:
                if c.coin in open_coins:
                    logger.info(f"   ⏭️ {c.coin} atlanıyor (Zaten açık pozisyon var)")
                    continue
                
                # UTC Cooldown Kontrolü
                if c.coin in self.cooldowns or c.symbol in self.cooldowns:
                    cooldown_key = c.coin if c.coin in self.cooldowns else c.symbol
                    if datetime.now(timezone.utc) < self.cooldowns[cooldown_key]:
                        kalan_dk = int((self.cooldowns[cooldown_key] - datetime.now(timezone.utc)).total_seconds() / 60)
                        logger.info(f"   ❄️ {c.coin} atlanıyor (Soğuma süresinde - Kalan: {kalan_dk} dk)")
                        continue
                    else:
                        if c.coin in self.cooldowns:
                            del self.cooldowns[c.coin]
                        if c.symbol in self.cooldowns:
                            del self.cooldowns[c.symbol]
                    
                r = self._analyze_coin(c.symbol, c.coin)
                
                results.append(r)
                report.total_analyzed += 1
                if r.ic_confidence >= IC_TRADE:
                    report.total_above_gate += 1

            logger.info(f"\n💹 Execution...")
            for r in results:
                if r.status == "ready":
                    r = self._execute_trade(r)
                    if r.trade_executed:
                        report.total_traded += 1

            report.coins        = results
            report.paper_balance= self.paper_trader.balance
            report.balance      = self._balance

            pt_stats = self.paper_trader.get_summary()
            total_closed = pt_stats.get("closed_trades", 0)
            real_win_rate = pt_stats.get("win_rate_pct", 0.0)

            retrain_threshold = 30
            current_retrain_count = getattr(self.lgbm_model, 'retrain_count', 0)
            
            if total_closed >= retrain_threshold:
                target_retrain_count = total_closed // retrain_threshold
                
                if target_retrain_count > current_retrain_count:
                    logger.info(f"\n🧠 [RETRAIN] {total_closed} kapalı işleme ulaşıldı! Model kendi tecrübelerinden yeniden eğitiliyor...")
                    try:
                        if hasattr(self, 'retrain_from_experience'):
                            self.retrain_from_experience() 
                    except Exception as e:
                        logger.error(f"Eğitim tetiklenemedi: {e}")
                    
                    self.lgbm_model.retrain_count = target_retrain_count

            next_target = ((total_closed // retrain_threshold) + 1) * retrain_threshold
            kalan_islem = next_target - total_closed

            report.ml_stats = {
                "closed_trades":  total_closed,
                "win_rate":       real_win_rate,
                "retrain_count":  getattr(self.lgbm_model, 'retrain_count', 0),
                "next_retrain_in": kalan_islem,
                "model_trained":  self.lgbm_model.is_trained,
            }
            report.status = (
                CycleStatus.SUCCESS   if report.total_traded > 0
                else CycleStatus.PARTIAL   if report.total_above_gate > 0
                else CycleStatus.NO_SIGNAL
            )
            self._consecutive_errors = 0

        except Exception as e:
            report.status = CycleStatus.ERROR
            report.errors.append(str(e))
            self._consecutive_errors += 1
            logger.error(f"❌ Döngü hatası: {e}", exc_info=True)

        report.elapsed = time.time() - start
        self._log_cycle_summary(report)
        return report

    def retrain_from_experience(self):
        logger.info("🧠 Kendi tecrübelerinden öğrenme (Retrain) başlatılıyor...")
        
        try:
            memory_file = self.trade_memory.log_dir / "ml_trade_memory.json"
            if not memory_file.exists():
                logger.warning("   ⚠️ Trade hafızası bulunamadı, standart BTC eğitimine dönülüyor.")
                return self.initial_train()
                
            import json
            with open(memory_file, 'r', encoding='utf-8') as f:
                trades = json.load(f)
                
            closed_trades = [t for t in trades if t.get('exit_reason') or t.get('pnl', 0) != 0]
            
            if len(closed_trades) < 30:
                logger.warning(f"   ⚠️ Yeterli kapalı işlem yok ({len(closed_trades)} < 30), standart eğitime dönülüyor.")
                return self.initial_train()

            rows_X = []
            rows_y = []
            
            for t in closed_trades:
                fv = t.get('feature_snapshot', {})
                if not fv:
                    continue
                    
                direction = t.get('direction', '')
                exit_reason = t.get('exit_reason', '')
                pnl = t.get('pnl', 0.0)
                
                won = (exit_reason == "TP Hit" or pnl > 0)
                
                if direction == "LONG":
                    target = 1 if won else 0
                elif direction == "SHORT":
                    target = 0 if won else 1
                else:
                    continue
                    
                rows_X.append(fv)
                rows_y.append(target)

            if len(rows_X) == 0:
                return False

            win_ratio = sum(rows_y) / len(rows_y) * 100
            logger.info(f"   📥 Hafızadan {len(rows_X)} adet gerçek işlem tecrübesi yüklendi. (Piyasa yönü=LONG oranı: %{win_ratio:.1f})")
            
            X_experience = pd.DataFrame(rows_X).replace([np.inf, -np.inf], np.nan)
            y_experience = pd.Series(rows_y)
            
            metrics = self.lgbm_model.train(X_experience, y_experience)
            
            logger.info(f"✅ Gerçek işlemlerden öğrenildi! Yeni Win Rate: %{metrics.accuracy*100:.1f}")
            return True

        except Exception as e:
            logger.error(f"❌ Tecrübeden öğrenme (Retrain) hatası: {e}", exc_info=True)
            return False
    
    def initial_train(self, symbol: str = "BTC/USDT:USDT") -> bool:
        logger.info(f"🎓 İlk eğitim: {symbol} 1h verisi kullanılıyor...")

        try:
            TRAIN_TF    = "1h"
            TRAIN_LIMIT = 1000

            df_raw = self.fetcher.fetch_ohlcv(symbol, TRAIN_TF, limit=TRAIN_LIMIT)
            if df_raw is None or len(df_raw) < 200:
                logger.error("❌ Yeterli veri çekilemedi")
                return False

            df_clean = self.preprocessor.full_pipeline(df_raw)
            if df_clean is None or len(df_clean) < 100:
                logger.error("❌ Preprocessing sonrası yetersiz veri")
                return False

            df_ind = self.calculator.calculate_all(df_clean)
            df_ind = self.calculator.add_forward_returns(df_ind, periods=[self.fwd_period])

            target_col = f'fwd_ret_{self.fwd_period}'
            if target_col not in df_ind.columns:
                logger.error(f"❌ Target kolon bulunamadı: {target_col}")
                return False

            class DummyAnalysis:
                def __init__(self, sym):
                    self.symbol          = sym
                    self.coin            = sym.split('/')[0]
                    self.price           = 0.0
                    self.change_24h      = 0.0
                    self.volume_24h      = 0.0
                    self.ic_confidence   = 65.0
                    self.ic_direction    = 'NEUTRAL'
                    self.significant_count = 10
                    self.market_regime   = 'trending'
                    self.category_tops   = {}              
                    self.tf_rankings     = []
                    self.atr             = 0.0
                    self.atr_pct         = 0.0
                    self.sl_price        = 0.0
                    self.tp_price        = 0.0
                    self.risk_reward     = 0.0
                    self.position_size   = 0.0
                    self.leverage        = 1

            analysis_stub = DummyAnalysis(symbol)
            rows_X = []
            rows_y = []

            train_cat_tops = {}                                    
            all_scores = self.selector.evaluate_all_indicators(df_ind, target_col)
            CATEGORIES = {
                'volume':  ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI'],
                'momentum':['RSI', 'Stoch', 'MFI', 'UO', 'MACD', 'PPO', 'ROC','TSI', 'CCI', 'Williams'],
                'trend':   ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST','Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'ATR', 'Keltner', 'Donchian', 'STD'],
            }
            for cat, cat_indicators in CATEGORIES.items():
                cat_scores = [s for s in all_scores if any(s.name.lower().startswith(ci.lower()) for ci in cat_indicators)]
                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    train_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}

            MIN_MOVE = 0.0025
            start_idx = 250
            end_idx   = len(df_ind) - self.fwd_period

            for i in range(start_idx, end_idx):
                fwd_val = df_ind[target_col].iloc[i]
                if pd.isna(fwd_val):
                    continue
                if abs(fwd_val) < MIN_MOVE:
                    continue

                target = 1 if fwd_val > 0 else 0
                df_slice = df_ind.iloc[max(0, i-50):i+1].copy()
                if len(df_slice) < 40:
                    continue

                analysis_stub.category_tops = {}
                for cat, cat_indicators in CATEGORIES.items():
                    cat_scores = [s for s in all_scores if any(s.name.lower().startswith(ci.lower()) for ci in cat_indicators)]
                    if not cat_scores:
                        cat_ind_lower = [x.lower() for x in cat_indicators]
                        matched = []
                        for s in all_scores:
                            s_lower = s.name.lower()
                            if not cat_scores:
                                for ci in cat_ind_lower:
                                    if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                        if len(ci) >= 3:
                                            matched.append(s)
                                            break
                        cat_scores = matched
                        
                    if cat_scores:
                        top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                        dynamic_cat_tops = {}
                        for cat, cat_indicators in CATEGORIES.items():
                            cat_scores = [s for s in all_scores if any(
                                s.name.lower().startswith(ci.lower()) or ci.lower() in s.name.lower()
                                for ci in cat_indicators
                            )]
                            if not cat_scores:
                                cat_ind_lower = [x.lower() for x in cat_indicators]
                                matched = []
                                for s in all_scores:
                                    s_lower = s.name.lower()
                                    if not cat_scores:
                                        continue
                                    for ci in cat_ind_lower:
                                        if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                            if len(ci) >= 3:
                                                matched.append(s)
                                                break
                                cat_scores = matched
                                
                            if cat_scores:
                                top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                                dynamic_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}
                    
                    analysis_stub.category_tops = dynamic_cat_tops

                fv = self.feature_eng.build_features(
                    analysis=analysis_stub,
                    ohlcv_df=df_slice
                )

                rows_X.append(fv.to_dict())
                rows_y.append(1 if target > 0 else 0)

            if len(rows_X) < 30:
                logger.error(f"❌ Yetersiz eğitim verisi: {len(rows_X)} < 30")
                return False

            n_total_raw = end_idx - start_idx
            n_filtered  = n_total_raw - len(rows_X)
            logger.info(
                f"  Eğitim Verisi : {len(rows_X)} satır × "
                f"{len(rows_X[0]) if rows_X else 0} feature | "
                f"WIN={sum(rows_y)/len(rows_y):.1%} | "
                f"Dead-band: {n_filtered}/{n_total_raw} bar çıkarıldı "
                f"({n_filtered/max(n_total_raw,1):.1%}) | "
                f"MIN_MOVE={MIN_MOVE:.3f}"
            )

            X = pd.DataFrame(rows_X).replace([np.inf, -np.inf], np.nan)
            y = pd.Series(rows_y)

            metrics = self.lgbm_model.train(X, y)
            
            try:
                from pathlib import Path
                
                report_dir = Path("logs/reports")
                report_dir.mkdir(parents=True, exist_ok=True)
                report_path = report_dir / "model_egitim_raporu.xlsx"
                
                with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                    df_metrics = pd.DataFrame([{
                        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Eğitim Satır Sayısı": len(X),
                        "Kazanma Oranı (Win Rate)": f"{y.mean():.1%}",
                        "Doğruluk (Accuracy)": metrics.accuracy,
                        "AUC Skoru": metrics.auc_roc,
                        "F1 Skoru": getattr(metrics, 'f1', 0.0)
                    }])
                    df_metrics.to_excel(writer, sheet_name="1_Genel_Metrikler", index=False)
                    
                    if hasattr(self.lgbm_model, 'model') and self.lgbm_model.model is not None:
                        importance = self.lgbm_model.model.feature_importances_
                        df_imp = pd.DataFrame({
                            "Feature (Kolon)": X.columns,
                            "Önem Puanı": importance
                        }).sort_values(by="Önem Puanı", ascending=False)
                        df_imp.to_excel(writer, sheet_name="2_Kolon_Onemleri", index=False)
                    
                    df_raw = X.copy()
                    df_raw['TARGET_SONUC'] = y.values
                    df_raw['TARGET_ACIKLAMA'] = df_raw['TARGET_SONUC'].apply(lambda x: "KÂR (1)" if x == 1 else "ZARAR (0)")
                    df_raw.to_excel(writer, sheet_name="3_Gecmis_Ham_Veri", index=False)
                    
                logger.info(f"📊 Detaylı Eğitim Raporu Excel olarak kaydedildi: {report_path}")
            except Exception as ex:
                logger.error(f"⚠️ Excel raporu oluşturulurken hata: {ex}")

            logger.info(f"✅ İlk eğitim tamamlandı | Metrik: AUC={metrics.auc_roc:.3f}, Acc={metrics.accuracy:.2f}")
            return True

        except Exception as e:
            logger.error(f"❌ İlk eğitim hatası: {e}", exc_info=True)
            return False

    def _log_cycle_summary(self, report: CycleReport) -> None:
        emoji = {"success":"✅","partial":"⚡","no_signal":"😴",
                 "error":"❌","killed":"⛔"}.get(report.status.value, "❓")
        logger.info(f"\n{'─'*50}")
        logger.info(f"  {emoji} Döngü | Taranan={report.total_scanned} "
                    f"IC-geçen={report.total_above_gate} "
                    f"İşlem={report.total_traded} "
                    f"Süre={report.elapsed:.1f}s")
        if report.ml_stats:
            s = report.ml_stats
            logger.info(f"  ML: eğitildi={s.get('model_trained')} | "
                        f"win={s.get('win_rate',0):.1f}% | "
                        f"retrain#{s.get('retrain_count',0)}")
        logger.info(f"{'─'*50}\n")

    def print_performance(self) -> None:
        PerformanceAnalyzer(self.paper_trader).print_report(
            PerformanceAnalyzer(self.paper_trader).full_analysis()
        )
        self.trade_memory.print_summary()


# =============================================================================
# SCHEDULER
# =============================================================================

# Varsayılan çalışma süresi 10 dakika olarak güncellendi
def run_scheduler(pipeline: MLTradingPipeline, interval_minutes: int = 10) -> None:
    pipeline._is_running = True

    def _stop(signum, frame):
        logger.info(f"\n🛑 Sinyal {signum} — durduruluyor...")
        pipeline._is_running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info(f"⏰ Scheduler: {interval_minutes}dk aralık")

    if not pipeline._init_balance():
        return

    if not pipeline.lgbm_model.is_trained:
        pipeline.initial_train()

    while pipeline._is_running:
        if pipeline._kill_switch:
            break
        if pipeline._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            time.sleep(ERROR_COOLDOWN_SECONDS)
            pipeline._consecutive_errors = 0
            continue

        report = pipeline.run_cycle()
        if report.status == CycleStatus.KILLED:
            break

        logger.info(f"⏰ Sonraki Tarama: {(datetime.now()+timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}")
        
        # Bot uyurken de açık işlemleri (TP/SL) takip etmesi için özel döngü
        for i in range(interval_minutes * 60):
            if not pipeline._is_running:
                break
            
            # Her 30 saniyede bir Binance API'yi yokla ve askıda kalanları temizle
            if i > 0 and i % 30 == 0:
                try:
                    pipeline._check_open_positions()
                except Exception:
                    pass
                    
            time.sleep(1)

    logger.info("🏁 Scheduler kapatıldı.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Crypto Bot v2.1.3")
    parser.add_argument("--live",     action="store_true", help="Canlı trade (ÖNERİLEN)")
    parser.add_argument("--top",      type=int, default=20, help="Top N coin")
    parser.add_argument("--schedule", action="store_true", help="Scheduler modu")
    # Interval varsayılan değeri 10 dakikaya düşürüldü
    parser.add_argument("-i","--interval", type=int, default=10, help="Aralık (dk)")
    parser.add_argument("--report",   action="store_true", help="Performans raporu")
    parser.add_argument("--train",    action="store_true", help="Sadece eğitim")
    parser.add_argument("--verbose",  action="store_true", help="Debug")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = MLTradingPipeline(dry_run=not args.live, top_n=args.top)

    if args.report:
        pipeline.print_performance()
        return

    if args.train:
        pipeline.initial_train()
        return

    if not pipeline._init_balance():
        sys.exit(1)

    if not pipeline.lgbm_model.is_trained:
        logger.info("🎓 Model eğitilmemiş — ilk eğitim başlıyor...")
        pipeline.initial_train()

    if args.schedule:
        run_scheduler(pipeline, args.interval)
    else:
        report = pipeline.run_cycle()
        logger.info(f"Döngü: {report.status.value}")


if __name__ == "__main__":
    main()