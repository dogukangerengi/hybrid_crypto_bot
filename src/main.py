# =============================================================================
# MAIN.PY — ML-DRIVEN TRADING PIPELINE v2.1.0
# =============================================================================
# Gemini tamamen kaldırıldı → LightGBM pipeline entegre edildi.
#
# Gerçek API:
#   FeatureEngineer.build_features(analysis, ohlcv_df, all_tf_analyses) → MLFeatureVector
#   LGBMSignalModel.predict(fv, ic_score, ic_direction) → MLDecisionResult
#   SignalValidator.validate(fv, model, decision, confidence, ...) → ValidationResult
#   LGBMSignalModel.train(X, y) → ModelMetrics
#
# Çalıştırma:
#   python main.py              ← paper trade (varsayılan)
#   python main.py --live       ← canlı trade
#   python main.py --train      ← sadece eğitim
#   python main.py --report     ← performans raporu
#   python main.py --schedule   ← 75dk scheduler
# =============================================================================

import sys, os, time, signal, argparse, logging, traceback
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

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
from data import BitgetFetcher, DataPreprocessor
from indicators import IndicatorCalculator, IndicatorSelector
from execution import RiskManager, BitgetExecutor
from notifications import TelegramNotifier
from paper_trader import PaperTrader
from performance_analyzer import PerformanceAnalyzer

# ── ML modülleri (v2.0) ───────────────────────────────────────────────────────
from ml.feature_engineer import FeatureEngineer, MLFeatureVector
from ml.lgbm_model import LGBMSignalModel, MLDecisionResult
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

VERSION                = "2.1.0"
MAX_COINS_PER_CYCLE    = 20
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

IC_NO_TRADE = 55.0   # IC < bu → analizi atla, işlem yapma
IC_TRADE    = 60.0   # IC >= bu → ML pipeline'a gönder

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
    """
    Tek bir coin'in tüm analiz sonuçlarını tutan veri yapısı.
    FeatureEngineer bu nesneyi doğrudan kullanır.
    """
    # ── Kimlik ──
    coin:             str   = ""               # Kısa sembol: 'BTC'
    full_symbol:      str   = ""               # Tam sembol: 'BTC/USDT:USDT'
    price:            float = 0.0              # Son fiyat ($)
    change_24h:       float = 0.0             # 24h % değişim
    volume_24h:       float = 0.0             # 24h USDT hacim

    # ── IC Analiz (FeatureEngineer bu alanları okur) ──
    best_timeframe:   str   = ""              # En yüksek IC skorlu TF
    ic_confidence:    float = 0.0            # Composite IC skoru (0-100)
    ic_direction:     str   = ""            # 'LONG' / 'SHORT' / 'NEUTRAL'
    significant_count: int  = 0             # İstatistiksel anlamlı indikatör sayısı
    market_regime:    str   = ""            # 'trending' / 'ranging' / 'volatile'

    # ── FeatureEngineer için ek alanlar ──
    category_tops:    Dict  = field(default_factory=dict)
    # IC analizinin kategori bazlı en iyi indikatörleri
    # {'trend': {'name': 'EMA_20', 'ic': 0.15}, ...}
    tf_rankings:      List  = field(default_factory=list)
    # TF sıralama listesi — cross-TF feature'lar için
    # [{'tf': '1h', 'composite': 65, 'direction': 'LONG', 'sig_count': 10}, ...]

    # ── Risk ──
    atr:              float = 0.0
    atr_pct:          float = 0.0
    sl_price:         float = 0.0
    tp_price:         float = 0.0
    position_size:    float = 0.0
    leverage:         int   = 1
    risk_reward:      float = 0.0

    # ── ML Karar ──
    ml_result:        Optional[MLDecisionResult] = None
    val_result:       Optional[ValidationResult] = None
    ml_skipped:       bool  = False           # Model henüz eğitilmediyse True

    # ── Execution ──
    trade_executed:   bool  = False
    status:           str   = "pending"
    error:            str   = ""
    execution_result: Any   = None
    paper_trade_id:   str   = ""


@dataclass
class CycleReport:
    """Bir tarama döngüsünün özet raporu."""
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
    """
    LightGBM tabanlı trading pipeline.

    Adımlar:
    1. CoinScanner → Top N coin
    2. OHLCV + İndikatör + IC Analizi
    3. FeatureEngineer → MLFeatureVector
    4. LGBMSignalModel.predict() → MLDecisionResult
    5. SignalValidator.validate() → ValidationResult
    6. RiskManager → SL/TP/pozisyon
    7. Execution (paper veya canlı)
    8. TradeMemory → kayıt + retrain feedback loop
    9. Telegram bildirimi
    """

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

        # ── Mevcut modüller ──
        self.scanner      = CoinScanner()
        self.fetcher      = BitgetFetcher()
        self.preprocessor = DataPreprocessor()
        self.calculator   = IndicatorCalculator()
        self.selector     = IndicatorSelector(alpha=0.05)
        self.risk_manager = RiskManager()
        self.executor     = BitgetExecutor(dry_run=dry_run)
        self.notifier     = TelegramNotifier()
        self.paper_trader = PaperTrader()

        # ── ML modülleri ──
        self.feature_eng  = FeatureEngineer()     # IC + context → MLFeatureVector
        self.lgbm_model   = LGBMSignalModel()     # LightGBM model (train + predict)
        self.validator    = SignalValidator()      # Bootstrap CI + regime filter
        self.trade_memory = TradeMemory(
            log_dir = _root_dir / "logs"
        )                                          # Kalıcı trade hafızası

        # ── State ──
        self._balance          = 0.0
        self._initial_balance  = 0.0
        self._kill_switch      = False
        self._is_running       = False
        self._consecutive_errors = 0

        logger.info(f"🚀 ML Trading Pipeline v{VERSION} (dry_run={dry_run})")

    # =========================================================================
    # BAKIYE
    # =========================================================================

    def _init_balance(self) -> bool:
        """Başlangıç bakiyesini başlatır. Paper trade'de sabit değer kullanır."""
        try:
            if self.dry_run:
                self._balance = self._initial_balance = 75.0
                logger.info(f"💰 Paper bakiye: ${self._balance:.2f}")
            else:
                b = self.executor.fetch_balance()
                self._balance = self._initial_balance = b
                logger.info(f"💰 Canlı bakiye: ${self._balance:.2f}")
            return True
        except Exception as e:
            logger.error(f"❌ Bakiye hatası: {e}")
            return False

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def _check_kill_switch(self) -> bool:
        """Drawdown >= eşik ise tüm işlemleri durdurur."""
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

    # =========================================================================
    # REJİM TESPİTİ
    # =========================================================================

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """ADX bazlı piyasa rejimi: 'trending' / 'ranging' / 'volatile'"""
        try:
            if 'ADX_14' in df.columns:
                adx = df['ADX_14'].iloc[-1]
                if adx > 25: return 'trending'
                if adx > 15: return 'ranging'
                return 'volatile'
        except Exception:
            pass
        return 'unknown'

    # =========================================================================
    # TEK COİN ANALİZİ
    # =========================================================================

    def _analyze_coin(self, symbol: str, coin: str) -> CoinAnalysisResult:
        """
        Tek bir coin için tam ML pipeline çalıştırır.

        1. OHLCV veri çek (multi-TF)
        2. İndikatör hesapla
        3. IC analizi → composite skor + kategori bazlı top indikatörler
        4. IC eşiği kontrolü
        5. FeatureEngineer → MLFeatureVector
        6. LGBMSignalModel.predict() → MLDecisionResult
        7. SignalValidator.validate() → ValidationResult
        8. RiskManager → SL/TP/pozisyon
        """
        result = CoinAnalysisResult(coin=coin, full_symbol=symbol)

        try:
            # ── 1. Veri çek ─────────────────────────────────────────────────
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

            # ── 2. İndikatörler ──────────────────────────────────────────────
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

            # ── 3. IC Analizi ────────────────────────────────────────────────
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
                significant = [s for s in scores if s.is_significant]
                if not significant:
                    continue

                ic_mean = np.mean([abs(s.ic_mean) for s in significant]) * 100

                # TF sıralama listesi (cross-TF feature için)
                top_ic_val = max(significant, key=lambda s: abs(s.ic_mean))
                tf_dir     = "LONG" if top_ic_val.ic_mean > 0 else "SHORT"
                tf_rankings.append({
                    "tf":        tf,
                    "composite": round(ic_mean, 2),
                    "direction": tf_dir,
                    "sig_count": len(significant),
                    "top_ic":    round(top_ic_val.ic_mean, 4),
                })

                if ic_mean > best_ic:
                    best_ic     = ic_mean
                    best_tf     = tf
                    best_scores = scores

            if best_tf is None:
                result.status = "no_ic"; return result

            result.best_timeframe    = best_tf
            result.ic_confidence     = round(best_ic, 2)
            result.significant_count = sum(1 for s in best_scores if s.is_significant)
            result.tf_rankings       = sorted(tf_rankings, key=lambda x: -x["composite"])

            # IC yön tespiti
            sig_sorted = sorted(
                [s for s in best_scores if s.is_significant],
                key=lambda s: abs(s.ic_mean), reverse=True
            )
            result.ic_direction = (
                "LONG"  if sig_sorted and sig_sorted[0].ic_mean > 0
                else "SHORT" if sig_sorted
                else "NEUTRAL"
            )

            # Kategori bazlı top indikatörler (FeatureEngineer için)
            from indicators.categories import get_category_names, get_indicators_by_category
            category_tops = {}
            for cat in get_category_names():
                cat_indicators = {i['name'] for i in get_indicators_by_category(cat)}
                cat_scores = [s for s in best_scores
                              if s.name in cat_indicators and s.is_significant]
                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    category_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}
            result.category_tops = category_tops

            # ── 4. IC eşiği kontrolü ─────────────────────────────────────────
            if result.ic_confidence < IC_NO_TRADE:
                result.status = "low_ic"
                logger.debug(f"  ⏭ {coin}: IC={result.ic_confidence:.1f} < {IC_NO_TRADE}")
                return result

            # ── 5. Piyasa rejimi ─────────────────────────────────────────────
            best_df = indicator_data[best_tf]
            result.market_regime = self._detect_regime(best_df)

            # ── 6. Feature Engineering ───────────────────────────────────────
            try:
                fv = self.feature_eng.build_features(
                    analysis        = result,          # CoinAnalysisResult
                    ohlcv_df        = best_df,         # İndikatörlü DataFrame
                    all_tf_analyses = result.tf_rankings,  # TF listesi
                )
            except Exception as e:
                logger.warning(f"  ⚠️ {coin} FeatureEngineer hatası: {e}")
                result.status = "feature_error"; return result

            # ── 7. LightGBM Tahmini ──────────────────────────────────────────
            if not self.lgbm_model.is_trained:
                result.status    = "model_not_trained"
                result.ml_skipped = True
                logger.info(f"  ⏳ {coin}: Model henüz eğitilmedi → atlanıyor")
                return result

            try:
                ml_result = self.lgbm_model.predict(
                    feature_vector = fv,
                    ic_score       = result.ic_confidence,
                    ic_direction   = result.ic_direction,
                )
                result.ml_result = ml_result
            except Exception as e:
                logger.warning(f"  ⚠️ {coin} predict hatası: {e}")
                result.status = "predict_error"; return result

            # ── 8. İstatistiksel Doğrulama ───────────────────────────────────
            try:
                val_result = self.validator.validate(
                    feature_vector   = fv,
                    model            = self.lgbm_model,
                    model_decision   = ml_result.decision,
                    model_confidence = ml_result.confidence,
                    ic_direction     = result.ic_direction,
                    ic_score         = result.ic_confidence,
                    regime           = result.market_regime,
                )
                result.val_result = val_result
            except Exception as e:
                logger.warning(f"  ⚠️ {coin} validate hatası: {e}")
                result.status = "validate_error"; return result

            # ── 9. Risk Hesapla ──────────────────────────────────────────────
            if val_result.approved and str(ml_result.decision).upper() != "WAIT":
                direction = str(ml_result.decision).upper()
                try:
                    trade_calc = self.risk_manager.validate_trade(
                        symbol      = symbol,
                        direction   = direction,
                        entry_price = result.price,
                        balance     = self._balance,
                    )
                    if trade_calc and trade_calc.is_approved():
                        result.sl_price      = trade_calc.stop_loss.price
                        result.tp_price      = trade_calc.take_profit.price
                        result.position_size = trade_calc.position.size
                        result.leverage      = trade_calc.position.leverage
                        result.risk_reward   = trade_calc.take_profit.risk_reward
                        result.status        = "ready"
                    else:
                        result.status = "risk_rejected"
                        result.error  = ", ".join(getattr(trade_calc, 'rejection_reasons', []))
                except Exception as e:
                    result.status = "risk_error"
                    result.error  = str(e)
            else:
                result.status = "ml_rejected"
                result.error  = getattr(val_result, 'reason', "Doğrulama başarısız")

            # Özet log
            decision_str = str(ml_result.decision) if ml_result else "N/A"
            logger.info(
                f"  🔬 {coin:8} | IC={result.ic_confidence:.0f} | "
                f"{'✅' if val_result and val_result.approved else '❌'} "
                f"ML={decision_str} | Rejim={result.market_regime}"
            )

        except Exception as e:
            result.status = "error"
            result.error  = str(e)
            logger.error(f"  ❌ {coin} analiz hatası: {e}", exc_info=True)

        return result

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    def _execute_trade(self, result: CoinAnalysisResult) -> CoinAnalysisResult:
        """
        Analiz sonucuna göre trade açar ve TradeMemory'ye kaydeder.
        Paper trade veya canlı işlem — dry_run flag'ine göre seçilir.
        """
        if result.status != "ready" or result.ml_result is None:
            return result

        direction = str(result.ml_result.decision).upper()
        if direction == "WAIT":
            return result

        try:
            if self.dry_run:
                paper_id = self.paper_trader.open_trade(
                    symbol        = result.full_symbol,
                    coin          = result.coin,
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
                result.paper_trade_id = paper_id
                result.trade_executed = True
                result.status         = "paper_executed"

            else:
                # Canlı: max pozisyon kontrolü
                try:
                    open_count = len(self.executor.fetch_positions())
                except Exception:
                    open_count = 0

                if open_count >= MAX_OPEN_POSITIONS:
                    result.status = "position_limit"
                    return result

                class _Adapter:
                    """BitgetExecutor'un beklediği arayüzü sağlar."""
                    def __init__(self, r, d):
                        self.symbol = r.full_symbol
                        self.direction = d
                        self.entry_price = r.price
                        self.rejection_reasons = []
                        class _Pos: size = r.position_size; leverage = r.leverage
                        class _Tgt:
                            def __init__(self, p): self.price = p
                        self.position   = _Pos()
                        self.stop_loss  = _Tgt(r.sl_price)
                        self.take_profit= _Tgt(r.tp_price)
                    def is_approved(self): return True

                exec_res = self.executor.execute_trade(_Adapter(result, direction))
                result.execution_result = exec_res

                if exec_res.success:
                    result.trade_executed = True
                    result.status = "executed"
                else:
                    result.status = "execution_error"
                    result.error  = exec_res.error

            # TradeMemory'ye kaydet
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

    # =========================================================================
    # AÇIK POZİSYON KONTROLÜ
    # =========================================================================

    def _check_open_positions(self) -> List:
        """
        PaperTrader'daki açık pozisyonları kontrol eder.
        SL/TP tetiklenmiş trade'leri kapatır ve TradeMemory'yi günceller.
        Kapanan her trade için retrain_if_ready() çağrılır.
        """
        if not self.paper_trader.open_trades:
            return []

        prices = {}
        for trade in self.paper_trader.open_trades.values():
            try:
                df = self.fetcher.fetch_ohlcv(trade.symbol, '1m', limit=2)
                if df is not None and len(df) > 0:
                    prices[trade.symbol] = float(df['close'].iloc[-1])
            except Exception:
                pass

        closed = self.paper_trader.check_exits(prices)

        for trade in closed:
            self.trade_memory.close_trade(
                trade_id    = trade.trade_id,
                exit_price  = trade.exit_price,
                pnl         = trade.net_pnl,
                exit_reason = trade.exit_reason or "SL_TP",
            )
            # Yeterli trade birikince modeli retrain et
            self.trade_memory.retrain_if_ready(self.lgbm_model)
            emoji = "✅" if trade.net_pnl > 0 else "❌"
            logger.info(f"{emoji} Kapandı: {trade.symbol} | PnL=${trade.net_pnl:+.2f}")

        return closed

    # =========================================================================
    # ANA DÖNGÜ
    # =========================================================================

    def run_cycle(self) -> CycleReport:
        """
        Tek bir tarama→analiz→execution döngüsü.
        Scheduler bu metodu periyodik olarak çağırır.
        """
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
            # Açık pozisyonları kontrol et (SL/TP kapanışları işle)
            self._check_open_positions()

            # Coin tarama
            logger.info(f"\n📡 Coin taraması (top {self.top_n})...")
            coins = self.scanner.scan(top_n=self.top_n)
            if not coins:
                report.status = CycleStatus.ERROR
                return report
            report.total_scanned = len(coins)

            # Her coin için ML analizi
            logger.info(f"\n🔬 ML analizi ({len(coins)} coin)...")
            results = []
            for c in coins:
                r = self._analyze_coin(c.symbol, c.coin)
                results.append(r)
                report.total_analyzed += 1
                if r.ic_confidence >= IC_TRADE:
                    report.total_above_gate += 1

            # Execution
            logger.info(f"\n💹 Execution...")
            for r in results:
                if r.status == "ready":
                    r = self._execute_trade(r)
                    if r.trade_executed:
                        report.total_traded += 1

            report.coins        = results
            report.paper_balance= self.paper_trader.balance
            report.balance      = self._balance

            # ML istatistikleri
            mem_stats = self.trade_memory.get_stats()
            report.ml_stats = {
                "closed_trades":  mem_stats["closed_trades"],
                "win_rate":       mem_stats["win_rate"],
                "retrain_count":  mem_stats["total_retrain_count"],
                "next_retrain_in":mem_stats["next_retrain_in"],
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

    # =========================================================================
    # İLK EĞİTİM (Walk-forward)
    # =========================================================================

    def initial_train(self, symbol: str = "BTC/USDT:USDT") -> bool:
        """
        Pipeline ilk başladığında LightGBM'i tarihsel veri ile eğitir.
        TradeMemory'de yeterli geçmiş yoksa bu metod çağrılır.

        BTC 1h verisini kullanır → feature colonlarını öğrenir → train().
        """
        logger.info(f"🎓 İlk eğitim: {symbol} 1h verisi kullanılıyor...")

        try:
            df_raw = self.fetcher.fetch_ohlcv(symbol, "1h", limit=500)
            if df_raw is None or len(df_raw) < 200:
                logger.error("❌ Yeterli veri çekilemedi")
                return False

            df_clean = self.preprocessor.full_pipeline(df_raw)
            df_ind   = self.calculator.calculate_all(df_clean)
            df_ind   = self.calculator.add_forward_returns(df_ind, periods=[self.fwd_period])
            df_ind   = df_ind.dropna()

            target_col   = f'fwd_ret_{self.fwd_period}'
            skip_cols    = {"open","high","low","close","volume",target_col}
            feature_cols = [c for c in df_ind.columns
                            if c not in skip_cols and not c.startswith("fwd_")]

            X = df_ind[feature_cols].replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            y = (df_ind[target_col] > 0).astype(int)  # Binary: fiyat artarsa 1

            logger.info(f"  Eğitim: {X.shape[0]}×{X.shape[1]} | WIN={y.mean():.1%}")

            metrics = self.lgbm_model.train(X, y)  # LightGBM eğit

            logger.info(f"✅ İlk eğitim tamamlandı | Metrik: {metrics}")
            return True

        except Exception as e:
            logger.error(f"❌ İlk eğitim hatası: {e}", exc_info=True)
            return False

    # =========================================================================
    # YARDIMCI
    # =========================================================================

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
                        f"win={s.get('win_rate',0):.0%} | "
                        f"retrain#{s.get('retrain_count',0)}")
        logger.info(f"{'─'*50}\n")

    def print_performance(self) -> None:
        """Paper trade performans raporunu konsola yazdırır."""
        PerformanceAnalyzer(self.paper_trader).print_report(
            PerformanceAnalyzer(self.paper_trader).full_analysis()
        )
        self.trade_memory.print_summary()


# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(pipeline: MLTradingPipeline, interval_minutes: int = 75) -> None:
    """Pipeline'ı belirli aralıklarla otomatik çalıştırır. Ctrl+C ile durur."""
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

        logger.info(f"⏰ Sonraki: {(datetime.now()+timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}")
        for _ in range(interval_minutes * 60):
            if not pipeline._is_running:
                break
            time.sleep(1)

    logger.info("🏁 Scheduler kapatıldı.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"ML Crypto Bot v{VERSION}")
    parser.add_argument("--live",     action="store_true", help="Canlı trade")
    parser.add_argument("--top",      type=int, default=15, help="Top N coin")
    parser.add_argument("--schedule", action="store_true", help="Scheduler modu")
    parser.add_argument("-i","--interval", type=int, default=75, help="Aralık (dk)")
    parser.add_argument("--report",   action="store_true", help="Performans raporu")
    parser.add_argument("--train",    action="store_true", help="Sadece eğitim")
    parser.add_argument("--verbose",  action="store_true", help="Debug çıktısı")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = MLTradingPipeline(dry_run=not args.live, top_n=args.top)

    if args.report:
        pipeline.print_performance(); return

    if args.train:
        pipeline.initial_train(); return

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
