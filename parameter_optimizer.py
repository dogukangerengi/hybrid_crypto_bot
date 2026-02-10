# =============================================================================
# PARAMETER OPTIMIZER — Grid Search ile Parametre Optimizasyonu (ADIM 10)
# =============================================================================
# Bu modül strateji parametrelerini optimize eder.
# Paper trade sonuçlarını kullanarak en iyi parametre kombinasyonunu bulur.
#
# Optimize edilen parametreler:
# - IC Gate eşikleri (no_trade, full_trade)
# - Risk per trade (%)
# - Leverage limitleri
# - ATR çarpanı (SL mesafesi)
# - Risk/Reward minimum
# - Timeframe tercihleri
#
# Yöntemler:
# - Grid Search: Tüm kombinasyonları dene
# - Random Search: Rastgele örnekleme (büyük alanlar için)
# - Walk-Forward: Zaman serisi cross-validation
#
# Kullanım:
#   from parameter_optimizer import ParameterOptimizer
#   opt = ParameterOptimizer(paper_trader, historical_data)
#   best_params = opt.grid_search(param_grid)
#   opt.print_results()
#
# =============================================================================

import itertools                               # Parametre kombinasyonları için
import random                                  # Random search için
import logging                                 # Loglama
import json                                    # Sonuçları kaydetmek için
from datetime import datetime                  # Zaman damgaları
from typing import Dict, List, Optional, Tuple, Any, Callable  # Tip belirteçleri
from dataclasses import dataclass, field, asdict  # Yapılandırılmış veri
from pathlib import Path                       # Dosya yolları
from concurrent.futures import ProcessPoolExecutor, as_completed  # Paralel işleme
import copy                                    # Derin kopya için

import numpy as np                             # Sayısal hesaplamalar

# Logger yapılandırması
logger = logging.getLogger(__name__)


# =============================================================================
# SABİTLER
# =============================================================================

# Varsayılan parametre aralıkları
DEFAULT_PARAM_GRID = {
    # IC Gate eşikleri
    "ic_no_trade": [50, 55, 60],               # IC < bu değer → işlem yapma
    "ic_full_trade": [65, 70, 75, 80],         # IC > bu değer → tam işlem
    
    # Risk yönetimi
    "risk_per_trade_pct": [1.0, 1.5, 2.0, 2.5],  # Trade başına risk (%)
    "min_leverage": [2, 3, 5],                 # Minimum kaldıraç
    "max_leverage": [10, 15, 20],              # Maksimum kaldıraç
    
    # SL/TP parametreleri
    "atr_multiplier": [1.0, 1.5, 2.0],         # ATR × bu = SL mesafesi
    "min_risk_reward": [1.5, 2.0, 2.5],        # Minimum RR oranı
    
    # Kill switch
    "kill_switch_pct": [10, 15, 20],           # Drawdown eşiği (%)
}

# Optimizasyon hedefleri
OPTIMIZATION_TARGETS = [
    "total_return",                            # Toplam getiri (%)
    "sharpe_ratio",                            # Risk-adjusted getiri
    "profit_factor",                           # Kâr faktörü
    "win_rate",                                # Kazanma oranı (%)
    "calmar_ratio",                            # Getiri / Max DD
    "expectancy",                              # Beklenen değer ($)
]


# =============================================================================
# OPTİMİZASYON SONUÇ DATACLASS'I
# =============================================================================

@dataclass
class OptimizationResult:
    """
    Tek bir parametre kombinasyonunun sonucu.
    """
    # ---- Parametreler ----
    params: Dict[str, Any]                     # Test edilen parametreler
    
    # ---- Performans Metrikleri ----
    total_return: float = 0.0                  # Toplam getiri (%)
    sharpe_ratio: float = 0.0                  # Sharpe oranı
    sortino_ratio: float = 0.0                 # Sortino oranı
    profit_factor: float = 0.0                 # Kâr faktörü
    win_rate: float = 0.0                      # Kazanma oranı (%)
    max_drawdown: float = 0.0                  # Maksimum drawdown (%)
    calmar_ratio: float = 0.0                  # Calmar oranı
    expectancy: float = 0.0                    # Beklenti ($)
    
    # ---- Trade Detayları ----
    total_trades: int = 0                      # Toplam trade
    winning_trades: int = 0                    # Kazanan trade
    losing_trades: int = 0                     # Kaybeden trade
    avg_trade_pnl: float = 0.0                 # Ortalama trade PnL
    
    # ---- Meta ----
    run_time_seconds: float = 0.0             # Çalışma süresi
    timestamp: str = ""                        # Zaman damgası
    
    def get_score(self, target: str = "sharpe_ratio") -> float:
        """Belirtilen hedefe göre skor döndür."""
        return getattr(self, target, 0.0)


@dataclass
class OptimizationReport:
    """
    Tam optimizasyon raporu.
    """
    # ---- Meta ----
    start_time: str = ""                       # Başlangıç zamanı
    end_time: str = ""                         # Bitiş zamanı
    total_combinations: int = 0                # Test edilen kombinasyon sayısı
    optimization_target: str = ""              # Hedef metrik
    
    # ---- Sonuçlar ----
    all_results: List[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None
    worst_result: Optional[OptimizationResult] = None
    
    # ---- En İyi Parametreler ----
    best_params: Dict[str, Any] = field(default_factory=dict)
    
    # ---- İstatistikler ----
    avg_return: float = 0.0                    # Ortalama getiri
    avg_sharpe: float = 0.0                    # Ortalama Sharpe
    std_return: float = 0.0                    # Getiri std sapma
    
    # ---- Parametre Sensitivity ----
    param_sensitivity: Dict[str, Dict] = field(default_factory=dict)


# =============================================================================
# BACKTESTER (HIZLI SİMÜLASYON)
# =============================================================================

class QuickBacktester:
    """
    Parametre optimizasyonu için hızlı backtester.
    
    Tam pipeline'ı çalıştırmak yerine, mevcut trade sinyallerini
    farklı parametrelerle simüle eder.
    """

    def __init__(
        self,
        signals: List[Dict],                   # Trade sinyalleri [{symbol, direction, entry, atr, ic, ...}]
        initial_balance: float = 75.0,         # Başlangıç bakiyesi
    ):
        """
        Backtester'ı başlat.
        
        Parameters:
        ----------
        signals : list
            Trade sinyalleri listesi. Her sinyal şunları içermeli:
            - symbol: Coin sembolü
            - direction: 'LONG' veya 'SHORT'
            - entry_price: Giriş fiyatı
            - atr: ATR değeri
            - ic_confidence: IC skoru
            - high_after: Sonraki N bar'daki en yüksek fiyat
            - low_after: Sonraki N bar'daki en düşük fiyat
            - close_after: N bar sonraki kapanış
        initial_balance : float
            Simülasyon başlangıç bakiyesi
        """
        self.signals = signals
        self.initial_balance = initial_balance

    def run(self, params: Dict[str, Any]) -> OptimizationResult:
        """
        Belirtilen parametrelerle backtest çalıştır.
        
        Parameters:
        ----------
        params : dict
            Test edilecek parametreler
            
        Returns:
        -------
        OptimizationResult
            Backtest sonuçları
        """
        import time
        start_time = time.time()
        
        # Parametreleri çıkar
        ic_no_trade = params.get("ic_no_trade", 55)
        ic_full_trade = params.get("ic_full_trade", 70)
        risk_per_trade = params.get("risk_per_trade_pct", 2.0) / 100
        atr_mult = params.get("atr_multiplier", 1.5)
        min_rr = params.get("min_risk_reward", 1.5)
        min_lev = params.get("min_leverage", 2)
        max_lev = params.get("max_leverage", 20)
        kill_pct = params.get("kill_switch_pct", 15) / 100
        
        # Simülasyon değişkenleri
        balance = self.initial_balance
        initial = self.initial_balance
        peak_balance = balance
        
        trades = []
        total_pnl = 0.0
        wins = 0
        losses = 0
        
        for signal in self.signals:
            # Kill switch kontrolü
            drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
            if drawdown >= kill_pct:
                break
            
            # IC filtresi
            ic = signal.get("ic_confidence", 0)
            if ic < ic_no_trade:
                continue                       # IC çok düşük, atla
            
            # Trade mi report mu?
            if ic < ic_full_trade:
                continue                       # Sadece report, trade yok
            
            # Fiyat bilgileri
            entry = signal.get("entry_price", 0)
            atr = signal.get("atr", 0)
            direction = signal.get("direction", "LONG")
            high_after = signal.get("high_after", entry)
            low_after = signal.get("low_after", entry)
            
            if entry <= 0 or atr <= 0:
                continue
            
            # SL/TP hesapla
            sl_distance = atr * atr_mult
            tp_distance = sl_distance * min_rr
            
            if direction == "LONG":
                sl = entry - sl_distance
                tp = entry + tp_distance
            else:
                sl = entry + sl_distance
                tp = entry - tp_distance
            
            # Risk miktarı ve pozisyon büyüklüğü
            risk_amount = balance * risk_per_trade
            position_size = risk_amount / sl_distance
            position_value = entry * position_size
            
            # Kaldıraç hesapla
            required_margin = position_value / max_lev
            if required_margin > balance * 0.5:  # Max %50 margin
                continue                       # Yetersiz margin
            
            leverage = min(max_lev, max(min_lev, int(position_value / balance)))
            
            # ---- TRADE SİMÜLASYONU ----
            pnl = 0.0
            exit_price = entry
            exit_reason = "NONE"
            
            if direction == "LONG":
                # LONG: Önce SL mi TP mi tetiklendi?
                if low_after <= sl:
                    # SL tetiklendi
                    exit_price = sl
                    pnl = (sl - entry) * position_size
                    exit_reason = "SL"
                elif high_after >= tp:
                    # TP tetiklendi
                    exit_price = tp
                    pnl = (tp - entry) * position_size
                    exit_reason = "TP"
                else:
                    # Ne SL ne TP, periyod sonunda kapat
                    exit_price = signal.get("close_after", entry)
                    pnl = (exit_price - entry) * position_size
                    exit_reason = "TIMEOUT"
            else:
                # SHORT: Önce SL mi TP mi tetiklendi?
                if high_after >= sl:
                    # SL tetiklendi
                    exit_price = sl
                    pnl = (entry - sl) * position_size
                    exit_reason = "SL"
                elif low_after <= tp:
                    # TP tetiklendi
                    exit_price = tp
                    pnl = (entry - tp) * position_size
                    exit_reason = "TP"
                else:
                    # Ne SL ne TP, periyod sonunda kapat
                    exit_price = signal.get("close_after", entry)
                    pnl = (entry - exit_price) * position_size
                    exit_reason = "TIMEOUT"
            
            # Fee düş (%0.06 × 2)
            fee = position_value * 0.0006 * 2
            net_pnl = pnl - fee
            
            # Bakiyeyi güncelle
            balance += net_pnl
            total_pnl += net_pnl
            
            if balance > peak_balance:
                peak_balance = balance
            
            # Sayaçları güncelle
            if net_pnl > 0:
                wins += 1
            else:
                losses += 1
            
            trades.append({
                "pnl": net_pnl,
                "exit_reason": exit_reason,
            })
        
        # ---- METRİKLERİ HESAPLA ----
        total_trades = len(trades)
        
        result = OptimizationResult(
            params=params,
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            timestamp=datetime.now().isoformat(),
        )
        
        if total_trades == 0:
            result.run_time_seconds = time.time() - start_time
            return result
        
        # PnL listesi
        pnl_list = [t["pnl"] for t in trades]
        
        # Temel metrikler
        result.total_return = (balance - initial) / initial * 100
        result.win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        result.avg_trade_pnl = total_pnl / total_trades
        
        # Max Drawdown
        result.max_drawdown = (peak_balance - min(balance, *[
            initial + sum(pnl_list[:i+1]) for i in range(len(pnl_list))
        ])) / peak_balance * 100 if peak_balance > 0 else 0
        
        # Profit Factor
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe Ratio (basitleştirilmiş)
        if len(pnl_list) > 1:
            returns = np.array(pnl_list) / initial
            avg_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            result.sharpe_ratio = (avg_ret / std_ret * np.sqrt(365)) if std_ret > 0 else 0
            
            # Sortino (sadece downside)
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0:
                downside_std = np.std(neg_returns, ddof=1)
                result.sortino_ratio = (avg_ret / downside_std * np.sqrt(365)) if downside_std > 0 else 0
        
        # Calmar Ratio
        if result.max_drawdown > 0:
            result.calmar_ratio = result.total_return / result.max_drawdown
        
        # Expectancy
        avg_win = np.mean([p for p in pnl_list if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnl_list if p < 0]) if losses > 0 else 0
        result.expectancy = (result.win_rate/100 * avg_win) + ((1 - result.win_rate/100) * avg_loss)
        
        result.run_time_seconds = time.time() - start_time
        
        return result


# =============================================================================
# PARAMETER OPTIMIZER ANA SINIFI
# =============================================================================

class ParameterOptimizer:
    """
    Strateji parametrelerini optimize eden ana sınıf.
    """

    def __init__(
        self,
        signals: List[Dict],                   # Trade sinyalleri
        initial_balance: float = 75.0,         # Başlangıç bakiyesi
        output_dir: Optional[Path] = None,     # Sonuç kayıt dizini
    ):
        """
        Optimizer'ı başlat.
        
        Parameters:
        ----------
        signals : list
            Backtest için trade sinyalleri
        initial_balance : float
            Simülasyon başlangıç bakiyesi
        output_dir : Path, optional
            Sonuçların kaydedileceği dizin
        """
        self.signals = signals
        self.initial_balance = initial_balance
        self.output_dir = output_dir or Path(__file__).parent.parent / "logs" / "optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._report: Optional[OptimizationReport] = None
        
        logger.info(f"🔧 ParameterOptimizer başlatıldı | {len(signals)} sinyal")

    # =========================================================================
    # GRID SEARCH
    # =========================================================================

    def grid_search(
        self,
        param_grid: Dict[str, List] = None,
        target: str = "sharpe_ratio",
        n_jobs: int = 1,                       # Paralel işlem sayısı (1=sıralı)
        verbose: bool = True,
    ) -> OptimizationReport:
        """
        Grid Search ile tüm parametre kombinasyonlarını dene.
        
        Parameters:
        ----------
        param_grid : dict
            Parametre aralıkları {param_name: [values]}
        target : str
            Optimizasyon hedefi (sharpe_ratio, total_return, etc.)
        n_jobs : int
            Paralel işlem sayısı (1 = sıralı çalışma)
        verbose : bool
            İlerleme gösterimi
            
        Returns:
        -------
        OptimizationReport
            Optimizasyon sonuçları
        """
        import time
        start_time = time.time()
        
        param_grid = param_grid or DEFAULT_PARAM_GRID
        
        # Tüm kombinasyonları oluştur
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        total_combinations = len(combinations)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 GRID SEARCH BAŞLADI")
        logger.info(f"{'='*60}")
        logger.info(f"   Parametreler: {len(param_names)}")
        logger.info(f"   Kombinasyonlar: {total_combinations}")
        logger.info(f"   Hedef: {target}")
        logger.info(f"{'='*60}\n")
        
        # Rapor başlat
        report = OptimizationReport(
            start_time=datetime.now().isoformat(),
            total_combinations=total_combinations,
            optimization_target=target,
        )
        
        # Backtester oluştur
        backtester = QuickBacktester(self.signals, self.initial_balance)
        
        # Tüm kombinasyonları test et
        results = []
        
        for i, combo in enumerate(combinations):
            # Parametre dict'i oluştur
            params = dict(zip(param_names, combo))
            
            # Backtest çalıştır
            result = backtester.run(params)
            results.append(result)
            
            # İlerleme göster
            if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / total_combinations * 100
                best_so_far = max(results, key=lambda r: r.get_score(target))
                print(
                    f"   [{progress:>5.1f}%] {i+1}/{total_combinations} | "
                    f"En iyi {target}: {best_so_far.get_score(target):.3f}"
                )
        
        # Sonuçları sırala
        results.sort(key=lambda r: r.get_score(target), reverse=True)
        
        # Raporu doldur
        report.all_results = results
        report.best_result = results[0] if results else None
        report.worst_result = results[-1] if results else None
        report.best_params = results[0].params if results else {}
        report.end_time = datetime.now().isoformat()
        
        # İstatistikler
        if results:
            returns = [r.total_return for r in results]
            sharpes = [r.sharpe_ratio for r in results]
            report.avg_return = np.mean(returns)
            report.avg_sharpe = np.mean(sharpes)
            report.std_return = np.std(returns)
        
        # Parametre sensitivity analizi
        report.param_sensitivity = self._analyze_sensitivity(results, param_names, target)
        
        # Cache'le ve kaydet
        self._report = report
        self._save_report(report)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ GRID SEARCH TAMAMLANDI ({elapsed:.1f}s)")
        logger.info(f"{'='*60}")
        logger.info(f"   En iyi {target}: {report.best_result.get_score(target):.3f}")
        logger.info(f"   En iyi parametreler:")
        for k, v in report.best_params.items():
            logger.info(f"      {k}: {v}")
        logger.info(f"{'='*60}\n")
        
        return report

    # =========================================================================
    # RANDOM SEARCH
    # =========================================================================

    def random_search(
        self,
        param_grid: Dict[str, List] = None,
        n_iter: int = 100,                     # Deneme sayısı
        target: str = "sharpe_ratio",
        verbose: bool = True,
    ) -> OptimizationReport:
        """
        Random Search ile rastgele parametre örnekleri dene.
        
        Büyük parametre alanları için Grid Search'ten daha verimli.
        
        Parameters:
        ----------
        param_grid : dict
            Parametre aralıkları
        n_iter : int
            Deneme sayısı
        target : str
            Optimizasyon hedefi
        verbose : bool
            İlerleme gösterimi
            
        Returns:
        -------
        OptimizationReport
            Optimizasyon sonuçları
        """
        import time
        start_time = time.time()
        
        param_grid = param_grid or DEFAULT_PARAM_GRID
        param_names = list(param_grid.keys())
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎲 RANDOM SEARCH BAŞLADI")
        logger.info(f"{'='*60}")
        logger.info(f"   Denemeler: {n_iter}")
        logger.info(f"   Hedef: {target}")
        logger.info(f"{'='*60}\n")
        
        # Rapor başlat
        report = OptimizationReport(
            start_time=datetime.now().isoformat(),
            total_combinations=n_iter,
            optimization_target=target,
        )
        
        # Backtester
        backtester = QuickBacktester(self.signals, self.initial_balance)
        
        results = []
        seen_combos = set()
        
        for i in range(n_iter):
            # Rastgele kombinasyon seç (tekrar etmeden)
            while True:
                combo = tuple(random.choice(values) for values in param_grid.values())
                if combo not in seen_combos:
                    seen_combos.add(combo)
                    break
            
            params = dict(zip(param_names, combo))
            result = backtester.run(params)
            results.append(result)
            
            if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                progress = (i + 1) / n_iter * 100
                best_so_far = max(results, key=lambda r: r.get_score(target))
                print(
                    f"   [{progress:>5.1f}%] {i+1}/{n_iter} | "
                    f"En iyi {target}: {best_so_far.get_score(target):.3f}"
                )
        
        # Sonuçları sırala
        results.sort(key=lambda r: r.get_score(target), reverse=True)
        
        # Raporu doldur
        report.all_results = results
        report.best_result = results[0] if results else None
        report.worst_result = results[-1] if results else None
        report.best_params = results[0].params if results else {}
        report.end_time = datetime.now().isoformat()
        
        # İstatistikler
        if results:
            returns = [r.total_return for r in results]
            sharpes = [r.sharpe_ratio for r in results]
            report.avg_return = np.mean(returns)
            report.avg_sharpe = np.mean(sharpes)
            report.std_return = np.std(returns)
        
        # Sensitivity
        report.param_sensitivity = self._analyze_sensitivity(results, param_names, target)
        
        self._report = report
        self._save_report(report)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ RANDOM SEARCH TAMAMLANDI ({elapsed:.1f}s)")
        logger.info(f"   En iyi {target}: {report.best_result.get_score(target):.3f}")
        
        return report

    # =========================================================================
    # SENSITIVITY ANALİZİ
    # =========================================================================

    def _analyze_sensitivity(
        self,
        results: List[OptimizationResult],
        param_names: List[str],
        target: str
    ) -> Dict[str, Dict]:
        """
        Parametrelerin hedef metriğe etkisini analiz et.
        
        Her parametre için: hangi değer en iyi ortalama sonucu veriyor?
        """
        sensitivity = {}
        
        for param in param_names:
            # Bu parametrenin tüm değerlerini grupla
            value_scores = {}
            
            for result in results:
                value = result.params.get(param)
                score = result.get_score(target)
                
                if value not in value_scores:
                    value_scores[value] = []
                value_scores[value].append(score)
            
            # Her değer için ortalama hesapla
            value_avg = {
                v: np.mean(scores) for v, scores in value_scores.items()
            }
            
            # En iyi değeri bul
            best_value = max(value_avg.items(), key=lambda x: x[1])
            
            sensitivity[param] = {
                "values": list(value_avg.keys()),
                "avg_scores": list(value_avg.values()),
                "best_value": best_value[0],
                "best_avg_score": best_value[1],
                "importance": max(value_avg.values()) - min(value_avg.values()),  # Önem = aralık
            }
        
        return sensitivity

    # =========================================================================
    # RAPORLAMA
    # =========================================================================

    def print_report(self, report: Optional[OptimizationReport] = None) -> None:
        """Optimizasyon sonuçlarını konsola yazdır."""
        report = report or self._report
        
        if not report:
            print("⚠️ Önce grid_search() veya random_search() çalıştırın")
            return
        
        print(f"\n{'='*70}")
        print(f"🔧 OPTİMİZASYON RAPORU")
        print(f"{'='*70}")
        print(f"   Hedef: {report.optimization_target}")
        print(f"   Kombinasyonlar: {report.total_combinations}")
        print(f"   Süre: {report.start_time} → {report.end_time}")
        
        # En iyi sonuç
        if report.best_result:
            best = report.best_result
            print(f"\n{'─'*50}")
            print(f"🏆 EN İYİ SONUÇ")
            print(f"{'─'*50}")
            print(f"   Return:        {best.total_return:+.2f}%")
            print(f"   Sharpe:        {best.sharpe_ratio:.3f}")
            print(f"   Sortino:       {best.sortino_ratio:.3f}")
            print(f"   Profit Factor: {best.profit_factor:.2f}")
            print(f"   Win Rate:      {best.win_rate:.1f}%")
            print(f"   Max DD:        {best.max_drawdown:.1f}%")
            print(f"   Calmar:        {best.calmar_ratio:.2f}")
            print(f"   Trades:        {best.total_trades}")
            
            print(f"\n   📋 PARAMETRELER:")
            for k, v in best.params.items():
                print(f"      {k}: {v}")
        
        # Sensitivity analizi
        if report.param_sensitivity:
            print(f"\n{'─'*50}")
            print(f"📊 PARAMETRE ÖNEMİ (Sensitivity)")
            print(f"{'─'*50}")
            
            # Öneme göre sırala
            sorted_params = sorted(
                report.param_sensitivity.items(),
                key=lambda x: x[1]["importance"],
                reverse=True
            )
            
            for param, info in sorted_params:
                print(f"   {param}:")
                print(f"      En iyi: {info['best_value']} (avg={info['best_avg_score']:.3f})")
                print(f"      Önem:   {info['importance']:.3f}")
        
        # İstatistikler
        print(f"\n{'─'*50}")
        print(f"📈 GENEL İSTATİSTİKLER")
        print(f"{'─'*50}")
        print(f"   Ort. Return: {report.avg_return:.2f}%")
        print(f"   Ort. Sharpe: {report.avg_sharpe:.3f}")
        print(f"   Std Return:  {report.std_return:.2f}%")
        
        print(f"\n{'='*70}\n")

    def get_top_n(self, n: int = 10, target: str = None) -> List[OptimizationResult]:
        """En iyi N sonucu döndür."""
        if not self._report:
            return []
        
        target = target or self._report.optimization_target
        sorted_results = sorted(
            self._report.all_results,
            key=lambda r: r.get_score(target),
            reverse=True
        )
        return sorted_results[:n]

    def _save_report(self, report: OptimizationReport) -> None:
        """Raporu JSON dosyasına kaydet."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"optimization_{timestamp}.json"
        
        # Dataclass'ları dict'e çevir
        data = {
            "meta": {
                "start_time": report.start_time,
                "end_time": report.end_time,
                "total_combinations": report.total_combinations,
                "optimization_target": report.optimization_target,
            },
            "best_params": report.best_params,
            "best_result": asdict(report.best_result) if report.best_result else None,
            "statistics": {
                "avg_return": report.avg_return,
                "avg_sharpe": report.avg_sharpe,
                "std_return": report.std_return,
            },
            "param_sensitivity": report.param_sensitivity,
            "top_10_results": [asdict(r) for r in report.all_results[:10]],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Rapor kaydedildi: {filepath}")


# =============================================================================
# SINYAL ÜRETİCİ (TEST İÇİN)
# =============================================================================

def generate_sample_signals(n: int = 100, seed: int = 42) -> List[Dict]:
    """
    Test için örnek trade sinyalleri üret.
    
    Gerçek kullanımda bunlar IC analizinden gelecek.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    signals = []
    base_prices = {
        "BTC": 95000,
        "ETH": 3500,
        "SOL": 180,
        "DOGE": 0.35,
        "XRP": 2.5,
    }
    
    for i in range(n):
        symbol = random.choice(list(base_prices.keys()))
        base = base_prices[symbol]
        
        # Rastgele fiyat variasyonu
        entry = base * (1 + np.random.normal(0, 0.05))
        atr = entry * np.random.uniform(0.01, 0.04)  # %1-4 ATR
        
        direction = random.choice(["LONG", "SHORT"])
        ic = np.random.uniform(40, 90)         # IC 40-90 arası
        
        # Gelecekteki fiyatlar (simülasyon)
        # Gerçekte bunlar historical veri olacak
        volatility = atr * np.random.uniform(1, 3)
        
        if direction == "LONG":
            # LONG için yukarı bias
            high_after = entry + volatility * np.random.uniform(0.5, 2)
            low_after = entry - volatility * np.random.uniform(0.3, 1.5)
        else:
            # SHORT için aşağı bias
            high_after = entry + volatility * np.random.uniform(0.3, 1.5)
            low_after = entry - volatility * np.random.uniform(0.5, 2)
        
        close_after = (high_after + low_after) / 2 + np.random.normal(0, volatility * 0.3)
        
        signals.append({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "atr": atr,
            "ic_confidence": ic,
            "high_after": high_after,
            "low_after": low_after,
            "close_after": close_after,
            "market_regime": random.choice(["trending_up", "trending_down", "ranging"]),
        })
    
    return signals


# =============================================================================
# MODÜL TESTİ
# =============================================================================

if __name__ == "__main__":
    # Örnek kullanım
    print("🔧 Parameter Optimizer Test\n")
    
    # Örnek sinyaller üret
    signals = generate_sample_signals(n=200)
    print(f"   {len(signals)} örnek sinyal üretildi")
    
    # Optimizer oluştur
    optimizer = ParameterOptimizer(signals, initial_balance=75.0)
    
    # Küçük bir grid ile test
    small_grid = {
        "ic_no_trade": [50, 55],
        "ic_full_trade": [70, 75],
        "risk_per_trade_pct": [1.5, 2.0],
        "atr_multiplier": [1.0, 1.5],
        "min_risk_reward": [1.5, 2.0],
    }
    
    # Grid search
    report = optimizer.grid_search(small_grid, target="sharpe_ratio", verbose=True)
    
    # Sonuçları yazdır
    optimizer.print_report()
