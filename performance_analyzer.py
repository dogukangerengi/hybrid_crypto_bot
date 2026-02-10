# =============================================================================
# PERFORMANCE ANALYZER — Strateji Performans Analizi (ADIM 10)
# =============================================================================
# Bu modül paper trade sonuçlarını analiz eder ve detaylı metrikler üretir.
# Profesyonel quant fonları tarafından kullanılan standart metrikleri içerir.
#
# Metrikler:
# - Win Rate, Profit Factor
# - Sharpe Ratio, Sortino Ratio
# - Maximum Drawdown, Calmar Ratio
# - Expectancy (R-multiple)
# - Trade distribution analysis
# - Time-based performance (gün/saat)
# - Direction analysis (LONG vs SHORT)
# - Regime analysis (trending vs ranging)
#
# Kullanım:
#   from performance_analyzer import PerformanceAnalyzer
#   analyzer = PerformanceAnalyzer(paper_trader)
#   report = analyzer.full_analysis()
#   analyzer.print_report()
#   analyzer.plot_equity_curve()
#
# =============================================================================

import math                                    # Matematiksel hesaplamalar
import statistics                              # İstatistiksel fonksiyonlar
import logging                                 # Loglama
from datetime import datetime, timedelta      # Zaman hesaplamaları
from typing import Dict, List, Optional, Tuple, Any  # Tip belirteçleri
from dataclasses import dataclass, field      # Yapılandırılmış veri
from collections import defaultdict           # Varsayılan dict
from pathlib import Path                       # Dosya yolları

import numpy as np                             # Sayısal hesaplamalar

# Logger yapılandırması
logger = logging.getLogger(__name__)


# =============================================================================
# SABİTLER
# =============================================================================

# Risk-free rate (yıllık) - ABD 10 yıllık tahvil yaklaşık değeri
RISK_FREE_RATE = 0.045                         # %4.5 yıllık

# Yıllık gün sayısı (kripto 365/7/24 işlem görür)
TRADING_DAYS_PER_YEAR = 365

# Minimum trade sayısı (anlamlı istatistik için)
MIN_TRADES_FOR_STATS = 5


# =============================================================================
# ANALİZ SONUÇ DATACLASS'I
# =============================================================================

@dataclass
class PerformanceReport:
    """
    Tam performans analizi raporu.
    
    Tüm metrikleri tek bir objede toplar.
    """
    # ---- Zaman Bilgisi ----
    analysis_date: str = ""                    # Analiz tarihi
    period_start: str = ""                     # İlk trade tarihi
    period_end: str = ""                       # Son trade tarihi
    period_days: int = 0                       # Toplam gün sayısı
    
    # ---- Bakiye Metrikleri ----
    initial_balance: float = 0.0               # Başlangıç bakiyesi
    final_balance: float = 0.0                 # Son bakiye
    total_return_pct: float = 0.0              # Toplam getiri (%)
    total_pnl: float = 0.0                     # Toplam kar/zarar ($)
    total_fees: float = 0.0                    # Toplam ücretler ($)
    
    # ---- Trade Sayıları ----
    total_trades: int = 0                      # Toplam trade
    winning_trades: int = 0                    # Kazanan trade
    losing_trades: int = 0                     # Kaybeden trade
    breakeven_trades: int = 0                  # Başa baş trade
    
    # ---- Temel Oranlar ----
    win_rate: float = 0.0                      # Kazanma oranı (%)
    loss_rate: float = 0.0                     # Kaybetme oranı (%)
    profit_factor: float = 0.0                 # Kâr faktörü
    payoff_ratio: float = 0.0                  # Ödeme oranı (avg_win/avg_loss)
    
    # ---- PnL Metrikleri ----
    avg_pnl: float = 0.0                       # Ortalama PnL
    avg_win: float = 0.0                       # Ortalama kazanç
    avg_loss: float = 0.0                      # Ortalama kayıp
    max_win: float = 0.0                       # En büyük kazanç
    max_loss: float = 0.0                      # En büyük kayıp
    median_pnl: float = 0.0                    # Medyan PnL
    std_pnl: float = 0.0                       # PnL standart sapma
    
    # ---- Risk Metrikleri ----
    max_drawdown_pct: float = 0.0              # Maksimum drawdown (%)
    max_drawdown_abs: float = 0.0              # Maksimum drawdown ($)
    avg_drawdown: float = 0.0                  # Ortalama drawdown (%)
    max_consecutive_wins: int = 0              # Maksimum ardışık kazanç
    max_consecutive_losses: int = 0            # Maksimum ardışık kayıp
    
    # ---- Risk-Adjusted Metrikler ----
    sharpe_ratio: float = 0.0                  # Sharpe oranı (yıllık)
    sortino_ratio: float = 0.0                 # Sortino oranı (yıllık)
    calmar_ratio: float = 0.0                  # Calmar oranı
    expectancy: float = 0.0                    # Beklenti (R-multiple)
    expectancy_pct: float = 0.0                # Beklenti (%)
    
    # ---- Süre Metrikleri ----
    avg_trade_duration_min: float = 0.0        # Ortalama trade süresi (dk)
    avg_winning_duration: float = 0.0          # Kazanan trade ort. süresi
    avg_losing_duration: float = 0.0           # Kaybeden trade ort. süresi
    
    # ---- Yön Analizi ----
    long_trades: int = 0                       # LONG trade sayısı
    short_trades: int = 0                      # SHORT trade sayısı
    long_win_rate: float = 0.0                 # LONG win rate (%)
    short_win_rate: float = 0.0                # SHORT win rate (%)
    long_pnl: float = 0.0                      # LONG toplam PnL
    short_pnl: float = 0.0                     # SHORT toplam PnL
    
    # ---- Rejim Analizi ----
    trending_trades: int = 0                   # Trending rejimde trade
    ranging_trades: int = 0                    # Ranging rejimde trade
    trending_win_rate: float = 0.0             # Trending win rate (%)
    ranging_win_rate: float = 0.0              # Ranging win rate (%)
    
    # ---- Timeframe Analizi ----
    tf_performance: Dict[str, Dict] = field(default_factory=dict)  # TF bazlı performans
    
    # ---- IC Analizi ----
    ic_correlation: float = 0.0                # IC skoru ile PnL korelasyonu
    avg_ic_winners: float = 0.0                # Kazananların ort. IC'si
    avg_ic_losers: float = 0.0                 # Kaybedenlerin ort. IC'si
    
    # ---- Günlük Analiz ----
    best_day: str = ""                         # En iyi gün
    worst_day: str = ""                        # En kötü gün
    best_day_pnl: float = 0.0                  # En iyi gün PnL
    worst_day_pnl: float = 0.0                 # En kötü gün PnL
    profitable_days: int = 0                   # Kârlı gün sayısı
    losing_days: int = 0                       # Zararlı gün sayısı
    
    # ---- Saatlik Analiz ----
    best_hour: int = 0                         # En iyi saat (0-23)
    worst_hour: int = 0                        # En kötü saat
    hourly_performance: Dict[int, float] = field(default_factory=dict)  # Saat → PnL
    
    # ---- Equity Curve ----
    equity_curve: List[float] = field(default_factory=list)  # Bakiye geçmişi
    timestamps: List[str] = field(default_factory=list)       # Zaman damgaları


# =============================================================================
# PERFORMANCE ANALYZER SINIFI
# =============================================================================

class PerformanceAnalyzer:
    """
    Paper trade sonuçlarını analiz eden ana sınıf.
    
    PaperTrader objesini alır ve detaylı performans raporu üretir.
    """

    def __init__(self, paper_trader):
        """
        Analyzer'ı başlat.
        
        Parameters:
        ----------
        paper_trader : PaperTrader
            Analiz edilecek PaperTrader objesi
        """
        self.pt = paper_trader                 # PaperTrader referansı
        self._report: Optional[PerformanceReport] = None  # Cache'lenmiş rapor

    # =========================================================================
    # ANA ANALİZ
    # =========================================================================

    def full_analysis(self) -> PerformanceReport:
        """
        Tam performans analizi yap.
        
        Tüm metrikleri hesaplar ve PerformanceReport döndürür.
        
        Returns:
        -------
        PerformanceReport
            Tüm metrikleri içeren rapor objesi
        """
        report = PerformanceReport()
        
        # Kapalı trade yoksa boş rapor döndür
        if not self.pt.closed_trades:
            logger.warning("⚠️ Analiz için kapalı trade yok")
            report.analysis_date = datetime.now().isoformat()
            return report
        
        trades = self.pt.closed_trades
        
        # ---- ZAMAN BİLGİSİ ----
        report.analysis_date = datetime.now().isoformat()
        report.period_start = min(t.opened_at for t in trades)
        report.period_end = max(t.closed_at or t.opened_at for t in trades)
        
        start_dt = datetime.fromisoformat(report.period_start)
        end_dt = datetime.fromisoformat(report.period_end)
        report.period_days = max((end_dt - start_dt).days, 1)
        
        # ---- BAKİYE METRİKLERİ ----
        report.initial_balance = self.pt.initial_balance
        report.final_balance = self.pt.balance
        report.total_pnl = self.pt.total_pnl
        report.total_fees = self.pt.total_fees
        report.total_return_pct = (
            (report.final_balance - report.initial_balance) / 
            report.initial_balance * 100
        )
        
        # ---- TRADE SAYILARI ----
        report.total_trades = len(trades)
        report.winning_trades = sum(1 for t in trades if t.net_pnl and t.net_pnl > 0)
        report.losing_trades = sum(1 for t in trades if t.net_pnl and t.net_pnl < 0)
        report.breakeven_trades = report.total_trades - report.winning_trades - report.losing_trades
        
        # ---- TEMEL ORANLAR ----
        if report.total_trades > 0:
            report.win_rate = report.winning_trades / report.total_trades * 100
            report.loss_rate = report.losing_trades / report.total_trades * 100
        
        # PnL listeleri
        all_pnl = [t.net_pnl for t in trades if t.net_pnl is not None]
        winning_pnl = [p for p in all_pnl if p > 0]
        losing_pnl = [p for p in all_pnl if p < 0]
        
        # Profit Factor
        gross_profit = sum(winning_pnl) if winning_pnl else 0
        gross_loss = abs(sum(losing_pnl)) if losing_pnl else 0
        report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # ---- PNL METRİKLERİ ----
        if all_pnl:
            report.avg_pnl = statistics.mean(all_pnl)
            report.median_pnl = statistics.median(all_pnl)
            report.std_pnl = statistics.stdev(all_pnl) if len(all_pnl) > 1 else 0
            report.max_win = max(all_pnl)
            report.max_loss = min(all_pnl)
        
        if winning_pnl:
            report.avg_win = statistics.mean(winning_pnl)
        if losing_pnl:
            report.avg_loss = statistics.mean(losing_pnl)
        
        # Payoff Ratio (avg_win / |avg_loss|)
        if report.avg_loss != 0:
            report.payoff_ratio = abs(report.avg_win / report.avg_loss)
        
        # ---- RİSK METRİKLERİ ----
        report.max_drawdown_pct = self.pt.max_drawdown
        report.max_drawdown_abs = self.pt.peak_balance - min(
            self.pt.balance,
            min(t.net_pnl for t in trades if t.net_pnl) if trades else 0
        )
        
        # Ardışık kazanç/kayıp
        report.max_consecutive_wins, report.max_consecutive_losses = self._calc_consecutive_streaks(trades)
        
        # ---- RİSK-ADJUSTED METRİKLER ----
        report.sharpe_ratio = self._calc_sharpe_ratio(all_pnl, report.period_days)
        report.sortino_ratio = self._calc_sortino_ratio(all_pnl, report.period_days)
        
        if report.max_drawdown_pct > 0:
            annualized_return = report.total_return_pct * (365 / report.period_days)
            report.calmar_ratio = annualized_return / report.max_drawdown_pct
        
        # Expectancy
        report.expectancy = self._calc_expectancy(report.win_rate, report.avg_win, report.avg_loss)
        if report.initial_balance > 0:
            report.expectancy_pct = report.expectancy / report.initial_balance * 100
        
        # ---- SÜRE METRİKLERİ ----
        durations = [t.duration_minutes for t in trades if t.duration_minutes]
        if durations:
            report.avg_trade_duration_min = statistics.mean(durations)
        
        winning_durations = [t.duration_minutes for t in trades if t.net_pnl and t.net_pnl > 0 and t.duration_minutes]
        losing_durations = [t.duration_minutes for t in trades if t.net_pnl and t.net_pnl < 0 and t.duration_minutes]
        
        if winning_durations:
            report.avg_winning_duration = statistics.mean(winning_durations)
        if losing_durations:
            report.avg_losing_duration = statistics.mean(losing_durations)
        
        # ---- YÖN ANALİZİ ----
        self._analyze_direction(trades, report)
        
        # ---- REJİM ANALİZİ ----
        self._analyze_regime(trades, report)
        
        # ---- TIMEFRAME ANALİZİ ----
        self._analyze_timeframe(trades, report)
        
        # ---- IC ANALİZİ ----
        self._analyze_ic(trades, report)
        
        # ---- GÜNLÜK ANALİZ ----
        self._analyze_daily(trades, report)
        
        # ---- SAATLİK ANALİZ ----
        self._analyze_hourly(trades, report)
        
        # ---- EQUITY CURVE ----
        self._build_equity_curve(trades, report)
        
        # Cache'le
        self._report = report
        
        return report

    # =========================================================================
    # YARDIMCI HESAPLAMA METODLARİ
    # =========================================================================

    def _calc_consecutive_streaks(self, trades) -> Tuple[int, int]:
        """Maksimum ardışık kazanç ve kayıp sayısını hesapla."""
        max_wins = max_losses = 0
        current_wins = current_losses = 0
        
        for trade in trades:
            if trade.net_pnl and trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.net_pnl and trade.net_pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = current_losses = 0
        
        return max_wins, max_losses

    def _calc_sharpe_ratio(self, pnl_list: List[float], period_days: int) -> float:
        """
        Sharpe Ratio hesapla.
        
        Sharpe = (Ortalama Getiri - Risk-free Rate) / Standart Sapma
        Yıllık bazda normalize edilir.
        
        Parameters:
        ----------
        pnl_list : list
            Trade PnL'leri
        period_days : int
            Toplam gün sayısı
            
        Returns:
        -------
        float
            Yıllık Sharpe Ratio
        """
        if len(pnl_list) < MIN_TRADES_FOR_STATS:
            return 0.0
        
        # Günlük getiri varsayımı (ortalama 1 trade/gün)
        returns = np.array(pnl_list) / self.pt.initial_balance
        
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Günlük risk-free rate
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        
        # Günlük Sharpe
        daily_sharpe = (avg_return - daily_rf) / std_return
        
        # Yıllık Sharpe (√252 ile çarp, kripto için √365)
        annualized_sharpe = daily_sharpe * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return float(annualized_sharpe)

    def _calc_sortino_ratio(self, pnl_list: List[float], period_days: int) -> float:
        """
        Sortino Ratio hesapla.
        
        Sharpe'a benzer ama sadece downside volatiliteyi kullanır.
        Pozitif volatiliteyi cezalandırmaz.
        
        Returns:
        -------
        float
            Yıllık Sortino Ratio
        """
        if len(pnl_list) < MIN_TRADES_FOR_STATS:
            return 0.0
        
        returns = np.array(pnl_list) / self.pt.initial_balance
        
        avg_return = np.mean(returns)
        
        # Sadece negatif getiriler (downside)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')              # Hiç kayıp yok
        
        downside_std = np.std(negative_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        
        daily_sortino = (avg_return - daily_rf) / downside_std
        annualized_sortino = daily_sortino * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return float(annualized_sortino)

    def _calc_expectancy(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Beklenti (Expectancy) hesapla.
        
        E = (Win% × Avg Win) + (Loss% × Avg Loss)
        
        Pozitif beklenti = uzun vadede kârlı strateji
        
        Parameters:
        ----------
        win_rate : float
            Kazanma oranı (%)
        avg_win : float
            Ortalama kazanç ($)
        avg_loss : float
            Ortalama kayıp ($) - negatif değer
            
        Returns:
        -------
        float
            Trade başına beklenen PnL ($)
        """
        if win_rate == 0:
            return 0.0
        
        win_prob = win_rate / 100
        loss_prob = 1 - win_prob
        
        expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)
        
        return expectancy

    def _analyze_direction(self, trades, report: PerformanceReport) -> None:
        """Yön bazlı (LONG/SHORT) analiz."""
        long_trades = [t for t in trades if t.direction == "LONG"]
        short_trades = [t for t in trades if t.direction == "SHORT"]
        
        report.long_trades = len(long_trades)
        report.short_trades = len(short_trades)
        
        # LONG performans
        if long_trades:
            long_wins = sum(1 for t in long_trades if t.net_pnl and t.net_pnl > 0)
            report.long_win_rate = long_wins / len(long_trades) * 100
            report.long_pnl = sum(t.net_pnl for t in long_trades if t.net_pnl)
        
        # SHORT performans
        if short_trades:
            short_wins = sum(1 for t in short_trades if t.net_pnl and t.net_pnl > 0)
            report.short_win_rate = short_wins / len(short_trades) * 100
            report.short_pnl = sum(t.net_pnl for t in short_trades if t.net_pnl)

    def _analyze_regime(self, trades, report: PerformanceReport) -> None:
        """Piyasa rejimi bazlı analiz."""
        trending = [t for t in trades if 'trending' in (t.market_regime or '').lower()]
        ranging = [t for t in trades if 'ranging' in (t.market_regime or '').lower()]
        
        report.trending_trades = len(trending)
        report.ranging_trades = len(ranging)
        
        if trending:
            trending_wins = sum(1 for t in trending if t.net_pnl and t.net_pnl > 0)
            report.trending_win_rate = trending_wins / len(trending) * 100
        
        if ranging:
            ranging_wins = sum(1 for t in ranging if t.net_pnl and t.net_pnl > 0)
            report.ranging_win_rate = ranging_wins / len(ranging) * 100

    def _analyze_timeframe(self, trades, report: PerformanceReport) -> None:
        """Timeframe bazlı analiz."""
        tf_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        
        for trade in trades:
            tf = trade.best_timeframe or "unknown"
            tf_stats[tf]["trades"] += 1
            if trade.net_pnl and trade.net_pnl > 0:
                tf_stats[tf]["wins"] += 1
            if trade.net_pnl:
                tf_stats[tf]["pnl"] += trade.net_pnl
        
        report.tf_performance = {}
        for tf, stats in tf_stats.items():
            report.tf_performance[tf] = {
                "trades": stats["trades"],
                "wins": stats["wins"],
                "win_rate": stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0,
                "total_pnl": stats["pnl"],
                "avg_pnl": stats["pnl"] / stats["trades"] if stats["trades"] > 0 else 0,
            }

    def _analyze_ic(self, trades, report: PerformanceReport) -> None:
        """IC skoru analizi."""
        ic_pnl_pairs = [
            (t.ic_confidence, t.net_pnl)
            for t in trades
            if t.ic_confidence > 0 and t.net_pnl is not None
        ]
        
        if len(ic_pnl_pairs) < MIN_TRADES_FOR_STATS:
            return
        
        ics = [p[0] for p in ic_pnl_pairs]
        pnls = [p[1] for p in ic_pnl_pairs]
        
        # Korelasyon hesapla (Pearson)
        try:
            correlation = np.corrcoef(ics, pnls)[0, 1]
            report.ic_correlation = float(correlation) if not np.isnan(correlation) else 0.0
        except:
            report.ic_correlation = 0.0
        
        # Kazanan/kaybedenlerin IC ortalaması
        winners_ic = [t.ic_confidence for t in trades if t.net_pnl and t.net_pnl > 0]
        losers_ic = [t.ic_confidence for t in trades if t.net_pnl and t.net_pnl < 0]
        
        if winners_ic:
            report.avg_ic_winners = statistics.mean(winners_ic)
        if losers_ic:
            report.avg_ic_losers = statistics.mean(losers_ic)

    def _analyze_daily(self, trades, report: PerformanceReport) -> None:
        """Günlük performans analizi."""
        daily_pnl = defaultdict(float)
        
        for trade in trades:
            if trade.closed_at and trade.net_pnl:
                day = trade.closed_at[:10]     # YYYY-MM-DD
                daily_pnl[day] += trade.net_pnl
        
        if not daily_pnl:
            return
        
        # En iyi/kötü gün
        best_day = max(daily_pnl.items(), key=lambda x: x[1])
        worst_day = min(daily_pnl.items(), key=lambda x: x[1])
        
        report.best_day = best_day[0]
        report.best_day_pnl = best_day[1]
        report.worst_day = worst_day[0]
        report.worst_day_pnl = worst_day[1]
        
        # Kârlı/zararlı gün sayısı
        report.profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)
        report.losing_days = sum(1 for pnl in daily_pnl.values() if pnl < 0)

    def _analyze_hourly(self, trades, report: PerformanceReport) -> None:
        """Saatlik performans analizi."""
        hourly_pnl = defaultdict(float)
        
        for trade in trades:
            if trade.opened_at and trade.net_pnl:
                try:
                    hour = datetime.fromisoformat(trade.opened_at).hour
                    hourly_pnl[hour] += trade.net_pnl
                except:
                    pass
        
        if not hourly_pnl:
            return
        
        report.hourly_performance = dict(hourly_pnl)
        
        best_hour = max(hourly_pnl.items(), key=lambda x: x[1])
        worst_hour = min(hourly_pnl.items(), key=lambda x: x[1])
        
        report.best_hour = best_hour[0]
        report.worst_hour = worst_hour[0]

    def _build_equity_curve(self, trades, report: PerformanceReport) -> None:
        """Equity curve (bakiye geçmişi) oluştur."""
        # Trade'leri kapanış zamanına göre sırala
        sorted_trades = sorted(trades, key=lambda t: t.closed_at or t.opened_at)
        
        balance = self.pt.initial_balance
        report.equity_curve = [balance]
        report.timestamps = ["START"]
        
        for trade in sorted_trades:
            if trade.net_pnl is not None:
                balance += trade.net_pnl
                report.equity_curve.append(balance)
                report.timestamps.append(trade.closed_at or "")

    # =========================================================================
    # RAPORLAMA
    # =========================================================================

    def print_report(self, report: Optional[PerformanceReport] = None) -> None:
        """
        Detaylı performans raporunu konsola yazdır.
        
        Parameters:
        ----------
        report : PerformanceReport, optional
            Yazdırılacak rapor (None ise full_analysis çağrılır)
        """
        if report is None:
            report = self._report or self.full_analysis()
        
        print(f"\n{'='*70}")
        print(f"📊 PERFORMANS ANALİZ RAPORU")
        print(f"{'='*70}")
        print(f"📅 Analiz: {report.analysis_date[:19]}")
        print(f"📅 Period: {report.period_start[:10]} → {report.period_end[:10]} ({report.period_days} gün)")
        
        # ---- BAKİYE ----
        print(f"\n{'─'*40}")
        print(f"💰 BAKİYE")
        print(f"{'─'*40}")
        print(f"   Başlangıç:    ${report.initial_balance:,.2f}")
        print(f"   Son:          ${report.final_balance:,.2f}")
        print(f"   Toplam PnL:   ${report.total_pnl:+,.2f}")
        print(f"   Getiri:       {report.total_return_pct:+.2f}%")
        print(f"   Ücretler:     ${report.total_fees:.2f}")
        
        # ---- TRADE İSTATİSTİKLERİ ----
        print(f"\n{'─'*40}")
        print(f"📈 TRADE İSTATİSTİKLERİ")
        print(f"{'─'*40}")
        print(f"   Toplam:       {report.total_trades}")
        print(f"   Kazanan:      {report.winning_trades} ({report.win_rate:.1f}%)")
        print(f"   Kaybeden:     {report.losing_trades} ({report.loss_rate:.1f}%)")
        print(f"   Başa baş:     {report.breakeven_trades}")
        
        # ---- PERFORMANS METRİKLERİ ----
        print(f"\n{'─'*40}")
        print(f"📊 PERFORMANS METRİKLERİ")
        print(f"{'─'*40}")
        print(f"   Win Rate:        {report.win_rate:.1f}%")
        print(f"   Profit Factor:   {report.profit_factor:.2f}")
        print(f"   Payoff Ratio:    {report.payoff_ratio:.2f}")
        print(f"   Expectancy:      ${report.expectancy:+.2f} ({report.expectancy_pct:+.2f}%)")
        
        # ---- PNL DETAY ----
        print(f"\n{'─'*40}")
        print(f"💵 PnL DETAY")
        print(f"{'─'*40}")
        print(f"   Ortalama:     ${report.avg_pnl:+.2f}")
        print(f"   Medyan:       ${report.median_pnl:+.2f}")
        print(f"   Std Dev:      ${report.std_pnl:.2f}")
        print(f"   Ort. Kazanç:  ${report.avg_win:+.2f}")
        print(f"   Ort. Kayıp:   ${report.avg_loss:+.2f}")
        print(f"   Max Kazanç:   ${report.max_win:+.2f}")
        print(f"   Max Kayıp:    ${report.max_loss:+.2f}")
        
        # ---- RİSK METRİKLERİ ----
        print(f"\n{'─'*40}")
        print(f"⚠️ RİSK METRİKLERİ")
        print(f"{'─'*40}")
        print(f"   Max Drawdown:    {report.max_drawdown_pct:.1f}%")
        print(f"   Max Ardışık Win: {report.max_consecutive_wins}")
        print(f"   Max Ardışık Loss:{report.max_consecutive_losses}")
        
        # ---- RİSK-ADJUSTED ----
        print(f"\n{'─'*40}")
        print(f"📐 RİSK-ADJUSTED METRİKLER")
        print(f"{'─'*40}")
        print(f"   Sharpe Ratio:  {report.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {report.sortino_ratio:.2f}")
        print(f"   Calmar Ratio:  {report.calmar_ratio:.2f}")
        
        # ---- YÖN ANALİZİ ----
        print(f"\n{'─'*40}")
        print(f"🧭 YÖN ANALİZİ")
        print(f"{'─'*40}")
        print(f"   LONG:  {report.long_trades} trade, {report.long_win_rate:.1f}% WR, ${report.long_pnl:+.2f}")
        print(f"   SHORT: {report.short_trades} trade, {report.short_win_rate:.1f}% WR, ${report.short_pnl:+.2f}")
        
        # ---- REJİM ANALİZİ ----
        print(f"\n{'─'*40}")
        print(f"📊 REJİM ANALİZİ")
        print(f"{'─'*40}")
        print(f"   Trending: {report.trending_trades} trade, {report.trending_win_rate:.1f}% WR")
        print(f"   Ranging:  {report.ranging_trades} trade, {report.ranging_win_rate:.1f}% WR")
        
        # ---- TF ANALİZİ ----
        if report.tf_performance:
            print(f"\n{'─'*40}")
            print(f"⏱️ TIMEFRAME ANALİZİ")
            print(f"{'─'*40}")
            for tf, stats in sorted(report.tf_performance.items()):
                print(f"   {tf:>5}: {stats['trades']:>3} trade, {stats['win_rate']:>5.1f}% WR, ${stats['total_pnl']:>+8.2f}")
        
        # ---- IC ANALİZİ ----
        if report.ic_correlation != 0 or report.avg_ic_winners != 0:
            print(f"\n{'─'*40}")
            print(f"🔬 IC ANALİZİ")
            print(f"{'─'*40}")
            print(f"   IC-PnL Korelasyon: {report.ic_correlation:+.3f}")
            print(f"   Kazanan Ort. IC:   {report.avg_ic_winners:.1f}")
            print(f"   Kaybeden Ort. IC:  {report.avg_ic_losers:.1f}")
        
        # ---- GÜNLÜK ----
        if report.best_day:
            print(f"\n{'─'*40}")
            print(f"📅 GÜNLÜK ANALİZ")
            print(f"{'─'*40}")
            print(f"   En İyi Gün:    {report.best_day} (${report.best_day_pnl:+.2f})")
            print(f"   En Kötü Gün:   {report.worst_day} (${report.worst_day_pnl:+.2f})")
            print(f"   Kârlı Günler:  {report.profitable_days}")
            print(f"   Zararlı Günler:{report.losing_days}")
        
        # ---- SÜRE ----
        print(f"\n{'─'*40}")
        print(f"⏰ SÜRE ANALİZİ")
        print(f"{'─'*40}")
        print(f"   Ort. Trade Süresi:    {report.avg_trade_duration_min:.0f} dk")
        print(f"   Ort. Kazanan Süresi:  {report.avg_winning_duration:.0f} dk")
        print(f"   Ort. Kaybeden Süresi: {report.avg_losing_duration:.0f} dk")
        
        print(f"\n{'='*70}\n")

    def to_dict(self, report: Optional[PerformanceReport] = None) -> Dict[str, Any]:
        """Raporu sözlük formatına çevir (JSON/API için)."""
        if report is None:
            report = self._report or self.full_analysis()
        
        from dataclasses import asdict
        return asdict(report)

    def export_json(self, filepath: Path, report: Optional[PerformanceReport] = None) -> Path:
        """Raporu JSON dosyasına export et."""
        import json
        
        if report is None:
            report = self._report or self.full_analysis()
        
        data = self.to_dict(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Rapor export: {filepath}")
        return filepath


# =============================================================================
# MODÜL TESTİ
# =============================================================================

if __name__ == "__main__":
    # Basit test — PaperTrader'a bağımlı
    print("PerformanceAnalyzer modülü yüklendi.")
    print("Kullanım: analyzer = PerformanceAnalyzer(paper_trader)")
