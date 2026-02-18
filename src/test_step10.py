# =============================================================================
# TEST_STEP10.PY — Paper Trading + Optimizasyon Testleri (ADIM 10)
# =============================================================================
# Bu test dosyası Adım 10'daki tüm modülleri test eder:
# - PaperTrader (trade kayıtları)
# - PerformanceAnalyzer (metrik hesaplama)
# - ParameterOptimizer (grid search)
#
# Çalıştırma:
#   cd hybrid_crypto_bot/src
#   python test_step10.py
#
# =============================================================================

import sys                                     # Sistem çıkış kodu
import time                                    # Performans ölçümü
import tempfile                                # Geçici dizin
from pathlib import Path                       # Dosya yolları
from datetime import datetime, timedelta      # Zaman hesaplamaları

# =============================================================================
# TEST YARDIMCI FONKSİYONLARI
# =============================================================================

def run_test(test_num: int, test_name: str, test_func, skip_reason: str = None) -> bool:
    """Test çalıştırıcı wrapper."""
    print(f"\n{'─'*55}")
    print(f"  TEST {test_num:>2}: {test_name}")
    print(f"{'─'*55}")
    
    if skip_reason:
        print(f"  ⏭️  ATLANDI: {skip_reason}")
        return None
    
    start = time.time()
    try:
        test_func()
        elapsed = time.time() - start
        print(f"\n  ✅ BAŞARILI ({elapsed:.2f}s)")
        return True
    except AssertionError as e:
        elapsed = time.time() - start
        print(f"\n  ❌ BAŞARISIZ ({elapsed:.2f}s)")
        print(f"     Hata: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ HATA ({elapsed:.2f}s)")
        print(f"     {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 1: PAPER TRADER İMPORT
# =============================================================================

def test_01_paper_trader_import():
    """PaperTrader modülü import edilebiliyor mu?"""
    from paper_trader import (
        PaperTrader,
        PaperTrade,
        TradeStatus,
        TradeDirection
    )
    
    assert PaperTrader is not None, "PaperTrader import edilemedi"
    assert PaperTrade is not None, "PaperTrade import edilemedi"
    assert TradeStatus.OPEN.value == "open", "TradeStatus hatalı"
    assert TradeDirection.LONG.value == "LONG", "TradeDirection hatalı"
    
    print(f"  ✓ PaperTrader import başarılı")
    print(f"  ✓ PaperTrade dataclass mevcut")
    print(f"  ✓ TradeStatus enum'ları doğru")


# =============================================================================
# TEST 2: PAPER TRADE AÇMA
# =============================================================================

def test_02_open_trade():
    """Trade açma işlemi doğru çalışıyor mu?"""
    from paper_trader import PaperTrader, TradeStatus
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # Trade aç
        trade = pt.open_trade(
            symbol="BTC",
            full_symbol="BTC/USDT:USDT",
            direction="LONG",
            entry_price=95000.0,
            position_size=0.01,
            stop_loss=94000.0,
            take_profit=97000.0,
            leverage=5,
            ic_confidence=75.0,
            ic_direction="LONG",
            best_timeframe="4h",
            market_regime="trending_up",
        )
        
        # Doğrulamalar
        assert trade is not None, "Trade None"
        assert trade.trade_id != "", "Trade ID boş"
        assert trade.symbol == "BTC", f"Symbol hatalı: {trade.symbol}"
        assert trade.direction == "LONG", f"Direction hatalı: {trade.direction}"
        assert trade.entry_price == 95000.0, f"Entry hatalı: {trade.entry_price}"
        assert trade.status == TradeStatus.OPEN.value, f"Status hatalı: {trade.status}"
        
        # Hesaplamalar
        assert trade.position_value == 950.0, f"Position value hatalı: {trade.position_value}"
        assert trade.risk_amount == 10.0, f"Risk amount hatalı: {trade.risk_amount}"  # (95000-94000)*0.01
        assert trade.risk_reward == 2.0, f"RR hatalı: {trade.risk_reward}"  # 2000/1000
        
        # Koleksiyonlar
        assert len(pt.open_trades) == 1, "Open trades sayısı hatalı"
        assert trade.trade_id in pt.open_trades, "Trade open_trades'de yok"
        assert pt.total_trades == 1, "Total trades sayısı hatalı"
        
        print(f"  ✓ Trade açıldı: {trade.trade_id}")
        print(f"  ✓ Position value: ${trade.position_value:.2f}")
        print(f"  ✓ Risk: ${trade.risk_amount:.2f} | RR: {trade.risk_reward:.1f}")


# =============================================================================
# TEST 3: SL/TP SİMÜLASYONU
# =============================================================================

def test_03_sl_tp_simulation():
    """SL/TP tetikleme simülasyonu doğru çalışıyor mu?"""
    from paper_trader import PaperTrader, TradeStatus
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # ---- TEST 1: TP Tetikleme (LONG) ----
        trade1 = pt.open_trade(
            symbol="BTC",
            full_symbol="BTC/USDT:USDT",
            direction="LONG",
            entry_price=95000.0,
            position_size=0.01,
            stop_loss=94000.0,
            take_profit=97000.0,
            leverage=5,
            ic_confidence=75,
            ic_direction="LONG",
            best_timeframe="4h",
            market_regime="trending_up",
        )
        
        # Fiyat TP'nin üzerine çıktı
        closed = pt.check_exits({"BTC": 97500.0})
        
        assert len(closed) == 1, "TP tetiklenmedi"
        assert closed[0].status == TradeStatus.CLOSED_TP.value, f"Status hatalı: {closed[0].status}"
        assert closed[0].exit_price == 97000.0, f"Exit price hatalı: {closed[0].exit_price}"
        assert closed[0].net_pnl > 0, f"PnL negatif olmamalı: {closed[0].net_pnl}"
        
        print(f"  ✓ LONG TP tetiklendi: Exit=${closed[0].exit_price}, PnL=${closed[0].net_pnl:.2f}")
        
        # ---- TEST 2: SL Tetikleme (SHORT) ----
        trade2 = pt.open_trade(
            symbol="ETH",
            full_symbol="ETH/USDT:USDT",
            direction="SHORT",
            entry_price=3500.0,
            position_size=0.1,
            stop_loss=3600.0,  # SHORT için SL yukarıda
            take_profit=3300.0,  # SHORT için TP aşağıda
            leverage=5,
            ic_confidence=70,
            ic_direction="SHORT",
            best_timeframe="1h",
            market_regime="trending_down",
        )
        
        # Fiyat SL'nin üzerine çıktı
        closed = pt.check_exits({"ETH": 3650.0})
        
        assert len(closed) == 1, "SL tetiklenmedi"
        assert closed[0].status == TradeStatus.CLOSED_SL.value, f"Status hatalı: {closed[0].status}"
        assert closed[0].exit_price == 3600.0, f"Exit price hatalı: {closed[0].exit_price}"
        assert closed[0].net_pnl < 0, f"SL'de PnL pozitif olamaz: {closed[0].net_pnl}"
        
        print(f"  ✓ SHORT SL tetiklendi: Exit=${closed[0].exit_price}, PnL=${closed[0].net_pnl:.2f}")
        
        # ---- BAKIYE KONTROLÜ ----
        assert pt.balance != 100.0, "Bakiye değişmemiş"
        assert len(pt.open_trades) == 0, "Açık trade kalmış"
        assert len(pt.closed_trades) == 2, f"Kapalı trade sayısı hatalı: {len(pt.closed_trades)}"
        
        print(f"  ✓ Bakiye güncellendi: ${pt.balance:.2f}")


# =============================================================================
# TEST 4: PNL HESAPLAMA
# =============================================================================

def test_04_pnl_calculation():
    """PnL hesaplamaları matematiksel olarak doğru mu?"""
    from paper_trader import PaperTrade, TradeStatus
    
    # ---- LONG TRADE ----
    long_trade = PaperTrade(
        trade_id="TEST001",
        symbol="SOL",
        full_symbol="SOL/USDT:USDT",
        direction="LONG",
        entry_price=180.0,
        position_size=1.0,
        position_value=180.0,
        leverage=5,
        stop_loss=175.0,
        take_profit=190.0,
        risk_amount=5.0,
        risk_reward=2.0,
        ic_confidence=80,
        ic_direction="LONG",
        best_timeframe="4h",
        market_regime="trending_up",
    )
    
    # Kârlı kapanış
    pnl_abs, pnl_pct = long_trade.calculate_pnl(190.0)
    
    # LONG: (exit - entry) × size = (190-180) × 1 = $10
    assert pnl_abs == 10.0, f"LONG PnL hatalı: {pnl_abs}"
    # Yüzde: (10/180) × 5 × 100 = 27.78%
    expected_pct = (10/180) * 5 * 100
    assert abs(pnl_pct - expected_pct) < 0.01, f"LONG PnL% hatalı: {pnl_pct}"
    
    print(f"  ✓ LONG kâr: ${pnl_abs:.2f} ({pnl_pct:.1f}%)")
    
    # ---- SHORT TRADE ----
    short_trade = PaperTrade(
        trade_id="TEST002",
        symbol="BTC",
        full_symbol="BTC/USDT:USDT",
        direction="SHORT",
        entry_price=95000.0,
        position_size=0.01,
        position_value=950.0,
        leverage=5,
        stop_loss=96000.0,
        take_profit=93000.0,
        risk_amount=10.0,
        risk_reward=2.0,
        ic_confidence=75,
        ic_direction="SHORT",
        best_timeframe="4h",
        market_regime="trending_down",
    )
    
    # Kârlı kapanış (fiyat düştü)
    pnl_abs, pnl_pct = short_trade.calculate_pnl(93000.0)
    
    # SHORT: (entry - exit) × size = (95000-93000) × 0.01 = $20
    assert pnl_abs == 20.0, f"SHORT PnL hatalı: {pnl_abs}"
    
    print(f"  ✓ SHORT kâr: ${pnl_abs:.2f} ({pnl_pct:.1f}%)")
    
    # ---- FEE DÜŞÜMÜ ----
    short_trade.close(93000.0, TradeStatus.CLOSED_TP, "TP hit", fee_rate=0.0006)
    
    # Fee = 950 × 0.0006 × 2 = $1.14
    expected_fee = 950 * 0.0006 * 2
    assert abs(short_trade.fees - expected_fee) < 0.01, f"Fee hatalı: {short_trade.fees}"
    
    # Net PnL = 20 - 1.14 = 18.86
    assert short_trade.net_pnl < short_trade.pnl_absolute, "Net PnL > Gross PnL olmamalı"
    
    print(f"  ✓ Fee: ${short_trade.fees:.2f}")
    print(f"  ✓ Net PnL: ${short_trade.net_pnl:.2f}")


# =============================================================================
# TEST 5: PERFORMANCE ANALYZER
# =============================================================================

def test_05_performance_analyzer():
    """PerformanceAnalyzer metrikleri doğru hesaplıyor mu?"""
    from paper_trader import PaperTrader
    from performance_analyzer import PerformanceAnalyzer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # Birkaç trade aç ve kapat
        trades_data = [
            ("BTC", "LONG", 95000, 0.01, 94000, 97000, 97500),  # TP
            ("ETH", "SHORT", 3500, 0.1, 3600, 3300, 3250),      # TP
            ("SOL", "LONG", 180, 1.0, 175, 190, 173),           # SL
            ("DOGE", "LONG", 0.35, 100, 0.33, 0.40, 0.42),      # TP
        ]
        
        for symbol, direction, entry, size, sl, tp, final_price in trades_data:
            trade = pt.open_trade(
                symbol=symbol,
                full_symbol=f"{symbol}/USDT:USDT",
                direction=direction,
                entry_price=entry,
                position_size=size,
                stop_loss=sl,
                take_profit=tp,
                leverage=5,
                ic_confidence=75,
                ic_direction=direction,
                best_timeframe="4h",
                market_regime="trending_up",
            )
            pt.check_exits({symbol: final_price})
        
        # Analyzer ile analiz et
        analyzer = PerformanceAnalyzer(pt)
        report = analyzer.full_analysis()
        
        # Doğrulamalar
        assert report is not None, "Report None"
        assert report.total_trades == 4, f"Toplam trade hatalı: {report.total_trades}"
        assert report.winning_trades == 3, f"Kazanan trade hatalı: {report.winning_trades}"
        assert report.losing_trades == 1, f"Kaybeden trade hatalı: {report.losing_trades}"
        assert report.win_rate == 75.0, f"Win rate hatalı: {report.win_rate}"
        
        # Profit factor > 1 olmalı (kârlı)
        assert report.profit_factor > 1, f"Profit factor hatalı: {report.profit_factor}"
        
        # Sharpe ratio hesaplanmış mı?
        assert isinstance(report.sharpe_ratio, float), "Sharpe ratio hesaplanmamış"
        
        print(f"  ✓ Toplam trade: {report.total_trades}")
        print(f"  ✓ Win rate: {report.win_rate:.1f}%")
        print(f"  ✓ Profit factor: {report.profit_factor:.2f}")
        print(f"  ✓ Sharpe ratio: {report.sharpe_ratio:.2f}")
        print(f"  ✓ Max drawdown: {report.max_drawdown_pct:.1f}%")
        print(f"  ✓ Total return: {report.total_return_pct:+.2f}%")


# =============================================================================
# TEST 6: DIRECTION ANALYSIS
# =============================================================================

def test_06_direction_analysis():
    """Yön bazlı analiz doğru mu?"""
    from paper_trader import PaperTrader
    from performance_analyzer import PerformanceAnalyzer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # 2 LONG, 2 SHORT
        # LONG'lar: 1 win, 1 loss
        # SHORT'lar: 2 win
        
        # LONG WIN
        pt.open_trade("BTC", "BTC/USDT:USDT", "LONG", 95000, 0.01, 94000, 97000, 5, 75, "LONG", "4h", "trending_up")
        pt.check_exits({"BTC": 97500})
        
        # LONG LOSS
        pt.open_trade("ETH", "ETH/USDT:USDT", "LONG", 3500, 0.1, 3400, 3700, 5, 70, "LONG", "4h", "trending_up")
        pt.check_exits({"ETH": 3350})
        
        # SHORT WIN
        pt.open_trade("SOL", "SOL/USDT:USDT", "SHORT", 180, 1, 185, 170, 5, 72, "SHORT", "4h", "trending_down")
        pt.check_exits({"SOL": 168})
        
        # SHORT WIN
        pt.open_trade("DOGE", "DOGE/USDT:USDT", "SHORT", 0.35, 100, 0.38, 0.30, 5, 68, "SHORT", "4h", "trending_down")
        pt.check_exits({"DOGE": 0.29})
        
        # Analiz
        analyzer = PerformanceAnalyzer(pt)
        report = analyzer.full_analysis()
        
        # Doğrulamalar
        assert report.long_trades == 2, f"LONG trade sayısı hatalı: {report.long_trades}"
        assert report.short_trades == 2, f"SHORT trade sayısı hatalı: {report.short_trades}"
        assert report.long_win_rate == 50.0, f"LONG win rate hatalı: {report.long_win_rate}"
        assert report.short_win_rate == 100.0, f"SHORT win rate hatalı: {report.short_win_rate}"
        
        print(f"  ✓ LONG: {report.long_trades} trade, {report.long_win_rate:.0f}% WR, ${report.long_pnl:+.2f}")
        print(f"  ✓ SHORT: {report.short_trades} trade, {report.short_win_rate:.0f}% WR, ${report.short_pnl:+.2f}")


# =============================================================================
# TEST 7: CSV EXPORT
# =============================================================================

def test_07_csv_export():
    """CSV export çalışıyor mu?"""
    from paper_trader import PaperTrader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # Birkaç trade
        pt.open_trade("BTC", "BTC/USDT:USDT", "LONG", 95000, 0.01, 94000, 97000, 5, 75, "LONG", "4h", "trending_up")
        pt.check_exits({"BTC": 97500})
        
        pt.open_trade("ETH", "ETH/USDT:USDT", "SHORT", 3500, 0.1, 3600, 3300, 5, 70, "SHORT", "1h", "ranging")
        pt.check_exits({"ETH": 3250})
        
        # CSV export
        csv_path = pt.export_to_csv()
        
        assert csv_path.exists(), f"CSV dosyası oluşturulmadı: {csv_path}"
        
        # Dosyayı oku ve kontrol et
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2, f"CSV satır sayısı hatalı: {len(rows)}"
        assert rows[0]["symbol"] == "BTC", "İlk satır BTC olmalı"
        assert rows[1]["symbol"] == "ETH", "İkinci satır ETH olmalı"
        
        print(f"  ✓ CSV export: {csv_path}")
        print(f"  ✓ {len(rows)} satır yazıldı")


# =============================================================================
# TEST 8: PARAMETER OPTIMIZER IMPORT
# =============================================================================

def test_08_optimizer_import():
    """ParameterOptimizer modülü import edilebiliyor mu?"""
    from parameter_optimizer import (
        ParameterOptimizer,
        OptimizationResult,
        OptimizationReport,
        QuickBacktester,
        generate_sample_signals,
        DEFAULT_PARAM_GRID,
    )
    
    assert ParameterOptimizer is not None, "ParameterOptimizer import edilemedi"
    assert OptimizationResult is not None, "OptimizationResult import edilemedi"
    assert generate_sample_signals is not None, "generate_sample_signals import edilemedi"
    
    # Örnek sinyal üret
    signals = generate_sample_signals(n=10)
    assert len(signals) == 10, f"Sinyal sayısı hatalı: {len(signals)}"
    assert "symbol" in signals[0], "Sinyal formatı hatalı"
    assert "ic_confidence" in signals[0], "IC confidence eksik"
    
    print(f"  ✓ ParameterOptimizer import başarılı")
    print(f"  ✓ {len(DEFAULT_PARAM_GRID)} varsayılan parametre")
    print(f"  ✓ Örnek sinyal üretimi çalışıyor")


# =============================================================================
# TEST 9: QUICK BACKTESTER
# =============================================================================

def test_09_quick_backtester():
    """QuickBacktester doğru çalışıyor mu?"""
    from parameter_optimizer import QuickBacktester, generate_sample_signals
    
    # Örnek sinyaller
    signals = generate_sample_signals(n=50, seed=42)
    
    # Backtester oluştur
    bt = QuickBacktester(signals, initial_balance=100.0)
    
    # Varsayılan parametrelerle çalıştır
    params = {
        "ic_no_trade": 55,
        "ic_full_trade": 70,
        "risk_per_trade_pct": 2.0,
        "atr_multiplier": 1.5,
        "min_risk_reward": 1.5,
        "min_leverage": 2,
        "max_leverage": 20,
        "kill_switch_pct": 15,
    }
    
    result = bt.run(params)
    
    # Doğrulamalar
    assert result is not None, "Result None"
    assert result.params == params, "Params eşleşmiyor"
    assert isinstance(result.total_return, float), "total_return float değil"
    assert isinstance(result.sharpe_ratio, float), "sharpe_ratio float değil"
    assert result.run_time_seconds > 0, "Çalışma süresi 0"
    
    print(f"  ✓ Backtest çalıştı: {result.total_trades} trade")
    print(f"  ✓ Return: {result.total_return:+.2f}%")
    print(f"  ✓ Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  ✓ Win rate: {result.win_rate:.1f}%")
    print(f"  ✓ Süre: {result.run_time_seconds:.3f}s")


# =============================================================================
# TEST 10: GRID SEARCH
# =============================================================================

def test_10_grid_search():
    """Grid search doğru çalışıyor mu?"""
    from parameter_optimizer import ParameterOptimizer, generate_sample_signals
    
    # Örnek sinyaller
    signals = generate_sample_signals(n=100, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Optimizer oluştur
        optimizer = ParameterOptimizer(
            signals, 
            initial_balance=100.0,
            output_dir=Path(tmpdir)
        )
        
        # Küçük grid (hızlı test için)
        small_grid = {
            "ic_no_trade": [50, 55],
            "ic_full_trade": [70, 75],
            "risk_per_trade_pct": [2.0],
            "atr_multiplier": [1.5],
            "min_risk_reward": [1.5, 2.0],
        }
        
        # Grid search
        report = optimizer.grid_search(
            small_grid, 
            target="sharpe_ratio",
            verbose=False
        )
        
        # Doğrulamalar
        assert report is not None, "Report None"
        assert report.total_combinations == 8, f"Kombinasyon sayısı hatalı: {report.total_combinations}"  # 2×2×1×1×2
        assert report.best_result is not None, "Best result None"
        assert len(report.best_params) > 0, "Best params boş"
        assert len(report.all_results) == 8, f"Sonuç sayısı hatalı: {len(report.all_results)}"
        
        # Sensitivity analizi
        assert len(report.param_sensitivity) > 0, "Sensitivity analizi yok"
        
        print(f"  ✓ Grid search: {report.total_combinations} kombinasyon")
        print(f"  ✓ En iyi Sharpe: {report.best_result.sharpe_ratio:.3f}")
        print(f"  ✓ En iyi return: {report.best_result.total_return:+.2f}%")
        print(f"  ✓ En iyi parametreler:")
        for k, v in report.best_params.items():
            print(f"      {k}: {v}")


# =============================================================================
# TEST 11: RANDOM SEARCH
# =============================================================================

def test_11_random_search():
    """Random search doğru çalışıyor mu?"""
    from parameter_optimizer import ParameterOptimizer, generate_sample_signals
    
    signals = generate_sample_signals(n=100, seed=42)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        optimizer = ParameterOptimizer(
            signals, 
            initial_balance=100.0,
            output_dir=Path(tmpdir)
        )
        
        # Random search
        report = optimizer.random_search(
            n_iter=20,
            target="profit_factor",
            verbose=False
        )
        
        assert report is not None, "Report None"
        assert report.total_combinations == 20, f"Kombinasyon sayısı hatalı"
        assert report.best_result is not None, "Best result None"
        
        print(f"  ✓ Random search: {report.total_combinations} deneme")
        print(f"  ✓ En iyi profit factor: {report.best_result.profit_factor:.2f}")
        print(f"  ✓ En iyi return: {report.best_result.total_return:+.2f}%")


# =============================================================================
# TEST 12: KILL SWITCH (PAPER TRADER)
# =============================================================================

def test_12_kill_switch_simulation():
    """Paper trader kill switch doğru çalışıyor mu?"""
    from paper_trader import PaperTrader, TradeStatus
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # Birkaç zararlı trade yap
        for i in range(5):
            pt.open_trade(
                symbol=f"COIN{i}",
                full_symbol=f"COIN{i}/USDT:USDT",
                direction="LONG",
                entry_price=100.0,
                position_size=1.0,
                stop_loss=95.0,  # %5 SL
                take_profit=110.0,
                leverage=5,
                ic_confidence=75,
                ic_direction="LONG",
                best_timeframe="4h",
                market_regime="trending_up",
            )
            # SL tetikle
            pt.check_exits({f"COIN{i}": 93.0})
        
        # Bakiye düşmüş olmalı
        assert pt.balance < 100.0, "Bakiye düşmemiş"
        
        # Drawdown kontrolü
        drawdown = (pt.peak_balance - pt.balance) / pt.peak_balance * 100
        
        print(f"  ✓ Başlangıç: $100.00")
        print(f"  ✓ Güncel:    ${pt.balance:.2f}")
        print(f"  ✓ Drawdown:  {drawdown:.1f}%")
        print(f"  ✓ Max DD:    {pt.max_drawdown:.1f}%")
        
        # Kill switch all close testi
        pt.open_trade("TEST", "TEST/USDT:USDT", "LONG", 100, 0.1, 95, 110, 5, 70, "LONG", "4h", "ranging")
        closed = pt.close_all_trades({"TEST": 98.0}, reason="Test kill switch")
        
        assert len(closed) == 1, "Close all çalışmadı"
        assert closed[0].status == TradeStatus.CLOSED_KILL.value, "Kill status hatalı"
        
        print(f"  ✓ Kill switch close çalışıyor")


# =============================================================================
# TEST 13: JSON KAYIT/YÜKLEME
# =============================================================================

def test_13_json_persistence():
    """Trade logları JSON'a kaydedilip yükleniyor mu?"""
    from paper_trader import PaperTrader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # İlk session
        pt1 = PaperTrader(initial_balance=100.0, log_dir=log_dir)
        
        pt1.open_trade("BTC", "BTC/USDT:USDT", "LONG", 95000, 0.01, 94000, 97000, 5, 75, "LONG", "4h", "trending_up")
        pt1.check_exits({"BTC": 97500})
        
        pt1.open_trade("ETH", "ETH/USDT:USDT", "SHORT", 3500, 0.1, 3600, 3300, 5, 70, "SHORT", "1h", "ranging")
        # Bu trade açık kalsın
        
        balance_after = pt1.balance
        closed_count = len(pt1.closed_trades)
        open_count = len(pt1.open_trades)
        
        # Yeni session (aynı dizinden yükle)
        pt2 = PaperTrader(initial_balance=100.0, log_dir=log_dir)
        
        # Veriler yüklendi mi?
        assert pt2.balance == balance_after, f"Bakiye yüklenmedi: {pt2.balance} vs {balance_after}"
        assert len(pt2.closed_trades) == closed_count, f"Kapalı trade sayısı hatalı"
        assert len(pt2.open_trades) == open_count, f"Açık trade sayısı hatalı"
        
        print(f"  ✓ Bakiye persist: ${pt2.balance:.2f}")
        print(f"  ✓ Kapalı trade: {len(pt2.closed_trades)}")
        print(f"  ✓ Açık trade: {len(pt2.open_trades)}")


# =============================================================================
# TEST 14: ÖZET RAPOR
# =============================================================================

def test_14_summary_report():
    """Paper trader özet raporu doğru mu?"""
    from paper_trader import PaperTrader
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = PaperTrader(initial_balance=100.0, log_dir=Path(tmpdir))
        
        # Çeşitli tradeler
        trades = [
            ("BTC", "LONG", 95000, 0.01, 94000, 97000, 97500, True),   # Win
            ("ETH", "SHORT", 3500, 0.1, 3600, 3300, 3250, True),       # Win
            ("SOL", "LONG", 180, 1, 175, 190, 173, False),             # Loss
            ("DOGE", "LONG", 0.35, 100, 0.33, 0.40, 0.42, True),       # Win
        ]
        
        for symbol, direction, entry, size, sl, tp, final, is_win in trades:
            pt.open_trade(symbol, f"{symbol}/USDT:USDT", direction, entry, size, sl, tp, 5, 75, direction, "4h", "trending")
            pt.check_exits({symbol: final})
        
        # Özet al
        summary = pt.get_summary()
        
        # Doğrulamalar
        assert summary["total_trades"] == 4, "Total trades hatalı"
        assert summary["winning_trades"] == 3, "Winning trades hatalı"
        assert summary["losing_trades"] == 1, "Losing trades hatalı"
        assert summary["win_rate_pct"] == 75.0, "Win rate hatalı"
        assert summary["profit_factor"] > 1, "Profit factor hatalı"
        
        print(f"  ✓ Toplam: {summary['total_trades']} trade")
        print(f"  ✓ Win rate: {summary['win_rate_pct']:.1f}%")
        print(f"  ✓ Profit factor: {summary['profit_factor']:.2f}")
        print(f"  ✓ Return: {summary['total_return_pct']:+.2f}%")
        print(f"  ✓ Max DD: {summary['max_drawdown_pct']:.1f}%")


# =============================================================================
# ANA TEST RUNNER
# =============================================================================

def main():
    """Tüm testleri çalıştır."""
    print("=" * 55)
    print("  ADIM 10: PAPER TRADING + OPTİMİZASYON TESTLERİ")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    
    tests = [
        (1,  "PaperTrader Import",            test_01_paper_trader_import,   None),
        (2,  "Trade Açma",                     test_02_open_trade,            None),
        (3,  "SL/TP Simülasyonu",              test_03_sl_tp_simulation,      None),
        (4,  "PnL Hesaplama",                  test_04_pnl_calculation,       None),
        (5,  "Performance Analyzer",           test_05_performance_analyzer,  None),
        (6,  "Direction Analysis",             test_06_direction_analysis,    None),
        (7,  "CSV Export",                     test_07_csv_export,            None),
        (8,  "Optimizer Import",               test_08_optimizer_import,      None),
        (9,  "Quick Backtester",               test_09_quick_backtester,      None),
        (10, "Grid Search",                    test_10_grid_search,           None),
        (11, "Random Search",                  test_11_random_search,         None),
        (12, "Kill Switch Simulation",         test_12_kill_switch_simulation, None),
        (13, "JSON Persistence",               test_13_json_persistence,      None),
        (14, "Summary Report",                 test_14_summary_report,        None),
    ]
    
    results = []
    total_start = time.time()
    
    for num, name, func, skip in tests:
        success = run_test(num, name, func, skip_reason=skip)
        results.append((num, name, success))
    
    total_time = time.time() - total_start
    
    # Özet
    print("\n" + "=" * 55)
    print("  TEST SONUÇLARI")
    print("=" * 55)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for num, name, success in results:
        if success is None:
            status = "⏭️"
            skipped += 1
        elif success:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        print(f"  {status} Test {num:>2}: {name}")
    
    print(f"\n  {'─' * 40}")
    print(f"  Toplam: {len(results)} | ✅ {passed} | ❌ {failed} | ⏭️ {skipped}")
    print(f"  Süre: {total_time:.1f}s")
    
    if failed == 0:
        print(f"\n  🎉 ADIM 10 TAMAMLANDI! Tüm testler geçti.")
        print(f"  → Sonraki: Paper trading ile 1 hafta test")
        print(f"  → Ardından: Gerçek paraya geçiş")
    else:
        print(f"\n  ⚠️  {failed} test başarısız. Hataları kontrol edin.")
    
    print("=" * 55)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
