# =============================================================================
# HİBRİT KRİPTO TRADING BOT — ANA ORKESTRASYON v3.0 (ADIM 9)
# =============================================================================
# Pipeline: Scanner → IC Analiz → GateKeeper → AI Optimizer → Risk → Execute → Telegram
#
# Yeni özellikler (v3.0):
# - Dinamik 500+ coin tarama (CoinScanner entegrasyonu)
# - IC bazlı Gatekeeper karar mekanizması
# - Gemini AI optimizer (IC > 55 olan coinler için)
# - Tam trade execution (IC > 70 ve AI onayı)
# - APScheduler ile periyodik çalışma (15dk / 1saat)
# - Hata yönetimi + retry + circuit breaker
# - macOS LaunchAgent desteği
# - Paper trade modu (dry_run=True)
#
# Çalıştırma:
#   cd hybrid_crypto_bot/src
#   python main.py                      # Tek seferlik analiz
#   python main.py --schedule           # Sürekli çalışma (varsayılan 60dk)
#   python main.py --schedule -i 15     # 15 dakikada bir
#   python main.py --dry-run            # Paper trade modu
#   python main.py --top 10             # Sadece top 10 coin
#   python main.py --symbol SOL         # Tek coin analiz
#
# Mimari (Roadmap):
# ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐
# │  1. IC BEYİN  │─▶│  2. AI GÖZ   │─▶│  3. BİTGET EL      │
# │  (Scanner +  │  │  (Gemini)    │  │  (Execution)       │
# │   IC Analiz) │  │              │  │                    │
# └──────────────┘  └──────────────┘  └────────────────────┘
# =============================================================================

import sys                                     # Sistem çıkış kodları ve path yönetimi
import os                                      # Ortam değişkenleri ve process yönetimi
import time                                    # Performans ölçümü ve bekleme
import signal                                  # Graceful shutdown (SIGINT, SIGTERM)
import argparse                                # CLI argüman parse
import logging                                 # Yapılandırılmış loglama
import traceback                               # Hata detay raporu
from pathlib import Path                       # Platform-bağımsız dosya yolları
from datetime import datetime, timedelta, timezone  # Zaman damgaları ve hesaplamaları
from typing import Dict, List, Optional, Tuple, Any  # Tip belirteçleri (mypy uyumlu)
from dataclasses import dataclass, field       # Yapılandırılmış veri sınıfları
from enum import Enum                          # Sabit değer enumları
from concurrent.futures import ThreadPoolExecutor, as_completed  # Paralel veri çekme

import numpy as np                             # Sayısal hesaplamalar
import pandas as pd                            # DataFrame işlemleri


# =============================================================================
# .ENV DOSYASINI YÜKLE (TÜM İMPORTLARDAN ÖNCE)
# =============================================================================
from dotenv import load_dotenv                 # Ortam değişkeni yöneticisi

CURRENT_FILE = Path(__file__).resolve()        # Bu dosyanın mutlak yolu
PROJECT_ROOT = CURRENT_FILE.parent.parent      # hybrid_crypto_bot/
SRC_DIR = CURRENT_FILE.parent                  # hybrid_crypto_bot/src/
ENV_FILE = PROJECT_ROOT / '.env'               # API key'ler burada

# .env dosyasını yükle → os.environ'a ekle
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)                      # → BITGET_API_KEY, GEMINI_API_KEY, TELEGRAM_* yüklenir
else:
    # Alternatif konum: src/.env
    alt_env = SRC_DIR / '.env'
    if alt_env.exists():
        load_dotenv(alt_env)


# =============================================================================
# PATH AYARLARI (MODÜL İMPORTLARI İÇİN)
# =============================================================================
# src/ dizini ve alt modüllerini Python path'e ekle
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))           # → from config import cfg çalışsın


# =============================================================================
# PROJE MODÜL İMPORTLARI
# =============================================================================

# Merkezi yapılandırma (.env + settings.yaml)
from config import cfg, AppConfig              # Tüm config'ler tek yerden

# Dinamik coin tarayıcı (Adım 4)
from scanner import CoinScanner, CoinScanResult  # 500+ USDT çifti tarama

# Veri katmanı (Adım 2)
from data import BitgetFetcher, DataPreprocessor  # OHLCV çekme + ön işleme

# İndikatör katmanı (Adım 3)
from indicators import (
    IndicatorCalculator,                       # 64+ indikatör hesaplama
    IndicatorSelector,                         # IC bazlı istatistiksel seçim
    IndicatorScore,                            # Tek indikatörün IC skoru
)

# AI karar modülü (Adım 6)
from ai import (
    GeminiOptimizer,                           # Gemini API entegrasyonu
    AIAnalysisInput,                           # AI'ya gönderilecek veri paketi
    AIDecisionResult,                          # AI karar sonucu
    AIDecision,                                # LONG / SHORT / WAIT
    GateAction,                                # NO_TRADE / REPORT_ONLY / FULL_TRADE
)

# Emir yönetimi (Adım 5 + 7)
from execution import (
    RiskManager,                               # ATR bazlı pozisyon sizing
    TradeCalculation,                          # Trade hesaplama sonucu
    BitgetExecutor,                            # Bitget Futures emir gönderme
    ExecutionResult,                           # Emir sonucu
)

# Bildirim sistemi (Adım 8)
from notifications import (
    TelegramNotifier,                          # Telegram bildirim gönderme
    AnalysisReport,                            # Analiz raporu formatı
)


# =============================================================================
# LOGGING YAPILANDIRMASI
# =============================================================================
# Dosya + konsol loglama — zaman damgalı, seviye etiketli
LOG_DIR = PROJECT_ROOT / 'logs'                # Log dosyaları dizini
LOG_DIR.mkdir(parents=True, exist_ok=True)     # Yoksa oluştur

logging.basicConfig(
    level=logging.INFO,                        # INFO ve üstü logla
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),     # Konsola yaz
        logging.FileHandler(                   # Dosyaya yaz
            LOG_DIR / f"bot_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        ),
    ]
)

logger = logging.getLogger('main')             # Ana modül logger'ı


# =============================================================================
# SABİTLER VE VARSAYILANLAR
# =============================================================================

# IC analiz için varsayılan timeframe'ler ve bar sayıları
DEFAULT_TIMEFRAMES = {
    '15m': 400,                                # Day trading — 400 bar ≈ 4 gün
    '30m': 300,                                # Kısa swing — 300 bar ≈ 6 gün
    '1h':  250,                                # Intraday — 250 bar ≈ 10 gün
    '4h':  150,                                # Swing — 150 bar ≈ 25 gün
}

# Forward return periyodu (IC hedef değişkeni)
DEFAULT_FWD_PERIOD = 5                         # 5 bar ileri getiri

# Pipeline limitleri
MAX_CONCURRENT_FETCHES = 4                     # Paralel veri çekme thread sayısı
MAX_COINS_PER_CYCLE = 20                       # Cycle başına max analiz edilecek coin
COIN_ANALYSIS_TIMEOUT = 120                    # Tek coin analiz timeout (saniye)

# Circuit breaker (ardışık hata koruması)
MAX_CONSECUTIVE_ERRORS = 5                     # Bu kadar ardışık hata → döngüyü durdur
ERROR_COOLDOWN_SECONDS = 300                   # Hata sonrası bekleme (5 dakika)

# Versiyon
VERSION = "3.0.0"                              # Adım 9 — tam pipeline orkestrasyon


# =============================================================================
# DATACLASS TANIMLARI
# =============================================================================

class CycleStatus(Enum):
    """Pipeline döngüsünün durumu."""
    SUCCESS = "success"                        # Tüm aşamalar başarılı
    PARTIAL = "partial"                        # Bazı coinler başarısız
    NO_SIGNAL = "no_signal"                    # Hiçbir coin eşiği geçemedi
    ERROR = "error"                            # Kritik hata
    KILLED = "killed"                          # Kill switch tetiklendi


@dataclass
class CoinAnalysisResult:
    """
    Tek bir coin'in analiz sonucu (IC Analiz → AI Karar → Execution).

    Pipeline'ın her aşamasında bu obje güncellenir:
    1. IC analiz → ic_* alanları doldurulur
    2. GateKeeper → gate_action belirlenir
    3. AI karar → ai_decision doldurulur
    4. Execution → execution_result doldurulur

    İstatistiksel Not:
    - composite_score: Ağırlıklı IC composite (top IC %40, avg IC %25, count %15, consistency %20)
    - IC > 0.02 olan indikatörler "anlamlı" kabul edilir (noise threshold)
    """
    # Coin bilgileri
    symbol: str = ""                           # 'SOL/USDT:USDT'
    coin: str = ""                             # 'SOL'
    price: float = 0.0                         # Son fiyat ($)
    change_24h: float = 0.0                    # 24h % değişim

    # IC analiz sonuçları
    best_timeframe: str = ""                   # En iyi TF (composite score bazlı)
    ic_confidence: float = 0.0                 # Composite güven skoru (0-100)
    ic_direction: str = "NEUTRAL"              # IC'nin önerdiği yön
    top_ic: float = 0.0                        # En yüksek |IC| değeri
    top_indicator: str = ""                    # En iyi indikatör adı
    significant_count: int = 0                 # Anlamlı indikatör sayısı
    market_regime: str = "unknown"             # Piyasa rejimi (ADX bazlı)

    # Kategori bazlı en iyi indikatörler
    category_tops: Dict[str, Dict] = field(default_factory=dict)

    # TF sıralaması (tüm timeframe'ler)
    tf_rankings: List[Dict] = field(default_factory=list)

    # Risk hesaplamaları
    atr: float = 0.0                           # ATR değeri ($)
    atr_pct: float = 0.0                       # ATR / fiyat (%)
    sl_price: float = 0.0                      # Stop-Loss fiyatı
    tp_price: float = 0.0                      # Take-Profit fiyatı
    risk_reward: float = 0.0                   # Risk/Reward oranı
    position_size: float = 0.0                 # Pozisyon büyüklüğü (coin)
    leverage: int = 0                          # Kaldıraç

    # GateKeeper kararı
    gate_action: str = ""                      # NO_TRADE / REPORT_ONLY / FULL_TRADE

    # AI karar (Gemini)
    ai_decision: Optional[AIDecisionResult] = None

    # Execution sonucu
    execution_result: Optional[ExecutionResult] = None

    # Durum
    status: str = "pending"                    # pending / analyzed / executed / skipped / error
    error: str = ""                            # Hata mesajı (varsa)
    elapsed: float = 0.0                       # Analiz süresi (saniye)


@dataclass
class CycleReport:
    """
    Bir pipeline döngüsünün (cycle) özet raporu.

    Her 15dk/1saat'te bir cycle çalışır:
    Scan → Analyze → Gate → AI → Execute → Report

    Bu rapor Telegram'a gönderilir.
    """
    timestamp: str = ""                        # Döngü zamanı
    status: CycleStatus = CycleStatus.NO_SIGNAL
    total_scanned: int = 0                     # Taranan toplam coin
    total_analyzed: int = 0                    # IC analiz yapılan coin
    total_above_gate: int = 0                  # Gate eşiğini geçen coin
    total_traded: int = 0                      # İşlem açılan coin
    coins: List[CoinAnalysisResult] = field(default_factory=list)
    balance: float = 0.0                       # Güncel bakiye ($)
    errors: List[str] = field(default_factory=list)
    elapsed: float = 0.0                       # Toplam süre (saniye)


# =============================================================================
# ANA ORKESTRASYON SINIFI
# =============================================================================

class HybridTradingPipeline:
    """
    Tüm modülleri birleştiren ana pipeline sınıfı.

    Flow:
    1. CoinScanner → Top N coin seç (hacim, spread, volatilite)
    2. Her coin için:
       a. BitgetFetcher → OHLCV verisi çek (çoklu TF)
       b. IndicatorCalculator → 64+ indikatör hesapla
       c. IndicatorSelector → IC analiz + anlamlı indikatörleri seç
       d. Composite skor hesapla → en iyi TF ve yön belirle
    3. GateKeeper filtresi:
       - IC < 55 → atla (NO_TRADE)
       - IC 55-70 → rapor et (REPORT_ONLY)
       - IC > 70 → AI'ya gönder (FULL_TRADE)
    4. GeminiOptimizer → nihai LONG/SHORT/WAIT kararı
    5. RiskManager → SL/TP/pozisyon büyüklüğü hesapla
    6. BitgetExecutor → emir gönder (dry_run veya canlı)
    7. TelegramNotifier → tüm sonuçları bildir
    """

    def __init__(
        self,
        dry_run: bool = True,                  # Paper trade modu (varsayılan: güvenli)
        top_n: int = MAX_COINS_PER_CYCLE,      # Analiz edilecek max coin sayısı
        timeframes: Dict[str, int] = None,     # TF → bar sayısı mapping
        fwd_period: int = DEFAULT_FWD_PERIOD,  # Forward return periyodu
        verbose: bool = True,                  # Detaylı çıktı
    ):
        """
        Pipeline'ı başlat ve tüm modülleri initialize et.

        Parameters:
        ----------
        dry_run : bool
            True → emir gönderilmez (paper trade), False → canlı işlem
        top_n : int
            CoinScanner'dan kaç coin alınacak (max 20)
        timeframes : dict
            Her TF için kaç bar çekileceği (örn: {'1h': 250, '4h': 150})
        fwd_period : int
            IC hedef değişkeni: kaç bar ilerinin getirisi
        verbose : bool
            True → konsola detaylı çıktı
        """
        self.dry_run = dry_run                 # Paper trade mi canlı mı?
        self.top_n = min(top_n, MAX_COINS_PER_CYCLE)  # Güvenlik sınırı
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES  # TF konfigürasyonu
        self.fwd_period = fwd_period           # IC forward return periyodu
        self.verbose = verbose                 # Detaylı log

        # ---- MODÜL İNİTİALİZASYONU ----

        # Coin tarayıcı — 500+ USDT Futures çifti tara, top N döndür
        self.scanner = CoinScanner(verbose=verbose)

        # Veri çekici — Bitget Futures OHLCV
        self.fetcher = BitgetFetcher()

        # Veri ön işlemci — return hesaplama, outlier temizleme
        self.preprocessor = DataPreprocessor()

        # İndikatör hesaplayıcı — pandas-ta ile 64+ teknik indikatör
        self.calculator = IndicatorCalculator(verbose=False)

        # IC seçici — Spearman IC + FDR düzeltmesi
        self.selector = IndicatorSelector(
            alpha=0.05,                        # %5 anlamlılık seviyesi
            correction_method='fdr',           # Benjamini-Hochberg FDR düzeltmesi
            verbose=False                      # Her indikatör için log basma
        )

        # AI optimizer — Gemini ile nihai karar
        self.ai_optimizer = GeminiOptimizer()

        # Risk yöneticisi — başlangıç bakiyesi ile (canlı modda API'den çekilecek)
        self._risk_manager = None              # Lazy init (bakiye gerekli)

        # Emir yöneticisi — dry_run veya canlı
        self.executor = BitgetExecutor(dry_run=dry_run)

        # Telegram bildirici
        self.notifier = TelegramNotifier()

        # ---- DURUM DEĞİŞKENLERİ ----
        self._balance: float = 0.0             # Güncel USDT bakiye
        self._initial_balance: float = 0.0     # Başlangıç bakiyesi (kill switch için)
        self._consecutive_errors: int = 0      # Ardışık hata sayacı
        self._is_running: bool = False         # Pipeline çalışıyor mu?
        self._kill_switch: bool = False        # Kill switch aktif mi?
        self._cycle_count: int = 0             # Toplam döngü sayısı

        logger.info(
            f"🚀 HybridTradingPipeline v{VERSION} başlatıldı | "
            f"Mode: {'🧪 DRY RUN' if dry_run else '🔴 CANLI'} | "
            f"Top N: {self.top_n} | TFs: {list(self.timeframes.keys())}"
        )

    # =========================================================================
    # BAKİYE YÖNETİMİ
    # =========================================================================

    def _init_balance(self) -> bool:
        """
        Bakiyeyi API'den çek veya DRY RUN için varsayılan ata.

        DRY RUN modda config'den okunan varsayılan bakiye kullanılır.
        Canlı modda Bitget API'den gerçek USDT bakiye çekilir.

        Returns:
        -------
        bool
            Bakiye başarıyla alındıysa True
        """
        try:
            if self.dry_run:
                # DRY RUN: varsayılan bakiye (config veya $75)
                self._balance = 75.0           # Roadmap varsayılan sermaye
                self._initial_balance = 75.0
                logger.info(f"💰 DRY RUN bakiye: ${self._balance:.2f}")
            else:
                # CANLI: Bitget API'den gerçek bakiye çek
                balance_info = self.executor.fetch_balance()
                self._balance = balance_info.get('free', 0.0)  # Kullanılabilir USDT
                self._initial_balance = balance_info.get('total', self._balance)
                logger.info(
                    f"💰 Canlı bakiye: ${self._balance:.2f} "
                    f"(Total: ${self._initial_balance:.2f})"
                )

            # Risk manager'ı bakiye ile başlat
            self._risk_manager = RiskManager(
                balance=self._balance,
                initial_balance=self._initial_balance
            )

            return self._balance > 0           # Bakiye 0'dan büyükse başarılı

        except Exception as e:
            logger.error(f"❌ Bakiye çekme hatası: {e}")
            return False

    def _refresh_balance(self) -> None:
        """Her cycle başında bakiyeyi güncelle."""
        if not self.dry_run:
            try:
                balance_info = self.executor.fetch_balance()
                self._balance = balance_info.get('free', 0.0)
                if self._risk_manager:
                    self._risk_manager.update_balance(self._balance)
                logger.info(f"💰 Bakiye güncellendi: ${self._balance:.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Bakiye güncelleme hatası (eski bakiye kullanılıyor): {e}")

    # =========================================================================
    # KILL SWITCH KONTROLÜ
    # =========================================================================

    def _check_kill_switch(self) -> bool:
        """
        Drawdown bazlı kill switch kontrolü.

        Kill switch kuralı (config'den):
        - Mevcut bakiye, başlangıç bakiyesinin %15'inden fazla düştüyse → DURDUR

        Returns:
        -------
        bool
            True → kill switch AKTİF (işlem yapma!)
        """
        if self._initial_balance <= 0:
            return False                       # Başlangıç bakiyesi bilinmiyorsa kontrol etme

        drawdown_pct = ((self._initial_balance - self._balance) / self._initial_balance) * 100

        threshold = cfg.risk.kill_switch_drawdown_pct  # Varsayılan: %15

        if drawdown_pct >= threshold:
            self._kill_switch = True
            logger.critical(
                f"🚨 KILL SWITCH AKTİF! Drawdown: %{drawdown_pct:.1f} "
                f"(Eşik: %{threshold}) — TÜM İŞLEMLER DURDURULDU"
            )

            # Telegram'dan acil bildirim
            try:
                self.notifier.send_risk_alert_sync(
                    alert_type="kill_switch",
                    message=(
                        f"🚨 KILL SWITCH AKTİF!\n"
                        f"Drawdown: %{drawdown_pct:.1f} (Eşik: %{threshold})\n"
                        f"Başlangıç: ${self._initial_balance:.2f}\n"
                        f"Mevcut: ${self._balance:.2f}\n"
                        f"Kayıp: ${self._initial_balance - self._balance:.2f}\n\n"
                        f"⛔ Tüm işlemler durduruldu."
                    ),
                    severity="critical"
                )
            except Exception:
                pass                           # Bildirim hatasını yut, kill switch'i engelleme

            return True

        return False

    # =========================================================================
    # AŞAMA 1: COIN TARAMA (SCANNER)
    # =========================================================================

    def _scan_market(self) -> List[CoinScanResult]:
        """
        CoinScanner ile market taraması yap.

        500+ USDT Futures çiftini tarar, hacim/spread/volatilite filtreler,
        composite skor ile sıralar ve top N coin döndürür.

        Returns:
        -------
        List[CoinScanResult]
            Sıralanmış coin listesi (en iyi → en kötü)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📡 AŞAMA 1: MARKET TARAMASI (Top {self.top_n})")
        logger.info(f"{'='*60}")

        try:
            # CoinScanner.scan() → batch ticker + filtre + composite skor
            top_coins = self.scanner.scan(top_n=self.top_n)

            if not top_coins:
                logger.warning("⚠️ Tarama sonucu boş — coin bulunamadı")
                return []

            logger.info(f"✅ {len(top_coins)} coin seçildi (toplam taranan: 500+)")

            if self.verbose:
                for i, coin in enumerate(top_coins[:5], 1):
                    logger.info(
                        f"  #{i} {coin.symbol} | "
                        f"Vol: ${coin.volume_24h:,.0f} | "
                        f"Score: {coin.composite_score:.1f}"
                    )

            return top_coins

        except Exception as e:
            logger.error(f"❌ Tarama hatası: {e}")
            traceback.print_exc()
            return []

    # =========================================================================
    # AŞAMA 2: IC ANALİZ (TEK COİN)
    # =========================================================================

    def _analyze_coin(self, symbol: str, coin_name: str) -> Optional[CoinAnalysisResult]:
        """
        Tek bir coin için tam IC analiz pipeline'ı.

        Pipeline:
        1. Tüm timeframe'ler için OHLCV verisi çek
        2. Her TF'de indikatör hesapla + forward return ekle
        3. IC analiz yap (Spearman korelasyon + FDR)
        4. Composite skor hesapla (en iyi TF ve yön belirle)
        5. Market rejimi tespit et (ADX bazlı)

        Parameters:
        ----------
        symbol : str
            Bitget Futures sembolü (örn: 'SOL/USDT:USDT')
        coin_name : str
            Kısa coin adı (örn: 'SOL')

        Returns:
        -------
        CoinAnalysisResult veya None (hata durumunda)
        """
        start_time = time.time()
        result = CoinAnalysisResult(symbol=symbol, coin=coin_name)

        try:
            # ---- 2a. VERİ ÇEKME (tüm TF'ler paralel) ----
            tf_data: Dict[str, pd.DataFrame] = {}  # TF → OHLCV DataFrame

            for tf, bars in self.timeframes.items():
                try:
                    # Bitget Futures OHLCV çek
                    df = self.fetcher.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=tf,
                        limit=bars
                    )

                    # Minimum bar kontrolü (IC güvenilirliği için en az 50 bar)
                    if df is not None and len(df) >= 50:
                        tf_data[tf] = df
                    else:
                        logger.debug(f"  {symbol} {tf}: Yetersiz veri ({len(df) if df is not None else 0} bar)")

                except Exception as e:
                    logger.debug(f"  {symbol} {tf}: Veri çekme hatası — {e}")
                    continue

                time.sleep(0.1)                # Rate limit koruması (100ms)

            if not tf_data:
                result.status = "error"
                result.error = "Hiçbir TF'de yeterli veri alınamadı"
                return result

            # ---- Son fiyat ve 24h değişim ----
            latest_tf = list(tf_data.keys())[0]        # En kısa TF'den al
            latest_df = tf_data[latest_tf]
            result.price = float(latest_df['close'].iloc[-1])  # Son kapanış fiyatı

            # 24h değişim (varsa)
            if len(latest_df) >= 2:
                first_close = float(latest_df['close'].iloc[0])
                result.change_24h = ((result.price - first_close) / first_close) * 100

            # ---- 2b. İNDİKATÖR HESAPLAMA + IC ANALİZ (her TF için) ----
            tf_analyses = []                   # (tf, top_ic, direction, composite, scores, regime)

            for tf, df in tf_data.items():
                try:
                    # İndikatörleri hesapla (trend, momentum, volatility, volume)
                    df_ind = self.calculator.calculate_all(
                        df,
                        categories=['trend', 'momentum', 'volatility', 'volume']
                    )

                    # Fiyat türev özellikleri ekle (returns, log_returns, vb.)
                    df_ind = self.calculator.add_price_features(df_ind)

                    # Forward return ekle (IC hedef değişkeni)
                    df_ind = self.calculator.add_forward_returns(
                        df_ind,
                        periods=[1, self.fwd_period, 10]
                    )

                    # IC analiz — Spearman korelasyon + p-value + FDR
                    target_col = f'fwd_ret_{self.fwd_period}'  # Hedef: 5-bar ileri getiri
                    scores = self.selector.evaluate_all_indicators(
                        df_ind,
                        target_col=target_col
                    )

                    # Anlamlı indikatörleri filtrele (|IC| > 0.02)
                    valid_categories = ['trend', 'momentum', 'volatility', 'volume']
                    sig_scores = [
                        s for s in scores
                        if abs(s.ic_mean) > 0.02        # Noise threshold
                        and not np.isnan(s.ic_mean)      # NaN kontrolü
                        and s.category in valid_categories
                    ]

                    if not sig_scores:
                        continue                         # Bu TF'de anlamlı sinyal yok

                    # ---- Composite skor hesaplama ----
                    # En iyi indikatör (en yüksek |IC|)
                    top_score = max(sig_scores, key=lambda x: abs(x.ic_mean))
                    top_ic_val = abs(top_score.ic_mean)

                    # Ortalama |IC| (tüm anlamlı indikatörler)
                    avg_ic = np.mean([abs(s.ic_mean) for s in sig_scores])

                    # Yön tutarlılığı (positive IC → LONG, negative → SHORT)
                    pos_count = sum(1 for s in sig_scores if s.ic_mean > 0)
                    neg_count = sum(1 for s in sig_scores if s.ic_mean < 0)
                    consistency = max(pos_count, neg_count) / len(sig_scores)

                    # Dominant yön belirleme
                    if neg_count > pos_count * 1.5:
                        direction = 'SHORT'              # Net bearish sinyal
                    elif pos_count > neg_count * 1.5:
                        direction = 'LONG'               # Net bullish sinyal
                    else:
                        direction = 'NEUTRAL'            # Karışık sinyal

                    # ---- Market rejimi tespiti (ADX bazlı) ----
                    regime = self._detect_regime(df_ind)

                    # ---- Normalize + ağırlıklı composite ----
                    # top_ic: 0.02-0.40 aralığını 0-100'e map et
                    top_norm = min((top_ic_val - 0.02) / 0.38 * 100, 100)
                    # avg_ic: 0.02-0.15 aralığını 0-100'e map et
                    avg_norm = min((avg_ic - 0.02) / 0.13 * 100, 100)
                    # Anlamlı indikatör sayısı: 0-50 → 0-100
                    cnt_norm = min(len(sig_scores) / 50 * 100, 100)
                    # Tutarlılık: 0.5-1.0 → 0-100
                    cons_norm = max(0, min((consistency - 0.5) / 0.5 * 100, 100))

                    # Ağırlıklı composite (roadmap'teki formül)
                    composite = (
                        top_norm  * 0.40 +               # Top IC ağırlığı: %40
                        avg_norm  * 0.25 +               # Avg IC ağırlığı: %25
                        cnt_norm  * 0.15 +               # Count ağırlığı: %15
                        cons_norm * 0.20                  # Consistency ağırlığı: %20
                    )

                    # Rejim bazlı düzeltme (ranging/volatile → güvenilirlik düşer)
                    regime_multipliers = {
                        'ranging': 0.85,                 # Yatay piyasada IC daha az güvenilir
                        'volatile': 0.80,                # Aşırı volatilitede sinyal gürültülü
                        'transitioning': 0.90,           # Geçiş döneminde dikkatli ol
                    }
                    composite *= regime_multipliers.get(regime, 1.0)

                    # Kategori bazlı en iyi indikatörleri kaydet
                    cat_tops = {}
                    for cat in valid_categories:
                        cat_scores = [s for s in sig_scores if s.category == cat]
                        if cat_scores:
                            best_cat = max(cat_scores, key=lambda x: abs(x.ic_mean))
                            cat_tops[cat] = {
                                'name': best_cat.name,
                                'ic': best_cat.ic_mean,
                            }

                    tf_analyses.append({
                        'tf': tf,
                        'top_ic': top_ic_val,
                        'top_indicator': top_score.name,
                        'avg_ic': avg_ic,
                        'sig_count': len(sig_scores),
                        'consistency': consistency,
                        'direction': direction,
                        'composite': composite,
                        'regime': regime,
                        'scores': scores,
                        'cat_tops': cat_tops,
                        'df': df_ind,                    # ATR hesabı için sakla
                    })

                except Exception as e:
                    logger.debug(f"  {symbol} {tf}: IC analiz hatası — {e}")
                    continue

            # ---- 2c. EN İYİ TF SEÇİMİ ----
            if not tf_analyses:
                result.status = "skipped"
                result.error = "Hiçbir TF'de anlamlı IC bulunamadı"
                result.elapsed = time.time() - start_time
                return result

            # Composite skora göre sırala (en yüksek → en iyi TF)
            tf_analyses.sort(key=lambda x: x['composite'], reverse=True)
            best = tf_analyses[0]              # En iyi timeframe

            # ---- SONUÇ DOLDURMA ----
            result.best_timeframe = best['tf']
            result.ic_confidence = best['composite']
            result.ic_direction = best['direction']
            result.top_ic = best['top_ic']
            result.top_indicator = best['top_indicator']
            result.significant_count = best['sig_count']
            result.market_regime = best['regime']
            result.category_tops = best['cat_tops']

            # TF rankings (Telegram raporu için)
            result.tf_rankings = [
                {
                    'tf': a['tf'],
                    'score': a['composite'],
                    'direction': a['direction'],
                    'top_ic': a['top_ic'],
                    'regime': a['regime'],
                }
                for a in tf_analyses
            ]

            # ---- ATR HESAPLAMA (Risk Manager için) ----
            best_df = best['df']               # En iyi TF'nin DataFrame'i
            if 'ATRr_14' in best_df.columns:
                result.atr = float(best_df['ATRr_14'].iloc[-1])
            elif 'NATR_14' in best_df.columns:
                # NATR yüzdeyse, fiyata çevir
                natr = float(best_df['NATR_14'].iloc[-1])
                result.atr = result.price * natr / 100
            else:
                # Manuel ATR hesabı (14 periyot)
                high = best_df['high']
                low = best_df['low']
                close = best_df['close']
                tr = pd.concat([
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()
                ], axis=1).max(axis=1)
                result.atr = float(tr.rolling(14).mean().iloc[-1])

            result.atr_pct = (result.atr / result.price * 100) if result.price > 0 else 0

            result.status = "analyzed"
            result.elapsed = time.time() - start_time

            logger.info(
                f"  ✅ {coin_name}: TF={best['tf']} | "
                f"IC={best['composite']:.0f} | {best['direction']} | "
                f"Regime={best['regime']} | {best['sig_count']} sig | "
                f"{result.elapsed:.1f}s"
            )

            return result

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            result.elapsed = time.time() - start_time
            logger.error(f"  ❌ {coin_name}: Analiz hatası — {e}")
            return result

    # =========================================================================
    # AŞAMA 3: GATEKEEPER + AI KARAR
    # =========================================================================

    def _evaluate_coin(self, analysis: CoinAnalysisResult) -> CoinAnalysisResult:
        """
        IC analiz sonucunu GateKeeper + AI Optimizer'a gönder.

        GateKeeper Kuralları (Roadmap):
        - IC < 55  → NO_TRADE (AI'a bile sorma)
        - IC 55-70 → REPORT_ONLY (AI'a sor, rapor et, emir girme)
        - IC > 70  → FULL_TRADE (AI optimize etsin + emir gir)

        Parameters:
        ----------
        analysis : CoinAnalysisResult
            IC analiz sonucu doldurulmuş obje

        Returns:
        -------
        CoinAnalysisResult
            AI karar ve gate_action ile güncellenmiş obje
        """
        # ---- AI INPUT HAZIRLA ----
        ai_input = AIAnalysisInput(
            symbol=analysis.symbol,
            coin=analysis.coin,
            price=analysis.price,
            change_24h=analysis.change_24h,
            best_timeframe=analysis.best_timeframe,
            ic_confidence=analysis.ic_confidence,
            ic_direction=analysis.ic_direction,
            category_tops=analysis.category_tops,
            tf_rankings=analysis.tf_rankings,
            atr=analysis.atr,
            atr_pct=analysis.atr_pct,
            market_regime=analysis.market_regime,
            volume_24h=0.0,                    # Scanner'dan doldurulacak
            volatility=analysis.atr_pct,
        )

        # ---- GEMİNİ OPTİMİZER → KARAR ----
        try:
            ai_result = self.ai_optimizer.get_decision(ai_input)
            analysis.ai_decision = ai_result
            analysis.gate_action = ai_result.gate_action.value

            # AI'dan gelen optimize edilmiş SL/TP değerlerini al
            if ai_result.sl_price > 0:
                analysis.sl_price = ai_result.sl_price
            if ai_result.tp_price > 0:
                analysis.tp_price = ai_result.tp_price
            if ai_result.risk_reward > 0:
                analysis.risk_reward = ai_result.risk_reward

            logger.info(
                f"  🤖 {analysis.coin}: Gate={ai_result.gate_action.value} | "
                f"AI={ai_result.decision.value} | Güven={ai_result.confidence:.0f}"
            )

        except Exception as e:
            logger.error(f"  ❌ {analysis.coin}: AI karar hatası — {e}")
            analysis.gate_action = "NO_TRADE"  # Hata durumunda güvenli tarafta kal
            analysis.error = f"AI hatası: {e}"

        return analysis

    # =========================================================================
    # AŞAMA 4: RİSK HESAPLAMA + EMİR GÖNDERİM
    # =========================================================================

    def _execute_trade(self, analysis: CoinAnalysisResult) -> CoinAnalysisResult:
        """
        AI onaylı coin için risk hesapla ve emir gönder.

        Sadece şu durumda çalışır:
        - gate_action == FULL_TRADE
        - ai_decision.should_execute() == True
        - Kill switch AKTİF DEĞİL

        Pipeline:
        1. RiskManager.calculate_trade() → SL/TP/pozisyon/kaldıraç
        2. Trade onay kontrolü (is_approved?)
        3. BitgetExecutor.execute_trade() → emir gönder
        4. Bakiye güncelle

        Parameters:
        ----------
        analysis : CoinAnalysisResult
            AI kararı doldurulmuş obje

        Returns:
        -------
        CoinAnalysisResult
            Execution sonucu ile güncellenmiş obje
        """
        # ---- KONTROLLER ----
        if self._kill_switch:
            analysis.status = "killed"
            analysis.error = "Kill switch aktif"
            return analysis

        ai = analysis.ai_decision
        if not ai or not ai.should_execute():
            analysis.status = "skipped"
            return analysis

        if not self._risk_manager:
            analysis.status = "error"
            analysis.error = "RiskManager başlatılmamış"
            return analysis

        # ---- RİSK HESAPLAMA ----
        try:
            direction = ai.decision.value      # 'LONG' veya 'SHORT'

            # ATR multiplier — AI'dan gelirse onu kullan, yoksa varsayılan
            atr_mult = ai.atr_multiplier if ai.atr_multiplier > 0 else 1.5

            # RiskManager → SL/TP/pozisyon büyüklüğü/kaldıraç hesapla
            trade_calc = self._risk_manager.calculate_trade(
                entry_price=analysis.price,
                direction=direction,
                atr=analysis.atr,
                symbol=analysis.symbol,
                atr_multiplier=atr_mult,
                risk_reward=cfg.risk.min_risk_reward_ratio,  # Min 1.5 RR
            )

            # Trade onay kontrolü (margin yeterli mi? risk limiti aşılıyor mu?)
            if not trade_calc.is_approved():
                analysis.status = "rejected"
                analysis.error = f"Risk kontrolü red: {trade_calc.rejection_reasons}"
                logger.warning(f"  ⚠️ {analysis.coin}: Trade reddedildi — {trade_calc.rejection_reasons}")
                return analysis

            # Sonuçları kaydet
            analysis.sl_price = trade_calc.stop_loss.price
            analysis.tp_price = trade_calc.take_profit.price
            analysis.risk_reward = trade_calc.take_profit.distance / trade_calc.stop_loss.distance if trade_calc.stop_loss.distance > 0 else 0
            analysis.position_size = trade_calc.position.size
            analysis.leverage = trade_calc.position.leverage

            logger.info(
                f"  📊 {analysis.coin}: Size={trade_calc.position.size} | "
                f"SL=${trade_calc.stop_loss.price:,.2f} | "
                f"TP=${trade_calc.take_profit.price:,.2f} | "
                f"Lev={trade_calc.position.leverage}x"
            )

        except Exception as e:
            analysis.status = "error"
            analysis.error = f"Risk hesaplama hatası: {e}"
            logger.error(f"  ❌ {analysis.coin}: Risk hatası — {e}")
            return analysis

        # ---- EMİR GÖNDER ----
        try:
            exec_result = self.executor.execute_trade(trade_calc)
            analysis.execution_result = exec_result

            if exec_result.success:
                analysis.status = "executed"
                logger.info(
                    f"  {'🧪' if self.dry_run else '🔴'} {analysis.coin}: "
                    f"{direction} emri gönderildi ✅"
                )
            else:
                analysis.status = "error"
                analysis.error = f"Execution hatası: {exec_result.error}"
                logger.error(f"  ❌ {analysis.coin}: Execution hatası — {exec_result.error}")

        except Exception as e:
            analysis.status = "error"
            analysis.error = f"Execution exception: {e}"
            logger.error(f"  ❌ {analysis.coin}: Execution exception — {e}")

        return analysis

    # =========================================================================
    # AŞAMA 5: TELEGRAM BİLDİRİM
    # =========================================================================

    def _send_cycle_report(self, report: CycleReport) -> None:
        """
        Döngü raporunu Telegram'a gönder.

        Rapor formatı:
        - Taranan coin sayısı
        - Gate eşiğini geçenler
        - Açılan işlemler
        - Hata özeti
        """
        try:
            if not self.notifier.is_configured():
                logger.warning("⚠️ Telegram yapılandırılmamış, bildirim atlanıyor")
                return

            # ---- ÖZET MESAJ ----
            mode = "🧪 DRY RUN" if self.dry_run else "🔴 CANLI"
            status_emoji = {
                CycleStatus.SUCCESS: "✅",
                CycleStatus.PARTIAL: "⚠️",
                CycleStatus.NO_SIGNAL: "↔️",
                CycleStatus.ERROR: "❌",
                CycleStatus.KILLED: "🚨",
            }

            lines = [
                f"{status_emoji.get(report.status, '❓')} {mode} — DÖNGÜ #{self._cycle_count}",
                f"⏰ {report.timestamp}",
                f"",
                f"📡 Taranan: {report.total_scanned}",
                f"🔬 Analiz: {report.total_analyzed}",
                f"🚦 Gate+: {report.total_above_gate}",
                f"📈 İşlem: {report.total_traded}",
                f"💰 Bakiye: ${report.balance:,.2f}",
                f"⏱ Süre: {report.elapsed:.0f}s",
            ]

            # ---- DETAY: Gate eşiğini geçen coinler ----
            gate_coins = [c for c in report.coins if c.gate_action in ('REPORT_ONLY', 'FULL_TRADE')]
            if gate_coins:
                lines.append(f"\n{'─'*30}")
                lines.append("📊 SİNYAL DETAY:")
                for c in gate_coins:
                    dir_emoji = "🟢" if c.ic_direction == "LONG" else "🔴" if c.ic_direction == "SHORT" else "↔️"
                    status_icon = "✅" if c.status == "executed" else "📋" if c.gate_action == "REPORT_ONLY" else "⏭"

                    lines.append(
                        f"  {status_icon} {c.coin} {dir_emoji} | "
                        f"IC={c.ic_confidence:.0f} | TF={c.best_timeframe} | "
                        f"{c.market_regime}"
                    )

                    # Execution detayı (varsa)
                    if c.execution_result and c.execution_result.success:
                        er = c.execution_result
                        lines.append(
                            f"     📍 Entry: ${er.actual_entry:,.2f} | "
                            f"SL: ${c.sl_price:,.2f} | TP: ${c.tp_price:,.2f}"
                        )

            # ---- HATALAR ----
            if report.errors:
                lines.append(f"\n⚠️ Hatalar ({len(report.errors)}):")
                for err in report.errors[:3]:  # Max 3 hata göster
                    lines.append(f"  • {err[:80]}")

            message = "\n".join(lines)

            # Gönder
            self.notifier.send_alert_sync(
                title=f"📊 Cycle #{self._cycle_count}",
                message=message,
                severity="info" if report.status in (CycleStatus.SUCCESS, CycleStatus.NO_SIGNAL) else "warning"
            )

        except Exception as e:
            logger.error(f"❌ Telegram bildirim hatası: {e}")

    # =========================================================================
    # ANA PIPELINE DÖNGÜSÜ
    # =========================================================================

    def run_cycle(self) -> CycleReport:
        """
        Tek bir pipeline döngüsü çalıştır.

        Pipeline:
        1. Bakiye güncelle + kill switch kontrol
        2. CoinScanner → Top N coin
        3. Her coin için IC analiz
        4. Gate eşiğini geçenler → AI optimizer
        5. AI onaylılar → Risk hesapla + emir gönder
        6. Rapor oluştur + Telegram bildir

        Returns:
        -------
        CycleReport
            Döngü özet raporu
        """
        self._cycle_count += 1
        cycle_start = time.time()

        report = CycleReport(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 HYBRID TRADING PIPELINE v{VERSION} — DÖNGÜ #{self._cycle_count}")
        logger.info(f"⏰ {report.timestamp}")
        logger.info(f"🔧 Mode: {'🧪 DRY RUN' if self.dry_run else '🔴 CANLI'}")
        logger.info(f"{'='*70}")

        try:
            # ---- 0. BAKİYE + KILL SWITCH ----
            self._refresh_balance()
            report.balance = self._balance

            if self._check_kill_switch():
                report.status = CycleStatus.KILLED
                report.elapsed = time.time() - cycle_start
                self._send_cycle_report(report)
                return report

            # ---- 1. MARKET TARAMASI ----
            top_coins = self._scan_market()
            report.total_scanned = len(top_coins) if top_coins else 0

            if not top_coins:
                report.status = CycleStatus.NO_SIGNAL
                report.elapsed = time.time() - cycle_start
                logger.info("⏭ Taramada uygun coin bulunamadı")
                self._send_cycle_report(report)
                return report

            # ---- 2. IC ANALİZ (her coin için) ----
            logger.info(f"\n{'='*60}")
            logger.info(f"🔬 AŞAMA 2: IC ANALİZ ({len(top_coins)} coin)")
            logger.info(f"{'='*60}")

            analyzed_coins: List[CoinAnalysisResult] = []

            for i, coin_result in enumerate(top_coins, 1):
                logger.info(f"\n  [{i}/{len(top_coins)}] {coin_result.symbol} analiz ediliyor...")

                analysis = self._analyze_coin(
                    symbol=coin_result.symbol,
                    coin_name=coin_result.symbol.split('/')[0]  # 'SOL/USDT:USDT' → 'SOL'
                )

                if analysis and analysis.status == "analyzed":
                    analyzed_coins.append(analysis)

                time.sleep(0.2)                # API rate limit koruması

            report.total_analyzed = len(analyzed_coins)

            if not analyzed_coins:
                report.status = CycleStatus.NO_SIGNAL
                report.elapsed = time.time() - cycle_start
                logger.info("⏭ Hiçbir coinde anlamlı IC bulunamadı")
                self._send_cycle_report(report)
                return report

            # IC skoruna göre sırala (en yüksek → en umut verici)
            analyzed_coins.sort(key=lambda x: x.ic_confidence, reverse=True)

            # ---- 3. GATEKEEPER + AI KARAR ----
            logger.info(f"\n{'='*60}")
            logger.info(f"🚦 AŞAMA 3: GATEKEEPER + AI KARAR")
            logger.info(f"{'='*60}")

            for analysis in analyzed_coins:
                # Gate kontrolü — IC < 55 → atla
                if analysis.ic_confidence < cfg.gate.no_trade:
                    analysis.gate_action = "NO_TRADE"
                    analysis.status = "skipped"
                    logger.info(
                        f"  ❌ {analysis.coin}: IC={analysis.ic_confidence:.0f} "
                        f"< {cfg.gate.no_trade} → NO_TRADE"
                    )
                    continue

                # Gate eşiğini geçti → AI'ya gönder
                report.total_above_gate += 1
                analysis = self._evaluate_coin(analysis)

            # ---- 4. TRADE EXECUTION ----
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 AŞAMA 4: TRADE EXECUTION")
            logger.info(f"{'='*60}")

            for analysis in analyzed_coins:
                if (analysis.ai_decision
                    and analysis.ai_decision.should_execute()
                    and analysis.gate_action == "FULL_TRADE"):

                    analysis = self._execute_trade(analysis)

                    if analysis.status == "executed":
                        report.total_traded += 1

            # ---- 5. RAPOR ----
            report.coins = analyzed_coins
            report.elapsed = time.time() - cycle_start

            # Durum belirleme
            if report.total_traded > 0:
                report.status = CycleStatus.SUCCESS
            elif report.total_above_gate > 0:
                report.status = CycleStatus.PARTIAL
            else:
                report.status = CycleStatus.NO_SIGNAL

            # Hataları topla
            report.errors = [
                f"{c.coin}: {c.error}"
                for c in analyzed_coins
                if c.error
            ]

            # Ardışık hata sayacını sıfırla (başarılı cycle)
            self._consecutive_errors = 0

            # ---- ÖZET LOG ----
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 DÖNGÜ #{self._cycle_count} ÖZET")
            logger.info(f"{'='*70}")
            logger.info(f"  Taranan: {report.total_scanned}")
            logger.info(f"  Analiz: {report.total_analyzed}")
            logger.info(f"  Gate+: {report.total_above_gate}")
            logger.info(f"  İşlem: {report.total_traded}")
            logger.info(f"  Bakiye: ${report.balance:,.2f}")
            logger.info(f"  Süre: {report.elapsed:.1f}s")
            logger.info(f"{'='*70}")

            # ---- TELEGRAM BİLDİRİM ----
            self._send_cycle_report(report)

            return report

        except Exception as e:
            self._consecutive_errors += 1
            logger.exception(f"❌ Döngü #{self._cycle_count} kritik hata: {e}")

            report.status = CycleStatus.ERROR
            report.errors.append(str(e))
            report.elapsed = time.time() - cycle_start

            self._send_cycle_report(report)
            return report

    # =========================================================================
    # TEK COİN ANALİZ MODU
    # =========================================================================

    def analyze_single(self, symbol: str) -> Optional[CoinAnalysisResult]:
        """
        Tek bir coin için tam analiz + karar pipeline'ı.

        Kullanım: python main.py --symbol SOL

        Parameters:
        ----------
        symbol : str
            Coin adı (örn: 'SOL') veya tam sembol (örn: 'SOL/USDT:USDT')

        Returns:
        -------
        CoinAnalysisResult veya None
        """
        # Sembol formatı normalize et
        if '/' not in symbol:
            symbol = f"{symbol.upper()}/USDT:USDT"

        coin_name = symbol.split('/')[0]

        logger.info(f"\n🔍 TEK COİN ANALİZ: {symbol}")

        # Bakiye başlat
        if not self._init_balance():
            logger.error("❌ Bakiye başlatılamadı")
            return None

        # Analiz
        analysis = self._analyze_coin(symbol, coin_name)
        if not analysis or analysis.status != "analyzed":
            logger.warning(f"⚠️ {coin_name}: Analiz başarısız — {analysis.error if analysis else 'None'}")
            return analysis

        # Gatekeeper + AI
        analysis = self._evaluate_coin(analysis)

        # Execution (sadece uygunsa)
        if (analysis.ai_decision
            and analysis.ai_decision.should_execute()
            and analysis.gate_action == "FULL_TRADE"):
            analysis = self._execute_trade(analysis)

        # Özet yazdır
        self._print_single_analysis(analysis)

        return analysis

    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """
        ADX bazlı piyasa rejimi tespiti.

        ADX > 25 → trending (yön: DI+ vs DI- karşılaştırması)
        ADX < 20 → ranging (yatay piyasa)
        20 ≤ ADX ≤ 25 → transitioning

        Parameters:
        ----------
        df : pd.DataFrame
            İndikatörleri hesaplanmış DataFrame

        Returns:
        -------
        str : 'trending_up', 'trending_down', 'ranging', 'transitioning', 'unknown'
        """
        if 'ADX_14' not in df.columns:
            return 'unknown'

        adx = df['ADX_14'].iloc[-1]
        if pd.isna(adx):
            return 'unknown'

        # DI+ ve DI- (yön belirlemek için)
        dmp = df.get('DMP_14', pd.Series([50])).iloc[-1] if 'DMP_14' in df.columns else 50
        dmn = df.get('DMN_14', pd.Series([50])).iloc[-1] if 'DMN_14' in df.columns else 50

        if adx > 25:
            return 'trending_up' if dmp > dmn else 'trending_down'
        elif adx < 20:
            return 'ranging'
        return 'transitioning'

    def _print_single_analysis(self, a: CoinAnalysisResult) -> None:
        """Tek coin analiz sonucunu konsola yazdır."""
        print(f"\n{'='*60}")
        print(f"📊 {a.coin} ANALİZ SONUCU")
        print(f"{'='*60}")
        print(f"  💵 Fiyat: ${a.price:,.2f} ({a.change_24h:+.1f}%)")
        print(f"  📈 TF: {a.best_timeframe} | IC: {a.ic_confidence:.0f}/100")
        print(f"  🧭 Yön: {a.ic_direction} | Rejim: {a.market_regime}")
        print(f"  🔬 Anlamlı: {a.significant_count} indikatör")
        print(f"  📏 ATR: ${a.atr:.2f} ({a.atr_pct:.1f}%)")

        if a.category_tops:
            print(f"\n  Kategori Tops:")
            for cat, info in a.category_tops.items():
                print(f"    {cat}: {info['name']} (IC={info['ic']:+.3f})")

        if a.tf_rankings:
            print(f"\n  TF Sıralaması:")
            for r in a.tf_rankings:
                marker = "→" if r['tf'] == a.best_timeframe else " "
                print(f"   {marker}{r['tf']:<5} Score={r['score']:.0f} {r['direction']:<8} {r['regime']}")

        print(f"\n  🚦 Gate: {a.gate_action}")

        if a.ai_decision:
            ai = a.ai_decision
            print(f"  🤖 AI: {ai.decision.value} (Güven: {ai.confidence:.0f})")
            print(f"  💬 {ai.reasoning[:100]}")

        if a.sl_price > 0:
            print(f"\n  🛑 SL: ${a.sl_price:,.2f}")
            print(f"  🎯 TP: ${a.tp_price:,.2f}")
            print(f"  ⚖️ RR: {a.risk_reward:.1f}")
            print(f"  📦 Size: {a.position_size}")
            print(f"  ⚡ Leverage: {a.leverage}x")

        if a.execution_result:
            print(f"\n  📋 Execution: {'✅' if a.execution_result.success else '❌'}")
            print(f"  {a.execution_result.summary()}")

        print(f"\n  Status: {a.status}")
        if a.error:
            print(f"  ⚠️ {a.error}")
        print(f"{'='*60}")


# =============================================================================
# SCHEDULER (PERİYODİK ÇALIŞTIRICI)
# =============================================================================

def run_scheduler(pipeline: HybridTradingPipeline, interval_minutes: int = 60):
    """
    Pipeline'ı belirtilen aralıkla periyodik çalıştırır.

    APScheduler yerine basit while-loop + sleep kullanılıyor:
    - Daha az bağımlılık
    - Daha kolay debug
    - SIGINT ile temiz kapatma

    Parameters:
    ----------
    pipeline : HybridTradingPipeline
        Çalıştırılacak pipeline
    interval_minutes : int
        Çalışma aralığı (dakika), varsayılan 60
    """
    pipeline._is_running = True

    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info(f"\n🛑 Sinyal alındı ({signum}). Scheduler durduruluyor...")
        pipeline._is_running = False

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)   # kill komutu

    logger.info(f"\n{'='*70}")
    logger.info(f"⏰ SCHEDULER BAŞLATILDI")
    logger.info(f"   Aralık: {interval_minutes} dakika")
    logger.info(f"   Mode: {'🧪 DRY RUN' if pipeline.dry_run else '🔴 CANLI'}")
    logger.info(f"   Durdurma: Ctrl+C veya SIGTERM")
    logger.info(f"{'='*70}")

    # Bakiye başlat (scheduler başlangıcında bir kez)
    if not pipeline._init_balance():
        logger.error("❌ Bakiye başlatılamadı. Scheduler durduruluyor.")
        return

    while pipeline._is_running:
        try:
            # Kill switch kontrolü
            if pipeline._kill_switch:
                logger.critical("🚨 Kill switch aktif — scheduler durduruluyor")
                break

            # Ardışık hata kontrolü (circuit breaker)
            if pipeline._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error(
                    f"⚠️ {MAX_CONSECUTIVE_ERRORS} ardışık hata! "
                    f"{ERROR_COOLDOWN_SECONDS}s bekleniyor..."
                )
                time.sleep(ERROR_COOLDOWN_SECONDS)
                pipeline._consecutive_errors = 0  # Cooldown sonrası sıfırla
                continue

            # ---- DÖNGÜ ÇALIŞTIR ----
            report = pipeline.run_cycle()

            # Sonraki çalışma zamanı
            next_run = datetime.now() + timedelta(minutes=interval_minutes)
            logger.info(f"\n⏰ Sonraki döngü: {next_run.strftime('%H:%M:%S')}")
            logger.info(f"   ({interval_minutes} dakika bekleniyor...)")

            # Bekleme (her saniye kontrol ederek — temiz kapatma için)
            wait_seconds = interval_minutes * 60
            for _ in range(wait_seconds):
                if not pipeline._is_running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n🛑 Scheduler Ctrl+C ile durduruldu")
            break
        except Exception as e:
            pipeline._consecutive_errors += 1
            logger.exception(f"❌ Scheduler döngü hatası: {e}")
            time.sleep(60)                     # Hata sonrası 1 dakika bekle

    logger.info("🏁 Scheduler temiz şekilde kapatıldı.")


# =============================================================================
# CLI ARGÜMAN PARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Komut satırı argümanlarını parse et.

    Kullanım örnekleri:
      python main.py                      # Tek seferlik full pipeline
      python main.py --schedule           # Sürekli çalışma (60dk)
      python main.py --schedule -i 15     # 15 dakikada bir
      python main.py --dry-run            # Paper trade
      python main.py --symbol SOL         # Tek coin analiz
      python main.py --top 10             # Top 10 coin
      python main.py --live               # Canlı işlem (DİKKAT!)
    """
    parser = argparse.ArgumentParser(
        description=f'Hybrid Crypto Trading Bot v{VERSION}',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Çalışma modu
    parser.add_argument(
        '--schedule', '-s',
        action='store_true',
        help='Sürekli çalışma modu (periyodik döngü)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Çalışma aralığı dakika cinsinden (varsayılan: 60)'
    )

    # İşlem modu
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Paper trade modu — emir göndermez (varsayılan)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='🔴 CANLI İŞLEM — gerçek emir gönderir (DİKKAT!)'
    )

    # Analiz parametreleri
    parser.add_argument(
        '--symbol', '-sym',
        type=str,
        default=None,
        help='Tek coin analiz (örn: SOL, BTC, ETH)'
    )
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=MAX_COINS_PER_CYCLE,
        help=f'Analiz edilecek coin sayısı (varsayılan: {MAX_COINS_PER_CYCLE})'
    )

    # Telegram
    parser.add_argument(
        '--no-telegram',
        action='store_true',
        help='Telegram bildirimlerini kapat'
    )

    # Debug
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Detaylı çıktı'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimum çıktı (sadece hatalar ve sonuçlar)'
    )

    return parser.parse_args()


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================

def main():
    """
    Ana giriş noktası — CLI argümanlarını parse et ve pipeline'ı başlat.
    """
    args = parse_args()

    # ---- YAPILANDIRMA ----
    dry_run = not args.live                    # --live verilmediyse DRY RUN
    verbose = not args.quiet                   # --quiet verilmediyse VERBOSE

    # Log seviyesi ayarla
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Banner
    print(f"\n{'='*70}")
    print(f"  🚀 HYBRID CRYPTO TRADING BOT v{VERSION}")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🔧 Mode: {'🧪 DRY RUN' if dry_run else '🔴 CANLI İŞLEM'}")
    if args.schedule:
        print(f"  ⏰ Scheduler: Her {args.interval} dakikada bir")
    if args.symbol:
        print(f"  🎯 Hedef: {args.symbol.upper()}")
    print(f"{'='*70}\n")

    # ---- CANLI MOD UYARISI ----
    if not dry_run:
        print("⚠️  CANLI İŞLEM MODU AKTİF — Gerçek emir gönderilecek!")
        print("    Devam etmek için 'EVET' yazın: ", end="")
        confirm = input().strip()
        if confirm != "EVET":
            print("❌ İptal edildi.")
            sys.exit(0)

    # ---- CONFIG DURUMU ----
    cfg.print_status()

    # ---- PİPELİNE BAŞLAT ----
    pipeline = HybridTradingPipeline(
        dry_run=dry_run,
        top_n=args.top,
        verbose=verbose,
    )

    # Telegram devre dışı bırakma
    if args.no_telegram:
        pipeline.notifier = TelegramNotifier(token="", chat_id="")

    # ---- ÇALIŞTIRMA MODU ----
    if args.symbol:
        # TEK COİN ANALİZ
        result = pipeline.analyze_single(args.symbol)
        sys.exit(0 if result and result.status in ("analyzed", "executed") else 1)

    elif args.schedule:
        # SCHEDULER (sürekli çalışma)
        run_scheduler(pipeline, interval_minutes=args.interval)

    else:
        # TEK DÖNGÜ (varsayılan)
        if not pipeline._init_balance():
            logger.error("❌ Bakiye başlatılamadı")
            sys.exit(1)

        report = pipeline.run_cycle()
        sys.exit(0 if report.status != CycleStatus.ERROR else 1)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
