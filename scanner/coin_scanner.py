# =============================================================================
# DİNAMİK COİN TARAYICI (COIN SCANNER)
# =============================================================================
# Amaç: Bitget USDT-M Futures çiftlerini tarayıp IC analize gönderilecek
#        en iyi coin'leri otomatik seçmek.
#
# Pipeline:
# 1. Tüm USDT Futures çiftlerini çek (500+)
# 2. Batch ticker ile hacim/spread/değişim verisi al (TEK API çağrısı)
# 3. Filtreler uygula (min hacim, max spread, min fiyat)
# 4. Composite skor hesapla (percentile rank bazlı)
# 5. Top N coin döndür → IC analize gönder
#
# İstatistiksel Gerekçe:
# - Düşük hacimli coinlerde IC güvenilmez (thin trading → noisy returns)
# - Yüksek spread → execution slippage, backtest-live gap büyür
# - Volatilite filtreleme → IC'nin capture edebileceği sinyal varlığı
# - Percentile rank → outlier-robust normalization (BTC hacmi 100x olabilir)
#
# Kullanım:
# --------
# from scanner.coin_scanner import CoinScanner
# scanner = CoinScanner()
# top_coins = scanner.scan()       # → [CoinScanResult, ...]
# symbols = scanner.get_symbols()  # → ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]
# =============================================================================

import sys                                     # Path ayarları
import time                                    # Performans ölçümü
import logging                                 # Log yönetimi
import numpy as np                             # Sayısal hesaplamalar (percentile rank)
import pandas as pd                            # DataFrame (rapor çıktısı)
from pathlib import Path                       # Platform-bağımsız dosya yolları
from typing import Dict, List, Optional        # Tip belirteçleri
from dataclasses import dataclass              # Yapılandırılmış veri sınıfı
from datetime import datetime, timezone        # Zaman damgası (cache için)

# Proje config import
sys.path.insert(0, str(Path(__file__).parent.parent))  # → src/
from config import cfg, get_setting            # Merkezi config + yaml okuyucu

# Data modülü import (BitgetFetcher)
from data.fetcher import BitgetFetcher         # CCXT üzerinden Bitget API

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# TARAMA SONUCU DATACLASS
# =============================================================================

@dataclass
class CoinScanResult:
    """
    Tek bir coin'in tarama sonucu.
    
    IC analize gönderilecek coin seçiminde kullanılır.
    Composite score → hangi coin'lerin IC analizi için en uygun olduğunu belirler.
    
    Attributes:
    ----------
    symbol : str
        Bitget Futures sembolü (örn: 'BTC/USDT:USDT')
    coin : str
        Kısa isim (örn: 'BTC') — Telegram mesajlarında kullanılır
    price : float
        Son fiyat (USDT cinsinden)
    volume_24h : float
        24 saatlik USDT cinsinden işlem hacmi
        → IC güvenilirliği ile doğrudan ilişkili (sample quality)
    change_24h : float
        24 saatlik yüzde değişim
    spread_pct : float
        Bid-ask spread yüzdesi → likidite göstergesi
        Düşük = iyi execution, backtest-live gap küçük
    volatility : float
        24h (high-low)/midprice yüzdesi
        → IC'nin capture edebileceği sinyal büyüklüğü
    composite_score : float
        Ağırlıklı composite skor (0-100)
        Percentile rank bazlı → outlier-robust
    passed_filters : bool
        Tüm hard filtrelerden geçti mi?
    filter_reason : str
        Elenme nedeni (boş string ise geçti)
    """
    symbol: str                                # 'BTC/USDT:USDT'
    coin: str                                  # 'BTC'
    price: float                               # Son fiyat ($)
    volume_24h: float                          # 24h USDT hacim
    change_24h: float                          # 24h % değişim
    spread_pct: float                          # Bid-ask spread (%)
    volatility: float                          # 24h range / mid (%)
    composite_score: float = 0.0               # Composite skor (0-100)
    passed_filters: bool = True                # Filtrelerden geçti mi?
    filter_reason: str = ""                    # Elenme nedeni


# =============================================================================
# ANA TARAYICI SINIFI
# =============================================================================

class CoinScanner:
    """
    Bitget USDT-M Futures çiftlerini tarayıp IC analiz için en uygun
    coin'leri seçen dinamik tarayıcı.
    
    Filtreleme Mantığı (sırasıyla):
    ------------------------------
    1. Blacklist   → Stablecoin, leveraged token, sorunlu coinleri eler
    2. Hacim       → Düşük hacim = thin trading = unreliable IC
    3. Spread      → Yüksek spread = execution cost, backtest-live gap
    4. Fiyat       → $0'a çok yakın coinler = veri kalitesi sorunu
    
    Skorlama (percentile rank bazlı):
    --------------------------------
    score = w_vol × PercentileRank(volume)
          + w_volatility × PercentileRank(volatility)
          + w_liquidity × PercentileRank(1/spread)
    
    Neden percentile rank?
    → Outlier-robust (BTC hacmi diğerlerinin 100x'i olabilir)
    → [0, 100] normalize → karşılaştırılabilir
    → Ordinal bilgi yeterli (exact magnitude gerekmez)
    """
    
    # =========================================================================
    # SKORLAMA AĞIRLIKLARI
    # =========================================================================
    # Composite skor ağırlıkları — toplamı 1.0
    # Hacim en önemli: IC sample quality'si doğrudan volume'a bağlı
    # Volatilite ikinci: düşük vol = yakalayacak sinyal yok
    # Likidite üçüncü: execution quality, slippage riski
    
    DEFAULT_WEIGHTS = {
        'volume': 0.45,                        # %45 — IC güvenilirliği için kritik
        'volatility': 0.30,                    # %30 — sinyal büyüklüğü
        'liquidity': 0.25,                     # %25 — execution quality
    }
    
    # =========================================================================
    # BLACKLIST (IC ANALİZE UYGUN OLMAYAN COİNLER)
    # =========================================================================
    # Stablecoinler: Volatilite ~0, IC anlamsız
    # Leveraged token'lar: Yapay fiyat hareketi, IC yanıltıcı
    # Sorunlu coinler: Delist riski, veri kalitesi sorunu
    
    BLACKLIST_KEYWORDS = [
        'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'FDUSD',  # Stablecoinler
        'UP', 'DOWN', 'BULL', 'BEAR',                       # Leveraged token kısaltmaları
        '3L', '3S', '5L', '5S',                             # Leveraged token varyantları
        'BTTC',                                               # Bilinen sorunlu coin
    ]
    
    def __init__(
        self,
        fetcher: Optional[BitgetFetcher] = None,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """
        CoinScanner başlatır.
        
        Parameters:
        ----------
        fetcher : BitgetFetcher, optional
            Veri çekme nesnesi. None ise yeni oluşturulur.
            Dışarıdan verme avantajı: aynı CCXT exchange bağlantısını paylaşır
            (market data tekrar yüklenmez → ~2s tasarruf).
            
        weights : Dict[str, float], optional
            Composite skor ağırlıkları. None ise DEFAULT_WEIGHTS kullanılır.
            Toplamı 1.0 olmalı.
            
        verbose : bool
            True → tarama adımları ve istatistikler loglanır
        """
        self.fetcher = fetcher or BitgetFetcher()
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.verbose = verbose
        
        # settings.yaml'dan filtre parametrelerini oku
        # get_setting(): config modülündeki yaml okuyucu (nokta-ayrımlı key path)
        self.min_volume = get_setting('scanner.min_24h_volume_usdt', 5_000_000)
        self.max_spread_pct = get_setting('scanner.max_spread_pct', 0.10)
        self.top_n = get_setting('scanner.top_n_coins', 20)
        self.min_price = 0.0001                # Min fiyat eşiği ($0.0001)
        
        # Cache mekanizması — aynı taramayı 5dk içinde tekrar yapmaz
        self._last_scan: Optional[List[CoinScanResult]] = None
        self._last_scan_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300          # 5 dakika cache ömrü
        
        logger.info(
            f"CoinScanner başlatıldı | "
            f"min_vol=${self.min_volume/1e6:.0f}M | "
            f"max_spread={self.max_spread_pct}% | "
            f"top_n={self.top_n}"
        )
    
    # =========================================================================
    # ANA TARAMA FONKSİYONU
    # =========================================================================
    
    def scan(
        self,
        top_n: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[CoinScanResult]:
        """
        Tam tarama pipeline'ını çalıştırır ve top coin'leri döndürür.
        
        Pipeline Akışı:
        1. get_all_usdt_futures() → 500+ sembol listesi
        2. fetch_tickers()        → tek çağrıda TÜM ticker verileri
        3. build_scan_results()   → ham metrikleri hesapla
        4. apply_filters()        → hard filter'lardan geçir
        5. calculate_scores()     → percentile rank bazlı composite skor
        6. sort & select top N
        
        Parameters:
        ----------
        top_n : int, optional
            Seçilecek coin sayısı. None ise config'den okunur (varsayılan 20).
            
        force_refresh : bool
            True ise cache'i yoksay, Bitget API'den taze veri çek.
            
        Returns:
        -------
        List[CoinScanResult]
            Composite score'a göre azalan sıralanmış top coin'ler.
        """
        top_n = top_n or self.top_n
        
        # ---- Cache kontrolü (5dk TTL) ----
        if self._is_cache_valid() and not force_refresh:
            logger.info(f"Cache kullanılıyor (TTL: {self._cache_ttl_seconds}s)")
            return self._last_scan[:top_n]
        
        scan_start = time.time()
        
        if self.verbose:
            logger.info("=" * 60)
            logger.info("🔍 COİN TARAMASI BAŞLIYOR")
            logger.info("=" * 60)
        
        # ---- ADIM 1: Tüm USDT Futures çiftlerini çek ----
        all_symbols = self.fetcher.get_all_usdt_futures()
        
        if self.verbose:
            logger.info(f"  [1/5] {len(all_symbols)} USDT Futures çifti bulundu")
        
        # Blacklist filtresi (stablecoin, leveraged token eleme)
        symbols = self._apply_blacklist(all_symbols)
        
        if self.verbose:
            logger.info(f"  [1/5] Blacklist sonrası: {len(symbols)} çift "
                        f"({len(all_symbols) - len(symbols)} elendi)")
        
        # ---- ADIM 2: Batch ticker verisi çek (tek API çağrısı) ----
        tickers = self._fetch_all_tickers(symbols)
        
        if self.verbose:
            logger.info(f"  [2/5] {len(tickers)} ticker çekildi")
        
        if not tickers:
            logger.error("❌ Ticker verisi alınamadı! API bağlantısını kontrol edin.")
            return []
        
        # ---- ADIM 3: CoinScanResult'ları oluştur ----
        results = self._build_scan_results(tickers)
        
        if self.verbose:
            logger.info(f"  [3/5] {len(results)} coin analiz edildi")
        
        # ---- ADIM 4: Hard filtreler uygula ----
        self._apply_filters(results)
        passed = [r for r in results if r.passed_filters]
        failed = [r for r in results if not r.passed_filters]
        
        if self.verbose:
            logger.info(f"  [4/5] Filtre sonrası: {len(passed)}/{len(results)} geçti")
            self._log_filter_stats(failed)
        
        # ---- ADIM 5: Composite skorlama + sıralama ----
        scored = self._calculate_scores(passed)
        scored.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Top N seç
        top_coins = scored[:top_n]
        
        # Cache güncelle
        self._last_scan = scored
        self._last_scan_time = datetime.now(timezone.utc)
        
        elapsed = time.time() - scan_start
        
        if self.verbose:
            logger.info(f"  [5/5] Top {len(top_coins)} coin seçildi ({elapsed:.1f}s)")
            logger.info("")
            self._print_top_table(top_coins)
        
        return top_coins
    
    # =========================================================================
    # CACHE KONTROL
    # =========================================================================
    
    def _is_cache_valid(self) -> bool:
        """
        Cache'in hâlâ geçerli olup olmadığını kontrol eder.
        
        TTL (Time-To-Live): 5 dakika.
        Gerekçe: Ticker verisi hızlı değişir ama 5dk'da bir taramak yeterli.
        IC analiz zaten 15dk-1saat aralıkla çalışacak.
        """
        if self._last_scan is None or self._last_scan_time is None:
            return False
        
        age = (datetime.now(timezone.utc) - self._last_scan_time).total_seconds()
        return age < self._cache_ttl_seconds
    
    # =========================================================================
    # BLACKLIST FİLTRESİ
    # =========================================================================
    
    def _apply_blacklist(self, symbols: List[str]) -> List[str]:
        """
        Stablecoin ve leveraged token'ları eler.
        
        Sembol formatı: 'BTC/USDT:USDT' → coin = 'BTC'
        Coin adı BLACKLIST_KEYWORDS'den herhangi birini içeriyorsa elenir.
        
        Parameters:
        ----------
        symbols : List[str]
            Tam sembol listesi
            
        Returns:
        -------
        List[str]
            Blacklist'ten geçen semboller
        """
        filtered = []
        
        for symbol in symbols:
            coin = symbol.split('/')[0].upper()   # 'BTC/USDT:USDT' → 'BTC'
            
            # Keyword bazlı kontrol (O(n*k), ama n ve k küçük)
            is_blacklisted = any(kw in coin for kw in self.BLACKLIST_KEYWORDS)
            
            if not is_blacklisted:
                filtered.append(symbol)
        
        return filtered
    
    # =========================================================================
    # BATCH TİCKER ÇEKİM (TEK API ÇAĞRISI)
    # =========================================================================
    
    def _fetch_all_tickers(self, symbols: List[str]) -> Dict:
        """
        Tüm semboller için ticker verisi çeker.
        
        CCXT fetch_tickers(): tek çağrıda TÜM market ticker'larını döndürür.
        500 ayrı API çağrısı yerine 1 çağrı → ~100x daha hızlı.
        
        Parameters:
        ----------
        symbols : List[str]
            İstenen semboller
            
        Returns:
        -------
        Dict
            {symbol: ticker_data} formatında
        """
        try:
            # Bitget fetch_tickers: tek çağrıda tüm market'lar
            all_tickers = self.fetcher.exchange.fetch_tickers()
            
            # Sadece istenen sembolleri döndür (memory efficient)
            return {s: all_tickers[s] for s in symbols if s in all_tickers}
            
        except Exception as e:
            logger.error(f"Batch ticker hatası: {e}")
            return {}
    
    # =========================================================================
    # SCAN RESULT OLUŞTURMA
    # =========================================================================
    
    def _build_scan_results(self, tickers: Dict) -> List[CoinScanResult]:
        """
        Ham ticker verisinden CoinScanResult listesi oluşturur.
        
        Her ticker'dan hesaplanan metrikler:
        - price: Son işlem fiyatı
        - volume_24h: 24h USDT cinsinden hacim (quoteVolume)
        - change_24h: 24h yüzde değişim
        - spread_pct: (ask - bid) / last × 100 → likidite proxy'si
        - volatility: (high - low) / mid_price × 100 → 24h range volatilitesi
        
        Parameters:
        ----------
        tickers : Dict
            CCXT fetch_tickers() çıktısı
            
        Returns:
        -------
        List[CoinScanResult]
            Henüz filtrelenmemiş ham scan sonuçları
        """
        results = []
        
        for symbol, ticker in tickers.items():
            try:
                # Temel fiyat verileri (None → 0 dönüşümü)
                last = ticker.get('last', 0) or 0          # Son fiyat
                bid = ticker.get('bid', 0) or 0            # En iyi alış
                ask = ticker.get('ask', 0) or 0            # En iyi satış
                high = ticker.get('high', 0) or 0          # 24h yüksek
                low = ticker.get('low', 0) or 0            # 24h düşük
                volume_24h = ticker.get('quoteVolume', 0) or 0  # 24h USDT hacim
                change_24h = ticker.get('percentage', 0) or 0   # 24h % değişim
                
                # Geçersiz veri → atla
                if last <= 0 or volume_24h <= 0:
                    continue
                
                # Spread hesaplama: (ask - bid) / last × 100
                # Düşük spread = yüksek likidite = daha iyi execution
                # Spread > 1% olan coinlerde slippage ciddi problem
                if bid > 0 and ask > 0:
                    spread_pct = ((ask - bid) / last) * 100
                else:
                    spread_pct = 999.0                     # Geçersiz → filtrele
                
                # 24h volatilite: (high - low) / midprice × 100
                # → IC'nin yakalayabileceği sinyal büyüklüğünün proxy'si
                # Çok düşük vol (< 0.5%) = sinyal yok
                # Çok yüksek vol (> 30%) = noise dominates
                if high > 0 and low > 0:
                    mid_price = (high + low) / 2
                    volatility = ((high - low) / mid_price) * 100
                else:
                    volatility = 0.0
                
                # Coin kısa adı: 'BTC/USDT:USDT' → 'BTC'
                coin = symbol.split('/')[0]
                
                results.append(CoinScanResult(
                    symbol=symbol,
                    coin=coin,
                    price=last,
                    volume_24h=volume_24h,
                    change_24h=change_24h,
                    spread_pct=spread_pct,
                    volatility=volatility,
                ))
                
            except Exception as e:
                logger.debug(f"Ticker parse hatası {symbol}: {e}")
                continue
        
        return results
    
    # =========================================================================
    # HARD FİLTRELER
    # =========================================================================
    
    def _apply_filters(self, results: List[CoinScanResult]) -> None:
        """
        Hard filter'ları uygular. Geçemeyenlere sebep yazar.
        
        Filtre sırası (en çok eleyenden en az eleyene):
        1. Hacim  → $5M min (IC sample quality)
        2. Spread → %0.10 max (execution quality)
        3. Fiyat  → $0.0001 min (veri kalitesi)
        
        In-place güncelleme: passed_filters ve filter_reason alanları set edilir.
        
        Parameters:
        ----------
        results : List[CoinScanResult]
            Filtrelenmemiş sonuçlar (in-place güncellenir)
        """
        for r in results:
            # Filtre 1: Minimum 24h USDT hacim
            # Gerekçe: Düşük hacim = thin trading = noisy returns = unreliable IC
            if r.volume_24h < self.min_volume:
                r.passed_filters = False
                r.filter_reason = f"volume<${self.min_volume/1e6:.0f}M"
                continue
            
            # Filtre 2: Maksimum bid-ask spread
            # Gerekçe: Yüksek spread = execution cost ↑, backtest-live gap ↑
            if r.spread_pct > self.max_spread_pct:
                r.passed_filters = False
                r.filter_reason = f"spread>{self.max_spread_pct}%"
                continue
            
            # Filtre 3: Minimum fiyat
            # Gerekçe: Çok düşük fiyatlı coinlerde precision sorunu
            if r.price < self.min_price:
                r.passed_filters = False
                r.filter_reason = "price_too_low"
                continue
            
            # Tüm filtrelerden geçti
            r.passed_filters = True
            r.filter_reason = ""
    
    # =========================================================================
    # PERCENTILE RANK BAZLI COMPOSİTE SKORLAMA
    # =========================================================================
    
    def _calculate_scores(self, results: List[CoinScanResult]) -> List[CoinScanResult]:
        """
        Percentile rank bazlı composite skor hesaplar.
        
        Metod:
        ------
        1. Her metrik (volume, volatility, 1/spread) için percentile rank (0-100)
        2. Ağırlıklı toplam → composite score
        
        Neden percentile rank?
        - Outlier-robust: BTC hacmi $50B, altcoin $10M olabilir
        - [0, 100] normalization → metrikler karşılaştırılabilir
        - Ordinal bilgi yeterli (exact magnitude gerekmez)
        
        Likidite hesabı:
        - spread ↓ = likidite ↑ → spread'i tersine çeviriyoruz
        - inverted_spread = max(spread) - spread + ε
        
        Parameters:
        ----------
        results : List[CoinScanResult]
            Filtreden geçmiş sonuçlar
            
        Returns:
        -------
        List[CoinScanResult]
            composite_score güncellenmiş sonuçlar
        """
        if len(results) < 2:
            for r in results:
                r.composite_score = 100.0      # Tek coin → max skor
            return results
        
        # Metrikleri array'e çıkar
        volumes = np.array([r.volume_24h for r in results])
        volatilities = np.array([r.volatility for r in results])
        spreads = np.array([r.spread_pct for r in results])
        
        # Percentile rank hesapla (0-100)
        vol_ranks = self._percentile_rank(volumes)
        volatility_ranks = self._percentile_rank(volatilities)
        
        # Likidite: düşük spread = iyi → tersine çevir
        inverted_spreads = spreads.max() - spreads + 0.001  # +ε: sıfır bölme önle
        liquidity_ranks = self._percentile_rank(inverted_spreads)
        
        # Composite score = ağırlıklı toplam
        w = self.weights
        for i, r in enumerate(results):
            r.composite_score = round(
                w['volume'] * vol_ranks[i] +
                w['volatility'] * volatility_ranks[i] +
                w['liquidity'] * liquidity_ranks[i],
                2
            )
        
        return results
    
    @staticmethod
    def _percentile_rank(arr: np.ndarray) -> np.ndarray:
        """
        Percentile rank hesaplar (0-100 arası).
        
        scipy.stats.rankdata ile ties='average' kullanır.
        
        Formül:
        percentile_i = (rank_i - 1) / (N - 1) × 100
        
        rank=1 → percentile=0, rank=N → percentile=100
        
        Parameters:
        ----------
        arr : np.ndarray
            Sıralanacak değerler
            
        Returns:
        -------
        np.ndarray
            0-100 arası percentile rank'lar
        """
        from scipy.stats import rankdata       # Lazy import (her çağrıda yükleme yok)
        
        n = len(arr)
        if n <= 1:
            return np.array([50.0] * n)        # Tek eleman → median
        
        # rankdata: 1'den N'e sıralar (ties → ortalama rank)
        ranks = rankdata(arr, method='average')
        
        # Percentile'a normalize: (rank - 1) / (N - 1) × 100
        percentiles = (ranks - 1) / (n - 1) * 100
        
        return percentiles
    
    # =========================================================================
    # YARDIMCI: FİLTRE İSTATİSTİKLERİ LOG
    # =========================================================================
    
    def _log_filter_stats(self, failed: List[CoinScanResult]) -> None:
        """Elenme nedenlerini gruplar ve loglar."""
        if not failed:
            return
        
        reasons = {}
        for r in failed:
            reasons[r.filter_reason] = reasons.get(r.filter_reason, 0) + 1
        
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"       Elenen: {reason} ({count} coin)")
    
    # =========================================================================
    # YARDIMCI: TOP COİN TABLOSU LOG
    # =========================================================================
    
    def _print_top_table(self, top_coins: List[CoinScanResult]) -> None:
        """Top coin tablosunu okunabilir formatta loglar."""
        header = (f"  {'#':<4} {'Coin':<10} {'Fiyat':>12} {'24h Hacim':>14} "
                  f"{'24h%':>7} {'Spread%':>9} {'Vol%':>6} {'Skor':>6}")
        
        logger.info("=" * 75)
        logger.info(header)
        logger.info("-" * 75)
        
        for i, c in enumerate(top_coins, 1):
            # Dinamik fiyat formatı (düşük fiyatlı coinler için)
            if c.price >= 100:
                price_str = f"${c.price:>10,.2f}"
            elif c.price >= 1:
                price_str = f"${c.price:>10,.4f}"
            else:
                price_str = f"${c.price:>10,.6f}"
            
            # Hacim formatı (B/M)
            if c.volume_24h >= 1e9:
                vol_str = f"${c.volume_24h/1e9:>6,.1f}B"
            else:
                vol_str = f"${c.volume_24h/1e6:>6,.1f}M"
            
            logger.info(
                f"  {i:<4} {c.coin:<10} {price_str} {vol_str:>14} "
                f"{'📈' if c.change_24h >= 0 else '📉'}{c.change_24h:>+5.1f}% "
                f"{c.spread_pct:>8.4f}% "
                f"{c.volatility:>5.1f} "
                f"{c.composite_score:>5.1f}"
            )
        
        logger.info("=" * 75)
    
    # =========================================================================
    # KOLAY ERİŞİM FONKSİYONLARI
    # =========================================================================
    
    def get_symbols(
        self,
        top_n: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[str]:
        """
        Top coin sembol listesi döndürür (IC analiz modülü için).
        
        main.py bu fonksiyonu çağırarak hangi coin'leri tarayacağını öğrenir:
        >>> symbols = scanner.get_symbols()
        >>> for symbol in symbols:
        ...     ic_analyze(symbol)
        
        Returns:
        -------
        List[str]
            Semboller (örn: ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...])
        """
        return [c.symbol for c in self.scan(top_n=top_n, force_refresh=force_refresh)]
    
    def get_coins(
        self,
        top_n: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[str]:
        """
        Kısa coin isimleri döndürür (Telegram mesajı için).
        
        Returns:
        -------
        List[str]
            Kısa isimler (örn: ['BTC', 'ETH', 'SOL', ...])
        """
        return [c.coin for c in self.scan(top_n=top_n, force_refresh=force_refresh)]
    
    # =========================================================================
    # RAPOR (DataFrame)
    # =========================================================================
    
    def get_report(
        self,
        top_n: Optional[int] = None,
        include_failed: bool = False
    ) -> pd.DataFrame:
        """
        Tarama sonuçlarını pandas DataFrame olarak döndürür.
        
        Debug, Streamlit dashboard ve Telegram raporu için.
        
        Parameters:
        ----------
        top_n : int, optional
            Gösterilecek coin sayısı
        include_failed : bool
            True → filtreden elenenleri de göster (debug için)
            
        Returns:
        -------
        pd.DataFrame
            Tarama sonuç tablosu (skor'a göre sıralı)
        """
        # Taramayı çalıştır (cache varsa kullanır)
        _ = self.scan(top_n=top_n or 100)
        
        # Cache'deki tüm sonuçları kullan
        all_results = self._last_scan or []
        
        data = []
        for r in all_results:
            if not include_failed and not r.passed_filters:
                continue
            
            data.append({
                'Coin': r.coin,
                'Symbol': r.symbol,
                'Fiyat ($)': round(r.price, 6),
                '24h Hacim ($)': round(r.volume_24h, 0),
                '24h Değişim (%)': round(r.change_24h, 2),
                'Spread (%)': round(r.spread_pct, 4),
                'Volatilite (%)': round(r.volatility, 2),
                'Skor': round(r.composite_score, 1),
                'Geçti': '✓' if r.passed_filters else '✗',
                'Elenme Nedeni': r.filter_reason,
            })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            df = df.sort_values('Skor', ascending=False).reset_index(drop=True)
            df.index += 1                      # 1'den başlat
            df.index.name = '#'
        
        return df


# =============================================================================
# BAĞIMSIZ ÇALIŞTIRMA TESTİ
# =============================================================================

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 65)
    print("  🔍 COİN SCANNER — BAĞIMSIZ TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    scanner = CoinScanner(verbose=True)
    
    # Tam tarama
    print("\n[1] Tam tarama başlıyor...")
    top = scanner.scan(top_n=20)
    
    print(f"\n[2] Top {len(top)} coin:")
    for i, c in enumerate(top, 1):
        print(f"   {i:>2}. {c.coin:<8} ${c.price:>12,.2f} | "
              f"Vol: ${c.volume_24h/1e6:>8,.1f}M | "
              f"Skor: {c.composite_score:>5.1f}")
    
    # Sembol listesi
    print(f"\n[3] Sembol listesi (top 10):")
    print(f"   {scanner.get_symbols(top_n=10)}")
    
    # Cache testi
    print(f"\n[4] Cache testi:")
    start = time.time()
    _ = scanner.scan(top_n=20)
    print(f"   Cache süresi: {time.time()-start:.4f}s")
    
    # DataFrame rapor
    print(f"\n[5] DataFrame raporu:")
    report = scanner.get_report(top_n=10)
    if not report.empty:
        print(report[['Coin', 'Fiyat ($)', '24h Hacim ($)', 'Spread (%)', 'Skor']].to_string())
    
    print(f"\n{'=' * 65}")
    print(f"  ✅ TEST TAMAMLANDI")
    print(f"{'=' * 65}")
