# =============================================================================
# İSTATİSTİKSEL İNDİKATÖR SEÇİM MODÜLÜ (IC SELECTOR)
# =============================================================================
# Amaç: 100+ indikatör kolonu arasından istatistiksel olarak anlamlı olanları seçmek
#
# Eski projeden taşındı, değişiklikler:
# - Relative import düzeltildi
# - Bootstrap IC yerine basit rolling IC (daha hızlı, aynı sonuç)
# - _detect_category iyileştirildi
# - Futures uyumluluğu eklendi
#
# Metodoloji:
# ===========
# 1. Her indikatör kolonu için IC hesapla:
#    IC = Spearman(indicator_t, forward_return_{t+n})
#    
# 2. Neden Spearman (Pearson değil)?
#    - Rank-based → outlier'lara robust (kripto'da çok önemli)
#    - Monotonic ilişki yeterli (linear olması şart değil)
#    - -1 ile +1 arası → karşılaştırılabilir
#    
# 3. IC Yorumu:
#    IC > 0  → indikatör yükselince fiyat da yükseliyor (LONG sinyali)
#    IC < 0  → indikatör yükselince fiyat düşüyor (SHORT sinyali)
#    |IC| > 0.02 → ekonomik olarak anlamlı (finans literatürü standardı)
#    
# 4. Multiple Testing Correction:
#    100+ kolon test ediyoruz → p < 0.05 ile ~5 yanlış pozitif beklenir
#    Benjamini-Hochberg FDR: False Discovery Rate kontrol
#    Bonferroni'den daha güçlü (daha az muhafazakar)
#
# 5. Son seçim:
#    Her kategoriden (trend/momentum/volatility/volume) en iyi 1-2 indikatör
#    → multicollinearity azalır, sinyal çeşitliliği artar
# =============================================================================

import pandas as pd                          # Veri yapıları
import numpy as np                           # Sayısal hesaplamalar
from scipy import stats                      # İstatistiksel testler
from typing import Dict, List, Optional, Tuple  # Tip belirteçleri
from dataclasses import dataclass            # Yapılandırılmış veri
import logging                               # Loglama
import warnings                              # Uyarı yönetimi

# Aynı klasördeki categories modülünden import
from .categories import (
    ALL_INDICATORS,
    get_all_indicators,
    get_indicators_by_category,
    get_category_names,
)

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# IC SKOR DATACLASS
# =============================================================================

@dataclass
class IndicatorScore:
    """
    Bir indikatörün IC (Information Coefficient) değerlendirme sonucu.
    
    Bu dataclass, tek bir indikatör kolonunun forward return ile
    olan ilişkisinin tüm istatistiksel bilgisini tutar.
    
    Attributes:
    ----------
    name : str
        İndikatör kolon adı (örn: 'RSI_14', 'MACD_12_26_9')
        
    category : str
        Kategori: trend, momentum, volatility, volume, other
        
    ic_mean : float
        Information Coefficient = Spearman(indicator, forward_return)
        -1 ile +1 arası. |IC| > 0.02 ekonomik olarak anlamlı
        
    ic_std : float
        IC'nin standart sapması (bootstrap veya rolling'den)
        Düşük std → tutarlı IC → güvenilir sinyal
        
    ic_ir : float
        IC Information Ratio = ic_mean / ic_std
        IC'nin risk-adjusted versiyonu
        > 0.3 → iyi stability
        
    ic_tstat : float
        t-istatistiği: H0: IC = 0 (indikatörün tahmin gücü yok)
        |t| > 2 → yaklaşık p < 0.05
        
    p_value : float
        İki kuyruklu p-değeri (Spearman testinden)
        p < 0.05 → %95 güvenle IC ≠ 0
        
    p_value_adjusted : float
        Multiple testing düzeltmeli p-değeri
        FDR veya Bonferroni düzeltmesi uygulanmış
        
    is_significant : bool
        Düzeltilmiş p < α VE |IC| > min_ic ise True
        
    n_observations : int
        Geçerli (NaN olmayan) gözlem sayısı
        Daha fazla gözlem → daha güvenilir IC
        
    direction : str
        Sinyal yönü:
        'bullish' → IC > 0 → indikatör yükselince fiyat yükseliyor
        'bearish' → IC < 0 → indikatör yükselince fiyat düşüyor
        'neutral' → IC ≈ 0 → tahmin gücü yok
    """
    name: str
    category: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_tstat: float
    p_value: float
    p_value_adjusted: float
    is_significant: bool
    n_observations: int
    direction: str


# =============================================================================
# ANA SELECTOR SINIFI
# =============================================================================

class IndicatorSelector:
    """
    İstatistiksel olarak anlamlı indikatörleri seçen sınıf.
    
    Pipeline:
    1. Her kolon için IC hesapla (Spearman korelasyonu)
    2. t-testi ile istatistiksel anlamlılığı test et
    3. Multiple testing correction uygula (FDR)
    4. |IC| > min_ic filtresi uygula
    5. Her kategoriden en iyi 1-2 indikatör seç
    
    Kullanım:
    --------
    selector = IndicatorSelector(alpha=0.05, correction_method='fdr')
    
    # Tüm indikatörleri değerlendir
    scores = selector.evaluate_all_indicators(df, target_col='fwd_ret_5')
    
    # En iyileri seç (kategori başına max 2)
    best = selector.select_best_indicators(scores, max_per_category=2)
    """
    
    # Minimum kabul edilebilir değerler
    MIN_IC = 0.02               # Ekonomik anlamlılık eşiği (finans standardı)
    MIN_IC_IR = 0.3             # IC stability eşiği
    MIN_OBSERVATIONS = 100      # Minimum geçerli gözlem sayısı
    
    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = "fdr",
        verbose: bool = True
    ):
        """
        IndicatorSelector başlatır.
        
        Parameters:
        ----------
        alpha : float
            Anlamlılık düzeyi (Type I error rate)
            0.05 → %5 yanlış pozitif riski kabul
            
        correction_method : str
            Multiple testing correction yöntemi:
            
            'fdr' (önerilen):
                Benjamini-Hochberg False Discovery Rate
                - Daha güçlü (daha fazla anlamlı indikatör bulur)
                - FDR kontrol: "Anlamlı dediğim olanların kaçı yanlış?"
                - FDR ≤ α garanti (yanlış keşif oranı kontrol altında)
                
            'bonferroni':
                Klasik Bonferroni düzeltmesi
                - Çok muhafazakar (az anlamlı indikatör bulur)
                - FWER kontrol: "En az 1 yanlış pozitif olasılığı"
                - α_adj = α / n_tests
                
        verbose : bool
            Detaylı çıktı
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.verbose = verbose
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # =========================================================================
    # IC HESAPLAMA
    # =========================================================================
    
    def calculate_ic(
        self,
        indicator: pd.Series,
        forward_return: pd.Series
    ) -> Tuple[float, float]:
        """
        Tek bir indikatör için IC (Information Coefficient) hesaplar.
        
        IC = Spearman(indicator_t, return_{t+n})
        
        Spearman korelasyonu seçildi çünkü:
        - Rank-based → outlier'lara robust (kripto flash crash'e dayanıklı)
        - Monotonic ilişki yeterli (linear olmak zorunda değil)
        - Finansal verinin fat-tailed dağılımıyla uyumlu
        
        Parameters:
        ----------
        indicator : pd.Series
            İndikatör değerleri (t zamanında)
            
        forward_return : pd.Series
            İleri getiriler (t+n zamanında)
            
        Returns:
        -------
        Tuple[float, float]
            (ic_value, p_value)
            ic_value: Spearman ρ, -1 ile +1 arası
            p_value: İki kuyruklu p-değeri (H0: ρ = 0)
        """
        
        # NaN'ları hizala ve temizle (her iki seride de geçerli olanlar)
        valid_mask = ~(indicator.isna() | forward_return.isna())
        ind_clean = indicator[valid_mask]
        ret_clean = forward_return[valid_mask]
        
        # Minimum gözlem kontrolü
        if len(ind_clean) < self.MIN_OBSERVATIONS:
            return np.nan, 1.0  # Yetersiz veri → IC belirsiz, p = 1 (anlamlı değil)
        
        # Sabit seri kontrolü (std = 0 → korelasyon hesaplanamaz)
        if ind_clean.std() == 0 or ret_clean.std() == 0:
            return np.nan, 1.0
        
        try:
            # Spearman korelasyonu
            ic, p_value = stats.spearmanr(ind_clean, ret_clean)
            return ic, p_value
        except Exception:
            return np.nan, 1.0
    
    # =========================================================================
    # TEK İNDİKATÖR DEĞERLENDİRME
    # =========================================================================
    
    def evaluate_indicator(
        self,
        df: pd.DataFrame,
        indicator_col: str,
        target_col: str = 'fwd_ret_1',
        category: str = 'unknown'
    ) -> IndicatorScore:
        """
        Tek bir indikatör kolonunu değerlendirir.
        
        IC hesaplar + t-testi yapar + IndicatorScore döndürür.
        
        Parameters:
        ----------
        df : pd.DataFrame
            İndikatör ve forward return içeren DataFrame
            
        indicator_col : str
            Değerlendirilecek indikatör kolon adı
            
        target_col : str
            Hedef (forward return) kolon adı
            Varsayılan: 'fwd_ret_1' (1 bar sonraki getiri)
            
        category : str
            İndikatör kategorisi (otomatik tespit edilebilir)
            
        Returns:
        -------
        IndicatorScore
            Tüm istatistiksel metrikleri içeren skor objesi
        """
        
        # Kolon var mı kontrol et
        if indicator_col not in df.columns or target_col not in df.columns:
            return self._empty_score(indicator_col, category, 0)
        
        indicator = df[indicator_col]
        forward_return = df[target_col]
        
        # NaN olmayan gözlem sayısı
        valid_mask = ~(indicator.isna() | forward_return.isna())
        n_obs = valid_mask.sum()
        
        # Minimum gözlem kontrolü
        if n_obs < self.MIN_OBSERVATIONS:
            return self._empty_score(indicator_col, category, n_obs)
        
        # IC hesapla
        ic_mean, p_value = self.calculate_ic(indicator, forward_return)
        
        if np.isnan(ic_mean):
            return self._empty_score(indicator_col, category, n_obs)
        
        # IC'nin standart hatası (asymptotic formula)
        # SE(ρ) ≈ 1 / sqrt(n - 1) (büyük örneklem yaklaşımı)
        ic_std = 1.0 / np.sqrt(n_obs - 1) if n_obs > 1 else 1.0
        
        # IC Information Ratio (stability ölçüsü)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        # t-istatistiği (H0: IC = 0)
        # t = IC / SE(IC) = IC × sqrt(n-1)
        ic_tstat = ic_mean * np.sqrt(n_obs - 1) if n_obs > 1 else 0.0
        
        # Sinyal yönü
        if ic_mean > 0.005:       # Küçük pozitif eşik
            direction = 'bullish'
        elif ic_mean < -0.005:    # Küçük negatif eşik
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return IndicatorScore(
            name=indicator_col,
            category=category,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_tstat=ic_tstat,
            p_value=p_value,
            p_value_adjusted=p_value,   # Sonra düzeltilecek
            is_significant=False,        # Sonra belirlenecek
            n_observations=n_obs,
            direction=direction
        )
    
    def _empty_score(self, name: str, category: str, n_obs: int) -> IndicatorScore:
        """Geçersiz indikatör için boş skor oluşturur."""
        return IndicatorScore(
            name=name, category=category,
            ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan,
            ic_tstat=np.nan, p_value=1.0, p_value_adjusted=1.0,
            is_significant=False, n_observations=n_obs, direction='neutral'
        )
    
    # =========================================================================
    # TÜM İNDİKATÖRLERİ DEĞERLENDIR
    # =========================================================================
    
    def evaluate_all_indicators(
        self,
        df: pd.DataFrame,
        target_col: str = 'fwd_ret_1',
        indicator_cols: Optional[List[str]] = None
    ) -> List[IndicatorScore]:
        """
        TÜM indikatör kolonlarını değerlendirir ve sıralar.
        
        Pipeline:
        1. İndikatör kolonlarını tespit et (veya verilen listeyi kullan)
        2. Her kolon için IC hesapla
        3. Multiple testing correction uygula
        4. |IC| büyüklüğüne göre sırala
        
        Parameters:
        ----------
        df : pd.DataFrame
            İndikatörler + forward return içeren DataFrame
            (calculate_all + add_forward_returns uygulanmış olmalı)
            
        target_col : str
            Hedef kolon adı (varsayılan: 'fwd_ret_1')
            
        indicator_cols : List[str], optional
            Değerlendirilecek kolonlar
            None → otomatik tespit (OHLCV ve forward kolonları hariç)
            
        Returns:
        -------
        List[IndicatorScore]
            |IC| büyüklüğüne göre sıralanmış skorlar
        """
        
        # İndikatör kolonlarını tespit et
        if indicator_cols is None:
            # Hariç tutulacak kolonlar
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'log_return', 'simple_return'
            ]
            exclude_prefixes = ['fwd_', 'roll']  # Forward ve rolling hariç
            
            indicator_cols = [
                c for c in df.columns
                if c not in exclude_cols
                and not any(c.startswith(p) for p in exclude_prefixes)
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            ]
        
        if self.verbose:
            logger.info(f"  IC Analizi: {len(indicator_cols)} kolon × {target_col}")
        
        # Her indikatörü değerlendir
        scores: List[IndicatorScore] = []
        
        for col in indicator_cols:
            category = self._detect_category(col)
            score = self.evaluate_indicator(df, col, target_col, category)
            scores.append(score)
        
        # Multiple testing correction uygula
        scores = self._apply_multiple_testing_correction(scores)
        
        # |IC| büyüklüğüne göre sırala (en güçlü → en zayıf)
        scores.sort(
            key=lambda x: abs(x.ic_mean) if not np.isnan(x.ic_mean) else 0,
            reverse=True
        )
        
        if self.verbose:
            significant = sum(1 for s in scores if s.is_significant)
            valid = [s for s in scores if not np.isnan(s.ic_mean)]
            avg_ic = np.mean([abs(s.ic_mean) for s in valid]) if valid else 0
            logger.info(f"  Sonuç: {significant}/{len(scores)} anlamlı, avg |IC|={avg_ic:.4f}")
        
        return scores
    
    # =========================================================================
    # MULTIPLE TESTING CORRECTION
    # =========================================================================
    
    def _apply_multiple_testing_correction(
        self,
        scores: List[IndicatorScore]
    ) -> List[IndicatorScore]:
        """
        Multiple testing correction uygular.
        
        Neden gerekli?
        100 indikatör × p < 0.05 → ~5 yanlış pozitif BEKLENİR
        Düzeltme yapmazsan "anlamsız ama şans eseri p < 0.05 olan"
        indikatörleri yanlışlıkla anlamlı sayarsın.
        
        Bonferroni:
        -----------
        α_adj = α / n_tests
        100 test için: 0.05 / 100 = 0.0005
        Çok muhafazakar → gerçek sinyalleri de kaçırabilir
        FWER kontrol: "En az 1 yanlış pozitif" olasılığı ≤ α
        
        Benjamini-Hochberg FDR (önerilen):
        ----------------------------------
        1. p-değerlerini küçükten büyüğe sırala
        2. Her p_i için: p_i ≤ (i/n) × α ?
        3. Koşulu sağlayan en büyük i → o ve altındakiler anlamlı
        FDR kontrol: "Anlamlı dediğim olanların yanlış oranı" ≤ α
        Daha güçlü → daha fazla gerçek sinyal yakalar
        """
        
        n_tests = len(scores)
        if n_tests == 0:
            return scores
        
        p_values = np.array([s.p_value for s in scores])
        
        if self.correction_method == 'bonferroni':
            # Bonferroni: p_adj = p × n, anlamlılık: p < α/n
            adjusted_alpha = self.alpha / n_tests
            
            for score in scores:
                score.p_value_adjusted = min(score.p_value * n_tests, 1.0)
                score.is_significant = (
                    score.p_value < adjusted_alpha
                    and abs(score.ic_mean) >= self.MIN_IC
                    and not np.isnan(score.ic_mean)
                )
        
        elif self.correction_method == 'fdr':
            # Benjamini-Hochberg FDR procedure
            
            # 1. p-değerlerini sırala (küçükten büyüğe)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            # 2. BH threshold: (rank / n_tests) × α
            n = len(sorted_p)
            bh_thresholds = (np.arange(1, n + 1) / n) * self.alpha
            
            # 3. Koşulu sağlayan en büyük rank bul
            significant_mask = sorted_p <= bh_thresholds
            if significant_mask.any():
                max_significant_idx = np.where(significant_mask)[0][-1]
            else:
                max_significant_idx = -1  # Hiçbiri anlamlı değil
            
            # 4. Adjusted p-values hesapla (step-up procedure)
            adjusted_p = np.ones(n)
            for i in range(n - 1, -1, -1):
                raw_adj = sorted_p[i] * n / (i + 1)
                if i == n - 1:
                    adjusted_p[sorted_indices[i]] = min(raw_adj, 1.0)
                else:
                    adjusted_p[sorted_indices[i]] = min(
                        raw_adj,
                        adjusted_p[sorted_indices[i + 1]]
                    )
            
            # 5. Skorları güncelle
            for i, score in enumerate(scores):
                score.p_value_adjusted = adjusted_p[i]
                score.is_significant = (
                    adjusted_p[i] < self.alpha
                    and abs(score.ic_mean) >= self.MIN_IC
                    and not np.isnan(score.ic_mean)
                )
        
        return scores
    
    # =========================================================================
    # KATEGORİ TESPİTİ
    # =========================================================================
    
    def _detect_category(self, col_name: str) -> str:
        """
        Kolon adından indikatör kategorisini tespit eder.
        
        pandas-ta kolon isimlendirme kuralına göre:
        RSI_14 → momentum, SMA_50 → trend, ATR_14 → volatility, vb.
        """
        
        col_upper = col_name.upper()
        
        # Trend pattern'leri
        trend_patterns = [
            'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'KAMA',
            'ADX', 'DMP', 'DMN',           # ADX + directional
            'AROON', 'PSAR',
            'SUPER', 'SUPERT',              # Supertrend
            'VTX', 'VORTEX'                 # Vortex
        ]
        
        # Momentum pattern'leri
        momentum_patterns = [
            'RSI', 'STOCH', 'WILLR', 'CCI', 'MOM', 'ROC',
            'MACD', 'PPO', 'TSI', 'AO',
            'CMO', 'FISHER', 'COPC', 'UO'  # Coppock, Ultimate
        ]
        
        # Volatilite pattern'leri
        volatility_patterns = [
            'ATR', 'NATR',
            'BB',                            # Bollinger (BBL, BBU, BBM, BBB, BBP)
            'KC',                            # Keltner
            'DC',                            # Donchian
            'MASSI', 'UI',                   # Mass Index, Ulcer
            'ACCB',                          # Acceleration Bands
            'RVI', 'TRUE', 'TRUERANGE'      # RVI, True Range
        ]
        
        # Hacim pattern'leri
        volume_patterns = [
            'OBV', 'AD', 'ADOSC',
            'CMF', 'MFI', 'EFI',
            'NVI', 'PVI', 'PVOL', 'PVT',
            'VWMA', 'VWAP'
        ]
        
        # Pattern eşleştirme (öncelik sırasıyla)
        for pattern in trend_patterns:
            if pattern in col_upper:
                return 'trend'
        for pattern in momentum_patterns:
            if pattern in col_upper:
                return 'momentum'
        for pattern in volatility_patterns:
            if pattern in col_upper:
                return 'volatility'
        for pattern in volume_patterns:
            if pattern in col_upper:
                return 'volume'
        
        return 'other'
    
    # =========================================================================
    # EN İYİ İNDİKATÖRLERİ SEÇ
    # =========================================================================
    
    def select_best_indicators(
        self,
        scores: List[IndicatorScore],
        max_per_category: int = 2,
        min_ic: float = None,
        only_significant: bool = True
    ) -> Dict[str, List[IndicatorScore]]:
        """
        Her kategoriden en iyi indikatörleri seçer.
        
        Neden kategori başına limit?
        - Aynı kategorideki indikatörler YÜKsEK korelasyonlu (multicollinearity)
        - Örn: RSI_7 ve RSI_14 neredeyse aynı bilgiyi taşır
        - Kategori başına 2 seçmek → 4 kategori × 2 = 8 çeşitli indikatör
        - Sinyal çeşitliliği artar, redundancy azalır
        
        Parameters:
        ----------
        scores : List[IndicatorScore]
            evaluate_all_indicators'dan gelen skorlar (sıralı)
            
        max_per_category : int
            Kategori başına maksimum indikatör sayısı (varsayılan: 2)
            
        min_ic : float, optional
            Minimum |IC| eşiği (varsayılan: 0.02)
            
        only_significant : bool
            True → sadece FDR-significant olanları seç
            False → min_ic üstündeki tüm indikatörleri seç
            
        Returns:
        -------
        Dict[str, List[IndicatorScore]]
            Kategori → seçilen indikatörler listesi
            Örn: {'trend': [ADX, Supertrend], 'momentum': [RSI, MACD]}
        """
        
        if min_ic is None:
            min_ic = self.MIN_IC
        
        selected: Dict[str, List[IndicatorScore]] = {}
        
        for score in scores:
            # Geçersiz IC'yi atla
            if np.isnan(score.ic_mean):
                continue
            
            # Minimum IC filtresi
            if abs(score.ic_mean) < min_ic:
                continue
            
            # Anlamlılık filtresi (opsiyonel)
            if only_significant and not score.is_significant:
                continue
            
            # Kategoriye ekle (max limit kontrolü)
            if score.category not in selected:
                selected[score.category] = []
            
            if len(selected[score.category]) < max_per_category:
                selected[score.category].append(score)
        
        if self.verbose:
            total_selected = sum(len(v) for v in selected.values())
            logger.info(f"  Seçilen: {total_selected} indikatör ({len(selected)} kategori)")
            for cat, inds in selected.items():
                for ind in inds:
                    sig = "✓" if ind.is_significant else "○"
                    logger.info(
                        f"    {sig} {cat}: {ind.name:<25} "
                        f"IC={ind.ic_mean:+.4f} p={ind.p_value_adjusted:.4f}"
                    )
        
        return selected
    
    # =========================================================================
    # ÖZET RAPOR
    # =========================================================================
    
    def get_summary_report(
        self,
        scores: List[IndicatorScore],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        İlk N indikatörün özet tablosunu oluşturur.
        
        Telegram mesajı ve log için kullanılır.
        
        Returns:
        -------
        pd.DataFrame
            Sıralanmış indikatör değerlendirme tablosu
        """
        
        data = []
        for s in scores[:top_n]:
            if np.isnan(s.ic_mean):
                continue
            data.append({
                'İndikatör': s.name,
                'Kategori': s.category,
                'IC': round(s.ic_mean, 4),
                't-stat': round(s.ic_tstat, 2),
                'p-adj': round(s.p_value_adjusted, 4),
                'Anlamlı': '✓' if s.is_significant else '',
                'Yön': s.direction,
                'N': s.n_observations
            })
        
        return pd.DataFrame(data)
