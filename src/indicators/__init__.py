# =============================================================================
# İNDİKATÖR MODÜLÜ (INDICATORS MODULE)
# =============================================================================
# 3 bileşen:
# - categories.py: 58 indikatör tanımı (4 kategori)
# - calculator.py: pandas-ta ile hesaplama motoru
# - selector.py:   IC bazlı istatistiksel seçim (Spearman + FDR)
#
# Kullanım:
# --------
# from indicators import IndicatorCalculator, IndicatorSelector
#
# calc = IndicatorCalculator()
# df = calc.calculate_all(ohlcv_df)
#
# selector = IndicatorSelector()
# scores = selector.evaluate_all_indicators(df)
# best = selector.select_best_indicators(scores)
# =============================================================================

from .categories import (
    IndicatorConfig,
    ALL_INDICATORS,
    get_all_indicators,
    get_indicators_by_category,
    get_category_names,
    get_indicator_count,
    get_total_output_columns,
)

from .calculator import IndicatorCalculator

from .selector import IndicatorScore, IndicatorSelector


__all__ = [
    'IndicatorConfig',
    'ALL_INDICATORS',
    'get_all_indicators',
    'get_indicators_by_category',
    'get_category_names',
    'get_indicator_count',
    'get_total_output_columns',
    'IndicatorCalculator',
    'IndicatorScore',
    'IndicatorSelector',
]

__version__ = '2.0.0'
