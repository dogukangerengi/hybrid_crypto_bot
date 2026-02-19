#!/usr/bin/env python3
"""
main.py _analyze_coin() iç döngüsü düzeltme scripti.

SORUN:
  main.py 3 hayalet method çağırıyor — hepsi except: continue ile yutuluyor:
    1. preprocessor.prepare(df)           → DataPreprocessor'da 'prepare' yok
    2. calculator.add_all_indicators(df)  → Doğrusu: calculate_all(df)
    3. selector.analyze(df, fwd_period)   → IndicatorSelector'da 'analyze' yok

ÇÖZÜM:
  main_backup.py'deki doğru implementasyonu main.py'ye uygular:
    1. calculator.calculate_all(df) 
    2. calculator.add_price_features(df)
    3. calculator.add_forward_returns(df, periods=[fwd_period])
    4. selector.evaluate_all_indicators(df, target_col)
    5. Composite score hesaplama (IC weights ile)
    6. Yön belirleme + rejim tespiti + ATR hesabı

Kullanım:
  cd hybrid_crypto_bot/src
  python fix_main_analyze.py          # Önizleme (dry-run)
  python fix_main_analyze.py --apply  # Uygula
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

MAIN_PY = Path("main.py")

# ═══════════════════════════════════════════════════════════════════════════════
# ESKİ KOD (bulanacak) — for döngüsü gövdesi
# ═══════════════════════════════════════════════════════════════════════════════
# Aranacak pattern: preprocessor.prepare + add_all_indicators + selector.analyze bloğu
OLD_BLOCK_MARKER_START = "for tf, limit in self.timeframes.items():"
OLD_BLOCK_MARKER_END = "if not tf_results:"

# ═══════════════════════════════════════════════════════════════════════════════
# YENİ KOD — main_backup.py'den adapte edilmiş doğru implementasyon
# ═══════════════════════════════════════════════════════════════════════════════

NEW_LOOP_BLOCK = '''            for tf, limit in self.timeframes.items():
                try:
                    # ── 1. OHLCV Veri Çekme ──
                    df = self.fetcher.fetch_ohlcv(full_symbol, timeframe=tf, limit=limit)
                    if df is None or len(df) < 50:
                        continue

                    # ── 2. İndikatör Hesaplama (64+ teknik indikatör) ──
                    df = self.calculator.calculate_all(df)          # pandas-ta ile hesapla
                    df = self.calculator.add_price_features(df)     # log_return, simple_return
                    df = self.calculator.add_forward_returns(       # IC hedef değişkeni
                        df, periods=[1, self.fwd_period]
                    )

                    # ── 3. IC Analiz — Spearman + FDR ──
                    target_col = f'fwd_ret_{self.fwd_period}'
                    scores = self.selector.evaluate_all_indicators(
                        df, target_col=target_col
                    )

                    # ── 4. Anlamlı İndikatörleri Filtrele ──
                    valid_categories = ['trend', 'momentum', 'volatility', 'volume']
                    sig_scores = [
                        s for s in scores
                        if not np.isnan(s.ic_mean)           # NaN kontrolü
                        and abs(s.ic_mean) > 0.02            # Noise threshold
                        and s.category in valid_categories   # Sadece bilinen kategoriler
                    ]

                    if not sig_scores:
                        continue                             # Bu TF'de anlamlı sinyal yok

                    # ── 5. Composite Score Hesaplama ──
                    # En yüksek |IC| (top sinyal gücü)
                    top_score = max(sig_scores, key=lambda x: abs(x.ic_mean))
                    top_ic_val = abs(top_score.ic_mean)

                    # Ortalama |IC| (genel sinyal seviyesi)
                    avg_ic = np.mean([abs(s.ic_mean) for s in sig_scores])

                    # Yön tutarlılığı (bullish/bearish consensus)
                    pos_count = sum(1 for s in sig_scores if s.ic_mean > 0)
                    neg_count = sum(1 for s in sig_scores if s.ic_mean < 0)
                    consistency = max(pos_count, neg_count) / len(sig_scores)

                    # Dominant yön belirleme
                    if neg_count > pos_count * 1.5:
                        direction = 'SHORT'
                    elif pos_count > neg_count * 1.5:
                        direction = 'LONG'
                    else:
                        direction = 'NEUTRAL'

                    # ── 6. Market Rejimi Tespiti (ADX bazlı) ──
                    regime = 'unknown'
                    if hasattr(self, '_detect_regime'):
                        regime = self._detect_regime(df)
                    elif 'ADX_14' in df.columns:
                        adx_val = float(df['ADX_14'].dropna().iloc[-1]) if not df['ADX_14'].dropna().empty else 20
                        if adx_val > 25:
                            regime = 'trending'
                        elif adx_val < 15:
                            regime = 'ranging'
                        else:
                            regime = 'transitioning'

                    # ── 7. Normalize + Ağırlıklı Composite (0-100) ──
                    top_norm  = min((top_ic_val - 0.02) / 0.38 * 100, 100)
                    avg_norm  = min((avg_ic - 0.02) / 0.13 * 100, 100)
                    cnt_norm  = min(len(sig_scores) / 50 * 100, 100)
                    cons_norm = max(0, min((consistency - 0.5) / 0.5 * 100, 100))

                    composite = (
                        top_norm  * 0.40 +               # Top IC ağırlığı
                        avg_norm  * 0.25 +               # Avg IC ağırlığı
                        cnt_norm  * 0.15 +               # Anlamlı sayı ağırlığı
                        cons_norm * 0.20                  # Tutarlılık ağırlığı
                    )

                    # Rejim bazlı düzeltme
                    regime_mult = {'ranging': 0.85, 'volatile': 0.80, 'transitioning': 0.90}
                    composite *= regime_mult.get(regime, 1.0)

                    # ── 8. ATR Hesaplama (Risk Manager için) ──
                    atr_val = 0.0
                    atr_pct = 0.0
                    if 'ATRr_14' in df.columns and not df['ATRr_14'].dropna().empty:
                        atr_val = float(df['ATRr_14'].dropna().iloc[-1])
                    elif 'NATR_14' in df.columns and not df['NATR_14'].dropna().empty:
                        natr = float(df['NATR_14'].dropna().iloc[-1])
                        last_price = float(df['close'].iloc[-1])
                        atr_val = last_price * natr / 100
                    else:
                        # Manuel ATR hesabı (14 periyot)
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

                    # ── 9. Sonuç Ekle ──
                    if composite > 0:
                        tf_results.append({
                            'tf': tf,
                            'score': composite,
                            'direction': direction,
                            'regime': regime,
                            'significant': len(sig_scores),
                            'atr': atr_val,
                            'atr_pct': atr_pct,
                        })

                except Exception as e:
                    logger.debug(f"  {full_symbol} {tf}: Analiz hatası — {e}")
                    continue
'''


def find_and_replace(content: str) -> str:
    """main.py içeriğinde eski döngüyü bulup yenisiyle değiştirir."""
    
    # Eski for döngüsünün başlangıç ve bitiş konumlarını bul
    start_idx = content.find(OLD_BLOCK_MARKER_START)
    if start_idx == -1:
        print("❌ HATA: Eski döngü başlangıcı bulunamadı!")
        print(f"   Aranan: '{OLD_BLOCK_MARKER_START}'")
        sys.exit(1)
    
    # "if not tf_results:" satırını bul (döngüden sonraki ilk oluşum)
    end_marker_idx = content.find(OLD_BLOCK_MARKER_END, start_idx)
    if end_marker_idx == -1:
        print("❌ HATA: Eski döngü bitişi bulunamadı!")
        sys.exit(1)
    
    # Satır başına geri git (for satırının başlangıcı)
    line_start = content.rfind('\n', 0, start_idx) + 1
    
    # end_marker'ın satır başına geri git
    end_line_start = content.rfind('\n', 0, end_marker_idx) + 1
    
    # Eski bloğu kes
    old_block = content[line_start:end_line_start]
    
    # Yeni bloğu yerleştir
    new_content = content[:line_start] + NEW_LOOP_BLOCK + '\n' + content[end_line_start:]
    
    return new_content, old_block


def main():
    apply = '--apply' in sys.argv
    
    if not MAIN_PY.exists():
        print(f"❌ {MAIN_PY} bulunamadı!")
        print("   Bu scripti hybrid_crypto_bot/src/ dizininden çalıştırın.")
        sys.exit(1)
    
    content = MAIN_PY.read_text(encoding='utf-8')
    
    # Zaten düzeltilmiş mi kontrol et
    if 'calculator.calculate_all(df)' in content and 'evaluate_all_indicators' in content:
        # İkisi de varsa — for döngüsü içinde mi kontrol et
        for_block_start = content.find(OLD_BLOCK_MARKER_START)
        if for_block_start != -1:
            for_block_end = content.find(OLD_BLOCK_MARKER_END, for_block_start)
            block = content[for_block_start:for_block_end]
            if 'calculate_all' in block and 'evaluate_all_indicators' in block:
                print("✅ main.py zaten düzeltilmiş görünüyor!")
                sys.exit(0)
    
    # Eski hayalet method'lar var mı?
    has_prepare = 'preprocessor.prepare' in content
    has_add_all = 'add_all_indicators' in content
    has_analyze = 'selector.analyze' in content
    
    print("=" * 60)
    print("  🔧 main.py _analyze_coin() İç Döngü Düzeltmesi")
    print("=" * 60)
    print()
    print("  Tespit edilen hayalet method'lar:")
    print(f"    {'✗' if has_prepare else '✓'} preprocessor.prepare()       {'→ KALDIRILACAK' if has_prepare else '(yok)'}")
    print(f"    {'✗' if has_add_all else '✓'} add_all_indicators()        {'→ calculate_all()' if has_add_all else '(yok)'}")
    print(f"    {'✗' if has_analyze else '✓'} selector.analyze()          {'→ evaluate_all_indicators()' if has_analyze else '(yok)'}")
    print()
    
    if not (has_prepare or has_add_all or has_analyze):
        print("  ℹ️  Hayalet method bulunamadı — düzeltme gerekli olmayabilir.")
    
    # Değiştir
    new_content, old_block = find_and_replace(content)
    
    # Değişiklik istatistikleri
    old_lines = old_block.count('\n')
    new_lines = NEW_LOOP_BLOCK.count('\n')
    
    print(f"  Değişiklik:")
    print(f"    Eski:  {old_lines} satır")
    print(f"    Yeni:  {new_lines} satır")
    print(f"    Fark:  +{new_lines - old_lines} satır")
    print()
    
    if apply:
        # Yedek al
        backup_name = f"main.py.bak_analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(MAIN_PY, backup_name)
        print(f"  💾 Yedek: {backup_name}")
        
        # Yaz
        MAIN_PY.write_text(new_content, encoding='utf-8')
        print(f"  ✅ main.py güncellendi!")
        print()
        
        # np import kontrolü
        if 'import numpy as np' not in new_content and 'import numpy' not in new_content:
            print("  ⚠️  UYARI: numpy import'u eksik olabilir!")
            print("     main.py'nin başına şu satırı ekleyin:")
            print("     import numpy as np")
        
        # pd import kontrolü (ATR hesabı için)
        if 'import pandas as pd' not in new_content and 'import pandas' not in new_content:
            print("  ⚠️  UYARI: pandas import'u eksik olabilir!")
            print("     main.py'nin başına şu satırı ekleyin:")
            print("     import pandas as pd")
        
        print()
        print("  🧪 Doğrulama için:")
        print("     python integration_test_v1_2.py")
        
    else:
        print("  ⚡ DRY-RUN modu — değişiklik yapılmadı.")
        print()
        print("  Uygulamak için:")
        print("     python fix_main_analyze.py --apply")


if __name__ == "__main__":
    main()
