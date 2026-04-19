#!/usr/bin/env python3
"""
second_patch.py — Kalan düzeltmeler

Düzelttikleri:
    1. main.py → initial_train: binary {0,1} target → R-multiple bidirectional
    2. main.py → retrain_from_experience: TIMEOUT magic number 0.16 → dinamik
    3. signal_validator.py → lgbm_model import'u → ensemble_model'den al
    4. ml/__init__.py → yorum satırı temizliği

Kullanım:
    cd hybrid_crypto_bot/src
    python second_patch.py
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).parent

def patch_file(path: Path, changes: list, label: str) -> list:
    """
    Bir dosyaya birden fazla string replacement uygula.
    changes: [(old, new, description), ...]
    Başarılı/başarısız mesajları döndürür.
    """
    content = path.read_text(encoding="utf-8")
    results = []
    for old, new, desc in changes:
        if old in content:
            content = content.replace(old, new, 1)
            results.append(f"  ✅ {label}: {desc}")
        else:
            results.append(f"  ⚠️  {label}: Bulunamadı — {desc}")
    path.write_text(content, encoding="utf-8")
    return results

all_results = []

# =============================================================================
# 1. signal_validator.py — lgbm_model import düzelt
# =============================================================================
sv_path = SRC / "ml" / "signal_validator.py"

old_sv_import = """from .lgbm_model import (
    LGBMSignalModel,
    HAS_LIGHTGBM,
)"""

new_sv_import = """# [SORUN 8 DÜZELTMESİ] — lgbm_model.py artık kullanılmıyor, ensemble_model'dan al
from .ensemble_model import (
    EnsemblePredictor as LGBMSignalModel,
    EnsemblePredictor,
)
try:
    import lightgbm  # noqa
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False"""

if sv_path.exists():
    r = patch_file(sv_path, [(old_sv_import, new_sv_import, "lgbm_model → ensemble_model import")], "signal_validator.py")
    all_results.extend(r)
else:
    all_results.append("  ⚠️  signal_validator.py bulunamadı")

# =============================================================================
# 2. ml/__init__.py — yorum temizliği
# =============================================================================
init_path = SRC / "ml" / "__init__.py"

if init_path.exists():
    init_content = init_path.read_text(encoding="utf-8")
    if "lgbm_model" in init_content:
        # Sadece yorum satırlarını temizle
        lines = init_content.splitlines()
        cleaned = []
        for line in lines:
            if "lgbm_model" in line and line.strip().startswith("#"):
                cleaned.append(f"# {line.strip()} (DEPRECATED — lgbm_model.py kaldırıldı)")
            else:
                cleaned.append(line)
        init_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        all_results.append("  ✅ ml/__init__.py: lgbm_model yorum satırı güncellendi")
    else:
        all_results.append("  ⏭️  ml/__init__.py: lgbm_model referansı yok, atlandı")
else:
    all_results.append("  ⚠️  ml/__init__.py bulunamadı")

# =============================================================================
# 3. main.py — retrain_from_experience TIMEOUT magic number 0.16
# =============================================================================
main_path = SRC / "main.py"
backup_path = SRC / f"main.py.bak_second_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy2(main_path, backup_path)
print(f"📦 Yedek alındı: {backup_path.name}")

main_content = main_path.read_text(encoding="utf-8")

# retrain_from_experience TIMEOUT bloğu — birkaç format varyantı dene
TIMEOUT_FIXES = [
    # Varyant 1: orijinal tam format
    (
        '                                else:  # TIMEOUT\n'
        '                                    if pnl != 0:\n'
        '                                        r_multiple = pnl / 0.16 \n'
        '                                        r_multiple = max(-2.0, min(2.0, r_multiple))\n'
        '                                    else:\n'
        '                                        r_multiple = 0.0\n'
        '                                        \n'
        '                # --- YENİ EKLENEN DEAD-BAND (ÖLÜ BÖLGE) FİLTRESİ ---',
        '                                else:  # TIMEOUT\n'
        '                                    if pnl != 0:\n'
        '                                        # [SORUN 1 DÜZELTMESİ] — Magic number 0.16 kaldırıldı\n'
        '                                        # Gerçek risk = balance × %2 varsayımı\n'
        '                                        risk_est = self._balance * 0.02 if self._balance > 0 else 1.0\n'
        '                                        r_multiple = pnl / risk_est if risk_est > 0 else 0.0\n'
        '                                        r_multiple = max(-2.0, min(2.0, r_multiple))\n'
        '                                    else:\n'
        '                                        r_multiple = 0.0\n'
        '                                        \n'
        '                # --- YENİ EKLENEN DEAD-BAND (ÖLÜ BÖLGE) FİLTRESİ ---',
        "TIMEOUT magic number 0.16 → risk_est = balance×0.02 (varyant 1)"
    ),
    # Varyant 2: farklı girinti
    (
        '                    else:  # TIMEOUT\n'
        '                        if pnl != 0:\n'
        '                            r_multiple = pnl / 0.16 \n',
        '                    else:  # TIMEOUT\n'
        '                        if pnl != 0:\n'
        '                            # [SORUN 1 DÜZELTMESİ] — Magic number 0.16 → balance×0.02\n'
        '                            risk_est = self._balance * 0.02 if self._balance > 0 else 1.0\n'
        '                            r_multiple = pnl / risk_est if risk_est > 0 else 0.0\n',
        "TIMEOUT magic number 0.16 → risk_est (varyant 2)"
    ),
    # Varyant 3: sadece o satırı değiştir
    (
        '                                        r_multiple = pnl / 0.16 \n',
        '                                        risk_est = self._balance * 0.02 if self._balance > 0 else 1.0\n'
        '                                        r_multiple = pnl / risk_est if risk_est > 0 else 0.0  # [SORUN 1: magic 0.16 → balance×0.02]\n',
        "TIMEOUT tek satır fix (varyant 3)"
    ),
]

timeout_fixed = False
for old, new, desc in TIMEOUT_FIXES:
    if old in main_content:
        main_content = main_content.replace(old, new, 1)
        all_results.append(f"  ✅ main.py retrain: {desc}")
        timeout_fixed = True
        break

if not timeout_fixed:
    # Son çare: regex ile bul
    pattern = r'r_multiple = pnl / 0\.16\s'
    match = re.search(pattern, main_content)
    if match:
        old_line = match.group(0)
        new_line = ('risk_est = self._balance * 0.02 if self._balance > 0 else 1.0\n'
                    '                                        r_multiple = pnl / risk_est if risk_est > 0 else 0.0  # [SORUN 1 FIX]\n')
        main_content = main_content[:match.start()] + new_line + main_content[match.end():]
        all_results.append("  ✅ main.py retrain: TIMEOUT magic 0.16 → risk_est (regex)")
    else:
        all_results.append("  ⚠️  main.py retrain: r_multiple = pnl / 0.16 satırı bulunamadı — zaten düzeltilmiş mi kontrol et")

# =============================================================================
# 4. main.py — initial_train binary loop → R-multiple bidirectional
#
# Strateji: İlk patch zaten CATEGORIES bloğunu değiştirdi.
# Şimdi mevcut durumda loop şöyle:
#
#   for i in range(start_idx, end_idx):
#       fwd_val = df_ind[target_col].iloc[i]
#       if pd.isna(fwd_val):
#           continue
#       if abs(fwd_val) < MIN_MOVE:
#           continue
#
#       target = 1 if fwd_val > 0 else 0
#       fake_direction = "LONG" if fwd_val > 0 else "SHORT" # YENİ EKLENDİ
#
# Bu bloku R-multiple yapısıyla değiştiriyoruz.
# =============================================================================

INIT_LOOP_VARIANTS = [
    # Varyant A: İlk patch CATEGORIES değiştirdi, loop hâlâ binary
    (
        '            for i in range(start_idx, end_idx):\n'
        '                fwd_val = df_ind[target_col].iloc[i]\n'
        '                if pd.isna(fwd_val):\n'
        '                    continue\n'
        '                if abs(fwd_val) < MIN_MOVE:\n'
        '                    continue\n'
        '\n'
        '                target = 1 if fwd_val > 0 else 0\n'
        '                fake_direction = "LONG" if fwd_val > 0 else "SHORT" # YENİ EKLENDİ\n'
        '                \n'
        '                df_slice = df_ind.iloc[max(0, i-50):i+1].copy()\n'
        '                if len(df_slice) < 40:\n'
        '                    continue\n'
        '\n'
        '                analysis_stub.category_tops = self._compute_category_tops(all_scores, CATEGORIES)\n'
        '\n'
        '                fv = self.feature_eng.build_features(\n'
        '                    analysis=analysis_stub,\n'
        '                    ohlcv_df=df_slice\n'
        '                )\n'
        '\n'
        '                rows_X.append(fv.to_dict())\n'
        '                rows_y.append(1 if target > 0 else 0)\n'
        '                rows_dir.append(fake_direction) # YENİ EKLENDİ',
        '''            # [SORUN 2 DÜZELTMESİ] — category_tops loop dışında bir kez hesaplanır
            analysis_stub.category_tops = self._compute_category_tops(all_scores, CATEGORIES)

            # [SORUN 1+9 DÜZELTMESİ] — ATR kolonu: R-multiple hesabı için zorunlu
            atr_col_train = None
            for _cand in ['ATRr_14', 'ATR_14', 'ATRr_7', 'NATR_14']:
                if _cand in df_ind.columns:
                    atr_col_train = _cand
                    break

            if atr_col_train is None:
                logger.error("❌ ATR kolonu bulunamadı — initial_train R-multiple hesaplayamıyor.")
                return False

            ATR_MULT = 3.0   # SL mesafesi (risk_manager ile tutarlı)
            RR_RATIO = 1.5   # TP/SL oranı

            logger.info(
                f"  [initial_train] R-multiple mode: ATR_col={atr_col_train}, "
                f"ATR_mult={ATR_MULT}, RR={RR_RATIO}"
            )

            for i in range(start_idx, end_idx):
                fwd_val = df_ind[target_col].iloc[i]
                if pd.isna(fwd_val):
                    continue

                entry_price = df_ind['close'].iloc[i]
                atr_val     = df_ind[atr_col_train].iloc[i]

                if atr_col_train == 'NATR_14':
                    atr_val = atr_val * entry_price / 100  # % → $

                if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0 or entry_price <= 0:
                    continue

                sl_distance   = atr_val * ATR_MULT
                tp_distance   = sl_distance * RR_RATIO
                price_move_pct = np.exp(fwd_val) - 1
                price_move_usd = price_move_pct * entry_price

                df_slice = df_ind.iloc[max(0, i - 50):i + 1].copy()
                if len(df_slice) < 40:
                    continue

                # [SORUN 9] Her bar için HER İKİ YÖN → tautoloji yok, is_long bilgilendirici
                for fake_direction in ["LONG", "SHORT"]:
                    if fake_direction == "LONG":
                        if price_move_usd >= tp_distance:
                            r_multiple = RR_RATIO        # +1.5R  (TP hit)
                        elif price_move_usd <= -sl_distance:
                            r_multiple = -1.0             # -1.0R  (SL hit)
                        else:
                            r_multiple = price_move_usd / sl_distance
                    else:  # SHORT — perspektif ters
                        if price_move_usd <= -tp_distance:
                            r_multiple = RR_RATIO        # +1.5R  (TP hit)
                        elif price_move_usd >= sl_distance:
                            r_multiple = -1.0             # -1.0R  (SL hit)
                        else:
                            r_multiple = -price_move_usd / sl_distance

                    if abs(r_multiple) < 0.25:  # Dead-band: noise at
                        continue

                    r_multiple = max(-2.0, min(2.0, r_multiple))

                    fv = self.feature_eng.build_features(
                        analysis=analysis_stub,
                        ohlcv_df=df_slice,
                    )

                    rows_X.append(fv.to_dict())
                    rows_y.append(float(r_multiple))
                    rows_dir.append(fake_direction)''',
        "initial_train binary→R-multiple (varyant A: category_tops ayrı)"
    ),
    # Varyant B: category_tops çağrısı loop içinde
    (
        '            for i in range(start_idx, end_idx):\n'
        '                fwd_val = df_ind[target_col].iloc[i]\n'
        '                if pd.isna(fwd_val):\n'
        '                    continue\n'
        '                if abs(fwd_val) < MIN_MOVE:\n'
        '                    continue\n'
        '\n'
        '                target = 1 if fwd_val > 0 else 0\n'
        '                fake_direction = "LONG" if fwd_val > 0 else "SHORT" # YENİ EKLENDİ',
        '''            # [SORUN 2] category_tops loop dışında bir kez hesapla
            analysis_stub.category_tops = self._compute_category_tops(all_scores, CATEGORIES)

            # [SORUN 1+9] ATR kolonu bul
            atr_col_train = next(
                (c for c in ['ATRr_14', 'ATR_14', 'ATRr_7', 'NATR_14'] if c in df_ind.columns),
                None
            )
            if atr_col_train is None:
                logger.error("❌ ATR kolonu bulunamadı — initial_train başarısız")
                return False

            ATR_MULT, RR_RATIO = 3.0, 1.5

            for i in range(start_idx, end_idx):
                fwd_val = df_ind[target_col].iloc[i]
                if pd.isna(fwd_val):
                    continue

                entry_price = df_ind['close'].iloc[i]
                atr_val     = df_ind[atr_col_train].iloc[i]
                if atr_col_train == 'NATR_14':
                    atr_val = atr_val * entry_price / 100
                if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0 or entry_price <= 0:
                    continue

                sl_distance    = atr_val * ATR_MULT
                tp_distance    = sl_distance * RR_RATIO
                price_move_usd = (np.exp(fwd_val) - 1) * entry_price''',
        "initial_train binary→R-multiple (varyant B: compact)"
    ),
]

init_fixed = False
for old, new, desc in INIT_LOOP_VARIANTS:
    if old in main_content:
        main_content = main_content.replace(old, new, 1)
        all_results.append(f"  ✅ main.py initial_train: {desc}")
        init_fixed = True
        break

if not init_fixed:
    # Son çare: "target = 1 if fwd_val > 0 else 0" satırını bul
    SIMPLE_TARGET = '                target = 1 if fwd_val > 0 else 0\n'
    SIMPLE_DIR    = '                fake_direction = "LONG" if fwd_val > 0 else "SHORT" # YENİ EKLENDİ\n'
    
    if SIMPLE_TARGET in main_content and SIMPLE_DIR in main_content:
        # Sadece bu iki satırı yorum haline getir ve R-multiple hazırlığı ekle
        COMBINED = SIMPLE_TARGET + SIMPLE_DIR
        COMBINED_NEW = (
            '                # [SORUN 1+9 DÜZELTMESİ] — binary target kaldırıldı, R-multiple hesabı aşağıda\n'
        )
        main_content = main_content.replace(COMBINED, COMBINED_NEW, 1)

        # rows_y.append(1 if target > 0 else 0) satırını değiştir
        OLD_ROWS_Y = (
            '                rows_X.append(fv.to_dict())\n'
            '                rows_y.append(1 if target > 0 else 0)\n'
            '                rows_dir.append(fake_direction) # YENİ EKLENDİ'
        )
        NEW_ROWS_Y = (
            '                # NOT: R-multiple hesabı için initial_train\'i tamamen\n'
            '                # yeniden yapılandırman gerekiyor. Şimdilik binary korunsun.\n'
            '                # Tam düzeltme için: UYGULAMA_REHBERI.md → SORUN 1+9 MANUAL bölümü\n'
            '                rows_X.append(fv.to_dict())\n'
            '                rows_y.append(1 if fwd_val > 0 else -1.0)  # kısmi fix: -1/+1\n'
            '                rows_dir.append("LONG" if fwd_val > 0 else "SHORT")'
        )
        if OLD_ROWS_Y in main_content:
            main_content = main_content.replace(OLD_ROWS_Y, NEW_ROWS_Y, 1)

        all_results.append("  ⚠️  main.py initial_train: Kısmi fix uygulandı — tam R-multiple için manual gerekli (bak UYGULAMA_REHBERI.md)")
    else:
        all_results.append("  ❌ main.py initial_train: Otomatik patch başarısız — manuel uygulama gerekli")
        all_results.append("       Aşağıdaki komutu çalıştır:")
        all_results.append("       grep -n 'target = 1 if fwd_val' main.py  # satır numarasını bul")
        all_results.append("       Sonra UYGULAMA_REHBERI.md'deki SORUN 1+9 bloğunu o satıra uygula")

# main.py'yi kaydet
main_path.write_text(main_content, encoding="utf-8")

# =============================================================================
# Özet
# =============================================================================
print("\n" + "=" * 60)
print("  SECOND PATCH SONUCU")
print("=" * 60)
for r in all_results:
    print(r)

warnings = [r for r in all_results if r.strip().startswith("⚠️")]
errors = [r for r in all_results if r.strip().startswith("❌")]

print()
print(f"  ✅ Başarılı: {len([r for r in all_results if r.strip().startswith('✅')])}")
print(f"  ⚠️  Uyarı:   {len(warnings)}")
print(f"  ❌ Hata:     {len(errors)}")

print()
print("Sonraki adım:")
print("  python cleanup.py   → Tüm testleri tekrar çalıştır")
print("  python main.py --train  → İlk eğitimi dene")
