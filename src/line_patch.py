#!/usr/bin/env python3
"""
line_patch.py — initial_train binary loop → R-multiple bidirectional

Satır 2127'de `rows_y.append(1 if target > 0 else 0)` ifadesini bulup
oradan geriye `for i in range(start_idx, end_idx):` satırına kadar
tüm bloğu R-multiple bidirectional yapısıyla değiştiriyor.

Kullanım:
    cd hybrid_crypto_bot/src
    python line_patch.py
"""

import shutil
from datetime import datetime
from pathlib import Path

main_path = Path(__file__).parent / "main.py"
backup    = main_path.with_suffix(f".bak_line_{datetime.now().strftime('%H%M%S')}")
shutil.copy2(main_path, backup)
print(f"📦 Yedek: {backup.name}")

lines = main_path.read_text(encoding="utf-8").splitlines(keepends=True)
total = len(lines)

# ─── 1. Hedef satırı bul (rows_y.append binary) ───────────────────────────
target_line_idx = None
for i, line in enumerate(lines):
    if "rows_y.append(1 if target > 0 else 0)" in line:
        target_line_idx = i
        print(f"✓ Binary append satırı: {i+1} → {line.rstrip()}")

if target_line_idx is None:
    print("❌ 'rows_y.append(1 if target > 0 else 0)' satırı bulunamadı.")
    print("   Belki zaten düzeltilmiş? cleanup.py çalıştır.")
    exit(0)

# ─── 2. rows_dir.append satırını bul (bir sonraki satır) ──────────────────
rows_dir_idx = target_line_idx + 1
if "rows_dir.append" not in lines[rows_dir_idx]:
    # Birkaç satır ileriye bak
    for delta in range(1, 5):
        if "rows_dir.append" in lines[target_line_idx + delta]:
            rows_dir_idx = target_line_idx + delta
            break

loop_end_idx = rows_dir_idx  # Bu satırı da değiştireceğiz

print(f"✓ rows_dir satırı: {rows_dir_idx+1} → {lines[rows_dir_idx].rstrip()}")

# ─── 3. Geriye giderek `for i in range(start_idx, end_idx):` bul ──────────
loop_start_idx = None
for i in range(target_line_idx, max(0, target_line_idx - 200), -1):
    if "for i in range(start_idx, end_idx):" in lines[i]:
        loop_start_idx = i
        print(f"✓ Loop başlangıcı: {i+1} → {lines[i].rstrip()}")
        break

if loop_start_idx is None:
    print("❌ 'for i in range(start_idx, end_idx):' satırı bulunamadı (200 satır geriye bakıldı).")
    exit(1)

# ─── 4. Loop'tan önce ATR hazırlık bloğu var mı kontrol et ────────────────
# "ATR_MULT" veya "atr_col_train" zaten eklenmişse tekrar ekleme
pre_block_start = loop_start_idx
has_atr_prep = any(
    "atr_col_train" in lines[j]
    for j in range(max(0, loop_start_idx - 30), loop_start_idx)
)

# ─── 5. Girintisini al ────────────────────────────────────────────────────
loop_indent = len(lines[loop_start_idx]) - len(lines[loop_start_idx].lstrip())
ind  = " " * loop_indent          # for döngüsü girintisi
ind2 = ind + "    "               # loop gövdesi
ind3 = ind2 + "    "              # for fake_direction gövdesi
ind4 = ind3 + "    "              # if/elif/else gövdesi

print(f"✓ Girinti: {loop_indent} boşluk")
print(f"✓ Değiştirilecek satır aralığı: {loop_start_idx+1} – {loop_end_idx+1}")

# ─── 6. Yeni blok ─────────────────────────────────────────────────────────
ATR_PREP = ""
if not has_atr_prep:
    ATR_PREP = (
        f"{ind}# [SORUN 2] category_tops loop dışında bir kez hesapla\n"
        f"{ind}analysis_stub.category_tops = self._compute_category_tops(all_scores, CATEGORIES)\n"
        f"\n"
        f"{ind}# [SORUN 1+9] ATR kolonu bul — R-multiple hesabı için zorunlu\n"
        f"{ind}atr_col_train = next(\n"
        f"{ind}    (c for c in ['ATRr_14', 'ATR_14', 'ATRr_7', 'NATR_14'] if c in df_ind.columns),\n"
        f"{ind}    None\n"
        f"{ind})\n"
        f"{ind}if atr_col_train is None:\n"
        f"{ind}    logger.error('❌ ATR kolonu bulunamadı — initial_train başarısız')\n"
        f"{ind}    return False\n"
        f"\n"
        f"{ind}ATR_MULT, RR_RATIO = 3.0, 1.5  # risk_manager ile tutarlı\n"
        f"{ind}logger.info(\n"
        f"{ind}    f'  [initial_train] R-multiple: ATR_col={{atr_col_train}}, '\n"
        f"{ind}    f'ATR_mult={{ATR_MULT}}, RR={{RR_RATIO}}'\n"
        f"{ind})\n"
        f"\n"
    )

NEW_LOOP = (
    ATR_PREP
    + f"{ind}for i in range(start_idx, end_idx):\n"
    + f"{ind2}fwd_val = df_ind[target_col].iloc[i]\n"
    + f"{ind2}if pd.isna(fwd_val):\n"
    + f"{ind3}continue\n"
    + f"\n"
    + f"{ind2}entry_price = df_ind['close'].iloc[i]\n"
    + f"{ind2}atr_val     = df_ind[atr_col_train].iloc[i]\n"
    + f"{ind2}if atr_col_train == 'NATR_14':\n"
    + f"{ind3}atr_val = atr_val * entry_price / 100  # % → $\n"
    + f"{ind2}if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0 or entry_price <= 0:\n"
    + f"{ind3}continue\n"
    + f"\n"
    + f"{ind2}sl_distance    = atr_val * ATR_MULT\n"
    + f"{ind2}tp_distance    = sl_distance * RR_RATIO\n"
    + f"{ind2}price_move_usd = (np.exp(fwd_val) - 1) * entry_price\n"
    + f"\n"
    + f"{ind2}df_slice = df_ind.iloc[max(0, i - 50):i + 1].copy()\n"
    + f"{ind2}if len(df_slice) < 40:\n"
    + f"{ind3}continue\n"
    + f"\n"
    + f"{ind2}# [SORUN 9] Her bar için LONG ve SHORT → tautoloji yok\n"
    + f'{ind2}for fake_direction in ["LONG", "SHORT"]:\n'
    + f"{ind3}if fake_direction == 'LONG':\n"
    + f"{ind4}if price_move_usd >= tp_distance:\n"
    + f"{ind4}    r_multiple = RR_RATIO\n"
    + f"{ind4}elif price_move_usd <= -sl_distance:\n"
    + f"{ind4}    r_multiple = -1.0\n"
    + f"{ind4}else:\n"
    + f"{ind4}    r_multiple = price_move_usd / sl_distance\n"
    + f"{ind3}else:  # SHORT\n"
    + f"{ind4}if price_move_usd <= -tp_distance:\n"
    + f"{ind4}    r_multiple = RR_RATIO\n"
    + f"{ind4}elif price_move_usd >= sl_distance:\n"
    + f"{ind4}    r_multiple = -1.0\n"
    + f"{ind4}else:\n"
    + f"{ind4}    r_multiple = -price_move_usd / sl_distance\n"
    + f"\n"
    + f"{ind3}if abs(r_multiple) < 0.25:  # dead-band: noise at\n"
    + f"{ind4}continue\n"
    + f"{ind3}r_multiple = max(-2.0, min(2.0, r_multiple))\n"
    + f"\n"
    + f"{ind3}fv = self.feature_eng.build_features(\n"
    + f"{ind4}analysis=analysis_stub,\n"
    + f"{ind4}ohlcv_df=df_slice,\n"
    + f"{ind3})\n"
    + f"{ind3}rows_X.append(fv.to_dict())\n"
    + f"{ind3}rows_y.append(float(r_multiple))\n"
    + f"{ind3}rows_dir.append(fake_direction)\n"
)

# ─── 7. Satırları değiştir ────────────────────────────────────────────────
new_lines = (
    lines[:loop_start_idx]
    + [NEW_LOOP]
    + lines[loop_end_idx + 1:]
)

main_path.write_text("".join(new_lines), encoding="utf-8")

print(f"\n✅ Değiştirildi: satır {loop_start_idx+1}–{loop_end_idx+1} → R-multiple bidirectional")

# ─── 8. Hızlı doğrulama ──────────────────────────────────────────────────
result = main_path.read_text(encoding="utf-8")
checks = {
    'for fake_direction in ["LONG", "SHORT"]:': "Bidirectional loop",
    "rows_y.append(float(r_multiple))":         "R-multiple append",
    "rows_y.append(1 if target > 0 else 0)":    "Binary append (olmamalı)",
}
print("\n[Doğrulama]")
for needle, label in checks.items():
    found = needle in result
    if label == "Binary append (olmamalı)":
        icon = "❌" if found else "✅"
    else:
        icon = "✅" if found else "❌"
    print(f"  {icon} {label}: {'MEVCUT' if found else 'YOK'}")

print("\nSonraki adım:")
print("  python cleanup.py")
print("  python main.py --train")
