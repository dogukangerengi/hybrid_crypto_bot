#!/usr/bin/env python3
"""
third_patch.py — Son 2 sorunun manuel tamamlanması

Düzelttikleri:
    1. signal_validator.py  → lgbm_model importunu elle düzelt
    2. main.py initial_train → bidirectional loop tamamla + rows_y R-multiple yap

Kullanım:
    cd hybrid_crypto_bot/src
    python third_patch.py
    python cleanup.py   # doğrulama
"""

import shutil
import re
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).parent

# ─────────────────────────────────────────────
# 1. signal_validator.py — import düzelt
# ─────────────────────────────────────────────
sv_path = SRC / "ml" / "signal_validator.py"
sv = sv_path.read_text(encoding="utf-8")

# Dosyada tam olarak hangi import satırları var?
for i, line in enumerate(sv.splitlines(), 1):
    if "lgbm_model" in line:
        print(f"[signal_validator.py satır {i}]: {repr(line)}")

# Satır bazlı değiştir — string eşleştirme yerine
sv_lines = sv.splitlines()
new_sv_lines = []
skip_until_paren = False

for i, line in enumerate(sv_lines):
    if "from .lgbm_model import" in line:
        # Bu satırı ve kapanan paranteze kadar olanları yeni importla değiştir
        skip_until_paren = True
        new_sv_lines.append("# [SORUN 8 DÜZELTMESİ] — lgbm_model.py kaldırıldı, ensemble_model'dan al")
        new_sv_lines.append("from .ensemble_model import EnsemblePredictor as LGBMSignalModel  # noqa: F401")
        new_sv_lines.append("try:")
        new_sv_lines.append("    import lightgbm  # noqa")
        new_sv_lines.append("    HAS_LIGHTGBM = True")
        new_sv_lines.append("except ImportError:")
        new_sv_lines.append("    HAS_LIGHTGBM = False")
        continue
    if skip_until_paren:
        if ")" in line:
            skip_until_paren = False   # kapanan parantezi gördük, atlamayı durdur
        continue                        # bu satırı (eski import gövdesini) atla
    new_sv_lines.append(line)

sv_new = "\n".join(new_sv_lines) + "\n"
sv_path.write_text(sv_new, encoding="utf-8")
print("✅ signal_validator.py: lgbm_model → ensemble_model import düzeltildi")

# ─────────────────────────────────────────────
# 2. main.py initial_train — mevcut durumu bul, tamamla
# ─────────────────────────────────────────────
main_path = SRC / "main.py"
backup = SRC / f"main.py.bak_third_{datetime.now().strftime('%H%M%S')}"
shutil.copy2(main_path, backup)
print(f"📦 Yedek: {backup.name}")

main = main_path.read_text(encoding="utf-8")

# Mevcut durumu raporla
has_bidirectional = 'for fake_direction in ["LONG", "SHORT"]:' in main
has_r_multiple_append = 'rows_y.append(float(r_multiple))' in main
has_binary_append = 'rows_y.append(1 if target > 0 else 0)' in main
has_partial_append = "rows_y.append(1 if fwd_val > 0 else -1.0)" in main

print(f"\n[Mevcut durum]")
print(f"  Bidirectional loop var mı?  {has_bidirectional}")
print(f"  R-multiple append var mı?   {has_r_multiple_append}")
print(f"  Binary append var mı?       {has_binary_append}")
print(f"  Kısmi fix append var mı?    {has_partial_append}")

if has_bidirectional and has_r_multiple_append:
    print("✅ initial_train zaten tam düzelmiş görünüyor!")
    main_path.write_text(main, encoding="utf-8")

elif has_partial_append:
    # second_patch varyant B kısmen uygulandı: loop başı var ama
    # rows_y append kısmi fix yapıldı. Tamamlayalım.

    # Önce: kısmi rows_y satırını düzelt
    OLD_PARTIAL = (
        "                    rows_X.append(fv.to_dict())\n"
        "                    rows_y.append(1 if fwd_val > 0 else -1.0)  # kısmi fix: -1/+1\n"
        '                    rows_dir.append("LONG" if fwd_val > 0 else "SHORT")'
    )
    NEW_FULL = (
        "                    rows_X.append(fv.to_dict())\n"
        "                    rows_y.append(float(r_multiple))\n"
        "                    rows_dir.append(fake_direction)"
    )
    if OLD_PARTIAL in main:
        main = main.replace(OLD_PARTIAL, NEW_FULL, 1)
        print("✅ rows_y kısmi fix → float(r_multiple) tamamlandı")

    # Şimdi: bidirectional loop başlığı var mı kontrol et
    if 'for fake_direction in ["LONG", "SHORT"]:' not in main:
        # Compact varyant B'nin başını bul ve genişlet
        COMPACT_ANCHOR = (
            "                sl_distance    = atr_val * ATR_MULT\n"
            "                tp_distance    = sl_distance * RR_RATIO\n"
            "                price_move_usd = (np.exp(fwd_val) - 1) * entry_price"
        )
        COMPACT_EXPAND = (
            "                sl_distance    = atr_val * ATR_MULT\n"
            "                tp_distance    = sl_distance * RR_RATIO\n"
            "                price_move_usd = (np.exp(fwd_val) - 1) * entry_price\n"
            "\n"
            "                df_slice = df_ind.iloc[max(0, i - 50):i + 1].copy()\n"
            "                if len(df_slice) < 40:\n"
            "                    continue\n"
            "\n"
            '                for fake_direction in ["LONG", "SHORT"]:\n'
            "                    if fake_direction == \"LONG\":\n"
            "                        if price_move_usd >= tp_distance:\n"
            "                            r_multiple = RR_RATIO\n"
            "                        elif price_move_usd <= -sl_distance:\n"
            "                            r_multiple = -1.0\n"
            "                        else:\n"
            "                            r_multiple = price_move_usd / sl_distance\n"
            "                    else:\n"
            "                        if price_move_usd <= -tp_distance:\n"
            "                            r_multiple = RR_RATIO\n"
            "                        elif price_move_usd >= sl_distance:\n"
            "                            r_multiple = -1.0\n"
            "                        else:\n"
            "                            r_multiple = -price_move_usd / sl_distance\n"
            "\n"
            "                    if abs(r_multiple) < 0.25:\n"
            "                        continue\n"
            "                    r_multiple = max(-2.0, min(2.0, r_multiple))\n"
            "\n"
            "                    fv = self.feature_eng.build_features(\n"
            "                        analysis=analysis_stub,\n"
            "                        ohlcv_df=df_slice,\n"
            "                    )"
        )
        if COMPACT_ANCHOR in main:
            main = main.replace(COMPACT_ANCHOR, COMPACT_EXPAND, 1)
            print("✅ Bidirectional loop (LONG+SHORT) eklendi")

            # Varsa eski df_slice satırlarını temizle (duplicate olmasın)
            DUP_SLICE = (
                "\n"
                "                df_slice = df_ind.iloc[max(0, i - 50):i + 1].copy()\n"
                "                if len(df_slice) < 40:\n"
                "                    continue\n"
                "\n"
                "                fv = self.feature_eng.build_features(\n"
                "                    analysis=analysis_stub,\n"
                "                    ohlcv_df=df_slice,\n"
                "                )"
            )
            if DUP_SLICE in main:
                main = main.replace(DUP_SLICE, "", 1)
                print("  🧹 Duplicate df_slice bloğu temizlendi")
        else:
            print("⚠️  compact_anchor bulunamadı — aşağıdaki yaklaşımı dene")

    main_path.write_text(main, encoding="utf-8")

elif has_binary_append:
    # Hiçbir şey değişmemiş, sıfırdan uygula
    print("⚠️  initial_train hâlâ tamamen binary — satır numarasını raporluyorum:")
    for i, line in enumerate(main.splitlines(), 1):
        if "target = 1 if fwd_val > 0 else 0" in line:
            print(f"  Satır {i}: {line.rstrip()}")
        if "rows_y.append(1 if target > 0 else 0)" in line:
            print(f"  Satır {i}: {line.rstrip()}")
    print()
    print("  Manuel düzeltme için: UYGULAMA_REHBERI.md → SORUN 1+9 MANUAL bölümüne bak")
    main_path.write_text(main, encoding="utf-8")
else:
    print("? Durum belirsiz — cleanup.py'yi çalıştır, sonucu paylaş")
    main_path.write_text(main, encoding="utf-8")

# ─────────────────────────────────────────────
# ml/__init__.py — çift yorum temizliği
# ─────────────────────────────────────────────
init_path = SRC / "ml" / "__init__.py"
if init_path.exists():
    init = init_path.read_text(encoding="utf-8")
    # "# #" → "#" (çift yorum düzelt)
    init_new = init.replace(
        "# # from ml.lgbm_model import LGBMModel (DEPRECATED — lgbm_model.py kaldırıldı)",
        "# from ml.lgbm_model import LGBMModel  # DEPRECATED: lgbm_model.py kaldırıldı"
    )
    if init_new != init:
        init_path.write_text(init_new, encoding="utf-8")
        print("✅ ml/__init__.py çift yorum (# #) düzeltildi")

print()
print("=" * 50)
print("Sonraki adım:")
print("  python cleanup.py")
print("  python main.py --train")
