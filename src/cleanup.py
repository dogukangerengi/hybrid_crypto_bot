#!/usr/bin/env python3
"""
Sorun 8: lgbm_model.py ölü kod temizliği + tüm düzeltmelerin doğrulama testi.

Kullanım:
    cd hybrid_crypto_bot/src
    python cleanup.py

Bu script:
1. lgbm_model.py'yi .deprecated olarak arşivler (git'ten çıkarmaz, önce test et)
2. ml/__init__.py'daki eski referansları kontrol eder
3. Tüm düzeltmelerin doğru uygulandığını doğrular
"""

import subprocess
from pathlib import Path
import sys

SRC = Path(__file__).parent
ML_DIR = SRC / "ml"

print("=" * 60)
print("  SORUN 8: lgbm_model.py Temizliği")
print("=" * 60)

# ─── 1. lgbm_model.py arşivle ───
lgbm_file = ML_DIR / "lgbm_model.py"
lgbm_deprecated = ML_DIR / "lgbm_model.py.deprecated"

if lgbm_file.exists():
    lgbm_file.rename(lgbm_deprecated)
    print(f"✅ Arşivlendi: lgbm_model.py → lgbm_model.py.deprecated")
    print(f"   (Sorunla karşılaşırsan geri al: mv lgbm_model.py.deprecated lgbm_model.py)")
elif lgbm_deprecated.exists():
    print(f"⏭️  lgbm_model.py zaten arşivlenmiş (.deprecated uzantılı)")
else:
    print(f"⏭️  lgbm_model.py bulunamadı (zaten silinmiş olabilir)")

# ─── 2. ml/__init__.py kontrol ───
init_file = ML_DIR / "__init__.py"
if init_file.exists():
    init_content = init_file.read_text()
    if "lgbm_model" in init_content:
        print(f"\n⚠️  ml/__init__.py içinde 'lgbm_model' referansı var:")
        for i, line in enumerate(init_content.splitlines(), 1):
            if "lgbm_model" in line:
                print(f"   Satır {i}: {line}")
        print("   Bu satırları manuel olarak kaldır!")
    else:
        print(f"✅ ml/__init__.py temiz (lgbm_model referansı yok)")

# ─── 3. Tüm dosyalarda lgbm_model referansı tara ───
print(f"\n{'─'*40}")
print("  lgbm_model referanslarını tara:")
print(f"{'─'*40}")
for py_file in SRC.rglob("*.py"):
    if py_file.name.startswith("lgbm_model"):
        continue
    try:
        content = py_file.read_text(encoding="utf-8")
        if "lgbm_model" in content and "lgbm_model.py" not in content:
            lines = [(i+1, l.strip()) for i, l in enumerate(content.splitlines()) if "lgbm_model" in l]
            print(f"  {py_file.relative_to(SRC)}: {len(lines)} referans")
            for lineno, line in lines[:3]:
                print(f"    Satır {lineno}: {line[:80]}")
    except Exception:
        pass

# ─── 4. Düzeltme doğrulama testi ───
print(f"\n{'='*60}")
print("  DÜZELTME DOĞRULAMA TESTLERİ")
print(f"{'='*60}")

tests = []

# Test 1: ensemble_model.py — _purged_walk_forward dinamik mi?
try:
    em_content = (ML_DIR / "ensemble_model.py").read_text()
    if "effective_splits" in em_content and "max_splits_by_size" in em_content:
        tests.append(("SORUN 5", "✅", "_purged_walk_forward dinamik n_splits"))
    else:
        tests.append(("SORUN 5", "❌", "_purged_walk_forward hâlâ sabit n_splits"))
except Exception as e:
    tests.append(("SORUN 5", "⚠️", f"ensemble_model.py okunamadı: {e}"))

# Test 2: ensemble_model.py — binary detection log hata mı?
try:
    if "logger.error" in em_content and "BINARY TARGET DETECTED" in em_content:
        tests.append(("SORUN 1", "✅", "Binary detection WARNING→ERROR seviyesine yükseltildi"))
    else:
        tests.append(("SORUN 1", "❌", "Binary detection log seviyesi güncellenmemiş"))
except Exception as e:
    tests.append(("SORUN 1", "⚠️", str(e)))

# Test 3: ensemble_model.py — train() sonunda is_trained kontrolü var mı?
try:
    if "n_folds == 0" in em_content and "is_trained = False" in em_content:
        tests.append(("SORUN 5", "✅", "train() n_folds=0 durumunda is_trained=False set ediyor"))
    else:
        tests.append(("SORUN 5", "❌", "train() içinde n_folds=0 kontrolü eksik"))
except Exception as e:
    tests.append(("SORUN 5", "⚠️", str(e)))

# Test 4: risk_manager.py — hard-code 1.5 kaldırıldı mı?
try:
    rm_content = (SRC / "execution" / "risk_manager.py").read_text()
    if "getattr(self.risk_cfg, 'min_risk_reward_ratio', 1.5)" in rm_content:
        tests.append(("SORUN 7", "✅", "risk_manager.py config'den min_risk_reward_ratio okuyor"))
    elif "config_rr = 1.5" in rm_content:
        tests.append(("SORUN 7", "❌", "risk_manager.py hâlâ hard-code 1.5 kullanıyor"))
    else:
        tests.append(("SORUN 7", "⚠️", "risk_manager.py bulunamadı veya farklı format"))
except Exception as e:
    tests.append(("SORUN 7", "⚠️", f"risk_manager.py okunamadı: {e}"))

# Test 5: main.py — _compute_category_tops var mı?
try:
    main_content = (SRC / "main.py").read_text()
    if "_compute_category_tops" in main_content:
        tests.append(("SORUN 2", "✅", "main.py _compute_category_tops helper mevcut"))
    else:
        tests.append(("SORUN 2", "❌", "main.py _compute_category_tops eksik — main_patch.py çalıştır"))
except Exception as e:
    tests.append(("SORUN 2", "⚠️", f"main.py okunamadı: {e}"))

# Test 6: main.py — _now_utc helper var mı?
try:
    if "_now_utc" in main_content:
        tests.append(("SORUN 4", "✅", "main.py UTC helper fonksiyonları mevcut"))
    else:
        tests.append(("SORUN 4", "❌", "main.py UTC helper yok — main_patch.py çalıştır"))
except Exception as e:
    tests.append(("SORUN 4", "⚠️", str(e)))

# Test 7: main.py — R-multiple loop var mı (initial_train)?
try:
    if 'for fake_direction in ["LONG", "SHORT"]:' in main_content:
        tests.append(("SORUN 1+9", "✅", "initial_train bidirectional loop (LONG+SHORT) mevcut"))
    else:
        tests.append(("SORUN 1+9", "❌", "initial_train hâlâ binary — main_patch.py çalıştır"))
except Exception as e:
    tests.append(("SORUN 1+9", "⚠️", str(e)))

# Test 8: main.py — magic number 0.16 kaldırıldı mı?
try:
    if "pnl / 0.16" in main_content:
        tests.append(("SORUN 1", "❌", "retrain_from_experience magic number 0.16 hâlâ var"))
    else:
        tests.append(("SORUN 1", "✅", "retrain_from_experience magic number 0.16 kaldırıldı"))
except Exception as e:
    tests.append(("SORUN 1", "⚠️", str(e)))

# Sonuçları yazdır
passed = sum(1 for _, s, _ in tests if s == "✅")
failed = sum(1 for _, s, _ in tests if s == "❌")
warned = sum(1 for _, s, _ in tests if s == "⚠️")

for sorun, status, msg in tests:
    print(f"  {status} [{sorun}] {msg}")

print(f"\n  Toplam: {passed} geçti, {failed} başarısız, {warned} uyarı")

if failed > 0:
    print(f"\n⚠️  {failed} test başarısız — ilgili patch scriptlerini çalıştır:")
    print("    cd src && python main_patch.py")
else:
    print(f"\n✅ Tüm testler geçti!")
    print("   Sonraki adım: python main.py --train")
    print("   Kontrol: Log'da 'Binary target detected' ÇIKMAMALI")
