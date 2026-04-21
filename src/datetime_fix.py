#!/usr/bin/env python3
"""
datetime_fix.py — Kalan naive datetime sorunları (Sorun 4 tamamlama)

Sorunlar:
  1469/1622: kapanis_zamani (UTC-aware) - acilis_zamani (naive) → TypeError
  2595:      datetime.now() naive → Süre (dk) yanlış
  2583:      datetime.now() naive → Excel'e yanlış zaman
  1913/2140: datetime.now() → Excel rapor tarihleri naive

Kullanım:
    cd hybrid_crypto_bot/src
    python datetime_fix.py
"""

import shutil
from datetime import datetime
from pathlib import Path

main_path = Path(__file__).parent / "main.py"
backup    = main_path.with_suffix(f".bak_dt_{datetime.now().strftime('%H%M%S')}")
shutil.copy2(main_path, backup)
print(f"📦 Yedek: {backup.name}")

content = main_path.read_text(encoding="utf-8")
results = []

# ─── Yardımcı: bir satırı değiştir, sonucu raporla ───────────────────────
def fix(old, new, desc):
    global content
    if old in content:
        count = content.count(old)
        content = content.replace(old, new)
        results.append(f"✅ {desc} ({count}x)")
    else:
        results.append(f"⚠️  Bulunamadı: {desc}")

# ─────────────────────────────────────────────────────────────────────────
# 1. Satır 1468-1469 ve 1621-1622: acilis_zamani naive → UTC-aware yap
#    kapanis_zamani = _now_utc() (aware)
#    acilis_zamani  = pd.to_datetime(acilis_str)  → naive
#    Çözüm: acilis_zamani'ye UTC ekle, sonra TR'e çevir
# ─────────────────────────────────────────────────────────────────────────
OLD_SURE_1 = (
    "                                        try:\n"
    "                                            acilis_str = str(df.at[idx, 'Tarih (Açılış)'])\n"
    "                                            acilis_zamani = pd.to_datetime(acilis_str)\n"
    "                                            df.at[idx, 'Süre (dk)'] = int((kapanis_zamani - acilis_zamani).total_seconds() / 60)\n"
    "                                        except Exception:\n"
    "                                            df.at[idx, 'Süre (dk)'] = 0"
)
NEW_SURE_1 = (
    "                                        try:\n"
    "                                            acilis_str = str(df.at[idx, 'Tarih (Açılış)'])\n"
    "                                            acilis_zamani = pd.to_datetime(acilis_str)\n"
    "                                            # UTC-aware yap (aware - naive → TypeError önle)\n"
    "                                            if acilis_zamani.tzinfo is None:\n"
    "                                                acilis_zamani = acilis_zamani.tz_localize('UTC')\n"
    "                                            df.at[idx, 'Süre (dk)'] = int((kapanis_zamani - acilis_zamani).total_seconds() / 60)\n"
    "                                        except Exception:\n"
    "                                            df.at[idx, 'Süre (dk)'] = 0"
)
fix(OLD_SURE_1, NEW_SURE_1, "Satır ~1469: acilis_zamani.tz_localize('UTC')")

# ─── Aynı pattern ikinci kez (satır ~1622) ────────────────────────────────
OLD_SURE_2 = (
    "                                    try:\n"
    "                                        acilis_str = str(df.at[idx, 'Tarih (Açılış)'])\n"
    "                                        acilis_zamani = pd.to_datetime(acilis_str)\n"
    "                                        df.at[idx, 'Süre (dk)'] = int((kapanis_zamani - acilis_zamani).total_seconds() / 60)\n"
    "                                    except Exception:\n"
    "                                        df.at[idx, 'Süre (dk)'] = 0"
)
NEW_SURE_2 = (
    "                                    try:\n"
    "                                        acilis_str = str(df.at[idx, 'Tarih (Açılış)'])\n"
    "                                        acilis_zamani = pd.to_datetime(acilis_str)\n"
    "                                        if acilis_zamani.tzinfo is None:\n"
    "                                            acilis_zamani = acilis_zamani.tz_localize('UTC')\n"
    "                                        df.at[idx, 'Süre (dk)'] = int((kapanis_zamani - acilis_zamani).total_seconds() / 60)\n"
    "                                    except Exception:\n"
    "                                        df.at[idx, 'Süre (dk)'] = 0"
)
fix(OLD_SURE_2, NEW_SURE_2, "Satır ~1622: acilis_zamani.tz_localize('UTC') (ikinci)")

# ─────────────────────────────────────────────────────────────────────────
# 2. Satır ~2595: orphan reconcile'daki Süre (dk) hesabı
#    datetime.now() - acilis.to_pydatetime() → naive - naive (şimdilik OK)
#    ama tutarlılık için _now_utc() yapalım
# ─────────────────────────────────────────────────────────────────────────
fix(
    "                    df.at[idx, 'Süre (dk)'] = int((datetime.now() - acilis.to_pydatetime()).total_seconds() / 60)",
    "                    _acilis_aware = acilis.to_pydatetime()\n"
    "                    if _acilis_aware.tzinfo is None:\n"
    "                        _acilis_aware = _acilis_aware.replace(tzinfo=timezone.utc)\n"
    "                    df.at[idx, 'Süre (dk)'] = int((_now_utc() - _acilis_aware).total_seconds() / 60)",
    "Satır ~2595: orphan Süre (dk) → _now_utc()"
)

# ─────────────────────────────────────────────────────────────────────────
# 3. Satır ~2583: orphan now_str naive → UTC-aware
# ─────────────────────────────────────────────────────────────────────────
fix(
    '                now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")',
    '                now_str = _iso_tr(_now_utc())  # UTC üret, TR göster',
    "Satır ~2583: now_str naive → _iso_tr(_now_utc())"
)

# ─────────────────────────────────────────────────────────────────────────
# 4. Satır ~1913 ve ~2140: Excel rapor tarihleri ("Tarih" kolonu)
# ─────────────────────────────────────────────────────────────────────────
fix(
    '                        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),',
    '                        "Tarih": _now_local().strftime("%Y-%m-%d %H:%M:%S"),  # TR saati',
    "Satır ~1913/2140: Excel rapor tarihi → _now_local()"
)

# ─────────────────────────────────────────────────────────────────────────
# 5. Satır ~2388: Summary sheet "Oluşturulma" zamanı
# ─────────────────────────────────────────────────────────────────────────
fix(
    '            ws_summary[\'A2\'] = f\'Oluşturulma: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\'',
    "            ws_summary['A2'] = f'Oluşturulma: {_now_local().strftime(\"%Y-%m-%d %H:%M:%S\")} (TR)'",
    "Satır ~2388: Summary sheet Oluşturulma → _now_local()"
)

# ─────────────────────────────────────────────────────────────────────────
# 6. Satır ~2472: Alt dosya adı (PermissionError fallback)
# ─────────────────────────────────────────────────────────────────────────
fix(
    'filepath.with_stem(filepath.stem + f"_{datetime.now().strftime(\'%H%M%S\')}")',
    'filepath.with_stem(filepath.stem + f"_{_now_local().strftime(\'%H%M%S\')}")',
    "Satır ~2472: fallback dosya adı → _now_local()"
)

# ─────────────────────────────────────────────────────────────────────────
# 7. Satır ~1697: Döngü log mesajı
# ─────────────────────────────────────────────────────────────────────────
fix(
    'logger.info(f"🔄 YENİ DÖNGÜ — {datetime.now().strftime(\'%H:%M:%S\')} "',
    'logger.info(f"🔄 YENİ DÖNGÜ — {_now_local().strftime(\'%H:%M:%S\')} "',
    "Satır ~1697: döngü log → _now_local()"
)

# ─────────────────────────────────────────────────────────────────────────
# 8. Satır ~2662: Scheduler "Sonraki Tarama" log
# ─────────────────────────────────────────────────────────────────────────
fix(
    'logger.info(f"⏰ Sonraki Tarama: {(datetime.now()+timedelta(minutes=interval_minutes)).strftime(\'%H:%M:%S\')}")',
    'logger.info(f"⏰ Sonraki Tarama: {(_now_local()+timedelta(minutes=interval_minutes)).strftime(\'%H:%M:%S\')}")',
    "Satır ~2662: Sonraki Tarama log → _now_local()"
)

# ─────────────────────────────────────────────────────────────────────────
# Kaydet
# ─────────────────────────────────────────────────────────────────────────
main_path.write_text(content, encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────
# Rapor
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DATETIME FIX SONUCU")
print("=" * 55)
for r in results:
    print(f"  {r}")

# Kalan naive datetime.now() kontrolü
remaining = [
    (i+1, line.rstrip())
    for i, line in enumerate(content.splitlines())
    if "datetime.now()" in line
    and "# helper" not in line
    and "def _now" not in line
    and line.strip().startswith("#") is False
]

print()
if remaining:
    print(f"⚠️  Kalan {len(remaining)} datetime.now() çağrısı:")
    for lineno, line in remaining:
        print(f"   Satır {lineno}: {line.strip()[:80]}")
else:
    print("✅ Tüm datetime.now() çağrıları temizlendi!")

print()
print("Doğrulama:")
print("  grep -n 'datetime.now()' main.py | grep -v '# helper\\|def _now\\|^.*#'")
