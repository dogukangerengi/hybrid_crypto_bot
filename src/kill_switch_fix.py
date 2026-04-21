#!/usr/bin/env python3
"""
kill_switch_fix.py — Kill switch ve Telegram parametre hatası düzelt

Sorun 1: dry_run'da _initial_balance=1000 set ediliyor ama
         _refresh_live_balance hemen ardından self._balance=paper_trader.balance($556)
         yapıyor. initial hâlâ $1000 → DD=%44 → yanlış kill switch.

Sorun 2: send_risk_alert_sync(message=...) yerine details= bekleniyor,
         balance/drawdown parametresi de yok.
"""

import shutil
from datetime import datetime
from pathlib import Path

main_path = Path(__file__).parent / "main.py"
backup    = main_path.with_suffix(f".bak_ks_{datetime.now().strftime('%H%M%S')}")
shutil.copy2(main_path, backup)
print(f"📦 Yedek: {backup.name}")

content = main_path.read_text(encoding="utf-8")
results = []

# ─── Düzeltme 1: _init_balance dry_run → paper_trader'dan al ─────────────
OLD_INIT = (
    "            if self.dry_run:\n"
    "                self._balance = self._initial_balance = 1000.0\n"
    "                logger.info(f\"💰 Paper bakiye: ${self._balance:.2f}\")"
)
NEW_INIT = (
    "            if self.dry_run:\n"
    "                # Paper trader'ın mevcut bakiyesini kullan (1000.0 değil)\n"
    "                # Böylece initial_balance gerçek başlangıç noktasını yansıtır\n"
    "                # ve kill switch yanlış tetiklenmez.\n"
    "                self._balance = self._initial_balance = self.paper_trader.balance\n"
    "                logger.info(f\"💰 Paper bakiye: ${self._balance:.2f} (paper_trader'dan)\")"
)
if OLD_INIT in content:
    content = content.replace(OLD_INIT, NEW_INIT, 1)
    results.append("✅ _init_balance: dry_run → paper_trader.balance kullanıyor")
else:
    results.append("⚠️  _init_balance: eski satır bulunamadı")

# ─── Düzeltme 2: send_risk_alert_sync parametre hatası ───────────────────
OLD_ALERT = (
    "            if self.notifier.is_configured():\n"
    "                self.notifier.send_risk_alert_sync(\n"
    "                    alert_type=\"KILL_SWITCH\",\n"
    "                    message=f\"⛔ Kill switch! DD={dd:.1f}%\",\n"
    "                    balance=self._balance, drawdown=dd,\n"
    "                )"
)
NEW_ALERT = (
    "            if self.notifier.is_configured():\n"
    "                self.notifier.send_risk_alert_sync(\n"
    "                    alert_type=\"KILL_SWITCH\",\n"
    "                    details=(\n"
    "                        f\"⛔ Kill switch tetiklendi!\\n\"\n"
    "                        f\"📉 Drawdown: %{dd:.1f}\\n\"\n"
    "                        f\"💰 Başlangıç: ${self._initial_balance:.2f}\\n\"\n"
    "                        f\"💰 Güncel: ${self._balance:.2f}\"\n"
    "                    ),\n"
    "                    severity=\"critical\",\n"
    "                )"
)
if OLD_ALERT in content:
    content = content.replace(OLD_ALERT, NEW_ALERT, 1)
    results.append("✅ send_risk_alert_sync: message= → details= düzeltildi")
else:
    results.append("⚠️  send_risk_alert_sync: eski satır bulunamadı, alternatif deneniyor...")
    # Alternatif: daha geniş eşleşme
    OLD_ALERT2 = (
        "                self.notifier.send_risk_alert_sync(\n"
        "                    alert_type=\"KILL_SWITCH\",\n"
        "                    message=f\"⛔ Kill switch! DD={dd:.1f}%\","
    )
    NEW_ALERT2 = (
        "                self.notifier.send_risk_alert_sync(\n"
        "                    alert_type=\"KILL_SWITCH\",\n"
        "                    details=f\"⛔ Kill switch! DD={dd:.1f}% | Başlangıç: ${self._initial_balance:.2f} | Güncel: ${self._balance:.2f}\","
    )
    if OLD_ALERT2 in content:
        content = content.replace(OLD_ALERT2, NEW_ALERT2, 1)
        # balance= ve drawdown= parametrelerini kaldır
        content = content.replace(
            "                    balance=self._balance, drawdown=dd,\n",
            "                    severity=\"critical\",\n",
            1
        )
        results.append("✅ send_risk_alert_sync: alternatif eşleşme ile düzeltildi")
    else:
        results.append("❌ send_risk_alert_sync: manuel düzeltme gerekli (satır ~389)")

main_path.write_text(content, encoding="utf-8")

# ─── Rapor ───────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  KILL SWITCH FIX SONUCU")
print("=" * 50)
for r in results:
    print(f"  {r}")

# Doğrulama
c = main_path.read_text(encoding="utf-8")
checks = [
    ("paper_trader.balance" in c and "self._initial_balance = 1000.0" not in c,
     "_initial_balance artık paper_trader'dan geliyor"),
    ("message=f\"⛔ Kill switch" not in c,
     "Eski 'message=' parametresi kaldırıldı"),
    ("details=" in c and "KILL_SWITCH" in c,
     "Yeni 'details=' parametresi mevcut"),
]
print()
for ok, label in checks:
    print(f"  {'✅' if ok else '❌'} {label}")

print()
print("Sonraki adım:")
print("  python main.py --schedule")
print()
print("Beklenen: kill switch TETIKLENMEMELI ($556 → initial=$556, DD=%0)")
