import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# 1. VERİYİ OKUMA VE TEMİZLEME
veri_yolu = "src/logs/reports/model_egitim_raporu.xlsx"
print("📊 Veri yükleniyor...")
df = pd.read_excel(veri_yolu, sheet_name="3_Gecmis_Ham_Veri")

# HATA ÇÖZÜMÜ (DEVASA SAYILARI KIRPMA)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
sayisal_kolonlar = df.select_dtypes(include=[np.number]).columns
df[sayisal_kolonlar] = df[sayisal_kolonlar].clip(lower=-1000, upper=1000)

X = df.drop(columns=['TARGET_SONUC', 'TARGET_ACIKLAMA'])
y = df['TARGET_SONUC']

# 2. EĞİTİM VE TEST OLARAK BÖLME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 3. MODELLERİ EĞİTME
print("🌲 1. Beyin: Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

print("⚡ 2. Beyin: LightGBM eğitiliyor...")
lgbm_model = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', verbose=-1)
lgbm_model.fit(X_train, y_train)

# 4. TAHMİNLER
rf_tahminler = rf_model.predict(X_test)
lgbm_tahminler = lgbm_model.predict(X_test)

# İkisi de 1 (Kâr) diyorsa işleme gir (Sıkı Oylama / Hard Voting)
ortak_tahminler = (rf_tahminler == 1) & (lgbm_tahminler == 1)

# 5. TRADE METRİKLERİ HESAPLAMA
def trade_raporu(isim, tahminler, gercek_degerler):
    isleme_girilen = sum(tahminler == 1)
    if isleme_girilen == 0:
        return f"{isim:<18}: Hiç işleme girilmedi (Çok defansif)"
    kazanc = sum((tahminler == 1) & (gercek_degerler == 1))
    win_rate = kazanc / isleme_girilen
    return f"{isim:<18}: {isleme_girilen:2d} işleme girdi -> {kazanc:2d} kazanç | Win Rate: %{win_rate*100:.1f}"

print("\n==================================================")
print("🏆 İŞLEM BAŞARISI (Sadece Girilen İşlemlerdeki Win Rate)")
print("==================================================")
print(trade_raporu("🌲 Random Forest", rf_tahminler, y_test))
print(trade_raporu("⚡ LightGBM", lgbm_tahminler, y_test))
print(trade_raporu("🤝 İKİLİ ORTAK KARAR", ortak_tahminler, y_test))
print("==================================================")