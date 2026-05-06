# 🚀 Hybrid Crypto Quant Bot

> **Binance USD-M Perpetual Futures** için Kurumsal Düzey (Quant) İstatistiksel Analiz + Ensemble ML (LightGBM & Random Forest) tabanlı otomatik işlem ve risk yönetim sistemi.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Binance](https://img.shields.io/badge/Exchange-Binance-orange)](https://binance.com)
[![Version](https://img.shields.io/badge/Version-Quant_Edition-green)](https://github.com)

---

## 📋 İçindekiler

1. [Proje Nedir?](#-proje-nedir)
2. [Sistem Mimarisi (Pipeline)](#-sistem-mimarisi-pipeline)
3. [Quant Özellikleri & İstatistik](#-quant-özellikleri--istatistik)
4. [ML Katmanı & Üçlü Bariyer](#-ml-katmanı--üçlü-bariyer-tbm)
5. [Gelişmiş Risk Yönetimi](#-gelişmiş-risk-yönetimi)
6. [Kurulum & Yapılandırma](#-kurulum--yapılandırma)
7. [Çalıştırma](#-çalıştırma)

---

## 🎯 Proje Nedir?

**Hedef:** Binance borsasında USDT-M Perpetual Futures işlemleri yapan, duygu ve sezgiden arındırılmış, pür istatistiksel ve makine öğrenimi temelli bir al-sat botu.

**Felsefe (Quant Yaklaşımı):**
- Kararlar "RSI kesişti" gibi basit kurallara değil, **FDR Düzeltmeli IC (Information Coefficient)** testlerine dayanır.
- İşlemler "Yöne Karar Ver" mantığıyla değil, **Beklenen Değer (Expected Value - EV > 0)** formülüyle onaylanır.
- Model eğitimi klasik zaman ufkuyla değil, Hedge Fonlarının kullandığı **Triple Barrier Method (TBM)** ile yapılır.
- Portföy riski sadece bakiye ile değil, açık işlemlerin **Korelasyon Matrisi** ile yönetilir.

| Katman | Teknoloji / Yöntem | Amaç |
|--------|-----------|-------|
| **Veri** | `ccxt` + Binance API | Hızlı ve güvenilir OHLCV çekimi |
| **İstatistik** | `scipy`, FDR (Benjamini-Hochberg) | Sinyal ayıklama ve False Positive engelleme |
| **ML Karar** | LightGBM & Random Forest (Ensemble) | Olasılık (Probability) tabanlı yön ve EV hesabı |
| **Risk** | ATR, Korelasyon Kontrolü, TBM | Dinamik SL/TP, Flash-crash koruması |
| **Zamanlama** | macOS LaunchAgent | Tam otomatik 7/24 döngü |

---

## ⚙️ Sistem Mimarisi (Pipeline)

```text
┌─────────────────────────────────────────────────────────┐
│  Otomatik Zamanlayıcı (Cron / LaunchAgent)              │
└────────────────────┬────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │ 1. COIN TARAYICI    │  Top 15-20 hacimli coin seçimi
          └──────────┬──────────┘  Min spread & Min hacim filtresi
                     ▼
          ┌─────────────────────┐
          │ 2. İSTATİSTİK (IC)  │  58 indikatör x 4 Timeframe test edilir.
          └──────────┬──────────┘  FDR (p-value) düzeltmesi uygulanır.
                     ▼
          ┌─────────────────────┐
          │ 3. REGIME BLOCK     │  Piyasa "RANGING" (Yatay) ise işlem 
          └──────────┬──────────┘  anında reddedilir (Hard Block).
                     ▼
          ┌─────────────────────┐
          │ 4. ML & EV KARARI   │  LGBM + RF Ensemble Modeli ile
          └──────────┬──────────┘  Kazanma Olasılığı x RR - Kaybetme x Risk > 0 (EV)
                     ▼
          ┌─────────────────────┐
          │ 5. RISK & PORTFÖY   │  Yeni pozisyon mevcut pozisyonlarla 
          └──────────┬──────────┘  yüksek korelasyonluysa (> 0.80) REDDEDİLİR.
                     ▼
          ┌─────────────────────┐
          │ 6. EXECUTION        │  Hesaplanan ATR bazlı SL ve TP ile
          └─────────────────────┘  Borsaya Atomik (Garantili) Emir Gönderimi.
```

---

## 📊 Quant Özellikleri & İstatistik

### 1. IC (Information Coefficient) ve FDR
Sistem klasik teknik analiz kullanmaz. Her indikatör geçmişe dönük olarak **Spearman Rank Korelasyonu** ile test edilir. 
- *Çoklu Test Sapması (Multiple Testing Bias)* riskini ortadan kaldırmak için **Benjamini-Hochberg (FDR)** p-value düzeltmesi uygulanır. 
- Yalnızca istatistiksel olarak anlamlı indikatörlerin ağırlıkları toplanarak ana bir **IC Score** oluşturulur.

### 2. Piyasa Rejimi (Regime Awareness)
Bot her coin için ADX ve volatilite bazlı piyasa rejimi tespiti yapar:
- `TRENDING`: Güçlü sinyal, çarpanlar artırılır.
- `VOLATILE`: Belirsizlik yüksek, risk azaltılır.
- `RANGING`: Yatay piyasalarda ML modelleri gürültü (noise) öğrenir. Bu nedenle **Hard Block** devreye girer ve işlem tamamen iptal edilir.

---

## 🧠 ML Katmanı & Üçlü Bariyer (TBM)

### 1. Triple Barrier Method (TBM) ile Eğitim
Eski ML sistemleri "10 mum sonra fiyat arttı mı?" diye sorar. Ancak bu yöntem fiyat 3. mumda stop (SL) olsa bile hedefi başarılı sayar. 
Projemiz, Marcos Lopez de Prado'nun **Triple Barrier Method**'unu kullanarak modeli eğitir:
1. **Üst Bariyer (Take Profit)**
2. **Alt Bariyer (Stop Loss)**
3. **Dikey Bariyer (Zaman Aşımı)**
Model sadece fiyatın hangi yöne gideceğini değil, *"Önce TP'ye mi yoksa SL'ye mi çarpacağını"* öğrenir.

### 2. Beklenen Değer (Expected Value)
ML Kararı, "Kazanma Şansı" ile yetinmez. **Risk-Reward** (Örn: 1.5R) hesaba katılır:
`EV = (Win_Probability * Reward) - (Loss_Probability * Risk)`
- Eğer `EV > 0` ise (istatistiksel olarak kârlıysa) işlem onaylanır.

---

## 💰 Gelişmiş Risk Yönetimi

Risk yönetimi, botun en agresif olduğu alandır. Kurallar esnetilemez:

| Kural | Değer / İşlev | Gerekçe |
|-------|-------|---------|
| **Korelasyon Koruması** | Limit: `0.80` | Aynı anda düşecek coinlere arka arkaya Long açmayı engeller. |
| **Risk Per Trade** | %2 | Kasa yönetiminde Kelly Kriteri sınırları. |
| **Max Açık İşlem** | 5 Adet | Sermayenin belli bir alanda sıkışmasını önler. |
| **Dinamik SL / TP** | ATR Bazlı | Sabit yüzdeler yerine piyasa volatilitesine göre genişler/daralır. |
| **Cooldown** | 4 Saat | Stop-Loss olan coinde intikam trade'ini (revenge trading) önler. |
| **Kill Switch** | %15 | Ciddi bir çöküşte veya piyasa değişiminde tüm sistemi kapatır. |

---

## 🛠️ Kurulum & Yapılandırma

### Sistem Gereksinimleri
- macOS veya Linux / Python 3.12+
- Binance API Key (Sadece Okuma ve Vadeli İşlem izni, Çekim İzni KAPALI olmalı)

### Kurulum Adımları
```bash
git clone https://github.com/KULLANICI_ADI/hybrid_crypto_bot.git
cd hybrid_crypto_bot

# Sanal ortam oluştur ve aktif et
python3.12 -m venv venv
source venv/bin/activate

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

### Çevresel Değişkenler (.env)
Ana dizine bir `.env` dosyası oluşturun ve Binance API bilgilerinizi girin:
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🚀 Çalıştırma

Bot iki farklı modda çalışabilir: **Paper Trading (Sanal)** ve **Live Trading (Canlı)**.

### Paper Trading (Tavsiye Edilen İlk Adım)
Sistem sanal bakiye ile çalışır ve piyasaya hiçbir emir göndermez. Sadece izler ve sonuçları raporlar.
```bash
source venv/bin/activate
python src/main.py
```

### Modeli Eğitme (TBM)
Triple Barrier Method kullanarak modeli sıfırdan eğitmek için:
```bash
python src/main.py --train
```

### Canlı İşlemler (Live)
```bash
python src/main.py --live
```

> ⚠️ **Uyarı:** Canlı moda geçmeden önce botu birkaç gün Paper modda çalıştırmanız, modelin piyasa koşullarına göre eğitildiğinden ve `logs/` altındaki raporların kârlılığından emin olmanız tavsiye edilir.

---
*Geliştirici: Doğukan Gerengi*
*Sistem Türü: Kantitatif İstatistik + Makine Öğrenimi (Quant/ML)*
