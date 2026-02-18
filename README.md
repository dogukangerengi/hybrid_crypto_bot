# 🚀 Hibrit Kripto Trading Bot

**IC (Information Coefficient) bazlı istatistiksel analiz + AI optimizasyon + otomatik emir yönetimi**

## 📋 Genel Bakış

Bu sistem üç katmanlı bir hibrit yaklaşım kullanır:

```
IC BEYİN (Tarama + Analiz)  →  AI GÖZ (Optimize)  →  BİTGET EL (Execution)
```

| Katman | Görev | Teknoloji |
|--------|-------|-----------|
| **IC Beyin** | 500+ coin tara, IC analiz, TF seç, yön belirle | pandas-ta, scipy |
| **AI Göz** | Entry/SL/TP optimize, grafik analiz | Google Gemini |
| **Bitget El** | Futures emir gönder, pozisyon izle, PnL raporla | CCXT, Bitget API |

### 🚦 IC Kapı Bekçisi

```
IC Güven < 55  →  ❌ İŞLEM YAPMA (AI'a bile sorma)
IC Güven 55-70 →  👁️ AI'a sor, rapor et AMA emir girme  
IC Güven > 70  →  ✅ AI optimize etsin + emir gir
```

## 🏗️ Proje Yapısı

```
hybrid_crypto_bot/
├── src/
│   ├── main.py                    # Ana orkestrasyon + scheduler
│   ├── config.py                  # Merkezi yapılandırma (.env + yaml)
│   ├── test_bitget_connection.py  # Bağlantı test scripti
│   │
│   ├── data/                      # Veri çekme + ön işleme
│   │   ├── fetcher.py             # Bitget OHLCV çekme (CCXT)
│   │   └── preprocessor.py        # Return, outlier, volatilite
│   │
│   ├── indicators/                # Teknik indikatörler
│   │   ├── categories.py          # 64+ indikatör tanımı
│   │   ├── calculator.py          # pandas-ta hesaplama motoru
│   │   └── selector.py            # IC bazlı istatistiksel seçim
│   │
│   ├── scanner/                   # Dinamik coin tarayıcı
│   │   └── coin_scanner.py        # Bitget tüm çiftler → filtre → Top N
│   │
│   ├── ai/                        # AI entry optimizasyon
│   │   ├── gemini_client.py       # Gemini API bağlantısı
│   │   ├── chart_generator.py     # Teknik grafik oluşturma
│   │   └── entry_optimizer.py     # Entry/SL/TP optimize + RR filtre
│   │
│   ├── execution/                 # Bitget emir yönetimi
│   │   ├── bitget_trader.py       # Futures emir gönderme
│   │   ├── position_manager.py    # Açık pozisyon izleme
│   │   └── risk_manager.py        # Pozisyon sizing + risk kontrolü
│   │
│   ├── notifications/             # Bildirim sistemi
│   │   └── telegram_bot.py        # Telegram bildirim + komutlar
│   │
│   └── utils/                     # Yardımcı araçlar
│       ├── trade_logger.py        # İşlem kayıt sistemi
│       └── helpers.py             # Ortak fonksiyonlar
│
├── config/
│   └── settings.yaml              # Tüm sistem parametreleri
├── data/                          # Veri cache
├── logs/                          # Log dosyaları
├── tests/                         # Birim testleri
├── .env                           # API key'ler (GİZLİ)
├── .env.example                   # Örnek .env şablonu
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚡ Hızlı Başlangıç

### 1. Projeyi Kur

```bash
# Projeyi klonla / dizine git
cd ~/hybrid_crypto_bot

# Python sanal ortam oluştur
python3.12 -m venv venv

# Sanal ortamı aktif et
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. API Key'leri Ayarla

```bash
# .env dosyası oluştur
cp .env.example .env

# .env dosyasını düzenle (nano, vim veya VSCode)
code .env
```

`.env` dosyasına gerçek key'lerini yaz:
```
BITGET_API_KEY=senin_key
BITGET_API_SECRET=senin_secret
BITGET_PASSPHRASE=senin_passphrase
GEMINI_API_KEY=senin_gemini_key
TELEGRAM_BOT_TOKEN=senin_bot_token
TELEGRAM_CHAT_ID=senin_chat_id
```

### 3. Bağlantıyı Test Et

```bash
cd src
python test_bitget_connection.py
```

Başarılı çıktı:
```
✅ Temel Bağlantı
✅ Ticker  
✅ OHLCV
✅ Multi-TF
✅ Bakiye
✅ Market Info
```

## ⚙️ Risk Parametreleri

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Risk/İşlem | %2 | Sermaye koruma |
| Max Açık Pozisyon | 2 | Küçük sermaye için yeterli |
| Max Margin/İşlem | %25 | Likidasyon önleme |
| Max Toplam Margin | %60 | Reserve bırak |
| Min Kaldıraç | 2x | Minimum etkinlik |
| Max Kaldıraç | 20x | Risk kontrolü |
| Min RR Oranı | 1.5 | Beklenen değer pozitif |
| Günlük Max Kayıp | %6 | Tilt önleme |
| Kill Switch | %15 DD | Sistemi durdur |

## 🔬 İstatistiksel Metodoloji

### Information Coefficient (IC)
```
IC = Spearman(indicator_t, return_{t+n})
```
- **Spearman korelasyonu**: Rank-based, outlier'lara robust
- **|IC| > 0.02**: Ekonomik olarak anlamlı
- **Benjamini-Hochberg FDR**: Multiple testing correction

### Composite Skor
| Faktör | Ağırlık |
|--------|---------|
| Top |IC| | %40 |
| Ortalama |IC| | %25 |
| Anlamlı İndikatör Sayısı | %15 |
| Yön Tutarlılığı | %20 |

## 🔒 Güvenlik

- API key'ler **sadece** `.env` dosyasında
- `.env` dosyası `.gitignore`'da (git'e girmez)
- Bitget API: Sadece Trade + Read izni
- Withdraw izni **KAPALI**
- Kill switch: Telegram `/durdur` komutu

## ⚠️ Uyarılar

1. **Yatırım tavsiyesi değildir** - Karar destek + execution sistemidir
2. **Paper trading ile test edin** - `settings.yaml`'da `environment: paper`
3. **Risk yönetimi sizin sorumluluğunuzdadır**
4. **Geçmiş performans gelecek sonuçları garanti etmez**

---

**Geliştirici**: Doğukan Gerengi  
**Versiyon**: 0.1.0-alpha  
**Borsa**: Bitget (USDT-M Perpetual Futures)  
**AI**: Google Gemini Pro
