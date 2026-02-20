# 🤖 Hybrid Crypto Trading Bot

> **Bitget USDT-M Perpetual Futures** için IC istatistiksel analizi + Gemini AI kararı + otomatik emir execution sistemi.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Bitget](https://img.shields.io/badge/Exchange-Bitget-orange)](https://bitget.com)
[![Version](https://img.shields.io/badge/Version-1.3.0-green)](https://github.com)

---

## 📋 İçindekiler

1. [Proje Nedir?](#-proje-nedir)
2. [Nasıl Çalışır?](#-nasıl-çalışır-pipeline)
3. [IC Analizi — Neden?](#-ic-information-coefficient-analizi--neden)
4. [AI Katmanı — Neden Gemini?](#-ai-katmanı--neden-gemini)
5. [Risk Yönetimi Matematiği](#-risk-yönetimi-matematiği)
6. [Proje Yapısı](#-proje-yapısı)
7. [Kurulum](#-kurulum)
8. [Konfigürasyon](#-konfigürasyon)
9. [Çalıştırma](#-çalıştırma)
10. [LaunchAgent (Otomatik Zamanlama)](#-launchagent-otomatik-zamanlama)
11. [Aktif Ayarlar](#-aktif-ayarlar)
12. [Log & İzleme](#-log--izleme)
13. [Güvenlik](#-güvenlik)
14. [Bilinen Limitler](#-bilinen-limitler)

---

## 🎯 Proje Nedir?

**Hedef:** Bitget borsasında USDT-M Perpetual Futures işlemi yapan, tamamen otomatik, istatistiksel olarak temelli bir al-sat botu.

**Felsefe:**
- Kararlar sezgiye değil **istatistiksel anlam** testine dayanır
- AI tek başına karar vermez — istatistiksel filtreden geçen sinyalleri **optimize** eder
- Her işlem **SL/TP zorunlu**, korumasız pozisyon açılamaz
- Sistem MacBook arka planında **LaunchAgent** ile her 75 dakikada otomatik çalışır

**Kullanılan Teknolojiler:**

| Katman | Teknoloji | Amaç |
|--------|-----------|-------|
| Veri | `ccxt` + Bitget API | OHLCV veri çekme |
| Analiz | `pandas-ta` | 58+ teknik indikatör |
| İstatistik | `scipy` + `numpy` | IC hesaplama, anlamlılık testi |
| AI | Google Gemini 2.5 Flash | Karar optimizasyonu |
| Execution | Bitget USDT-M API | Emir gönderme |
| Bildirim | Telegram Bot API | Anlık trade bildirimleri |
| Zamanlama | macOS LaunchAgent | Her 75 dk otomatik tetikleme |

---

## ⚙️ Nasıl Çalışır? (Pipeline)

```
┌─────────────────────────────────────────────────────────┐
│  Her 75 dakikada bir LaunchAgent tetikler                │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   1. COIN TARAYICI    │
         │   535 USDT çifti      │
         │   → Hacim filtresi    │  min $2M günlük hacim
         │   → Spread filtresi   │  max %0.1 spread
         │   → Top 15 coin seç   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   2. IC ANALİZİ       │
         │   Her coin için:      │
         │   4 timeframe × 58    │
         │   indikatör = 232     │
         │   IC hesabı           │
         │   → Güven skoru (0-100│
         │   → Yön (LONG/SHORT)  │
         │   → En iyi TF seç     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   3. GATE KEEPER      │
         │   IC < 40  → ❌ Atla  │
         │   IC 40-70 → 👁️ Rapor│
         │   IC > 70  → ✅ AI'a  │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   4. GEMİNİ AI        │
         │   IC sonuçları +      │
         │   market bağlamı      │
         │   → LONG/SHORT/WAIT   │
         │   → Güven skoru       │
         │   Güven < 60 → ❌ Atla│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   5. RİSK MOTORU      │
         │   ATR bazlı SL/TP     │
         │   Pozisyon sizing     │
         │   Kaldıraç hesabı     │
         │   Max 5 pozisyon kont.│
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   6. EXECUTION        │
         │   Market order → SL   │
         │   → TP gönder         │
         │   SL/TP başarısız →   │
         │   Pozisyonu KAPAT     │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   7. TELEGRAM         │
         │   Trade bildirimi     │
         │   SL/TP seviyeleri    │
         │   Cycle özeti         │
         └───────────────────────┘
```

---

## 📊 IC (Information Coefficient) Analizi — Neden?

### Ne Ölçer?

IC (Information Coefficient), bir teknik indikatörün **gelecekteki fiyat hareketini ne kadar iyi tahmin ettiğini** ölçen istatistiksel bir metrik. Spearman rank korelasyonu kullanılır:

```
IC = SpearmanCorr(indikatör_değeri[t], fiyat_getirisi[t+6])
```

- `IC = +1.0` → Mükemmel pozitif tahmin gücü
- `IC = 0.0` → Rastgele (indikatörün anlamı yok)
- `IC = -1.0` → Mükemmel negatif tahmin gücü

### Neden Klasik Crossover/Sinyal Yerine IC?

Klasik yaklaşım: *"RSI < 30 ise AL"* — bu **korelasyon yok, kural var**. Hangi piyasa koşulunda, hangi coin için geçerli olduğu test edilmemiş.

IC yaklaşımı: Her indikatörü **o coin için, o zaman diliminde, o piyasa koşulunda** istatistiksel olarak test eder. Sadece **gerçekten anlamlı** olanlar seçilir.

### Nasıl Hesaplanır?

```python
# 58+ indikatör hesapla (trend, momentum, volatilite, hacim)
# 4 timeframe: 15m, 30m, 1h, 2h
# Her indikatör için:

IC_mean  = mean(SpearmanCorr(ind, fwd_return_6bar))   # Ortalama tahmin gücü
IC_std   = std(IC_scores)                              # Kararlılık ölçüsü
IC_IR    = IC_mean / IC_std                            # Information Ratio

# Anlamlılık testi (p < 0.05):
# Sadece istatistiksel olarak anlamlı indikatörler geçer
```

### Güven Skoru Hesabı

```
top_norm  = normalize(en_iyi_IC, 0.02-0.40)  × 40%  # En iyi indikatörün gücü
avg_norm  = normalize(ortalama_IC, 0.02-0.15) × 25%  # Genel indikatör kalitesi
cnt_norm  = normalize(anlamlı_sayı, 0-50)     × 15%  # Kaç indikatör anlamlı
cons_norm = normalize(yön_tutarlılık, 0.5-1)  × 20%  # İndikatörler aynı yönde mi?

# Rejim düzeltmesi:
# ranging piyasa → ×0.85 (trend indikatörleri çalışmaz)
# volatile piyasa → ×0.80

composite_score = (top_norm + avg_norm + cnt_norm + cons_norm) × rejim_katsayısı
```

### Yön Belirleme

```
LONG  sinyali veren indikatörler > SHORT × 1.5  → LONG
SHORT sinyali veren indikatörler > LONG  × 1.5  → SHORT
Eşit / belirsiz                                  → NEUTRAL (trade yok)
```

---

## 🤖 AI Katmanı — Neden Gemini?

### IC Yeterli Değil mi?

IC istatistiksel olarak *geçmişte ne çalıştı* sorusunu yanıtlar. Ama piyasa dinamik — bazen:

- Makroekonomik haber var (Fed açıklaması, likidite krizi)
- Coin spesifik gelişme var (listing, hack, partnership)
- Genel piyasa rejimi değişmiş (boğa → ayı)

Gemini bu **bağlamsal** faktörleri değerlendirerek IC sinyalini onaylar veya reddeder.

### Prompt Yapısı

Gemini'ye şunlar gönderilir:
- En iyi IC skoru ve yönü
- Hangi indikatörler anlamlı (kategori bazlı)
- ATR, volatilite, piyasa rejimi
- 24h fiyat değişimi, hacim
- Hesaplanmış SL/TP seviyeleri
- Risk/Ödül oranı

Dönen cevap: `{"decision": "SHORT", "confidence": 85, "reasoning": "...", "atr_multiplier": 1.5}`

### Güven Eşiği

```
Gemini güveni < 60  → İşlem AÇILMAZ (low_confidence)
Gemini güveni ≥ 60  → Risk hesaplanır, işlem açılır
```

### Quota Yönetimi (Free Tier)

Gemini Free Tier: 5 istek/dakika. Her coin analizi arasında **12 saniye beklenir**. Quota aşılırsa sistem IC-only moduna geçer (AI atlanır, IC yönü kullanılır, güven ×0.8 düşürülür).

---

## 💰 Risk Yönetimi Matematiği

### Pozisyon Büyüklüğü (ATR Bazlı)

```
Risk per trade     = Bakiye × %2              # Örn: $56 × 0.02 = $1.12
Stop distance      = ATR × atr_multiplier      # Örn: 0.85 × 1.5 = $1.275
Position size      = Risk / Stop distance      # $1.12 / $1.275 = 0.878 SOL
Leverage           = (Position × Entry) / Margin_limit
```

### SL/TP Hesabı

```
SHORT işlem için:
  SL = entry + (ATR × multiplier)    # Zarar yönü: yukarı
  TP = entry - (ATR × multiplier × RR)  # Kâr yönü: aşağı
  
LONG işlem için:
  SL = entry - (ATR × multiplier)
  TP = entry + (ATR × multiplier × RR)

Minimum RR: 1.5  (TP mesafesi SL'nin en az 1.5 katı)
```

### Hard Limitler

| Kural | Değer | Gerekçe |
|-------|-------|---------|
| Risk per trade | %2 | Kelly kriteri altında güvenli bölge |
| Max açık pozisyon | 5 | Sermaye dağıtımı + margin yönetimi |
| Max SL mesafesi | %8 | ATR bazlı SL'nin cap'i |
| Min RR oranı | 1.5 | Pozitif beklenen değer için minimum |
| Max kaldıraç | 20x | Likidasyon riski kontrolü |
| Günlük kayıp limiti | %6 | Tilt önleme |
| Kill switch DD | %15 | Sistemi tamamen durdur |
| SL/TP zorunlu | ✅ | Korumasız pozisyon = pozisyon kapatılır |

### Kill Switch Mantığı

```python
drawdown = (initial_balance - total_balance) / initial_balance × 100

# total_balance = free + margin (açık pozisyon marjini dahil)
# NOT: Sadece free bakiye kullanmak yanlış — pozisyon açıkken
# free düşer ama para kaybolmaz, marginde kilitlidir.

if drawdown >= 15%:
    KILL SWITCH → Tüm işlemler durdurulur
    Telegram bildirimi gönderilir
```

---

## 📁 Proje Yapısı

```
hybrid_crypto_bot/
│
├── src/                          # Ana kaynak kodu
│   ├── main.py                   # Pipeline orkestrasyon + scheduler
│   ├── config.py                 # Merkezi config (.env + settings.yaml)
│   ├── paper_trader.py           # Simülasyon trade motoru
│   ├── performance_analyzer.py   # Trade performans analizi
│   │
│   ├── scanner/
│   │   └── coin_scanner.py       # 535 coin tarama + filtreleme
│   │
│   ├── data/
│   │   ├── fetcher.py            # Bitget OHLCV veri çekme (ccxt)
│   │   └── preprocessor.py       # Veri temizleme + normalize
│   │
│   ├── indicators/
│   │   ├── categories.py         # 58+ indikatör tanımı (4 kategori)
│   │   ├── calculator.py         # pandas-ta ile hesaplama motoru
│   │   └── selector.py           # IC bazlı istatistiksel seçim
│   │
│   ├── ai/
│   │   └── gemini_optimizer.py   # Gemini API entegrasyonu + karar motoru
│   │
│   ├── execution/
│   │   ├── bitget_executor.py    # Bitget Futures emir motoru (SL/TP/market)
│   │   └── risk_manager.py       # ATR bazlı pozisyon sizing
│   │
│   └── notifications/
│       └── telegram_notifier.py  # Telegram bildirim sistemi
│
├── config/
│   └── settings.yaml             # Tüm sistem parametreleri
│
├── logs/
│   ├── stdout.log                # Bot çıktısı
│   ├── stderr.log                # Hata logları
│   └── paper_trades/             # Paper trade kayıtları (JSON)
│
├── com.hybrid.crypto.bot.plist   # macOS LaunchAgent tanımı
├── .env                          # API anahtarları (GİZLİ - git'e girmez)
├── .env.example                  # .env şablonu
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Kurulum

### Gereksinimler

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.12+
- Bitget hesabı + API key
- Google AI Studio hesabı (Gemini API key - ücretsiz)
- Telegram Bot Token

### Adım 1: Repo'yu Klonla

```bash
git clone https://github.com/KULLANICI_ADI/hybrid_crypto_bot.git
cd hybrid_crypto_bot
```

### Adım 2: Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Adım 3: API Anahtarları

```bash
cp .env.example .env
# .env dosyasını düzenle:
```

```env
# Bitget API (Trade + Read izni, Withdraw KAPALI)
BITGET_API_KEY=your_api_key
BITGET_API_SECRET=your_api_secret
BITGET_PASSPHRASE=your_passphrase

# Google Gemini (https://aistudio.google.com)
GEMINI_API_KEY=your_gemini_key

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Adım 4: Bağlantı Testi

```bash
cd src
python -c "
from execution.bitget_executor import BitgetExecutor
ex = BitgetExecutor(dry_run=False)
bal = ex.fetch_balance()
print(f'Bağlantı OK | Bakiye: \${bal[\"total\"]:.2f}')
"
```

---

## ⚙️ Konfigürasyon

`config/settings.yaml` dosyası tüm sistem parametrelerini içerir:

```yaml
# === TARAYICI ===
scanner:
  min_volume_usd: 2_000_000    # Min günlük hacim ($2M)
  max_spread_pct: 0.1          # Max bid-ask spread
  top_n: 20                    # Taranacak max coin

# === IC ANALİZ ===
analysis:
  timeframes:                  # Aktif zaman dilimleri
    15m: 400                   # 400 bar veri
    30m: 300
    1h:  250
    2h:  200
  # NOT: 4h kaldırıldı (kısa vade odaklı)
  forward_period: 6            # IC hedef: 6 bar ilerisi
  alpha: 0.05                  # Anlamlılık eşiği (p < 0.05)

# === GATE KEEPER ===
gate:
  no_trade: 40                 # IC < 40 → işlem yok
  report_only: 55              # IC 40-70 → sadece rapor
  full_trade: 70               # IC > 70 → AI + trade

# === AI ===
ai:
  model: gemini-2.5-flash
  temperature: 0.3             # Düşük = tutarlı cevap
  min_confidence: 60           # AI güveni < 60 → trade yok

# === RİSK ===
risk:
  risk_per_trade_pct: 2.0      # Bakiyenin %2'si
  max_positions: 5             # Max açık pozisyon
  max_leverage: 20             # Max kaldıraç
  min_leverage: 2              # Min kaldıraç
  min_rr: 1.5                  # Min risk/ödül oranı
  max_sl_pct: 8.0              # Max SL mesafesi %8
  daily_loss_limit_pct: 6.0    # Günlük max kayıp
  kill_switch_pct: 15.0        # Kill switch DD eşiği

# === TELEGRAM ===
telegram:
  enabled: true
```

---

## 🚀 Çalıştırma

### Paper Trade (Simülasyon - Güvenli Test)

```bash
cd src
python main.py --top 15
```

### Canlı Trade (Gerçek Para)

```bash
cd src
python main.py --live --top 15
```

### Tek Çalıştırma (Scheduler Olmadan)

```bash
python main.py --live --top 15
# Bir cycle çalışır, çıkar
```

### Performans Raporu

```bash
python main.py --report
```

---

## ⏰ LaunchAgent (Otomatik Zamanlama)

Bot macOS LaunchAgent ile her **75 dakikada** otomatik çalışır.

### Kurulum

```bash
# LaunchAgents klasörünü oluştur
mkdir -p ~/Library/LaunchAgents

# plist dosyasını kopyala
cp com.hybrid.crypto.bot.plist ~/Library/LaunchAgents/

# Syntax kontrolü
plutil -lint ~/Library/LaunchAgents/com.hybrid.crypto.bot.plist

# Yükle ve başlat
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hybrid.crypto.bot.plist
```

### Durum Kontrol

```bash
launchctl list | grep hybrid     # Aktif mi?
```

Çıktı:
```
-    0    com.hybrid.crypto.bot
# - = şu an çalışmıyor (bekleme)
# 0 = son çalışma başarılı
```

### Yönetim Komutları

```bash
# Durdur
launchctl bootout gui/$(id -u)/com.hybrid.crypto.bot

# Tekrar başlat
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hybrid.crypto.bot.plist

# Hemen tetikle (75 dk beklemeden)
launchctl kickstart gui/$(id -u)/com.hybrid.crypto.bot
```

---

## 📊 Aktif Ayarlar (v1.3.0)

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Borsa | Bitget USDT-M Futures | Perpetual kontratlar |
| Çalışma aralığı | 75 dakika | LaunchAgent StartInterval |
| Taranan coin | 535 → Top 15 | Hacim + spread filtresi |
| Timeframe | 15m, 30m, 1h, 2h | 4h kaldırıldı |
| IC eşiği | > 70 → trade | Gate keeper |
| AI min güven | 60% | Altında trade açılmaz |
| Risk per trade | %2 bakiye | ATR bazlı sizing |
| Max pozisyon | 5 | Aynı anda max açık |
| Max kaldıraç | 20x | Hard limit |
| SL/TP | Zorunlu | Başarısız = pozisyon kapatılır |
| Kill switch | %15 drawdown | Total bakiye bazlı |
| AI model | Gemini 2.5 Flash | Free tier (quota yönetimli) |
| Bildirim | Telegram | Her trade + cycle özeti |

---

## 📋 Log & İzleme

### Canlı Log İzle

```bash
# Bot çıktısı
tail -f ~/hybrid_crypto_bot/logs/stdout.log

# Hata logu
tail -f ~/hybrid_crypto_bot/logs/stderr.log
```

### Trade Geçmişi

```bash
# Açılan tüm canlı tradelar
grep "CANLI TRADE AÇILDI" ~/hybrid_crypto_bot/logs/stdout.log

# SL/TP seviyeleri
grep -E "SL|TP" ~/hybrid_crypto_bot/logs/stdout.log | tail -20

# Cycle özetleri
grep "CYCLE.*ÖZET" ~/hybrid_crypto_bot/logs/stdout.log
```

### Açık Pozisyonlar

```bash
cd ~/hybrid_crypto_bot/src
python3 -c "
from execution.bitget_executor import BitgetExecutor
ex = BitgetExecutor(dry_run=False)
for p in ex.fetch_positions():
    print(f\"{p['symbol']} | {p['side'].upper()} | Entry: {p['entry_price']} | Qty: {p['amount']}\")
"
```

---

## 🔒 Güvenlik

### API Key Güvenliği

- API key'ler SADECE `.env` dosyasında — asla kod içinde değil
- `.env` dosyası `.gitignore`'da — GitHub'a gitmez
- Bitget API: **Trade + Read** izni — **Withdraw KAPALI**
- IP Whitelist önerilir (Bitget API ayarları)

### Risk Kontrolleri

- Tek pozisyon max risk: %2
- Günlük max kayıp: %6 → trading durdurulur
- Kill switch: %15 drawdown → sistem kapanır
- SL/TP zorunlu: başarısız gönderimde pozisyon otomatik kapatılır
- Max 5 açık pozisyon: 6.'sı açılmaz

---

## ⚠️ Bilinen Limitler

### Teknik

- **Gemini Free Tier:** 5 istek/dakika → her coin analizi 12s bekler → cycle süresi uzar
- **Düşük fiyatlı coinler:** `triggerPrice` precision hatası (48001) — `_format_trigger_price()` ile çözüldü
- **Bitget One-Way Mode:** `tradeSide=open/close` zorunlu — hedge mode ile uyumsuz

### İstatistiksel

- IC analizi **geçmiş veri** üzerinden hesaplanır — gelecek performansı garanti etmez
- Kısa veri geçmişine sahip yeni coinler (< 150 bar) analizden elenir
- Piyasa rejim değişimlerinde (sudden regime shift) IC sinyalleri gecikebilir

### Operasyonel

- MacBook kapalıysa LaunchAgent çalışmaz
- Bitget API rate limit aşımında işlemler atlanır (circuit breaker aktif)
- Gemini quota aşımında IC-only moda geçilir (AI atlanır)

---

## 📈 Performans Takibi

```bash
cd ~/hybrid_crypto_bot/src
python main.py --report
```

Rapor şunları içerir:
- Toplam trade sayısı + win rate
- Ortalama PnL per trade
- Max drawdown
- Sharpe ratio tahmini
- En iyi/kötü performans gösteren coinler
- Timeframe bazlı başarı oranı

---

## 🔄 Versiyon Geçmişi

| Versiyon | Tarih | Değişiklik |
|----------|-------|-----------|
| v1.3.0 | Şubat 2026 | Max 5 pozisyon, 4h TF kaldırıldı, 75dk interval |
| v1.2.0 | Şubat 2026 | AI güven filtresi %60, kill switch fix |
| v1.1.0 | Şubat 2026 | SL/TP precision fix (48001), one-way mode |
| v1.0.0 | Şubat 2026 | İlk canlı deployment |

---

*Geliştirici: Doğukan Gerengi | Şubat 2026*
*Lisans: Özel Kullanım*
