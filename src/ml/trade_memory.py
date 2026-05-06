# =============================================================================
# TRADE MEMORY — Geçmiş Trade Hafızası + Retrain Feedback Loop (ADIM 4)
# =============================================================================
# Amaç: Her trade'in açılış özelliklerini ve kapanış sonucunu kaydet.
#       Yeterli trade birikince LightGBM modeli gerçek trade verisiyle retrain et.
#
# - TradeMemory → her trade bir eğitim örneği haline gelir
# - Model zamanla kendi hatalarından öğrenir (online/incremental learning)
#
# Pipeline içindeki yeri:
#   SignalValidator → [TRADE MEMORY: open_trade()] → BitgetExecutor
#   BitgetExecutor kapanınca → [TRADE MEMORY: close_trade()] → retrain_if_ready()
#   retrain_if_ready() → LGBMModel.retrain(X_train, y_train)
#
# Kayıt yöntemi:
# - Her trade JSON olarak logs/ml_trade_memory.json dosyasına kaydedilir
# - Dosya append edilir, silinmez → kalıcı hafıza
# - Başlangıçta dosya okunur, mevcut trade'ler RAM'e yüklenir
#
# İstatistiksel not:
# - Retrain için minimum trade sayısı: MIN_TRADES_FOR_RETRAIN (varsayılan 30)
# - Her RETRAIN_INTERVAL yeni kapanan trade'de tekrar eğitilir
# - Target: fwd_ret > 0 ise 1 (WIN), değilse 0 (LOSS) — binary classification
# =============================================================================

import json                                    # Trade verilerini JSON formatında kayıt
import logging                                 # Loglama
import uuid                                    # Benzersiz trade ID üretimi
from datetime import datetime, timezone        # UTC zaman damgaları
from pathlib import Path                       # Platform-bağımsız dosya yolları
from typing import Dict, List, Optional, Any   # Tip belirteçleri
from dataclasses import dataclass, field, asdict  # Yapılandırılmış veri sınıfları
from enum import Enum                          # Sabit değer enumları

import numpy as np                             # Sayısal hesaplamalar
import pandas as pd                            # DataFrame işlemleri

logger = logging.getLogger(__name__)           # Bu modüle özel logger


# =============================================================================
# SABİTLER
# =============================================================================

MIN_TRADES_FOR_RETRAIN = 30                    # İlk retrain için minimum kapalı trade sayısı
RETRAIN_INTERVAL = 15                          # [MADDE 14] Her X yeni kapanan trade'de retrain yap
                                               # Eski: 10. 225 trade'de 22 retrain çok fazlaydı.
                                               # 340 satırlık veriyle her retrain model parametrelerini
                                               # ciddi değiştiriyor, fold-arası IC varyansı artıyor.
                                               # 30'a çekmek stabilitiyi artırır → IR yükselir.
MEMORY_FILE_NAME = "ml_trade_memory.json"      # Kalıcı hafıza dosyası adı
MAX_MEMORY_SIZE = 2000                         # RAM'de max trade sayısı (eski olanlar silinir)

# [LABEL NOISE FİLTRESİ]
# Çok küçük |R| değerleri (gürültü bölgesi) eğitim kalitesini düşürür.
# Model "ne kadar kazandı?" sorusunu öğrenemez çünkü target sıfıra yakın.
# Bu eşiğin altındaki trade'ler get_training_data()'dan çıkarılır.
MIN_ABS_R_FOR_TRAINING = 0.12  # |R| < 0.12 → eğitimden çıkar (gürültü)


# =============================================================================
# ENUM: TRADE DURUMU
# =============================================================================

class TradeStatus(str, Enum):
    """Trade'in yaşam döngüsü durumu."""
    OPEN   = "OPEN"                            # Pozisyon açık, sonuç bilinmiyor
    CLOSED = "CLOSED"                          # Pozisyon kapandı, PnL biliniyor
    ERROR  = "ERROR"                           # Execution hatası, kullanılmaz


class TradeOutcome(str, Enum):
    """Kapanan trade'in sonucu."""
    WIN       = "WIN"                            # PnL > 0 (kâr)
    LOSS      = "LOSS"                           # PnL < 0 (zarar)
    BREAKEVEN = "BREAKEVEN"                      # PnL = 0 (başa baş, bug veya gerçek BE)
    UNKNOWN   = "UNKNOWN"                        # Henüz kapanmadı


# =============================================================================
# DATACLASS: TEK TRADE KAYDI
# =============================================================================

@dataclass
class TradeRecord:
    """
    Bir trade'in tüm bilgilerini tutan veri yapısı.

    Açılışta: trade_id, symbol, direction, entry_price, feature_snapshot doldurulur.
    Kapanışta: exit_price, pnl, outcome, status=CLOSED güncellenir.

    feature_snapshot → LightGBM'in o an kullandığı feature vektörü.
    Bu sayede retrain için X (features) ve y (outcome) matrisleri oluşturulabilir.
    """
    # ── Kimlik ──
    trade_id:         str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    # Benzersiz trade kimliği (UUID'nin ilk 12 karakteri, okunabilirlik için)

    # ── Coin bilgisi ──
    symbol:           str   = ""               # Tam sembol: 'BTC/USDT:USDT'
    coin:             str   = ""               # Kısa kod: 'BTC'
    direction:        str   = ""               # 'LONG' veya 'SHORT'
    timeframe:        str   = ""               # Kullanılan en iyi TF: '1h', '15m'

    # ── Fiyat bilgisi ──
    entry_price:      float = 0.0              # Pozisyon açılış fiyatı ($)
    exit_price:       float = 0.0              # Pozisyon kapanış fiyatı ($), 0 = henüz açık
    sl_price:         float = 0.0             # Stop-Loss fiyatı
    tp_price:         float = 0.0             # Take-Profit fiyatı

    # ── ML sinyal bilgisi ──
    ml_confidence:    float = 0.0              # LightGBM'in tahmin güveni (0-1 arası olasılık)
    ml_direction:     str   = ""               # LightGBM'in önerdiği yön
    ic_confidence:    float = 0.0             # IC skoruna dayalı güven (0-100)
    ic_direction:     str   = ""              # IC'nin önerdiği yön
    market_regime:    str   = ""              # Piyasa rejimi: 'trending', 'ranging', 'volatile'
    validated_conf:   float = 0.0             # SignalValidator sonrası güven (CI düzeltmeli)

    # ── Feature snapshot ──
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    # LightGBM'e verilen feature vektörü (kolon adı → değer).
    # Retrain sırasında bu dict'ten X matrisi oluşturulur.
    # Dikkat: JSON'a kaydedilir, float olmayan değerler dışlanır.

    # ── Risk ──
    position_size:    float = 0.0             # Pozisyon büyüklüğü (coin adedi)
    leverage:         int   = 1               # Kaldıraç
    risk_reward:      float = 0.0             # Risk/Ödül oranı
    atr:              float = 0.0             # ATR değeri ($)

    # ── Sonuç ──
    pnl:              float = 0.0             # Gerçekleşen PnL ($), 0 = açık
    pnl_pct:          float = 0.0            # PnL yüzde
    outcome:          str   = TradeOutcome.UNKNOWN  # WIN / LOSS / UNKNOWN
    exit_reason:      str   = ""             # 'SL', 'TP', 'MANUAL', 'TIMEOUT'

    # ── Zaman ──
    opened_at:        str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Trade açılış zamanı (UTC ISO format)
    closed_at:        str   = ""             # Trade kapanış zamanı (UTC ISO format)
    duration_minutes: float = 0.0            # Açık kalma süresi (dakika)

    # ── Durum ──
    status:           str   = TradeStatus.OPEN   # OPEN, CLOSED veya ERROR


# =============================================================================
# ANA SINIF: TRADE MEMORY
# =============================================================================

class TradeMemory:
    """
    Tüm trade'leri kalıcı olarak kaydeden ve retrain feedback loop yöneten sınıf.

    Kullanım:
    --------
    memory = TradeMemory(log_dir=Path("logs"))

    # Trade açıldığında
    record = memory.open_trade(symbol="BTC/USDT:USDT", ...)
    trade_id = record.trade_id  # paper_trade_id olarak sakla

    # Trade kapandığında
    memory.close_trade(trade_id, exit_price=50000, pnl=12.5, exit_reason="TP")

    # Retrain feedback
    from ml.lgbm_model import LGBMModel
    model = LGBMModel()
    memory.retrain_if_ready(model)  # Yeterli trade varsa retrain yapar
    """

    def __init__(
        self,
        log_dir: Path = Path("logs"),          # Trade'lerin kaydedileceği klasör
        min_trades: int = MIN_TRADES_FOR_RETRAIN,  # İlk retrain için minimum kapalı trade
        retrain_interval: int = RETRAIN_INTERVAL,  # Kaç yeni trade'de bir retrain
        max_memory: int = MAX_MEMORY_SIZE,     # RAM'de tutulacak max trade sayısı
    ):
        """
        TradeMemory başlat.

        Parameters:
        ----------
        log_dir : Path
            Trade kayıtlarının saklanacağı dizin.
            Yoksa otomatik oluşturulur.
        min_trades : int
            Retrain tetiklenmesi için gereken minimum kapalı trade sayısı.
        retrain_interval : int
            Her kaç yeni kapanan trade'de retrain yapılacağı.
        max_memory : int
            RAM'de tutulacak maksimum trade sayısı (eski kayıtlar temizlenir).
        """
        self.log_dir        = Path(log_dir)    # Dosya yolu
        self.min_trades     = min_trades       # İlk retrain eşiği
        self.retrain_int    = retrain_interval # Periyodik retrain eşiği
        self.max_memory     = max_memory       # Bellek limiti

        self._records: Dict[str, TradeRecord] = {}
        # trade_id → TradeRecord mapping (RAM'deki aktif kayıtlar)

        self._closed_since_last_retrain: int = 0
        # Son retrain'den bu yana kapanan trade sayacı

        self._total_retrain_count: int = 0     # Toplam kaç kez retrain yapıldı

        # Klasörü oluştur
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Kalıcı dosyadan mevcut trade'leri yükle
        self._load_from_disk()

        logger.info(
            f"📦 TradeMemory başlatıldı | "
            f"Yüklenen: {len(self._records)} trade | "
            f"Kapalı: {self._count_closed()} | "
            f"Retrain eşiği: {self.min_trades}"
        )

    # =========================================================================
    # PUBLIC: TRADE YAŞAM DÖNGÜSÜ
    # =========================================================================

    def open_trade(
        self,
        symbol:           str,
        coin:             str,
        direction:        str,
        entry_price:      float,
        sl_price:         float,
        tp_price:         float,
        timeframe:        str          = "",
        ml_confidence:    float        = 0.0,
        ml_direction:     str          = "",
        ic_confidence:    float        = 0.0,
        ic_direction:     str          = "",
        market_regime:    str          = "",
        validated_conf:   float        = 0.0,
        feature_snapshot: Dict         = None,
        position_size:    float        = 0.0,
        leverage:         int          = 1,
        risk_reward:      float        = 0.0,
        atr:              float        = 0.0,
    ) -> TradeRecord:
        """
        Yeni bir trade kaydı açar ve diske yazar.

        Bu fonksiyon BitgetExecutor veya PaperTrader'dan trade açılınca çağrılmalı.
        feature_snapshot → model o an hangi featurelar ile karar verdi?

        Returns:
        -------
        TradeRecord
            Oluşturulan kayıt. trade_id'yi sakla, close_trade() için gerekli.
        """
        record = TradeRecord(
            symbol           = symbol,
            coin             = coin,
            direction        = direction,
            entry_price      = entry_price,
            sl_price         = sl_price,
            tp_price         = tp_price,
            timeframe        = timeframe,
            ml_confidence    = ml_confidence,
            ml_direction     = ml_direction,
            ic_confidence    = ic_confidence,
            ic_direction     = ic_direction,
            market_regime    = market_regime,
            validated_conf   = validated_conf,
            feature_snapshot = self._sanitize_features(feature_snapshot or {}),
            # Sadece float değerleri kaydet, NaN/inf temizle
            position_size    = position_size,
            leverage         = leverage,
            risk_reward      = risk_reward,
            atr              = atr,
            status           = TradeStatus.OPEN,
        )

        self._records[record.trade_id] = record  # RAM'e ekle
        self._trim_memory()                      # Bellek limitini kontrol et
        self._append_to_disk(record)             # Diske yaz

        logger.info(f"📝 Trade hafızaya eklendi: {symbol} {direction} @ ${entry_price:.4f} | ML güven: {ml_confidence:.2f}%")
        
        return record

    def close_trade(
        self,
        trade_id:    str,
        exit_price:  float,
        pnl:         float,
        exit_reason: str  = "UNKNOWN",
    ) -> Optional[TradeRecord]:
        """
        Açık bir trade'i kapatır, sonucu kaydeder ve diske günceller.

        Parameters:
        ----------
        trade_id : str
            open_trade() tarafından döndürülen trade kimliği.
        exit_price : float
            Kapanış fiyatı ($).
        pnl : float
            Gerçekleşen kâr/zarar ($). Pozitif = kâr, negatif = zarar.
        exit_reason : str
            Kapanış nedeni: 'SL', 'TP', 'MANUAL', 'TIMEOUT', 'KILL_SWITCH'.

        Returns:
        -------
        TradeRecord veya None
            Güncellenen kayıt. trade_id bulunamazsa None döner (uyarı loglar).
        """
        record = self._records.get(trade_id)    # RAM'den bul

        if record is None:
            # Disk'ten yüklenmiş ama RAM'de olmayabilir, disk'ten ara
            record = self._find_on_disk(trade_id)

        if record is None:
            logger.warning(f"⚠️ close_trade: trade_id bulunamadı → {trade_id}")
            return None

        # Kapanış bilgilerini doldur
        now             = datetime.now(timezone.utc)
        record.exit_price    = exit_price
        record.pnl           = pnl
        record.exit_reason   = exit_reason
        record.status        = TradeStatus.CLOSED
        record.closed_at     = now.isoformat()
        record.outcome       = (
            TradeOutcome.WIN if pnl > 0
            else TradeOutcome.BREAKEVEN if pnl == 0
            else TradeOutcome.LOSS
        )
        # PnL=0 → BREAKEVEN: Model eğitiminde kullanılMAZ.
        # Fiyat halfway'e kadar gitmiş (doğru yön), breakeven aktifleşmiş,
        # sonra entry'ye dönmüş. Bu ne başarı ne başarısızlık — nötr veri.

        # PnL yüzde hesapla (entry_price sıfır değilse)
        if record.entry_price > 0:
            record.pnl_pct = (exit_price - record.entry_price) / record.entry_price * 100
            if record.direction == "SHORT":
                record.pnl_pct = -record.pnl_pct  # SHORT'ta yön tersine döner

        # Süre hesapla (dakika)
        try:
            opened = datetime.fromisoformat(record.opened_at)
            record.duration_minutes = (now - opened).total_seconds() / 60
        except Exception:
            record.duration_minutes = 0.0       # Parse hatası → 0

        self._records[trade_id] = record        # RAM'i güncelle
        self._update_on_disk(record)            # Disk'i güncelle

        self._closed_since_last_retrain += 1    # Retrain sayacını artır

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{emoji} Trade kapandı: {trade_id} | "
            f"{record.coin} {record.direction} | "
            f"PnL: ${pnl:+.2f} ({record.pnl_pct:+.1f}%) | "
            f"Neden: {exit_reason}"
        )

        return record

    # =========================================================================
    # PUBLIC: RETRAIN FEEDBACK LOOP
    # =========================================================================

    def retrain_if_ready(self, model: Any) -> bool:
        """
        Yeterli trade birikince LightGBM modelini gerçek trade verisiyle retrain eder.

        Koşullar:
        - Kapalı trade sayısı >= min_trades (ilk retrain)
        - VEYA son retrain'den bu yana >= retrain_interval yeni trade kapandı

        Bu fonksiyon her trade kapandıktan sonra çağrılmalı.

        Parameters:
        ----------
        model : LGBMModel
            lgbm_model.py'deki model nesnesi. retrain(X, y) metodu çağrılır.

        Returns:
        -------
        bool
            True → retrain yapıldı, False → henüz yeterli trade yok.
        """
        closed_count = self._count_closed()     # Toplam kapalı trade sayısı

        # İlk retrain: minimum trade sayısına ulaşıldı mı?
        first_ready   = closed_count >= self.min_trades and self._total_retrain_count == 0

        # Periyodik retrain: son retrain'den beri yeterli trade kapandı mı?
        periodic_ready = (
            self._total_retrain_count > 0
            and self._closed_since_last_retrain >= self.retrain_int
        )

        if not (first_ready or periodic_ready):
            remaining = (
                self.min_trades - closed_count if self._total_retrain_count == 0
                else self.retrain_int - self._closed_since_last_retrain
            )
            logger.debug(
                f"⏳ Retrain bekleniyor: {remaining} trade daha gerekli "
                f"(toplam kapalı: {closed_count})"
            )
            return False

        # Eğitim verisi hazırla
        X, y = self.get_training_data()

        if X is None or len(X) < self.min_trades:
            logger.warning(
                f"⚠️ Retrain için yeterli özellik verisi yok "
                f"(mevcut: {len(X) if X is not None else 0})"
            )
            return False

        logger.info(
            f"🔄 RETRAIN BAŞLIYOR | "
            f"Kapalı trade: {closed_count} | "
            f"Eğitim verisi: {len(X)} satır × {len(X.columns)} özellik | "
            f"WIN oranı: {y.mean():.2%}"
        )

        try:
            model.retrain(X, y)                 # LightGBM'i yeniden eğit

            self._total_retrain_count += 1      # Retrain sayısını artır
            self._closed_since_last_retrain = 0 # Sayacı sıfırla

            logger.info(
                f"✅ RETRAIN TAMAMLANDI | "
                f"#{self._total_retrain_count} | "
                f"{len(X)} örnek ile"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Retrain hatası: {e}", exc_info=True)
            return False

    def get_training_data(self) -> tuple:
        """
        Kapalı trade'lerden LightGBM eğitim matrisini oluşturur.

        Her kapalı trade bir satır → feature_snapshot kolon değerleri.
        Target (y): WIN = 1, LOSS = 0 (binary classification).

        Returns:
        -------
        (X, y) : (pd.DataFrame, pd.Series)
            X → feature matrisi, y → binary hedef.
            Hiç kapalı trade yoksa (None, None) döner.
        """
        closed = [
            r for r in self._records.values()
            if r.status == TradeStatus.CLOSED
            and r.feature_snapshot                 # Feature snapshot boş değilse
            and r.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS)
        ]

        if not closed:
            logger.warning("⚠️ get_training_data: Kapalı + feature'lı trade bulunamadı")
            return None, None

        rows = []
        labels = []

        for r in closed:
            # [MADDE 1] — Binary 0/1 yerine kontinü R-multiple kullan
            # Eski: 1 if WIN else 0 (bilgi kaybı: +0.1R ve +1.5R aynı görünüyordu)
            # Yeni: gerçek fiyat hareketi / SL mesafesi → model magnitude öğrenir
            r_val = TradeMemory.compute_actual_r_multiple(r)

            # [LABEL NOISE FİLTRESİ] |R| < 0.12 → gürültü bölgesi, eğitimden çıkar
            # Model "ne kadar kazandı?" sorusunu öğrenemez, sadece noise öğrenir
            # Bu filtre retrain kalitesini artırır ve IC'yi stabilize eder
            if abs(r_val) < MIN_ABS_R_FOR_TRAINING:
                continue

            rows.append(r.feature_snapshot)        # Dict → satır (sadece geçerlileri ekle)
            labels.append(r_val)

        X = pd.DataFrame(rows)                     # Feature matrisi
        y = pd.Series(labels, name="r_multiple")   # Target: kontinü R-multiple değerleri

        # Sonsuz değerleri ve aşırı NaN'lı kolonları temizle
        X = X.replace([np.inf, -np.inf], np.nan)  # inf → NaN
        nan_ratio = X.isna().mean()                # Her kolonun NaN oranı
        valid_cols = nan_ratio[nan_ratio < 0.5].index  # %50'den az NaN olan kolonlar
        X = X[valid_cols].fillna(X[valid_cols].median())  # Kalan NaN'ları medyan ile doldur

        win_count = sum(1 for v in labels if v > 0)   # Pozitif R → kârlı trade
        noise_filtered = len(closed) - len(rows)       # Noise filtresi ile çıkarılan
        logger.info(
            f"📊 Eğitim verisi hazırlandı (R-multiple): "
            f"{len(X)} satır × {len(X.columns)} kolon | "
            f"Pozitif R: {win_count}/{len(y)} (%{win_count/max(1,len(y))*100:.0f}) | "
            f"Ort R: {float(y.mean()):+.3f} | "
            f"Noise filtresi: {noise_filtered} trade çıkarıldı (|R|<{MIN_ABS_R_FOR_TRAINING})"
        )

        return X, y

    # =========================================================================
    # PUBLIC: SORGULAMA VE İSTATİSTİK
    # =========================================================================

    def get_stats(self) -> Dict:
        """
        Trade hafızasının özet istatistiklerini döndürür.

        Returns:
        -------
        dict
            total, open_count, closed_count, win_rate, avg_pnl, avg_duration
        """
        all_records = list(self._records.values())
        closed = [r for r in all_records if r.status == TradeStatus.CLOSED]
        wins   = [r for r in closed if r.outcome == TradeOutcome.WIN]

        return {
            "total_trades":           len(all_records),        # Toplam kayıt
            "open_trades":            sum(1 for r in all_records if r.status == TradeStatus.OPEN),
            "closed_trades":          len(closed),             # Kapanan trade sayısı
            "win_rate":               len(wins) / len(closed) if closed else 0.0,
            # Kazanma oranı (0-1)
            "avg_pnl":                np.mean([r.pnl for r in closed]) if closed else 0.0,
            # Ortalama PnL ($)
            "total_pnl":              sum(r.pnl for r in closed),
            # Toplam PnL ($)
            "avg_duration_minutes":   np.mean([r.duration_minutes for r in closed]) if closed else 0.0,
            # Ortalama trade süresi
            "total_retrain_count":    self._total_retrain_count,  # Kaç kez retrain yapıldı
            "closed_since_retrain":   self._closed_since_last_retrain,
            # Son retrain'den beri kapanan trade
            "next_retrain_in":        max(0, (
                self.min_trades - len(closed) if self._total_retrain_count == 0
                else self.retrain_int - self._closed_since_last_retrain
            )),
            # Bir sonraki retrain için kaç trade daha gerekli
        }

    def get_open_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Belirli bir açık trade kaydını döndürür."""
        r = self._records.get(trade_id)
        return r if r and r.status == TradeStatus.OPEN else None

    @property
    def open_trades(self) -> Dict[str, TradeRecord]:
        """
        Tüm açık trade'leri {trade_id: TradeRecord} dict olarak döndürür.
        
        main.py'deki close senkronizasyonu bu property üzerinden iterasyon yapar:
            for mem_id, mem_trade in self.trade_memory.open_trades.items():
        """
        return {
            tid: rec for tid, rec in self._records.items()
            if rec.status == TradeStatus.OPEN
        }

    def get_all_open(self) -> List[TradeRecord]:
        """Tüm açık trade'leri listeler."""
        return [r for r in self._records.values() if r.status == TradeStatus.OPEN]

    def get_recent_closed(self, n: int = 20) -> List[TradeRecord]:
        """
        En son kapanan N trade'i döndürür.

        Parameters:
        ----------
        n : int
            Döndürülecek trade sayısı (varsayılan: 20).
        """
        closed = [r for r in self._records.values() if r.status == TradeStatus.CLOSED]
        closed.sort(key=lambda r: r.closed_at, reverse=True)  # En yeni önce
        return closed[:n]

    def print_summary(self) -> None:
        """İnsan-okunabilir özet tablosunu konsola yazdırır."""
        stats = self.get_stats()
        closed = self.get_recent_closed(5)       # Son 5 trade

        print(f"\n{'═'*55}")
        print(f"  📦 TRADE MEMORY ÖZET")
        print(f"{'─'*55}")
        print(f"  Toplam Trade     : {stats['total_trades']}")
        print(f"  Açık             : {stats['open_trades']}")
        print(f"  Kapalı           : {stats['closed_trades']}")
        print(f"  Win Rate         : {stats['win_rate']:.1%}")
        print(f"  Ortalama PnL     : ${stats['avg_pnl']:+.2f}")
        print(f"  Toplam PnL       : ${stats['total_pnl']:+.2f}")
        print(f"  Retrain Sayısı   : {stats['total_retrain_count']}")
        print(f"  Sonraki Retrain  : {stats['next_retrain_in']} trade sonra")
        print(f"{'─'*55}")

        if closed:
            print(f"  Son {len(closed)} Kapalı Trade:")
            for r in closed:
                emoji = "✅" if r.pnl > 0 else "❌"
                print(
                    f"    {emoji} {r.coin:6} {r.direction:5} "
                    f"${r.pnl:+6.2f} ({r.pnl_pct:+.1f}%) "
                    f"{r.exit_reason}"
                )
        print(f"{'═'*55}\n")

    # =========================================================================
    # PRIVATE: DISK İŞLEMLERİ
    # =========================================================================

    @property
    def _memory_file(self) -> Path:
        """Kalıcı hafıza dosyasının tam yolu."""
        return self.log_dir / MEMORY_FILE_NAME

    def _load_from_disk(self) -> None:
        """
        Başlangıçta disk'ten mevcut trade kayıtlarını RAM'e yükler.

        Dosya yoksa ya da bozuksa boş başlar (hata fırlatmaz).
        Büyük dosyalarda son MAX_MEMORY_SIZE kaydı yükler.
        """
        if not self._memory_file.exists():
            logger.info("📂 Trade memory dosyası yok, sıfırdan başlanıyor")
            return

        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                raw = json.load(f)               # Tüm dosyayı JSON olarak oku

            # Son MAX_MEMORY_SIZE kaydı al (bellek limiti)
            if len(raw) > self.max_memory:
                raw = raw[-self.max_memory:]
                logger.info(f"⚠️ Bellek limiti: Son {self.max_memory} trade yüklendi")

            for entry in raw:
                try:
                    # Dict → TradeRecord (alan adları eşleşmeli)
                    record = TradeRecord(**entry)
                    self._records[record.trade_id] = record
                except (TypeError, KeyError) as e:
                    logger.debug(f"Bozuk kayıt atlandı: {e}")
                    continue

            logger.info(f"✅ Diskten {len(self._records)} trade yüklendi")

        except json.JSONDecodeError as e:
            logger.error(f"❌ Memory dosyası JSON hatası: {e}. Sıfırdan başlanıyor.")
            self._records = {}
        except Exception as e:
            logger.error(f"❌ Memory dosyası okunamadı: {e}")
            self._records = {}

    def _save_all_to_disk(self) -> None:
        """
        RAM'deki tüm trade'leri diske yazar (tam üzerine yaz).

        Kullanım: Toplu güncelleme veya compact işleminden sonra.
        """
        records_list = [self._record_to_dict(r) for r in self._records.values()]
        with open(self._memory_file, "w", encoding="utf-8") as f:
            json.dump(records_list, f, ensure_ascii=False, indent=2)

    def _append_to_disk(self, record: TradeRecord) -> None:
        """
        Yeni bir trade kaydını disk dosyasına ekler (append).

        Varolan dosyayı tamamen yeniden yazmak yerine sadece bu kaydı ekler.
        Dosya yoksa yeni oluşturur.
        """
        records_list = []

        if self._memory_file.exists():
            try:
                with open(self._memory_file, "r", encoding="utf-8") as f:
                    records_list = json.load(f)
            except Exception:
                records_list = []                # Bozuksa sıfırdan başla

        records_list.append(self._record_to_dict(record))

        with open(self._memory_file, "w", encoding="utf-8") as f:
            json.dump(records_list, f, ensure_ascii=False, indent=2)

    def _update_on_disk(self, record: TradeRecord) -> None:
        """
        Disk'teki mevcut bir trade kaydını günceller (trade_id ile bulur).

        close_trade() çağrısında kullanılır.
        """
        if not self._memory_file.exists():
            self._append_to_disk(record)         # Dosya yoksa direkt ekle
            return

        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                records_list = json.load(f)

            updated = False
            for i, entry in enumerate(records_list):
                if entry.get("trade_id") == record.trade_id:
                    records_list[i] = self._record_to_dict(record)  # Güncelle
                    updated = True
                    break

            if not updated:
                records_list.append(self._record_to_dict(record))   # Bulunamadıysa ekle

            with open(self._memory_file, "w", encoding="utf-8") as f:
                json.dump(records_list, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ Disk güncelleme hatası: {e}")
            # Fallback: tam kaydet
            self._save_all_to_disk()

    def _find_on_disk(self, trade_id: str) -> Optional[TradeRecord]:
        """
        RAM'de olmayan bir trade'i disk'te arar.

        Returns:
        -------
        TradeRecord veya None
        """
        if not self._memory_file.exists():
            return None

        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                records_list = json.load(f)

            for entry in records_list:
                if entry.get("trade_id") == trade_id:
                    record = TradeRecord(**entry)
                    self._records[trade_id] = record  # RAM'e yükle (cache)
                    return record

        except Exception as e:
            logger.error(f"❌ Disk arama hatası: {e}")

        return None

    # =========================================================================
    # PRIVATE: YARDIMCI METODLAR
    # =========================================================================

    def _count_closed(self) -> int:
        """RAM'deki kapalı trade sayısını döndürür."""
        return sum(1 for r in self._records.values() if r.status == TradeStatus.CLOSED)

    def _trim_memory(self) -> None:
        """
        RAM'deki trade sayısı max_memory'yi aşarsa en eski kapalı trade'leri siler.

        Açık trade'ler hiçbir zaman silinmez.
        """
        if len(self._records) <= self.max_memory:
            return

        # Kapalı trade'leri açılış zamanına göre sırala (en eski önce)
        closed_sorted = sorted(
            [(k, r) for k, r in self._records.items() if r.status == TradeStatus.CLOSED],
            key=lambda x: x[1].opened_at
        )

        to_delete = len(self._records) - self.max_memory  # Kaç tane silinecek
        for k, _ in closed_sorted[:to_delete]:
            del self._records[k]                # RAM'den sil (disk'te kalır)

        logger.debug(f"🧹 Bellek temizlendi: {to_delete} eski kapalı trade RAM'den kaldırıldı")

    @staticmethod
    def _sanitize_features(features: Dict) -> Dict[str, float]:
        """
        Feature dict'inden sadece geçerli float değerleri alır.

        NaN, inf, None değerleri dışlanır. Listeler/dicts dışlanır.
        LightGBM sadece sayısal feature ister.

        Parameters:
        ----------
        features : dict
            Ham feature vektörü (kolon adı → değer).

        Returns:
        -------
        dict
            Temizlenmiş {str: float} vektörü.
        """
        clean = {}
        for k, v in features.items():
            try:
                fv = float(v)                   # Float'a çevir
                if np.isfinite(fv):             # NaN ve inf'i dışla
                    clean[str(k)] = fv
            except (TypeError, ValueError):
                pass                            # Çevrilemeyen değerleri atla
        return clean

    @staticmethod
    def _record_to_dict(record: TradeRecord) -> Dict:
        """TradeRecord dataclass'ını JSON-serileştirilebilir dict'e çevirir."""
        d = asdict(record)                      # dataclasses.asdict() → iç içe dict
        return d

    # =========================================================================
    # [MADDE 1] — GERÇEK R-MULTIPLE HESABI
    # =========================================================================

    @staticmethod
    def compute_actual_r_multiple(record: 'TradeRecord') -> float:
        """
        Gerçek R-multiple hesaplar: (çıkış - giriş) / SL mesafesi, yöne göre.

        Binary etiketleme (+1.5 / -1.0) yerine sürekli değer döndürür.
        Bu sayede model "kazandı mı?" değil "ne kadar kazandı?" sorusunu öğrenir.

        Örnekler:
          LONG @ 100, SL @ 95, TP @ 107.5 (TP vuruldu)  → r = +7.5/5 = +1.5
          LONG @ 100, SL @ 95, çıkış @ 97 (SL'den önce)  → r = -3/5  = -0.6
          SHORT @ 100, SL @ 105, çıkış @ 92              → r = +8/5  = +1.6

        Parameters:
        ----------
        record : TradeRecord
            Kapalı bir trade kaydı (exit_price, entry_price, sl_price dolu olmalı).

        Returns:
        -------
        float
            R-multiple değeri, [-2.5, +2.5] aralığında sınırlandırılmış.
            Veri eksikse 0.0 döner.
        """
        entry      = record.entry_price          # Pozisyon giriş fiyatı ($)
        exit_price = record.exit_price            # Gerçek çıkış fiyatı ($)
        sl         = record.sl_price              # Stop-Loss fiyatı ($)

        sl_distance = abs(entry - sl)             # SL mesafesi (mutlak, $)

        if sl_distance <= 0 or entry <= 0 or exit_price <= 0:
            # Eksik veya hatalı veri — sıfır döndür (eğitimde dead-band filtresi atar)
            return 0.0

        if record.direction == "LONG":
            price_move = exit_price - entry       # LONG: yukarı hareket = kâr
        else:
            price_move = entry - exit_price       # SHORT: aşağı hareket = kâr

        r = price_move / sl_distance              # Sürekli R değeri

        # Aşırı outlier'ları kırp: -2.5R / +2.5R sınırı
        # Veri hatası veya slippage kaynaklı uç değerler modeli bozmasın.
        return max(-2.5, min(2.5, r))