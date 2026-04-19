#!/usr/bin/env python3
"""
main.py için kapsamlı patch scripti.

Düzelttikleri sorunlar:
    Sorun 4  — Timezone/naive datetime → UTC-aware helper fonksiyonlar
    Sorun 2  — Bozuk kategori for-döngüsü → _compute_category_tops helper
    Sorun 1+9 — initial_train binary target → R-multiple unified
    Sorun     — retrain_from_experience TIMEOUT magic number (0.16) → gerçek risk

Kullanım:
    cd hybrid_crypto_bot/src
    python main_patch.py

Önemli: Bu script orijinal main.py'yi yedekler (main.py.bak_sorun_fix).
Bir şeyler yanlış giderse:
    cp main.py.bak_sorun_fix main.py
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path(__file__).parent / "main.py"
BACKUP = Path(__file__).parent / f"main.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not TARGET.exists():
    raise FileNotFoundError(f"main.py bulunamadı: {TARGET}")

# ─── Yedek al ───
shutil.copy2(TARGET, BACKUP)
print(f"📦 Yedek alındı: {BACKUP}")

content = TARGET.read_text(encoding="utf-8")
changes_made = []

# =============================================================================
# [SORUN 4] — UTC-aware datetime helper fonksiyonları ekle
# LOCAL_TZ tanımından hemen sonra ekliyoruz
# =============================================================================
TZ_HELPERS = '''
# =============================================================================
# [SORUN 4 DÜZELTMESİ] — UTC-aware datetime helper fonksiyonları
# =============================================================================
# Eski kodda datetime.now() (naive, timezone'suz) kullanılıyordu.
# _compute_daily_pnl bu naive string'leri UTC sanıp +3 saat çevirince
# 3 saatlik kayma oluşuyordu. Artık tüm zaman üretimi UTC veya TR-aware.
#
# Kullanım:
#   _now_utc()  → UTC datetime (karşılaştırma, hesaplama)
#   _now_local() → TR datetime (log, display)
#   _iso_tr()   → Excel için TR zaman string'i (timezone bilgisi olmadan)

def _now_utc() -> datetime:
    """UTC-aware şimdi — karşılaştırma ve hesaplamalar için."""
    return datetime.now(timezone.utc)

def _now_local() -> datetime:
    """TR (LOCAL_TZ) aware şimdi — log ve gösterim için."""
    return datetime.now(LOCAL_TZ)

def _iso_tr(dt: datetime = None) -> str:
    """
    ISO formatında TR zaman string'i (Excel için okunabilir).
    Excel tz-aware ISO'yu parse edemeyebilir, bu yüzden
    saf string olarak yazıyoruz ama TR saatiyle.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%dT%H:%M:%S.%f")

'''

# LOCAL_TZ tanımından sonra helper'ları ekle
LOCAL_TZ_DEF = 'LOCAL_TZ = timezone(timedelta(hours=3), name="TRT")'
if LOCAL_TZ_DEF in content and "_now_utc" not in content:
    content = content.replace(
        LOCAL_TZ_DEF,
        LOCAL_TZ_DEF + "\n" + TZ_HELPERS,
        1
    )
    changes_made.append("✅ SORUN 4: UTC helper fonksiyonları eklendi (_now_utc, _now_local, _iso_tr)")
else:
    if "_now_utc" in content:
        changes_made.append("⏭️  SORUN 4: Helper fonksiyonlar zaten mevcut, atlandı")
    else:
        changes_made.append("⚠️  SORUN 4: LOCAL_TZ tanımı bulunamadı — helper eklenemedi, manuel ekle!")

# =============================================================================
# [SORUN 4] — Kritik datetime.now() çağrılarını UTC-aware yap
# =============================================================================

# Excel açılış tarihi kaydı (en çok kullanılan yer)
old_acilis = '"Tarih (Açılış)": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),'
new_acilis = '"Tarih (Açılış)": _iso_tr(_now_utc()),  # UTC üretilir, TR gösterilir'
if old_acilis in content:
    content = content.replace(old_acilis, new_acilis, 1)
    changes_made.append("✅ SORUN 4: Excel açılış tarihi → _iso_tr(_now_utc())")
else:
    changes_made.append("⚠️  SORUN 4: Excel açılış tarihi satırı bulunamadı — manuel kontrol")

# paper trade kapanış zamanı (birden fazla yerde)
old_kapanis_1 = "kapanis_zamani = datetime.now()\n                                    df.at[idx, 'Tarih (Kapanış)'] = kapanis_zamani.strftime"
new_kapanis_1 = "kapanis_zamani = _now_utc()\n                                    df.at[idx, 'Tarih (Kapanış)'] = _iso_tr(kapanis_zamani)"
count_1 = content.count("kapanis_zamani = datetime.now()")
if count_1 > 0:
    content = content.replace("kapanis_zamani = datetime.now()", "kapanis_zamani = _now_utc()")
    # strftime çağrısını da güncelle
    content = content.replace(
        "df.at[idx, 'Tarih (Kapanış)'] = kapanis_zamani.strftime(\"%Y-%m-%dT%H:%M:%S.%f\")",
        "df.at[idx, 'Tarih (Kapanış)'] = _iso_tr(kapanis_zamani)"
    )
    changes_made.append(f"✅ SORUN 4: kapanis_zamani × {count_1} → _now_utc()")
else:
    changes_made.append("⚠️  SORUN 4: kapanis_zamani = datetime.now() bulunamadı")

# _compute_daily_pnl içindeki naive datetime parse uyarısı güncelle
old_naive_warn = (
    "if closed_dt.tzinfo is None:\n"
    "                        closed_dt = closed_dt.replace(tzinfo=timezone.utc)"
)
new_naive_warn = (
    "if closed_dt.tzinfo is None:\n"
    "                        # Naive datetime geldi — UTC varsayıyoruz\n"
    "                        # Bu dal artık tetiklenmemeli (_iso_tr() her yerde UTC üretiyor)\n"
    "                        logger.debug(f'Naive datetime: {closed_at} — UTC varsayılıyor')\n"
    "                        closed_dt = closed_dt.replace(tzinfo=timezone.utc)"
)
if old_naive_warn in content:
    content = content.replace(old_naive_warn, new_naive_warn, 1)
    changes_made.append("✅ SORUN 4: _compute_daily_pnl naive datetime dalına açıklama eklendi")

# =============================================================================
# [SORUN 2] — _compute_category_tops static metod ekle
# Ekleme yeri: _analyze_coin metodundan hemen önce (class içinde)
# =============================================================================

COMPUTE_CATEGORY_TOPS = '''
    @staticmethod
    def _compute_category_tops(
        all_scores: list,
        categories: dict
    ) -> dict:
        """
        [SORUN 2 DÜZELTMESİ] — Kategori bazlı en güçlü indikatörü bulur.

        Eski kodda iç içe bozuk for-döngüsü vardı:
        - `if 'matched' in locals()` her zaman True dönerdi
        - `if not cat_scores:` döngü içinde değişkeni güncellemiyordu
        - Sonuç: category_tops neredeyse her zaman boş dict dönüyordu
        - ic_cat_trend/momentum/volatility/volume feature'ları hep NaN oluyordu

        Bu implementasyon iki aşamalı temiz eşleştirme yapar:
        1. startswith eşleşmesi (tam ön ek)
        2. contains eşleşmesi (daha esnek, min 3 karakter)

        Parameters
        ----------
        all_scores : List[IndicatorScore]
            selector.evaluate_all_indicators() çıktısı
        categories : Dict[str, List[str]]
            {'trend': ['ADX', 'Aroon', ...], 'momentum': ['RSI', ...], ...}

        Returns
        -------
        Dict[str, Dict]
            {'trend': {'name': 'ADX_14', 'ic': -0.05}, ...}
            Boş kategoriler sonuç dict'e dahil edilmez.
        """
        result = {}

        for cat, cat_indicators in categories.items():
            # Aşama 1: startswith (tam ön ek eşleşmesi)
            cat_scores = [
                s for s in all_scores
                if any(
                    s.name.lower().startswith(ci.lower() + '_') or
                    s.name.lower().startswith(ci.lower())
                    for ci in cat_indicators
                )
            ]

            # Aşama 2: contains (esnek eşleşme, en az 3 karakter)
            if not cat_scores:
                cat_scores = [
                    s for s in all_scores
                    if any(
                        ci.lower() in s.name.lower()
                        for ci in cat_indicators
                        if len(ci) >= 3
                    )
                ]

            if cat_scores:
                top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                result[cat] = {
                    "name": top.name,
                    "ic": round(top.ic_mean, 4),
                }

        return result

'''

# _analyze_coin metodundan önce ekle
ANCHOR = "    def _analyze_coin(self, symbol: str, coin: str, scan_result=None) -> CoinAnalysisResult:"
if ANCHOR in content and "_compute_category_tops" not in content:
    content = content.replace(ANCHOR, COMPUTE_CATEGORY_TOPS + "    " + ANCHOR.strip())
    changes_made.append("✅ SORUN 2: _compute_category_tops static metod eklendi")
elif "_compute_category_tops" in content:
    changes_made.append("⏭️  SORUN 2: _compute_category_tops zaten mevcut, atlandı")
else:
    changes_made.append("⚠️  SORUN 2: _analyze_coin metod başlangıcı bulunamadı — _compute_category_tops eklenemedi")

# =============================================================================
# [SORUN 2] — _analyze_coin içindeki bozuk for-döngüsünü temizle
# Hedef: "dynamic_cat_tops = {}" ile başlayan bölümü _compute_category_tops çağrısına çevir
# =============================================================================

# _analyze_coin içindeki CATEGORIES tanımı ve bozuk döngüyü bul
OLD_CAT_BLOCK = '''            CATEGORIES = {
                'volume':  ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI'],
                'momentum':['RSI', 'Stoch', 'MFI', 'UO', 'MACD', 'PPO', 'ROC',
                            'TSI', 'CCI', 'Williams'],
                'trend':   ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST',
                            'Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'ATR', 'Keltner', 'Donchian', 'STD'],
            }
            dynamic_cat_tops = {}
            for cat, cat_indicators in CATEGORIES.items():
                cat_scores = [s for s in best_scores if any(
                    s.name.lower().startswith(ci.lower()) or ci.lower() in s.name.lower()
                    for ci in cat_indicators
                )]
                if not cat_scores:
                    cat_ind_lower = [x.lower() for x in cat_indicators]
                    matched = []
                    for s in best_scores:
                        s_lower = s.name.lower()
                        if not cat_scores:
                                for ci in cat_ind_lower:
                                    if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                        if len(ci) >= 3:
                                            matched.append(s)
                                            break
                        if 'matched' in locals():
                            cat_scores = matched
                        else:
                            cat_scores = []

                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    dynamic_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}

            result.category_tops = dynamic_cat_tops'''

NEW_CAT_BLOCK = '''            # [SORUN 2 DÜZELTMESİ] — Bozuk iç içe for-döngüsü yerine temiz helper
            CATEGORIES = {
                'volume':     ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI', 'MFI'],
                'momentum':   ['RSI', 'Stoch', 'UO', 'MACD', 'PPO', 'ROC', 'TSI', 'CCI', 'Williams', 'WILLR'],
                'trend':      ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST', 'Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'BBU', 'BBL', 'BBM', 'ATR', 'NATR', 'Keltner', 'Donchian'],
            }
            result.category_tops = self._compute_category_tops(best_scores, CATEGORIES)'''

if OLD_CAT_BLOCK in content:
    content = content.replace(OLD_CAT_BLOCK, NEW_CAT_BLOCK, 1)
    changes_made.append("✅ SORUN 2: _analyze_coin bozuk for-döngüsü → _compute_category_tops çağrısı")
else:
    changes_made.append("⚠️  SORUN 2: _analyze_coin içindeki bozuk for-döngüsü bulunamadı — manuel kontrol gerekli")

# =============================================================================
# [SORUN 1+9] — initial_train: binary target → R-multiple + bidirectional
# =============================================================================

# MIN_MOVE tanımını değiştir
OLD_MIN_MOVE = "MIN_MOVE = 0.0025"
NEW_MIN_MOVE = "MIN_MOVE = 0.0025  # Küçük fiyat hareketleri (TIMEOUT dead-band) için alt eşik"
content = content.replace(OLD_MIN_MOVE, NEW_MIN_MOVE, 1)

# initial_train içindeki ana loop bloğunu değiştir
OLD_INIT_LOOP = '''            for i in range(start_idx, end_idx):
                fwd_val = df_ind[target_col].iloc[i]
                if pd.isna(fwd_val):
                    continue
                if abs(fwd_val) < MIN_MOVE:
                    continue

                target = 1 if fwd_val > 0 else 0
                fake_direction = "LONG" if fwd_val > 0 else "SHORT" # YENİ EKLENDİ
                
                df_slice = df_ind.iloc[max(0, i-50):i+1].copy()
                if len(df_slice) < 40:
                    continue

                analysis_stub.category_tops = {}
                for cat, cat_indicators in CATEGORIES.items():
                    cat_scores = [s for s in all_scores if any(s.name.lower().startswith(ci.lower()) for ci in cat_indicators)]
                    if not cat_scores:
                        cat_ind_lower = [x.lower() for x in cat_indicators]
                        matched = []
                        for s in all_scores:
                            s_lower = s.name.lower()
                            if not cat_scores:
                                for ci in cat_ind_lower:
                                    if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                        if len(ci) >= 3:
                                            matched.append(s)
                                            break
                        if 'matched' in locals():
                            cat_scores = matched
                        else:
                            cat_scores = []

                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    dynamic_cat_tops = {}
                    for cat, cat_indicators in CATEGORIES.items():
                        cat_scores = [s for s in all_scores if any(
                            s.name.lower().startswith(ci.lower()) or ci.lower() in s.name.lower()
                            for ci in cat_indicators
                        )]
                        if not cat_scores:
                            cat_ind_lower = [x.lower() for x in cat_indicators]
                            matched = []
                            for s in all_scores:
                                s_lower = s.name.lower()
                                if not cat_scores:
                                    continue
                                for ci in cat_ind_lower:
                                    if s_lower.startswith(ci + '_') or s_lower.startswith(ci):
                                        if len(ci) >= 3:
                                            matched.append(s)
                                            break
                        if 'matched' in locals():
                            cat_scores = matched
                        else:
                            cat_scores = []

                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    dynamic_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}
                    
                analysis_stub.category_tops = dynamic_cat_tops

                fv = self.feature_eng.build_features(
                    analysis=analysis_stub,
                    ohlcv_df=df_slice
                )

                rows_X.append(fv.to_dict())
                rows_y.append(1 if target > 0 else 0)
                rows_dir.append(fake_direction) # YENİ EKLENDİ'''

NEW_INIT_LOOP = '''            # [SORUN 2 DÜZELTMESİ] — Kategori top'ları bir kere hesapla (döngü dışında)
            # all_scores değişmediği için her bar'da tekrar hesaplamaya gerek yok.
            analysis_stub.category_tops = self._compute_category_tops(all_scores, CATEGORIES)

            # [SORUN 1+9 DÜZELTMESİ] — ATR kolonu bul (R-multiple hesabı için gerekli)
            atr_col_train = None
            for cand in ['ATRr_14', 'ATR_14', 'ATRr_7', 'NATR_14']:
                if cand in df_ind.columns:
                    atr_col_train = cand
                    break

            if atr_col_train is None:
                logger.error("❌ ATR kolonu bulunamadı — initial_train R-multiple hesaplayamıyor.")
                logger.error("   pandas-ta'nın ATR hesabını yaptığından emin ol (calculator.calculate_all çağrıldı mı?)")
                return False

            # R-multiple sabitleri (risk_manager ile tutarlı)
            ATR_MULT = 3.0   # SL mesafesi = ATR × 3.0
            RR_RATIO = 1.5   # TP mesafesi = SL × 1.5

            logger.info(f"  R-multiple parametreleri: ATR_mult={ATR_MULT}, RR={RR_RATIO}, ATR_col={atr_col_train}")

            for i in range(start_idx, end_idx):
                fwd_val = df_ind[target_col].iloc[i]
                if pd.isna(fwd_val):
                    continue

                entry_price = df_ind['close'].iloc[i]
                atr_val = df_ind[atr_col_train].iloc[i]

                # NATR yüzde cinsinden gelir → $ cinsine çevir
                if atr_col_train == 'NATR_14':
                    atr_val = atr_val * entry_price / 100

                if pd.isna(entry_price) or pd.isna(atr_val) or atr_val <= 0 or entry_price <= 0:
                    continue

                # Synthetic SL/TP mesafeleri ($ cinsinden)
                sl_distance = atr_val * ATR_MULT
                tp_distance = sl_distance * RR_RATIO

                # Fiyat hareketi log-return'den simple'a çevir
                price_move_pct = np.exp(fwd_val) - 1
                price_move_usd = price_move_pct * entry_price

                df_slice = df_ind.iloc[max(0, i - 50):i + 1].copy()
                if len(df_slice) < 40:
                    continue

                # [SORUN 9 DÜZELTMESİ] — Her bar için HER İKİ YÖN ayrı eğitim örneği
                # Eski kod: fwd_val > 0 ise fake_direction=LONG → tautoloji (is_long feature target'ı açıklıyordu)
                # Yeni kod: Hem LONG hem SHORT için bağımsız R-multiple hesapla
                for fake_direction in ["LONG", "SHORT"]:
                    if fake_direction == "LONG":
                        # LONG: fiyat yukarı giderse kâr
                        if price_move_usd >= tp_distance:
                            r_multiple = RR_RATIO          # +1.5R (TP hit)
                        elif price_move_usd <= -sl_distance:
                            r_multiple = -1.0              # -1.0R (SL hit)
                        else:
                            # TIMEOUT: gerçek hareket / risk mesafesi
                            r_multiple = price_move_usd / sl_distance
                    else:  # SHORT
                        # SHORT: fiyat aşağı giderse kâr (perspektif ters)
                        if price_move_usd <= -tp_distance:
                            r_multiple = RR_RATIO          # +1.5R (TP hit)
                        elif price_move_usd >= sl_distance:
                            r_multiple = -1.0              # -1.0R (SL hit)
                        else:
                            # TIMEOUT: ters perspektif
                            r_multiple = -price_move_usd / sl_distance

                    # Dead-band filtresi: çok küçük hareketler noise, at
                    if abs(r_multiple) < 0.25:
                        continue

                    # R-multiple clipping: aşırı uç değerleri sınırla
                    r_multiple = max(-2.0, min(2.0, r_multiple))

                    fv = self.feature_eng.build_features(
                        analysis=analysis_stub,
                        ohlcv_df=df_slice,
                    )

                    rows_X.append(fv.to_dict())
                    rows_y.append(float(r_multiple))
                    rows_dir.append(fake_direction)'''

if OLD_INIT_LOOP in content:
    content = content.replace(OLD_INIT_LOOP, NEW_INIT_LOOP, 1)
    changes_made.append("✅ SORUN 1+9: initial_train binary target → R-multiple bidirectional (2 örnek/bar)")
else:
    changes_made.append("⚠️  SORUN 1+9: initial_train iç döngüsü bulunamadı (uzun satır farklılığı olabilir) — manuel uygulama gerekebilir")

# =============================================================================
# [SORUN 1] — retrain_from_experience: TIMEOUT magic number 0.16 düzelt
# =============================================================================
OLD_TIMEOUT_R = "r_multiple = pnl / 0.16 \n                                        r_multiple = max(-2.0, min(2.0, r_multiple))"
NEW_TIMEOUT_R = (
    "# Gerçek PnL / risk amount (SL mesafesi = ATR×3 × position_size olarak tahmin)\n"
    "                                        # %2 risk varsayımı: risk_amount ≈ balance × 0.02\n"
    "                                        # Normalize R ≈ pnl / risk_amount\n"
    "                                        risk_est = self._balance * 0.02 if self._balance > 0 else 1.0\n"
    "                                        r_multiple = pnl / risk_est if risk_est > 0 else 0.0\n"
    "                                        r_multiple = max(-2.0, min(2.0, r_multiple))"
)

if OLD_TIMEOUT_R in content:
    content = content.replace(OLD_TIMEOUT_R, NEW_TIMEOUT_R, 1)
    changes_made.append("✅ SORUN 1: retrain_from_experience TIMEOUT magic number 0.16 → balance×0.02 tahmini")
else:
    # Farklı format dene
    OLD_TIMEOUT_R_v2 = "r_multiple = pnl / 0.16 "
    if OLD_TIMEOUT_R_v2 in content:
        changes_made.append("⚠️  SORUN 1: TIMEOUT formülü satır formatı farklı — manual kontrol gerekli")
    else:
        changes_made.append("⚠️  SORUN 1: retrain_from_experience TIMEOUT formülü bulunamadı")

# =============================================================================
# [SORUN 2] — initial_train başındaki CATEGORIES tanımı ve bozuk kodu temizle
# (initial_train içindeki farklı bir CATEGORIES bloğu daha var)
# =============================================================================
OLD_INIT_CATEGORIES = '''            CATEGORIES = {
                'volume':  ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI'],
                'momentum':['RSI', 'Stoch', 'MFI', 'UO', 'MACD', 'PPO', 'ROC','TSI', 'CCI', 'Williams'],
                'trend':   ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST','Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'ATR', 'Keltner', 'Donchian', 'STD'],
            }
            for cat, cat_indicators in CATEGORIES.items():
                cat_scores = [s for s in all_scores if any(s.name.lower().startswith(ci.lower()) for ci in cat_indicators)]
                if cat_scores:
                    top = max(cat_scores, key=lambda s: abs(s.ic_mean))
                    train_cat_tops[cat] = {"name": top.name, "ic": round(top.ic_mean, 4)}'''

NEW_INIT_CATEGORIES = '''            # [SORUN 2 DÜZELTMESİ] — Temiz helper kullan
            CATEGORIES = {
                'volume':     ['OBV', 'CMF', 'VPT', 'FI', 'EOM', 'ADI', 'NVI', 'MFI'],
                'momentum':   ['RSI', 'Stoch', 'UO', 'MACD', 'PPO', 'ROC', 'TSI', 'CCI', 'Williams', 'WILLR'],
                'trend':      ['ADX', 'Aroon', 'PSAR', 'DPO', 'Vortex', 'KST', 'Ichimoku', 'SMA', 'EMA', 'WMA'],
                'volatility': ['BBW', 'BBU', 'BBL', 'BBM', 'ATR', 'NATR', 'Keltner', 'Donchian'],
            }
            train_cat_tops = self._compute_category_tops(all_scores, CATEGORIES)'''

if OLD_INIT_CATEGORIES in content:
    content = content.replace(OLD_INIT_CATEGORIES, NEW_INIT_CATEGORIES, 1)
    changes_made.append("✅ SORUN 2: initial_train CATEGORIES+döngüsü → _compute_category_tops")

# =============================================================================
# Dosyayı yaz
# =============================================================================
TARGET.write_text(content, encoding="utf-8")

# =============================================================================
# Özet raporu
# =============================================================================
print("\n" + "=" * 60)
print("  PATCH SONUCU")
print("=" * 60)
for c in changes_made:
    print(f"  {c}")

warnings = [c for c in changes_made if c.startswith("⚠️")]
print()
if warnings:
    print(f"⚠️  {len(warnings)} uyarı var — manuel kontrol gerekli:")
    for w in warnings:
        print(f"   {w}")
else:
    print("✅ Tüm değişiklikler başarıyla uygulandı!")

print()
print(f"📦 Yedek dosya: {BACKUP.name}")
print("   Sorun olursa:")
print(f"   cp {BACKUP} {TARGET}")
print()
print("Sonraki adım: cd src && python main.py --train")
print("Beklenen log: 'Binary target detected' mesajı ÇIKMAMALI")
print("              'Eğitim Verisi: ~1500 satır' (her bar 2 yön)")
