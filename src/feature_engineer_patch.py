#!/usr/bin/env python3
"""
feature_engineer.py için Sorun 6 patch scripti.

Kullanım:
    cd hybrid_crypto_bot/src
    python feature_engineer_patch.py

Yaptığı işlem:
    feature_engineer.py içindeki ctf_* feature açıklamalarını günceller.
    Kod mantığı DEĞİŞMİYOR — sadece yorumlar netleştiriliyor.
"""

from pathlib import Path

target = Path(__file__).parent / "ml" / "feature_engineer.py"
if not target.exists():
    raise FileNotFoundError(f"Dosya bulunamadı: {target}")

content = target.read_text(encoding="utf-8")
original = content  # Rollback için

# ─── Değişiklik 1: _build_cross_tf_features içi yorum ───
content = content.replace(
    "features['ctf_best_score'] = float(max(scores))     # En yüksek TF composite skoru",
    "features['ctf_best_score'] = float(max(scores))     # En iyi TF'nin IC×100 skoru (tipik 5-20, yüzde DEĞİL)"
)

# ─── Değişiklik 2: ctf_avg_score yorum ───
content = content.replace(
    "features['ctf_avg_score'] = float(np.mean(scores))  # Ortalama TF skoru",
    "features['ctf_avg_score'] = float(np.mean(scores))  # Ortalama TF IC×100 skoru"
)

# ─── Değişiklik 3: ctf_score_std yorum ───
content = content.replace(
    "features['ctf_score_std'] = float(                  # TF skorları standart sapması",
    "features['ctf_score_std'] = float(                  # TF IC×100 skorları std (düşük=tutarlı)"
)

# ─── Değişiklik 4: get_feature_descriptions() içi açıklama ───
content = content.replace(
    "'ctf_best_score':         'En iyi TF composite skoru',",
    "'ctf_best_score':         'En iyi TF IC×100 skoru (0-40 tipik, yüzde değil)',"
)

content = content.replace(
    "'ctf_avg_score':          'Ortalama TF skoru',",
    "'ctf_avg_score':          'Ortalama TF IC×100 skoru (0-40 tipik)',"
)

content = content.replace(
    "'ctf_score_std':          'TF skorları std (düşük=tutarlı)',",
    "'ctf_score_std':          'TF IC×100 skorları std (düşük=tutarlı TF)  ',"
)

if content == original:
    print("⚠️  Hiçbir değişiklik yapılmadı — satırlar zaten güncel veya eşleşme bulunamadı.")
    print("    feature_engineer.py'yi manuel kontrol edin.")
else:
    changed = sum(1 for a, b in zip(original.splitlines(), content.splitlines()) if a != b)
    target.write_text(content, encoding="utf-8")
    print(f"✅ feature_engineer.py güncellendi — {changed} satır değişti.")
    print("   Değişen açıklamalar:")
    print("   - ctf_best_score: 'composite skor' → 'IC×100 skoru (0-40 tipik)'")
    print("   - ctf_avg_score:  'Ortalama TF skoru' → 'Ortalama TF IC×100 skoru'")
    print("   - ctf_score_std:  'TF skorları std' → 'TF IC×100 skorları std'")
    print()
    print("   NOT: Kod mantığı değişmedi, sadece yorumlar netleştirildi.")
