# House MD — Metin Sınıflandırma Projesi (Güncellendi)

**Görev:** Eldeki tüm özellik sütunlarını kullanarak `correct_prediction` (hastalık adı) tahmin etmek  
**Yaklaşım:** Seçenek C — Metin + Kategorik + Sayısal özelliklerin birleşimi  
**Veri:** House MD Türkçe veri seti (7282 satır, top 20 hastalık → ~1567 satır)

---

## Sütun Rolleri

| Sütun | Tür | Rol |
|---|---|---|
| `text` | Serbest metin | Özellik → TF-IDF |
| `Symptom` | Serbest metin | Özellik → TF-IDF (tıbbi metin grubu) |
| `Test` | Serbest metin | Özellik → TF-IDF (tıbbi metin grubu) |
| `Drug` | Serbest metin | Özellik → TF-IDF (tıbbi metin grubu) |
| `Procedure` | Serbest metin | Özellik → TF-IDF (tıbbi metin grubu) |
| `Organ` | Serbest metin | Özellik → TF-IDF (tıbbi metin grubu) |
| `Intent` | Kategorik | Özellik → One-Hot Encoding |
| `diagnosis_stage` | Kategorik | Özellik → One-Hot Encoding |
| `Emotion` | Kategorik | Özellik → One-Hot Encoding |
| `speaker` | Kategorik | Özellik → One-Hot Encoding |
| `Sarcasm` | Binary (0/1) | Özellik → direkt kullan |
| `season` | Sayısal | Özellik → direkt kullan |
| `episode` | Sayısal | Özellik → direkt kullan |
| `model_prediction` | Metin | **Dışla** (çoğu boş, veri sızıntısı riski) |
| `medical_entities` | JSON | Özellik → JSON çözümle, entity type'a göre düzleştir |
| `correct_prediction` | Kategorik | **Hedef (y)** |

---

## Adım 1 — Ortam Kurulumu

```bash
pip install pandas scikit-learn matplotlib seaborn scipy
```

---

## Adım 2 — Veri Yükleme ve EDA

- CSV'yi pandas ile oku (`sep=';'`, `encoding='utf-8-sig'`)
- Tüm sütunları gör, veri tiplerini kontrol et
- `correct_prediction` dağılımını incele (725 unique değer!)
- Boş değer oranlarını sütun sütun kontrol et

---

## Adım 3 — Hedef Sütun Temizleme

`correct_prediction` sütununda sorunlar:
- 1471 boş satır → sil
- "none", "1", "0" gibi gürültülü değerler → sil
- 725 unique sınıf → sadece **top 20** hastalığı tut (~1567 satır)

Top 20 hastalık seçildikten sonra veri seti kullanılabilir boyuta iner.

---

## Adım 4 — Özellik Mühendisliği

### 4a — Metin Özellikleri (TF-IDF)

`text` sütunu için ayrı TF-IDF:
```python
TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
```

`Symptom`, `Test`, `Drug`, `Procedure`, `Organ` sütunları **birleştirilir** ve tek TF-IDF:
```python
df['tibbi_metin'] = df[['Symptom','Test','Drug','Procedure','Organ']].fillna('').agg(' '.join, axis=1)
TfidfVectorizer(max_features=3000)
```

### 4b — Kategorik Özellikler (One-Hot Encoding)

`Intent`, `diagnosis_stage`, `Emotion`, `speaker` → boş değerleri `"bilinmiyor"` ile doldur → `pd.get_dummies()`

### 4c — JSON Çözümleme: medical_entities

`medical_entities` sütunu JSON formatında — her satırda `text` ve `type` alanları var.

**Entity type → Ana tip eşlemesi:**
| Ham type | Ana tip |
|---|---|
| Disease, Diagnosis, DS, Condition | `ent_Disease` |
| Symptom, SYMP | `ent_Symptom` |
| Test, TEST, Test Sonucu | `ent_Test` |
| Drug, DRUG, Medication | `ent_Drug` |
| Procedure, PROC, Treatment | `ent_Procedure` |
| Anatomy, Organ, ORG | `ent_Anatomy` |

Her satır için aynı tipteki entity metinleri birleştirilir → 6 yeni metin sütunu oluşur → TF-IDF uygulanır.

```python
# Örnek çıktı (tek satır):
ent_Disease  = "beyin tümörü vaskülit"
ent_Symptom  = "nöbet afazi lezyon"
ent_Test     = "MR gadolinyum kan testi"
ent_Drug     = "prednizon epinefrin"
ent_Procedure= "biyopsi cerrahi hava yolu"
ent_Anatomy  = "beyin damar boğaz"
```

### 4d — Sayısal Özellikler

`Sarcasm`, `season`, `episode` → direkt sayısal, boş ise 0 ile doldur

---

## Adım 5 — Özellik Birleştirme

Tüm özellik grupları `scipy.sparse.hstack` ile tek matrise birleştirilir:

```
X = hstack([X_text, X_tibbi, X_entity, X_kategorik, X_sayisal])
```

| Grup | Boyut |
|---|---|
| X_text (text TF-IDF) | 5000 |
| X_tibbi (Symptom+Test+Drug+Procedure+Organ TF-IDF) | 3000 |
| X_entity (medical_entities TF-IDF, 6 tip) | 3000 |
| X_kategorik (One-Hot) | ~50-100 |
| X_sayisal (Sarcasm, season, episode) | 3 |

---

## Adım 6 — Train/Test Bölme

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Adım 7 — Model Eğitimi

Denenecek modeller:

1. **Logistic Regression** — baseline
2. **LinearSVC** — metin + karışık veride güçlü
3. **Random Forest** — kategorik özellikler için iyi

---

## Adım 8 — Değerlendirme

- `classification_report` (her hastalık için precision / recall / F1)
- Confusion Matrix ısı haritası
- Model karşılaştırma grafiği (accuracy bar plot)

---

## Özet Akışı

```
Adım 1  →  Ortam kurulumu
Adım 2  →  Veri yükleme + EDA
Adım 3  →  Hedef temizleme (top 20 hastalık)
Adım 4  →  Özellik mühendisliği (metin + kategorik + sayısal)
Adım 5  →  Özellik birleştirme (hstack)
Adım 6  →  Train/test bölme
Adım 7  →  Model eğitimi
Adım 8  →  Değerlendirme
```
