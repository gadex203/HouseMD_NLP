# House MD Teşhis Sınıflandırıcı

House MD diyaloglarından hastalık teşhisi yapan LinearSVC tabanlı NLP uygulaması.  
60 hastalık sınıfı · Türkçe metin ön işleme · sklearn Pipeline · Streamlit arayüzü

---

## Gereksinimler

- Python 3.10+
- `Last_HouseMD_DataSet(Sayfa1).csv` dosyası proje klasöründe mevcut olmalı

---

## Kurulum

```bash
pip install streamlit scikit-learn scipy pandas numpy joblib
```

---

## Çalıştırma

### 1. Adım — Modeli Eğit

> Sadece ilk seferinde çalıştırılır. `model/` klasörü oluşturulup model kaydedilir.

```bash
python train_model.py
```

Beklenen çıktı:
```
CSV yukleniyor...
Veri: 3648 satir, 60 sinif
Bagla penceresi +-1 uygulandi.
HouseMDFeaturizer tum veriye fit ediliyor...
X_full: (3648, 11480)
Egitim: (2918, 11480), Test: (730, 11480)
LinearSVC egitiliyor (sadece train)...
  Train Acc : 0.9983
  Test Acc  : 0.9479
Kaydedildi : model/pipeline.pkl
```

### 2. Adım — Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

---

## Proje Yapısı

```
nlp_project/
├── Last_HouseMD_DataSet(Sayfa1).csv   # Ham veri seti (gerekli)
├── featurizer.py                      # HouseMDFeaturizer sınıfı (pipeline için zorunlu)
├── train_model.py                     # Model eğitim scripti
├── app.py                             # Streamlit uygulaması
├── housemd_classification.ipynb       # Orijinal araştırma notebook'u
├── model/
│   └── pipeline.pkl                   # Eğitilmiş pipeline (train_model.py oluşturur)
└── images/                            # Notebook görselleri
```

---

## Model Pipeline

| Bileşen | Detay |
|---|---|
| Algoritma | LinearSVC (`C=1.0`, `class_weight='balanced'`, `max_iter=2000`) |
| Bağlam penceresi | ±1 komşu diyalog satırı birleştiriliyor (`PENCERE=1`) |
| Entity imputation | `bilgi_yok` tıbbi sütunlar `medical_entities` JSON'undan dolduruluyor |
| Metin özellikleri | TF-IDF (max 5000 özellik, 1-2 gram) |
| Tıbbi metin | TF-IDF (max 3000 özellik, 1-2 gram) |
| Entity özellikleri | TF-IDF (max 3000 özellik, 1-2 gram) |
| Kategorik | OneHotEncoder (Intent, Emotion, Speaker, Stage) |
| Sayısal | Sarcasm, Sezon, Bölüm, Entity sayıları |
| Ön işleme | Türkçe stop-word temizleme, sarcasm prefix (`ALAY_`) |
| Test Accuracy | **%94.79** (notebook: %94.52) |

---

## Mimari Not

`featurizer.py` dosyası pipeline'ın joblib ile doğru serialize/deserialize edilmesi için zorunludur.
`app.py`'de preprocessing kodu yoktur — tüm adımlar `pipeline.pkl` içinde paketlenmiştir.

---

## Sık Karşılaşılan Sorunlar

**`ModuleNotFoundError`**  
→ `pip install streamlit scikit-learn scipy pandas numpy joblib` komutunu çalıştır.

**`Model bulunamadı` hatası**  
→ Önce `python train_model.py` komutuyla modeli eğit.

**`FileNotFoundError: Last_HouseMD_DataSet...csv`**  
→ CSV dosyasının `train_model.py` ile aynı klasörde olduğunu kontrol et.

**`Can't get attribute 'HouseMDFeaturizer'` hatası**  
→ `featurizer.py` dosyasının proje klasöründe olduğundan emin ol.

**Port zaten kullanımda**  
→ `streamlit run app.py --server.port 8502` ile farklı port kullan.
