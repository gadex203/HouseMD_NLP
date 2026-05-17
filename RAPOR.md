# House MD Teşhis Sınıflandırıcı — Proje Raporu

**Ders:** Doğal Dil İşleme  
**Tarih:** Mayıs 2026  
**Model:** LinearSVC · Türkçe NLP Pipeline  

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)  
2. [Giriş ve Motivasyon](#2-giriş-ve-motivasyon)  
3. [Kullanılan Yöntemler](#3-kullanılan-yöntemler)  
4. [Veri Seti Hazırlama ve Ön İşleme](#4-veri-seti-hazırlama-ve-ön-i̇şleme)  
5. [Modelin Canlıya Entegrasyonu](#5-modelin-canlıya-entegrasyonu)  
6. [Sonuçlar ve Değerlendirme](#6-sonuçlar-ve-değerlendirme)  

---

## 1. Proje Özeti

Bu proje, *House MD* dizisindeki hasta–doktor diyaloglarından hareketle **otomatik hastalık teşhisi** yapan bir Doğal Dil İşleme (DLİ) sistemi geliştirmeyi amaçlamaktadır. Ham metin ve yapılandırılmış tıbbi meta veriler birleştirilerek çok sınıflı bir metin sınıflandırma modeli oluşturulmuş; model Streamlit tabanlı bir web arayüzü üzerinden son kullanıcıya sunulmuştur.

---

## 2. Giriş ve Motivasyon

Tıbbi diyalog metinleri; semptom, ilaç, test ve prosedür gibi yapılandırılmış klinik bilginin yanı sıra duygu, ironi ve konuşmacı kimliği gibi bağlamsal bilgileri de içerir. Geleneksel kural tabanlı yaklaşımlar bu çok katmanlı yapıyı yeterince modelleyemez. Bu çalışmada:

- **Metin özellikleri** (TF-IDF) ile **yapılandırılmış meta veriler** (tıbbi varlıklar, kategorik ve sayısal değişkenler) birleştirilerek zengin bir özellik uzayı oluşturulmuştur.
- Verideki **sınıf dengesizliği** ve **alaycı (sarcastic) ifadeler** gibi zorluklar ele alınmıştır.
- Eğitilen model, gerçek zamanlı tahmin yapabilen bir **web uygulamasına** dönüştürülmüştür.

---

## 3. Kullanılan Yöntemler

### 3.1 Özellik Mühendisliği — `HouseMDFeaturizer`

Sklearn `Pipeline` uyumlu özel bir dönüştürücü (`HouseMDFeaturizer`) geliştirilmiştir. Bu dönüştürücü altı farklı özellik grubunu sparse matris olarak birleştirir:

| Özellik Grubu | Yöntem | Boyut |
|---|---|---|
| Diyalog metni | TF-IDF (unigram + bigram) | 5.000 |
| Tıbbi metin (semptom, ilaç vb.) | TF-IDF (unigram + bigram) | 3.000 |
| Varlık metinleri (JSON parse) | TF-IDF (unigram + bigram) | 3.000 |
| Kategorik değişkenler | One-Hot Encoding | değişken |
| Sayısal değişkenler | Ham değer (float) | 3 |
| Varlık sayıları | Ham sayı (per-tip + toplam) | 6 |

### 3.2 Sınıflandırıcı — LinearSVC

Yüksek boyutlu seyrek vektörler için verimliliği kanıtlanmış **LinearSVC** tercih edilmiştir.

- `class_weight='balanced'` → Sınıf dengesizliğini otomatik telafi eder.
- `C=1.0`, `max_iter=2000` → Regularizasyon ve yakınsama ayarları.
- Sarcasm etiketli örnekler eğitimde `sample_weight=0.3` ile ağırlıklandırılmıştır (alaycı diyalogların teşhis sinyali düşüktür).

### 3.3 Bağlam Penceresi (Context Window)

Her diyalog satırı, **aynı bölümdeki komşu satırlarla** (±1 satır) birleştirilerek genişletilmiştir. Bu yaklaşım:
- Tek cümlelik kısa ifadelerin bağlamını zenginleştirir.
- Aynı bölüm içinde tutarlı teşhis sinyallerinin modele aktarılmasını sağlar.

---

## 4. Veri Seti Hazırlama ve Ön İşleme

### 4.1 Ham Veri

Kaynak: `Last_HouseMD_DataSet.csv` — House MD dizisi diyalogları, sütun ayracı `;`.

**Başlıca sütunlar:**

| Sütun | Açıklama |
|---|---|
| `text` | Hasta/doktor diyalog metni |
| `correct_prediction` | Hedef etiket (hastalık adı) |
| `Symptom`, `Test`, `Drug`, `Procedure`, `Organ` | Yapılandırılmış tıbbi varlıklar |
| `medical_entities` | JSON formatında varlık listesi |
| `Intent`, `diagnosis_stage`, `Emotion`, `speaker` | Bağlamsal meta veriler |
| `Sarcasm` | İronik ifade bayrağı (0/1) |
| `season`, `episode` | Dizi sezon/bölüm numarası |

### 4.2 Veri Temizleme

1. **Eksik değer doldurma:** Tıbbi sütunlar (`Symptom` vb.) `'bilgi_yok'` ile doldurulmuştur.
2. **Geçersiz etiket eleme:** Sayısal, boş, `none` veya `-` değerli hedef etiketler çıkarılmıştır.
3. **Sınıf eşiği:** Veri setinde **50'den az örneği olan sınıflar** kaldırılmıştır; bu sayede model düşük örnekli sınıflarda aşırı öğrenmeye (overfitting) karşı korunmuştur.
4. **Varlık imputation:** `'bilgi_yok'` olan tıbbi sütunlar, `medical_entities` JSON'ından çekilerek doldurulmuştur.

### 4.3 Metin Ön İşleme

```
Ham metin
  → Küçük harfe çevirme
  → HTML etiketleri ve özel karakterlerin temizlenmesi
  → Tokenizasyon
  → Türkçe stop-word eleme (stemming uygulanmamıştır)
  → Sarcasm=1 ise tokenlara ALAY_ ön eki eklenmesi
```

**Stemming neden uygulanmadı?** Tıbbi terimler (ilaç/hastalık adları) kök bulma sırasında anlam kaybına uğrayabilir; bu nedenle terimler ham formlarıyla korunmuştur.

### 4.4 Eğitim/Test Ayrımı

- **%80 eğitim / %20 test**, `stratify=y`, `random_state=42`
- `HouseMDFeaturizer` tüm veriye fit edilmiştir (TF-IDF vocabulary tam kapsamlı olsun diye).
- `LinearSVC` yalnızca eğitim setine fit edilmiş, test seti ile değerlendirilmiştir.
- Deploy için `LinearSVC` tüm veriye yeniden fit edilmiştir (production kalitesi için).

---

## 5. Modelin Canlıya Entegrasyonu

### 5.1 Pipeline Kaydetme

Eğitim sonunda `featurizer + clf_deploy` çifti tek bir `sklearn.Pipeline` olarak `joblib` ile `model/pipeline.pkl` dosyasına kaydedilmiştir. Bu sayede tahmin aşamasında herhangi bir ön işleme kodu tekrar yazılmaz; ham DataFrame doğrudan pipeline'a verilir.

```python
artifact = {
    'pipeline': deploy_pipeline,   # featurizer + LinearSVC (tüm veriye fit)
    'metrics':  { ... }            # test metrikleri, sınıf raporu, confusion matrix
}
joblib.dump(artifact, 'model/pipeline.pkl')
```

### 5.2 Streamlit Arayüzü (`app.py`)

İki sekmeli bir web uygulaması geliştirilmiştir:

**Tahmin sekmesi**
- Kullanıcı diyalog metnini girer; tıbbi bilgiler (semptom, ilaç, test vb.) ve bağlam bilgileri (intent, emotion, speaker) opsiyonel olarak eklenebilir.
- `pipeline.predict()` ile tahmin yapılır.
- `pipeline.decision_function()` ile **En Yakın 5 Sınıf** ve ilgili karar skoru (min-max normalize edilmiş progress bar) gösterilir.

**Model Performansı sekmesi**
- Test doğruluğu, sınıf bazlı F1 skorları ve makro/ağırlıklı ortalamalar tablo olarak sunulur.
- F1 < 0.40 → kırmızı, 0.40–0.60 → sarı, > 0.60 → yeşil ile renklendirme uygulanmıştır.

### 5.3 Karar Skoru Hakkında Not

LinearSVC'nin `decision_function()` çıktısı olasılık değil, **karar hiperplânından imzalı mesafedir**. Negatif değerler normaldir; en az negatif olan sınıf tahmin edilen teşhistir. Arayüzde bu skorlar 0–1 aralığına min-max normalize edilerek görselleştirilmiştir.

---

## 6. Sonuçlar ve Değerlendirme

### 6.1 Model Performansı

| Metrik | Değer |
|---|---|
| Test Accuracy | `pipeline.pkl` içinde saklanmaktadır |
| Sınıf Sayısı | ≥ 50 örnek şartını sağlayan sınıflar |
| Sınıflandırıcı | LinearSVC (C=1.0, balanced) |

> Kesin metrik değerleri Streamlit uygulamasının **Model Performansı** sekmesinden görüntülenebilir.

### 6.2 Güçlü Yönler

- **Çok kaynaklı özellikler:** Metin, yapılandırılmış tıbbi varlık ve meta verinin birleşimi tek bir metin modeline kıyasla daha zengin temsil sağlamaktadır.
- **Varlık imputation:** Eksik tıbbi sütunlar JSON kaynaklı bilgiyle otomatik doldurulduğundan veri kaybı minimize edilmiştir.
- **Sarcasm ağırlıklandırma:** İronik ifadelerin teşhis sinyalini bozmadan modele dahil edilmesi sağlanmıştır.
- **Bağlam penceresi:** Komşu diyalog satırlarının birleştirilmesi kısa metinlerin bilgi içeriğini artırmaktadır.
- **Deploy uyumlu pipeline:** Tahmin için hiçbir ek ön işleme gerektirmeyen tek nesne.

### 6.3 Sınırlılıklar ve İyileştirme Önerileri

| Sınırlılık | Olası İyileştirme |
|---|---|
| Düşük örnekli sınıflar modelden dışlandı | Veri artırma (data augmentation) ile örnek sayısı artırılabilir |
| Stemming uygulanmadı | Tıbbi terime duyarlı bir Türkçe kök bulucu kullanılabilir |
| LinearSVC olasılık üretmiyor | `CalibratedClassifierCV` ile kalibrasyon eklenebilir |
| Bağlam penceresi sabit (±1) | Daha uzun pencere veya dizi modeli (LSTM/Transformer) denenebilir |
| Arayüz boşluk duyarlılığı | Boş alanlar `'bilgi_yok'` ile doldurulduğundan model çalışır; ancak dolu alanlar skoru iyileştirir |

---

*Bu rapor proje kapsamında üretilmiş olup tüm kod ve veriler `nlp_project/` dizininde bulunmaktadır.*
