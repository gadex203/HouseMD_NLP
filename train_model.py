"""
House MD — Pipeline egitim scripti.
Notebook ile birebir ayni yaklasim:
  1. Bagla penceresi (PENCERE=1) uygula
  2. HouseMDFeaturizer → tum veriye fit (TF-IDF vocabulary tam)
  3. X_full'u split et → LinearSVC sadece X_train'e fit (notebook gibi)
  4. Test skoru hesapla → deploy icin LinearSVC'yi tum X'e refit et
  5. Pipeline kaydet
"""
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from featurizer import HouseMDFeaturizer, sample_weight_arr

CSV_PATH  = 'Last_HouseMD_DataSet(Sayfa1).csv'
MODEL_DIR = 'model'
ESIK      = 50
PENCERE   = 1


def load_and_clean(path):
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    tibbi = ['Symptom', 'Test', 'Drug', 'Procedure', 'Organ']
    df[tibbi] = df[tibbi].fillna('bilgi_yok')
    df = df.dropna(subset=['text', 'Emotion']).copy()
    df = df[df['correct_prediction'].notna()].copy()
    df['hedef'] = df['correct_prediction'].astype(str).str.strip().str.lower()
    df = df[~df['hedef'].str.match(r'^\d+$')]
    df = df[~df['hedef'].isin({'', 'none', '-', 'nan'})]
    counts = df['hedef'].value_counts()
    return df[df['hedef'].isin(counts[counts >= ESIK].index)].reset_index(drop=True)


def apply_context_window(df, pencere=1):
    """Notebook Cell 12 — birebir ayni mantik."""
    df = df.sort_values(['season', 'episode']).reset_index(drop=True)
    df['text_orijinal'] = df['text'].copy()
    bolum_gruplari = df.groupby(['season', 'episode']).indices
    pos_harita = {}
    for pozlar in bolum_gruplari.values():
        sirali = sorted(pozlar)
        for sira, pos in enumerate(sirali):
            pos_harita[pos] = (sira, sirali)
    sonuc = []
    for pos in range(len(df)):
        sira, bolum_pozlar = pos_harita[pos]
        bas = max(0, sira - pencere)
        bit = min(len(bolum_pozlar), sira + pencere + 1)
        metin = ' '.join(
            str(df.at[p, 'text_orijinal'])
            for p in bolum_pozlar[bas:bit]
            if pd.notna(df.at[p, 'text_orijinal'])
        )
        sonuc.append(metin)
    df['text'] = sonuc
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('CSV yukleniyor...')
    df = load_and_clean(CSV_PATH)
    print(f'Veri: {len(df)} satir, {df["hedef"].nunique()} sinif')

    df = apply_context_window(df, PENCERE)
    print(f'Bagla penceresi +-{PENCERE} uygulandi.')

    y  = np.array(df['hedef'].tolist())
    sw = sample_weight_arr(df)

    # ── Adim 1: HouseMDFeaturizer → TUM VERIYE fit (notebook gibi) ────────────
    print('\nHouseMDFeaturizer tum veriye fit ediliyor...')
    featurizer = HouseMDFeaturizer()
    featurizer.fit(df)
    X_full = featurizer.transform(df)
    print(f'X_full: {X_full.shape}')

    # ── Adim 2: X_full'u split et (notebook Cell 20 ile ayni) ─────────────────
    X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
        X_full, y, sw, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Egitim: {X_train.shape}, Test: {X_test.shape}')

    # ── Adim 3: LinearSVC sadece X_train'e fit (notebook Cell 25) ─────────────
    print('\nLinearSVC egitiliyor (sadece train)...')
    clf_eval = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced', C=1.0)
    clf_eval.fit(X_train, y_train, sample_weight=sw_train)

    y_pred    = clf_eval.predict(X_test)
    test_acc  = (y_pred == y_test).mean()
    train_acc = (clf_eval.predict(X_train) == y_train).mean()
    print(f'  Train Acc : {train_acc:.4f}')
    print(f'  Test Acc  : {test_acc:.4f}  (notebook ile karsilastir)')

    report_df = pd.DataFrame(
        classification_report(y_te=y_test, y_pred=y_pred, output_dict=True, zero_division=0)
        if False else
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).T.round(3)
    labels = sorted(np.unique(np.concatenate([y_train, y_test])))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # ── Adim 4: Deploy icin LinearSVC → TUM VERIYE refit ─────────────────────
    print('\nDeploy icin LinearSVC tum veriye refit ediliyor...')
    clf_deploy = LinearSVC(max_iter=2000, random_state=42, class_weight='balanced', C=1.0)
    clf_deploy.fit(X_full, y, sample_weight=sw)

    # ── Adim 5: Pipeline olarak paketle ve kaydet ──────────────────────────────
    # Pipeline nesnesi: featurizer (tum veriye fit) + clf_deploy (tum veriye fit)
    deploy_pipeline = Pipeline([
        ('features', featurizer),
        ('clf',      clf_deploy),
    ])

    artifact = {
        'pipeline': deploy_pipeline,
        'pencere':  PENCERE,
        'metrics': {
            'accuracy':         float(test_acc),
            'train_accuracy':   float(train_acc),
            'report_df':        report_df,
            'confusion_matrix': cm,
            'cm_labels':        labels,
            'n_train':          X_train.shape[0],
            'n_test':           X_test.shape[0],
            'n_classes':        df['hedef'].nunique(),
        },
    }
    out = os.path.join(MODEL_DIR, 'pipeline.pkl')
    joblib.dump(artifact, out)
    print(f'\nKaydedildi : {out}')
    print(f'Test Acc   : {test_acc:.4f}')


if __name__ == '__main__':
    main()
