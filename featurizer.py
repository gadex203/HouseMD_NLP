"""
HouseMDFeaturizer — notebook ile birebir ayni on isleme adimlarini uygular.
Ayri modul olmasi zorunlu: joblib pkl'den deserialize ederken
featurizer.HouseMDFeaturizer olarak import eder.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# ── Notebook'taki BIREBIR stopwords (Turkce karakterler korundu) ───────────────
TURKCE_STOPWORDS = {
    'bir', 've', 'bu', 'da', 'de', 'ile', 'için', 'mi', 'mu', 'mı', 'mü',
    'ki', 'ne', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
    'ama', 'fakat', 'lakin', 'ancak', 'ya', 'veya', 'yahut', 'hem',
    'çünkü', 'eğer', 'ise', 'gibi', 'kadar', 'daha', 'en',
    'her', 'bazı', 'tüm', 'bütün', 'hangi', 'nasıl', 'neden', 'niçin',
    'şu', 'şey', 'olan', 'olarak', 'üzere', 'göre', 'karşı',
    'yani', 'sadece', 'bile', 'dahi', 'hatta', 'artık', 'zaten',
    'peki', 'evet', 'hayır', 'tamam', 'diye', 'bunu',
    'buna', 'bunun', 'şunu', 'şuna', 'onun', 'beni', 'seni', 'onu',
    'bizi', 'sizi', 'onları', 'bende', 'sende', 'onda',
}

# ── Notebook'taki BIREBIR TYPE_MAP ────────────────────────────────────────────
TYPE_MAP = {
    'Symptom': 'ent_Symptom', 'SYMP': 'ent_Symptom',
    'Test': 'ent_Test', 'TEST': 'ent_Test', 'Test Sonucu': 'ent_Test',
    'Drug': 'ent_Drug', 'DRUG': 'ent_Drug', 'Medication': 'ent_Drug',
    'Procedure': 'ent_Procedure', 'PROC': 'ent_Procedure', 'Treatment': 'ent_Procedure',
    'Anatomy': 'ent_Anatomy', 'Organ': 'ent_Anatomy', 'ORG': 'ent_Anatomy',
}

# ── Notebook'taki BIREBIR ENT_TIBBI_MAP (imputation icin) ─────────────────────
ENT_TIBBI_MAP = {
    'Symptom':   ['Symptom', 'SYMP'],
    'Test':      ['Test', 'TEST', 'Test Sonucu'],
    'Drug':      ['Drug', 'DRUG', 'Medication'],
    'Procedure': ['Procedure', 'PROC', 'Treatment'],
    'Organ':     ['Anatomy', 'Organ', 'ORG'],
}

ENT_COLS       = ['ent_Symptom', 'ent_Test', 'ent_Drug', 'ent_Procedure', 'ent_Anatomy']
ENT_COUNT_COLS = [c.replace('ent_', 'n_') for c in ENT_COLS]
TIBBI          = ['Symptom', 'Test', 'Drug', 'Procedure', 'Organ']
KATEGORIK      = ['Intent', 'diagnosis_stage', 'Emotion', 'speaker']
SAYISAL        = ['Sarcasm', 'season', 'episode']
SARCASM_WEIGHT = 0.3


# ── Notebook fonksiyonlari (birebir kopya) ────────────────────────────────────
def on_isle(metin):
    """Lowercasing + HTML / ozel karakter temizleme. Sayilar KORUNUR."""
    metin = str(metin).lower()
    metin = re.sub(r'<[^>]+>', ' ', metin)
    metin = re.sub(r'[^\w\s]', ' ', metin)
    return re.sub(r'\s+', ' ', metin).strip()


def on_isle_tam(metin):
    """Temizleme + tokenization + stop-words. Stemming yok."""
    metin = on_isle(metin)
    tokens = [t for t in metin.split() if t not in TURKCE_STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


def isle_ve_etiketle(row):
    metin = on_isle_tam(row['text'])
    if str(row.get('Sarcasm', '0')).strip() == '1':
        metin = ' '.join('ALAY_' + t for t in metin.split())
    return metin


def parse_entities(json_str):
    sonuc = {col: [] for col in ENT_COLS}
    try:
        for e in json.loads(str(json_str)):
            if not isinstance(e, dict):
                continue
            tip = TYPE_MAP.get(e.get('type', ''))
            if tip:
                sonuc[tip].append(str(e.get('text', '')).lower())
    except Exception:
        pass
    return {col: ' '.join(v) for col, v in sonuc.items()}


def count_entities(json_str):
    counts = {col: 0 for col in ENT_COLS}
    try:
        for e in json.loads(str(json_str)):
            if not isinstance(e, dict):
                continue
            tip = TYPE_MAP.get(e.get('type', ''))
            if tip:
                counts[tip] += 1
    except Exception:
        pass
    return counts


def impute_from_entities(row):
    """bilgi_yok olan tibbi sutunlari medical_entities JSON'undan doldurur."""
    try:
        entities = json.loads(str(row.get('medical_entities', '[]')))
    except Exception:
        entities = []

    result = {}
    for col, tipler in ENT_TIBBI_MAP.items():
        val = str(row.get(col, 'bilgi_yok'))
        if val == 'bilgi_yok':
            eslesen = [
                e['text'] for e in entities
                if isinstance(e, dict) and e.get('type') in tipler
            ]
            result[col] = ' '.join(eslesen) if eslesen else 'bilgi_yok'
        else:
            result[col] = val
    return result


def sample_weight_arr(df):
    return np.array(
        df['Sarcasm'].apply(
            lambda x: SARCASM_WEIGHT if str(x).strip() == '1' else 1.0
        ).tolist(),
        dtype=float,
    )


# ── HouseMDFeaturizer ─────────────────────────────────────────────────────────
class HouseMDFeaturizer(BaseEstimator, TransformerMixin):
    """
    Ham DataFrame alir, sparse ozellik matrisi dondurur.
    Notebook'taki on isleme adimlarini birebir uygular:
      - on_isle / on_isle_tam / isle_ve_etiketle
      - Entity imputation (bilgi_yok → medical_entities)
      - TF-IDF x3 (metin, tibbi, entity)
      - OneHotEncoder (kategorik)
      - Sayisal + entity sayilari
    """

    def __init__(self, max_text=5000, max_tibbi=3000, max_entity=3000):
        self.max_text   = max_text
        self.max_tibbi  = max_tibbi
        self.max_entity = max_entity

    def fit(self, X: pd.DataFrame, y=None):
        p = self._preprocess(X)
        self.tfidf_text_   = TfidfVectorizer(max_features=self.max_text,   ngram_range=(1, 2))
        self.tfidf_tibbi_  = TfidfVectorizer(max_features=self.max_tibbi,  ngram_range=(1, 2))
        self.tfidf_entity_ = TfidfVectorizer(max_features=self.max_entity, ngram_range=(1, 2))
        self.ohe_          = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

        self.tfidf_text_.fit(p['text_clean'])
        self.tfidf_tibbi_.fit(p['tibbi_metin'])
        self.tfidf_entity_.fit(p['ent_all'])
        self.ohe_.fit(p['kat_vals'])
        return self

    def transform(self, X: pd.DataFrame):
        p = self._preprocess(X)

        X_text   = self.tfidf_text_.transform(p['text_clean'])
        X_tibbi  = self.tfidf_tibbi_.transform(p['tibbi_metin'])
        X_entity = self.tfidf_entity_.transform(p['ent_all'])
        X_kat    = self.ohe_.transform(p['kat_vals'])
        X_say    = csr_matrix(p['say_vals'])
        X_ent    = csr_matrix(p['ent_cnt_vals'])

        # Notebook'taki birebir hstack sirasi:
        # X_text, X_tibbi, X_entity, X_kat, X_say, X_ent_counts
        return hstack([X_text, X_tibbi, X_entity, X_kat, X_say, X_ent])

    def _preprocess(self, X: pd.DataFrame) -> dict:
        """Tum on isleme — hem fit hem transform cagirabilir."""
        X = X.copy()

        # 1. Entity imputation (bilgi_yok → medical_entities)
        imputed = X.apply(impute_from_entities, axis=1).apply(pd.Series)
        for col in TIBBI:
            X[col] = imputed[col]

        # 2. Metin temizleme + sarcasm prefix
        text_clean = X.apply(isle_ve_etiketle, axis=1).tolist()

        # 3. Tibbi metin
        tibbi_metin = (
            X[TIBBI].fillna('bilgi_yok')
            .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            .apply(on_isle_tam)
            .tolist()
        )

        # 4. Entity parse
        ent_dicts = X['medical_entities'].fillna('[]').apply(parse_entities).apply(pd.Series)
        ent_all = (
            ent_dicts.fillna('')
            .apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            .apply(on_isle)
            .tolist()
        )

        # 5. Kategorik
        kat_df = X[KATEGORIK].fillna('bilinmiyor').astype(str)
        kat_vals = kat_df.values

        # 6. Sayisal
        say_vals = (
            X[SAYISAL].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).values
        )

        # 7. Entity sayilari
        cnt_rows = X['medical_entities'].fillna('[]').apply(count_entities).apply(pd.Series)
        cnt_arr = np.zeros((len(X), len(ENT_COLS)), dtype=float)
        for j, col in enumerate(ENT_COLS):
            if col in cnt_rows.columns:
                cnt_arr[:, j] = cnt_rows[col].values
        totals = cnt_arr.sum(axis=1, keepdims=True)
        ent_cnt_vals = np.hstack([cnt_arr, totals])

        return {
            'text_clean':   text_clean,
            'tibbi_metin':  tibbi_metin,
            'ent_all':      ent_all,
            'kat_vals':     kat_vals,
            'say_vals':     say_vals,
            'ent_cnt_vals': ent_cnt_vals,
        }
