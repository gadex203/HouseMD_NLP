"""
House MD Teşhis Sınıflandırıcı — Streamlit Uygulaması.
Tüm ön işleme Pipeline içinde — burada hiç preprocessing yok.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from featurizer import HouseMDFeaturizer  # joblib deserialize icin gerekli

MODEL_PATH = os.path.join("model", "pipeline.pkl")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def predict(pipeline, inputs: dict):
    # Ham girdiyi tek satırlık DataFrame'e çevir
    row = {
        'text':               inputs.get('text', ''),
        'Sarcasm':            '1' if inputs['sarcasm'] else '0',
        'Symptom':            inputs.get('symptom', ''),
        'Test':               inputs.get('test', ''),
        'Drug':               inputs.get('drug', ''),
        'Procedure':          inputs.get('procedure', ''),
        'Organ':              inputs.get('organ', ''),
        'medical_entities':   inputs.get('medical_entities', '[]'),
        'Intent':             inputs.get('intent', 'bilinmiyor'),
        'diagnosis_stage':    inputs.get('diagnosis_stage', 'bilinmiyor'),
        'Emotion':            inputs.get('emotion', 'bilinmiyor'),
        'speaker':            inputs.get('speaker', 'bilinmiyor'),
        'season':             inputs.get('season', 0),
        'episode':            inputs.get('episode', 0),
    }
    df = pd.DataFrame([row])

    pred = pipeline.predict(df)[0]

    # Karar fonksiyonu → top-5 sınıf
    decision = pipeline.decision_function(df)[0]
    classes = pipeline.classes_
    top_idx = np.argsort(decision)[::-1][:5]
    top5 = [(classes[i], float(decision[i])) for i in top_idx]

    return pred, top5


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="House MD Teşhis",
        page_icon="🏥",
        layout="wide",
    )
    st.title("🏥 House MD — Teşhis Sınıflandırıcı")
    st.caption("LinearSVC · Türkçe NLP")

    if not os.path.exists(MODEL_PATH):
        st.error("Model bulunamadı. `python train_model.py` çalıştırın.")
        st.stop()

    artifact = load_artifact()
    pipeline = artifact['pipeline']

    tab_tahmin, tab_perf = st.tabs(["Tahmin", "Model Performansı"])

    # ── Performans ────────────────────────────────────────────────────────────
    with tab_perf:
        m = artifact.get('metrics', {})
        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{m.get('accuracy', 0):.1%}")
        c2.metric("Sınıf Sayısı", m.get('n_classes', '—'))

        st.subheader("Sınıf Bazlı Metrikler")
        report = m.get('report_df')
        if report is not None:
            summary_rows = ['accuracy', 'macro avg', 'weighted avg']
            class_df = report.drop(
                index=[r for r in summary_rows if r in report.index], errors='ignore'
            )[['precision', 'recall', 'f1-score', 'support']].copy()
            class_df.index.name = 'Hastalık'

            def color_f1(val):
                if not isinstance(val, float):
                    return ''
                if val < 0.40:
                    return 'background-color: #ffcccc'
                if val < 0.60:
                    return 'background-color: #fff3cc'
                return 'background-color: #ccffcc'

            st.dataframe(
                class_df.style.map(color_f1, subset=['f1-score']),
                use_container_width=True,
                height=500,
            )
            with st.expander("Macro / Weighted Ortalamalar"):
                summary = report.loc[[r for r in summary_rows if r in report.index]]
                st.dataframe(summary[['precision', 'recall', 'f1-score']], use_container_width=True)

        st.caption("Metrikler yalnızca test seti (%20) üzerinden hesaplanmıştır.")

    # ── Tahmin ────────────────────────────────────────────────────────────────
    with tab_tahmin:
        with st.form("tahmin_formu"):
            st.subheader("Diyalog Metni")
            text = st.text_area(
                "Hasta / Doktor diyaloğu *",
                height=150,
                placeholder="Örn: Hastanın karnında şiddetli ağrı var, ateşi 39 derece...",
            )
            sarcasm = st.checkbox("Alaycı/ironik ifade (Sarcasm)")

            st.subheader("Tıbbi Bilgiler (opsiyonel)")
            c1, c2, c3 = st.columns(3)
            with c1:
                symptom = st.text_input("Semptomlar", placeholder="ateş, öksürük")
                test_val = st.text_input("Testler", placeholder="kan testi, MRI")
            with c2:
                drug = st.text_input("İlaçlar", placeholder="aspirin, metformin")
                procedure = st.text_input("Prosedürler", placeholder="biyopsi")
            with c3:
                organ = st.text_input("Organlar", placeholder="karaciğer, akciğer")
                medical_entities = st.text_area(
                    "medical_entities (JSON)",
                    height=80,
                    placeholder='[{"type":"Symptom","text":"ateş"}]',
                )

            st.subheader("Bağlam Bilgileri (opsiyonel)")
            c4, c5, c6, c7 = st.columns(4)
            with c4:
                intent = st.text_input("Intent", value="bilinmiyor")
            with c5:
                diagnosis_stage = st.text_input("Diagnosis Stage", value="bilinmiyor")
            with c6:
                emotion = st.text_input("Emotion", value="bilinmiyor")
            with c7:
                speaker = st.text_input("Speaker", value="bilinmiyor")

            c8, c9 = st.columns(2)
            with c8:
                season = st.number_input("Sezon", min_value=0, max_value=10, value=1)
            with c9:
                episode = st.number_input("Bölüm", min_value=0, max_value=30, value=1)

            submitted = st.form_submit_button("Tahmin Et", use_container_width=True)

        if submitted:
            if not text.strip():
                st.warning("Lütfen diyalog metni girin.")
                st.stop()

            inputs = {
                'text': text, 'sarcasm': sarcasm,
                'symptom': symptom, 'test': test_val,
                'drug': drug, 'procedure': procedure, 'organ': organ,
                'medical_entities': medical_entities or '[]',
                'intent': intent, 'diagnosis_stage': diagnosis_stage,
                'emotion': emotion, 'speaker': speaker,
                'season': season, 'episode': episode,
            }

            with st.spinner("Tahmin yapılıyor..."):
                pred, top5 = predict(pipeline, inputs)

            st.divider()
            st.subheader("Tahmin Sonucu")
            st.success(f"**Teşhis:** {pred.upper()}", icon="✅")

            st.subheader("En Yakın 5 Sınıf")
            scores = [s for _, s in top5]
            lo, hi = min(scores), max(scores)
            score_range = hi - lo if hi != lo else 1.0
            for i, (sinif, skor) in enumerate(top5):
                cl, cr = st.columns([3, 7])
                with cl:
                    st.write(f"{'**' if i == 0 else ''}{i+1}. {sinif}{'**' if i == 0 else ''}")
                with cr:
                    norm = (skor - lo) / score_range  # 0.0–1.0, her zaman geçerli
                    st.progress(norm, text=f"{skor:.3f}")


if __name__ == '__main__':
    main()
