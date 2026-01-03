import os
import requests
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Contract Clause Risk Classifier", layout="wide")
st.title("📄 Construction Contract Clause Risk Classifier")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "voting_model.pkl")
VECT_PATH  = os.path.join(MODEL_DIR, "vectorizer.pkl")

MODEL_URL = "https://drive.google.com/uc?id=1NEL16kC-510kwXh7TNXwj-vuI1rxtVBY"
VECT_URL  = "https://drive.google.com/uc?id=1YP30UYdR2YNt75FqJXeKnLPhKw55SbP2"

def download_file(url, dest):
    with st.spinner(f"Downloading {os.path.basename(dest)}..."):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.info("Model bulunamadı. İndiriliyor...")
    download_file(MODEL_URL, MODEL_PATH)

if not os.path.exists(VECT_PATH):
    st.info("Vectorizer bulunamadı. İndiriliyor...")
    download_file(VECT_URL, VECT_PATH)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)


uploaded = st.file_uploader(
    "Excel, CSV veya TSV dosyası yükle",
    type=["xlsx", "csv", "tsv"])

if uploaded is not None:
    filename = uploaded.name.lower()

    if filename.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t")
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    st.subheader("📄 Yüklenen veri")
    st.dataframe(df, use_container_width=True)

    text_col = st.selectbox("Metin sütununu seç", df.columns)

    label_mode = st.selectbox("Çıktı formatı", ["0/1", "label (potential-risk / other)"])

    if st.button("🔍 Sınıflandır"):
        texts = df[text_col].astype(str).fillna("").tolist()

        X = vectorizer.transform(texts)
        X_dense = X.toarray()  

        preds = model.predict(X_dense)

        if label_mode.startswith("label"):
            df["predicted"] = np.where(preds == 1, "potential-risk", "other")
        else:
            df["predicted"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_dense)
            classes = list(model.classes_)

            for i, cls in enumerate(classes):
                df[f"prob_{cls}"] = proba[:, i]

            if 1 in classes:
                df["risk_score"] = proba[:, classes.index(1)]

        st.subheader("✅ Sonuçlar")
        st.dataframe(df, use_container_width=True)

        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            "📥 Sonuçları Excel olarak indir",
            data=buffer,
            file_name="classified_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
