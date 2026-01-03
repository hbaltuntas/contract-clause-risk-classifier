import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Contract Clause Risk Classifier", layout="wide")
st.title("📄 Construction Contract Clause Risk Classifier")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/voting_model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

uploaded = st.file_uploader("Excel dosyası yükle (.xlsx)", type=["xlsx"])

if uploaded is not None:
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
