import os
import time
import requests
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Contract Clause Risk Classifier",
    page_icon="https://cdn-icons-png.flaticon.com/512/2586/2586948.png",
    layout="wide"
)

st.title("📝 Construction Contract Clause Risk Classifier")

st.markdown(
    "This application automatically analyzes construction contract clauses and "
    "identifies potential risk-related provisions using a soft voting ensemble "
    "of machine learning models."
)

st.markdown(
    """
    **How it works**
    1. Upload an Excel, CSV, or TSV file containing contract clauses  
    2. Select the text column to be analyzed  
    3. Run the classification to identify potential risk clauses  
    4. Review and download the annotated results
    """
)

st.divider()

# ------------------------------------------------------------
# Model paths and download URLs
# ------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "voting_model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

MODEL_URL = "https://drive.google.com/uc?id=1NEL16kC-510kwXh7TNXwj-vuI1rxtVBY"
VECT_URL  = "https://drive.google.com/uc?id=1YP30UYdR2YNt75FqJXeKnLPhKw55SbP2"

# ------------------------------------------------------------
# Utility function
# ------------------------------------------------------------
def download_file(url, dest):
    with st.spinner(f"Downloading {os.path.basename(dest)}..."):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# ------------------------------------------------------------
# Load model artifacts (cached)
# ------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    st.info("Model file not found. Downloading...")
    download_file(MODEL_URL, MODEL_PATH)

if not os.path.exists(VECT_PATH):
    st.info("Vectorizer file not found. Downloading...")
    download_file(VECT_URL, VECT_PATH)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ------------------------------------------------------------
# File uploader
# ------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload an Excel, CSV, or TSV file",
    type=["xlsx", "csv", "tsv"]
)

# ------------------------------------------------------------
# File processing
# ------------------------------------------------------------
if uploaded is not None:
    filename = uploaded.name.lower()

    if filename.endswith(".tsv"):
        df = pd.read_csv(uploaded, sep="\t")
    elif filename.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head(50), width="stretch")

    default_index = (
        list(df.columns).index("clause_text")
        if "clause_text" in df.columns
        else 0
    )

    text_col = st.selectbox(
        "Select the text column to classify",
        df.columns,
        index=default_index
    )

    st.subheader("Classification")
    st.info("Prediction Encoding [ 0 = risk-free | 1 = potential-risk ]")


    # --------------------------------------------------------
    # Run Classification
    # --------------------------------------------------------
    run_clicked = st.button("🔍 Run Classification")

    if run_clicked:
        with st.spinner("Analyzing contract clauses, please wait..."):

            start_time = time.time()

            texts = df[text_col].astype(str).fillna("").tolist()

            X = vectorizer.transform(texts).toarray()

            predictions = model.predict(X)

            elapsed_time = time.time() - start_time

        # Labels (0 = risk-free, 1 = potential-risk)
        df["predicted"] = predictions
        risk_count = (df["predicted"] == 1).sum()


        # Probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            classes = list(model.classes_)

            for i, cls in enumerate(classes):
                df[f"prob_{cls}"] = probabilities[:, i]

            if 1 in classes:
                df["risk_score"] = probabilities[:, classes.index(1)]
                df = df.sort_values("risk_score", ascending=False)

        num_clauses = len(texts)
        avg_time = elapsed_time / num_clauses if num_clauses > 0 else 0

        st.success(
            f"✅ {num_clauses:,} clauses were analyzed in "
            f"{elapsed_time:.2f} seconds."
        )

        st.caption(
            f"Detected {risk_count:,} potential-risk clauses. "
            f"Average processing time per clause: {avg_time:.4f} seconds."
        )

        st.subheader("✅ Classification Results")
        st.dataframe(df, width="stretch")

        # Download
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            "📥 Download results as Excel",
            data=buffer,
            file_name="classified_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("Please upload a data file to start the analysis.")
