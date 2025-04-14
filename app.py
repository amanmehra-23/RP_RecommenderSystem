# app.py

import os

# (1) Turn off Streamlit's file‑watcher before anything else loads
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle

MODELS_DIR = "models"  # or "models_gpu"

# ─── Heavy resources, loaded only once on demand ─────────────────────
@st.cache_resource
def load_resources():
    # Defer the SBERT (and thus torch) import until now
    from sentence_transformers import SentenceTransformer

    # 1) DataFrame
    df = pd.read_pickle(f"{MODELS_DIR}/dataset.pkl")
    # 2) Embeddings
    embeddings = np.load(f"{MODELS_DIR}/embeddings.npy")
    # 3) FAISS index
    index = faiss.read_index(f"{MODELS_DIR}/faiss_index.bin")
    # 4) SBERT model on CPU
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return df, embeddings, index, model

# ─── Simple FAISS search, cached by Streamlit ────────────────────────
@st.cache_data
def run_faiss_search(q_emb: np.ndarray, k: int):
    D, I = index.search(q_emb.astype(np.float32), k)
    return D[0], I[0]

# ─── Main app ───────────────────────────────────────────────────────
df, embeddings, index, model = load_resources()

st.title("🔍 Document Recommender")

st.sidebar.header("Query Document")
terms_in    = st.sidebar.text_input("Terms (comma‑separated)")
title_in    = st.sidebar.text_input("Title")
abstract_in = st.sidebar.text_area("Abstract")
k           = st.sidebar.slider("Number of recommendations", 1, 20, 5)

if st.sidebar.button("Recommend"):
    if not (terms_in or title_in or abstract_in):
        st.error("Please enter at least one field.")
    else:
        # Build the query string
        combo = "  ||  ".join([
            terms_in.replace(",", " "),
            title_in,
            abstract_in
        ]).strip()

        # Embed & normalize
        q_emb = model.encode([combo], convert_to_tensor=False)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # FAISS lookup
        scores, ids = run_faiss_search(q_emb, k)

        # Prepare results
        recs = []
        for score, idx in zip(scores, ids):
            recs.append({
                "Score": float(score),
                "Terms": df.at[idx, "terms"],
                "Title": df.at[idx, "titles"],
                "Abstract": df.at[idx, "abstracts"]
            })

        st.write("## Top Recommendations")
        st.dataframe(pd.DataFrame(recs), use_container_width=True)
