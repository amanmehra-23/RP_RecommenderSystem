# Research Paper Recommender System

A **content‑based** document recommender that uses:

1. **BM25** for fast lexical candidate retrieval  
2. **SBERT** (`all‑mpnet‑base‑v2`) for semantic embeddings  
3. **FAISS** (inner‑product on L2‑normalized vectors) for ANN lookup  
4. A simple **Streamlit** UI to query by terms/title/abstract  

---

## 📂 Project Structure

```
.
├── data/
│   └── arvix_data.csv         # raw CSV with columns: terms, titles, abstracts
├── models/                      # artifacts from GPU‐encode pipeline
│   ├── bm25.pkl                 # pickled BM25 index
│   ├── dataset.pkl              # pickled DataFrame (terms, titles, abstracts)
│   ├── embeddings.npy           # normalized SBERT embeddings
│   └── faiss_index.bin          # FAISS IndexFlatIP
├── notebooks/
│   └── train_embeddings.ipynb   # Jupyter notebook to build indices
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
└── README.md                    # this file
```

---

## 🛠️ Prerequisites

- **macOS** or **Linux**  
- Python 3.11 (on macOS) or 3.10/3.11 (Linux)  
- Xcode Command‑Line Tools (macOS):  
  ```bash
  xcode-select --install
  ```
- (Optional, recommended) [Watchdog](https://pypi.org/project/watchdog/) for Streamlit file watching:
  ```bash
  pip install watchdog
  ```

---

## 🚀 Setup

1. **Clone** this repo:
   ```bash
   git clone https://github.com/yourusername/ResearchPaperRecommenderSystem.git
   cd ResearchPaperRecommenderSystem
   ```

2. **Create & activate** a Python 3.11 venv:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ⚙️ Training / Indexing

### GPU‐accelerated embedding (Colab recommended)

If you have a GPU (e.g. on Colab):

```bash
python train_embeddings_gpu.py
```

This will:

- Build a **BM25** index (`bm25.pkl`)  
- Compute & normalize **SBERT** embeddings (`embeddings.npy`)  
- Build a **FAISS** inner‑product index (`faiss_index.bin`)  
- Save a minimal DataFrame (`dataset.pkl`)

### CPU fallback

If you cannot run GPU code locally, use:

```bash
python train_embeddings.py
```

---

## 📊 Offline Evaluation

To measure P@5, R@5, nDCG@5 on the first 100 docs (using BM25 top‑5 as pseudo ground‑truth):

```bash
python evaluate.py
```

You should see output like:

```
Offline evaluation (first 100 docs):
P@5       0.60
R@5       0.60
nDCG@5    0.90
```

---

## 🌐 Running the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

### Known Issue on macOS

Due to a Streamlit file‑watcher bug crashing on PyTorch’s MPS backend:

- We **disable** the watcher and force **CPU** inference in `app.py`.  
- If you still see a **segmentation fault**, you can:
  1. **Disable dev mode**:  
     ```bash
     streamlit run app.py --global.developmentMode=false
     ```
  2. **Run on Linux** or in a **Docker** container.  
  3. **Use Jupyter** to call `model.encode()` and FAISS directly as a fallback UI.

---

## 🔭 Next Steps

- Fine‑tune a **cross‑encoder** for better ranking.  
- Add **MMR** diversification or a two‑stage BM25 → semantic pipeline.  
- Collect **user feedback** for collaborative filtering.  
- Deploy via **Docker** or **Streamlit Cloud** for production.

---

