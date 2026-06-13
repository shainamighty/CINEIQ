# 🎬 CINEIQ — Open Explainable Movie Recommendation Engine

> *Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. CINEIQ fixes that.*

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-3.x-blue?style=flat-square&logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 What is CINEIQ?

CINEIQ is a **hybrid ML movie recommendation engine** that combines three recommendation strategies, re-ranks results using real audience sentiment, and explains every single recommendation in plain English — with zero paid promotions.

Unlike Netflix or Prime Video, CINEIQ tells you **why** it recommended something.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔀 **Hybrid Engine** | Combines content-based filtering + collaborative filtering (SVD) into a weighted ensemble |
| 💬 **Sentiment Re-Ranking** | Uses VADER on TMDB reviews to boost genuinely loved movies and suppress poorly received ones |
| 💡 **Explainability Layer** | Every recommendation shows: genre match, shared cast, director, and theme |
| 🎭 **Taste Dashboard** | Genre radar chart, director affinity, decade preference, actor affinity — all from your watch history |
| 📊 **MLflow Tracking** | All SVD experiments logged with RMSE/MAE metrics for reproducibility |
| 🚫 **No Promoted Titles** | 100% open ML — no paid placements, no hidden agenda |

---

## 🏗️ System Architecture

```
User Input (movie title)
        ↓
┌───────────────────────────────────┐
│         HYBRID ENGINE             │
│                                   │
│  Content-Based    Collaborative   │
│  TF-IDF+Cosine    Surprise SVD    │
│  (TMDB metadata)  (MovieLens 25M) │
│         ↘              ↙          │
│      Weighted Ensemble            │
│   score = 0.6×content + 0.4×collab│
└───────────────┬───────────────────┘
                ↓
    Sentiment Re-Ranker (VADER)
                ↓
    Explainability Layer
                ↓
    Streamlit Dashboard
```

---

## 📦 Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| MovieLens 25M | GroupLens | 25M ratings | Collaborative filtering |
| TMDB 5000 Movies | Kaggle | 4,803 movies | Content-based filtering |
| TMDB 5000 Credits | Kaggle | 4,803 movies | Cast & crew extraction |
| IMDB 50K Reviews | Kaggle | 50,000 reviews | Sentiment model training |

---

## 🛠️ Tech Stack

```
ML Pipeline
├── scikit-learn      → TF-IDF vectorization, cosine similarity
├── Surprise (SVD)    → Collaborative filtering, matrix factorization
├── VADER Sentiment   → Review scoring and re-ranking
├── NLTK              → Porter stemming
└── Pandas / NumPy    → Data processing

Frontend
└── Streamlit + Plotly → Dashboard, radar charts, bar charts

Experiment Tracking
└── MLflow            → SVD hyperparameter experiments

Data
└── Pickle            → Serialized models and dataframes
```

---

## 📊 Model Performance

| Experiment | n_factors | n_epochs | lr | RMSE | MAE |
|---|---|---|---|---|---|
| SVD_baseline | 100 | 20 | 0.005 | **0.9186** | 0.7072 |
| SVD_more_factors | 150 | 25 | 0.005 | 0.9269 | 0.7134 |
| SVD_high_lr | 100 | 20 | 0.010 | 0.9335 | 0.7198 |

> Baseline configuration wins. Tracked via MLflow at `localhost:5000`.

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/cineiq.git
cd cineiq
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) → `data/raw/ratings.csv`, `data/raw/ml_movies.csv`
- [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) → `data/raw/tmdb_5000_movies.csv`, `data/raw/tmdb_5000_credits.csv`

### 5. Run data prep notebook
```
notebooks/01_data_prep.ipynb  → Run all cells
```

### 6. Get TMDB API key
Sign up at [themoviedb.org](https://www.themoviedb.org/settings/api) and paste your key in `main.py`:
```python
API_KEY = "your_key_here"
```

### 7. Launch the app
```bash
.venv\Scripts\python.exe -m streamlit run main.py
```

Open **http://localhost:8501**

---

## 📁 Project Structure

```
CINEIQ/
│
├── main.py                    ← Streamlit app (UI + all logic)
│
├── notebooks/
│   └── 01_data_prep.ipynb     ← Data prep, SVD training, MLflow experiments
│
├── models/
│   ├── movies.pkl             ← Processed content dataframe
│   ├── similarity.pkl         ← Cosine similarity matrix
│   ├── svd_model.pkl          ← Trained SVD model
│   └── vectorizer.pkl         ← TF-IDF vectorizer
│
├── data/
│   ├── raw/                   ← Original CSVs (not tracked by git)
│   └── processed/             ← Cleaned pickle files
│
├── requirements.txt
└── README.md
```

---

## 💡 How the Recommendation Works

**Step 1 — Content Score**
Each movie is represented as a bag of stemmed words from its overview, genres, cast, crew, and keywords. TF-IDF vectorizes these into 5000-dimensional vectors. Cosine similarity finds the closest movies.

**Step 2 — Collaborative Score**
SVD matrix factorization decomposes the 25M rating matrix to find latent user-movie patterns. It predicts what rating a user would give to unseen movies.

**Step 3 — Hybrid Ensemble**
```
final_score = 0.6 × content_score + 0.4 × collab_score
```

**Step 4 — Sentiment Re-Ranking**
TMDB reviews are fetched and scored with VADER. The hybrid score is adjusted:
```
final_score = 0.8 × hybrid_score + 0.2 × sentiment_score
```

**Step 5 — Explainability**
Shared genres, cast, director, and keywords between input and recommended movie are extracted and formatted into a human-readable reason string.

---

## 🎭 Taste Dashboard

Select movies you've watched and CINEIQ builds your personal taste profile:
- **Genre DNA** — Radar chart of your genre distribution
- **Director Affinity** — Bar chart of directors you watch most
- **Decade Preference** — Which era of cinema you gravitate toward
- **Actor Affinity** — Most watched actors across your history
- **Taste Summary** — One-line description of your cinema personality

---

## 🔮 Future Roadmap

- [ ] FastAPI backend (`/recommend` and `/similar` endpoints)
- [ ] DistilBERT sentiment upgrade (more accurate than VADER)
- [ ] LIME explainability (model-agnostic feature importance)
- [ ] User login + persistent watchlist
- [ ] Deployment on Render / HuggingFace Spaces

---

## 👤 Author

**Your Name**
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — free to use, modify, and distribute.