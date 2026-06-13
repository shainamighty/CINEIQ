from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# ── Config ───────────────────────────────────────────────
API_KEY  = "b2cb7ba21343d87c5624b90f86f6fa87"
analyzer = SentimentIntensityAnalyzer()
app      = FastAPI(title="CINEIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Models ──────────────────────────────────────────
print("Loading models...")
content_df = pickle.load(open('models/movies.pkl', 'rb'))
svd_model  = pickle.load(open('models/svd_model.pkl', 'rb'))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors    = cv.fit_transform(content_df['tags']).toarray()
cosine_sim = cosine_similarity(vectors)
print("Models loaded!")

# ── Helpers ──────────────────────────────────────────────
def get_content_scores(movie_title, n=20):
    matches = content_df[content_df['title'] == movie_title]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
    idx = matches.index[0]
    distances = cosine_sim[idx]
    ranked = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:n+1]
    return {content_df.iloc[i]['id']: score for i, score in ranked}

def get_svd_scores(user_id, candidate_ids):
    return {mid: svd_model.predict(user_id, mid).est / 5.0 for mid in candidate_ids}

def get_sentiment(movie_id):
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/reviews",
            params={'api_key': API_KEY}, timeout=8
        ).json()
        reviews = [x['content'] for x in r.get('results', [])]
        if not reviews:
            return 0.5
        scores = [analyzer.polarity_scores(rv)['compound'] for rv in reviews]
        return round((sum(scores)/len(scores) + 1) / 2, 3)
    except:
        return 0.5

def generate_reason(input_title, rec_id):
    inp = content_df[content_df['title'] == input_title].iloc[0]
    rec = content_df[content_df['id'] == rec_id]
    if rec.empty:
        return "Similar overall style and tone"
    rec = rec.iloc[0]
    reasons = []
    shared_genres = set(inp['genres']) & set(rec['genres'])
    if shared_genres:
        reasons.append(f"{', '.join(list(shared_genres)[:2])} genre match")
    shared_cast = set(inp['cast']) & set(rec['cast'])
    if shared_cast:
        reasons.append(f"Features {list(shared_cast)[0]}")
    shared_crew = set(inp['crew']) & set(rec['crew'])
    if shared_crew:
        reasons.append(f"Directed by {list(shared_crew)[0]}")
    shared_kw = set(inp['keywords']) & set(rec['keywords'])
    if shared_kw:
        reasons.append(f"Theme: {list(shared_kw)[0]}")
    return " · ".join(reasons) if reasons else "Similar overall style and tone"

# ── Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "CINEIQ API",
        "version": "1.0.0",
        "endpoints": {
            "recommend": "/recommend?movie=Avatar&n=5&user_id=1",
            "similar":   "/similar?movie=Avatar&n=10",
            "movies":    "/movies?search=dark",
            "health":    "/health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": len(content_df),
        "model": "SVD + TF-IDF Hybrid"
    }

@app.get("/movies")
def search_movies(search: str = "", limit: int = 20):
    """Search available movies by title"""
    if search:
        results = content_df[
            content_df['title'].str.contains(search, case=False, na=False)
        ]['title'].tolist()[:limit]
    else:
        results = content_df['title'].tolist()[:limit]
    return {"results": results, "count": len(results)}

@app.get("/recommend")
def recommend(movie: str, n: int = 5, user_id: int = 1):
    """
    Hybrid recommendation endpoint.
    Combines content-based + collaborative filtering + sentiment re-ranking.
    """
    # Hybrid scores
    c_scores  = get_content_scores(movie, n=20)
    cf_scores = get_svd_scores(user_id, list(c_scores.keys()))
    hybrid    = {mid: 0.6*c_scores[mid] + 0.4*cf_scores.get(mid, 0)
                 for mid in c_scores}

    # Build results
    results = []
    for mid, h_score in hybrid.items():
        row = content_df[content_df['id'] == mid]
        if row.empty:
            continue
        sentiment  = get_sentiment(mid)
        final      = round(0.8*h_score + 0.2*sentiment, 4)
        results.append({
            "title":          row.iloc[0]['title'],
            "movie_id":       int(mid),
            "hybrid_score":   round(h_score, 4),
            "sentiment_score": sentiment,
            "final_score":    final,
            "reason":         generate_reason(movie, mid),
            "genres":         row.iloc[0]['genres']
        })

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return {
        "query":        movie,
        "user_id":      user_id,
        "total":        len(results[:n]),
        "recommendations": results[:n]
    }

@app.get("/similar")
def similar(movie: str, n: int = 10):
    """
    Content-only similarity endpoint.
    Returns movies most similar by genre, cast, keywords.
    """
    c_scores = get_content_scores(movie, n=n+1)

    results = []
    for mid, score in c_scores.items():
        row = content_df[content_df['id'] == mid]
        if row.empty:
            continue
        results.append({
            "title":            row.iloc[0]['title'],
            "movie_id":         int(mid),
            "similarity_score": round(score, 4),
            "genres":           row.iloc[0]['genres'],
            "reason":           generate_reason(movie, mid)
        })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return {
        "query": movie,
        "total": len(results[:n]),
        "similar": results[:n]
    }