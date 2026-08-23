import streamlit as st
import re
import pickle
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY  = "b2cb7ba21343d87c5624b90f86f6fa87"
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="CINEIQ", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0a0a0a; color: #f0f0f0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #141414;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #222;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #888;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: #E50914 !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1a0000 0%, #0a0a0a 50%, #0d0d1a 100%);
        border-bottom: 1px solid #2a2a2a;
        padding: 2.5rem 2rem 2rem 2rem;
        margin: -2rem -6rem 2rem -6rem;
        text-align: center;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #E50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-sub { color: #888; font-size: 0.95rem; margin-top: 0.4rem; }

    /* Search panel */
    .search-panel {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }

    /* Widgets */
    .stSelectbox > div > div {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stMultiSelect > div > div {
        background-color: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #E50914, #c40812) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(229,9,20,0.4) !important;
    }

    /* Section title */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E50914;
        display: inline-block;
    }

    /* Movie card */
    .movie-card {
        background: #141414;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #222;
        transition: all 0.25s ease;
    }
    .movie-card:hover {
        border-color: #E50914;
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(229,9,20,0.2);
    }
    .card-body { padding: 12px; }
    .movie-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #f0f0f0;
        margin: 0 0 8px 0;
        line-height: 1.3;
        min-height: 2.4em;
    }
    .sentiment-bar-bg {
        background: #2a2a2a;
        border-radius: 99px;
        height: 5px;
        margin: 6px 0 10px 0;
        overflow: hidden;
    }
    .sentiment-bar-fill { height: 5px; border-radius: 99px; }
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 99px;
        margin-bottom: 8px;
    }
    .badge-green  { background: rgba(34,197,94,0.15);  color: #22c55e; }
    .badge-yellow { background: rgba(234,179,8,0.15);  color: #eab308; }
    .badge-red    { background: rgba(239,68,68,0.15);  color: #ef4444; }
    .reason-text {
        font-size: 0.72rem;
        color: #777;
        line-height: 1.5;
        border-top: 1px solid #222;
        padding-top: 8px;
        margin-top: 4px;
    }
    .reason-text span { color: #E50914; }
    .poster-wrap { position: relative; overflow: hidden; }
    .rank-badge {
        position: absolute;
        top: 8px; left: 8px;
        background: #E50914;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        width: 24px; height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ctrl-label { font-size: 0.75rem; color: #888; margin-bottom: 4px; }

    /* Stat card */
    .stat-card {
        background: #141414;
        border: 1px solid #222;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #E50914;
        margin: 0;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #666;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────
@st.cache_resource
def load_models():
    content_df = pickle.load(open('models/movies.pkl', 'rb'))
    svd_model  = pickle.load(open('models/svd_model.pkl', 'rb'))
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(content_df['tags']).toarray()
    cosine_sim = cosine_similarity(vectors)
    return content_df, svd_model, cv, cosine_sim

content_df, svd_model, cv, cosine_sim = load_models()

# ── Also load TMDB raw for release years ─────────────────
@st.cache_data
def load_tmdb_years():
    try:
        tmdb = pd.read_csv(r'C:\Users\shain\OneDrive\Desktop\cineiq\data\raw\tmdb_5000_movies.csv')
        tmdb['year'] = pd.to_datetime(tmdb['release_date'], errors='coerce').dt.year
        return tmdb[['id', 'year']].dropna()
    except:
        return pd.DataFrame(columns=['id', 'year'])

tmdb_years = load_tmdb_years()

# ── Helpers ──────────────────────────────────────────────
def get_content_scores(movie_title, n=20):
    idx = content_df[content_df['title'] == movie_title].index[0]
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
        reasons.append(f"<span>{', '.join(list(shared_genres)[:2])}</span> genre match")
    shared_cast = set(inp['cast']) & set(rec['cast'])
    if shared_cast:
        reasons.append(f"Features <span>{list(shared_cast)[0]}</span>")
    shared_crew = set(inp['crew']) & set(rec['crew'])
    if shared_crew:
        reasons.append(f"Directed by <span>{list(shared_crew)[0]}</span>")
    shared_kw = set(inp['keywords']) & set(rec['keywords'])
    if shared_kw:
        reasons.append(f"Theme: <span>{list(shared_kw)[0]}</span>")
    return " · ".join(reasons) if reasons else "Similar overall style and tone"
RUNTIME_MAP = {"No limit": 999, "90 min": 90, "2 hrs": 120, "2.5 hrs": 150}

MOOD_MAP = {
    "Comfort watch":     {"genres": ["Comedy", "Family", "Romance"]},
    "Thrilling":         {"genres": ["Thriller", "Action", "Mystery"]},
    "Thought-provoking": {"genres": ["Drama", "Science Fiction"]},
    "Feel-good":         {"genres": ["Comedy", "Animation", "Adventure"]},
    "Dark/intense":      {"genres": ["Thriller", "Crime", "Horror"]},
    "Romantic/emotional": {"genres": ["Romance", "Drama"]},
}
WEIGHT_MAP = {
    "Light": ["Comedy", "Family", "Adventure", "Animation"],
    "Heavy": ["War", "Drama", "Crime", "Thriller"],
}

def mood_fit_score(rec_genres, mood_choice):
    if mood_choice == "Any":
        return 1.0
    target = set(MOOD_MAP.get(mood_choice, {}).get("genres", []))
    return 1.2 if target & set(rec_genres) else 0.85

def weight_fit_score(rec_genres, weight_choice):
    if weight_choice == "Any":
        return 1.0
    target = set(WEIGHT_MAP.get(weight_choice, []))
    return 1.15 if target & set(rec_genres) else 0.9

def get_runtime(movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                          params={'api_key': API_KEY}, timeout=8).json()
        return r.get('runtime', 999)
    except:
        return 999
def parse_nl_query(text):
    text = text.lower()
    
    # Runtime
    runtime_choice = "No limit"
    match = re.search(r'(\d+(\.\d+)?)\s*(hour|hr)', text)
    if match:
        hrs = float(match.group(1))
        if hrs <= 1.5: runtime_choice = "90 min"
        elif hrs <= 2: runtime_choice = "2 hrs"
        elif hrs <= 2.5: runtime_choice = "2.5 hrs"
    
    # Weight
    light_words = ["light", "funny", "not heavy", "not too emotional", "easy watch", "feel-good", "fun"]
    heavy_words = ["heavy", "intense", "dark", "serious", "emotional", "sad"]
    weight_choice = "Any"
    if any(w in text for w in light_words):
        weight_choice = "Light"
    elif any(w in text for w in heavy_words):
        weight_choice = "Heavy"
    
    # Mood
    mood_keywords = {
        "Comfort watch": ["comfort", "cozy", "familiar", "relax"],
        "Thrilling": ["thrill", "suspense", "action", "exciting"],
        "Thought-provoking": ["thought", "deep", "philosophical", "meaningful"],
        "Feel-good": ["feel-good", "feel good", "uplifting", "happy"],
        "Dark/intense": ["dark", "intense", "disturbing", "gritty"],
    }
    mood_choice = "Any"
    for mood, kws in mood_keywords.items():
        if any(kw in text for kw in kws):
            mood_choice = mood
            break
    
    # Romance/emotional isn't in your current MOOD_MAP genres — add it
    if "romantic" in text or "romance" in text or "emotional" in text:
        mood_choice = "Romantic/emotional"
    
    return runtime_choice, mood_choice, weight_choice        
def generate_full_sentence(input_title, rec_id, mood_choice, weight_choice):
    short = generate_reason(input_title, rec_id).replace("<span>", "").replace("</span>", "")
    parts = [short.lower()]
    if mood_choice != "Any":
        parts.append(f"fits the {mood_choice.lower()} mood you picked")
    if weight_choice != "Any":
        parts.append(f"is a {weight_choice.lower()} watch")
    return f"Recommended because it " + ", and ".join(parts) + "."
def fetch_poster(movie_id):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={'api_key': API_KEY}, headers=headers, timeout=10
        ).json()
        path = r.get('poster_path')
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except:
        return None

def sentiment_badge(sent):
    pct = int(sent * 100)
    if sent > 0.7:
        return f'<span class="badge badge-green">● {pct}% positive</span>', "#22c55e"
    elif sent > 0.4:
        return f'<span class="badge badge-yellow">● {pct}% mixed</span>', "#eab308"
    else:
        return f'<span class="badge badge-red">● {pct}% negative</span>', "#ef4444"

def plotly_dark(fig):
    fig.update_layout(
        paper_bgcolor='#141414',
        plot_bgcolor='#141414',
        font_color='#f0f0f0',
        margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig

# ── Hero ─────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">CINEIQ</p>
    <p class="hero-sub">Tell us a movie you love. We'll find your next obsession.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Recommend", "🎭 My Taste Profile"])

# ════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDER
# ════════════════════════════════════════════════════════
with tab1:
    
    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown('<p class="ctrl-label">🎬 Pick a movie you loved</p>', unsafe_allow_html=True)
        selected_movie = st.selectbox("", content_df['title'].values, label_visibility="collapsed")
    with c2:
        st.markdown('<p class="ctrl-label">&nbsp;</p>', unsafe_allow_html=True)
        search_btn = st.button("Find Movies →", use_container_width=True)
    nl_query = st.text_input("💬 Or just tell us what you're in the mood for", 
                              placeholder="e.g. I have 3 hours and want something romantic and emotional")
    # NEW — mood/situation controls
    m1, m2, m3 = st.columns(3)
    with m1:
        runtime_choice = st.selectbox("⏱ Time you have", ["No limit", "90 min", "2 hrs", "2.5 hrs"])
    with m2:
        mood_choice = st.selectbox("🎭 Mood", ["Any", "Comfort watch", "Thrilling", "Thought-provoking", "Feel-good", "Dark/intense", "Romantic/emotional"])
    with m3:
        weight_choice = st.radio("Light or heavy?", ["Any", "Light", "Heavy"], horizontal=True)
    if nl_query.strip():
        parsed_runtime, parsed_mood, parsed_weight = parse_nl_query(nl_query)
        runtime_choice = parsed_runtime
        mood_choice = parsed_mood
        weight_choice = parsed_weight
    alpha = 0.6
    user_id = 1
    

    st.markdown("""
    <div style="display:flex; gap:1rem; margin:1rem 0 2rem 0; flex-wrap:wrap;">
        <div style="flex:1; min-width:160px; background:#141414; border-radius:12px; padding:1rem 1.2rem; border:1px solid #222;">
            <div style="font-size:1.4rem;">🧠</div>
            <div style="font-size:0.8rem; font-weight:600; color:#f0f0f0; margin:6px 0 4px 0;">Hybrid AI Engine</div>
            <div style="font-size:0.72rem; color:#666;">Content similarity + what users like you actually watched</div>
        </div>
        <div style="flex:1; min-width:160px; background:#141414; border-radius:12px; padding:1rem 1.2rem; border:1px solid #222;">
            <div style="font-size:1.4rem;">💬</div>
            <div style="font-size:0.8rem; font-weight:600; color:#f0f0f0; margin:6px 0 4px 0;">Sentiment Ranked</div>
            <div style="font-size:0.72rem; color:#666;">Re-ranks using real audience review scores — not just algorithms</div>
        </div>
        <div style="flex:1; min-width:160px; background:#141414; border-radius:12px; padding:1rem 1.2rem; border:1px solid #222;">
            <div style="font-size:1.4rem;">💡</div>
            <div style="font-size:0.8rem; font-weight:600; color:#f0f0f0; margin:6px 0 4px 0;">Always Explained</div>
            <div style="font-size:0.72rem; color:#666;">Every pick tells you exactly why it was recommended</div>
        </div>
        <div style="flex:1; min-width:160px; background:#141414; border-radius:12px; padding:1rem 1.2rem; border:1px solid #222;">
            <div style="font-size:1.4rem;">🚫</div>
            <div style="font-size:0.8rem; font-weight:600; color:#f0f0f0; margin:6px 0 4px 0;">No Promoted Titles</div>
            <div style="font-size:0.72rem; color:#666;">100% open ML — no paid placements, no hidden agenda</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if search_btn:
    with st.spinner("Analysing 4,800+ movies for you..."):
        c_scores  = get_content_scores(selected_movie, n=15)
        cf_scores = get_svd_scores(user_id, list(c_scores.keys()))
        beta = 1 - alpha
        hybrid = {mid: alpha*c_scores[mid] + beta*cf_scores.get(mid, 0) for mid in c_scores}

        results = []
        for mid, h_score in hybrid.items():
            row = content_df[content_df['id'] == mid]
            if row.empty:
                continue

            runtime_min = get_runtime(mid)
            if runtime_min > RUNTIME_MAP[runtime_choice]:
                continue  # hard filter

            rec_genres = row.iloc[0]['genres']
            sentiment = get_sentiment(mid)
            final = round(0.8*h_score + 0.2*sentiment, 4)
            final *= mood_fit_score(rec_genres, mood_choice)
            final *= weight_fit_score(rec_genres, weight_choice)

            results.append({
                'title':       row.iloc[0]['title'],
                'movie_id':    mid,
                'runtime':     runtime_min,
                'sentiment':   sentiment,
                'final_score': round(final, 4),
                'reason':      generate_reason(selected_movie, mid)
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        top5 = results[:5]

        st.markdown(f'<p class="section-title">Because you loved &nbsp;<em>{selected_movie}</em></p>',
                    unsafe_allow_html=True)

        cols = st.columns(5, gap="medium")
        for idx, rec in enumerate(top5):
            with cols[idx]:
                poster = fetch_poster(rec['movie_id'])
                badge_html, bar_color = sentiment_badge(rec['sentiment'])
                bar_width = int(rec['sentiment'] * 100)
                poster_html = (
                    f'<img src="{poster}" style="width:100%;border-radius:10px 10px 0 0;">'
                    if poster else
                    '<div style="width:100%;height:220px;background:#1a1a1a;border-radius:10px 10px 0 0;'
                    'display:flex;align-items:center;justify-content:center;color:#444;font-size:2rem;">🎬</div>'
                )
                st.markdown(f"""
                <div class="movie-card">
                    <div class="poster-wrap">
                        {poster_html}
                        <div class="rank-badge">#{idx+1}</div>
                    </div>
                    <div class="card-body">
                        <p class="movie-title">{rec['title']}</p>
                        <p style="font-size:0.72rem;color:#888;margin:0 0 6px 0;">⏱ {rec['runtime']} min</p>
                        {badge_html}
                        <div class="sentiment-bar-bg">
                            <div class="sentiment-bar-fill"
                                 style="width:{bar_width}%;background:{bar_color};"></div>
                        </div>
                        <p class="reason-text">💡 {rec['reason']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Why this?"):
                    st.write(generate_full_sentence(selected_movie, rec['movie_id'], mood_choice, weight_choice))       

# ════════════════════════════════════════════════════════
# TAB 2 — TASTE DASHBOARD
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <p style="color:#888; font-size:0.9rem; margin-bottom:1.5rem;">
    Select movies you've watched and we'll build your personal taste profile.
    </p>
    """, unsafe_allow_html=True)

    # Movie picker
    watched = st.multiselect(
        "🎬 Movies you've watched",
        options=content_df['title'].tolist(),
        default=['Avatar', 'The Dark Knight', 'Inception', 'Interstellar', 'The Avengers'],
        placeholder="Search and add movies..."
    )

    if len(watched) < 3:
        st.info("Add at least 3 movies to generate your taste profile.")
    else:
        watched_df = content_df[content_df['title'].isin(watched)].copy()

        # ── Stats Row ────────────────────────────────────
        all_genres    = [g for genres in watched_df['genres'] for g in genres]
        all_directors = [d for crew in watched_df['crew']    for d in crew]
        all_cast      = [a for cast  in watched_df['cast']   for a in cast]

        top_genre    = Counter(all_genres).most_common(1)[0][0]    if all_genres    else "N/A"
        top_director = Counter(all_directors).most_common(1)[0][0] if all_directors else "N/A"
        top_actor    = Counter(all_cast).most_common(1)[0][0]      if all_cast      else "N/A"

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{len(watched)}</p>
                <p class="stat-label">Movies Tracked</p>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number" style="font-size:1.2rem;">{top_genre}</p>
                <p class="stat-label">Top Genre</p>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number" style="font-size:1rem;">{top_director}</p>
                <p class="stat-label">Favourite Director</p>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number" style="font-size:1rem;">{top_actor}</p>
                <p class="stat-label">Favourite Actor</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Genre Radar + Director Bar ────────────
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown('<p class="section-title">Genre DNA</p>', unsafe_allow_html=True)
            genre_counts = Counter(all_genres).most_common(8)
            if genre_counts:
                labels = [g[0] for g in genre_counts]
                values = [g[1] for g in genre_counts]
                # Close the radar shape
                labels += [labels[0]]
                values += [values[0]]

                fig_radar = go.Figure(go.Scatterpolar(
                    r=values, theta=labels,
                    fill='toself',
                    fillcolor='rgba(229,9,20,0.15)',
                    line=dict(color='#E50914', width=2),
                    marker=dict(color='#E50914', size=6)
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='#1a1a1a',
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values)+1],
                            tickfont=dict(color='#555', size=9),
                            gridcolor='#2a2a2a',
                            linecolor='#2a2a2a'
                        ),
                        angularaxis=dict(
                            tickfont=dict(color='#aaa', size=11),
                            gridcolor='#2a2a2a',
                            linecolor='#2a2a2a'
                        )
                    ),
                    paper_bgcolor='#141414',
                    plot_bgcolor='#141414',
                    font_color='#f0f0f0',
                    margin=dict(t=20, b=20, l=40, r=40),
                    height=360
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_right:
            st.markdown('<p class="section-title">Director Affinity</p>', unsafe_allow_html=True)
            dir_counts = Counter(all_directors).most_common(10)
            if dir_counts:
                dir_df = pd.DataFrame(dir_counts, columns=['Director', 'Movies'])
                fig_dir = px.bar(
                    dir_df, x='Movies', y='Director',
                    orientation='h',
                    color='Movies',
                    color_continuous_scale=[[0, '#3a0000'], [1, '#E50914']],
                    text='Movies'
                )
                fig_dir.update_traces(textposition='outside', textfont_color='#aaa')
                fig_dir.update_layout(
                    paper_bgcolor='#141414',
                    plot_bgcolor='#141414',
                    font_color='#f0f0f0',
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(gridcolor='#222', tickfont=dict(size=11)),
                    margin=dict(t=20, b=20, l=10, r=60),
                    height=360
                )
                st.plotly_chart(fig_dir, use_container_width=True)

        # ── Row 2: Decade Preference + Actor Affinity ────
        col_l2, col_r2 = st.columns(2, gap="large")

        with col_l2:
            st.markdown('<p class="section-title">Decade Preference</p>', unsafe_allow_html=True)
            year_data = tmdb_years[tmdb_years['id'].isin(watched_df['id'].values)]
            if not year_data.empty:
                year_data = year_data.copy()
                year_data['decade'] = (year_data['year'] // 10 * 10).astype(int).astype(str) + 's'
                decade_counts = year_data['decade'].value_counts().sort_index()

                fig_decade = px.bar(
                    x=decade_counts.index,
                    y=decade_counts.values,
                    labels={'x': 'Decade', 'y': 'Movies'},
                    color=decade_counts.values,
                    color_continuous_scale=[[0, '#1a0000'], [1, '#E50914']],
                    text=decade_counts.values
                )
                fig_decade.update_traces(textposition='outside', textfont_color='#aaa')
                fig_decade.update_layout(
                    paper_bgcolor='#141414',
                    plot_bgcolor='#141414',
                    font_color='#f0f0f0',
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor='#222', tickfont=dict(size=12)),
                    yaxis=dict(showgrid=False, visible=False),
                    margin=dict(t=20, b=20, l=10, r=10),
                    height=300
                )
                st.plotly_chart(fig_decade, use_container_width=True)
            else:
                st.caption("Not enough year data for selected movies.")

        with col_r2:
            st.markdown('<p class="section-title">Actor Affinity</p>', unsafe_allow_html=True)
            actor_counts = Counter(all_cast).most_common(10)
            if actor_counts:
                actor_df = pd.DataFrame(actor_counts, columns=['Actor', 'Appearances'])
                fig_actor = px.bar(
                    actor_df, x='Appearances', y='Actor',
                    orientation='h',
                    color='Appearances',
                    color_continuous_scale=[[0, '#001a00'], [1, '#22c55e']],
                    text='Appearances'
                )
                fig_actor.update_traces(textposition='outside', textfont_color='#aaa')
                fig_actor.update_layout(
                    paper_bgcolor='#141414',
                    plot_bgcolor='#141414',
                    font_color='#f0f0f0',
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(gridcolor='#222', tickfont=dict(size=11)),
                    margin=dict(t=20, b=20, l=10, r=60),
                    height=300
                )
                st.plotly_chart(fig_actor, use_container_width=True)

        # ── Taste Summary Card ────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        top3_genres = [g[0] for g in Counter(all_genres).most_common(3)]
        top2_dirs   = [d[0] for d in Counter(all_directors).most_common(2)]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a0000, #141414);
                    border: 1px solid #E50914; border-radius: 16px;
                    padding: 1.5rem 2rem; margin-top: 0.5rem;">
            <p style="font-size:0.75rem; color:#E50914; font-weight:600;
                      letter-spacing:1px; margin:0 0 8px 0;">YOUR TASTE SUMMARY</p>
            <p style="font-size:1.1rem; font-weight:600; color:#f0f0f0; margin:0 0 6px 0;">
                You gravitate toward <span style="color:#E50914;">
                {" · ".join(top3_genres)}</span> films
            </p>
            <p style="font-size:0.85rem; color:#888; margin:0;">
                Biggest influences: <strong style="color:#ccc;">
                {" and ".join(top2_dirs)}</strong>
                {"— you clearly have a director loyalty 🎬" if len(top2_dirs) > 0 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)
