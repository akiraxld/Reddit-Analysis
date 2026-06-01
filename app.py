# ── IMPORTS ────────────────────────────────────────────────────────────────────
import os, re, json, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import urllib.parse

import nltk
import torch
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from nrclex import NRCLex
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
import hdbscan
import umap.umap_ as umap

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ── NLTK ───────────────────────────────────────────────────────────────────────
for corpus in ('wordnet', 'punkt', 'omw-1.4'):
    try:
        nltk.data.find(f'corpora/{corpus}')
    except LookupError:
        nltk.download(corpus, quiet=True)

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reddit Thread Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN TOKENS ──────────────────────────────────────────────────────────────
BG          = "#050000"
BG_CARD     = "#100404"
BG_CARD2    = "#1a0505"
BORDER      = "#3d0f0f"
BORDER2     = "#5a1515"
RED         = "#c0392b"
RED_BRIGHT  = "#e74c3c"
RED_PALE    = "#f1948a"
RED_DEEP    = "#7b241c"
RED_DARKER  = "#641e16"
TEXT        = "#f0e0e0"
MUTED       = "#9a7070"
MUTED2      = "#c0a0a0"

PALETTE = [
    RED_BRIGHT, '#e67e22', RED, '#d35400',
    RED_DEEP, RED_PALE, RED_DARKER, '#cd6155',
    '#a93226', '#f0b27a',
]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  .stApp {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'DM Sans', sans-serif;
  }}

  /* ─── typography ─── */
  h1,h2,h3,h4 {{ font-family:'DM Serif Display', Georgia, serif; }}

  .main-title {{
    font-family:'DM Serif Display', Georgia, serif;
    font-size: 84px;
    line-height: 0.9;
    letter-spacing: -3px;
    background: linear-gradient(140deg, {RED_BRIGHT} 0%, {RED} 45%, {RED_DARKER} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
  }}

  .subtitle {{
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 40px;
  }}

  .badge {{
    display: inline-block;
    border: 1px solid {BORDER2};
    border-radius: 40px;
    padding: 3px 14px;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {RED_PALE};
    background: rgba(192,57,43,0.08);
    margin-bottom: 20px;
  }}

  /* ─── cards ─── */
  .card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
  }}

  .metric-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {RED};
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
  }}

  .metric-label {{
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 4px;
  }}

  .metric-value {{
    font-size: 28px;
    font-weight: 700;
    color: {RED_BRIGHT};
    font-family: 'DM Serif Display', serif;
  }}

  /* ─── inputs ─── */
  .stTextInput input {{
    background: {BG_CARD} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s;
  }}
  .stTextInput input:focus {{
    border-color: {RED} !important;
    box-shadow: 0 0 0 2px rgba(192,57,43,0.2) !important;
  }}

  /* ─── buttons ─── */
  div.stButton > button {{
    background: linear-gradient(135deg, {RED} 0%, {RED_DEEP} 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s ease;
    cursor: pointer;
  }}
  div.stButton > button:hover {{
    background: linear-gradient(135deg, {RED_BRIGHT} 0%, {RED} 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(192,57,43,0.35);
  }}
  div.stButton > button:active {{ transform: translateY(0); }}

  /* ─── tabs ─── */
  .stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD};
    border-bottom: 1px solid {BORDER};
    gap: 0;
    padding: 0 8px;
    border-radius: 10px 10px 0 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {MUTED};
    padding: 10px 20px;
    font-size: 13px;
    font-family: 'DM Sans', sans-serif;
    border-bottom: 2px solid transparent;
    background: transparent;
  }}
  .stTabs [aria-selected="true"] {{
    color: {RED_BRIGHT} !important;
    border-bottom: 2px solid {RED} !important;
    background: transparent !important;
  }}

  /* ─── expander ─── */
  details summary {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif;
  }}
  details[open] summary {{ border-radius: 8px 8px 0 0 !important; }}
  details > div {{ border: 1px solid {BORDER}; border-top: none; border-radius: 0 0 8px 8px; background: {BG_CARD2}; }}

  /* ─── metrics ─── */
  [data-testid="stMetricValue"] {{
    color: {RED_BRIGHT} !important;
    font-size: 26px !important;
    font-family: 'DM Serif Display', serif;
  }}
  [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 11px !important; }}

  /* ─── misc ─── */
  hr {{ border-color: {BORDER} !important; margin: 20px 0 !important; }}

  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {RED}; }}

  /* ─── info/warning/error boxes ─── */
  [data-testid="stAlert"] {{
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
  }}
  
  /* sidebar hidden */
  [data-testid="collapsedControl"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# REDDIT PARSER  (limit=500 + morechildren parallel expansion)
# ══════════════════════════════════════════════════════════════════════════════

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    )
}


def _extract_comment(c_data: dict, post_info: dict, depth: int) -> dict:
    parent_id = c_data.get('parent_id', '')
    replies_node = c_data.get('replies')
    visible = hidden = 0
    if isinstance(replies_node, dict):
        for ch in replies_node.get('data', {}).get('children', []):
            if ch['kind'] == 't1':
                visible += 1
            elif ch['kind'] == 'more':
                hidden += ch['data'].get('count', 0)
    utc = c_data.get('created_utc', 0)
    return {
        'id':              c_data.get('id'),
        'thread_id':       post_info['id'],
        'subreddit':       post_info['subreddit'],
        'parent_id':       parent_id,
        'is_root_comment': parent_id.startswith('t3_'),
        'author':          c_data.get('author'),
        'body':            c_data.get('body'),
        'score':           c_data.get('score', 0),
        'depth':           depth,
        'replies_count':   visible + hidden,
        'created_utc':     utc,
        'created_at':      datetime.fromtimestamp(utc).strftime('%Y-%m-%d %H:%M') if utc else None,
    }


def _parse_tree(children, post_info, rows, more_ids, depth=0):
    for child in children:
        if child['kind'] == 't1':
            c = child['data']
            rows.append(_extract_comment(c, post_info, depth))
            rn = c.get('replies')
            if isinstance(rn, dict):
                _parse_tree(rn['data'].get('children', []), post_info, rows, more_ids, depth + 1)
        elif child['kind'] == 'more':
            ids = child['data'].get('children', [])
            if ids:
                more_ids.extend(ids)


def _fetch_more_batch(session, post_id, batch):
    """Single morechildren request for one batch of IDs."""
    try:
        r = session.get(
            'https://www.reddit.com/api/morechildren.json',
            params={
                'api_type': 'json',
                'link_id':  f't3_{post_id}',
                'children': ','.join(batch),
                'depth':    '5',
            },
            timeout=20,
        )
        if r.status_code == 200:
            return r.json().get('json', {}).get('data', {}).get('things', [])
    except Exception:
        pass
    return []


def get_reddit_data(url: str, min_comments: int = 30, max_extra_batches: int = 20):
    """
    Returns (error_str, None) on failure or (post_info dict, DataFrame).
    Uses limit=500 for initial fetch + parallel morechildren expansion.
    """
    parsed   = urllib.parse.urlparse(url)
    base_url = f"https://{parsed.netloc}{parsed.path.rstrip('/')}"
    json_url = base_url + '.json'

    session = requests.Session()
    session.headers.update(_HEADERS)

    # ── initial fetch ──────────────────────────────────────────────────────────
    try:
        resp = session.get(json_url, params={'limit': 500, 'depth': 10}, timeout=15)
        if resp.status_code == 404:
            return "Post not found — double-check the URL.", None
        if resp.status_code != 200:
            return f"Reddit returned error {resp.status_code}. Try again later.", None
        data = resp.json()
    except requests.ConnectionError:
        return "Connection error. Check the URL and your internet.", None
    except Exception as exc:
        return f"Unexpected error: {exc}", None

    if not isinstance(data, list) or len(data) < 2:
        return "Unexpected Reddit API response.", None

    pr = data[0]['data']['children'][0]['data']
    post_info = {
        'id':           pr.get('id'),
        'title':        pr.get('title', ''),
        'text':         pr.get('selftext', ''),
        'author':       pr.get('author', ''),
        'score':        pr.get('score', 0),
        'num_comments': pr.get('num_comments', 0),
        'subreddit':    pr.get('subreddit', ''),
        'url':          base_url,
    }

    if post_info['num_comments'] < min_comments:
        return (
            f"Thread too small ({post_info['num_comments']} comments). "
            f"Need at least {min_comments}.", None
        )

    # ── walk initial tree ──────────────────────────────────────────────────────
    rows: list = []
    more_ids: list = []
    _parse_tree(data[1]['data']['children'], post_info, rows, more_ids)

    # ── expand morechildren in parallel ───────────────────────────────────────
    BATCH = 20
    batches = [
        more_ids[i:i + BATCH]
        for i in range(0, len(more_ids), BATCH)
    ][:max_extra_batches]

    if batches:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_fetch_more_batch, session, post_info['id'], b): b
                for b in batches
            }
            for fut in as_completed(futures):
                for thing in fut.result():
                    if thing['kind'] == 't1':
                        d = thing['data']
                        rows.append(_extract_comment(d, post_info, d.get('depth', 1)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df['body'].notna()]
        df = df[~df['body'].isin(['[deleted]', '[removed]'])]
        df = df[df['body'].str.strip().str.len() > 0]

    return post_info, df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_#~>]', '', text)
    text = re.sub(r'\/[ur]\/[A-Za-z0-9_-]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df['body'].astype(str).isin({'[deleted]', '[removed]', 'nan', 'None'})]
    df['body_clean'] = df['body'].apply(clean_text)
    df = df[df['body_clean'].str.len() > 0]
    df['word_count'] = df['body_clean'].str.split().str.len()
    df = df[df['word_count'] >= 5]
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING ENGINE  (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer('all-MiniLM-L6-v2', device=device)


def embed(model, texts):
    return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)


# ══════════════════════════════════════════════════════════════════════════════
# POLARIZATION METRIC
# ══════════════════════════════════════════════════════════════════════════════

def polarization_score(X, min_n=10):
    X = np.asarray(X, dtype=float)
    if len(X) < min_n:
        return 0.0
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    c1, c2 = km.cluster_centers_
    labels = km.labels_
    n1, n2 = (labels == 0).sum(), (labels == 1).sum()
    balance = 2 * min(n1, n2) / (n1 + n2)
    D = float(np.linalg.norm(c1 - c2))
    w1 = np.linalg.norm(X[labels == 0] - c1, axis=1).mean() if n1 else 0
    w2 = np.linalg.norm(X[labels == 1] - c2, axis=1).mean() if n2 else 0
    W = (w1 + w2) / 2
    sil = silhouette_score(X, labels)
    confidence = max(0.0, (sil + 1) / 2)
    raw = (D / (W + 1e-9)) * balance * confidence
    return round(min(10.0, max(0.0, raw * 12.0)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# COMMUNITY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class CommunityAnalyzer:
    def analyze(self, df: pd.DataFrame, embeddings: np.ndarray):
        # 5-D UMAP for robust clustering
        r5 = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
        e5 = r5.fit_transform(embeddings)

        clust = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
        df = df.copy()
        df['cluster'] = clust.fit_predict(e5)
        df['sentiment'] = df['body_clean'].apply(lambda t: TextBlob(t).sentiment.polarity)

        # 2-D UMAP from 5-D (much faster than from raw)
        r2 = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42)
        e2 = r2.fit_transform(e5)
        df['umap_x'] = e2[:, 0]
        df['umap_y'] = e2[:, 1]

        rows = []
        for cid in sorted(c for c in df['cluster'].unique() if c != -1):
            mask = df['cluster'] == cid
            cdf  = df[mask]
            cemb = embeddings[mask.values]

            # core: closest to centroid
            centroid = cemb.mean(axis=0, keepdims=True)
            idx, _   = pairwise_distances_argmin_min(centroid, cemb)
            core     = cdf.iloc[idx[0]]['body_clean']

            # popular: highest score
            popular = cdf.loc[cdf['score'].idxmax(), 'body_clean']

            # dissent: farthest from centroid
            dists  = pairwise_distances(centroid, cemb)[0]
            dissent = cdf.iloc[dists.argmax()]['body_clean']

            # TF-IDF keywords
            vec  = TfidfVectorizer(stop_words='english', max_features=500)
            Xt   = vec.fit_transform(cdf['body_clean'].tolist())
            mw   = np.asarray(Xt.mean(axis=0)).ravel()
            terms = vec.get_feature_names_out()
            kw   = ", ".join(terms[mw.argsort()[::-1][:5]]) or "misc"

            rows.append({
                'id':           cid,
                'keywords':     kw,
                'upvotes':      int(cdf['score'].sum()),
                'participants': len(cdf),
                'polarization': polarization_score(cemb),
                'sentiment':    round(float(cdf['sentiment'].mean()), 3),
                'popular':      popular,
                'core':         core,
                'dissent':      dissent,
            })

        summary = pd.DataFrame(rows).sort_values('upvotes', ascending=False).reset_index(drop=True)
        return df, summary


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color=TEXT,
    font_family='DM Sans, sans-serif',
)


def _L(**kw):
    return {**_LAYOUT, **kw}


class Visualizer:
    P = PALETTE

    # ── 1. UMAP Opinion Map ──────────────────────────────────────────────────
    def opinion_map(self, df):
        fig = px.scatter(
            df, x='umap_x', y='umap_y',
            color=df['cluster'].astype(str),
            hover_data={'body_clean': True, 'score': True},
            title="OPINION LANDSCAPE",
            color_discrete_sequence=self.P,
            labels={'color': 'Topic'},
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(**_L(
            height=500,
            title_font_size=13, title_font_family='DM Sans',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        ))
        return fig

    # ── 2. Polarization Gauge ────────────────────────────────────────────────
    def gauge(self, value):
        col = RED_BRIGHT if value > 6 else (RED if value > 3 else '#27ae60')
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': "Polarization Index", 'font': {'color': MUTED, 'size': 12}},
            number={'font': {'color': col, 'size': 38}, 'suffix': '/10'},
            gauge={
                'axis': {'range': [0, 10], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
                'bar': {'color': col},
                'bgcolor': BG_CARD,
                'steps': [
                    {'range': [0, 3],  'color': 'rgba(39,174,96,0.15)'},
                    {'range': [3, 7],  'color': 'rgba(230,126,34,0.10)'},
                    {'range': [7, 10], 'color': 'rgba(192,57,43,0.20)'},
                ],
            },
        ))
        fig.update_layout(**_L(height=260, margin=dict(l=20, r=20, t=50, b=10)))
        return fig

    # ── 3. Emotion Radar (NRCLex) ────────────────────────────────────────────
    def emotion_radar(self, df, cluster_id):
        cdf  = df[df['cluster'] == cluster_id]
        text = " ".join(cdf['body_clean'])
        raw  = NRCLex(text).raw_emotion_scores

        cats = ['fear', 'anger', 'trust', 'surprise', 'joy', 'sadness', 'disgust', 'anticipation']
        total = sum(raw.values()) or 1
        vals  = [(raw.get(c, 0) / total) * 100 for c in cats]
        vals.append(vals[0])
        labels = [c.capitalize() for c in cats] + [cats[0].capitalize()]

        # color per emotion
        emotion_colors = {
            'Fear':'#8e44ad','Anger':'#c0392b','Trust':'#27ae60',
            'Surprise':'#f39c12','Joy':'#f1c40f','Sadness':'#2980b9',
            'Disgust':'#16a085','Anticipation':'#d35400',
        }
        peak_idx = int(np.argmax(vals[:-1]))
        peak_cat = labels[peak_idx]
        fill_col = emotion_colors.get(peak_cat, RED)
        # Convert hex to rgba (Plotly doesn't support 8-digit hex alpha)
        hx = fill_col.lstrip('#')
        r_, g_, b_ = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
        fill_rgba = f'rgba({r_},{g_},{b_},0.25)'

        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=labels,
            fill='toself',
            line_color=fill_col,
            fillcolor=fill_rgba,
            hovertemplate='%{theta}: %{r:.1f}%<extra></extra>',
        ))
        fig.update_layout(**_L(
            height=420,
            title=dict(text=f"EMOTIONAL SPECTRUM — TOPIC #{cluster_id}", font_size=13),
            polar=dict(
                radialaxis=dict(visible=True, gridcolor=BORDER, range=[0, max(vals[:-1]) + 8]),
                angularaxis=dict(gridcolor=BORDER),
                bgcolor='rgba(0,0,0,0)',
            ),
        ))
        return fig

    # ── 4. Topic Network ────────────────────────────────────────────────────
    def topic_network(self, df, summary, embeddings):
        ids = summary['id'].tolist()
        centroids = np.array([
            embeddings[df['cluster'].values == cid].mean(axis=0) for cid in ids
        ])
        sim = cosine_similarity(centroids)
        n   = len(ids)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(ang), np.sin(ang)

        fig = go.Figure()
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] > 0.45:
                    alpha = round(float(sim[i, j]), 2)
                    fig.add_trace(go.Scatter(
                        x=[xs[i], xs[j]], y=[ys[i], ys[j]],
                        mode='lines',
                        line=dict(color=f'rgba(192,57,43,{alpha})', width=sim[i, j] * 5),
                        hoverinfo='none', showlegend=False,
                    ))

        sizes = (np.sqrt(summary['participants'].values.astype(float)) * 5).clip(8, 60)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            marker=dict(size=sizes, color=self.P[:n], line=dict(width=2, color='white')),
            text=[f"T{i}" for i in ids],
            textposition='top center',
            hovertext=[
                f"<b>Topic {k}</b><br>{kw}<br>Participants: {p}<br>Polarization: {pol:.1f}"
                for k, kw, p, pol in zip(
                    ids, summary['keywords'], summary['participants'], summary['polarization']
                )
            ],
            hoverinfo='text', showlegend=False,
        ))
        fig.update_layout(**_L(
            height=500,
            title=dict(text="TOPIC INTER-CONNECTIONS", font_size=13),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ))
        return fig

    # ── 5. Keyword Bar Chart (replaces random word cloud) ───────────────────
    def keyword_bar(self, df, cluster_id, n=15):
        texts = df[df['cluster'] == cluster_id]['body_clean'].tolist()
        if not texts:
            return go.Figure()
        vec    = TfidfVectorizer(stop_words='english', max_features=500)
        X      = vec.fit_transform(texts)
        terms  = vec.get_feature_names_out()
        weights = np.asarray(X.mean(axis=0)).ravel()

        top = weights.argsort()[::-1][:n]
        ws  = weights[top][::-1]
        wds = terms[top][::-1]
        mx  = ws[-1] if ws[-1] > 0 else 1
        colors = [f'rgba(192,57,43,{0.3 + 0.7 * v/mx:.2f})' for v in ws]

        fig = go.Figure(go.Bar(
            x=ws, y=wds, orientation='h',
            marker_color=colors,
            marker_line_color=RED_BRIGHT, marker_line_width=0.3,
            hovertemplate='%{y}: %{x:.4f}<extra></extra>',
        ))
        fig.update_layout(**_L(
            height=420,
            title=dict(text=f"TOP KEYWORDS — TOPIC #{cluster_id}", font_size=13),
            xaxis=dict(title='TF-IDF score', gridcolor=BORDER, zeroline=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=20, t=40, b=0),
        ))
        return fig

    # ── 6. Sentiment Violin (replaces boring histogram) ──────────────────────
    def sentiment_violin(self, topic_df):
        df2 = topic_df.copy()
        df2['label'] = pd.cut(
            df2['sentiment'],
            bins=[-1.01, -0.05, 0.05, 1.01],
            labels=['Negative 😡', 'Neutral 😐', 'Positive 😊'],
        )
        colors = {'Negative 😡': RED, 'Neutral 😐': '#7f8c8d', 'Positive 😊': '#27ae60'}
        fig = go.Figure()
        for lbl in ['Negative 😡', 'Neutral 😐', 'Positive 😊']:
            sub = df2[df2['label'] == lbl]['sentiment'].dropna()
            fig.add_trace(go.Violin(
                y=sub, x=[lbl] * len(sub), name=lbl,
                box_visible=True, meanline_visible=True,
                fillcolor=colors[lbl], line_color='white', opacity=0.75,
            ))
        fig.update_layout(**_L(
            height=320,
            title=dict(text="SENTIMENT DISTRIBUTION", font_size=13),
            yaxis=dict(title='Polarity', gridcolor=BORDER),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        ))
        return fig

    # ── 7. Activity Timeline (NEW) ───────────────────────────────────────────
    def activity_timeline(self, df):
        if 'created_utc' not in df.columns or df['created_utc'].isna().all():
            return go.Figure().update_layout(**_L(title="No time data"))
        df2 = df[df['created_utc'] > 0].copy()
        df2['dt']  = pd.to_datetime(df2['created_utc'], unit='s')
        df2['bin'] = df2['dt'].dt.floor('30T')
        agg = df2.groupby('bin').agg(
            count=('id', 'count'),
            avg_score=('score', 'mean'),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agg['bin'], y=agg['count'],
            name='Comments',
            marker_color=RED, marker_line_color=RED_BRIGHT, marker_line_width=0.4,
        ))
        fig.add_trace(go.Scatter(
            x=agg['bin'], y=agg['avg_score'],
            name='Avg Score', yaxis='y2',
            line=dict(color=RED_PALE, width=2),
            mode='lines+markers', marker=dict(size=4),
        ))
        fig.update_layout(**_L(
            height=340,
            title=dict(text="COMMENT ACTIVITY OVER TIME", font_size=13),
            xaxis_title='Time',
            yaxis=dict(title='Comments', gridcolor=BORDER),
            yaxis2=dict(title='Avg Score', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation='h', y=1.08, font_size=11),
            margin=dict(l=0, r=40, t=50, b=0),
        ))
        return fig

    # ── 8. Score by Comment Depth (NEW) ─────────────────────────────────────
    def depth_engagement(self, df):
        if 'depth' not in df.columns:
            return go.Figure().update_layout(**_L(title="No depth data"))
        agg = (
            df.groupby('depth')
            .agg(avg_score=('score', 'mean'), count=('id', 'count'))
            .reset_index()
        )
        agg = agg[agg['depth'] <= 8]
        mx  = agg['avg_score'].max() or 1
        colors = [
            f'rgba(192,57,43,{max(0.2, 1 - 0.1*d):.2f})' for d in agg['depth']
        ]
        fig = go.Figure(go.Bar(
            x=agg['depth'], y=agg['avg_score'],
            marker_color=colors, marker_line_color=RED_BRIGHT, marker_line_width=0.4,
            text=[f"n={c}" for c in agg['count']],
            textposition='outside',
            textfont=dict(color=MUTED, size=11),
        ))
        fig.update_layout(**_L(
            height=300,
            title=dict(text="ENGAGEMENT BY COMMENT DEPTH", font_size=13),
            xaxis=dict(title='Depth  (0 = top-level)', dtick=1, gridcolor=BORDER),
            yaxis=dict(title='Avg Upvotes', gridcolor=BORDER),
            margin=dict(l=0, r=0, t=40, b=0),
        ))
        return fig

    # ── 9. Top Contributors (NEW) ────────────────────────────────────────────
    def top_commenters(self, df, n=10):
        bad = {'[deleted]', 'AutoModerator', None}
        agg = (
            df[~df['author'].isin(bad)]
            .groupby('author')
            .agg(total_score=('score', 'sum'), comments=('id', 'count'))
            .reset_index()
            .sort_values('total_score', ascending=True)
            .tail(n)
        )
        fig = go.Figure(go.Bar(
            x=agg['total_score'], y=agg['author'], orientation='h',
            marker=dict(
                color=agg['total_score'],
                colorscale=[[0, RED_DARKER], [0.5, RED], [1, RED_BRIGHT]],
                showscale=False,
            ),
            text=[f"{c} posts" for c in agg['comments']],
            textposition='auto',
            hovertemplate='%{y}<br>Total score: %{x}<extra></extra>',
        ))
        fig.update_layout(**_L(
            height=340,
            title=dict(text=f"TOP {n} CONTRIBUTORS BY UPVOTES", font_size=13),
            xaxis=dict(title='Total upvotes', gridcolor=BORDER),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=20, t=40, b=0),
        ))
        return fig

    # ── 10. Controversy Map (NEW) ────────────────────────────────────────────
    def controversy_map(self, df):
        df2 = df[df['score'].notna() & df.get('replies_count', pd.Series(0, index=df.index)).notna()].copy()
        df2['replies_count'] = df2['replies_count'].fillna(0)
        df2 = df2[df2['replies_count'] > 0]
        if df2.empty:
            return go.Figure().update_layout(**_L(title="Not enough data"))
        fig = px.scatter(
            df2, x='score', y='replies_count',
            color=df2['cluster'].astype(str),
            size=df2['replies_count'].clip(1, 50),
            hover_data=['body_clean'] if 'body_clean' in df2 else None,
            title="CONTROVERSY MAP: Upvotes vs. Replies",
            color_discrete_sequence=self.P,
            labels={'color': 'Topic', 'score': 'Upvotes', 'replies_count': 'Replies'},
        )
        fig.update_layout(**_L(
            height=360,
            xaxis=dict(title='Upvotes (negative = downvoted)', gridcolor=BORDER, zeroline=True, zerolinecolor=BORDER2),
            yaxis=dict(title='Reply Count', gridcolor=BORDER),
            margin=dict(l=0, r=0, t=40, b=0),
        ))
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE  (cached by URL / df content)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_analysis(df_json: str):
    df = pd.read_json(df_json)
    df = preprocess(df)
    if df.empty:
        return None

    model     = get_model()
    embeddings = embed(model, df['body_clean'].tolist())

    analyzer  = CommunityAnalyzer()
    df_out, summary = analyzer.analyze(df, embeddings)

    return {
        'df':         df_out.to_json(),
        'summary':    summary.to_json(),
        'embeddings': embeddings.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# AI (Groq)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_ai():
    key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    except Exception:
        return None


_AI_MODEL = "llama-3.3-70b-versatile"


def _gen(client, prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model=_AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=700,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        return "__QUOTA__" if "429" in msg else f"__ERR__: {msg[:200]}"


def _chunks(texts, max_chars=8000):
    out, cur, n = [], [], 0
    for t in map(str, texts):
        if not t.strip():
            continue
        if n + len(t) > max_chars and cur:
            out.append("\n".join(cur)); cur, n = [], 0
        cur.append(t); n += len(t)
    if cur:
        out.append("\n".join(cur))
    return out


def ai_thread_summary(client, texts, meta=""):
    parts = []
    for i, ch in enumerate(_chunks(texts), 1):
        out = _gen(client, f"Summarize Reddit comments. Background: {meta}\n\nChunk {i}:\n{ch}\n\nBullet points: key points, viewpoints, paraphrased examples.")
        if out.startswith("__"): return out.replace("__QUOTA__","Rate limit hit.").replace("__ERR__:","Error: ")
        parts.append(out)
    if len(parts) > 1:
        return _gen(client, f"""Combine into ONE summary. Background: {meta}
Return:
1) Summary (3-5 sentences)
2) Main viewpoints (bullets)
3) Key disagreements (bullets)
4) Neutral takeaway (bullets)

PARTIALS:
{"---".join(parts)}""")
    return parts[0] if parts else "No content."


def ai_topic_explain(client, texts, meta=""):
    ch = _chunks(texts[:60], 9000)
    content = ch[0] if ch else ""
    return _gen(client, f"""Explain this Reddit topic simply.
{meta}

- What are people arguing about? (1-2 sentences)
- Main positions (bullets)
- Why they disagree (bullets)
- Neutral takeaway (bullets)

Rules: only use comments below, no quotes, no invented facts.

COMMENTS:
{content}""")


def ai_chart_explain(client, ctx: dict):
    return _gen(client, f"""Explain this analytics chart for a non-technical reader.

Context: {json.dumps(ctx, ensure_ascii=False)}

Return:
1) What this shows (1-2 sentences)
2) How to read it (bullets)
3) What's notable (bullets)
4) What NOT to over-interpret (bullets)""")


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

for k, v in [('page', 'landing'), ('data', None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label, value):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )


def quote_box(text, border_color=RED):
    return (
        f'<div style="background:{BG};padding:14px;border-radius:8px;'
        f'border-left:3px solid {border_color};color:{TEXT};'
        f'font-size:13px;line-height:1.6;margin-bottom:14px;">'
        f'{text}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LANDING
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == 'landing':
    st.markdown('<br>', unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="badge">Reddit Thread Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-title">Reddit<br>Analyzer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="subtitle">Opinion clusters · Emotion mapping · Polarization index</div>',
            unsafe_allow_html=True
        )

        url = st.text_input("", placeholder="Paste Reddit post URL here…", label_visibility="collapsed")

        c1, c2 = st.columns([1, 2])
        with c1:
            go_btn = st.button("⚡ Analyse", use_container_width=True)
        with c2:
            st.markdown(
                f'<p style="color:{MUTED};font-size:11px;margin-top:10px;letter-spacing:0.5px;">'
                f'Works on any public Reddit post · Loads all available comments</p>',
                unsafe_allow_html=True
            )

        if go_btn:
            if not url.strip():
                st.warning("Please enter a URL.")
            else:
                prog = st.progress(0, "Fetching Reddit data…")
                post_info, df_raw = get_reddit_data(url.strip())
                prog.progress(35, "Running NLP analysis…")

                if isinstance(post_info, str):
                    prog.empty()
                    st.error(post_info)
                else:
                    result = run_analysis(df_raw.to_json())
                    prog.progress(100, "Done!")
                    time.sleep(0.4)
                    prog.empty()

                    if result is None:
                        st.error("Not enough valid comments after cleaning.")
                    else:
                        st.session_state.data = {
                            'info':       post_info,
                            'df':         pd.read_json(result['df']),
                            'summary':    pd.read_json(result['summary']),
                            'embeddings': np.array(result['embeddings']),
                        }
                        st.session_state.page = 'overview'
                        st.rerun()

        st.markdown('<br><br>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:flex;gap:28px;color:{MUTED};font-size:12px;line-height:1.8;">
  <div>📊 <span style="color:{MUTED2};">Cluster Analysis</span><br>HDBSCAN semantic topics</div>
  <div>🎭 <span style="color:{MUTED2};">Emotion Radar</span><br>8-axis NRC spectrum</div>
  <div>🔥 <span style="color:{MUTED2};">Polarization</span><br>Two-camp divergence</div>
  <div>📈 <span style="color:{MUTED2};">Engagement</span><br>Activity & depth stats</div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
<div style="
  margin-top:80px;
  background:{BG_CARD};
  border:1px solid {BORDER};
  border-radius:16px;
  padding:36px 28px;
">
  <div style="
    width:60px;height:60px;border-radius:12px;
    background:linear-gradient(135deg,{RED} 0%,{RED_DEEP} 100%);
    display:flex;align-items:center;justify-content:center;
    font-size:28px;margin-bottom:20px;
  ">🔍</div>
  <div style="color:{TEXT};font-size:17px;font-weight:600;margin-bottom:6px;
              font-family:'DM Serif Display',serif;">
    Deep Thread Analysis
  </div>
  <div style="color:{MUTED};font-size:13px;line-height:2.0;">
    · Loads <b style="color:{MUTED2};">all</b> comments via Reddit API<br>
    · Semantic clustering with UMAP + HDBSCAN<br>
    · Emotion analysis across 8 dimensions<br>
    · Polarization index per topic cluster<br>
    · Activity timeline & engagement patterns<br>
    · Optional AI summaries via Groq<br>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 'overview':
    data   = st.session_state.data
    viz    = Visualizer()
    client = get_ai()

    # ── header ────────────────────────────────────────────────────────────────
    cb, ct = st.columns([1, 7])
    with cb:
        if st.button("← Back"):
            st.session_state.page = 'landing'; st.rerun()
    with ct:
        st.markdown(
            f'<h2 style="margin:0;color:{TEXT};">{data["info"]["title"]}</h2>',
            unsafe_allow_html=True
        )
    st.markdown(
        f'<p style="color:{MUTED};margin-top:4px;">r/{data["info"]["subreddit"]} · '
        f'{data["info"]["num_comments"]} total comments · '
        f'<b style="color:{MUTED2};">{len(data["df"])}</b> analysed</p>',
        unsafe_allow_html=True
    )
    st.divider()

    # ── global metrics ────────────────────────────────────────────────────────
    avg_pol = float(np.average(
        data['summary']['polarization'].astype(float),
        weights=data['summary']['participants'].astype(float)
    ))
    avg_sent = float(data['df']['sentiment'].mean())

    mc = st.columns(4)
    for col, lbl, val in zip(mc, [
        "Comments Analysed", "Topic Clusters", "Avg Polarization", "Avg Sentiment"
    ], [
        len(data['df']), len(data['summary']), f"{avg_pol:.2f}/10", f"{avg_sent:+.3f}"
    ]):
        col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # ── AI summary ────────────────────────────────────────────────────────────
    with st.expander("🤖 AI Thread Summary", expanded=False):
        if client is None:
            st.info("Add GROQ_API_KEY to Streamlit secrets to enable AI features.")
        else:
            if st.button("Generate AI Summary"):
                with st.spinner("AI is reading the thread…"):
                    meta = f'Post: "{data["info"]["title"]}" | r/{data["info"]["subreddit"]}'
                    texts = data['df']['body_clean'].dropna().astype(str).tolist()
                    st.session_state['ai_thread'] = ai_thread_summary(client, texts, meta)
            if 'ai_thread' in st.session_state:
                st.markdown(st.session_state['ai_thread'])
                st.caption("AI reads retrieved comments only — some context may be missing.")

    st.divider()

    # ── main tabs ─────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🗺  Opinion Map", "🕸  Topic Network", "📈  Activity", "👥  Contributors"
    ])

    with t1:
        cmap, cgauge = st.columns([3, 1])
        with cmap:
            st.plotly_chart(viz.opinion_map(data['df']), use_container_width=True)
            if client:
                if st.button("AI: Explain map", key="ai_map"):
                    with st.spinner("AI explaining…"):
                        ctx = {
                            "chart": "UMAP Opinion Landscape",
                            "clusters": int(len(data['summary'])),
                            "comments": int(len(data['df'])),
                            "topics": data['summary'][['id','keywords','participants','polarization']].head(6).to_dict('records'),
                        }
                        st.session_state['ai_map_txt'] = ai_chart_explain(client, ctx)
                if 'ai_map_txt' in st.session_state:
                    st.caption(st.session_state['ai_map_txt'])
        with cgauge:
            st.plotly_chart(viz.gauge(avg_pol), use_container_width=True)
            st.markdown(metric_card("Clusters", len(data['summary'])), unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(metric_card("Analysed", len(data['df'])), unsafe_allow_html=True)

    with t2:
        st.plotly_chart(
            viz.topic_network(data['df'], data['summary'], data['embeddings']),
            use_container_width=True
        )

    with t3:
        st.plotly_chart(viz.activity_timeline(data['df']), use_container_width=True)
        st.plotly_chart(viz.depth_engagement(data['df']), use_container_width=True)

    with t4:
        ca, cb_ = st.columns(2)
        with ca:
            st.plotly_chart(viz.top_commenters(data['df']), use_container_width=True)
        with cb_:
            st.plotly_chart(viz.controversy_map(data['df']), use_container_width=True)

    st.divider()

    # ── cluster cards ─────────────────────────────────────────────────────────
    st.markdown(f'<h3 style="color:{TEXT};font-family:DM Serif Display,serif;">DETECTED TOPIC CLUSTERS</h3>', unsafe_allow_html=True)

    for _, row in data['summary'].iterrows():
        pol = float(row['polarization'])
        icon = "🔴" if pol > 6 else ("🟡" if pol > 3 else "🟢")
        with st.expander(f"{icon}  TOPIC #{row['id']}  ·  {row['keywords']}"):
            c_a, c_b, c_c = st.columns([1, 2, 1])
            with c_a:
                st.metric("Upvotes",      int(row['upvotes']))
                st.metric("Participants", int(row['participants']))
                st.metric("Polarization", f"{pol:.2f}/10")
                st.metric("Sentiment",    f"{float(row['sentiment']):+.3f}")
            with c_b:
                st.markdown("**Core Message**")
                st.markdown(quote_box(row['core']), unsafe_allow_html=True)
            with c_c:
                st.markdown('<br>', unsafe_allow_html=True)
                if st.button("Deep Dive →", key=f"dd_{row['id']}", use_container_width=True):
                    st.session_state.selected_topic = row['id']
                    st.session_state.page = 'detail'
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DETAIL
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == 'detail':
    data     = st.session_state.data
    tid      = st.session_state.selected_topic
    viz      = Visualizer()
    client   = get_ai()

    row      = data['summary'][data['summary']['id'] == tid].iloc[0]
    topic_df = data['df'][data['df']['cluster'] == tid].copy()

    if st.button("← Back to Overview"):
        st.session_state.page = 'overview'; st.rerun()

    st.markdown(
        f'<h2 style="color:{TEXT};font-family:DM Serif Display,serif;">DEEP DIVE — TOPIC #{tid}</h2>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color:{MUTED};">Keywords: <span style="color:{RED_BRIGHT};">{row["keywords"]}</span></p>',
        unsafe_allow_html=True
    )

    # metrics
    pol = float(row['polarization'])
    mc  = st.columns(4)
    for col, lbl, val in zip(mc, [
        "Upvotes", "Participants", "Polarization", "Sentiment"
    ], [
        int(row['upvotes']), int(row['participants']),
        f"{pol:.2f}/10", f"{float(row['sentiment']):+.3f}"
    ]):
        col.markdown(metric_card(lbl, val), unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col_text, col_viz = st.columns(2, gap="large")

    # ── text panel ────────────────────────────────────────────────────────────
    with col_text:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f'**🎯 Core Message**')
        st.markdown(quote_box(row['core']), unsafe_allow_html=True)

        st.markdown(f'**⭐ Most Popular Comment**')
        st.markdown(quote_box(row['popular'], '#27ae60'), unsafe_allow_html=True)

        border = RED_BRIGHT if pol > 5 else '#7f8c8d'
        label  = "⚠️ Dissenting Voice" if pol > 5 else "💬 Alternative View"
        st.markdown(f'**{label}**')
        st.markdown(quote_box(row['dissent'], border), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if pol > 6:
            st.error(f"⚠️ Highly controversial — Polarization {pol:.1f}/10")
        elif pol > 3:
            st.warning(f"Moderate disagreement — Polarization {pol:.1f}/10")
        else:
            st.success(f"Stable consensus — Polarization {pol:.1f}/10")

        if client:
            if st.button("🤖 AI: Explain this topic", key=f"ai_t_{tid}"):
                with st.spinner("AI analysing…"):
                    meta  = f"Topic #{tid} | Keywords: {row['keywords']} | Participants: {int(row['participants'])}"
                    texts = topic_df['body_clean'].dropna().astype(str).tolist()
                    st.session_state[f'ai_t_{tid}'] = ai_topic_explain(client, texts, meta)
            if f'ai_t_{tid}' in st.session_state:
                st.info(st.session_state[f'ai_t_{tid}'])

    # ── viz panel ─────────────────────────────────────────────────────────────
    with col_viz:
        vt1, vt2, vt3, vt4 = st.tabs([
            "🎭 Emotions", "📊 Sentiment", "🔤 Keywords", "📈 Activity"
        ])

        with vt1:
            st.plotly_chart(viz.emotion_radar(data['df'], tid), use_container_width=True)
            if client:
                if st.button("AI: Explain emotions", key=f"ai_emo_{tid}"):
                    cdf = data['df'][data['df']['cluster'] == tid]
                    text = " ".join(cdf['body_clean'])
                    raw = NRCLex(text).raw_emotion_scores
                    total = sum(raw.values()) or 1
                    dist = {k: round(v/total*100, 1) for k,v in raw.items() if v > 0}
                    with st.spinner():
                        st.session_state[f'ai_emo_{tid}'] = ai_chart_explain(
                            client, {"chart": "Emotion Radar", "topic_id": int(tid), "distribution_%": dist}
                        )
                if f'ai_emo_{tid}' in st.session_state:
                    st.caption(st.session_state[f'ai_emo_{tid}'])

        with vt2:
            st.plotly_chart(viz.sentiment_violin(topic_df), use_container_width=True)

        with vt3:
            st.plotly_chart(viz.keyword_bar(data['df'], tid), use_container_width=True)

        with vt4:
            st.plotly_chart(viz.activity_timeline(topic_df), use_container_width=True)
