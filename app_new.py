import requests
import pandas as pd
import json
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
from sklearn.metrics import pairwise_distances_argmin_min
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import umap.umap_ as umap
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.parse
import torch
from textblob import TextBlob
import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import torch
from nrclex import NRCLex
import nltk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from textblob import TextBlob




try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')

#КОНФИГУРАЦИЯ СТРАНИЦЫ
st.set_page_config(page_title="Reddit Thread Intelligence", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Главный фон */
    .stApp {
        background-color: #050505;
        color: white;
    }
    
    /* Стилизация заголовков */
    .main-title {
        font-size: 80px;
        font-weight: 800;
        font-family: 'Impact', sans-serif;
        line-height: 0.9;
        margin-bottom: 20px;
    }
    
    /* Инпут и кнопка */
    .stTextInput input {
        background-color: #0a051a !important;
        color: white !important;
        border: 1px solid #2d1b4e !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    div.stButton > button {
        background-color: #0a051a;
        color: white;
        border: 1px solid #ffffff;
        border-radius: 20px;
        padding: 10px 40px;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #ffffff;
        color: black;
    }

    /* Карточки и контейнеры */
    .analysis-card {
        background-color: #0a051a;
        border: 1px solid #2d1b4e;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .topic-item {
        padding: 10px;
        border-bottom: 1px solid #2d1b4e;
        cursor: pointer;
    }
    
    .metric-box {
        border-right: 1px solid #2d1b4e;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#ЯДЕРНЫЙ КОД

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_reddit_data_advanced(url, min_comments=50):
    parsed_url = urllib.parse.urlparse(url)
    clean_url = f"https://{parsed_url.netloc}{parsed_url.path}"
    json_url = clean_url.rstrip('/') + '.json'

    headers = {
        'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(json_url, headers=headers, timeout=10)
        if response.status_code == 404:
            return "The link looks incorrect. Please check if the Reddit post exists.", None
        if response.status_code != 200:
            return f"Reddit is not responding (Error {response.status_code}). Try again later.", None
        data = response.json()

    except requests.exceptions.ConnectionError:
        return "Invalid link or connection error. Please enter a valid Reddit URL.", None
    except Exception:
        return "Something went wrong. Make sure the link is a public Reddit thread.", None

    if not isinstance(data, list) or len(data) < 2:
        return "Invalid data format received from Reddit.", None

    post_raw = data[0]['data']['children'][0]['data']
    post_info = {
        'id': post_raw.get('id'),
        'title': post_raw.get('title'),
        'text': post_raw.get('selftext'),
        'author': post_raw.get('author'),
        'score': post_raw.get('score'),
        'num_comments': post_raw.get('num_comments'),
        'subreddit': post_raw.get('subreddit'),
        'url': clean_url
    }

    if post_info['num_comments'] < min_comments:
        return f"Discussion too small ({post_info['num_comments']} comments). Minimum required: {min_comments}", None

    extracted_comments = []

    def parse_tree(children, depth=0):
        for child in children:
            if child['kind'] == 't1':
                c_data = child['data']

                visible_replies = 0
                hidden_replies = 0
                replies_node = c_data.get('replies')

                if isinstance(replies_node, dict):
                    inner_children = replies_node['data'].get('children', [])
                    for inner_child in inner_children:
                        if inner_child['kind'] == 't1':
                            visible_replies += 1
                        elif inner_child['kind'] == 'more':
                            hidden_replies += inner_child['data'].get('count', 0)

                parent_id = c_data.get('parent_id', '')
                parent_type = parent_id.split('_')[0] if '_' in parent_id else 'unknown'
                is_root = parent_id.startswith('t3_')

                extracted_comments.append({
                    'id': c_data.get('id'),
                    'thread_id': post_info['id'],
                    'subreddit': post_info['subreddit'],
                    'parent_id': parent_id,
                    'parent_type': parent_type,
                    'is_root_comment': is_root,
                    'author': c_data.get('author'),
                    'body': c_data.get('body'),
                    'score': c_data.get('score'),
                    'depth': depth,
                    'replies_count': visible_replies + hidden_replies,
                    'created_at': datetime.fromtimestamp(
                        c_data.get('created_utc')
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })

                if isinstance(replies_node, dict):
                    parse_tree(replies_node['data'].get('children'), depth + 1)

    parse_tree(data[1]['data']['children'])

    df = pd.DataFrame(extracted_comments)
    if not df.empty:
        df = df[df['body'].notnull()]
        df = df[~df['body'].isin(['[deleted]', '[removed]'])]
        df = df[df['body'].str.strip().str.len() > 0]

    return post_info, df

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_#~>]', '', text)
    text = re.sub(r'\/u\/[A-Za-z0-9_-]+', '', text)
    text = re.sub(r'\/r\/[A-Za-z0-9_-]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_reddit_data(df):
    print(f"Initial comments count: {len(df)}")
    
    clean_df = df.copy()
    noise_list = ['[deleted]', '[removed]', 'nan', 'None']
    clean_df = clean_df[~clean_df['body'].astype(str).isin(noise_list)]

    clean_df['body_clean'] = clean_df['body'].apply(clean_text)
    clean_df = clean_df[clean_df['body_clean'] != ""]

    clean_df['word_count'] = clean_df['body_clean'].apply(lambda x: len(x.split()))
    clean_df = clean_df[clean_df['word_count'] >= 5]

    print(f"Comments after preprocessing: {len(clean_df)}")
    return clean_df.reset_index(drop=True)

class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading model: {model_name}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model loaded on: {self.device}")

    def generate(self, df):
        if 'body_clean' not in df.columns:
            raise ValueError("DataFrame must contain 'body_clean' column.")
        
        sentences = df['body_clean'].tolist()
        print(f"Generating embeddings for {len(sentences)} comments...")
        
        embeddings = self.model.encode(
            sentences, 
            show_progress_bar=True, 
            batch_size=32,
            convert_to_numpy=True
        )
        return embeddings

@st.cache_resource
def get_embedding_engine():
    return EmbeddingEngine()
def polarization_two_camp(embeddings: np.ndarray, min_n: int = 30, random_state: int = 42):
    X = np.asarray(embeddings)
    n = X.shape[0]

    if n < min_n:
        return {
            "polarization": 0.0,
            "silhouette": float("nan"),
            "balance": float("nan"),
            "sep_over_within": float("nan"),
            "n": n,
            "n1": 0,
            "n2": 0
        }

    km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    c1, c2 = km.cluster_centers_[0], km.cluster_centers_[1]

    n1 = int(np.sum(labels == 0))
    n2 = int(np.sum(labels == 1))
    balance = (2.0 * min(n1, n2)) / (n1 + n2)  # 0..1

    D = float(np.linalg.norm(c1 - c2))  # separation

    d1 = np.linalg.norm(X[labels == 0] - c1, axis=1).mean() if n1 > 0 else np.inf
    d2 = np.linalg.norm(X[labels == 1] - c2, axis=1).mean() if n2 > 0 else np.inf
    W = float((d1 + d2) / 2.0)  # within spread

    sep_over_within = D / (W + 1e-9)

    sil = float(silhouette_score(X, labels))  # [-1,1]
    confidence = max(0.0, (sil + 1) / 2)      # [0,1] (если <=0 → 0)

    polarization = sep_over_within * balance * confidence

    return {
        "polarization": float(polarization),
        "silhouette": sil,
        "balance": float(balance),
        "sep_over_within": float(sep_over_within),
        "n": n,
        "n1": n1,
        "n2": n2
    }



class CommunityAnalyzer:
    def __init__(self, min_cluster_size=5):
        self.min_cluster_size = min_cluster_size
        

    def _get_sentiment(self, text):
        return TextBlob(text).sentiment.polarity

    def analyze(self, df, embeddings):
        print("Starting advanced clustering and polarization analysis...")
        
        reducer = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)
        umap_embeddings = reducer.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean', cluster_selection_method='eom')
        df['cluster'] = clusterer.fit_predict(umap_embeddings)
        
        df['sentiment'] = df['body_clean'].apply(self._get_sentiment)

        cluster_results = []
        unique_clusters = [c for c in set(df['cluster']) if c != -1]

        for cluster_id in unique_clusters:
            c_mask = df['cluster'] == cluster_id
            c_df = df[c_mask]
            c_embeddings = embeddings[c_mask]

            # 1. Opinion Leader
            leader_idx = c_df['score'].idxmax()
            opinion_leader = c_df.loc[leader_idx, 'body_clean']
            
            # 2. Representative 
            centroid = c_embeddings.mean(axis=0).reshape(1, -1)
            closest_idx, _ = pairwise_distances_argmin_min(centroid, c_embeddings)
            representative = c_df.iloc[closest_idx[0]]['body_clean']

            # 3. Dissenting Voice 
            distances = pairwise_distances(centroid, c_embeddings)[0]
            dissent_idx = np.argmax(distances)
            dissenting_voice = c_df.iloc[dissent_idx]['body_clean']

            pol_metrics = polarization_two_camp(c_embeddings, min_n=10, random_state=42)
            pol_0_10 = max(0.0, min(10.0, pol_metrics["polarization"] * 12.0)) 
            polarization_index = round(pol_0_10, 2)
            cluster_texts = c_df["body_clean"].tolist()
            vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
            X_tfidf = vectorizer.fit_transform(cluster_texts)
            mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()

            top_idx = mean_tfidf.argsort()[::-1][:5] 
            top_keywords = ", ".join([terms[i] for i in top_idx if mean_tfidf[i] > 0])
            
            if not top_keywords:
                top_keywords = "misc"

            cluster_results.append({
                'cluster_id': cluster_id,
                'topic_tags': top_keywords,
                'count': len(c_df),
                'total_score': c_df['score'].sum(),
                'polarization': polarization_index,
                'pol_balance': round(float(pol_metrics.get("balance", 0) or 0), 2),
                'pol_silhouette': round(float(pol_metrics.get("silhouette", 0) or 0), 2),
                'pol_n': int(pol_metrics.get("n", len(c_df))),
                'opinion_leader': opinion_leader,
                'representative': representative,
                'dissenting_voice': dissenting_voice,
                'avg_sentiment': round(c_df['sentiment'].mean(), 2)
            })

        summary_df = pd.DataFrame(cluster_results).sort_values(by='total_score', ascending=False)
        return df, summary_df

class Visualizer:
    def __init__(self):
        self.blue_palette = ['#0066ff', '#003399', '#4b5563', '#1f2937']

    def plot_opinion_map(self, df, embeddings):
        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)

        df['x'], df['y'] = coords[:, 0], coords[:, 1]

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data={'body_clean': True, 'score': True},
            title="COMMUNICATION LANDSCAPE",
            template="plotly_dark",
            color_discrete_sequence=self.blue_palette
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white"
        )

        return fig

    def plot_temperature_gauge(self, polarization_index):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=polarization_index,
            title={'text': "Polarization (Conflict Level)", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 10], 'tickcolor': "white"},
                'bar': {'color': "#0066ff"},
                'bgcolor': "#111827",
                'steps': [
                    {'range': [0, 3], 'color': "#064e3b"},
                    {'range': [7, 10], 'color': "#7f1d1d"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=300
        )

        return fig

    def plot_sentiment_distribution(self, topic_df):
        fig = px.histogram(
            topic_df,
            x='sentiment',
            nbins=20,
            title="SENTIMENT SPREAD",
            color_discrete_sequence=['#0066ff'],
            template="plotly_dark"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Negative <---> Positive",
            yaxis_title="Count"
        )

        return fig

    def plot_word_cloud(self, df, cluster_id):
        cluster_text = " ".join(df[df['cluster'] == cluster_id]['body_clean'])

        vectorizer = TfidfVectorizer(stop_words='english', max_features=30)
        tfidf_matrix = vectorizer.fit_transform([cluster_text])
        words = vectorizer.get_feature_names_out()
        weights = tfidf_matrix.toarray()[0]

        np.random.seed(42)
        x = np.random.uniform(0, 10, size=len(words))
        y = np.random.uniform(0, 10, size=len(words))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            text=words,
            mode='text',
            textfont={
                'size': [15 + (w * 100) for w in weights],
                'color': ['#0066ff', '#00ccff', '#ffffff', '#4b5563'][np.random.randint(0, 4)]
            },
            hovertext=[f"Importance: {round(w, 3)}" for w in weights],
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f"TOP KEYWORDS FOR CLUSTER #{cluster_id}",
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def plot_sentiment_radar(self, df, cluster_id):
        c_df = df[df['cluster'] == cluster_id]
        all_text = " ".join(c_df['body_clean'])

        emotion = NRCLex(all_text)
        raw_emotions = emotion.raw_emotion_scores

        if not raw_emotions:
            return go.Figure().update_layout(title="Not enough emotional data")

        categories = [
            'fear', 'anger', 'trust', 'surprise',
            'joy', 'sadness', 'disgust', 'anticipation'
        ]

        total = sum(raw_emotions.values()) if sum(raw_emotions.values()) > 0 else 1
        values = [(raw_emotions.get(cat, 0) / total) * 100 for cat in categories]

        values.append(values[0])
        categories.append(categories[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[c.capitalize() for c in categories],
            fill='toself',
            line_color='#0066ff',
            fillcolor='rgba(0, 102, 255, 0.3)',
            hovertemplate='%{theta}: %{r:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    gridcolor="#2d1b4e",
                    range=[0, max(values) + 5 if values else 100]
                ),
                angularaxis=dict(gridcolor="#2d1b4e"),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=400,
            title=f"EMOTIONAL SPECTRUM: TOPIC #{cluster_id} (%)"
        )

        return fig

    def plot_topic_network(self, df, summary_df, embeddings):
        centroids = []
        cluster_ids = summary_df['id'].tolist()

        for c_id in cluster_ids:
            mask = df['cluster'].values == c_id
            centroids.append(embeddings[mask].mean(axis=0))

        centroids = np.array(centroids)

        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(centroids)

        n_clusters = len(cluster_ids)
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        x_nodes = np.cos(angles)
        y_nodes = np.sin(angles)

        fig = go.Figure()

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if similarity_matrix[i, j] > 0.5:
                    fig.add_trace(go.Scatter(
                        x=[x_nodes[i], x_nodes[j]],
                        y=[y_nodes[i], y_nodes[j]],
                        mode='lines',
                        line=dict(
                            color='rgba(100, 100, 255, 0.2)',
                            width=similarity_matrix[i, j] * 5
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))

        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=[np.sqrt(c) * 5 for c in summary_df['participants']],
                color='#0066ff',
                line=dict(width=2, color='white')
            ),
            text=[f"T#{i}" for i in cluster_ids],
            textposition="top center",
            hovertext=[
                f"Topic: {k}<br>Size: {p}"
                for k, p in zip(summary_df['keywords'], summary_df['participants'])
            ],
            hoverinfo='text',
            showlegend=False
        ))

        fig.update_layout(
            title="TOPIC INTER-CONNECTIONS (Semantic Proximity)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=500
        )

        return fig

    def plot_sentiment_timeline(self, topic_comments: pd.DataFrame):
        df = topic_comments.copy()

        # safety
        if "created_utc" not in df.columns:
            return px.scatter(title="No time data (created_utc missing)")

        df = df.dropna(subset=["created_utc"])
        df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df = df.dropna(subset=["created_dt"])

        # ensure numeric
        if "sentiment" not in df.columns:
            return px.scatter(title="No sentiment column")

        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        df = df.dropna(subset=["sentiment"])

        # optional size
        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
            size_col = "score"
        else:
            df["score"] = 1
            size_col = "score"

        # label for color
        df["sent_bucket"] = pd.cut(df["sentiment"], bins=[-1, -0.05, 0.05, 1], labels=["Negative", "Neutral", "Positive"])

        fig = px.scatter(
            df.sort_values("created_dt"),
            x="created_dt",
            y="sentiment",
            size=size_col,
            color="sent_bucket",
            hover_data=["author", "score"] if "author" in df.columns else ["score"],
            title="Sentiment Trajectory Over Time",
        )

        # trend line (rolling mean on time bins)
        # bin by 30 minutes (change as you like)
        df_bin = df.set_index("created_dt").resample("30min")["sentiment"].mean().reset_index()
        fig.add_scatter(x=df_bin["created_dt"], y=df_bin["sentiment"], mode="lines", name="Avg (30min)")

        fig.update_layout(xaxis_title="Time", yaxis_title="Sentiment", legend_title="")
        return fig






def perform_analysis(df):
    df = preprocess_reddit_data(df) 
    if df.empty:
        st.error("After cleaning, no comments left for analysis.")
        return None, None, None
    engine = get_embedding_engine()
    embeddings = engine.generate(df) 
    analyzer = CommunityAnalyzer(min_cluster_size=5)
    df_ready, summary = analyzer.analyze(df, embeddings)
    summary = summary.rename(columns={
        'cluster_id': 'id',
        'topic_tags': 'keywords',
        'total_score': 'upvotes',
        'count': 'participants',
        'opinion_leader': 'popular',
        'representative': 'core',
        'dissenting_voice': 'dissent',
        'avg_sentiment': 'sentiment'
    })
    
    return df_ready, summary, embeddings



from openai import OpenAI

def get_ai_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
        return client
    except Exception as e:
        st.error(f"Failed to init Groq client: {e}")
        return None




def chunk_texts(texts, max_chars=9000):
    chunks, cur, cur_len = [], [], 0
    for t in texts:
        t = str(t)
        if not t.strip():
            continue
        if cur_len + len(t) + 2 > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(t)
        cur_len += len(t) + 2
    if cur:
        chunks.append("\n".join(cur))
    return chunks

def safe_generate(client, model: str, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            return "__QUOTA__"
        return "__ERROR__:" + msg

def ai_explain_topic_plain(client, texts, model="openai/gpt-oss-20b", meta=None):
    chunks = chunk_texts(texts, max_chars=9000)

    meta_block = ""
    if meta:
        meta_block = (
            "Context (for understanding only, do NOT repeat verbatim):\n"
            f"{meta}\n\n"
        )

    partials = []
    for i, ch in enumerate(chunks, start=1):
        prompt = f"""
You explain a Reddit TOPIC using only the COMMENTS below.
{meta_block}
Write in very simple, human language.

Return:
- What people are arguing about (1-2 sentences)
- The 2-4 main positions (bullets)
- Why they disagree (bullets)
- What a neutral person should take away (bullets)

Rules:
- Use ONLY the comments.
- No quotes.
- No invented facts.

COMMENTS (chunk #{i}):
{ch}
"""
        out = safe_generate(client, model, prompt)
        if out == "__QUOTA__":
            return "The request limit has been reached. Try again later."
        if out.startswith("__ERROR__:"):
            return "Error AI: " + out.replace("__ERROR__:", "")[:250]
        partials.append(out)

    # Reduce
    joined = "\n\n".join(partials)
    final_prompt = f"""
Combine the partial explanations into ONE final explanation.
{meta_block}

Return the same structure:
- What people are arguing about
- Main positions
- Why they disagree
- Neutral takeaway

Rules:
- No quotes.
- No invented facts.

PARTIALS:
{joined}
"""
    final_out = safe_generate(client, model, final_prompt)
    if final_out == "__QUOTA__":
        return "The request limit has been reached. Try again later."
    if final_out.startswith("__ERROR__:"):
        return "Error AI: " + final_out.replace("__ERROR__:", "")[:250]

    return final_out

def ai_explain_visual(client, ctx: dict, model="openai/gpt-oss-20b") -> str:
    prompt = f"""
You explain an analytics visualization in simple words for a non-technical user.

Context (JSON-like):
{json.dumps(ctx, ensure_ascii=False, indent=2)}

Return:
1) What this shows (1-2 sentences)
2) How to read it (bullets)
3) What looks notable here (bullets)
4) What NOT to overinterpret (bullets)
5) One practical takeaway (1 sentence)

Rules:
- Use ONLY the provided context.
- Do NOT invent extra numbers or facts.
- Keep it short and clear.
"""
    out = safe_generate(client, model, prompt)
    if out == "__QUOTA__":
        return "The request limit has been reached. Try again later."
    if out.startswith("__ERROR__:"):
        return "Error AI: " + out.replace("__ERROR__:", "")[:250]
    return out


def ai_map_reduce_summary(client, texts, prompt_role="THREAD", model="openai/gpt-oss-20b", meta=None):
    chunks = chunk_texts(texts, max_chars=9000)

    meta_block = ""
    if meta:
        meta_block = (
            "Background context (use only to understand comments; do NOT summarize this block):\n"
            f"{meta}\n\n"
        )

    # MAP
    partials = []
    for i, ch in enumerate(chunks, start=1):
        prompt = f"""
You are summarizing Reddit COMMENTS. Role={prompt_role}.
{meta_block}
IMPORTANT: Summarize ONLY the comments. Do NOT explain charts/metrics.

Summarize chunk #{i} in bullet points:
- main points
- distinct viewpoints (if any)
- 1-2 short paraphrased examples (no quotes)

COMMENTS:
{ch}
"""
        out = safe_generate(client, model, prompt)
        if out == "__QUOTA__":
            return "The request limit has been reached. Try again later."
        if out.startswith("__ERROR__:"):
            return "Error AI: " + out.replace("__ERROR__:", "")[:250]
        partials.append(out)
    joined = "\n\n".join(partials)
    final_prompt = f"""
Create a FINAL summary for Role={prompt_role}.
{meta_block}
Use ONLY the comment content from the partial summaries.

Return structure:

1) Summary (3-5 sentences)
2) Main viewpoints (2-6 bullets)
3) Key disagreements / uncertainty (bullets)
4) Notable examples (2-4 short paraphrases; no quotes)
5) What a neutral reader should know (bullets)

Rules:
- Do NOT explain charts/metrics.
- Do NOT invent facts.
- Simple language.

PARTIALS:
{joined}
"""
    final_out = safe_generate(client, model, final_prompt)
    if final_out == "__QUOTA__":
        return "The request limit has been reached. Try again later."
    if final_out.startswith("__ERROR__:"):
        return "Error AI: " + final_out.replace("__ERROR__:", "")[:250]

    return final_out



#НАВИГАЦИЯ
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'data' not in st.session_state:
    st.session_state.data = None

#СТРАНИЦА 1: LANDING 
if st.session_state.page == 'landing':
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<div class="main-title">Reddit Thread<br>Intelligence</div>',
            unsafe_allow_html=True
        )

        url_input = st.text_input(
            "",
            placeholder="Enter the link to the Reddit post...."
        )

        if st.button("Analyse"):
            if url_input:
                with st.spinner('Accessing Reddit and analyzing...'):
                    post_info, result_df = get_reddit_data_advanced(url_input)
                    if isinstance(post_info, str):
                        st.error(post_info)
                    elif result_df is not None:
                        df_ready, summary, embs = perform_analysis(result_df)

                        st.session_state.data = {
                            'info': post_info,
                            'df': df_ready,
                            'summary': summary,
                            'embeddings': embs
                        }

                        st.session_state.page = 'overview'
                        st.rerun()
            else:
                st.warning("Please enter a URL first!")

    with col2:
        st.image(
            "https://i.pinimg.com/474x/f8/fe/a6/f8fea6e8ab5328ea65edad66c74d4d92.jpg?nii=t",
            width=300
        )


#СТРАНИЦА 2: OVERVIEW
elif st.session_state.page == 'overview':
    data = st.session_state.data
    viz = Visualizer()
    client = get_ai_client()

    
    if st.button("← Back to Search", key="top_back"):
        st.session_state.page = 'landing'
        st.rerun()

    st.markdown(f"### DEEP INTELLIGENCE: {data['info']['title']}")

    #AI SUMMARY THREAD
    st.markdown("### AI Summary (Thread)")
    if client is None:
        st.info("AI is disabled: add GROQ_API_KEY to Streamlit secrets.")
    else:
        col_ai1, col_ai2 = st.columns([1, 3])
        with col_ai1:
            if st.button("Generate Thread Summary", use_container_width=True):
                with st.spinner("AI is summarizing the whole thread..."):
                    texts = data["df"]["body_clean"].dropna().astype(str).tolist()
                    st.session_state["ai_thread_summary"] = ai_map_reduce_summary(
                        client, texts, prompt_role="THREAD"
                    )

        with col_ai2:
            if "ai_thread_summary" in st.session_state:
                st.markdown(st.session_state["ai_thread_summary"])
                st.caption("Note: AI summary is based only on retrieved comments and may miss details.")

    st.divider()

    #ГЛОБАЛЬНАЯ АНАЛИТИКА 
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

    tab_map, tab_network = st.tabs(["🗺 Opinion Landscape", "🕸 Topic Inter-Connections"])

    with tab_map:
        col_m1, col_m2 = st.columns([3, 1])

        with col_m1:
            if 'embeddings' in data:
                fig_map = viz.plot_opinion_map(data['df'], data['embeddings'])
                st.plotly_chart(fig_map, use_container_width=True)
                if client is not None:
                    if st.button("AI: Explain Opinion Landscape", key="ai_explain_map"):
                        ctx = {
                            "chart": "Opinion Landscape (UMAP scatter)",
                            "total_comments": int(len(data["df"])),
                            "detected_clusters": int(len(data["summary"])),
                            "top_topics": data["summary"][["id", "keywords", "participants", "polarization"]]
                                .head(5).to_dict("records")
                        }
                        with st.spinner("AI is explaining this chart..."):
                            st.session_state["ai_explain_map_text"] = ai_explain_visual(client, ctx)

                    if "ai_explain_map_text" in st.session_state:
                        st.caption(st.session_state["ai_explain_map_text"])

        with col_m2:
            avg_pol = float(np.average(
                data['summary']['polarization'].astype(float),
                weights=data['summary']['participants'].astype(float)
            ))

            st.plotly_chart(viz.plot_temperature_gauge(avg_pol), use_container_width=True)
            st.metric("Total Comments", int(len(data["df"])))
            st.metric("Detected Clusters", int(len(data["summary"])))

            # AI EXPLAIN (GAUGE)
            if client is not None:
                if st.button("AI: Explain Polarization Gauge", key="ai_explain_gauge"):
                    ctx = {
                        "chart": "Polarization Gauge (0..10)",
                        "value": round(avg_pol, 2),
                        "note": "This is a weighted average across clusters (weighted by participants).",
                        "cluster_polarizations": data["summary"][["id", "participants", "polarization"]]
                            .sort_values("polarization", ascending=False).head(8).to_dict("records")
                    }
                    with st.spinner("AI is explaining the gauge..."):
                        st.session_state["ai_explain_gauge_text"] = ai_explain_visual(client, ctx)

                if "ai_explain_gauge_text" in st.session_state:
                    st.caption(st.session_state["ai_explain_gauge_text"])

    with tab_network:
        if 'embeddings' in data:
            fig_net = viz.plot_topic_network(data['df'], data['summary'], data['embeddings'])
            st.plotly_chart(fig_net, use_container_width=True)

            if client is not None:
                if st.button("AI: Explain Topic Network", key="ai_explain_network"):
                    ctx = {
                        "chart": "Topic Inter-Connections (semantic proximity network)",
                        "how_built": "Edges shown when centroid cosine similarity > 0.5",
                        "nodes": data["summary"][["id", "keywords", "participants"]].head(10).to_dict("records")
                    }
                    with st.spinner("AI is explaining the network..."):
                        st.session_state["ai_explain_network_text"] = ai_explain_visual(client, ctx)

                if "ai_explain_network_text" in st.session_state:
                    st.caption(st.session_state["ai_explain_network_text"])
        else:
            st.warning("Embeddings not found. Network graph unavailable.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # --- НИЖНИЙ БЛОК: КАРТОЧКИ ТЕМ ---
    st.markdown("### DETECTED TOPIC CLUSTERS")

    for _, row in data['summary'].iterrows():
        with st.expander(f"🔹 TOPIC #{row['id']}: {row['keywords']}"):
            c1, c2, c3 = st.columns([1, 2, 1])

            with c1:
                st.metric("Upvotes", int(row['upvotes']))
                st.metric("Participants", int(row['participants']))
                st.metric("Polarization", f"{float(row['polarization']):.2f}/10")

            with c2:
                st.markdown("**Core Message:**")
                st.markdown(f"""
                    <div style="
                        background-color: #0a051a; 
                        padding: 15px; 
                        border-radius: 10px; 
                        border: 1px solid #2d1b4e;
                        height: 160px; 
                        overflow-y: auto;
                        color: #e0e0e0;
                        font-size: 14px;
                        line-height: 1.5;
                    ">
                        {row['core']}
                    </div>
                """, unsafe_allow_html=True)

            with c3:
                st.write("")
                st.write("")
                if st.button(f"Analyze Topic #{row['id']} →", key=f"go_{row['id']}", use_container_width=True):
                    st.session_state.selected_topic = row['id']
                    st.session_state.page = 'detail'
                    st.rerun()
#СТРАНИЦА 3: DETAIL
elif st.session_state.page == 'detail':
    data = st.session_state.data
    topic_id = st.session_state.selected_topic
    viz = Visualizer()
    client = get_ai_client()

    row = data['summary'][data['summary']['id'] == topic_id].iloc[0]
    topic_comments = data['df'][data['df']['cluster'] == topic_id].copy()

    st.markdown(f"### DEEP DIVE: TOPIC #{topic_id}")
    st.markdown(f"**Keywords:** {row['keywords']}")

    # METRIC CARDS
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="analysis-card"><b>Upvotes</b><br><span style="color:#0066ff;font-size:24px;">{int(row["upvotes"])}</span></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="analysis-card"><b>Participants</b><br><span style="color:#0066ff;font-size:24px;">{int(row["participants"])}</span></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="analysis-card"><b>Polarization</b><br><span style="color:#0066ff;font-size:24px;">{float(row["polarization"]):.2f}/10</span></div>',
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f'<div class="analysis-card"><b>Avg Sentiment</b><br><span style="color:#0066ff;font-size:24px;">{float(row["sentiment"]):.2f}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_text, col_viz = st.columns([1, 1])

    # ---------- TEXT PANEL ----------
    with col_text:
        st.markdown('<div class="analysis-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("CORE MESSAGE")
        st.write(row['core'])

        st.subheader("MOST POPULAR")
        st.write(row['popular'])

        st.subheader("DISSENTING VOICE")
        if float(row['polarization']) > 5:
            st.markdown(
                f'<p style="color:#ff4b4b; border-left: 2px solid #ff4b4b; padding-left: 10px;">{row["dissent"]}</p>',
                unsafe_allow_html=True
            )
        else:
            st.write(row['dissent'])
        st.markdown('</div>', unsafe_allow_html=True)

        # AI: plain-language explanation
        if client is not None:
            if st.button("AI: Explain this topic in simple words", key=f"ai_explain_topic_{topic_id}"):
                with st.spinner("AI is explaining this topic..."):
                    c = topic_comments.dropna(subset=["body_clean"]).copy()
                    c["body_clean"] = c["body_clean"].astype(str)

                    top = c.sort_values("score", ascending=False).head(25)["body_clean"].tolist()
                    rnd = c.sample(min(25, len(c)), random_state=7)["body_clean"].tolist() if len(c) > 0 else []
                    texts = top + rnd

                    meta = (
                        f"Topic id: {int(topic_id)}\n"
                        f"Keywords: {row['keywords']}\n"
                        f"Participants: {int(row['participants'])}\n"
                        f"Polarization: {float(row['polarization']):.2f}/10"
                    )

                    st.session_state[f"ai_explain_topic_text_{topic_id}"] = ai_explain_topic_plain(
                        client, texts, meta=meta
                    )

            exp_key = f"ai_explain_topic_text_{topic_id}"
            if exp_key in st.session_state:
                st.caption(st.session_state[exp_key])

    # ---------- VISUALS PANEL ----------
    with col_viz:
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Distribution", "☁️ Word Cloud", "🎭 Emotions"])

        with viz_tab1:
            st.plotly_chart(viz.plot_sentiment_distribution(topic_comments), use_container_width=True)
            if client is not None:
                if st.button("AI: Explain Sentiment Distribution", key=f"ai_explain_dist_{topic_id}"):
                    ctx = {
                        "chart": "Sentiment Distribution Histogram",
                        "topic_id": int(topic_id),
                        "participants": int(row["participants"]),
                        "avg_sentiment": float(row["sentiment"]),
                        "note": "Histogram shows sentiment polarity (negative to positive) across comments in this topic."
                    }
                    with st.spinner("AI is explaining the chart..."):
                        st.session_state[f"ai_explain_dist_text_{topic_id}"] = ai_explain_visual(client, ctx)
                k = f"ai_explain_dist_text_{topic_id}"
                if k in st.session_state:
                    st.caption(st.session_state[k])

        with viz_tab2:
            st.plotly_chart(viz.plot_word_cloud(data['df'], topic_id), use_container_width=True)
            if client is not None:
                if st.button("AI: Explain Word Cloud", key=f"ai_explain_wc_{topic_id}"):
                    ctx = {
                        "chart": "TF-IDF Word Cloud",
                        "topic_id": int(topic_id),
                        "note": "Words are sized by TF-IDF importance within this topic."
                    }
                    with st.spinner("AI is explaining the chart..."):
                        st.session_state[f"ai_explain_wc_text_{topic_id}"] = ai_explain_visual(client, ctx)
                k = f"ai_explain_wc_text_{topic_id}"
                if k in st.session_state:
                    st.caption(st.session_state[k])

        with viz_tab3:
            fig_radar = viz.plot_sentiment_radar(data['df'], topic_id)
            st.plotly_chart(fig_radar, use_container_width=True)
            if client is not None:
                if st.button("AI: Explain Emotions Radar", key=f"ai_explain_radar_{topic_id}"):
                    ctx = {
                        "chart": "Emotion Radar (NRCLex)",
                        "topic_id": int(topic_id),
                        "note": "Radar shows relative emotion composition (fear/anger/joy/etc.) in this topic."
                    }
                    with st.spinner("AI is explaining the chart..."):
                        st.session_state[f"ai_explain_radar_text_{topic_id}"] = ai_explain_visual(client, ctx)
                k = f"ai_explain_radar_text_{topic_id}"
                if k in st.session_state:
                    st.caption(st.session_state[k])

        # Insight
        if float(row['polarization']) > 6:
            st.error("Highly controversial topic.")
        else:
            st.success("Stable consensus.")

    if st.button("← Back to Overview"):
        st.session_state.page = 'overview'
        st.rerun()
