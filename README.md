# 🔴 Reddit Thread Intelligence

> Deep semantic analysis of Reddit threads — opinion clusters, emotion mapping, polarization scoring, and engagement patterns.

![Python](https://img.shields.io/badge/Python-3.9%2B-c0392b?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-c0392b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-7b241c?style=flat-square)

---

## What it does

Paste any public Reddit URL and get a full breakdown of the discussion:

- **Semantic clustering** — groups comments into topic clusters using UMAP + HDBSCAN
- **Polarization index** — measures how split each cluster is via two-camp KMeans divergence
- **Emotion radar** — 8-axis NRCLex spectrum (fear, anger, joy, trust, sadness, disgust, surprise, anticipation)
- **Sentiment violin** — distribution of positive/negative/neutral tone per topic
- **Activity timeline** — comment volume and avg score over time (30-min bins)
- **Engagement by depth** — how upvotes change as threads go deeper
- **Top contributors** — who is driving the conversation
- **Controversy map** — upvotes vs replies scatter to find heated comments
- **AI summaries** — optional Groq-powered plain-language explanations per topic or chart

---

## Screenshots

| Landing | Overview | Deep Dive |
|---------|----------|-----------|
| Dark red input screen | UMAP opinion map + cluster cards | Per-topic emotion radar, keywords, sentiment |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/your-username/reddit-thread-intelligence.git
cd reddit-thread-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

---

## Requirements

Create a `requirements.txt` with:

```
streamlit>=1.30
requests
pandas
numpy
torch
sentence-transformers
umap-learn
hdbscan
scikit-learn
plotly
textblob
nrclex
nltk
openai
```

> **Note:** `torch` can be the CPU-only build if you don't have a GPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## AI Features (optional)

The app supports AI-powered summaries via [Groq](https://console.groq.com) (free tier available).

Add your key to Streamlit secrets:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_..."
```

Or set it as an environment variable:

```bash
export GROQ_API_KEY="gsk_..."
```

Without a key the app works fully — AI buttons are simply hidden.

---

## How the parser works

Reddit's default JSON endpoint returns ~25-1000 comments. This app does better:

1. Fetches with `?limit=500&depth=10` to maximize the initial payload
2. Collects all `"more"` continuation tokens from the comment tree
3. Fires parallel requests to `/api/morechildren.json` via `ThreadPoolExecutor(max_workers=5)`
4. Merges everything into a single DataFrame before analysis

In practice this retrieves **several times more comments** than a naive fetch, especially on large threads.

---

## Project structure

```
app.py                  # Full Streamlit application (single file)
requirements.txt        # Python dependencies
.streamlit/
  secrets.toml          # API keys (not committed)
README.md
```

---

## Tech stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Dimensionality reduction | UMAP |
| Clustering | HDBSCAN |
| Polarization | KMeans two-camp + silhouette |
| Emotion analysis | NRCLex |
| Sentiment | TextBlob |
| Visualization | Plotly |
| AI summaries | Groq API (llama-3.3-70b) |

---

## Limitations

- Only works on **public** Reddit posts (no login required)
- Reddit may rate-limit aggressive scraping — the app adds no artificial delays but uses parallel batching responsibly
- Very large threads (10k+ comments) will be partially sampled due to Reddit API caps
- Embeddings run on CPU by default — analysis of 1000+ comments takes ~30–60 seconds

---

## License

MIT — do whatever you want, attribution appreciated.
