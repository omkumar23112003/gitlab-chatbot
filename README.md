# 🦊 GitBot — GitLab Handbook & Direction Chatbot

An AI-powered chatbot that lets you search and ask questions across **GitLab's Handbook** and **Direction** pages using semantic search (embeddings + FAISS) and a free LLM (Groq).

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[ Streamlit UI ]
    │
    ▼
[ Query Embedding ]          ← sentence-transformers/all-MiniLM-L6-v2  (local, free)
    │
    ▼
[ FAISS Vector Search ]      ← Top-10 nearest neighbors (cosine similarity)
    │
    ▼
[ MMR Reranking ]            ← Maximal Marginal Relevance (diversity + relevance)
    │
    ▼
[ Groq LLM API ]             ← llama-3.3-70b-versatile  (free tier)
    │
    ▼
Answer + Source Citations
```

### Pipeline steps

| Step | Script | What it does |
|------|--------|-------------|
| **1. Scrape** | `scraper.py` | Crawls GitLab Handbook & Direction pages, chunks text |
| **2. Embed** | `embed.py` | Generates embeddings, builds FAISS index, saves to disk |
| **3. Chat** | `app.py` | Streamlit UI — loads index, retrieves, generates answers |

---

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.10+
- A **free Groq API key** → [console.groq.com](https://console.groq.com) (no credit card)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/gitlab-chatbot.git
cd gitlab-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

Or create a `.env` file:

```
GROQ_API_KEY=gsk_your_key_here
```

### 4. Scrape GitLab pages

```bash
python scraper.py
```

This crawls up to 500 pages from:
- `https://handbook.gitlab.com/`
- `https://about.gitlab.com/direction/`

Saves to `data/scraped_pages.json` and `data/chunks.json`.

> ⏱️ Takes ~5–10 minutes depending on your connection.

### 5. Build the FAISS index

```bash
python embed.py
```

Downloads the embedding model (~80MB, one time), generates embeddings for all chunks, saves:
- `data/faiss.index`
- `data/metadata.json`

> ⏱️ Takes ~1–5 minutes on CPU.

### 6. Run the chatbot

```bash
streamlit run app.py
```

Open `http://localhost:8501` → enter your Groq API key in the sidebar → start chatting!

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push this repo to **GitHub** (make sure `data/` files are included or re-run steps 4–5 after deploy)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Go to **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Click **Deploy** 🚀

> **Note:** You need to commit `data/faiss.index` and `data/metadata.json` to your repo for the deployed app to work. These files can be up to ~100MB — use [Git LFS](https://git-lfs.github.com/) if needed.

---

## 🆓 Free AI APIs Used

| Service | What for | Free tier |
|---------|----------|-----------|
| [Groq](https://console.groq.com) | LLM generation (Llama 3.3 70B) | ✅ Free, no credit card |
| [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Local embeddings | ✅ Runs locally, always free |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector search | ✅ Runs locally, always free |

### Other free LLM options you can use instead of Groq

| Provider | Models | How to switch |
|----------|--------|--------------|
| [OpenRouter](https://openrouter.ai) | 29+ free models (Llama, Mistral, Gemma) | Change base URL in `rag_engine.py` |
| [Mistral AI](https://mistral.ai) | Mistral Small, Codestral | Replace Groq client |
| [NVIDIA NIM](https://build.nvidia.com) | Llama 3.3 70B | OpenAI-compatible API |
| [Hugging Face](https://huggingface.co/inference-api) | 300+ models | Use `InferenceClient` |

---

## 📁 Project Structure

```
gitlab-chatbot/
├── scraper.py          # Web crawler — GitLab Handbook & Direction
├── embed.py            # Embedding generator + FAISS index builder
├── rag_engine.py       # RAG pipeline: FAISS retrieval + MMR + Groq generation
├── app.py              # Streamlit chat UI
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .streamlit/
│   └── secrets.toml    # Streamlit Cloud secrets template
├── data/               # Generated data (created by scripts)
│   ├── scraped_pages.json
│   ├── chunks.json
│   ├── faiss.index
│   └── metadata.json
└── README.md
```

---

## ⚙️ Configuration

Key settings you can tune:

| File | Variable | Default | Description |
|------|----------|---------|-------------|
| `scraper.py` | `MAX_PAGES` | 500 | Max pages to crawl |
| `scraper.py` | `CHUNK_SIZE` | 400 | Words per chunk |
| `scraper.py` | `CHUNK_OVERLAP` | 60 | Overlap between chunks |
| `embed.py` | `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `rag_engine.py` | `TOP_K` | 10 | FAISS candidates |
| `rag_engine.py` | `MMR_K` | 5 | Final chunks after MMR |
| `rag_engine.py` | `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |

---


MIT
