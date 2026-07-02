# 🏢 Company RAG — Streamlit · Groq · Tavily · FAISS

RAG platform with live web search enrichment.

```
Documents → Chunker → Embedder ──────────→ FAISS Index
                                                 ↓
Question → Embed → FAISS Search → Top-K Chunks ─┐
                                                  ├─→ Groq (stream) → Answer
Question → Tavily Web Search ──→ Live Results  ──┘
```

## Stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| LLM | Groq `llama-3.3-70b-versatile` (streaming) |
| Web Search | Tavily (optional — toggle per question) |
| Embeddings | `all-MiniLM-L6-v2` (local) |
| Vector Store | FAISS flat L2 |

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/company-rag-groq-tavily
cd company-rag-groq-tavily
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml with your keys

streamlit run app.py
```

## Deploy to Streamlit Cloud (free)

```bash
git init && git add . && git commit -m "init"
git remote add origin https://github.com/YOUR_USERNAME/company-rag-groq-tavily.git
git push -u origin main
```

1. [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo → `app.py`
2. **Settings → Secrets:**
```toml
GROQ_API_KEY   = "gsk_..."
TAVILY_API_KEY = "tvly-..."   # optional
```

## API Keys

| Service | URL | Free tier |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | ✅ Generous free tier |
| Tavily | [tavily.com](https://tavily.com) | ✅ 1,000 searches/month free |

## Features

- **Data tab** — company metrics, document library, add & index docs live
- **Ask AI tab** — streaming Groq answers with chunk inspector
- **🌐 Web search toggle** — Tavily enriches answers with live results per question
- **Multi-turn** — full conversation history per company
- **Save / Load** — persist companies to `companies.json`
