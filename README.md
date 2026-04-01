# Agentic RAG — Streamlit App

A single-turn Agentic RAG system powered by LangChain + GPT-4o (via OpenRouter).

## Architecture

```
User query
    │
    ▼
PDF uploaded? ──Yes──► RAG (FAISS + HuggingFace embeddings)
    │                        │
    │                  Answer found? ──Yes──► Return to user
    │                        │
    │                        No
    ▼                        ▼
    No             LangChain ReAct Agent
                   (decides which tool)
                   ┌──────────┬──────────┐
               Wikipedia   Tavily    ArXiv
                   └──────────┴──────────┘
                              │
                              ▼
                    Response + source + URLs
```

## Local Setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your real keys
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub (secrets.toml is gitignored).
2. Connect repo on [share.streamlit.io](https://share.streamlit.io).
3. In **Settings → Secrets**, paste:

```toml
OPENROUTER_API_KEY = "sk-or-..."
HF_API_KEY         = "hf_..."
TAVILY_API_KEY     = "tvly-..."
```

4. Set **Main file path** → `app.py`
5. Deploy.

## API Keys Required

| Key | Where to get |
|-----|-------------|
| `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai) |
| `HF_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `TAVILY_API_KEY` | [tavily.com](https://tavily.com) |

## Notes

- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (runs locally on CPU — no HF API calls needed for embeddings, but the key is accepted for future gated models).
- FAISS index is rebuilt each session (stateless — Streamlit Cloud doesn't persist files).
- Agent uses a ReAct loop with max 6 iterations before returning best answer.
