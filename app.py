import streamlit as st
import tempfile
import os
from pathlib import Path

from rag_engine import build_faiss_index, query_rag
from agent import run_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #151820;
    --border:    #252a35;
    --accent:    #4f8ef7;
    --accent2:   #7c5cf7;
    --green:     #3ecf8e;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --danger:    #f87171;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1 { font-family: 'Space Mono', monospace; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Mono', monospace; }

/* Cards */
.rag-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.tag-rag    { background: #1a3a2a; color: var(--green);  border: 1px solid #2d6a4f; }
.tag-wiki   { background: #1a2a3a; color: #60a5fa;       border: 1px solid #1e40af; }
.tag-tavily { background: #2a1a3a; color: #c084fc;       border: 1px solid #6d28d9; }
.tag-arxiv  { background: #3a2a1a; color: #fb923c;       border: 1px solid #92400e; }
.tag-agent  { background: #2a2a1a; color: #fbbf24;       border: 1px solid #78350f; }

.answer-box {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.7;
}

.source-box {
    background: #0f0f1a;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
    margin-top: 0.5rem;
}

.source-box a { color: var(--accent); text-decoration: none; }
.source-box a:hover { text-decoration: underline; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 1.5rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Status / info boxes */
.stAlert { border-radius: 8px !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_secret(key: str, fallback_session_key: str) -> str:
    """Return from st.secrets first, then session_state fallback."""
    try:
        return st.secrets[key]
    except Exception:
        return st.session_state.get(fallback_session_key, "")


def tag_html(label: str, cls: str) -> str:
    return f'<span class="tag {cls}">{label}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Agentic RAG")
    st.markdown('<p style="color:var(--muted);font-size:0.82rem;margin-top:-0.5rem;">Powered by LangChain + GPT-4o</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Settings expander ────────────────────────────────────────────────────
    with st.expander("⚙️ API Settings", expanded=False):
        st.markdown('<p style="font-size:0.78rem;color:var(--muted);">Keys are read from <code>st.secrets</code> first. Enter here to override.</p>', unsafe_allow_html=True)

        openrouter_input = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.get("openrouter_key", ""),
            type="password",
            placeholder="sk-or-...",
            key="openrouter_input",
        )
        hf_input = st.text_input(
            "HuggingFace API Key",
            value=st.session_state.get("hf_key", ""),
            type="password",
            placeholder="hf_...",
            key="hf_input",
        )
        tavily_input = st.text_input(
            "Tavily API Key",
            value=st.session_state.get("tavily_key", ""),
            type="password",
            placeholder="tvly-...",
            key="tavily_input",
        )
        if st.button("Save Keys", key="save_keys"):
            st.session_state["openrouter_key"] = openrouter_input
            st.session_state["hf_key"] = hf_input
            st.session_state["tavily_key"] = tavily_input
            st.success("Keys saved for this session.")

    st.markdown("---")

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown("### 📄 Document Upload")
    st.markdown('<p style="font-size:0.8rem;color:var(--muted);">Upload a PDF to enable RAG retrieval.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            with st.spinner("Building FAISS index…"):
                hf_key = get_secret("HF_API_KEY", "hf_key")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    retriever = build_faiss_index(tmp_path, hf_key)
                    st.session_state["retriever"] = retriever
                    st.session_state["last_uploaded"] = uploaded_file.name
                    st.success(f"✓ Indexed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                finally:
                    os.unlink(tmp_path)
        else:
            st.success(f"✓ Indexed: {uploaded_file.name}")
    else:
        st.session_state.pop("retriever", None)
        st.session_state.pop("last_uploaded", None)
        st.markdown('<p style="font-size:0.78rem;color:var(--muted);">No document loaded — agent will use web tools only.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tool legend ───────────────────────────────────────────────────────────
    st.markdown("### 🛠 Available Tools")
    st.markdown("""
<div style="font-size:0.78rem;line-height:2;">
  <span class="tag tag-rag">RAG</span> FAISS + PDF retrieval<br>
  <span class="tag tag-wiki">Wikipedia</span> Encyclopedic info<br>
  <span class="tag tag-tavily">Tavily</span> Web / current info<br>
  <span class="tag tag-arxiv">ArXiv</span> Academic papers
</div>
""", unsafe_allow_html=True)


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown("# Agentic RAG")
st.markdown(
    '<p style="color:var(--muted);font-size:0.95rem;max-width:680px;">'
    "Ask anything. If you've uploaded a PDF, the agent tries RAG first. "
    "Otherwise it autonomously selects the best tool — Wikipedia, Tavily, or ArXiv."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Query input ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Your question",
        placeholder="e.g. What are the latest transformer architectures for RAG?",
        height=100,
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    openrouter_key = get_secret("OPENROUTER_API_KEY", "openrouter_key")
    hf_key         = get_secret("HF_API_KEY",         "hf_key")
    tavily_key     = get_secret("TAVILY_API_KEY",      "tavily_key")

    missing = []
    if not openrouter_key: missing.append("OpenRouter API Key")
    if not tavily_key:     missing.append("Tavily API Key")
    if missing:
        st.error(f"Missing keys: {', '.join(missing)}. Add them in ⚙️ API Settings.")
        st.stop()

    retriever = st.session_state.get("retriever", None)

    with st.spinner("Agent thinking…"):
        try:
            result = run_agent(
                query=query,
                retriever=retriever,
                openrouter_key=openrouter_key,
                tavily_key=tavily_key,
                hf_key=hf_key,
            )
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    # ── Render result ─────────────────────────────────────────────────────────
    source  = result.get("source", "unknown")
    answer  = result.get("answer", "No answer returned.")
    urls    = result.get("urls", [])

    tag_map = {
        "rag":       ("tag-rag",    "RAG — PDF"),
        "tavily":    ("tag-tavily", "Tavily Search"),
        "wikipedia": ("tag-wiki",   "Wikipedia"),
        "arxiv":     ("tag-arxiv",  "ArXiv"),
        "agent":     ("tag-agent",  "Agent"),
    }
    tag_cls, tag_label = tag_map.get(source, ("tag-agent", source.upper()))

    st.markdown(f"""
<div class="rag-card">
  {tag_html(tag_label, tag_cls)}
  <div class="answer-box">{answer}</div>
  {''.join([f'<div class="source-box">🔗 <a href="{u}" target="_blank">{u}</a></div>' for u in urls]) if urls else ''}
</div>
""", unsafe_allow_html=True)
