import streamlit as st
import os
import shutil
import tempfile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
import wikipedia
import arxiv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic RAG", page_icon="🔍", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg: #0d0f14; --surface: #151820; --border: #252a35;
    --accent: #4f8ef7; --accent2: #7c5cf7; --green: #3ecf8e;
    --text: #e2e8f0; --muted: #64748b;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important; color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1 { font-family: 'Space Mono', monospace; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Mono', monospace; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important;
    padding: 0.6rem 1.5rem !important; letter-spacing: 0.05em !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text) !important;
}
[data-testid="stFileUploader"] {
    background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 10px !important;
}
[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME      = "openai/gpt-4o"

AGENT_SYSTEM_PROMPT = """You are a research assistant with three tools:
1. wikipedia_search — for biographical info, historical facts, established concepts, or any encyclopedic topic.
2. arxiv_search — for academic papers, recent ML/AI/science/engineering research.
3. tavily_search — for current events, news, weather, or anything time-sensitive.

Always pick the most appropriate tool. Prefer wikipedia_search for factual/biographical queries,
arxiv_search for research topics, and only use tavily_search when real-time information is needed."""

# ── Session state ─────────────────────────────────────────────────────────────
if "source_info" not in st.session_state:
    st.session_state.source_info = {"source": None, "urls": []}

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 Agentic RAG")
st.markdown(
    '<p style="color:#64748b;font-size:0.95rem;">'
    "Ask anything. If you upload a PDF, RAG is tried first. "
    "Otherwise the agent picks Wikipedia, ArXiv, or Tavily automatically.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_secret(key, fallback_session_key):
    try:
        return st.secrets[key]
    except Exception:
        return st.session_state.get(fallback_session_key, "")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    with st.expander("API Keys", expanded=False):
        st.markdown('<p style="font-size:0.78rem;color:#64748b;">Keys are read from st.secrets first. Enter here to override.</p>', unsafe_allow_html=True)
        openrouter_input = st.text_input("OpenRouter API Key", value=st.session_state.get("openrouter_key", ""), type="password", placeholder="sk-or-...")
        tavily_input     = st.text_input("Tavily API Key",     value=st.session_state.get("tavily_key", ""),     type="password", placeholder="tvly-...")
        if st.button("Save Keys"):
            st.session_state["openrouter_key"] = openrouter_input
            st.session_state["tavily_key"]     = tavily_input
            st.success("Keys saved.")
    st.markdown("---")
    st.markdown("### 🛠 Tools")
    st.markdown("- 📄 **RAG** — PDF retrieval\n- 📚 **Wikipedia** — Encyclopedic\n- 🔬 **ArXiv** — Academic\n- 🌐 **Tavily** — Web / current")

# ── Resolve keys ──────────────────────────────────────────────────────────────
openrouter_key = get_secret("OPENROUTER_API_KEY", "openrouter_key")
tavily_key     = get_secret("TAVILY_API_KEY",      "tavily_key")

missing = []
if not openrouter_key: missing.append("OpenRouter API Key")
if not tavily_key:     missing.append("Tavily API Key")
if missing:
    st.warning(f"Missing keys: {', '.join(missing)}. Add them in ⚙️ Settings.")
    st.stop()

os.environ["OPENAI_API_KEY"]  = openrouter_key
os.environ["TAVILY_API_KEY"]  = tavily_key

# ── LLM + Embeddings ──────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_base=OPENROUTER_BASE,
    openai_api_key=openrouter_key,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openrouter_key,
    openai_api_base=OPENROUTER_BASE,
)

# ── Tavily client ─────────────────────────────────────────────────────────────
tavily_client = TavilyClient(api_key=tavily_key)

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def tavily_search(query: str) -> str:
    """Search the web for current events, news, weather, or real-time information."""
    try:
        response = tavily_client.search(query)
        results  = response.get("results", [])
        if not results:
            return "NO_RESULTS: No search results found."
        formatted, urls = [], []
        for i, r in enumerate(results[:5], 1):
            formatted.append(f"{i}. {r.get('title','')}\n{r.get('content','')}\n")
            urls.append(r.get("url", ""))
        st.session_state.source_info = {"source": "Tavily", "urls": urls}
        return "\n".join(formatted)
    except Exception as e:
        return f"ERROR: {e}"

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for encyclopedic information: people, places, history, science concepts."""
    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(query, results=3)
        if not results:
            results = wikipedia.search(" ".join(query.split()[:3]), results=3)
        if not results:
            return "NO_RESULTS: No Wikipedia articles found."
        for name in results:
            try:
                page    = wikipedia.page(name)
                summary = wikipedia.summary(name, sentences=4)
                st.session_state.source_info = {"source": "Wikipedia", "urls": [page.url]}
                return f"Wikipedia: {page.title}\n\n{summary}\n\nSource: {page.url}"
            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        page    = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=4)
                        st.session_state.source_info = {"source": "Wikipedia", "urls": [page.url]}
                        return f"Wikipedia: {page.title}\n\n{summary}\n\nSource: {page.url}"
                    except Exception:
                        continue
            except wikipedia.exceptions.PageError:
                continue
        return "NO_RESULTS: Wikipedia pages not accessible."
    except Exception as e:
        return f"ERROR: {e}"

@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for academic papers in ML, AI, physics, math, or engineering."""
    try:
        client       = arxiv.Client()
        search       = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
        results_list = list(client.results(search))
        if not results_list:
            simplified   = " ".join(query.replace('"', "").split()[:5])
            search       = arxiv.Search(query=simplified, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
            results_list = list(client.results(search))
        if not results_list:
            return "NO_RESULTS: No papers found on ArXiv."
        formatted, urls = [], []
        for i, r in enumerate(results_list, 1):
            authors = ", ".join([a.name for a in r.authors[:3]])
            formatted.append(
                f"{i}. {r.title}\nAuthors: {authors}\n"
                f"Published: {r.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {r.summary[:300]}...\n"
            )
            urls.append(r.entry_id)
        st.session_state.source_info = {"source": "ArXiv", "urls": urls}
        return "\n".join(formatted)
    except Exception as e:
        return f"ERROR: {e}"

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = create_react_agent(llm, [tavily_search, wikipedia_search, arxiv_search])

# ── PDF Upload ────────────────────────────────────────────────────────────────
st.markdown("### 📄 Document Upload *(optional)*")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

retriever = None
if uploaded_file:
    if st.session_state.get("last_uploaded") != uploaded_file.name:
        with st.spinner("Building FAISS index…"):
            try:
                temp_dir  = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader   = PyPDFLoader(file_path)
                docs     = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts    = splitter.split_documents(docs)
                db       = FAISS.from_documents(texts, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 5})
                st.session_state["retriever"]     = retriever
                st.session_state["last_uploaded"] = uploaded_file.name
                shutil.rmtree(temp_dir)
                st.success(f"✓ Indexed: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
    else:
        retriever = st.session_state.get("retriever")
        st.success(f"✓ Indexed: {uploaded_file.name}")
else:
    st.session_state.pop("retriever", None)
    st.session_state.pop("last_uploaded", None)

st.markdown("---")

# ── Query input ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area("Your question", placeholder="e.g. What are the latest transformer architectures for RAG?", height=100, label_visibility="collapsed")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    answer_found_in_pdf = False

    # Step 1 — RAG
    if retriever:
        try:
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain

            rag_prompt = ChatPromptTemplate.from_template(
                "You must answer using ONLY the context below.\n"
                "If the context does not contain sufficient information, respond with exactly: INSUFFICIENT_CONTEXT\n\n"
                "Context: {context}\nQuestion: {input}\n\nAnswer:"
            )
            document_chain  = create_stuff_documents_chain(llm, rag_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            rag_result      = retrieval_chain.invoke({"input": query})
            rag_answer      = rag_result["answer"].strip()

            if rag_answer != "INSUFFICIENT_CONTEXT" and len(rag_answer) > 20:
                answer_found_in_pdf = True
                st.subheader("📄 Answer from Uploaded PDF")
                st.success("Source: PDF Documents")
                st.write(rag_answer)
                if rag_result.get("context"):
                    with st.expander("📋 Source Documents"):
                        for i, doc in enumerate(rag_result["context"], 1):
                            page  = doc.metadata.get("page", "N/A")
                            fname = os.path.basename(doc.metadata.get("source", "PDF"))
                            st.markdown(f"**Source {i}:** `{fname}` — Page {page}")
                            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.markdown("---")
            else:
                st.info("No relevant info found in PDF. Searching external sources…")
        except Exception as e:
            st.info(f"RAG error ({e}). Searching external sources…")

    # Step 2 — Agent
    if not answer_found_in_pdf:
        with st.spinner("Agent thinking…"):
            try:
                response = agent.invoke({
                    "messages": [
                        SystemMessage(content=AGENT_SYSTEM_PROMPT),
                        HumanMessage(content=query),
                    ]
                })

                response_text = ""
                for msg in reversed(response["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        response_text = msg.content
                        break

                tool_used = None
                for msg in response["messages"]:
                    if hasattr(msg, "name") and msg.name in ["tavily_search", "wikipedia_search", "arxiv_search"]:
                        tool_used = msg.name

                urls = st.session_state.source_info.get("urls", [])

                if tool_used == "arxiv_search":
                    icon, header, source = "🔬", "Academic Research Response", "ArXiv"
                elif tool_used == "wikipedia_search":
                    icon, header, source = "📚", "Encyclopedia Response", "Wikipedia"
                elif tool_used == "tavily_search":
                    icon, header, source = "🌐", "Web Search Response", "Tavily"
                else:
                    icon, header, source = "🤖", "Agent Response", "Agent"

                st.subheader(f"{icon} {header}")
                st.success(f"Source: {source}")
                st.write(response_text)
                if urls:
                    with st.expander("🔗 Sources"):
                        for url in urls:
                            st.markdown(f"- {url}")

            except Exception as e:
                st.error(f"Agent error: {e}")
