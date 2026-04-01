"""
rag_engine.py
Build a FAISS vector index from a PDF and query it.
"""

from __future__ import annotations
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4


def build_faiss_index(pdf_path: str, hf_api_key: str):
    """
    Load a PDF, split into chunks, embed with HuggingFace, store in FAISS.
    Returns a LangChain retriever.
    """
    # HF embeddings can run locally (no API key needed for sentence-transformers)
    # but we accept the key in case the user wants a gated model later.
    if hf_api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

    loader   = PyPDFLoader(pdf_path)
    docs     = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever


def query_rag(retriever, query: str) -> Optional[str]:
    """
    Query the FAISS retriever.
    Returns combined page content if docs found, else None.
    """
    docs = retriever.invoke(query)
    if not docs:
        return None

    combined = "\n\n".join(d.page_content for d in docs)
    # Return None if context is too sparse (very short chunks = likely noise)
    if len(combined.strip()) < 50:
        return None
    return combined
