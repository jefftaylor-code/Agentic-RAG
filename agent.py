"""
agent.py
LangChain agent that:
  1. Tries RAG (FAISS) if a retriever is available.
  2. If RAG fails or no PDF uploaded → LangChain agent picks from
     Wikipedia, Tavily, ArXiv.
  3. Tavily is used as the fallback when the agent needs current web info.
"""

from __future__ import annotations
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_engine import query_rag


# ── OpenRouter base URL ───────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME      = "openai/gpt-4o"


def _build_llm(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=openrouter_key,
        openai_api_base=OPENROUTER_BASE,
        temperature=0.2,
    )


def _build_tools(tavily_key: str) -> list:
    """Build Wikipedia, Tavily, ArXiv tools."""
    os.environ["TAVILY_API_KEY"] = tavily_key

    wiki_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    )
    wiki = Tool(
        name="Wikipedia",
        func=wiki_tool.run,
        description=(
            "Useful for encyclopedic or background knowledge questions. "
            "Use for factual, well-established topics."
        ),
    )

    tavily = TavilySearch(
        max_results=3,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )
    tavily_tool = Tool(
        name="Tavily",
        func=tavily.invoke,
        description=(
            "Useful for current events, recent news, or any question requiring "
            "up-to-date web information. Also used as a fallback when other tools fail."
        ),
    )

    arxiv_tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    )
    arxiv = Tool(
        name="ArXiv",
        func=arxiv_tool.run,
        description=(
            "Useful for academic research, scientific papers, machine learning, "
            "physics, math, or any research-oriented question."
        ),
    )

    return [wiki, tavily_tool, arxiv]


def _rag_answer(context: str, query: str, llm) -> str:
    """Use the LLM to synthesize a final answer from RAG context."""
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the following context to answer the question.\n"
        "If the context does not contain enough information to answer, respond with: NOT_FOUND\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def _extract_tool_used(agent_output: str, tools_used: list) -> str:
    """Best-effort detection of which tool the agent used."""
    lower = agent_output.lower()
    if "arxiv" in lower:
        return "arxiv"
    if "wikipedia" in lower:
        return "wikipedia"
    if "tavily" in lower:
        return "tavily"
    # fallback
    for t in tools_used:
        if t.lower() in lower:
            return t.lower()
    return "agent"


def _extract_urls(tool_result: str) -> list[str]:
    """Pull URLs from a tool result string."""
    import re
    return re.findall(r'https?://[^\s\'"<>]+', tool_result)


def run_agent(
    query: str,
    retriever,
    openrouter_key: str,
    tavily_key: str,
    hf_key: str,
) -> dict:
    """
    Main entry point.
    Returns dict with keys: source, answer, urls
    """
    llm   = _build_llm(openrouter_key)
    tools = _build_tools(tavily_key)

    # ── Step 1: Try RAG if retriever exists ───────────────────────────────────
    if retriever is not None:
        context = query_rag(retriever, query)
        if context:
            answer = _rag_answer(context, query, llm)
            if "NOT_FOUND" not in answer:
                return {"source": "rag", "answer": answer, "urls": []}
        # RAG didn't find it → fall through to agent

    # ── Step 2: Run LangChain ReAct agent ────────────────────────────────────
    try:
        # Pull a standard ReAct prompt from LangChain hub
        react_prompt = hub.pull("hwchase17/react")
    except Exception:
        # Offline fallback prompt
        react_prompt = PromptTemplate.from_template(
            "Answer the following question using the available tools.\n\n"
            "Tools: {tools}\n"
            "Tool names: {tool_names}\n\n"
            "Question: {input}\n\n"
            "{agent_scratchpad}"
        )

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": query})

    answer     = result.get("output", "No answer found.")
    steps      = result.get("intermediate_steps", [])

    # Determine which tool was actually used
    source = "agent"
    urls   = []
    if steps:
        last_action, last_observation = steps[-1]
        tool_name = getattr(last_action, "tool", "agent").lower()
        source    = tool_name if tool_name in {"wikipedia", "tavily", "arxiv"} else "agent"
        urls      = _extract_urls(str(last_observation))

    return {"source": source, "answer": answer, "urls": urls[:3]}
