"""
agent.py - compatible with langchain 0.2.x on any Python version
"""

from __future__ import annotations
import os
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from rag_engine import query_rag

# ── OpenRouter config ─────────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME      = "openai/gpt-4o"

# ── ReAct prompt ──────────────────────────────────────────────────────────────
REACT_PROMPT = PromptTemplate.from_template(
    "Answer the following question as best you can.\n"
    "You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer. Always mention which tool you used.\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)


def _build_llm(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=openrouter_key,
        openai_api_base=OPENROUTER_BASE,
        temperature=0.2,
    )


def _build_tools(tavily_key: str) -> list:
    os.environ["TAVILY_API_KEY"] = tavily_key

    wiki_tool = Tool(
        name="Wikipedia",
        func=WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
        ).run,
        description=(
            "Useful for encyclopedic or background knowledge. "
            "Use for factual, well-established topics."
        ),
    )

    tavily_tool = Tool(
        name="Tavily",
        func=lambda q: str(TavilySearchResults(max_results=3).invoke(q)),
        description=(
            "Useful for current events, recent news, or up-to-date web information. "
            "Use as fallback when other tools are insufficient."
        ),
    )

    arxiv_tool = Tool(
        name="ArXiv",
        func=ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
        ).run,
        description=(
            "Useful for academic research, scientific papers, ML, "
            "physics, math, or research-oriented questions."
        ),
    )

    return [wiki_tool, tavily_tool, arxiv_tool]


def _rag_answer(context: str, query: str, llm) -> str:
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the context is insufficient, respond with exactly: NOT_FOUND\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def _extract_urls(text: str) -> list:
    return re.findall(r'https?://[^\s\'"<>\]]+', text)


def run_agent(
    query: str,
    retriever,
    openrouter_key: str,
    tavily_key: str,
    hf_key: str,
) -> dict:
    llm   = _build_llm(openrouter_key)
    tools = _build_tools(tavily_key)

    # ── Step 1: Try RAG ───────────────────────────────────────────────────────
    if retriever is not None:
        context = query_rag(retriever, query)
        if context:
            answer = _rag_answer(context, query, llm)
            if "NOT_FOUND" not in answer:
                return {"source": "rag", "answer": answer, "urls": []}

    # ── Step 2: ReAct agent ───────────────────────────────────────────────────
    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": query})
    answer = result.get("output", "No answer found.")
    steps  = result.get("intermediate_steps", [])

    source = "agent"
    urls   = []
    if steps:
        last_action, last_observation = steps[-1]
        tool_name = getattr(last_action, "tool", "agent").lower()
        source    = tool_name if tool_name in {"wikipedia", "tavily", "arxiv"} else "agent"
        urls      = _extract_urls(str(last_observation))

    return {"source": source, "answer": answer, "urls": urls[:3]}
