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
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool

from rag_engine import query_rag


# ── OpenRouter config ─────────────────────────────────────────────────────────
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL_NAME      = "openai/gpt-4o"

# ── ReAct prompt (no hub dependency) ─────────────────────────────────────────
REACT_PROMPT = PromptTemplate.from_template("""Answer the following question as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Always mention which tool you used.

Begin!

Question: {input}
Thought:{agent_scratchpad}""")


def _build_llm(openrouter_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=openrouter_key,
        openai_api_base=OPENROUTER_BASE,
        temperature=0.2,
    )


def _build_tools(tavily_key: str) -> list:
    os.environ["TAVILY_API_KEY"] = tavily_key

    # Wikipedia
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    )
    wiki_tool = Tool(
        name="Wikipedia",
        func=wiki.run,
        description=(
            "Useful for encyclopedic or background knowledge questions. "
            "Use for factual, well-established topics."
        ),
    )

    # Tavily
    tavily_search = TavilySearchResults(max_results=3)
    tavily_tool = Tool(
        name="Tavily",
        func=lambda q: str(tavily_search.invoke(q)),
        description=(
            "Useful for current events, recent news, or any question requiring "
            "up-to-date web information. Also used as a fallback when other tools fail."
        ),
    )

    # ArXiv
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    )
    arxiv_tool = Tool(
        name="ArXiv",
        func=arxiv.run,
        description=(
            "Useful for academic research, scientific papers, machine learning, "
            "physics, math, or any research-oriented question."
        ),
    )

    return [wiki_tool, tavily_tool, arxiv_tool]


def _rag_answer(context: str, query: str, llm) -> str:
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the following context to answer the question.\n"
        "If the context does not contain enough information, respond with: NOT_FOUND\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def _extract_urls(text: str) -> list[str]:
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

    # Detect which tool was used
    source = "agent"
    urls   = []
    if steps:
        last_action, last_observation = steps[-1]
        tool_name = getattr(last_action, "tool", "agent").lower()
        source    = tool_name if tool_name in {"wikipedia", "tavily", "arxiv"} else "agent"
        urls      = _extract_urls(str(last_observation))

    return {"source": source, "answer": answer, "urls": urls[:3]}
