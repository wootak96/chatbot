"""General-purpose chat node — answers off-domain questions without RAG.

Used in two situations (LangGraph cyclic routing):
  1. query_analyze classifies the intent as `general` upfront → straight here.
  2. The retrieval cycle (hybrid_retrieve ↔ self_check) exhausts retries and
     still has no relevant evidence → escape to here as a graceful fallback.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import render_history, truncate_history
from app.graph.state import RAGState
from app.services.llm_factory import get_generator_llm


async def general_chat(state: RAGState) -> dict:
    query = state.get("resolved_query") or state["current_query"]
    history = truncate_history(state.get("messages", [])[:-1])
    prompt = prompts.GENERAL_CHAT.format(
        history=render_history(history),
        query=query,
    )
    response = await get_generator_llm().ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "sources": [],
        PROGRESS_KEY: "",
    }
