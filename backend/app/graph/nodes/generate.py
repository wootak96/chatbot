from __future__ import annotations

from langchain_core.messages import HumanMessage

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import render_docs_full
from app.graph.state import RAGState
from app.services.llm_factory import get_generator_llm


async def generate(state: RAGState) -> dict:
    intent = state.get("intent") or "question"
    query = state.get("resolved_query") or state["current_query"]

    if intent == "chitchat":
        prompt = prompts.CHITCHAT.format(query=query)
        sources: list[dict] = []
    else:
        candidates = state.get("candidates") or []
        if not candidates or not state.get("sufficient", False):
            return {
                "final_answer": "해당 정보를 찾을 수 없습니다.",
                "sources": [],
                PROGRESS_KEY: "",
            }
        prompt = prompts.GENERATE.format(
            query=query, docs=render_docs_full(candidates)
        )
        sources = [
            {"url": d.get("url", ""), "title": d.get("title", "")}
            for d in candidates
        ]

    response = await get_generator_llm().ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "sources": sources,
        PROGRESS_KEY: "",
    }
