"""History-aware query reformulation.

Runs only on the search branch (intent == question). Reads conversation history
and the current user query, and produces a self-contained Korean query string
suitable for downstream decomposition / rewrite. The output replaces follow-up
references ("그게", "어떻게") and elided subjects with their explicit referents
from prior turns, so query_decompose / query_rewrite never need to see history.

If history is empty (first user turn), the LLM call is skipped and the current
query is passed through unchanged — there is nothing to reform.
"""

from __future__ import annotations

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json, render_history, truncate_history
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def query_reform(state: RAGState) -> dict:
    query = state["current_query"]
    history = truncate_history(state.get("messages", [])[:-1])

    if not history:
        return {
            "resolved_query": query,
            PROGRESS_KEY: "📝 쿼리 재작성... (히스토리 없음, 원문 사용)",
        }

    prompt = prompts.QUERY_REFORM.format(
        history=render_history(history),
        query=query,
    )
    data = await llm_json(get_judge_llm(), prompt)
    reformed = (data.get("reformed_query") or "").strip() or query

    return {
        "resolved_query": reformed,
        PROGRESS_KEY: f"📝 쿼리 재작성... ({reformed})",
    }
