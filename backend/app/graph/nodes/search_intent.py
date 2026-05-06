"""Classify the SHAPE of the ES query needed.

Runs only on the search branch (intent == question/followup). Splits the
follow-up flow three ways:
  - lookup : RRF top-k retrieval (existing path through query_decompose ...)
  - count  : ES _count or size=0 + track_total_hits
  - list   : terms aggregation on title field

Lookup is the default fallback when the LLM gives us anything unexpected.
"""

from __future__ import annotations

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


_VALID = ("lookup", "count", "list")
_LABEL = {
    "lookup": "문서 내용 검색",
    "count": "문서 개수 조회",
    "list": "문서 목록 조회",
}

# `list` is temporarily disabled — title.keyword aggregation produced
# misleading results on the current corpus. The node, prompt, and graph
# wiring stay intact so we can re-enable by flipping this flag.
_DISABLED_INTENTS = {"list"}


async def search_intent(state: RAGState) -> dict:
    query = state.get("resolved_query") or state.get("current_query") or ""
    if not query:
        return {"search_intent": "lookup", PROGRESS_KEY: ""}

    data = await llm_json(
        get_judge_llm(),
        prompts.SEARCH_INTENT_CLASSIFY.format(query=query),
    )
    si = (data.get("search_intent") or "lookup").strip().lower()
    if si not in _VALID or si in _DISABLED_INTENTS:
        si = "lookup"

    return {
        "search_intent": si,
        PROGRESS_KEY: f"🎯 검색 유형 분석... ({_LABEL[si]})",
    }
