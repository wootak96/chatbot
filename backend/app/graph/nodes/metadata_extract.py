from __future__ import annotations

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def metadata_extract(state: RAGState) -> dict:
    query = state.get("resolved_query") or state["current_query"]
    if state.get("intent") == "chitchat":
        return {"metadata_filters": {}, PROGRESS_KEY: ""}

    data = await llm_json(get_judge_llm(), prompts.METADATA_EXTRACT.format(query=query))
    filters = {
        k: data.get(k)
        for k in ("source", "category", "date_range")
        if data.get(k)
    }
    return {
        "metadata_filters": filters,
        PROGRESS_KEY: "🏷️  메타데이터 필터 추출 중...",
    }
