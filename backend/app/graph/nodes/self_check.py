from __future__ import annotations

from app import prompts
from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json, render_docs_brief
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def self_check(state: RAGState) -> dict:
    if state.get("intent") == "chitchat":
        return {
            "sufficient": True,
            "sufficiency_reason": "chitchat — 검색 불필요",
            PROGRESS_KEY: "",
        }

    candidates = state.get("candidates") or []
    if not candidates:
        return {
            "sufficient": False,
            "sufficiency_reason": "검색 결과 없음",
            "retry_count": state.get("retry_count", 0) + 1,
            PROGRESS_KEY: "🔎 검색 결과 검증 중... ✗ 결과 없음",
        }

    query = state.get("resolved_query") or state["current_query"]
    prompt = prompts.SELF_CHECK.format(
        query=query, docs=render_docs_brief(candidates)
    )
    data = await llm_json(get_judge_llm(), prompt)
    sufficient = bool(data.get("sufficient", False))
    reason = data.get("reason") or ""

    update: dict = {
        "sufficient": sufficient,
        "sufficiency_reason": reason,
        PROGRESS_KEY: f"🔎 검색 결과 검증 중... {'✓ 충분' if sufficient else '✗ 불충분'}",
    }
    if not sufficient:
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update


def should_retry(state: RAGState) -> str:
    """Conditional edge:
      - 'generate' (sufficient): evidence is sufficient → grounded answer.
      - 'retry'                : insufficient & retry budget left → loop back
                                 through query_variate → hybrid_retrieve.
      - 'generate' (exhausted) : retry budget exhausted → still go to generate,
                                 which emits "해당 정보를 찾을 수 없습니다".
                                 Domain questions (Elasticsearch/Kafka) MUST
                                 NOT fall back to general LLM knowledge — the
                                 corpus is the single source of truth.
    """
    if state.get("sufficient"):
        return "generate"
    if state.get("retry_count", 0) >= get_settings().retrieval_max_retry:
        return "generate"
    return "retry"
