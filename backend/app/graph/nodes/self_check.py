from __future__ import annotations

from app import prompts
from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import (
    doc_dedup_key,
    doc_label,
    llm_json,
    render_docs_brief,
)
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

    # Aggregate per-doc verdicts to per-title (1-indexed match render_docs_brief).
    # Same title can appear in multiple chunks; mark title relevant if ANY chunk
    # was judged relevant. LLM returned items may be missing/malformed — default
    # to relevant=True so unjudged docs aren't penalized in the UI.
    per_doc_raw = data.get("per_doc") or []
    idx_to_relevant: dict[int, bool] = {}
    for item in per_doc_raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        idx_to_relevant[idx] = bool(item.get("relevant", True))

    # Aggregate per-chunk verdicts to per-source-doc using a stable dedup key
    # (title → url → id). Display label falls back the same way so something
    # human-readable shows even when `title` field is empty in the index.
    label_verdict: dict[str, bool] = {}
    label_order: list[str] = []
    label_for_key: dict[str, str] = {}
    for i, d in enumerate(candidates, 1):
        key = doc_dedup_key(d) or doc_label(d)
        if not key:
            continue
        chunk_relevant = idx_to_relevant.get(i, True)
        if key not in label_verdict:
            label_order.append(key)
            label_verdict[key] = chunk_relevant
            label_for_key[key] = doc_label(d) or key
        else:
            label_verdict[key] = label_verdict[key] or chunk_relevant

    if label_order:
        titles_block = "\n" + "\n".join(
            f"  {'✓' if label_verdict[k] else '✗'} {label_for_key[k]}"
            for k in label_order
        )
    else:
        titles_block = ""

    update: dict = {
        "sufficient": sufficient,
        "sufficiency_reason": reason,
        PROGRESS_KEY: (
            f"🔎 검색 결과 검증 중... {'✓ 충분' if sufficient else '✗ 불충분'}"
            f"{titles_block}"
        ),
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
