"""Post-generate answer-quality gate.

Runs immediately after `generate`. `self_check` only inspects the retrieved
documents *before* generation, so a falsely-optimistic sufficiency verdict
can still let a non-answer through (soft-escape, pure hedging, off-topic).
This node judges the FINAL ANSWER TEXT itself: when the answer is a
non-answer and the retry budget allows, the flow loops back through
`query_variate` for a wider re-search.

The retry budget is shared with the `self_check` loop — both gates bump
`retry_count`, which also drives the escalating top_k schedule in
`hybrid_retrieve`, so a rejected answer triggers a wider re-search.
"""

from __future__ import annotations

from app import prompts
from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def answer_check(state: RAGState) -> dict:
    # chitchat answers are greetings — there is no retrieval to re-run, so
    # there is nothing to gate. Always accept.
    if state.get("intent") == "chitchat":
        return {"answer_ok": True, PROGRESS_KEY: ""}

    answer = (state.get("final_answer") or "").strip()
    query = state.get("resolved_query") or state.get("current_query") or ""

    if not answer:
        # generate produced nothing — treat as a non-answer so the loop can
        # retry (budget permitting).
        return {
            "answer_ok": False,
            "sufficiency_reason": "생성된 답변이 비어 있음",
            "retry_count": state.get("retry_count", 0) + 1,
            PROGRESS_KEY: "✅ 답변 품질 검증... ✗ 빈 답변 — 재검색",
        }

    prompt = prompts.ANSWER_CHECK.format(query=query, answer=answer)
    data = await llm_json(get_judge_llm(), prompt)
    answer_ok = bool(data.get("answer_ok", True))
    reason = data.get("reason") or ""

    update: dict = {
        "answer_ok": answer_ok,
        # Only surface the gate in the trace when it actually does something
        # (rejects → re-search). A passing verdict is silent.
        PROGRESS_KEY: "" if answer_ok else "✅ 답변 품질 검증... ✗ 부족 — 재검색",
    }
    if not answer_ok:
        # Overwrite sufficiency_reason so query_variate varies the next search
        # based on *why the answer failed*, not a stale self_check verdict.
        update["sufficiency_reason"] = reason or "답변이 질문에 충분히 답하지 못함"
        update["retry_count"] = state.get("retry_count", 0) + 1
    return update


def should_regenerate(state: RAGState) -> str:
    """Conditional edge after answer_check:
      - 'end'  : answer accepted, OR the shared retry budget is exhausted,
                 OR there are no search plans to re-run (re-search can't help).
      - 'retry': answer rejected and budget remains → loop back through
                 query_variate → hybrid_retrieve for a wider re-search.

    Budget is len(retrieval_top_k_schedule), same as the self_check loop, and
    both gates increment retry_count — so the loop always terminates.
    """
    if state.get("answer_ok", True):
        return "end"
    if state.get("retry_count", 0) >= len(get_settings().retrieval_top_k_schedule):
        return "end"
    if not state.get("search_plans"):
        return "end"
    return "retry"
