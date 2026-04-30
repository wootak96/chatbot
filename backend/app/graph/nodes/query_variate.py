"""Re-write the search queries from a different angle when self_check fails.

Sits in the retrieval cycle between self_check (insufficient) and a fresh
hybrid_retrieve attempt. Without this, the cycle would just re-issue the
exact same ES request and get the exact same (insufficient) results.

Operates on `search_plans` so each (sub_query, index) plan is varied
independently — keeping the per-index language policy intact across retries.
"""

from __future__ import annotations

import asyncio

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState, SearchPlan
from app.services.llm_factory import get_judge_llm


async def _vary_one(plan: SearchPlan, reason: str, attempt: int) -> SearchPlan:
    prev_kw = plan.get("bm25") or ""
    prev_sem = plan.get("semantic") or ""
    prompt = prompts.QUERY_VARIATE.format(
        prev_keywords=prev_kw or "(none)",
        prev_semantic=prev_sem or "(none)",
        reason=reason or "(no specific reason)",
        attempt=attempt,
        target_index=plan.get("index") or "",
    )
    data = await llm_json(get_judge_llm(), prompt)
    new_kw = (data.get("keywords") or "").strip()
    new_sem = (data.get("semantic") or "").strip()
    # Defensive: never let a variation collapse to identical or empty strings.
    if not new_kw or new_kw.lower() == prev_kw.lower():
        new_kw = prev_kw
    if not new_sem or new_sem.lower() == prev_sem.lower():
        new_sem = prev_sem
    return {**plan, "bm25": new_kw, "semantic": new_sem}


async def query_variate(state: RAGState) -> dict:
    attempt = state.get("retry_count", 0)
    # No-op on the very first pass (shouldn't be wired in then anyway).
    if attempt <= 0:
        return {PROGRESS_KEY: ""}

    plans = state.get("search_plans") or []
    if not plans:
        return {PROGRESS_KEY: ""}

    reason = state.get("sufficiency_reason") or ""

    new_plans = list(
        await asyncio.gather(*[_vary_one(p, reason, attempt) for p in plans])
    )

    lines = [f"🔄 검색 쿼리 변형 (재시도 {attempt}회차) — {reason or '결과 부족'}"]
    for j, (old, new) in enumerate(zip(plans, new_plans)):
        branch = "└─" if j == len(plans) - 1 else "├─"
        idx = new.get("index", "")
        lines.append(
            f"   {branch} [{idx}] BM25:     {old.get('bm25', '')}  →  {new['bm25']}"
        )
        lines.append(
            f"   {'  ' if branch == '└─' else '│'}   semantic: "
            f"{old.get('semantic', '')}  →  {new['semantic']}"
        )

    return {
        "search_plans": new_plans,
        PROGRESS_KEY: "\n".join(lines),
    }
