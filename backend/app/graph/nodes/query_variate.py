"""Re-write the search queries from a different angle when self_check fails.

Sits in the retrieval cycle between self_check (insufficient) and a fresh
hybrid_retrieve attempt. Without this, the cycle would just re-issue the
exact same ES request and get the exact same (insufficient) results.
"""

from __future__ import annotations

import asyncio

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def _vary_one(
    prev_kw: str, prev_sem: str, reason: str, attempt: int
) -> tuple[str, str]:
    prompt = prompts.QUERY_VARIATE.format(
        prev_keywords=prev_kw or "(none)",
        prev_semantic=prev_sem or "(none)",
        reason=reason or "(no specific reason)",
        attempt=attempt,
    )
    data = await llm_json(get_judge_llm(), prompt)
    new_kw = (data.get("keywords") or "").strip()
    new_sem = (data.get("semantic") or "").strip()
    # Defensive: never let a variation collapse to identical or empty strings.
    if not new_kw or new_kw.lower() == prev_kw.lower():
        new_kw = prev_kw
    if not new_sem or new_sem.lower() == prev_sem.lower():
        new_sem = prev_sem
    return new_kw, new_sem


async def query_variate(state: RAGState) -> dict:
    attempt = state.get("retry_count", 0)
    # No-op on the very first pass (shouldn't be wired in then anyway).
    if attempt <= 0:
        return {PROGRESS_KEY: ""}

    bm25 = list(state.get("rewritten_queries") or [])
    sem = list(state.get("semantic_queries") or [])
    if not bm25:
        return {PROGRESS_KEY: ""}

    # Pad semantic to bm25 length defensively (matches hybrid_retrieve logic).
    if len(sem) < len(bm25):
        sem = sem + bm25[len(sem):]

    reason = state.get("sufficiency_reason") or ""

    pairs = await asyncio.gather(
        *[_vary_one(k, s, reason, attempt) for k, s in zip(bm25, sem)]
    )
    new_kw_list = [p[0] for p in pairs]
    new_sem_list = [p[1] for p in pairs]

    lines = [f"🔄 검색 쿼리 변형 (재시도 {attempt}회차) — {reason or '결과 부족'}"]
    for i, (old_kw, new_kw, old_sem, new_sem) in enumerate(
        zip(bm25, new_kw_list, sem, new_sem_list)
    ):
        branch = "└─" if i == len(bm25) - 1 else "├─"
        lines.append(f"   {branch} BM25:     {old_kw}  →  {new_kw}")
        lines.append(f"   {'  ' if branch == '└─' else '│'}   semantic: {old_sem}  →  {new_sem}")

    return {
        "rewritten_queries": new_kw_list,
        "semantic_queries": new_sem_list,
        PROGRESS_KEY: "\n".join(lines),
    }
