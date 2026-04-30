"""Index-aware sub-query rewriting.

Runs AFTER index_route. For each sub-query and each routed index, produces a
separate (BM25, semantic) pair tuned to that index's corpus language:
  - confluence_docs: Korean BM25 + Korean semantic (technical terms in English)
  - elasticsearch_docs / kafka_docs: English BM25 + English semantic

Output is a flat `search_plans` list — one entry per (sub_query, index) pair —
which `hybrid_retrieve` then executes in parallel.
"""

from __future__ import annotations

import asyncio

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState, SearchPlan
from app.services.llm_factory import get_judge_llm


async def _rewrite_one(query: str, target_index: str) -> tuple[str, str]:
    """Returns (bm25_keywords, semantic_query) for a single (query, index)."""
    data = await llm_json(
        get_judge_llm(),
        prompts.QUERY_REWRITE.format(query=query, target_index=target_index),
    )
    keywords = (data.get("keywords") or "").strip()
    semantic = (data.get("semantic") or "").strip()
    # Defensive fallbacks: if the model produced only one field, reuse it for
    # the other so retrieval still works.
    if not keywords and semantic:
        keywords = semantic
    if not semantic and keywords:
        semantic = keywords
    if not keywords and not semantic:
        keywords = semantic = query
    return keywords, semantic


async def query_rewrite(state: RAGState) -> dict:
    subs = state.get("sub_queries") or []
    indices_per_query = state.get("target_indices_per_query") or []

    if not subs:
        return {
            "search_plans": [],
            PROGRESS_KEY: "✏️ 검색 쿼리 최적화 중... (스킵)",
        }

    # Flatten (sub_query_idx, sub_query, index) tasks.
    tasks: list[tuple[int, str, str]] = []
    for i, sub in enumerate(subs):
        idxs = indices_per_query[i] if i < len(indices_per_query) else []
        for ix in idxs:
            tasks.append((i, sub, ix))

    if not tasks:
        return {
            "search_plans": [],
            PROGRESS_KEY: "✏️ 검색 쿼리 최적화 중... (라우팅된 인덱스 없음)",
        }

    rewrites = await asyncio.gather(
        *[_rewrite_one(sub, ix) for _, sub, ix in tasks]
    )
    plans: list[SearchPlan] = [
        {
            "sub_query_idx": i,
            "sub_query": sub,
            "index": ix,
            "bm25": kw,
            "semantic": sem,
        }
        for ((i, sub, ix), (kw, sem)) in zip(tasks, rewrites)
    ]

    lines = []
    for j, p in enumerate(plans):
        branch = "└─" if j == len(plans) - 1 else "├─"
        lines.append(f"   {branch} [{p['index']}] {p['sub_query']}")
        lines.append(f"      • BM25:     {p['bm25']}")
        lines.append(f"      • semantic: {p['semantic']}")
    tree = "\n".join(lines)

    return {
        "search_plans": plans,
        PROGRESS_KEY: f"✏️ 검색 쿼리 최적화 중...\n{tree}",
    }
