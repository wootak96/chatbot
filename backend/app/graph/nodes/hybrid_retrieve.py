from __future__ import annotations

import asyncio

from app.graph.nodes import PROGRESS_KEY
from app.graph.state import RAGState
from app.services.elasticsearch_client import hybrid_search


async def hybrid_retrieve(state: RAGState) -> dict:
    if state.get("intent") == "chitchat":
        return {"candidates": [], PROGRESS_KEY: ""}

    bm25_queries = state.get("rewritten_queries") or state.get("sub_queries") or []
    semantic_queries = state.get("semantic_queries") or []
    indices_per_query = state.get("target_indices_per_query") or []

    if not bm25_queries:
        return {
            "candidates": [],
            PROGRESS_KEY: "📚 검색 쿼리 없음 — 검색 스킵",
        }

    # Defensive: align lengths. If routing produced fewer entries than queries
    # (shouldn't happen), pad with the union of all known indices via empty list
    # (which downstream skips).
    if len(indices_per_query) < len(bm25_queries):
        indices_per_query = list(indices_per_query) + [[]] * (
            len(bm25_queries) - len(indices_per_query)
        )
    # Pad semantic_queries with the matching BM25 text when shorter (e.g. when
    # rewrite is skipped and we fall back to sub_queries directly).
    if len(semantic_queries) < len(bm25_queries):
        semantic_queries = list(semantic_queries) + bm25_queries[len(semantic_queries):]

    metadata = state.get("metadata_filters") or {}

    async def _search(bm25: str, sem: str, indices: list[str]):
        if not indices:
            return []
        return await hybrid_search(
            bm25_query_text=bm25,
            semantic_query_text=sem,
            indices=indices,
            metadata_filters=metadata,
        )

    results = await asyncio.gather(
        *[
            _search(bm25, sem, ix)
            for bm25, sem, ix in zip(bm25_queries, semantic_queries, indices_per_query)
        ]
    )

    seen: set[str] = set()
    merged: list[dict] = []
    for docs in results:
        for d in docs:
            key = d.get("id") or d.get("url") or ""
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

    return {
        "candidates": merged,
        PROGRESS_KEY: f"📚 Elasticsearch 검색 중... ({len(merged)}건 발견)",
    }
