from __future__ import annotations

import asyncio

from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import doc_dedup_key, doc_label
from app.graph.state import RAGState, SearchPlan
from app.services.elasticsearch_client import (
    hybrid_search,
    single_retriever_search,
)


async def hybrid_retrieve(state: RAGState) -> dict:
    if state.get("intent") == "chitchat":
        return {
            "candidates": [],
            "bm25_only_results": [],
            "semantic_only_results": [],
            PROGRESS_KEY: "",
        }

    plans: list[SearchPlan] = state.get("search_plans") or []
    if not plans:
        return {
            "candidates": [],
            "bm25_only_results": [],
            "semantic_only_results": [],
            PROGRESS_KEY: "📚 검색 계획 없음 — 검색 스킵",
        }

    metadata = state.get("metadata_filters") or {}

    # Escalating top_k: attempt 0 pulls a tight set, each re-search widens it.
    # `retry_count` is 0 on the first pass and is bumped by self_check before
    # the loop comes back here, so it indexes straight into the schedule.
    schedule = get_settings().retrieval_top_k_schedule
    attempt = state.get("retry_count", 0)
    top_k = schedule[min(attempt, len(schedule) - 1)]

    async def _search(p: SearchPlan):
        bm25 = p.get("bm25") or p.get("sub_query") or ""
        sem = p.get("semantic") or bm25
        index = p.get("index") or ""
        if not index or not bm25:
            return [], [], []
        # Run RRF + BM25-only + semantic-only in parallel. The two single-
        # retriever calls are diagnostic only — their hits go to chat_logs
        # so we can see which retriever surfaced each fused doc. They never
        # feed back into the candidate pool.
        rrf_hits, bm25_hits, sem_hits = await asyncio.gather(
            hybrid_search(
                bm25_query_text=bm25,
                semantic_query_text=sem,
                indices=[index],
                metadata_filters=metadata,
                size=top_k,
            ),
            single_retriever_search(
                retriever_kind="bm25",
                bm25_query_text=bm25,
                semantic_query_text=sem,
                indices=[index],
                metadata_filters=metadata,
                size=top_k,
            ),
            single_retriever_search(
                retriever_kind="semantic",
                bm25_query_text=bm25,
                semantic_query_text=sem,
                indices=[index],
                metadata_filters=metadata,
                size=top_k,
            ),
        )
        return rrf_hits, bm25_hits, sem_hits

    triples = await asyncio.gather(*[_search(p) for p in plans])
    results = [t[0] for t in triples]
    bm25_only_per_plan = [t[1] for t in triples]
    semantic_only_per_plan = [t[2] for t in triples]

    seen: set[str] = set()
    merged: list[dict] = []
    for docs in results:
        for d in docs:
            key = d.get("id") or d.get("url") or ""
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)

    # Dedup labels for UI: when `title` field is missing, fall back to URL
    # last segment or content excerpt so something useful is shown per doc.
    seen_keys: set[str] = set()
    labels: list[str] = []
    for d in merged:
        key = doc_dedup_key(d) or doc_label(d)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        label = doc_label(d)
        if label:
            labels.append(label)
    labels_block = "\n" + "\n".join(f"  • {t}" for t in labels) if labels else ""

    def _trim_hits(hits: list[dict]) -> list[dict]:
        # Diagnostic dump for chat_logs: only the per-retriever rank
        # (1-based position) plus title/url. Score is dropped — it's not
        # comparable across BM25 and cosine and was noise for human review.
        return [
            {
                "rank": i,
                "title": h.get("title", ""),
                "url": h.get("url", ""),
            }
            for i, h in enumerate(hits, 1)
        ]

    bm25_only_results = [
        {
            "sub_query": p.get("sub_query", ""),
            "index": p.get("index", ""),
            "bm25": p.get("bm25", ""),
            "hits": _trim_hits(hits),
        }
        for p, hits in zip(plans, bm25_only_per_plan)
    ]
    semantic_only_results = [
        {
            "sub_query": p.get("sub_query", ""),
            "index": p.get("index", ""),
            "semantic": p.get("semantic", ""),
            "hits": _trim_hits(hits),
        }
        for p, hits in zip(plans, semantic_only_per_plan)
    ]

    return {
        "candidates": merged,
        "bm25_only_results": bm25_only_results,
        "semantic_only_results": semantic_only_results,
        PROGRESS_KEY: (
            f"📚 Knowledge Base 검색 중.. (top_k={top_k}, {len(merged)}건 발견)"
            f"{labels_block}"
        ),
    }
