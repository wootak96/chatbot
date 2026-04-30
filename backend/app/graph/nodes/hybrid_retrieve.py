from __future__ import annotations

import asyncio

from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import doc_dedup_key, doc_label
from app.graph.state import RAGState, SearchPlan
from app.services.elasticsearch_client import hybrid_search


async def hybrid_retrieve(state: RAGState) -> dict:
    if state.get("intent") == "chitchat":
        return {"candidates": [], PROGRESS_KEY: ""}

    plans: list[SearchPlan] = state.get("search_plans") or []
    if not plans:
        return {
            "candidates": [],
            PROGRESS_KEY: "📚 검색 계획 없음 — 검색 스킵",
        }

    metadata = state.get("metadata_filters") or {}

    async def _search(p: SearchPlan):
        bm25 = p.get("bm25") or p.get("sub_query") or ""
        sem = p.get("semantic") or bm25
        index = p.get("index") or ""
        if not index or not bm25:
            return []
        return await hybrid_search(
            bm25_query_text=bm25,
            semantic_query_text=sem,
            indices=[index],
            metadata_filters=metadata,
        )

    results = await asyncio.gather(*[_search(p) for p in plans])

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

    return {
        "candidates": merged,
        PROGRESS_KEY: (
            f"📚 Knowledge Base 검색 중.. ({len(merged)}건 발견)"
            f"{labels_block}"
        ),
    }
