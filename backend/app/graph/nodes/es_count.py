"""Answer document-count questions with ES _count (no embedding, no RRF)."""

from __future__ import annotations

from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes.index_route import route_query
from app.graph.state import RAGState
from app.services.elasticsearch_client import count_documents


async def es_count(state: RAGState) -> dict:
    query = state.get("resolved_query") or state["current_query"]
    indices = await route_query(query)
    counts = await count_documents(indices=indices)
    total = sum(counts.values())

    if total == 0:
        answer = "현재 인덱스에 검색 가능한 문서가 없습니다."
    else:
        lines = [f"📊 사내 문서 통계 — 총 **{total:,}건**", ""]
        for idx, n in counts.items():
            lines.append(f"- `{idx}`: {n:,}건")
        answer = "\n".join(lines)

    return {
        "final_answer": answer,
        "sources": [],
        PROGRESS_KEY: f"📊 문서 개수 조회 중... (총 {total:,}건)",
    }
