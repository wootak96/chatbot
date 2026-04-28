"""Answer document-list questions with a terms aggregation on the title field."""

from __future__ import annotations

from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.state import RAGState
from app.services.elasticsearch_client import list_titles


async def es_list(state: RAGState) -> dict:
    s = get_settings()

    if not s.es_field_title:
        # Current production mapping has no title field — be honest about it.
        answer = (
            "현재 인덱스에 `title` 필드가 매핑되어 있지 않아 문서 목록을 표시할 수 없습니다.\n"
            "(인덱스에 `title` 필드를 보강하면 이 질문에 답할 수 있습니다.)"
        )
        return {
            "final_answer": answer,
            "sources": [],
            PROGRESS_KEY: "📚 문서 목록 조회 중... (title 필드 부재로 스킵)",
        }

    indices = s.all_indices
    by_index = await list_titles(indices=indices, size=30)
    total = sum(len(v) for v in by_index.values())

    if total == 0:
        answer = "표시할 문서 목록이 없습니다."
        return {
            "final_answer": answer,
            "sources": [],
            PROGRESS_KEY: "📚 문서 목록 조회 중... (결과 없음)",
        }

    lines = [f"📚 문서 목록 — 총 **{total}개 제목**"]
    for idx, items in by_index.items():
        if not items:
            continue
        lines.append("")
        lines.append(f"**`{idx}`** ({len(items)}개)")
        for title, chunk_count in items:
            display = title or "(제목 없음)"
            lines.append(f"- {display} ({chunk_count} chunks)")
    answer = "\n".join(lines)

    return {
        "final_answer": answer,
        "sources": [],
        PROGRESS_KEY: f"📚 문서 목록 조회 중... ({total}개 제목)",
    }
