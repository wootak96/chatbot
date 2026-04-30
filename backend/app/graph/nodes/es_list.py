"""Answer document-list questions with a terms aggregation on the title field."""

from __future__ import annotations

import re

from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes.index_route import route_query
from app.graph.state import RAGState
from app.services.elasticsearch_client import list_titles


_DEFAULT_LIST_SIZE = 30
_MAX_LIST_SIZE = 1000

# "10개", "20 건", "30개씩"
_SIZE_NUMERIC = re.compile(r"(\d+)\s*(개|건)")
# "전체 / 모두 / 모든 / 다 보여" — user wants everything (capped at _MAX_LIST_SIZE).
_SIZE_ALL = re.compile(r"(전체|모두|모든|\s다\s|^다\s|다\s*보여|all)")


def _extract_size(query: str) -> int:
    """Pull a list size out of the user query, defaulting to _DEFAULT_LIST_SIZE.

    Intentionally conservative: requires `개` / `건` suffix on the number to
    avoid false positives like '최근 30일'. Bumps to _MAX_LIST_SIZE when the
    user explicitly asks for everything ('전체', '모두', '다 보여줘')."""
    if not query:
        return _DEFAULT_LIST_SIZE
    m = _SIZE_NUMERIC.search(query)
    if m:
        try:
            n = int(m.group(1))
        except ValueError:
            return _DEFAULT_LIST_SIZE
        return max(1, min(n, _MAX_LIST_SIZE))
    if _SIZE_ALL.search(query):
        return _MAX_LIST_SIZE
    return _DEFAULT_LIST_SIZE


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

    query = state.get("resolved_query") or state["current_query"]
    indices = await route_query(query)
    requested_size = _extract_size(query)
    by_index = await list_titles(indices=indices, size=requested_size)
    total = sum(len(v) for v in by_index.values())

    if total == 0:
        answer = "표시할 문서 목록이 없습니다."
        return {
            "final_answer": answer,
            "sources": [],
            PROGRESS_KEY: "📚 문서 목록 조회 중... (결과 없음)",
        }

    cap_note = f" (인덱스별 최대 {requested_size}개)" if requested_size != _DEFAULT_LIST_SIZE else ""
    lines = [f"📚 문서 목록 — 총 **{total}개 제목**{cap_note}"]
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
        PROGRESS_KEY: f"📚 문서 목록 조회 중... ({total}개 제목, 요청 size={requested_size})",
    }
