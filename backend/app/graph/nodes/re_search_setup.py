"""Force-reroute node — re-runs the prior turn's search against user-named
indices.

Triggered when query_analyze sees a phrase like "confluence에서 검색해줘"
that combines a known index alias with a search-action verb. The current
user message itself is NOT a new question — it's a directive to redo the
prior question's search on a different index set.

Flow:
  1. Fetch the most recent stored turn from chat_logs (user_id + session_id).
  2. Pull its `sub_queries` and the original `resolved_query`/`metadata_filters`.
  3. Set `target_indices_per_query` so every prior sub_query routes to the
     user-named indices instead of the LLM-decided ones.
  4. Hand off to query_rewrite — NOT directly to hybrid_retrieve. The prior
     turn's BM25/semantic strings were tuned to the prior turn's indices
     (English for ES/Kafka, Korean for Confluence), so they MUST be
     re-rewritten when the index set changes.

If we can't reuse a prior turn (no auth, no session, no prior search turn),
emit a friendly Korean explanation and end the workflow — the user wasn't
asking for a *new* search, just a re-route, so we shouldn't silently fall
through to a different flow.
"""

from __future__ import annotations

from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.state import RAGState
from app.services.log_store import fetch_recent_turns


def _refusal(message: str, progress_tail: str) -> dict:
    return {
        "intent": "re_search",
        "final_answer": message,
        "sources": [],
        "sufficient": False,
        PROGRESS_KEY: f"🔁 강제 재검색 — {progress_tail}",
    }


async def re_search_setup(state: RAGState) -> dict:
    user_id = (state.get("user_id") or "").strip()
    session_id = (state.get("session_id") or "").strip()
    forced = state.get("forced_indices") or []

    if not user_id:
        return _refusal(
            "강제 재검색은 로그인된 사용자에 한해 동작합니다 "
            "(현재 요청에 user_id가 포함되지 않아 직전 검색 기록을 조회할 수 없습니다).",
            "user_id 미전송",
        )
    if not session_id:
        return _refusal(
            "강제 재검색은 세션 단위로 동작합니다 "
            "(현재 요청에 session_id가 포함되지 않아 직전 검색을 조회할 수 없습니다).",
            "session_id 미전송",
        )
    if not forced:
        return _refusal(
            "재검색할 인덱스가 지정되지 않았습니다. "
            "예: 'confluence에서 검색해줘', 'kafka에서 다시 찾아줘'.",
            "인덱스 미지정",
        )

    settings = get_settings()
    alias_map = settings.index_alias_map  # {"elasticsearch": "elasticsearch_docs", ...}
    resolved_targets = [alias_map[a] for a in forced if a in alias_map]
    if not resolved_targets:
        return _refusal(
            "인식 가능한 인덱스 이름이 없습니다 "
            "(지원: elasticsearch / kafka / confluence).",
            "인덱스 alias 매칭 실패",
        )

    turns = await fetch_recent_turns(user_id, session_id=session_id, n=1)
    if not turns:
        return _refusal(
            "현재 세션에 직전 검색 기록이 없어 재검색할 대상을 찾지 못했습니다.",
            "직전 턴 없음",
        )
    prev = turns[0]
    prev_sub_queries = prev.get("sub_queries") or []
    if not prev_sub_queries:
        return _refusal(
            "직전 턴이 검색 흐름이 아니어서 재라우팅할 서브쿼리가 없습니다 "
            "(이전 메시지가 일반 대화/지침/디버깅 이었을 수 있습니다).",
            "직전 턴이 검색이 아님",
        )

    # Build target_indices_per_query so each prior sub_query routes to the
    # full forced-index set. query_rewrite (next node) will then produce a
    # fresh (BM25, semantic) pair per (sub_query, new_index) — never reuse
    # the prior turn's strings, since they were tuned to a different index
    # language (English for ES/Kafka, Korean for Confluence).
    target_indices_per_query = [
        list(resolved_targets) for _ in prev_sub_queries
    ]
    resolved_query = prev.get("resolved_query") or prev.get("question") or ""

    return {
        # Keep intent="re_search" so chat_logs records the original intent
        # for later inspection. hybrid_retrieve / self_check / generate only
        # branch on `intent == "chitchat"`, so any non-chitchat intent (incl.
        # re_search) takes the standard RAG path without code changes.
        "search_intent": "lookup",
        "sub_queries": list(prev_sub_queries),
        "target_indices_per_query": target_indices_per_query,
        # Intentionally NOT setting `search_plans` — let query_rewrite
        # build fresh plans with index-aware BM25/semantic strings.
        "metadata_filters": prev.get("metadata_filters") or {},
        "resolved_query": resolved_query,
        "retry_count": 0,
        PROGRESS_KEY: (
            "🔁 강제 재검색 — 직전 질문을 다음 인덱스로 재라우팅: "
            + ", ".join(forced)
        ),
    }
