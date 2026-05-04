"""Session list and message-history endpoints for the chat UI sidebar.

`GET /v1/sessions?user_id=...`              → list of the user's sessions.
`GET /v1/sessions/{session_id}/messages?...` → messages of one session, in order.

These are read-only views over the shared `chat_logs` index, never mutate it.
Sessions are derived: a "session" is just a distinct `session_id` value seen
under a given `user_id`, with metadata (first user question as title, latest
timestamp, turn count) computed via aggregation.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.elasticsearch_client import get_es_client
from app.services.log_store import chat_logs_index_name

logger = logging.getLogger(__name__)

router = APIRouter()


_SESSION_LIST_LIMIT = 100  # cap returned sessions per user
_SESSION_MESSAGES_LIMIT = 200  # cap turns returned for a single session


@router.get("/v1/sessions")
async def list_sessions(user_id: str = Query(..., min_length=1)) -> dict:
    """Return the user's chat sessions, newest first.

    Each entry: {session_id, title, last_updated, turn_count}. `title` is the
    first user question of the session (truncated for display). Empty list
    when the user has no logged turns yet.
    """
    name = chat_logs_index_name()
    if not name:
        return {"sessions": []}
    es = get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            return {"sessions": []}
        body = {
            "size": 0,
            "query": {"bool": {"filter": [{"term": {"user_id": user_id}}]}},
            "aggs": {
                "sessions": {
                    "terms": {
                        "field": "session_id",
                        "size": _SESSION_LIST_LIMIT,
                        "order": {"latest": "desc"},
                    },
                    "aggs": {
                        "latest": {"max": {"field": "timestamp"}},
                        "first_q": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"timestamp": {"order": "asc"}}],
                                "_source": ["question", "timestamp"],
                            }
                        },
                    },
                }
            },
        }
        r = await es.search(index=name, body=body)
    except Exception as e:
        logger.warning("list_sessions(%s) failed: %s", user_id, e)
        return {"sessions": []}

    buckets = (
        r.get("aggregations", {}).get("sessions", {}).get("buckets", [])
    )
    sessions: list[dict[str, Any]] = []
    for b in buckets:
        sid = b.get("key", "")
        if not sid:
            continue
        first_hit = (
            b.get("first_q", {}).get("hits", {}).get("hits", [])
        )
        question = ""
        if first_hit:
            question = first_hit[0].get("_source", {}).get("question", "") or ""
        title = (question.strip() or "(제목 없음)")[:80]
        sessions.append(
            {
                "session_id": sid,
                "title": title,
                "last_updated": b.get("latest", {}).get("value_as_string", ""),
                "turn_count": int(b.get("doc_count", 0)),
            }
        )
    return {"sessions": sessions}


@router.get("/v1/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user_id: str = Query(..., min_length=1),
) -> dict:
    """Return the messages (user question + assistant answer pairs) of a
    single session in chronological order. Used by the UI when a user
    clicks a session in the sidebar."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    name = chat_logs_index_name()
    if not name:
        return {"messages": []}
    es = get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            return {"messages": []}
        body = {
            "size": _SESSION_MESSAGES_LIMIT,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user_id": user_id}},
                        {"term": {"session_id": session_id}},
                    ]
                }
            },
            "sort": [{"timestamp": {"order": "asc"}}],
            "_source": ["question", "final_answer", "timestamp"],
        }
        r = await es.search(index=name, body=body)
    except Exception as e:
        logger.warning(
            "get_session_messages(%s, %s) failed: %s", user_id, session_id, e
        )
        return {"messages": []}

    hits = r.get("hits", {}).get("hits", [])
    messages: list[dict[str, str]] = []
    for h in hits:
        src = h.get("_source", {})
        q = (src.get("question") or "").strip()
        a = (src.get("final_answer") or "").strip()
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})
    return {"messages": messages}
