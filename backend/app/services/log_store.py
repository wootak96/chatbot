"""Unified chat-log storage in Elasticsearch.

Every chat turn (question + final answer + full RAG trace) is stored as one
document in the shared `chat_logs` index, with `user_id` as a keyword field
on the document — no per-user indices. `debug_explain` filters by user_id
to retrieve a user's recent turns.

Failures are best-effort logged — saving a turn must never block the chat
response.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.config import get_settings
from app.services.elasticsearch_client import get_es_client

logger = logging.getLogger(__name__)


_LOG_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "question": {"type": "text"},
            "resolved_query": {"type": "text"},
            "intent": {"type": "keyword"},
            "search_intent": {"type": "keyword"},
            "sub_queries": {"type": "keyword"},
            "target_indices": {"type": "keyword"},
            "index_routing": {
                "type": "object",
                "properties": {
                    "sub_query": {"type": "keyword"},
                    "indices": {"type": "keyword"},
                },
            },
            "metadata_filters": {
                "type": "object",
                "properties": {
                    "source": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "gte": {"type": "keyword"},
                            "lte": {"type": "keyword"},
                        },
                    },
                },
            },
            "forced_indices": {"type": "keyword"},
            "search_plans": {
                "type": "object",
                "properties": {
                    "sub_query": {"type": "keyword"},
                    "index": {"type": "keyword"},
                    "bm25": {"type": "text"},
                    "semantic": {"type": "text"},
                },
            },
            "candidates": {
                "type": "object",
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "score": {"type": "float"},
                    "used": {"type": "boolean"},
                },
            },
            "bm25_only_results": {
                "type": "object",
                "properties": {
                    "sub_query": {"type": "keyword"},
                    "index": {"type": "keyword"},
                    "bm25": {"type": "text"},
                    "hits": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "keyword"},
                            "url": {"type": "keyword"},
                            "score": {"type": "float"},
                            "index": {"type": "keyword"},
                        },
                    },
                },
            },
            "semantic_only_results": {
                "type": "object",
                "properties": {
                    "sub_query": {"type": "keyword"},
                    "index": {"type": "keyword"},
                    "semantic": {"type": "text"},
                    "hits": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "keyword"},
                            "url": {"type": "keyword"},
                            "score": {"type": "float"},
                            "index": {"type": "keyword"},
                        },
                    },
                },
            },
            "sufficient": {"type": "boolean"},
            "sufficiency_reason": {"type": "text"},
            "retry_count": {"type": "integer"},
            "final_answer": {"type": "text"},
            "sources": {
                "type": "object",
                "properties": {
                    "url": {"type": "keyword"},
                    "title": {"type": "keyword"},
                },
            },
            "progress_log": {"type": "text"},
            "groundedness": {
                "properties": {
                    "grounded": {"type": "boolean"},
                    "score": {"type": "float"},
                    "supported_count": {"type": "integer"},
                    "total_claims": {"type": "integer"},
                    "claims": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "text"},
                            "citations": {"type": "integer"},
                            "supported": {"type": "boolean"},
                            "reason": {"type": "text"},
                        },
                    },
                }
            },
            "token_usage": {
                "properties": {
                    "total_input": {"type": "integer"},
                    "total_output": {"type": "integer"},
                    "total_tokens": {"type": "integer"},
                    "llm_calls": {"type": "integer"},
                    "by_node": {
                        "type": "object",
                        "properties": {
                            "node": {"type": "keyword"},
                            "input": {"type": "integer"},
                            "output": {"type": "integer"},
                            "total": {"type": "integer"},
                            "calls": {"type": "integer"},
                        },
                    },
                }
            },
        }
    }
}


def chat_logs_index_name() -> str:
    """Resolve the shared chat-logs index name from settings."""
    return get_settings().es_index_chat_logs


async def ensure_log_index(
    *, client: AsyncElasticsearch | None = None
) -> str:
    """Idempotently create the shared chat-logs index.

    Returns the resolved index name (empty string when the setting is blank
    so the caller can short-circuit)."""
    name = chat_logs_index_name()
    if not name:
        return ""
    es = client or get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            await es.indices.create(index=name, body=_LOG_INDEX_MAPPING)
    except Exception as e:
        logger.warning("ensure_log_index(%s) failed: %s", name, e)
    return name


async def save_turn(
    user_id: str,
    doc: dict[str, Any],
    *,
    session_id: str = "",
    client: AsyncElasticsearch | None = None,
) -> None:
    """Best-effort: index one chat-turn document. user_id and session_id are
    stored as keyword fields. Failures only log."""
    if not user_id:
        return  # unauthenticated — no log persistence
    name = chat_logs_index_name()
    if not name:
        return
    es = client or get_es_client()
    try:
        await ensure_log_index(client=es)
        body = dict(doc)
        # Always set user_id / session_id from the trusted caller args, never
        # from the incoming doc, to prevent cross-user / cross-session spoofing.
        body["user_id"] = user_id
        body["session_id"] = session_id or ""
        body.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        await es.index(index=name, body=body)
    except Exception as e:
        logger.warning(
            "save_turn(user_id=%s, session_id=%s) failed: %s",
            user_id,
            session_id,
            e,
        )


async def fetch_recent_turns(
    user_id: str,
    *,
    session_id: str = "",
    n: int = 3,
    client: AsyncElasticsearch | None = None,
) -> list[dict[str, Any]]:
    """Return the `n` most recent stored turns for (user_id, session_id),
    newest first. session_id is required: a fresh chat thread should NOT pull
    turns from the user's previous conversations. Empty list when either id
    is missing, the index doesn't exist yet, or on any ES error."""
    if not user_id or not session_id:
        return []
    name = chat_logs_index_name()
    if not name:
        return []
    es = client or get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            return []
        body = {
            "size": n,
            "sort": [{"timestamp": {"order": "desc"}}],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"user_id": user_id}},
                        {"term": {"session_id": session_id}},
                    ]
                }
            },
        }
        r = await es.search(index=name, body=body)
        hits = r.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]
    except Exception as e:
        logger.warning(
            "fetch_recent_turns(user_id=%s, session_id=%s) failed: %s",
            user_id,
            session_id,
            e,
        )
        return []
