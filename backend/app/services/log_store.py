"""Per-user chat log storage in Elasticsearch.

Each authenticated user has a dedicated `{sanitized_user_id}_logs` index.
Every chat turn (question + final answer + full RAG trace) is stored as one
document. Used by the `debug_explain` node to retrospectively explain why a
prior answer was produced.

The index name is derived deterministically from `user_id` after sanitization
(ES requires lowercase + a restricted alphabet). Index creation is lazy and
idempotent.

Failures are best-effort logged — saving a turn must never block the chat
response.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.services.elasticsearch_client import get_es_client

logger = logging.getLogger(__name__)


_LOG_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
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


def sanitize_user_id(user_id: str) -> str:
    """Convert a frontend user_id into a valid ES index-name fragment.

    ES requires lowercase + `[a-z0-9_-]`. Anything else collapses to `_`.
    Returns an empty string when no usable characters remain (caller must
    treat that as "no log persistence")."""
    if not user_id:
        return ""
    cleaned = re.sub(r"[^a-z0-9_-]", "_", user_id.lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned


def log_index_name(user_id: str) -> str:
    safe = sanitize_user_id(user_id)
    if not safe:
        return ""
    return f"{safe}_logs"


async def ensure_log_index(
    user_id: str, *, client: AsyncElasticsearch | None = None
) -> str:
    """Idempotently create `{user_id}_logs` with the chat-log mapping.

    Returns the resolved index name (empty string when the user_id is unusable
    so the caller can short-circuit)."""
    name = log_index_name(user_id)
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
    client: AsyncElasticsearch | None = None,
) -> None:
    """Best-effort: index one chat-turn document. Failures only log."""
    name = log_index_name(user_id)
    if not name:
        return
    es = client or get_es_client()
    try:
        await ensure_log_index(user_id, client=es)
        body = dict(doc)
        body.setdefault("user_id", user_id)
        body.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        await es.index(index=name, body=body)
    except Exception as e:
        logger.warning("save_turn(%s) failed: %s", name, e)


async def fetch_recent_turns(
    user_id: str,
    *,
    n: int = 3,
    client: AsyncElasticsearch | None = None,
) -> list[dict[str, Any]]:
    """Returns the `n` most recent stored turns (newest first). Empty list on
    any error or when the user has no logs yet."""
    name = log_index_name(user_id)
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
            "query": {"match_all": {}},
        }
        r = await es.search(index=name, body=body)
        hits = r.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]
    except Exception as e:
        logger.warning("fetch_recent_turns(%s) failed: %s", name, e)
        return []
