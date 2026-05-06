"""Per-user persistent instruction store.

One document per user in the `chat_md` index. The document holds the
accumulated answer-style preferences as a single markdown string the LLM
curates (additive, edit-in-place, or removal — all expressed by rewriting
the markdown). Cross-session by design: a user's preferences persist
beyond the conversation thread, similar to OpenClaw's memory.md.

Read path: every answer node loads the user's md and prepends it to the
prompt as a `[사용자 지침]` block. Write path: only the `instruction`
intent updates it, via `update_user_md`.

Failures are best-effort logged. A missing index, missing doc, or any ES
error degrades to an empty string — chat must keep working.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.config import get_settings
from app.services.elasticsearch_client import get_es_client

logger = logging.getLogger(__name__)


_INDEX_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "markdown": {"type": "text"},
            "updated_at": {"type": "date"},
        }
    }
}


def chat_md_index_name() -> str:
    return get_settings().es_index_chat_md


async def ensure_chat_md_index(
    *, client: AsyncElasticsearch | None = None
) -> str:
    name = chat_md_index_name()
    if not name:
        return ""
    es = client or get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            await es.indices.create(index=name, body=_INDEX_MAPPING)
    except Exception as e:
        logger.warning("ensure_chat_md_index(%s) failed: %s", name, e)
    return name


async def get_user_md(
    user_id: str, *, client: AsyncElasticsearch | None = None
) -> str:
    """Return the user's stored instruction markdown, or empty string when
    nothing has been saved yet / on any ES error."""
    if not user_id:
        return ""
    name = chat_md_index_name()
    if not name:
        return ""
    es = client or get_es_client()
    try:
        exists = await es.indices.exists(index=name)
        if not exists:
            return ""
        r = await es.get(index=name, id=user_id)
        return (r.get("_source") or {}).get("markdown", "") or ""
    except NotFoundError:
        return ""
    except Exception as e:
        logger.warning("get_user_md(user_id=%s) failed: %s", user_id, e)
        return ""


async def update_user_md(
    user_id: str,
    new_markdown: str,
    *,
    client: AsyncElasticsearch | None = None,
) -> None:
    """Upsert the user's instruction document. doc_id = user_id."""
    if not user_id:
        return
    name = chat_md_index_name()
    if not name:
        return
    es = client or get_es_client()
    try:
        await ensure_chat_md_index(client=es)
        await es.index(
            index=name,
            id=user_id,
            body={
                "user_id": user_id,
                "markdown": new_markdown or "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        logger.warning("update_user_md(user_id=%s) failed: %s", user_id, e)
