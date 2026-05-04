"""Tests for the unified chat-log store.

We never hit a real ES — `AsyncElasticsearch` is replaced with an in-memory
fake that records every method invocation. Index creation is asserted to be
idempotent and `fetch_recent_turns` round-trips the canned hits.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.services import log_store


# ---------- index name resolution ----------


def test_chat_logs_index_name_from_settings():
    assert log_store.chat_logs_index_name() == get_settings().es_index_chat_logs


# ---------- ES interactions ----------


class FakeIndices:
    def __init__(self, exists_value: bool = False):
        self.exists_value = exists_value
        self.create_calls: list[dict] = []
        self.exists_calls: list[str] = []

    async def exists(self, *, index: str):
        self.exists_calls.append(index)
        return self.exists_value

    async def create(self, *, index: str, body):
        self.create_calls.append({"index": index, "body": body})


class FakeES:
    def __init__(self, exists_value: bool = False, search_hits: list | None = None):
        self.indices = FakeIndices(exists_value=exists_value)
        self.index_calls: list[dict] = []
        self.search_calls: list[dict] = []
        self._search_hits = search_hits or []

    async def index(self, *, index: str, body):
        self.index_calls.append({"index": index, "body": body})

    async def search(self, *, index: str, body):
        self.search_calls.append({"index": index, "body": body})
        return {"hits": {"hits": self._search_hits}}


@pytest.mark.asyncio
async def test_ensure_log_index_creates_when_missing():
    es = FakeES(exists_value=False)
    name = await log_store.ensure_log_index(client=es)
    expected = log_store.chat_logs_index_name()
    assert name == expected
    assert es.indices.exists_calls == [expected]
    assert len(es.indices.create_calls) == 1
    assert es.indices.create_calls[0]["index"] == expected


@pytest.mark.asyncio
async def test_ensure_log_index_skips_when_exists():
    es = FakeES(exists_value=True)
    await log_store.ensure_log_index(client=es)
    assert es.indices.create_calls == []  # no-op


@pytest.mark.asyncio
async def test_save_turn_writes_to_shared_index_with_user_id_field():
    es = FakeES(exists_value=True)
    await log_store.save_turn(
        "alice",
        {"question": "Q", "final_answer": "A"},
        session_id="sess-1",
        client=es,
    )
    assert len(es.index_calls) == 1
    call = es.index_calls[0]
    assert call["index"] == log_store.chat_logs_index_name()
    body = call["body"]
    assert body["question"] == "Q"
    assert body["final_answer"] == "A"
    assert body["user_id"] == "alice"  # from the trusted arg
    assert body["session_id"] == "sess-1"  # from the trusted arg
    assert "timestamp" in body  # auto-defaulted


@pytest.mark.asyncio
async def test_save_turn_overrides_spoofed_user_id_and_session_id():
    """Caller-provided ids always win over doc-supplied ones."""
    es = FakeES(exists_value=True)
    await log_store.save_turn(
        "alice",
        {"question": "Q", "user_id": "mallory", "session_id": "evil-session"},
        session_id="sess-1",
        client=es,
    )
    body = es.index_calls[0]["body"]
    assert body["user_id"] == "alice"
    assert body["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_save_turn_skipped_for_empty_user_id():
    es = FakeES()
    await log_store.save_turn("", {"question": "Q"}, client=es)
    assert es.index_calls == []


@pytest.mark.asyncio
async def test_save_turn_swallows_es_errors():
    """ES failures must not propagate — chat response is already streamed."""

    class BoomES:
        class indices:
            @staticmethod
            async def exists(*args, **kwargs):
                raise RuntimeError("ES down")

        async def index(self, *args, **kwargs):
            raise RuntimeError("ES down")

    # Should NOT raise.
    await log_store.save_turn("alice", {"question": "Q"}, client=BoomES())


@pytest.mark.asyncio
async def test_fetch_recent_turns_filters_by_user_and_session():
    hits = [
        {"_source": {"question": "Q1", "user_id": "alice", "session_id": "sess-1"}},
        {"_source": {"question": "Q2", "user_id": "alice", "session_id": "sess-1"}},
    ]
    es = FakeES(exists_value=True, search_hits=hits)
    out = await log_store.fetch_recent_turns(
        "alice", session_id="sess-1", n=3, client=es
    )
    assert [t["question"] for t in out] == ["Q1", "Q2"]
    body = es.search_calls[0]["body"]
    assert body["size"] == 3
    assert body["sort"][0]["timestamp"]["order"] == "desc"
    flt = body["query"]["bool"]["filter"]
    assert {"term": {"user_id": "alice"}} in flt
    assert {"term": {"session_id": "sess-1"}} in flt
    assert es.search_calls[0]["index"] == log_store.chat_logs_index_name()


@pytest.mark.asyncio
async def test_fetch_recent_turns_empty_when_user_id_missing():
    es = FakeES()
    out = await log_store.fetch_recent_turns("", session_id="sess-1", client=es)
    assert out == []
    assert es.search_calls == []


@pytest.mark.asyncio
async def test_fetch_recent_turns_empty_when_session_id_missing():
    """No session_id → don't fetch anything (would mix prior conversations)."""
    es = FakeES(exists_value=True)
    out = await log_store.fetch_recent_turns("alice", session_id="", client=es)
    assert out == []
    assert es.search_calls == []


@pytest.mark.asyncio
async def test_fetch_recent_turns_empty_when_index_missing():
    es = FakeES(exists_value=False)
    out = await log_store.fetch_recent_turns("alice", client=es)
    assert out == []
    assert es.search_calls == []
