"""Tests for the per-user chat log store.

We never hit a real ES — `AsyncElasticsearch` is replaced with an in-memory
fake that records every method invocation. Index creation is asserted to be
idempotent and `fetch_recent_turns` round-trips the canned hits.
"""

from __future__ import annotations

import pytest

from app.services import log_store


# ---------- sanitize_user_id ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Alice", "alice"),                 # uppercase
        ("user.name", "user_name"),         # dot disallowed by ES
        ("user--name", "user--name"),       # hyphens allowed
        ("__weird__id__", "weird_id"),     # collapse + strip leading/trailing _
        ("", ""),                           # empty stays empty (no log persistence)
        (None, ""),                         # defensive None
        ("UPPER.WITH.dots", "upper_with_dots"),
        ("@@@", ""),                        # all-bad → empty
    ],
)
def test_sanitize_user_id(raw, expected):
    assert log_store.sanitize_user_id(raw) == expected


def test_log_index_name_appends_suffix():
    assert log_store.log_index_name("Alice") == "alice_logs"
    assert log_store.log_index_name("@@@") == ""  # unusable → ""


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
    name = await log_store.ensure_log_index("alice", client=es)
    assert name == "alice_logs"
    assert es.indices.exists_calls == ["alice_logs"]
    assert len(es.indices.create_calls) == 1
    assert es.indices.create_calls[0]["index"] == "alice_logs"


@pytest.mark.asyncio
async def test_ensure_log_index_skips_when_exists():
    es = FakeES(exists_value=True)
    await log_store.ensure_log_index("alice", client=es)
    assert es.indices.create_calls == []  # no-op


@pytest.mark.asyncio
async def test_ensure_log_index_empty_user_id_returns_empty():
    es = FakeES()
    assert await log_store.ensure_log_index("", client=es) == ""
    assert es.indices.exists_calls == []


@pytest.mark.asyncio
async def test_save_turn_indexes_doc_with_defaults():
    es = FakeES(exists_value=True)
    await log_store.save_turn(
        "alice",
        {"question": "Q", "final_answer": "A"},
        client=es,
    )
    assert len(es.index_calls) == 1
    call = es.index_calls[0]
    assert call["index"] == "alice_logs"
    body = call["body"]
    assert body["question"] == "Q"
    assert body["final_answer"] == "A"
    assert body["user_id"] == "alice"  # auto-defaulted
    assert "timestamp" in body  # auto-defaulted


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
async def test_fetch_recent_turns_returns_sources_newest_first():
    hits = [
        {"_source": {"question": "Q1", "timestamp": "2026-04-30T10:00:00Z"}},
        {"_source": {"question": "Q2", "timestamp": "2026-04-30T09:00:00Z"}},
        {"_source": {"question": "Q3", "timestamp": "2026-04-30T08:00:00Z"}},
    ]
    es = FakeES(exists_value=True, search_hits=hits)
    out = await log_store.fetch_recent_turns("alice", n=3, client=es)
    assert [t["question"] for t in out] == ["Q1", "Q2", "Q3"]
    # Verify the search body asked for desc timestamp and size=3.
    body = es.search_calls[0]["body"]
    assert body["size"] == 3
    assert body["sort"][0]["timestamp"]["order"] == "desc"


@pytest.mark.asyncio
async def test_fetch_recent_turns_empty_when_index_missing():
    es = FakeES(exists_value=False)
    out = await log_store.fetch_recent_turns("alice", client=es)
    assert out == []
    assert es.search_calls == []
