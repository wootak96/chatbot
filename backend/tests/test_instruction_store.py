"""Tests for the per-user instruction (chat_md) store.

We never hit a real ES — replace AsyncElasticsearch with an in-memory fake
that records every call. Both happy paths and degraded paths (missing
index, missing doc, ES error) are asserted to never raise.
"""

from __future__ import annotations

import pytest
from elasticsearch import NotFoundError

from app.config import get_settings
from app.services import instruction_store


def test_chat_md_index_name_from_settings():
    assert instruction_store.chat_md_index_name() == get_settings().es_index_chat_md


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
    def __init__(
        self,
        *,
        exists_value: bool = True,
        get_response: dict | None = None,
        get_raises: Exception | None = None,
    ):
        self.indices = FakeIndices(exists_value=exists_value)
        self.index_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self._get_response = get_response
        self._get_raises = get_raises

    async def index(self, *, index: str, id: str, body):
        self.index_calls.append({"index": index, "id": id, "body": body})

    async def get(self, *, index: str, id: str):
        self.get_calls.append({"index": index, "id": id})
        if self._get_raises:
            raise self._get_raises
        return self._get_response or {"_source": {}}


@pytest.mark.asyncio
async def test_get_user_md_returns_stored_markdown():
    es = FakeES(
        exists_value=True,
        get_response={"_source": {"markdown": "# 사용자 지침\n- 친근한 말투"}},
    )
    md = await instruction_store.get_user_md("alice", client=es)
    assert "친근한 말투" in md


@pytest.mark.asyncio
async def test_get_user_md_returns_empty_when_doc_missing():
    """A user who has never set instructions should not raise — return ''."""
    es = FakeES(
        exists_value=True,
        get_raises=NotFoundError("not found", meta=None, body=None),
    )
    md = await instruction_store.get_user_md("alice", client=es)
    assert md == ""


@pytest.mark.asyncio
async def test_get_user_md_returns_empty_when_index_missing():
    es = FakeES(exists_value=False)
    md = await instruction_store.get_user_md("alice", client=es)
    assert md == ""
    assert es.get_calls == []  # never reached the get


@pytest.mark.asyncio
async def test_get_user_md_returns_empty_for_empty_user():
    md = await instruction_store.get_user_md("", client=FakeES())
    assert md == ""


@pytest.mark.asyncio
async def test_update_user_md_upserts_with_user_id_as_doc_id():
    es = FakeES(exists_value=True)
    await instruction_store.update_user_md(
        "alice", "# 사용자 지침\n- 한 문단으로", client=es
    )
    assert len(es.index_calls) == 1
    call = es.index_calls[0]
    assert call["id"] == "alice"
    assert call["index"] == instruction_store.chat_md_index_name()
    body = call["body"]
    assert body["user_id"] == "alice"
    assert "한 문단으로" in body["markdown"]
    assert "updated_at" in body


@pytest.mark.asyncio
async def test_update_user_md_skips_for_empty_user():
    es = FakeES()
    await instruction_store.update_user_md("", "anything", client=es)
    assert es.index_calls == []


@pytest.mark.asyncio
async def test_get_user_md_swallows_arbitrary_es_errors():
    """Generic exceptions must degrade to '' so chat keeps working."""

    class BoomES:
        class indices:
            @staticmethod
            async def exists(*, index):
                return True

        async def get(self, *, index, id):
            raise RuntimeError("ES is down")

    md = await instruction_store.get_user_md("alice", client=BoomES())
    assert md == ""
