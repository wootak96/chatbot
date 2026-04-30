"""Shared pytest fixtures.

We never call the real LLM or Elasticsearch in tests. Two main fixtures:
- `mock_judge_llm`: replaces app.services.llm_factory.get_judge_llm with a
  programmable async stub that returns a queue of canned JSON strings.
- `mock_generator_llm`: same idea for the generator LLM, plus event-stream support.
- `mock_es`: monkeypatches `hybrid_search` to return canned docs.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk


class StubLLM:
    """Returns canned responses to ainvoke/astream. Records calls."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[Any] = []

    def _next(self) -> str:
        if not self._responses:
            return ""
        return self._responses.pop(0)

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self._next())

    async def astream(self, messages, *args, **kwargs):
        self.calls.append(messages)
        text = self._next()
        for ch in text:
            await asyncio.sleep(0)
            yield AIMessageChunk(content=ch)


@pytest.fixture
def stub_judge(monkeypatch):
    """Returns a function: install(responses) -> StubLLM."""

    def _install(responses: list[str]) -> StubLLM:
        from app.services import llm_factory

        stub = StubLLM(responses)
        monkeypatch.setattr(llm_factory, "get_judge_llm", lambda: stub)
        # Also patch the module-bound reference each node imported.
        from app.graph.nodes import (
            query_analyze,
            query_decompose,
            query_reform,
            query_rewrite,
            query_variate,
            search_intent,
            metadata_extract,
            index_route,
            self_check,
        )

        for mod in (
            query_analyze,
            query_decompose,
            query_reform,
            query_rewrite,
            query_variate,
            search_intent,
            metadata_extract,
            index_route,
            self_check,
        ):
            monkeypatch.setattr(mod, "get_judge_llm", lambda s=stub: s, raising=True)
        return stub

    return _install


@pytest.fixture
def stub_generator(monkeypatch):
    def _install(responses: list[str]) -> StubLLM:
        from app.services import llm_factory
        from app.graph.nodes import generate as generate_node
        from app.graph.nodes import general_chat as general_chat_node
        from app.graph.nodes import debug_explain as debug_explain_node

        stub = StubLLM(responses)
        monkeypatch.setattr(llm_factory, "get_generator_llm", lambda: stub)
        for mod in (generate_node, general_chat_node, debug_explain_node):
            monkeypatch.setattr(mod, "get_generator_llm", lambda s=stub: s, raising=True)
        return stub

    return _install


@pytest.fixture
def stub_es(monkeypatch):
    """Replaces hybrid_search with an async stub that returns canned docs."""

    def _install(docs_per_query: list[list[dict]] | list[dict]):
        # Either a flat list of docs (return same for every call) or a list-of-lists.
        if docs_per_query and isinstance(docs_per_query[0], dict):
            queue = [list(docs_per_query)]
        else:
            queue = [list(x) for x in docs_per_query]

        call_counter = {"n": 0}

        async def fake_hybrid_search(*args, **kwargs):
            i = call_counter["n"]
            call_counter["n"] += 1
            if not queue:
                return []
            return queue[i % len(queue)]

        from app.graph.nodes import hybrid_retrieve as hr_mod
        from app.services import elasticsearch_client as es_mod

        monkeypatch.setattr(es_mod, "hybrid_search", fake_hybrid_search)
        monkeypatch.setattr(hr_mod, "hybrid_search", fake_hybrid_search)
        return call_counter

    return _install
