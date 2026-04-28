"""HTTP-level tests against the FastAPI app."""

import json

from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_chat_ui():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "<!doctype html>" in body.lower() or "<!DOCTYPE html>" in body
    # UI should reference the chat endpoint and model id
    assert "/v1/chat/completions" in body
    assert "rag-chatbot" in body


def test_info_endpoint():
    client = TestClient(app)
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "rag-chatbot-backend"
    assert "POST /v1/chat/completions" in body["endpoints"]["chat"]
    assert body["endpoints"]["ui"] == "GET /"


def test_health_endpoint(monkeypatch):
    """/health pings ES; mock it to avoid network calls in tests."""
    from app.services import elasticsearch_client as es_mod
    from unittest.mock import AsyncMock, MagicMock

    fake_client = MagicMock()
    fake_client.info = AsyncMock(
        return_value={"cluster_name": "test-cluster", "version": {"number": "8.15.0"}}
    )
    fake_client.indices = MagicMock()
    fake_client.indices.exists = AsyncMock(return_value=True)
    monkeypatch.setattr(es_mod, "get_es_client", lambda: fake_client)
    es_mod.get_es_client.cache_clear() if hasattr(es_mod.get_es_client, "cache_clear") else None

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["elasticsearch"]["reachable"] is True
    assert body["elasticsearch"]["indices"]["elasticsearch_docs"] is True
    assert body["elasticsearch"]["indices"]["kafka_docs"] is True
    assert body["llm"]["provider"] in {"openai", "azure"}
    assert "model" in body["llm"]
    assert "api_key_configured" in body["llm"]


def test_health_endpoint_reports_es_failure(monkeypatch):
    from app.services import elasticsearch_client as es_mod
    from unittest.mock import MagicMock

    fake_client = MagicMock()

    async def boom():
        raise ConnectionError("connection refused")

    fake_client.info = boom
    monkeypatch.setattr(es_mod, "get_es_client", lambda: fake_client)

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["elasticsearch"]["reachable"] is False
    assert "ConnectionError" in body["elasticsearch"]["error"]


def test_models_endpoint():
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert any(m["id"] == "rag-chatbot" for m in body["data"])


def test_chat_completions_propagates_bearer_to_contextvar(
    monkeypatch, stub_judge, stub_generator, stub_es
):
    """Authorization: Bearer <key> should be picked up and set the LLM contextvar
    for the duration of the request."""
    captured: list[str | None] = []
    from app.services import llm_factory

    real_set = llm_factory.set_api_key
    real_resolve = llm_factory.resolve_api_key

    def spy_set(key):
        captured.append(key)
        real_set(key)

    # Patch in both the source module and chat.py's bound import.
    monkeypatch.setattr(llm_factory, "set_api_key", spy_set)
    from app.api import chat as chat_mod

    monkeypatch.setattr(chat_mod, "set_api_key", spy_set)

    stub_judge(
        [
            '{"intent": "question", "resolved_query": "x"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["x"]}',
            '{"keywords": "x", "semantic": "x"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"indices": ["elasticsearch"]}',
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["답."])
    stub_es([[{"id": "d", "title": "t", "url": "u", "content": "c"}]])

    client = TestClient(app)
    payload = {
        "model": "rag-chatbot",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer user-supplied-secret"},
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())  # drain

    assert captured == ["user-supplied-secret"]


def test_chat_completions_dummy_key_falls_back_to_env(
    monkeypatch, stub_judge, stub_generator, stub_es
):
    captured: list[str | None] = []
    from app.services import llm_factory

    real_set = llm_factory.set_api_key

    def spy_set(key):
        captured.append(key)
        real_set(key)

    monkeypatch.setattr(llm_factory, "set_api_key", spy_set)
    from app.api import chat as chat_mod

    monkeypatch.setattr(chat_mod, "set_api_key", spy_set)

    stub_judge(
        [
            '{"intent": "question", "resolved_query": "x"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["x"]}',
            '{"keywords": "x", "semantic": "x"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"indices": ["elasticsearch"]}',
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["답."])
    stub_es([[{"id": "d", "title": "t", "url": "u", "content": "c"}]])

    client = TestClient(app)
    payload = {
        "model": "rag-chatbot",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer dummy-key"},
    ) as r:
        assert r.status_code == 200
        b"".join(r.iter_bytes())

    # Dummy placeholder must be filtered to None so settings default is used.
    assert captured == [None]


def test_chat_completions_streams_sse(stub_judge, stub_generator, stub_es):
    stub_judge(
        [
            '{"intent": "question", "resolved_query": "ES RRF"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["ES RRF"]}',
            '{"keywords": "Elasticsearch RRF", "semantic": "definition of Elasticsearch RRF"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"indices": ["elasticsearch"]}',
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["RRF는 순위 결합 [1]."])
    stub_es([[{"id": "d1", "title": "T", "url": "https://x/y", "content": "c"}]])

    client = TestClient(app)
    payload = {
        "model": "rag-chatbot",
        "messages": [{"role": "user", "content": "RRF가 뭐야?"}],
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes()).decode("utf-8")

    # SSE structure
    assert body.startswith("data: ")
    assert body.rstrip().endswith("data: [DONE]")

    # Parse SSE chunks (excluding the [DONE] sentinel)
    chunks = []
    for raw in body.split("\n\n"):
        raw = raw.strip()
        if not raw or raw == "data: [DONE]":
            continue
        assert raw.startswith("data: ")
        chunks.append(json.loads(raw[len("data: "):]))

    # First chunk has role=assistant
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Concatenate all delta content
    full_text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    # Progress messages from at least a few nodes
    assert "🔍" in full_text  # query_analyze
    assert "📚" in full_text  # hybrid_retrieve
    # generate node no longer emits a progress message — answer streams directly
    # Separator before answer
    assert "─" in full_text
    # Final answer body
    assert "RRF" in full_text
    # Hidden CITES marker carries source URLs (no visible **출처** block)
    assert "<!--CITES:" in full_text
    assert "https://x/y" in full_text
    assert "**출처**" not in full_text

    # Last non-DONE chunk should have finish_reason="stop"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
