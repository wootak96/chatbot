"""Tests for the sessions sidebar API.

Both endpoints are read-only views over the chat_logs index. We swap in a
fake ES client that records the search bodies so we can assert the queries
filter by user_id (and session_id where appropriate) and that the returned
shape matches what the UI expects.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class FakeIndices:
    def __init__(self, exists_value: bool = True):
        self.exists_value = exists_value

    async def exists(self, *, index: str):
        return self.exists_value


class FakeES:
    def __init__(
        self,
        response: dict,
        exists: bool = True,
        delete_response: dict | None = None,
        delete_raises: Exception | None = None,
    ):
        self.indices = FakeIndices(exists_value=exists)
        self.search_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self._response = response
        self._delete_response = delete_response or {"deleted": 0}
        self._delete_raises = delete_raises

    async def search(self, *, index: str, body):
        self.search_calls.append({"index": index, "body": body})
        return self._response

    async def delete_by_query(self, *, index: str, body, **kwargs):
        self.delete_calls.append({"index": index, "body": body, "kwargs": kwargs})
        if self._delete_raises:
            raise self._delete_raises
        return self._delete_response


@pytest.fixture
def app_with_fake_es(monkeypatch):
    """Returns a function `install(response_for_each_call: dict)` that swaps
    `get_es_client` with a FakeES yielding the given response."""

    def _install(
        response: dict,
        *,
        exists: bool = True,
        delete_response: dict | None = None,
        delete_raises: Exception | None = None,
    ) -> FakeES:
        fake = FakeES(
            response=response,
            exists=exists,
            delete_response=delete_response,
            delete_raises=delete_raises,
        )
        from app.api import sessions as sessions_mod

        monkeypatch.setattr(sessions_mod, "get_es_client", lambda: fake)
        return fake

    return _install


def _client():
    from app.main import app

    return TestClient(app)


def test_list_sessions_aggregates_distinct_session_ids(app_with_fake_es):
    fake = app_with_fake_es(
        {
            "aggregations": {
                "sessions": {
                    "buckets": [
                        {
                            "key": "sess-2",
                            "doc_count": 3,
                            "latest": {"value_as_string": "2026-05-04T12:00:00Z"},
                            "first_q": {
                                "hits": {
                                    "hits": [
                                        {"_source": {"question": "Kafka 토픽?"}}
                                    ]
                                }
                            },
                        },
                        {
                            "key": "sess-1",
                            "doc_count": 1,
                            "latest": {"value_as_string": "2026-05-04T08:00:00Z"},
                            "first_q": {
                                "hits": {
                                    "hits": [
                                        {"_source": {"question": "ES RRF는?"}}
                                    ]
                                }
                            },
                        },
                    ]
                }
            }
        }
    )
    client = _client()
    r = client.get("/v1/sessions?user_id=alice")
    assert r.status_code == 200
    data = r.json()
    assert [s["session_id"] for s in data["sessions"]] == ["sess-2", "sess-1"]
    assert data["sessions"][0]["title"] == "Kafka 토픽?"
    assert data["sessions"][0]["turn_count"] == 3
    # Verify the search body filters by user_id and orders sessions by latest.
    body = fake.search_calls[0]["body"]
    flt = body["query"]["bool"]["filter"]
    assert {"term": {"user_id": "alice"}} in flt
    assert body["aggs"]["sessions"]["terms"]["order"] == {"latest": "desc"}


def test_list_sessions_returns_empty_when_no_aggs(app_with_fake_es):
    app_with_fake_es({"aggregations": {"sessions": {"buckets": []}}})
    r = _client().get("/v1/sessions?user_id=alice")
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


def test_list_sessions_requires_user_id():
    r = _client().get("/v1/sessions")
    assert r.status_code == 422  # missing required query param


def test_get_session_messages_filters_by_user_and_session(app_with_fake_es):
    fake = app_with_fake_es(
        {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "question": "ES RRF가 뭐야?",
                            "final_answer": "RRF는 Reciprocal Rank Fusion ...",
                        }
                    },
                    {
                        "_source": {
                            "question": "어떻게 설정해?",
                            "final_answer": "retriever DSL에 rrf 블록 ...",
                        }
                    },
                ]
            }
        }
    )
    r = _client().get("/v1/sessions/sess-1/messages?user_id=alice")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    # Each turn produces 2 messages (user + assistant) in chronological order.
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "ES RRF가 뭐야?"
    assert msgs[1]["content"].startswith("RRF는")
    body = fake.search_calls[0]["body"]
    flt = body["query"]["bool"]["filter"]
    assert {"term": {"user_id": "alice"}} in flt
    assert {"term": {"session_id": "sess-1"}} in flt
    assert body["sort"][0]["timestamp"]["order"] == "asc"


def test_get_session_messages_empty_when_index_missing(app_with_fake_es):
    app_with_fake_es({"hits": {"hits": []}}, exists=False)
    r = _client().get("/v1/sessions/sess-x/messages?user_id=alice")
    assert r.status_code == 200
    assert r.json() == {"messages": []}


def test_delete_session_runs_delete_by_query_with_user_and_session(
    app_with_fake_es,
):
    fake = app_with_fake_es({}, delete_response={"deleted": 3})
    r = _client().delete("/v1/sessions/sess-1?user_id=alice")
    assert r.status_code == 200
    assert r.json() == {"deleted": 3}
    assert len(fake.delete_calls) == 1
    flt = fake.delete_calls[0]["body"]["query"]["bool"]["filter"]
    # Both filters present so a user can never delete another user's session.
    assert {"term": {"user_id": "alice"}} in flt
    assert {"term": {"session_id": "sess-1"}} in flt
    assert fake.delete_calls[0]["kwargs"].get("refresh") is True


def test_delete_session_returns_zero_when_index_missing(app_with_fake_es):
    fake = app_with_fake_es({}, exists=False)
    r = _client().delete("/v1/sessions/sess-x?user_id=alice")
    assert r.status_code == 200
    assert r.json() == {"deleted": 0}
    assert fake.delete_calls == []  # nothing to delete


def test_delete_session_requires_user_id():
    r = _client().delete("/v1/sessions/sess-1")
    assert r.status_code == 422


def test_delete_session_returns_500_on_es_error(app_with_fake_es):
    app_with_fake_es({}, delete_raises=RuntimeError("boom"))
    r = _client().delete("/v1/sessions/sess-1?user_id=alice")
    assert r.status_code == 500
