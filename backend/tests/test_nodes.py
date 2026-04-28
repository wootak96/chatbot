"""Per-node tests with mocked LLM."""

import pytest

from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes.generate import generate
from app.graph.nodes.hybrid_retrieve import hybrid_retrieve
from app.graph.nodes.index_route import index_route
from app.graph.nodes.metadata_extract import metadata_extract
from app.graph.nodes.query_analyze import query_analyze
from app.graph.nodes.query_decompose import query_decompose
from app.graph.nodes.query_reform import query_reform
from app.graph.nodes.query_rewrite import query_rewrite
from app.graph.nodes.self_check import self_check, should_retry


@pytest.mark.asyncio
async def test_query_analyze_question(stub_judge):
    stub_judge(['{"intent": "question"}'])
    state = {"current_query": "RRF가 뭐야?", "messages": []}
    out = await query_analyze(state)
    assert out["intent"] == "question"
    assert "resolved_query" not in out  # reform is now a separate node
    assert PROGRESS_KEY in out


@pytest.mark.asyncio
async def test_query_analyze_chitchat(stub_judge):
    stub_judge(['{"intent": "chitchat"}'])
    state = {"current_query": "안녕", "messages": []}
    out = await query_analyze(state)
    assert out["intent"] == "chitchat"


@pytest.mark.asyncio
async def test_query_analyze_invalid_intent_falls_back(stub_judge):
    stub_judge(['{"intent": "garbage"}'])
    out = await query_analyze({"current_query": "x", "messages": []})
    assert out["intent"] == "question"


@pytest.mark.asyncio
async def test_query_analyze_domain_words_override_general(stub_judge):
    """If LLM misclassifies a domain question as general, the regex safety net
    kicks in and routes it to the search path."""
    stub_judge(['{"intent": "general"}'])
    out = await query_analyze(
        {"current_query": "Elasticsearch랑 Kafka 비교해줘", "messages": []}
    )
    assert out["intent"] == "question"  # overridden from general


@pytest.mark.asyncio
async def test_query_analyze_no_domain_words_keeps_general(stub_judge):
    """No domain keywords -> the safety net does not interfere with general."""
    stub_judge(['{"intent": "general"}'])
    out = await query_analyze({"current_query": "오늘 날씨 어때?", "messages": []})
    assert out["intent"] == "general"


@pytest.mark.asyncio
async def test_query_reform_no_history_passthrough(stub_judge):
    """Empty history -> reform is a no-op; no LLM call is made."""
    stub = stub_judge([])
    out = await query_reform({"current_query": "BM25가 뭐야?", "messages": []})
    assert out["resolved_query"] == "BM25가 뭐야?"
    assert stub.calls == []


@pytest.mark.asyncio
async def test_query_reform_expands_followup(stub_judge):
    """Follow-up reference resolved using prior turn."""
    stub_judge(['{"reformed_query": "Elasticsearch RRF 설정 방법"}'])
    state = {
        "current_query": "어떻게 설정해?",
        "messages": [
            {"role": "user", "content": "Elasticsearch RRF가 뭐야?"},
            {"role": "assistant", "content": "RRF는 ..."},
            {"role": "user", "content": "어떻게 설정해?"},
        ],
    }
    out = await query_reform(state)
    assert "Elasticsearch RRF" in out["resolved_query"]
    assert "설정" in out["resolved_query"]


@pytest.mark.asyncio
async def test_query_reform_falls_back_to_current_on_empty(stub_judge):
    """If LLM returns empty reformed_query, fall back to current_query."""
    stub_judge(['{"reformed_query": ""}'])
    state = {
        "current_query": "어떻게?",
        "messages": [
            {"role": "user", "content": "Kafka는?"},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "어떻게?"},
        ],
    }
    out = await query_reform(state)
    assert out["resolved_query"] == "어떻게?"


@pytest.mark.asyncio
async def test_query_decompose_single(stub_judge):
    stub_judge(['{"sub_queries": ["Elasticsearch RRF 동작 원리"]}'])
    state = {"resolved_query": "ES RRF 어떻게 동작?", "intent": "question"}
    out = await query_decompose(state)
    assert len(out["sub_queries"]) == 1


@pytest.mark.asyncio
async def test_query_decompose_multi(stub_judge):
    stub_judge(['{"sub_queries": ["A", "B", "C", "D"]}'])
    state = {"resolved_query": "X", "intent": "question"}
    out = await query_decompose(state)
    # capped at 3
    assert len(out["sub_queries"]) == 3


@pytest.mark.asyncio
async def test_query_decompose_skipped_for_chitchat(stub_judge):
    stub_judge([])  # should not be called
    out = await query_decompose({"resolved_query": "안녕", "intent": "chitchat"})
    assert out["sub_queries"] == []


@pytest.mark.asyncio
async def test_query_rewrite_parallel(stub_judge):
    stub_judge(
        [
            '{"keywords": "Elasticsearch RRF reciprocal rank fusion", "semantic": "mechanism of Reciprocal Rank Fusion in Elasticsearch"}',
            '{"keywords": "Kafka consumer group", "semantic": "definition of Kafka consumer group"}',
        ]
    )
    state = {"sub_queries": ["RRF 어떻게?", "kafka cg"]}
    out = await query_rewrite(state)
    assert len(out["rewritten_queries"]) == 2
    assert len(out["semantic_queries"]) == 2
    # BM25 list = keywords-only English
    assert "Elasticsearch" in out["rewritten_queries"][0]
    assert "RRF" in out["rewritten_queries"][0]
    # Semantic list = full English natural form
    assert "Reciprocal Rank Fusion" in out["semantic_queries"][0]


@pytest.mark.asyncio
async def test_metadata_extract_with_filter(stub_judge):
    stub_judge(['{"source": ["elasticsearch"], "category": null, "date_range": {"gte": "2024-01-01"}}'])
    out = await metadata_extract({"resolved_query": "최근 ES 문서", "intent": "question"})
    f = out["metadata_filters"]
    assert f["source"] == ["elasticsearch"]
    assert "category" not in f  # null dropped
    assert f["date_range"] == {"gte": "2024-01-01"}


@pytest.mark.asyncio
async def test_index_route_per_subquery(stub_judge):
    stub_judge(
        [
            '{"indices": ["elasticsearch"]}',
            '{"indices": ["kafka"]}',
        ]
    )
    out = await index_route(
        {
            "rewritten_queries": ["ES RRF 동작", "Kafka consumer group"],
            "intent": "question",
        }
    )
    assert out["target_indices_per_query"] == [
        ["elasticsearch_docs"],
        ["kafka_docs"],
    ]


@pytest.mark.asyncio
async def test_index_route_picks_both_for_one_subquery(stub_judge):
    stub_judge(['{"indices": ["elasticsearch", "kafka"]}'])
    out = await index_route(
        {"rewritten_queries": ["ES와 Kafka 비교"], "intent": "question"}
    )
    assert len(out["target_indices_per_query"]) == 1
    assert set(out["target_indices_per_query"][0]) == {
        "elasticsearch_docs",
        "kafka_docs",
    }


@pytest.mark.asyncio
async def test_index_route_empty_falls_back_to_both(stub_judge):
    stub_judge(['{"indices": []}'])
    out = await index_route(
        {"rewritten_queries": ["모호한 질문"], "intent": "question"}
    )
    assert set(out["target_indices_per_query"][0]) == {
        "elasticsearch_docs",
        "kafka_docs",
    }


@pytest.mark.asyncio
async def test_index_route_invalid_falls_back(stub_judge):
    stub_judge(['{"indices": ["fake_index"]}'])
    out = await index_route({"rewritten_queries": ["x"], "intent": "question"})
    assert set(out["target_indices_per_query"][0]) == {
        "elasticsearch_docs",
        "kafka_docs",
    }


@pytest.mark.asyncio
async def test_index_route_no_subqueries(stub_judge):
    stub_judge([])  # not called
    out = await index_route({"rewritten_queries": [], "intent": "question"})
    assert out["target_indices_per_query"] == []


@pytest.mark.asyncio
async def test_hybrid_retrieve_merges_dedupes(stub_es):
    stub_es(
        [
            [{"id": "a", "title": "A", "url": "u/a", "content": "..."}],
            [
                {"id": "a", "title": "A", "url": "u/a", "content": "..."},
                {"id": "b", "title": "B", "url": "u/b", "content": "..."},
            ],
        ]
    )
    out = await hybrid_retrieve(
        {
            "rewritten_queries": ["q1", "q2"],
            "target_indices_per_query": [["elasticsearch_docs"], ["kafka_docs"]],
            "metadata_filters": {},
            "intent": "question",
        }
    )
    ids = {d["id"] for d in out["candidates"]}
    assert ids == {"a", "b"}


@pytest.mark.asyncio
async def test_hybrid_retrieve_skips_when_no_indices(stub_es):
    stub_es([[]])
    out = await hybrid_retrieve(
        {
            "rewritten_queries": ["q"],
            "target_indices_per_query": [[]],
            "intent": "question",
        }
    )
    assert out["candidates"] == []


@pytest.mark.asyncio
async def test_hybrid_retrieve_uses_per_subquery_indices(monkeypatch):
    captured: list[tuple[str, str, list[str]]] = []

    async def fake_search(
        *, bm25_query_text, semantic_query_text=None, indices, metadata_filters=None, **kw
    ):
        captured.append((bm25_query_text, semantic_query_text, list(indices)))
        return [
            {
                "id": bm25_query_text,
                "title": "T",
                "url": "u/" + bm25_query_text,
                "content": "c",
            }
        ]

    from app.graph.nodes import hybrid_retrieve as hr_mod

    monkeypatch.setattr(hr_mod, "hybrid_search", fake_search)

    out = await hybrid_retrieve(
        {
            "rewritten_queries": ["es q", "kafka q"],
            "semantic_queries": ["es semantic", "kafka semantic"],
            "target_indices_per_query": [
                ["elasticsearch_docs"],
                ["kafka_docs"],
            ],
            "metadata_filters": {},
            "intent": "question",
        }
    )
    assert sorted(captured) == [
        ("es q", "es semantic", ["elasticsearch_docs"]),
        ("kafka q", "kafka semantic", ["kafka_docs"]),
    ]
    assert {d["id"] for d in out["candidates"]} == {"es q", "kafka q"}


@pytest.mark.asyncio
async def test_self_check_sufficient(stub_judge):
    stub_judge(['{"sufficient": true, "reason": "OK"}'])
    out = await self_check(
        {
            "resolved_query": "x",
            "candidates": [{"id": "a", "title": "A", "url": "u", "content": "c"}],
            "retry_count": 0,
            "intent": "question",
        }
    )
    assert out["sufficient"] is True
    assert "retry_count" not in out


@pytest.mark.asyncio
async def test_self_check_insufficient_increments_retry(stub_judge):
    stub_judge(['{"sufficient": false, "reason": "off-topic"}'])
    out = await self_check(
        {
            "resolved_query": "x",
            "candidates": [{"id": "a", "title": "A", "url": "u", "content": "c"}],
            "retry_count": 0,
            "intent": "question",
        }
    )
    assert out["sufficient"] is False
    assert out["retry_count"] == 1


@pytest.mark.asyncio
async def test_self_check_no_candidates(stub_judge):
    stub_judge([])  # not called
    out = await self_check(
        {"resolved_query": "x", "candidates": [], "retry_count": 0, "intent": "question"}
    )
    assert out["sufficient"] is False
    assert out["retry_count"] == 1


def test_should_retry_branches():
    assert should_retry({"sufficient": True}) == "generate"
    assert should_retry({"sufficient": False, "retry_count": 0}) == "retry"
    # When retry budget is exhausted without sufficient evidence, route to
    # generate (which emits "해당 정보를 찾을 수 없습니다."). Domain questions
    # MUST stay grounded in the ES corpus and never fall back to general_chat.
    assert should_retry({"sufficient": False, "retry_count": 99}) == "generate"


@pytest.mark.asyncio
async def test_generate_with_docs(stub_generator):
    stub_generator(["답변 본문 [1] 입니다."])
    out = await generate(
        {
            "intent": "question",
            "resolved_query": "x",
            "candidates": [{"id": "a", "title": "A", "url": "u/a", "content": "c"}],
            "sufficient": True,
        }
    )
    assert "답변" in out["final_answer"]
    assert out["sources"] == [{"url": "u/a", "title": "A"}]


@pytest.mark.asyncio
async def test_generate_without_sufficient_returns_no_info(stub_generator):
    stub_generator([])  # should not be called
    out = await generate(
        {
            "intent": "question",
            "resolved_query": "x",
            "candidates": [{"id": "a"}],
            "sufficient": False,
        }
    )
    assert out["final_answer"] == "해당 정보를 찾을 수 없습니다."
    assert out["sources"] == []


@pytest.mark.asyncio
async def test_generate_chitchat(stub_generator):
    stub_generator(["안녕하세요. 사내 ES/Kafka 문서 질의응답을 돕습니다."])
    out = await generate({"intent": "chitchat", "resolved_query": "hi"})
    assert "안녕" in out["final_answer"]
    assert out["sources"] == []
