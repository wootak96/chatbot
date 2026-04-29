"""End-to-end LangGraph workflow tests with all I/O mocked."""

import pytest

from app.graph.state import initial_state
from app.graph.workflow import build_workflow


@pytest.mark.asyncio
async def test_workflow_full_question_path(stub_judge, stub_generator, stub_es):
    stub_judge(
        [
            # query_analyze
            '{"intent": "question", "resolved_query": "Elasticsearch RRF 동작 원리"}',
            # search_intent
            '{"search_intent": "lookup"}',
            # query_decompose
            '{"sub_queries": ["Elasticsearch RRF 동작 원리"]}',
            # query_rewrite (1)
            '{"keywords": "Elasticsearch reciprocal rank fusion", "semantic": "mechanism of Reciprocal Rank Fusion in Elasticsearch"}',
            # metadata_extract
            '{"source": null, "category": null, "date_range": null}',
            # index_route
            '{"indices": ["elasticsearch"]}',
            # self_check
            '{"sufficient": true, "reason": "OK"}',
        ]
    )
    stub_generator(["RRF는 BM25와 kNN의 순위를 결합하는 방법입니다 [1]."])
    stub_es(
        [[{"id": "d1", "title": "RRF Guide", "url": "https://elastic.co/rrf", "content": "RRF combines ranks."}]]
    )

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "ES RRF 어떻게 동작해?"}])
    final = await workflow.ainvoke(state)

    assert final["intent"] == "question"
    assert final["target_indices_per_query"] == [["elasticsearch_docs"]]
    assert final["sufficient"] is True
    assert "RRF" in final["final_answer"]
    assert final["sources"] == [{"url": "https://elastic.co/rrf", "title": "RRF Guide"}]


@pytest.mark.asyncio
async def test_workflow_chitchat_skips_retrieval(stub_judge, stub_generator, stub_es):
    stub_judge(['{"intent": "chitchat", "resolved_query": "안녕"}'])
    stub_generator(["안녕하세요! 사내 ES/Kafka 문서 질문을 도와드립니다."])
    counter = stub_es([[]])

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "안녕"}])
    final = await workflow.ainvoke(state)

    assert final["intent"] == "chitchat"
    assert final["candidates"] == []
    assert counter["n"] == 0  # ES never called
    assert "안녕" in final["final_answer"]


@pytest.mark.asyncio
async def test_workflow_retry_on_insufficient(stub_judge, stub_generator, stub_es):
    stub_judge(
        [
            '{"intent": "question", "resolved_query": "k"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["k"]}',
            '{"keywords": "Kafka consumer group", "semantic": "definition of Kafka consumer group"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"indices": ["kafka"]}',
            # first self_check short-circuits on empty candidates (no LLM call),
            # so retry path: query_variate (1 LLM call per sub-query)
            '{"keywords": "Kafka consumer group rebalance", "semantic": "how Kafka consumer groups rebalance partitions"}',
            # second self_check after retry sees real candidates
            '{"sufficient": true, "reason": "now ok"}',
        ]
    )
    stub_generator(["답변 [1]."])
    counter = stub_es(
        [
            [],  # first retrieval: empty
            [{"id": "k1", "title": "K", "url": "u/k", "content": "kafka"}],  # second
        ]
    )

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "kafka 컨슈머 그룹"}])
    final = await workflow.ainvoke(state)

    assert counter["n"] == 2  # retry happened
    assert final["sufficient"] is True
    assert final["sources"] == [{"url": "u/k", "title": "K"}]


@pytest.mark.asyncio
async def test_workflow_routes_to_kafka(stub_judge, stub_generator, stub_es):
    stub_judge(
        [
            '{"intent": "question", "resolved_query": "Kafka 토픽 파티션"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["Kafka 토픽 파티션"]}',
            '{"keywords": "Kafka topic partition", "semantic": "concept of Kafka topic partitions"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"indices": ["kafka"]}',
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["답."])
    captured: dict = {}

    async def fake_search(*, bm25_query_text, indices, semantic_query_text=None, metadata_filters=None, **kw):
        captured["indices"] = indices
        return [{"id": "k1", "title": "K", "url": "u", "content": "c"}]

    from app.graph.nodes import hybrid_retrieve as hr_mod

    import pytest as _p

    monkey = _p.MonkeyPatch()
    monkey.setattr(hr_mod, "hybrid_search", fake_search)
    try:
        workflow = build_workflow()
        state = initial_state([{"role": "user", "content": "토픽 파티션 뭐야?"}])
        await workflow.ainvoke(state)
    finally:
        monkey.undo()

    assert captured["indices"] == ["kafka_docs"]


@pytest.mark.asyncio
async def test_workflow_routes_per_subquery(stub_judge, stub_generator):
    """Decompose into 2 sub-queries; each should route to its own index."""
    stub_judge(
        [
            # query_analyze
            '{"intent": "question", "resolved_query": "ES와 Kafka 비교"}',
            # search_intent
            '{"search_intent": "lookup"}',
            # query_decompose -> 2 sub-queries
            '{"sub_queries": ["Elasticsearch RRF", "Kafka consumer group"]}',
            # query_rewrite x2
            '{"keywords": "Elasticsearch RRF", "semantic": "how Elasticsearch RRF works"}',
            '{"keywords": "Kafka consumer group", "semantic": "definition of Kafka consumer group"}',
            # metadata_extract (single)
            '{"source": null, "category": null, "date_range": null}',
            # index_route x2
            '{"indices": ["elasticsearch"]}',
            '{"indices": ["kafka"]}',
            # self_check
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["답변 [1] [2]."])

    captured: list[tuple[str, list[str]]] = []

    async def fake_search(*, bm25_query_text, indices, semantic_query_text=None, metadata_filters=None, **kw):
        captured.append((bm25_query_text, list(indices)))
        return [
            {
                "id": bm25_query_text,
                "title": bm25_query_text,
                "url": "u/" + bm25_query_text,
                "content": "c",
            }
        ]

    from app.graph.nodes import hybrid_retrieve as hr_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(hr_mod, "hybrid_search", fake_search)
    try:
        workflow = build_workflow()
        state = initial_state(
            [{"role": "user", "content": "ES랑 Kafka 차이가 뭐야?"}]
        )
        final = await workflow.ainvoke(state)
    finally:
        monkey.undo()

    # Each sub-query should hit its routed index, not both.
    captured_map = dict(captured)
    assert captured_map["Elasticsearch RRF"] == ["elasticsearch_docs"]
    assert captured_map["Kafka consumer group"] == ["kafka_docs"]
    assert final["target_indices_per_query"] == [
        ["elasticsearch_docs"],
        ["kafka_docs"],
    ]
    assert len(final["sources"]) == 2


@pytest.mark.asyncio
async def test_workflow_search_intent_count(stub_judge, monkeypatch):
    """count search_intent routes to the relevant index and returns its count.

    For "ES 문서 몇 개?" the router picks elasticsearch only — the kafka
    index must not appear in the count or the final answer.
    """
    stub_judge(
        [
            '{"intent": "question"}',                # query_analyze
            '{"search_intent": "count"}',            # search_intent
            '{"indices": ["elasticsearch"]}',        # route_query inside es_count
        ]
    )

    captured: dict = {}

    async def fake_count(*, indices, metadata_filters=None, client=None):
        captured["indices"] = list(indices)
        return {idx: 50 for idx in indices}

    from app.graph.nodes import es_count as es_count_mod

    monkeypatch.setattr(es_count_mod, "count_documents", fake_count)

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "ES 문서 몇 개?"}])
    final = await workflow.ainvoke(state)

    assert final["search_intent"] == "count"
    assert captured["indices"] == ["elasticsearch_docs"]
    assert "50" in final["final_answer"]
    assert "elasticsearch_docs" in final["final_answer"]
    assert "kafka_docs" not in final["final_answer"]
    assert final["sources"] == []


@pytest.mark.asyncio
async def test_workflow_search_intent_count_kafka(stub_judge, monkeypatch):
    """Routing to kafka only excludes the elasticsearch index from the count."""
    stub_judge(
        [
            '{"intent": "question"}',
            '{"search_intent": "count"}',
            '{"indices": ["kafka"]}',
        ]
    )

    captured: dict = {}

    async def fake_count(*, indices, metadata_filters=None, client=None):
        captured["indices"] = list(indices)
        return {idx: 42 for idx in indices}

    from app.graph.nodes import es_count as es_count_mod

    monkeypatch.setattr(es_count_mod, "count_documents", fake_count)

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "kafka 문서 몇 개야?"}])
    final = await workflow.ainvoke(state)

    assert captured["indices"] == ["kafka_docs"]
    assert "42" in final["final_answer"]
    assert "kafka_docs" in final["final_answer"]
    assert "elasticsearch_docs" not in final["final_answer"]


@pytest.mark.asyncio
async def test_workflow_search_intent_count_ambiguous_searches_both(
    stub_judge, monkeypatch
):
    """Meta-collection question (no domain keyword) short-circuits to both
    indices without an LLM routing call.
    """
    # No third stub entry — route_query short-circuits when no domain term
    # is present in the query, so the routing LLM call is skipped entirely.
    stub_judge(
        [
            '{"intent": "question"}',
            '{"search_intent": "count"}',
        ]
    )

    captured: dict = {}

    async def fake_count(*, indices, metadata_filters=None, client=None):
        captured["indices"] = list(indices)
        return {"elasticsearch_docs": 50, "kafka_docs": 30}

    from app.graph.nodes import es_count as es_count_mod

    monkeypatch.setattr(es_count_mod, "count_documents", fake_count)

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "전체 문서 몇 개?"}])
    final = await workflow.ainvoke(state)

    assert set(captured["indices"]) == {"elasticsearch_docs", "kafka_docs"}
    assert "80" in final["final_answer"]


@pytest.mark.asyncio
async def test_workflow_search_intent_list_no_title_field(stub_judge, monkeypatch):
    """list search_intent gracefully degrades when title field is unmapped."""
    stub_judge(
        [
            '{"intent": "question", "resolved_query": "어떤 문서 있어?"}',
            '{"search_intent": "list"}',
        ]
    )

    # Force the degraded code path regardless of what .env / ES_FIELD_TITLE
    # is set to in the test host. This is exactly the scenario where the
    # index has no title field mapped.
    from app import config as app_config

    def _no_title_settings() -> app_config.Settings:
        s = app_config.Settings()
        return s.model_copy(update={"es_field_title": ""})

    monkeypatch.setattr(app_config, "get_settings", _no_title_settings)
    # es_list imports get_settings at call time via `from app.config import get_settings`
    # so we must patch the symbol there too.
    from app.graph.nodes import es_list as es_list_mod

    monkeypatch.setattr(es_list_mod, "get_settings", _no_title_settings)

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "어떤 문서 있어?"}])
    final = await workflow.ainvoke(state)

    assert final["search_intent"] == "list"
    assert "title" in final["final_answer"]
    assert final["sources"] == []
