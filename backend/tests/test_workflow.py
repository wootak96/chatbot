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
            # index_route (1 sub-query)
            '{"indices": ["elasticsearch"]}',
            # query_rewrite (1 plan: 1 sub × 1 routed index)
            '{"keywords": "Elasticsearch reciprocal rank fusion", "semantic": "mechanism of Reciprocal Rank Fusion in Elasticsearch"}',
            # metadata_extract
            '{"source": null, "category": null, "date_range": null}',
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
            # index_route (now before rewrite)
            '{"indices": ["kafka"]}',
            # query_rewrite (1 plan)
            '{"keywords": "Kafka consumer group", "semantic": "definition of Kafka consumer group"}',
            '{"source": null, "category": null, "date_range": null}',
            # first self_check short-circuits on empty candidates (no LLM call),
            # so retry path: query_variate (1 LLM call per plan)
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
            '{"indices": ["kafka"]}',
            '{"keywords": "Kafka topic partition", "semantic": "concept of Kafka topic partitions"}',
            '{"source": null, "category": null, "date_range": null}',
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
            # index_route x2 (one per sub-query, now before rewrite)
            '{"indices": ["elasticsearch"]}',
            '{"indices": ["kafka"]}',
            # query_rewrite x2 (one per plan: each sub × 1 routed index)
            '{"keywords": "Elasticsearch RRF", "semantic": "how Elasticsearch RRF works"}',
            '{"keywords": "Kafka consumer group", "semantic": "definition of Kafka consumer group"}',
            # metadata_extract (single)
            '{"source": null, "category": null, "date_range": null}',
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
async def test_workflow_fans_out_per_routed_index(stub_judge, stub_generator):
    """Single sub-query routed to confluence + elasticsearch fans out into
    two index-specific search plans with different language policies."""
    stub_judge(
        [
            '{"intent": "question", "resolved_query": "ES 클러스터 운영"}',
            '{"search_intent": "lookup"}',
            '{"sub_queries": ["ES 클러스터 운영"]}',
            # index_route routes the single sub-query to BOTH es_docs and confluence
            '{"indices": ["elasticsearch", "confluence"]}',
            # query_rewrite is now called twice — one per (sub, idx) plan.
            # English rewrite for elasticsearch_docs:
            '{"keywords": "Elasticsearch cluster operations", "semantic": "Elasticsearch cluster operations guide"}',
            # Korean rewrite for confluence_docs (technical term stays English):
            '{"keywords": "Elasticsearch 클러스터 운영 가이드", "semantic": "Elasticsearch 클러스터 운영 절차"}',
            '{"source": null, "category": null, "date_range": null}',
            '{"sufficient": true, "reason": "ok"}',
        ]
    )
    stub_generator(["답변 [1]."])

    captured: list[tuple[str, list[str]]] = []

    async def fake_search(
        *, bm25_query_text, indices, semantic_query_text=None, metadata_filters=None, **kw
    ):
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
            [{"role": "user", "content": "ES 클러스터 운영 어떻게 해?"}]
        )
        final = await workflow.ainvoke(state)
    finally:
        monkey.undo()

    # Two parallel ES calls, one per routed index, with index-specific BM25.
    by_index = {indices[0]: bm25 for bm25, indices in captured}
    assert "Elasticsearch cluster operations" == by_index["elasticsearch_docs"]
    assert "클러스터" in by_index["confluence_docs"]
    # search_plans should reflect the fan-out
    plans = final["search_plans"]
    assert len(plans) == 2
    assert {p["index"] for p in plans} == {"elasticsearch_docs", "confluence_docs"}
    assert all(p["sub_query_idx"] == 0 for p in plans)


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
async def test_workflow_search_intent_count_ambiguous_searches_all(
    stub_judge, monkeypatch
):
    """Meta-collection question (no domain keyword) short-circuits to all
    configured indices without an LLM routing call.
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
        return {
            "elasticsearch_docs": 50,
            "kafka_docs": 30,
            "confluence_docs": 20,
        }

    from app.graph.nodes import es_count as es_count_mod

    monkeypatch.setattr(es_count_mod, "count_documents", fake_count)

    workflow = build_workflow()
    state = initial_state([{"role": "user", "content": "전체 문서 몇 개?"}])
    final = await workflow.ainvoke(state)

    assert set(captured["indices"]) == {
        "elasticsearch_docs",
        "kafka_docs",
        "confluence_docs",
    }
    assert "100" in final["final_answer"]


@pytest.mark.asyncio
async def test_workflow_debugging_path_replays_logs(
    stub_judge, stub_generator, monkeypatch
):
    """`debugging` intent short-circuits the search pipeline and goes straight
    to debug_explain, which replays recent turns from {user_id}_logs."""
    stub_judge(['{"intent": "debugging"}'])
    stub_generator(["[Turn 1] Kafka 답변은 kafka_docs로 라우팅됐고..."])

    canned_turns = [
        {
            "question": "Kafka 뭐야?",
            "intent": "question",
            "search_intent": "lookup",
            "sub_queries": ["Kafka"],
            "target_indices": ["kafka_docs"],
            "search_plans": [
                {
                    "sub_query": "Kafka",
                    "index": "kafka_docs",
                    "bm25": "Kafka",
                    "semantic": "definition of Kafka",
                }
            ],
            "sufficient": True,
            "sufficiency_reason": "OK",
            "final_answer": "Kafka는... [1]",
            "sources": [{"url": "u/k", "title": "Kafka"}],
            "progress_log": "🔍 질문 분석 중...",
        }
    ]

    async def fake_fetch(user_id, *, session_id="", n=3, client=None):
        return canned_turns

    from app.graph.nodes import debug_explain as debug_explain_mod

    monkeypatch.setattr(debug_explain_mod, "fetch_recent_turns", fake_fetch)

    workflow = build_workflow()
    state = initial_state(
        [{"role": "user", "content": "왜 답변이 이렇게 나왔어?"}],
        user_id="alice",
        session_id="sess-1",
    )
    final = await workflow.ainvoke(state)

    assert final["intent"] == "debugging"
    assert "Turn 1" in final["final_answer"]
    # No retrieval pipeline ran on the debugging branch.
    assert final.get("candidates", []) == []
    assert final.get("search_plans", []) == []


@pytest.mark.asyncio
async def test_search_intent_list_is_disabled(stub_judge):
    """`list` is currently disabled — even when the judge returns it, the node
    must coerce search_intent back to `lookup` so es_list never runs."""
    from app.graph.nodes.search_intent import search_intent

    stub_judge(['{"search_intent": "list"}'])
    state = initial_state([{"role": "user", "content": "어떤 문서 있어?"}])
    state["resolved_query"] = "어떤 문서 있어?"
    out = await search_intent(state)
    assert out["search_intent"] == "lookup"
