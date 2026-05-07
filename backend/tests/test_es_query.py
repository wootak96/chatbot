"""ES query DSL construction tests (no network)."""

from app.config import Settings
from app.services.elasticsearch_client import build_rrf_query


def _settings_with_title() -> Settings:
    return Settings(es_field_title="title", es_field_content="content", es_field_semantic="content_embedding")


def _settings_no_title() -> Settings:
    """Both title fields disabled → single-field fallback on content."""
    return Settings(
        es_field_title="",
        es_field_ancestors_title="",
        es_field_content="content",
        es_field_semantic="content_embedding",
    )


def test_rrf_query_shape_with_title():
    s = _settings_with_title()
    q = build_rrf_query("Elasticsearch RRF", settings=s)
    assert "retriever" in q
    rrf = q["retriever"]["rrf"]
    assert rrf["rank_constant"] == s.retrieval_rank_constant
    assert rrf["rank_window_size"] == s.retrieval_rank_window
    assert len(rrf["retrievers"]) == 2

    bm25_q = rrf["retrievers"][0]["standard"]["query"]["multi_match"]
    assert bm25_q["query"] == "Elasticsearch RRF"
    assert "title^2" in bm25_q["fields"]
    assert "ancestors.title^1.5" in bm25_q["fields"]
    assert "content" in bm25_q["fields"]
    # `lenient` lets the cross-index search ignore ancestors.title on
    # indices (ES/Kafka docs) that don't map that field.
    assert bm25_q.get("lenient") is True

    sem_q = rrf["retrievers"][1]["standard"]["query"]["semantic"]
    assert sem_q["field"] == s.es_field_semantic
    assert sem_q["query"] == "Elasticsearch RRF"


def test_rrf_query_shape_content_only():
    """When title field is empty, BM25 should fall back to single-field match on content."""
    s = _settings_no_title()
    q = build_rrf_query("Elasticsearch RRF", settings=s)
    bm25_q = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]
    assert "match" in bm25_q
    assert bm25_q["match"]["content"]["query"] == "Elasticsearch RRF"
    assert "multi_match" not in bm25_q


def test_rrf_query_with_filters():
    s = _settings_with_title()
    q = build_rrf_query(
        "kafka consumer group",
        settings=s,
        metadata_filters={
            "category": ["guide", "ref"],
            "source": "kafka",
            "date_range": {"gte": "2024-01-01"},
        },
    )
    bm25 = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]
    assert "bool" in bm25
    filters = bm25["bool"]["filter"]
    # one terms filter, one term filter, one range
    has_terms = any("terms" in f for f in filters)
    has_term = any("term" in f for f in filters)
    has_range = any("range" in f for f in filters)
    assert has_terms and has_term and has_range


def test_rrf_query_empty_filters_omitted():
    s = _settings_with_title()
    q = build_rrf_query("test", settings=s, metadata_filters={})
    inner = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]
    assert "multi_match" in inner  # no bool wrapper
    assert "bool" not in inner


def test_rrf_query_distinct_bm25_and_semantic_text():
    """BM25 retriever uses keywords; semantic retriever uses the natural form."""
    s = _settings_with_title()
    q = build_rrf_query(
        "Elasticsearch RRF reciprocal rank fusion",
        semantic_query_text="mechanism of Reciprocal Rank Fusion in Elasticsearch",
        settings=s,
    )
    bm25_q = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]["multi_match"]
    sem_q = q["retriever"]["rrf"]["retrievers"][1]["standard"]["query"]["semantic"]
    assert bm25_q["query"] == "Elasticsearch RRF reciprocal rank fusion"
    assert sem_q["query"] == "mechanism of Reciprocal Rank Fusion in Elasticsearch"


def test_size_uses_top_k():
    s = _settings_with_title()
    q = build_rrf_query("x", settings=s)
    assert q["size"] == s.retrieval_top_k
    q2 = build_rrf_query("x", settings=s, size=3)
    assert q2["size"] == 3


def test_rrf_query_ancestors_title_field_omitted_when_setting_empty():
    """If the ancestors.title field is unconfigured (empty string), it must
    not show up in the BM25 multi_match — only title^2 + content remain."""
    s = Settings(
        es_field_title="title",
        es_field_content="content",
        es_field_semantic="content_embedding",
        es_field_ancestors_title="",
    )
    q = build_rrf_query("x", settings=s)
    bm25 = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]["multi_match"]
    assert "title^2" in bm25["fields"]
    assert "content" in bm25["fields"]
    assert not any("ancestors" in f for f in bm25["fields"])


def test_rrf_query_three_fields_total():
    """Default settings → BM25 multi_match searches exactly 3 fields:
    title (^2), ancestors.title (^1.5), content."""
    s = _settings_with_title()
    q = build_rrf_query("x", settings=s)
    bm25 = q["retriever"]["rrf"]["retrievers"][0]["standard"]["query"]["multi_match"]
    assert len(bm25["fields"]) == 3
