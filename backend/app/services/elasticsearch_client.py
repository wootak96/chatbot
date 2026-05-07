"""Elasticsearch hybrid retrieval (BM25 + semantic_text via RRF retriever).

Targets ES 8.14+ retriever DSL. Search is performed against one or more indices
selected by the index_route node (elasticsearch_docs / kafka_docs / confluence_docs).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.config import Settings, get_settings


@lru_cache
def get_es_client() -> AsyncElasticsearch:
    s = get_settings()
    kwargs: dict[str, Any] = dict(
        hosts=s.es_host_list,
        basic_auth=(s.es_username, s.es_password),
        verify_certs=s.es_verify_certs,
        ssl_show_warn=False,
        request_timeout=30,
    )
    if s.es_ca_certs:
        kwargs["ca_certs"] = s.es_ca_certs
    return AsyncElasticsearch(**kwargs)


def build_rrf_query(
    bm25_query_text: str,
    *,
    semantic_query_text: str | None = None,
    settings: Settings,
    metadata_filters: dict[str, Any] | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    """Build a single search request with RRF over BM25 + semantic.

    BM25 sees a keywords-only string (best for lexical match); semantic sees
    the full natural-language form (best for the embedding model). When the
    caller passes only one text, both retrievers reuse it.

    The semantic_text field is configured at index time with text-embedding-3-small,
    so the `semantic` query handles query embedding inside ES — we never embed
    on the client side.
    """
    s = settings
    rank_window = s.retrieval_rank_window
    rank_constant = s.retrieval_rank_constant
    final_size = size or s.retrieval_top_k
    sem_text = semantic_query_text if semantic_query_text else bm25_query_text

    # BM25: search content + title (^2 boost) + ancestors.title (^1.5 boost,
    # confluence page hierarchy). `lenient: true` so non-confluence indices
    # that don't map ancestors.title don't fail the cross-index search.
    bm25_fields: list[str] = []
    if s.es_field_title:
        bm25_fields.append(f"{s.es_field_title}^2")
    if s.es_field_ancestors_title:
        bm25_fields.append(f"{s.es_field_ancestors_title}^1.5")
    bm25_fields.append(s.es_field_content)
    if len(bm25_fields) == 1:
        bm25_query: dict[str, Any] = {
            "match": {bm25_fields[0]: {"query": bm25_query_text}}
        }
    else:
        bm25_query = {
            "multi_match": {
                "query": bm25_query_text,
                "fields": bm25_fields,
                "type": "best_fields",
                "lenient": True,
            }
        }

    bm25_retriever = {"standard": {"query": bm25_query}}
    semantic_retriever = {
        "standard": {
            "query": {
                "semantic": {
                    "field": s.es_field_semantic,
                    "query": sem_text,
                }
            }
        }
    }

    filter_clauses = _build_filter_clauses(metadata_filters or {})
    if filter_clauses:
        for retriever in (bm25_retriever, semantic_retriever):
            inner = retriever["standard"]["query"]
            retriever["standard"]["query"] = {
                "bool": {"must": [inner], "filter": filter_clauses}
            }

    return {
        "retriever": {
            "rrf": {
                "retrievers": [bm25_retriever, semantic_retriever],
                "rank_window_size": rank_window,
                "rank_constant": rank_constant,
            }
        },
        "size": final_size,
        "_source": [
            f for f in (
                s.es_field_title,
                s.es_field_content,
                s.es_field_url,
                "source",
                "category",
                "updated_at",
            )
            if f
        ],
    }


def _build_filter_clauses(filters: dict[str, Any]) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    for field in ("source", "category"):
        value = filters.get(field)
        if not value:
            continue
        if isinstance(value, list):
            clauses.append({"terms": {field: value}})
        else:
            clauses.append({"term": {field: value}})

    date_range = filters.get("date_range") or {}
    if date_range:
        rng = {k: v for k, v in date_range.items() if v}
        if rng:
            clauses.append({"range": {"updated_at": rng}})
    return clauses


async def hybrid_search(
    *,
    bm25_query_text: str,
    semantic_query_text: str | None = None,
    indices: list[str],
    metadata_filters: dict[str, Any] | None = None,
    size: int | None = None,
    client: AsyncElasticsearch | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid RRF search across the target indices and return hit docs."""
    settings = get_settings()
    body = build_rrf_query(
        bm25_query_text,
        semantic_query_text=semantic_query_text,
        settings=settings,
        metadata_filters=metadata_filters,
        size=size,
    )
    es = client or get_es_client()
    response = await es.search(index=",".join(indices), body=body)
    hits = response.get("hits", {}).get("hits", [])
    return [_hit_to_document(hit, settings) for hit in hits]


async def count_documents(
    *,
    indices: list[str],
    metadata_filters: dict[str, Any] | None = None,
    client: AsyncElasticsearch | None = None,
) -> dict[str, int]:
    """Returns {index_name: doc_count} for each requested index."""
    es = client or get_es_client()
    out: dict[str, int] = {}
    body: dict[str, Any] = {}
    fc = _build_filter_clauses(metadata_filters or {})
    if fc:
        body["query"] = {"bool": {"filter": fc}}
    for idx in indices:
        try:
            r = await es.count(index=idx, body=body if body else None)
            out[idx] = int(r.get("count", 0))
        except Exception:
            # Index missing / not authorised — treat as 0 rather than crashing
            out[idx] = 0
    return out


async def list_titles(
    *,
    indices: list[str],
    metadata_filters: dict[str, Any] | None = None,
    size: int = 30,
    client: AsyncElasticsearch | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """Returns {index_name: [(title, chunk_count), ...]} via terms aggregation
    on the configured title field. Empty dict if title field is not configured.
    """
    s = get_settings()
    if not s.es_field_title:
        return {}
    es = client or get_es_client()
    out: dict[str, list[tuple[str, int]]] = {}
    fc = _build_filter_clauses(metadata_filters or {})
    # terms agg requires a keyword/numeric field — use the .keyword subfield
    # of the analyzed title (text field cannot be aggregated without fielddata).
    agg_field = f"{s.es_field_title}.keyword"
    for idx in indices:
        body: dict[str, Any] = {
            "size": 0,
            "aggs": {
                "by_title": {
                    "terms": {"field": agg_field, "size": size}
                }
            },
        }
        if fc:
            body["query"] = {"bool": {"filter": fc}}
        try:
            r = await es.search(index=idx, body=body)
        except Exception:
            out[idx] = []
            continue
        buckets = (
            r.get("aggregations", {}).get("by_title", {}).get("buckets", [])
        )
        out[idx] = [(b.get("key", ""), int(b.get("doc_count", 0))) for b in buckets]
    return out


def _hit_to_document(hit: dict[str, Any], settings: Settings) -> dict[str, Any]:
    src = hit.get("_source", {})
    return {
        "id": hit.get("_id", ""),
        "index": hit.get("_index", ""),
        "score": hit.get("_score", 0.0),
        "title": src.get(settings.es_field_title, ""),
        "content": src.get(settings.es_field_content, ""),
        "url": src.get(settings.es_field_url, ""),
        "source": src.get("source", ""),
        "category": src.get("category", ""),
        "updated_at": src.get("updated_at", ""),
    }
