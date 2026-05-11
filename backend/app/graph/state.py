from __future__ import annotations

from typing import Any, Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class Document(TypedDict, total=False):
    id: str
    index: str
    score: float
    title: str
    content: str
    url: str
    source: str
    category: str
    updated_at: str


class SearchPlan(TypedDict, total=False):
    """Per-(sub_query, index) retrieval plan with index-aware rewrites.

    Different indices may use different languages — confluence_docs is a
    Korean corpus while elasticsearch_docs / kafka_docs are English. So the
    rewrite step fans out one plan per (sub_query, routed-index) pair, and
    BM25 / semantic strings follow each index's language policy.
    """

    sub_query_idx: int  # 0-based index into sub_queries
    sub_query: str  # original sub-query text (pre-rewrite)
    index: str  # target index name (e.g., "confluence_docs")
    bm25: str  # BM25 query (Korean for confluence, English for ES/Kafka)
    semantic: str  # semantic query (same per-index language policy)


Intent = Literal["question", "chitchat", "general", "debugging", "instruction"]
SearchIntent = Literal["lookup", "count", "list"]


class RAGState(TypedDict, total=False):
    # Input
    messages: list[Message]
    current_query: str
    # Logged-in user — required for log persistence and the `debugging` intent
    # which fetches recent turns from the shared chat_logs index. Empty when
    # the request arrived without authentication (chat still works, no log).
    user_id: str
    # Per-conversation session id. Scopes log persistence and debug history
    # so a fresh chat thread doesn't pull turns from a prior conversation.
    session_id: str

    # Query understanding
    # resolved_query is produced by query_reform on the search branch using
    # conversation history; query_analyze itself only emits the intent label.
    # On chitchat/general paths query_reform is skipped, so downstream nodes
    # fall back to current_query when resolved_query is absent.
    resolved_query: str
    intent: Intent
    # search_intent partitions ES query shape: lookup (RRF top-k), count
    # (size=0 + track_total_hits), list (terms agg on title). Set only on the
    # search branch (question); irrelevant for chitchat/general.
    search_intent: SearchIntent
    sub_queries: list[str]
    # Per-sub-query routing: target_indices_per_query[i] is the list of ES
    # indices to search for sub_queries[i]. Produced by index_route, which
    # now runs BEFORE query_rewrite so rewrites can be index-aware.
    target_indices_per_query: list[list[str]]
    # Flattened (sub_query, index) plans with per-index rewrite. One entry
    # per (sub_query_idx, target_index) pair.
    search_plans: list[SearchPlan]
    metadata_filters: dict[str, Any]

    # Retrieval
    candidates: list[Document]
    # Diagnostic-only: per-plan single-retriever results (title/url) so
    # chat_logs can show where each RRF-fused doc came from. Each entry is
    # a dict with `index`, `sub_query`, `bm25` query text, `semantic` query
    # text, and a `hits` list of {title, url, score} top documents.
    bm25_only_results: list[dict[str, Any]]
    semantic_only_results: list[dict[str, Any]]

    # Control
    retry_count: int
    sufficient: bool
    sufficiency_reason: str

    # Output
    final_answer: str
    sources: list[dict[str, str]]


def initial_state(
    messages: list[Message], user_id: str = "", session_id: str = ""
) -> RAGState:
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    current = last_user["content"] if last_user else ""
    return RAGState(
        messages=messages,
        current_query=current,
        user_id=user_id,
        session_id=session_id,
        retry_count=0,
        candidates=[],
        bm25_only_results=[],
        semantic_only_results=[],
        sub_queries=[],
        target_indices_per_query=[],
        search_plans=[],
        metadata_filters={},
        sources=[],
        final_answer="",
        sufficient=False,
        sufficiency_reason="",
    )
