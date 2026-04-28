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


Intent = Literal["question", "chitchat", "general"]
SearchIntent = Literal["lookup", "count", "list"]


class RAGState(TypedDict, total=False):
    # Input
    messages: list[Message]
    current_query: str

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
    # rewritten_queries[i] = English BM25 keywords for sub_queries[i].
    # semantic_queries[i]  = full English natural-language form for the
    # semantic_text retriever. Same length as rewritten_queries.
    rewritten_queries: list[str]
    semantic_queries: list[str]
    metadata_filters: dict[str, Any]
    # Per-sub-query routing: target_indices_per_query[i] is the list of ES
    # indices to search for sub_queries[i]. Length matches rewritten_queries.
    target_indices_per_query: list[list[str]]

    # Retrieval
    candidates: list[Document]

    # Control
    retry_count: int
    sufficient: bool
    sufficiency_reason: str

    # Output
    final_answer: str
    sources: list[dict[str, str]]


def initial_state(messages: list[Message]) -> RAGState:
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    current = last_user["content"] if last_user else ""
    return RAGState(
        messages=messages,
        current_query=current,
        retry_count=0,
        candidates=[],
        sub_queries=[],
        rewritten_queries=[],
        semantic_queries=[],
        metadata_filters={},
        target_indices_per_query=[],
        sources=[],
        final_answer="",
        sufficient=False,
        sufficiency_reason="",
    )
