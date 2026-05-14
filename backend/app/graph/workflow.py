"""LangGraph workflow assembly.

Flow:
  START
   -> query_analyze (intent classifier: question / chitchat / general / debugging / instruction)
        chitchat?    -> generate (greeting reply) -> END
        general?     -> general_chat -> END
        debugging?   -> debug_explain (replay chat_logs trace) -> END
        instruction? -> instruction_save (update user's chat_md doc) -> END
        question?    -> query_reform (history-aware self-contained rewrite)
   -> search_intent (lookup / count / list)
        count?     -> es_count -> END
        list?      -> es_list  -> END
        lookup?    -> query_decompose
   -> query_decompose -> index_route -> query_rewrite -> metadata_extract
   -> hybrid_retrieve
   -> self_check
        sufficient            -> generate (RAG-grounded) -> END
        retry budget left     -> query_variate -> hybrid_retrieve (cycle,
                                 each pass widens top_k: 10 -> 20 -> 30)
        budget exhausted      -> generate (soft-escape redirect) -> END

Routing now runs BEFORE rewrite so rewrites can be index-aware: the
confluence_docs corpus is Korean while elasticsearch_docs / kafka_docs are
English, and a single sub-query routed to multiple indices needs different
BM25/semantic strings per index. `query_rewrite` fans out one plan per
(sub_query, routed-index) pair into `search_plans`, which `hybrid_retrieve`
then executes in parallel.

History-aware query reformulation is isolated in `query_reform` (single
responsibility) so downstream nodes only see a self-contained query and never
need access to conversation history.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.debug_explain import debug_explain
from app.graph.nodes.es_count import es_count
from app.graph.nodes.es_list import es_list
from app.graph.nodes.general_chat import general_chat
from app.graph.nodes.generate import generate
from app.graph.nodes.hybrid_retrieve import hybrid_retrieve
from app.graph.nodes.index_route import index_route
from app.graph.nodes.instruction_save import instruction_save
from app.graph.nodes.metadata_extract import metadata_extract
from app.graph.nodes.query_analyze import query_analyze
from app.graph.nodes.query_decompose import query_decompose
from app.graph.nodes.query_reform import query_reform
from app.graph.nodes.query_rewrite import query_rewrite
from app.graph.nodes.query_variate import query_variate
from app.graph.nodes.re_search_setup import re_search_setup
from app.graph.nodes.search_intent import search_intent
from app.graph.nodes.self_check import self_check, should_retry
from app.graph.state import RAGState


def _branch_from_analyze(state: RAGState) -> str:
    intent = state.get("intent")
    if intent == "chitchat":
        return "chitchat"
    if intent == "general":
        return "general"
    if intent == "debugging":
        return "debugging"
    if intent == "instruction":
        return "instruction"
    if intent == "re_search":
        return "re_search"
    return "search"


def _branch_from_re_search(state: RAGState) -> str:
    """re_search_setup either populated sub_queries + target_indices_per_query
    (→ continue to query_rewrite for fresh per-index rewrites) or emitted a
    refusal message (→ end immediately)."""
    if state.get("final_answer"):
        return "end"
    return "rewrite"


def _branch_from_search_intent(state: RAGState) -> str:
    si = state.get("search_intent") or "lookup"
    if si == "count":
        return "count"
    if si == "list":
        return "list"
    return "lookup"


def build_workflow():
    builder = StateGraph(RAGState)

    builder.add_node("query_analyze", query_analyze)
    builder.add_node("query_reform", query_reform)
    builder.add_node("search_intent", search_intent)
    builder.add_node("query_decompose", query_decompose)
    builder.add_node("index_route", index_route)
    builder.add_node("query_rewrite", query_rewrite)
    builder.add_node("metadata_extract", metadata_extract)
    builder.add_node("hybrid_retrieve", hybrid_retrieve)
    builder.add_node("self_check", self_check)
    builder.add_node("query_variate", query_variate)
    builder.add_node("es_count", es_count)
    builder.add_node("es_list", es_list)
    builder.add_node("generate", generate)
    builder.add_node("general_chat", general_chat)
    builder.add_node("debug_explain", debug_explain)
    builder.add_node("instruction_save", instruction_save)
    builder.add_node("re_search_setup", re_search_setup)

    builder.add_edge(START, "query_analyze")
    builder.add_conditional_edges(
        "query_analyze",
        _branch_from_analyze,
        {
            "chitchat": "generate",
            "general": "general_chat",
            "debugging": "debug_explain",
            "instruction": "instruction_save",
            "re_search": "re_search_setup",
            "search": "query_reform",
        },
    )
    builder.add_conditional_edges(
        "re_search_setup",
        _branch_from_re_search,
        {
            # Re-run query_rewrite so BM25/semantic strings are regenerated
            # for the new forced indices (language differs per index).
            "rewrite": "query_rewrite",
            "end": END,
        },
    )
    builder.add_edge("query_reform", "search_intent")
    builder.add_conditional_edges(
        "search_intent",
        _branch_from_search_intent,
        {
            "lookup": "query_decompose",
            "count": "es_count",
            "list": "es_list",
        },
    )
    builder.add_edge("es_count", END)
    builder.add_edge("es_list", END)
    builder.add_edge("query_decompose", "index_route")
    builder.add_edge("index_route", "query_rewrite")
    builder.add_edge("query_rewrite", "metadata_extract")
    builder.add_edge("metadata_extract", "hybrid_retrieve")
    builder.add_edge("hybrid_retrieve", "self_check")
    builder.add_conditional_edges(
        "self_check",
        should_retry,
        {
            "retry": "query_variate",
            "generate": "generate",
        },
    )
    # Variation node loops back into retrieval with the new queries.
    builder.add_edge("query_variate", "hybrid_retrieve")
    builder.add_edge("generate", END)
    builder.add_edge("general_chat", END)
    builder.add_edge("debug_explain", END)
    builder.add_edge("instruction_save", END)

    return builder.compile()


_compiled = None


def get_workflow():
    global _compiled
    if _compiled is None:
        _compiled = build_workflow()
    return _compiled
