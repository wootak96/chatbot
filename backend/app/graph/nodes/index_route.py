"""Per-sub-query index routing.

For each sub-query (or its rewritten form when available), the LLM picks
which of the configured indices (`elasticsearch_docs`, `kafka_docs`) to
search. This lets a decomposed query like
   "ES와 Kafka의 차이"  →  ["ES throughput", "Kafka throughput"]
route to {ES} and {Kafka} respectively rather than searching both indices
for both sub-queries.
"""

from __future__ import annotations

import asyncio

from app import prompts
from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def _route_one(query: str, alias_map: dict[str, str]) -> list[str]:
    data = await llm_json(get_judge_llm(), prompts.INDEX_ROUTE.format(query=query))
    aliases = data.get("indices") or []
    if not isinstance(aliases, list):
        aliases = []
    valid = [a for a in aliases if a in alias_map]
    if not valid:
        valid = list(alias_map.keys())  # fallback: search both for recall
    return [alias_map[a] for a in valid]


async def index_route(state: RAGState) -> dict:
    settings = get_settings()
    alias_map = settings.index_alias_map

    if state.get("intent") == "chitchat":
        return {"target_indices_per_query": [], PROGRESS_KEY: ""}

    # Prefer rewritten queries when available (cleaner signal for the LLM);
    # fall back to sub_queries.
    queries = state.get("rewritten_queries") or state.get("sub_queries") or []
    if not queries:
        return {
            "target_indices_per_query": [],
            PROGRESS_KEY: "🧭 인덱스 라우팅: (서브쿼리 없음)",
        }

    routed = await asyncio.gather(*[_route_one(q, alias_map) for q in queries])
    target_indices_per_query = [list(r) for r in routed]

    # Brief summary for UI: aliases chosen for each sub-query.
    reverse = {v: k for k, v in alias_map.items()}
    summary = " | ".join(
        ",".join(reverse.get(i, i) for i in indices) for indices in target_indices_per_query
    )
    return {
        "target_indices_per_query": target_indices_per_query,
        PROGRESS_KEY: f"🧭 인덱스 라우팅: {summary}",
    }
