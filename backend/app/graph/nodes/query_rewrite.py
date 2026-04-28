from __future__ import annotations

import asyncio

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def _rewrite_one(query: str) -> tuple[str, str]:
    """Returns (bm25_keywords, semantic_query). Both English."""
    data = await llm_json(get_judge_llm(), prompts.QUERY_REWRITE.format(query=query))
    keywords = (data.get("keywords") or "").strip()
    semantic = (data.get("semantic") or "").strip()
    # Defensive fallbacks: if the model produced only one field, reuse it for
    # the other so retrieval still works.
    if not keywords and semantic:
        keywords = semantic
    if not semantic and keywords:
        semantic = keywords
    if not keywords and not semantic:
        keywords = semantic = query
    return keywords, semantic


async def query_rewrite(state: RAGState) -> dict:
    subs = state.get("sub_queries") or []
    if not subs:
        return {
            "rewritten_queries": [],
            "semantic_queries": [],
            PROGRESS_KEY: "✏️ 검색 쿼리 최적화 중... (스킵)",
        }
    pairs = list(await asyncio.gather(*[_rewrite_one(s) for s in subs]))
    keywords_list = [p[0] for p in pairs]
    semantic_list = [p[1] for p in pairs]

    lines = []
    for i, (before, kw, sem) in enumerate(zip(subs, keywords_list, semantic_list)):
        branch = "└─" if i == len(subs) - 1 else "├─"
        lines.append(f"   {branch} {before}")
        lines.append(f"      • BM25:     {kw}")
        lines.append(f"      • semantic: {sem}")
    tree = "\n".join(lines)

    return {
        "rewritten_queries": keywords_list,
        "semantic_queries": semantic_list,
        PROGRESS_KEY: f"✏️ 검색 쿼리 최적화 중...\n{tree}",
    }
