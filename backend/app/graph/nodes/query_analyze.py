from __future__ import annotations

import re

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json, render_history, truncate_history
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


# Safety net: if any of these domain tokens appear in the current query, the
# intent MUST be "question" — never chitchat or general. The LLM occasionally
# misclassifies comparison questions like "ES와 Kafka 비교해줘" as general;
# this regex overrides such mistakes.
_DOMAIN_PATTERN = re.compile(
    r"(?i)(elasticsearch|엘라스틱서치|엘라스틱|\bes\b|kafka|카프카|"
    r"\brrf\b|\bbm25\b|semantic|시맨틱|\bknn\b|벡터검색|"
    r"consumer|producer|topic|partition|broker|replica|"
    r"\bmapping\b|\bindex\b|인덱스|shard|샤드|"
    r"analyzer|tokenizer|embedding|임베딩|dense_vector|sparse_vector)"
)


def _has_domain_term(text: str) -> bool:
    return bool(_DOMAIN_PATTERN.search(text or ""))


async def query_analyze(state: RAGState) -> dict:
    """Intent classifier (3-way: question / chitchat / general).

    History-aware reformulation of the query (resolving follow-up references
    like "그게", "어떻게") is delegated to the next node `query_reform` on the
    search branch. This node returns ONLY the intent label.
    """
    query = state["current_query"]
    history = truncate_history(state.get("messages", [])[:-1])
    prompt = prompts.QUERY_ANALYZE.format(
        history=render_history(history),
        query=query,
    )
    data = await llm_json(get_judge_llm(), prompt)

    intent = data.get("intent") or "question"
    if intent not in ("question", "chitchat", "general"):
        intent = "question"

    # Override LLM misclassification when domain keywords are obviously present.
    if intent in ("chitchat", "general") and _has_domain_term(query):
        intent = "question"

    return {
        "intent": intent,
        PROGRESS_KEY: f"🔍 질문 분석 중... (intent={intent})",
    }
