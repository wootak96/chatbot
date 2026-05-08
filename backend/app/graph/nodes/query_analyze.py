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
# this regex overrides such mistakes. Three groups: public-tech (ES/Kafka),
# internal-wiki (Confluence ops/wiki vocabulary), and product names.
_DOMAIN_PATTERN = re.compile(
    r"(?i)("
    # Public-tech: Elasticsearch / Kafka
    r"elasticsearch|엘라스틱서치|엘라스틱|\bes\b|kafka|카프카|"
    r"\brrf\b|\bbm25\b|semantic|시맨틱|\bknn\b|벡터검색|"
    r"consumer|producer|topic|partition|broker|replica|"
    r"\bmapping\b|\bindex\b|인덱스|shard|샤드|"
    r"analyzer|tokenizer|embedding|임베딩|dense_vector|sparse_vector|"
    # Internal-wiki: Confluence
    r"confluence|컨플루언스|\bwiki\b|위키|"
    r"회의록|미팅록|미팅\s*노트|"
    r"운영\s*가이드|운영\s*매뉴얼|운영\s*절차|"
    r"장애\s*대응|장애\s*보고|"
    r"인수\s*인계|"
    r"사내\s*(표준|정책|가이드|매뉴얼|절차)|"
    r"팀\s*위키"
    r")"
)


# Internal-only proper nouns supplied by the user (HMG company terms,
# product names, location names, internal-namespace path prefixes). Anything
# matching this MUST also be treated as a domain question and MUST add
# confluence_docs to the routing — even if the LLM doesn't recognize the
# term. Kept as a separate pattern so `index_route` can reuse it.
_INTERNAL_PATTERN = re.compile(
    r"(?i)("
    # HMG cloud platforms / internal products
    r"hmgcloud|hcloud|hmgsearch|vaatz|evplatform|kafkaadm|"
    # Internal acronyms — word boundaries reduce false positives on
    # substrings inside unrelated English text
    r"\bvdsp\b|\bdsp\b|\bota\b|\baip\b|\bpam\b|\bhae\b|"
    r"\bhkmc\b|\bhmg\b|\bhchat\b|"
    # Internal locations (Korean, no boundary needed for CJK)
    r"상암|가산|광주|"
    # Team / industry context
    r"클라우드솔루션|완성차|"
    # ES path namespaces (leading slash distinguishes from generic words)
    r"/es_engine|/es_log|/es_data"
    r")"
)

# Meta-collection safety net: questions about the chatbot's document
# collection itself (count / list / total) should always be `question`,
# even when no specific domain token like "kafka" or "ES" is mentioned.
# search_intent will then classify them as count/list and route_query
# falls back to all indices on ambiguity (INDEX_ROUTE prompt rule).
_META_COLLECTION_PATTERN = re.compile(
    r"(전체\s*문서|사내\s*문서|사내\s*자료|"
    r"문서\s*(목록|리스트|개수|갯수|몇|얼마)|"
    r"몇\s*(개|건)|총\s*(몇|\d+)|"
    r"어떤\s*문서|문서들?\s*(뭐|뭔|어떤))"
)

# Debug-mode safety net: meta-questions about WHY a previous bot answer was
# produced. These should always be `debugging` regardless of LLM output, even
# when domain words appear (e.g., "왜 Kafka 답변이 이렇게 나왔어?" — the user
# is asking about a past answer, not requesting new Kafka info).
#
# Detection is a co-occurrence check: a reference to a previous bot output
# ("답변", "답", "응답", "결과") plus a meta-question word ("왜", "어떻게",
# "어디서", "근거"). Word-order independent so "왜 Kafka 답변이..." matches
# the same as "답변이 왜 이래?". Strong standalone phrases ("디버깅", "왜
# 이렇게 판단", etc.) trigger directly.

_DEBUG_ANSWER_REF = re.compile(r"(답변|응답|결과|답이|답은|답을|이\s*답)")
_DEBUG_META_QUESTION = re.compile(
    r"(왜|어떻게|어디서|어디에서|어디|근거|판단)"
)
_DEBUG_STRONG = re.compile(
    r"(디버깅|"
    r"왜\s*(이렇게|이런|그렇게|그런|충분|불충분|이렇다)|"
    r"어떻게\s*(판단|결정|찾았)|"
    r"근거가?\s*(뭐|뭔|뭐야|있어|어디)|"
    r"어디서?\s*(나왔|찾았|가져왔|뽑))"
)


def _has_domain_term(text: str) -> bool:
    if not text:
        return False
    return bool(_DOMAIN_PATTERN.search(text) or _INTERNAL_PATTERN.search(text))


def _has_internal_term(text: str) -> bool:
    """True when an HMG-internal proper noun appears — used by the index
    router to force-include confluence_docs regardless of the LLM verdict."""
    return bool(_INTERNAL_PATTERN.search(text or ""))


def _has_meta_collection_term(text: str) -> bool:
    return bool(_META_COLLECTION_PATTERN.search(text or ""))


def _is_debug_query(text: str) -> bool:
    if not text:
        return False
    if _DEBUG_STRONG.search(text):
        return True
    return bool(
        _DEBUG_ANSWER_REF.search(text) and _DEBUG_META_QUESTION.search(text)
    )


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
    if intent not in ("question", "chitchat", "general", "debugging", "instruction"):
        intent = "question"

    # Override 1: debug pattern wins over everything else. Meta-questions
    # about a prior bot answer ("왜 답변이 이렇게 나왔어?", even when they
    # contain domain words) must reach `debug_explain`, not the search path.
    if _is_debug_query(query):
        intent = "debugging"
    # Override 2: domain/meta terms force `question` (existing safety net).
    # Instruction is preserved — directives like "Kafka 답변할 때는 영어 용어
    # 그대로 써줘" contain domain words but are still preferences, not info
    # requests. We trust the LLM's `instruction` label here.
    elif intent in ("chitchat", "general") and (
        _has_domain_term(query) or _has_meta_collection_term(query)
    ):
        intent = "question"

    return {
        "intent": intent,
        PROGRESS_KEY: f"🔍 질문 분석 중... (intent={intent})",
    }
