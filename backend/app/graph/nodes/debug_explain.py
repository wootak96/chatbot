"""Debug-mode answer node — explains why a prior bot answer came out the way
it did, by replaying the stored RAG trace.

Reads up to 3 most recent turns from the user's `{user_id}_logs` index, hands
them to the generator LLM, and lets it pick the relevant turn (the user might
be asking about the most recent one, or about an earlier one referenced by
topic or position). The output is streamed via on_chat_model_stream just like
the regular generate node — chat.py adds 'debug_explain' to its
node-name allowlist so tokens reach the SSE client.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.state import RAGState
from app.services.llm_factory import get_generator_llm
from app.services.log_store import fetch_recent_turns


_RECENT_N = 3


def _render_turns(turns: list[dict]) -> str:
    """Render recent turns into a numbered text block for the LLM prompt.

    Turn 1 is the most recent. Each block contains question / final answer
    plus the structured trace fields most useful for explaining outcomes
    (intent, search_intent, sub_queries, search_plans, sufficiency, sources)
    and the raw progress_log so the LLM can quote specific lines.
    """
    if not turns:
        return "(저장된 최근 턴이 없습니다.)"

    blocks: list[str] = []
    for i, t in enumerate(turns, 1):
        question = (t.get("question") or "").strip()
        final = (t.get("final_answer") or "").strip()
        progress = (t.get("progress_log") or "").strip()
        intent = t.get("intent") or ""
        search_intent = t.get("search_intent") or ""
        sub_queries = t.get("sub_queries") or []
        target_indices = t.get("target_indices") or []
        plans = t.get("search_plans") or []
        sufficient = t.get("sufficient")
        reason = t.get("sufficiency_reason") or ""
        sources = t.get("sources") or []
        retry_count = t.get("retry_count")

        lines = [
            f"━━━ Turn {i} ━━━",
            f"질문: {question}",
            f"intent: {intent} | search_intent: {search_intent}"
            + (f" | retry_count: {retry_count}" if retry_count else ""),
        ]
        if sub_queries:
            lines.append(f"sub_queries: {sub_queries}")
        if target_indices:
            lines.append(f"target_indices: {target_indices}")
        if plans:
            lines.append("search_plans:")
            for p in plans:
                lines.append(
                    f"  - [{p.get('index', '')}] sub_query={p.get('sub_query', '')!r} "
                    f"bm25={p.get('bm25', '')!r} semantic={p.get('semantic', '')!r}"
                )
        if sufficient is not None:
            lines.append(f"sufficient: {sufficient} | reason: {reason}")
        if sources:
            lines.append("sources:")
            for s in sources:
                title = s.get("title", "")
                url = s.get("url", "")
                lines.append(f"  - {title} ({url})")
        if progress:
            lines.append("--- 진행 trace (UI에 표시됐던 내용) ---")
            lines.append(progress)
        lines.append("--- 최종 답변 ---")
        lines.append(final or "(빈 답변)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def debug_explain(state: RAGState) -> dict:
    user_id = state.get("user_id") or ""
    query = state.get("resolved_query") or state.get("current_query") or ""

    if not user_id:
        return {
            "final_answer": (
                "디버깅 모드는 로그인된 사용자에 한해 동작합니다 "
                "(현재 요청에 user_id가 포함되지 않아 로그를 조회할 수 없습니다)."
            ),
            "sources": [],
            PROGRESS_KEY: "🐛 디버깅 모드: user_id 미전송 — 로그 조회 불가",
        }

    turns = await fetch_recent_turns(user_id, n=_RECENT_N)

    if not turns:
        return {
            "final_answer": "최근 대화 기록이 없어 디버깅할 수 없습니다.",
            "sources": [],
            PROGRESS_KEY: "🐛 디버깅 모드: 최근 로그 없음",
        }

    context = _render_turns(turns)
    prompt = prompts.DEBUG_EXPLAIN.format(query=query, turns=context)

    response = await get_generator_llm().ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "sources": [],
        PROGRESS_KEY: f"🐛 디버깅 모드: 최근 {len(turns)}턴 분석 완료",
    }
