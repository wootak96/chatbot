from __future__ import annotations

from langchain_core.messages import HumanMessage

from app import prompts
from app.config import get_settings
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import render_docs_full
from app.graph.state import RAGState
from app.services.instruction_store import get_user_md
from app.services.llm_factory import get_generator_llm


def _format_user_md_block(md: str) -> str:
    """Wrap the user's stored instructions for injection into a prompt slot.
    Empty when the user has no saved md so the prompt has no leading
    blank line."""
    md = (md or "").strip()
    if not md:
        return ""
    return f"\n[사용자 지침]\n{md}\n"


async def generate(state: RAGState) -> dict:
    intent = state.get("intent") or "question"
    query = state.get("resolved_query") or state["current_query"]
    user_id = state.get("user_id") or ""
    user_md_block = _format_user_md_block(await get_user_md(user_id))

    if intent == "chitchat":
        prompt = prompts.CHITCHAT.format(query=query, user_md_block=user_md_block)
        sources: list[dict] = []
    else:
        settings = get_settings()
        candidates = state.get("candidates") or []
        # When generate can't ground an answer, what it should emit depends on
        # whether the retrieval loop will auto-retry after this pass:
        #   - non-final attempt → a single honest "not found yet" sentence
        #     (answer_check rejects it → wider re-search). Suggesting search
        #     keywords here would be odd — the bot would immediately re-search
        #     on its own right after asking the user to.
        #   - final attempt → the full redirect WITH alternative keywords,
        #     since there will be no more auto-retry.
        # Final = answer_check rejecting this pass can't trigger another retry:
        # the shared budget is (about to be) spent, or there are no plans to vary.
        budget = len(settings.retrieval_top_k_schedule)
        is_final_attempt = (
            state.get("retry_count", 0) >= budget - 1
            or not (state.get("search_plans") or [])
        )
        escape_directive = (
            prompts.GENERATE_ESCAPE_FINAL
            if is_final_attempt
            else prompts.GENERATE_ESCAPE_RETRYABLE
        )
        prompt = prompts.GENERATE.format(
            query=query,
            docs=render_docs_full(
                candidates,
                char_limit=settings.generate_doc_char_limit,
            ) if candidates else "(검색 결과 없음)",
            user_md_block=user_md_block,
            escape_directive=escape_directive,
        )
        # Sources only listed when we have docs — empty list in soft-escape
        # means the post-stream "📚 답변 인용 문서" block is skipped.
        sources = [
            {"url": d.get("url", ""), "title": d.get("title", "")}
            for d in candidates
        ] if state.get("sufficient", False) else []

    response = await get_generator_llm().ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "sources": sources,
        PROGRESS_KEY: "",
    }
