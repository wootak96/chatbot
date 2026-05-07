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
        candidates = state.get("candidates") or []
        if not candidates or not state.get("sufficient", False):
            return {
                "final_answer": "해당 정보를 찾을 수 없습니다.",
                "sources": [],
                PROGRESS_KEY: "",
            }
        prompt = prompts.GENERATE.format(
            query=query,
            docs=render_docs_full(
                candidates,
                char_limit=get_settings().generate_doc_char_limit,
            ),
            user_md_block=user_md_block,
        )
        sources = [
            {"url": d.get("url", ""), "title": d.get("title", "")}
            for d in candidates
        ]

    response = await get_generator_llm().ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": content,
        "sources": sources,
        PROGRESS_KEY: "",
    }
