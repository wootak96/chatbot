from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_settings


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON parse: strips ```json fences and finds the first {...} block."""
    if not text:
        return {}
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # find first balanced {...}
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}


async def llm_json(llm, prompt: str) -> dict[str, Any]:
    """Invoke the LLM and parse its response as JSON."""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, "content") else str(response)
    return parse_json(content)


def truncate_history(messages: list[dict], turns: int | None = None) -> list[dict]:
    """Keep only the last `turns` user/assistant pairs."""
    if turns is None:
        turns = get_settings().max_history_turns
    keep = turns * 2
    relevant = [m for m in messages if m["role"] in ("user", "assistant")]
    return relevant[-keep:] if keep > 0 else relevant


def render_history(messages: list[dict]) -> str:
    if not messages:
        return "(없음)"
    lines = []
    for m in messages:
        role = "사용자" if m["role"] == "user" else "어시스턴트"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def render_docs_brief(docs: list[dict]) -> str:
    if not docs:
        return "(검색된 문서 없음)"
    lines = []
    for i, d in enumerate(docs, 1):
        title = d.get("title", "")
        url = d.get("url", "")
        excerpt = (d.get("content", "") or "")[:200].replace("\n", " ")
        lines.append(f"[{i}] {title} | {url}\n    {excerpt}")
    return "\n".join(lines)


def render_docs_full(docs: list[dict], char_limit: int = 1500) -> str:
    if not docs:
        return "(검색된 문서 없음)"
    lines = []
    for i, d in enumerate(docs, 1):
        title = d.get("title", "")
        url = d.get("url", "")
        body = (d.get("content", "") or "")[:char_limit]
        lines.append(f"[{i}] {title}\nURL: {url}\n{body}")
    return "\n\n---\n\n".join(lines)
