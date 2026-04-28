from __future__ import annotations

from app import prompts
from app.graph.nodes import PROGRESS_KEY
from app.graph.nodes._helpers import llm_json
from app.graph.state import RAGState
from app.services.llm_factory import get_judge_llm


async def query_decompose(state: RAGState) -> dict:
    query = state.get("resolved_query") or state["current_query"]
    if state.get("intent") == "chitchat":
        return {
            "sub_queries": [],
            PROGRESS_KEY: "💬 잡담으로 판단되어 검색을 생략합니다.",
        }

    prompt = prompts.QUERY_DECOMPOSE.format(query=query)
    data = await llm_json(get_judge_llm(), prompt)
    raw = data.get("sub_queries") or [query]
    subs = [s.strip() for s in raw if isinstance(s, str) and s.strip()][:3]
    if not subs:
        subs = [query]

    if len(subs) == 1:
        tree = f"   └─ {subs[0]}"
    else:
        tree = "\n".join(
            f"   {'└─' if i == len(subs) - 1 else '├─'} {s}" for i, s in enumerate(subs)
        )
    return {
        "sub_queries": subs,
        PROGRESS_KEY: f"🧩 질의 분해 중... ({len(subs)}개 서브쿼리)\n{tree}",
    }
