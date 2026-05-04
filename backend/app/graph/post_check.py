"""Post-answer groundedness verification.

Runs AFTER `generate` finishes streaming so it can inspect the full answer
text. Checks whether each `[N]`-cited claim is actually supported by the
cited source document. Different from `self_check`, which judges retrieval
sufficiency BEFORE the answer is written and cannot detect hallucinations.

Invoked from `chat.py` post-stream (not as a LangGraph node) because
LangGraph nodes can't easily react to the streamed-then-finalized output of
a streaming sibling node. The result is appended to the per-turn log and
emitted as a final progress chunk for the SSE consumer.
"""

from __future__ import annotations

import logging
from typing import Any

from app import prompts
from app.graph.nodes._helpers import llm_json
from app.services.llm_factory import get_judge_llm

logger = logging.getLogger(__name__)


_DOC_CHAR_LIMIT = 1200  # smaller than generate's 1500 — verification only


async def run_groundedness_check(
    *,
    answer: str,
    candidates: list[dict[str, Any]],
    cited_indices: set[int],
) -> dict[str, Any]:
    """Verify cited claims in the answer against the cited source docs.

    Returns an empty dict when there's nothing to verify (no candidates or
    no citations — e.g., chitchat / general / debugging / "정보 없음" answers)
    so the caller can decide whether to log/emit anything."""
    if not candidates or not cited_indices or not answer:
        return {}

    cited_blocks: list[str] = []
    for i, d in enumerate(candidates, 1):
        if i not in cited_indices:
            continue
        title = d.get("title", "")
        url = d.get("url", "")
        body = (d.get("content", "") or "")[:_DOC_CHAR_LIMIT]
        cited_blocks.append(f"[{i}] {title}\nURL: {url}\n{body}")
    if not cited_blocks:
        return {}

    prompt = prompts.GROUNDEDNESS_CHECK.format(
        answer=answer,
        cited_docs="\n\n---\n\n".join(cited_blocks),
    )
    try:
        data = await llm_json(get_judge_llm(), prompt)
    except Exception as e:
        logger.warning("Groundedness check LLM call failed: %s", e)
        return {}

    claims_raw = data.get("claims") or []
    claims: list[dict[str, Any]] = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        cites_raw = c.get("citations") or []
        cites = [int(x) for x in cites_raw if isinstance(x, (int, str)) and str(x).isdigit()]
        claims.append(
            {
                "claim": str(c.get("claim", "")),
                "citations": cites,
                "supported": bool(c.get("supported", False)),
                "reason": str(c.get("reason", "")),
            }
        )

    total = len(claims)
    supported = sum(1 for c in claims if c["supported"])
    return {
        "grounded": bool(data.get("grounded", False)) if total else True,
        "score": (supported / total) if total else 1.0,
        "supported_count": supported,
        "total_claims": total,
        "claims": claims,
    }


def format_groundedness_progress(result: dict[str, Any]) -> str:
    """Render the groundedness verdict as a one-block progress message for
    the SSE stream. Returns "" when the check was skipped or empty so the
    caller can suppress the chunk."""
    if not result:
        return ""
    total = result.get("total_claims") or 0
    supported = result.get("supported_count") or 0
    score = float(result.get("score") or 0.0)
    if total == 0:
        return ""
    if result.get("grounded"):
        head = f"✅ Groundedness 검증 — {supported}/{total} 인용 주장 근거 있음"
    elif score >= 0.5:
        head = f"⚠️ Groundedness 검증 — {supported}/{total} 인용 주장만 근거 있음 (score {score:.2f})"
    else:
        head = f"❌ Groundedness 검증 — 인용 주장의 근거 부족 ({supported}/{total}, score {score:.2f})"
    # Show unsupported claims so the user/operator can spot potential hallucinations.
    unsupported = [c for c in (result.get("claims") or []) if not c.get("supported")]
    if unsupported:
        lines = [head]
        for c in unsupported[:3]:  # cap noise
            claim = c.get("claim", "")[:80]
            reason = c.get("reason", "")[:80]
            lines.append(f"  ✗ {claim} — {reason}")
        return "\n".join(lines)
    return head
