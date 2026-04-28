"""OpenAI-compatible /v1/chat/completions endpoint.

Drives the LangGraph workflow and emits:
  1. one SSE chunk per node (progress message)
  2. final answer streamed token-by-token from the generate node's LLM
  3. trailing CITES marker (hidden in UI, used to wrap inline [N] as links) + [DONE]
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Header
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.api import sse
from app.graph.nodes import PROGRESS_KEY
from app.graph.state import initial_state
from app.graph.workflow import get_workflow
from app.services.llm_factory import set_api_key

router = APIRouter()

MODEL_ID = "rag-chatbot"
SEPARATOR = "\n─────────────────────────────────────\n"

# When the GENERATE prompt is followed correctly the LLM never writes a
# **출처** section (the server appends one). When it disobeys, we truncate
# its stream at the marker to avoid duplicate source blocks. Detection is
# tolerant of the marker arriving split across multiple chunks.
SOURCE_MARKER = "**출처**"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None


async def _drive_workflow(request: ChatRequest) -> AsyncIterator[str]:
    completion_id = sse.make_completion_id()
    yield sse.role_chunk(model=request.model, completion_id=completion_id)

    state = initial_state([m.model_dump() for m in request.messages])
    workflow = get_workflow()

    final_state: dict[str, Any] = dict(state)
    answer_emitted = False
    pending_buf = ""           # holds last few chars to detect SOURCE_MARKER
    sources_truncated = False  # once True, drop the rest of the LLM stream

    async for event in workflow.astream_events(state, version="v2"):
        kind = event.get("event")
        node_name = event.get("metadata", {}).get("langgraph_node") or event.get("name")

        if kind == "on_chain_end" and event.get("name") in _NODE_NAMES:
            output = event.get("data", {}).get("output") or {}
            if isinstance(output, dict):
                msg = output.get(PROGRESS_KEY)
                if msg:
                    yield sse.text_chunk(
                        msg + "\n", model=request.model, completion_id=completion_id
                    )
                # accumulate state for source rendering
                for k, v in output.items():
                    if k == PROGRESS_KEY:
                        continue
                    final_state[k] = v

        elif kind == "on_chat_model_stream" and node_name in ("generate", "general_chat"):
            if sources_truncated:
                continue
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", "") if chunk is not None else ""
            if not content:
                continue
            if not answer_emitted:
                yield sse.text_chunk(
                    SEPARATOR, model=request.model, completion_id=completion_id
                )
                answer_emitted = True

            # Only the grounded `generate` node has a server-side **출처**
            # block to deduplicate against. general_chat has no sources, so
            # we let its tokens through unchanged.
            if node_name == "generate":
                pending_buf += content
                idx = pending_buf.find(SOURCE_MARKER)
                if idx >= 0:
                    safe = pending_buf[:idx].rstrip()
                    if safe:
                        yield sse.text_chunk(
                            safe, model=request.model, completion_id=completion_id
                        )
                    sources_truncated = True
                    pending_buf = ""
                elif len(pending_buf) > len(SOURCE_MARKER):
                    # Flush everything except the trailing window that could
                    # still complete into a marker on the next chunk.
                    flush = pending_buf[: -len(SOURCE_MARKER)]
                    pending_buf = pending_buf[-len(SOURCE_MARKER):]
                    if flush:
                        yield sse.text_chunk(
                            flush, model=request.model, completion_id=completion_id
                        )
            else:
                yield sse.text_chunk(
                    content, model=request.model, completion_id=completion_id
                )

    # After the LLM stream ends, flush any held-back tail (only if we didn't
    # already cut at SOURCE_MARKER).
    if not sources_truncated and pending_buf:
        yield sse.text_chunk(
            pending_buf, model=request.model, completion_id=completion_id
        )

    # If the generate node didn't stream (e.g., not-found case), fall back to
    # the final_answer captured in state.
    if not answer_emitted:
        fallback = final_state.get("final_answer", "")
        if fallback:
            yield sse.text_chunk(
                SEPARATOR, model=request.model, completion_id=completion_id
            )
            yield sse.text_chunk(
                fallback, model=request.model, completion_id=completion_id
            )

    # Emit a hidden CITES marker so the frontend can wrap inline [N] tokens in
    # the answer with clickable links to the corresponding source URL. We drop
    # the verbose **출처** block to save tokens — only N→url is needed since
    # the bracket text itself shows up inline already.
    sources = final_state.get("sources") or []
    cites = []
    for i, s in enumerate(sources, 1):
        url = (s.get("url") or "").strip()
        if not url:
            continue
        cites.append({"n": i, "url": url})
    if cites:
        marker = "\n<!--CITES:" + json.dumps(cites, ensure_ascii=False) + "-->"
        yield sse.text_chunk(
            marker, model=request.model, completion_id=completion_id
        )

    yield sse.stop_chunk(model=request.model, completion_id=completion_id)
    yield sse.done_marker()


_NODE_NAMES = {
    "query_analyze",
    "query_reform",
    "search_intent",
    "query_decompose",
    "query_rewrite",
    "metadata_extract",
    "index_route",
    "hybrid_retrieve",
    "self_check",
    "query_variate",
    "es_count",
    "es_list",
    "generate",
    "general_chat",
}


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    # chatbot-ui sends "dummy-key" as a placeholder when no real key is configured;
    # treat empty / clearly placeholder values as "use server default".
    if not token or token.lower() in {"dummy", "dummy-key", "placeholder"}:
        return None
    return token


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    if not request.stream:
        # For simplicity we only implement streaming. Non-streaming clients
        # can collect the SSE deltas themselves.
        return {"error": "non-streaming mode not supported", "code": "stream_required"}

    # Per-request LLM key: take the Bearer token from chatbot-ui and propagate
    # it via contextvar to all LLM clients constructed during this request.
    set_api_key(_extract_bearer(authorization))

    return StreamingResponse(
        _drive_workflow(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
