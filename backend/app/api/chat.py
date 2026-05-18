"""OpenAI-compatible /v1/chat/completions endpoint.

Drives the LangGraph workflow and emits:
  1. one SSE chunk per node (progress message)
  2. final answer streamed token-by-token from the generate node's LLM,
     with inline [N] citations renumbered in appearance order via
     `CitationRemapper` so the user sees [1], [2], [3]… instead of the
     sparse candidate positions the LLM emits (e.g., [10], [16], [22]).
  3. trailing CITES marker (hidden in UI, used to wrap inline [N] as links) + [DONE]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Header
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.api import sse
from app.graph.nodes import PROGRESS_KEY
from app.graph.post_check import (
    format_groundedness_progress,
    run_groundedness_check,
)
from app.graph.state import initial_state
from app.graph.workflow import get_workflow
from app.services.llm_factory import set_api_key
from app.services.log_store import save_turn

logger = logging.getLogger(__name__)

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
    # Logged-in user id from the frontend URL param. Empty / absent means
    # "no log persistence and debug-mode questions cannot fetch context".
    user_id: str | None = None
    # Per-conversation session id (UUID generated on the frontend). Resets on
    # "대화 초기화" / re-login so debug_explain only sees turns from the
    # current thread. Empty when frontend doesn't supply one — log persists
    # but debug fetch returns nothing.
    session_id: str | None = None


async def _drive_workflow(request: ChatRequest) -> AsyncIterator[str]:
    completion_id = sse.make_completion_id()
    yield sse.role_chunk(model=request.model, completion_id=completion_id)

    user_id = (request.user_id or "").strip()
    session_id = (request.session_id or "").strip()
    state = initial_state(
        [m.model_dump() for m in request.messages],
        user_id=user_id,
        session_id=session_id,
    )
    workflow = get_workflow()

    final_state: dict[str, Any] = dict(state)
    answer_emitted = False
    pending_buf = ""           # holds last few chars to detect SOURCE_MARKER
    sources_truncated = False  # once True, drop the rest of the LLM stream

    # Accumulators for log persistence — captured during the stream and
    # flushed to `{user_id}_logs` after the response is fully streamed out.
    progress_log_lines: list[str] = []
    # `streamed_answer_buf` holds the DISPLAYED text (with [N] renumbered in
    # citation-appearance order). `original_answer_buf` keeps the LLM's
    # pre-remap text — needed because groundedness / `used` candidate
    # tracking is indexed by the original 1-based candidate position, not
    # the displayed sequence.
    streamed_answer_buf: list[str] = []
    original_answer_buf: list[str] = []
    remapper = CitationRemapper()
    # Per-node token usage. Populated from `on_chat_model_end` events emitted
    # by every LLM call in the graph (judge nodes + streaming generator).
    token_usage_by_node: dict[str, dict[str, int]] = {}

    async for event in workflow.astream_events(state, version="v2"):
        kind = event.get("event")
        node_name = event.get("metadata", {}).get("langgraph_node") or event.get("name")

        if kind == "on_chain_end" and event.get("name") in _NODE_NAMES:
            output = event.get("data", {}).get("output") or {}
            if isinstance(output, dict):
                msg = output.get(PROGRESS_KEY)
                if msg:
                    progress_log_lines.append(msg)
                    yield sse.text_chunk(
                        msg + "\n", model=request.model, completion_id=completion_id
                    )
                # accumulate state for source rendering
                for k, v in output.items():
                    if k == PROGRESS_KEY:
                        continue
                    final_state[k] = v

        elif kind == "on_chat_model_end":
            usage = _extract_usage(event.get("data", {}).get("output"))
            if usage:
                bucket = token_usage_by_node.setdefault(
                    node_name or "unknown",
                    {"input": 0, "output": 0, "total": 0, "calls": 0},
                )
                bucket["input"] += usage["input"]
                bucket["output"] += usage["output"]
                bucket["total"] += usage["total"]
                bucket["calls"] += 1

        elif kind == "on_chat_model_stream" and node_name in (
            "generate",
            "general_chat",
            "debug_explain",
            "instruction_save",
        ):
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
            # block to deduplicate against. general_chat / debug_explain have
            # no sources and no [N] citations, so they bypass both the
            # SOURCE_MARKER buffer and the citation remapper.
            if node_name == "generate":
                pending_buf += content
                idx = pending_buf.find(SOURCE_MARKER)
                if idx >= 0:
                    safe = pending_buf[:idx].rstrip()
                    if safe:
                        original_answer_buf.append(safe)
                        # Stream ends here, so flush the remapper's
                        # bracket buffer in the same emission.
                        displayed = remapper.feed(safe) + remapper.flush()
                        if displayed:
                            streamed_answer_buf.append(displayed)
                            yield sse.text_chunk(
                                displayed,
                                model=request.model,
                                completion_id=completion_id,
                            )
                    sources_truncated = True
                    pending_buf = ""
                elif len(pending_buf) > len(SOURCE_MARKER):
                    # Flush everything except the trailing window that could
                    # still complete into a marker on the next chunk.
                    flush = pending_buf[: -len(SOURCE_MARKER)]
                    pending_buf = pending_buf[-len(SOURCE_MARKER):]
                    if flush:
                        original_answer_buf.append(flush)
                        displayed = remapper.feed(flush)
                        if displayed:
                            streamed_answer_buf.append(displayed)
                            yield sse.text_chunk(
                                displayed,
                                model=request.model,
                                completion_id=completion_id,
                            )
            else:
                streamed_answer_buf.append(content)
                original_answer_buf.append(content)
                yield sse.text_chunk(
                    content, model=request.model, completion_id=completion_id
                )

    # After the LLM stream ends, flush any held-back tail (only if we didn't
    # already cut at SOURCE_MARKER). The remapper also gets flushed so an
    # unclosed `[` at end-of-stream is emitted literally rather than dropped.
    if not sources_truncated and pending_buf:
        original_answer_buf.append(pending_buf)
        displayed = remapper.feed(pending_buf) + remapper.flush()
        if displayed:
            streamed_answer_buf.append(displayed)
            yield sse.text_chunk(
                displayed, model=request.model, completion_id=completion_id
            )
    else:
        trailing = remapper.flush()
        if trailing:
            streamed_answer_buf.append(trailing)
            yield sse.text_chunk(
                trailing, model=request.model, completion_id=completion_id
            )

    # If the generate node didn't stream (e.g., not-found case), fall back to
    # the final_answer captured in state. The fallback path bypasses the
    # remapper — those non-streamed answers don't carry [N] citations.
    if not answer_emitted:
        fallback = final_state.get("final_answer", "")
        if fallback:
            yield sse.text_chunk(
                SEPARATOR, model=request.model, completion_id=completion_id
            )
            yield sse.text_chunk(
                fallback, model=request.model, completion_id=completion_id
            )

    # Groundedness runs over the ORIGINAL (pre-remap) answer because
    # `cited_indices` here must index into `candidates` by its original
    # 1-based position — the remapping is a display-only concern.
    groundedness: dict[str, Any] = {}
    original_full = "".join(original_answer_buf) or final_state.get(
        "final_answer", ""
    )
    candidates_for_check = final_state.get("candidates") or []
    cited_for_check = _extract_cited_indices(original_full)
    if candidates_for_check and cited_for_check:
        groundedness = await run_groundedness_check(
            answer=original_full,
            candidates=candidates_for_check,
            cited_indices=cited_for_check,
        )
        verdict_msg = format_groundedness_progress(groundedness)
        if verdict_msg:
            progress_log_lines.append(verdict_msg)

    # Emit a hidden CITES marker so the frontend can wrap inline [N] tokens in
    # the answer with clickable links to the corresponding source URL. After
    # remapping, displayed `[new_N]` corresponds to the original candidate at
    # `mapping^-1(new_N)` — we resolve that here so the frontend mapping is
    # against the visible bracket numbers, not the original sparse ones.
    sources = final_state.get("sources") or []
    inverse_map = {new_n: orig_n for orig_n, new_n in remapper.mapping.items()}
    cites = []
    if inverse_map:
        for new_n in sorted(inverse_map):
            orig_n = inverse_map[new_n]
            if 1 <= orig_n <= len(sources):
                url = (sources[orig_n - 1].get("url") or "").strip()
                if url:
                    cites.append({"n": new_n, "url": url})
    else:
        # No remapping happened (chitchat / general / debugging): keep the
        # original 1-aligned mapping so callers without citations still get a
        # usable list when `sources` is populated through other code paths.
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

    # Persist this turn to `chat_logs` so the debug node can replay the
    # trace later. Skip when:
    #   - user_id is empty (no auth → no log)
    #   - intent is `debugging` (avoid recursive log noise)
    #   - intent is `instruction` (handled by its own chat_md store)
    # Save errors are best-effort logged inside save_turn — never raise out
    # to the SSE consumer (the response has already finished by this point
    # but we want chat to keep working even if ES is unhealthy).
    intent = final_state.get("intent") or ""
    if user_id and intent not in ("debugging", "instruction"):
        try:
            await save_turn(
                user_id,
                _build_log_doc(
                    request,
                    final_state,
                    progress_log_lines,
                    streamed_answer_buf,
                    original_answer_buf,
                    token_usage_by_node,
                    groundedness,
                ),
                session_id=session_id,
            )
        except Exception as e:  # belt-and-braces — save_turn already swallows
            logger.warning("Chat-turn log save failed: %s", e)


def _build_log_doc(
    request: ChatRequest,
    final_state: dict[str, Any],
    progress_log_lines: list[str],
    streamed_answer_buf: list[str],
    original_answer_buf: list[str],
    token_usage_by_node: dict[str, dict[str, int]] | None = None,
    groundedness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten the final RAGState into the `{user_id}_logs` document shape.

    `final_answer` is stored as the displayed (citation-remapped) text so the
    log matches what the user saw. `cited_indices` is extracted from the
    ORIGINAL pre-remap answer because it must index into `candidates` by
    their 1-based position — the `used` / `used_candidates` flags would be
    wrong if they pointed at displayed sequence numbers (1, 2, 3) instead of
    actual candidate positions (e.g., 10, 16, 22).
    """
    last_user = next(
        (m for m in reversed(request.messages) if m.role == "user"), None
    )
    user_question = last_user.content if last_user else ""
    final_answer = final_state.get("final_answer") or "".join(streamed_answer_buf)
    original_answer = "".join(original_answer_buf) or final_answer
    sub_queries = final_state.get("sub_queries") or []
    routing_per_query = final_state.get("target_indices_per_query") or []
    target_indices = sorted({idx for indices in routing_per_query for idx in indices})
    index_routing = [
        {"sub_query": sq, "indices": list(idxs)}
        for sq, idxs in zip(sub_queries, routing_per_query)
    ]
    plans = final_state.get("search_plans") or []
    candidates = final_state.get("candidates") or []
    cited_indices = _extract_cited_indices(original_answer)
    by_node = token_usage_by_node or {}
    total_in = sum(b.get("input", 0) for b in by_node.values())
    total_out = sum(b.get("output", 0) for b in by_node.values())
    total_calls = sum(b.get("calls", 0) for b in by_node.values())
    token_usage = {
        "total_input": total_in,
        "total_output": total_out,
        "total_tokens": total_in + total_out,
        "llm_calls": total_calls,
        "by_node": [
            {
                "node": node,
                "input": v.get("input", 0),
                "output": v.get("output", 0),
                "total": v.get("total", 0),
                "calls": v.get("calls", 0),
            }
            for node, v in sorted(by_node.items())
        ],
    }
    return {
        "question": user_question,
        "resolved_query": final_state.get("resolved_query") or "",
        "intent": final_state.get("intent") or "",
        "search_intent": final_state.get("search_intent") or "",
        "sub_queries": sub_queries,
        "target_indices": target_indices,
        "index_routing": index_routing,
        "metadata_filters": final_state.get("metadata_filters") or {},
        "search_plans": [
            {
                "sub_query": p.get("sub_query", ""),
                "index": p.get("index", ""),
                "bm25": p.get("bm25", ""),
                "semantic": p.get("semantic", ""),
            }
            for p in plans
        ],
        "candidates": [
            {
                "id": d.get("id", ""),
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "score": float(d.get("score", 0.0) or 0.0),
                "used": (i in cited_indices),
            }
            for i, d in enumerate(candidates, 1)
        ],
        # Pre-filtered view containing ONLY the candidates the answer
        # actually cited. Workaround for `candidates` being indexed as
        # `object` (parallel arrays flatten, so a `candidates.used == true`
        # filter can't return per-doc sub-objects). Querying this field
        # gives the cited docs directly without nested-query gymnastics.
        # `rank` is the doc's 1-based position in the full candidates list
        # so cited docs are still ordered/identifiable.
        "used_candidates": [
            {
                "rank": i,
                "title": d.get("title", ""),
                "url": d.get("url", ""),
            }
            for i, d in enumerate(candidates, 1)
            if i in cited_indices
        ],
        "bm25_only_results": final_state.get("bm25_only_results") or [],
        "semantic_only_results": final_state.get("semantic_only_results") or [],
        "forced_indices": final_state.get("forced_indices") or [],
        "sufficient": bool(final_state.get("sufficient", False)),
        "sufficiency_reason": final_state.get("sufficiency_reason") or "",
        "retry_count": int(final_state.get("retry_count", 0) or 0),
        "final_answer": final_answer,
        "sources": [
            {"url": s.get("url", ""), "title": s.get("title", "")}
            for s in (final_state.get("sources") or [])
        ],
        "progress_log": "\n".join(progress_log_lines),
        "token_usage": token_usage,
        "groundedness": groundedness or {},
    }


# Match bracket groups containing one or more digit-runs separated by commas:
# `[1]`, `[1, 2]`, `[1,2,3]`, `[ 1 , 2 ]`. Plain-text brackets like `[note]`
# are skipped because the inner alphabet is digits + comma + whitespace only.
_CITATION_RE = re.compile(r"\[([\d,\s]+)\]")

# Citation-inner allowed character set: digits, commas, and ASCII whitespace.
# Mirrors `_CITATION_RE`'s inner class so the streaming remapper makes the
# same accept/reject decision as the post-hoc regex extractor.
_CITATION_INNER_CHARS = frozenset("0123456789, \t")


class CitationRemapper:
    """Stream-safe renumberer for inline `[N]` citations.

    The LLM cites candidates by their 1-based position (so for 20 candidates
    it may emit `[10]`, `[16]`, `[22]`). We rewrite those to `[1]`, `[2]`,
    `[3]` in order of first appearance — the user-visible bracket numbers
    match the trailing source list / clickable order, not the (possibly
    sparse) candidate indices.

    Bracket tokens can arrive split across LLM chunks (`[1` then `0]`), so
    `feed()` buffers from `[` until either `]` or a non-citation char
    disqualifies the run. `flush()` releases any unclosed bracket literally —
    callers must invoke it once the stream ends.

    Non-citation brackets (markdown links `[text](url)`, code snippets) pass
    through unchanged: the first non-allowed char dumps the buffered content
    verbatim.
    """

    def __init__(self) -> None:
        self._mapping: dict[int, int] = {}
        self._buf = ""           # accumulates chars from '[' through ']'
        self._in_bracket = False # True iff _buf starts with '['

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if self._in_bracket:
                if ch == "]":
                    inner = self._buf[1:]
                    nums = re.findall(r"\d+", inner)
                    if nums and all(c in _CITATION_INNER_CHARS for c in inner):
                        new_parts: list[str] = []
                        for n_str in nums:
                            n = int(n_str)
                            if n not in self._mapping:
                                self._mapping[n] = len(self._mapping) + 1
                            new_parts.append(str(self._mapping[n]))
                        out.append("[" + ", ".join(new_parts) + "]")
                    else:
                        out.append(self._buf + ch)
                    self._buf = ""
                    self._in_bracket = False
                elif ch in _CITATION_INNER_CHARS:
                    self._buf += ch
                else:
                    # Disqualifies as a citation. Emit the buffered open-bracket
                    # run literally, then re-process the current char in the
                    # non-bracket state (it might be another '[').
                    out.append(self._buf)
                    self._buf = ""
                    self._in_bracket = False
                    if ch == "[":
                        self._buf = ch
                        self._in_bracket = True
                    else:
                        out.append(ch)
            else:
                if ch == "[":
                    self._buf = ch
                    self._in_bracket = True
                else:
                    out.append(ch)
        return "".join(out)

    def flush(self) -> str:
        """Release any unclosed bracket buffer literally. Idempotent."""
        out = self._buf
        self._buf = ""
        self._in_bracket = False
        return out

    @property
    def mapping(self) -> dict[int, int]:
        """{original_N: displayed_N}. Empty when no citations were seen."""
        return dict(self._mapping)


def _extract_usage(output: Any) -> dict[str, int] | None:
    """Pull token usage off an LLM response. Tolerates the multiple shapes
    LangChain emits: AIMessage.usage_metadata, response_metadata.token_usage,
    or a dict-shaped output. Returns None when no usage is available (e.g.,
    streaming responses that didn't enable include_usage)."""
    if output is None:
        return None
    meta = getattr(output, "usage_metadata", None)
    if not meta and isinstance(output, dict):
        meta = output.get("usage_metadata")
    if meta:
        return {
            "input": int(meta.get("input_tokens", 0) or 0),
            "output": int(meta.get("output_tokens", 0) or 0),
            "total": int(meta.get("total_tokens", 0) or 0),
        }
    rmeta = getattr(output, "response_metadata", None) or {}
    if isinstance(output, dict):
        rmeta = output.get("response_metadata") or rmeta
    tu = (rmeta or {}).get("token_usage") or {}
    if tu:
        in_ = int(tu.get("prompt_tokens", 0) or 0)
        out_ = int(tu.get("completion_tokens", 0) or 0)
        return {"input": in_, "output": out_, "total": in_ + out_}
    return None


def _extract_cited_indices(answer: str) -> set[int]:
    """Pull 1-based [N] citation numbers out of the generated answer.

    GENERATE prompt instructs the LLM to insert `[N]` inline citations matching
    the 1-based candidate order, so a doc was "used" iff its index appears
    bracketed in the answer body. Returns an empty set on chitchat / general /
    debugging answers (which have no candidates anyway)."""
    if not answer:
        return set()
    out: set[int] = set()
    for group in _CITATION_RE.findall(answer):
        for n in re.findall(r"\d+", group):
            out.add(int(n))
    return out


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
    "debug_explain",
    "instruction_save",
    "re_search_setup",
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
