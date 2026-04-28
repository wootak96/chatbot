"""OpenAI-compatible SSE chunk encoders."""

from __future__ import annotations

import json
import time
import uuid


def _chunk(content: str, *, model: str, completion_id: str, finish_reason: str | None = None) -> dict:
    delta: dict = {}
    if content:
        delta["content"] = content
    if finish_reason is None and not content:
        delta = {}
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def encode_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def role_chunk(*, model: str, completion_id: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    return encode_sse(payload)


def text_chunk(text: str, *, model: str, completion_id: str) -> str:
    return encode_sse(_chunk(text, model=model, completion_id=completion_id))


def stop_chunk(*, model: str, completion_id: str) -> str:
    return encode_sse(_chunk("", model=model, completion_id=completion_id, finish_reason="stop"))


def done_marker() -> str:
    return "data: [DONE]\n\n"
