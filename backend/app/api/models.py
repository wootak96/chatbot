from __future__ import annotations

import time

from fastapi import APIRouter

router = APIRouter()

MODEL_ID = "rag-chatbot"


@router.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "internal",
            }
        ],
    }
