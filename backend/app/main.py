from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.models import router as models_router
from app.api.web import router as web_router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="RAG Chatbot Backend", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(web_router)
    app.include_router(chat_router)
    app.include_router(models_router)

    @app.get("/info")
    async def info() -> dict:
        return {
            "service": "rag-chatbot-backend",
            "version": "0.1.0",
            "provider": settings.llm_provider,
            "model": settings.llm_model_label,
            "indices": settings.all_indices,
            "endpoints": {
                "ui": "GET /",
                "chat": "POST /v1/chat/completions",
                "models": "GET /v1/models",
                "health": "GET /health",
                "docs": "GET /docs",
            },
        }

    @app.get("/health")
    async def health() -> dict:
        """Liveness + readiness check.

        Pings Elasticsearch and reports per-index existence so the operator
        can verify .env values immediately after editing them.
        """
        from app.services.elasticsearch_client import get_es_client

        es_status: dict = {"reachable": False, "error": None, "indices": {}}
        try:
            client = get_es_client()
            info = await client.info()
            es_status["reachable"] = True
            es_status["cluster_name"] = info.get("cluster_name")
            es_status["version"] = info.get("version", {}).get("number")
            for idx in settings.all_indices:
                exists = await client.indices.exists(index=idx)
                es_status["indices"][idx] = bool(exists)
        except Exception as e:  # noqa: BLE001
            es_status["error"] = f"{type(e).__name__}: {e}"

        active_key = settings.active_api_key
        llm_status: dict = {
            "provider": settings.llm_provider,
            "model": settings.llm_model_label,
            "api_key_configured": bool(
                active_key and active_key not in {"dummy", "replace-me"}
            ),
        }
        if settings.llm_provider == "openai":
            llm_status["base_url"] = settings.openai_base_url.strip() or "https://api.openai.com/v1"
        else:
            llm_status["endpoint"] = settings.hchat_endpoint
        all_ok = es_status["reachable"] and all(es_status["indices"].values())
        return {
            "status": "ok" if all_ok else "degraded",
            "elasticsearch": es_status,
            "llm": llm_status,
        }

    return app


app = create_app()
