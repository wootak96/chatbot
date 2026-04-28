"""Verify per-request LLM API key resolution, header extraction, and provider switch."""

import pytest

from app.api.chat import _extract_bearer
from app.config import Settings
from app.services.llm_factory import resolve_api_key, set_api_key


def test_resolve_api_key_falls_back_to_settings():
    set_api_key(None)
    from app.config import get_settings

    assert resolve_api_key() == get_settings().active_api_key


def test_resolve_api_key_uses_override():
    try:
        set_api_key("custom-user-key")
        assert resolve_api_key() == "custom-user-key"
    finally:
        set_api_key(None)


def test_resolve_api_key_clears_on_none():
    set_api_key("temp")
    set_api_key(None)
    from app.config import get_settings

    assert resolve_api_key() == get_settings().active_api_key


def test_extract_bearer_valid():
    assert _extract_bearer("Bearer real-key-abc") == "real-key-abc"
    assert _extract_bearer("bearer real-key-abc") == "real-key-abc"


def test_extract_bearer_missing_or_invalid():
    assert _extract_bearer(None) is None
    assert _extract_bearer("") is None
    assert _extract_bearer("Basic foo") is None
    assert _extract_bearer("real-key-abc") is None  # no scheme


def test_extract_bearer_filters_placeholders():
    assert _extract_bearer("Bearer dummy") is None
    assert _extract_bearer("Bearer dummy-key") is None
    assert _extract_bearer("Bearer placeholder") is None
    assert _extract_bearer("Bearer ") is None


def test_active_api_key_picks_provider_specific():
    azure = Settings(
        llm_provider="azure",
        hchat_api_key="az-key",
        openai_api_key="oa-key",
    )
    assert azure.active_api_key == "az-key"
    assert azure.llm_model_label == azure.hchat_deployment

    openai = Settings(
        llm_provider="openai",
        hchat_api_key="az-key",
        openai_api_key="oa-key",
        openai_model="gpt-4o-mini",
    )
    assert openai.active_api_key == "oa-key"
    assert openai.llm_model_label == "gpt-4o-mini"


def _reset_caches():
    from app.config import get_settings
    from app.services import llm_factory

    get_settings.cache_clear()
    llm_factory._build.cache_clear()


def test_factory_builds_chat_openai_when_provider_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-public")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    _reset_caches()
    try:
        from app.services.llm_factory import get_judge_llm
        from langchain_openai import ChatOpenAI

        llm = get_judge_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o-mini"
    finally:
        _reset_caches()


def test_factory_builds_azure_when_provider_azure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setenv("HCHAT_API_KEY", "sk-test-azure")
    _reset_caches()
    try:
        from app.services.llm_factory import get_judge_llm
        from langchain_openai import AzureChatOpenAI

        llm = get_judge_llm()
        assert isinstance(llm, AzureChatOpenAI)
    finally:
        _reset_caches()


def test_factory_uses_custom_base_url_when_set(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-proxy.local/v1")
    _reset_caches()
    try:
        from app.services.llm_factory import get_judge_llm

        llm = get_judge_llm()
        # langchain_openai stores it as openai_api_base on the underlying client
        assert "my-proxy.local" in str(llm.openai_api_base or llm.root_client.base_url)
    finally:
        _reset_caches()
