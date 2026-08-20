"""Configure the explicit hosted or local model backend."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from backend_config import BackendPolicy, resolve_backend_policy

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass


def _ollama_alive() -> bool:
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    try:
        urllib.request.urlopen(base_url.replace("/v1", "") + "/api/version", timeout=0.6)
        return True
    except Exception:
        return False


POLICY: BackendPolicy = resolve_backend_policy(ollama_probe=_ollama_alive)


def _set_default_agent_model(model: str) -> None:
    if not model:
        return
    import agents

    if getattr(agents.Agent, "_seeu_default_model_patch", False):
        return
    original_init = agents.Agent.__init__

    def init_with_default_model(self, *args, **kwargs):
        if not kwargs.get("model"):
            kwargs["model"] = model
        original_init(self, *args, **kwargs)

    agents.Agent.__init__ = init_with_default_model
    agents.Agent._seeu_default_model_patch = True


if POLICY.ready and POLICY.backend == "ollama":
    from openai import AsyncOpenAI
    from agents import set_default_openai_api, set_default_openai_client

    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    set_default_openai_client(AsyncOpenAI(base_url=POLICY.base_url, api_key="ollama"))
    set_default_openai_api("chat_completions")
    _set_default_agent_model(POLICY.model)
elif POLICY.ready and POLICY.backend == "hosted":
    _set_default_agent_model(POLICY.model)


def backend_mode() -> str:
    """Return the explicitly selected backend name."""
    return POLICY.backend


def backend_ready() -> bool:
    """Return whether required model configuration is available."""
    return POLICY.ready


def backend_readiness_error() -> str | None:
    """Return a safe readiness error without credential details."""
    return POLICY.error
