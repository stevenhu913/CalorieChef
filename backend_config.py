"""Pure deployment policy for model backend and memory mode selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class BackendPolicy:
    """Resolved model backend configuration without credential values."""

    backend: str
    model: str
    base_url: str | None
    ready: bool
    error: str | None


def _clean(value: str | None) -> str:
    return (value or "").strip()


def resolve_backend_policy(
    environ: Mapping[str, str] | None = None,
    *,
    ollama_probe: Callable[[], bool] | None = None,
) -> BackendPolicy:
    """Resolve hosted or local configuration without silent fallback."""
    env = os.environ if environ is None else environ
    backend = _clean(env.get("CALORIECHEF_MODEL_BACKEND")) or "ollama"
    if backend == "hosted":
        model = _clean(env.get("CALORIECHEF_HOSTED_MODEL"))
        if not _clean(env.get("OPENAI_API_KEY")):
            return BackendPolicy(backend, model, None, False, "Hosted model credential is missing.")
        if not model:
            return BackendPolicy(backend, model, None, False, "Hosted model name is missing.")
        return BackendPolicy(backend, model, None, True, None)
    if backend == "ollama":
        model = _clean(env.get("LOCAL_MODEL")) or "qwen2.5:7b"
        base_url = _clean(env.get("LOCAL_BASE_URL")) or "http://localhost:11434/v1"
        ready = bool(ollama_probe and ollama_probe())
        return BackendPolicy(
            backend,
            model,
            base_url,
            ready,
            None if ready else "Local Ollama is unavailable.",
        )
    return BackendPolicy(backend, "", None, False, "Unsupported model backend.")


def resolve_long_term_memory_enabled(
    backend: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Default memory to full locally and limited for hosted deployment."""
    env = os.environ if environ is None else environ
    configured = _clean(env.get("CALORIECHEF_ENABLE_LONG_TERM_MEMORY")).lower()
    if configured:
        if configured not in {"true", "false"}:
            raise ValueError("CALORIECHEF_ENABLE_LONG_TERM_MEMORY must be true or false.")
        return configured == "true"
    return backend == "ollama"
