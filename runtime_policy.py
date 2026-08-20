"""Explicit architecture names and single-writer ownership policy."""

from __future__ import annotations

import os
from typing import Final, Literal


Architecture = Literal["single", "multi_experimental"]
ARCHITECTURE_ENV: Final = "CALORIECHEF_ARCHITECTURE"
SINGLE: Final[Architecture] = "single"
MULTI_EXPERIMENTAL: Final[Architecture] = "multi_experimental"
EXPERIMENTAL_WARNING: Final = (
    "Warning: multi_experimental is an isolated engineering experiment; its "
    "qwen2.5:7b grounded nutrition happy path is not verified."
)

# Each mutable resource maps to exactly one owner in either runtime mode.
SINGLE_WRITER_INVENTORY: Final[dict[str, str]] = {
    "final_user_response_default": "single_agent_runtime",
    "final_user_response_multi_experimental": "CalorieChefManager",
    "long_term_chroma_memory": "deterministic_memory_router_and_upsert",
    "sqlite_short_term_session": "main_request_session_runtime",
    "trace_jsonl": "configured_local_trace_processor",
    "evaluation_and_experiment_artifacts": "relevant_report_runner",
    "source_code_and_configuration": "one_human_or_coding_agent_at_a_time",
}


def selected_architecture(value: str | None = None) -> Architecture:
    """Return the validated runtime selection, defaulting to stable single."""
    selected = (value if value is not None else os.getenv(ARCHITECTURE_ENV, SINGLE)).strip()
    selected = selected or SINGLE
    if selected not in {SINGLE, MULTI_EXPERIMENTAL}:
        raise ValueError(
            f"Unsupported {ARCHITECTURE_ENV}={selected!r}; use "
            f"{SINGLE!r} or {MULTI_EXPERIMENTAL!r}."
        )
    return selected  # type: ignore[return-value]


def experimental_warning() -> str:
    """Return the required warning for the non-default experimental mode."""
    return EXPERIMENTAL_WARNING
