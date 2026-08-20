"""Instance-local short-term memory for CalorieChef."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agents import RunConfig, SessionSettings, SQLiteSession


ROOT = Path(__file__).resolve().parent
SESSION_DB_PATH = ROOT / "caloriechef.db"
DEFAULT_SESSION_ID = "caloriechef_default"
KEEP_HISTORY_ITEMS = 8


def keep_recent_history(
    history: list[Any],
    new_input: list[Any],
) -> list[Any]:
    """Send recent history plus the current input without deleting storage."""
    return history[-KEEP_HISTORY_ITEMS:] + new_input


def get_session_id() -> str:
    """Return the stable product session ID, with an optional override."""
    configured = os.getenv("CALORIECHEF_SESSION_ID", DEFAULT_SESSION_ID).strip()
    return configured or DEFAULT_SESSION_ID


def create_session(session_id: str | None = None) -> SQLiteSession:
    """Create the persistent SQLite session used by the CLI."""
    return SQLiteSession(session_id or get_session_id(), SESSION_DB_PATH)


RUN_CONFIG = RunConfig(
    session_input_callback=keep_recent_history,
    session_settings=SessionSettings(limit=50),
)
