"""Inspect stored short-term history and the smaller model-input window.

Run:
    python -m scripts.inspect_history

This developer-facing script reads the same persistent session as the CLI.
It reports item counts, not token counts, and does not modify the history.
"""

from __future__ import annotations

import asyncio

from memory import (
    KEEP_HISTORY_ITEMS,
    SESSION_DB_PATH,
    create_session,
    get_session_id,
    keep_recent_history,
)


async def main() -> None:
    """Print complete stored-history and simulated next-input counts."""
    session = create_session()
    stored = await session.get_items()
    next_input = [
        {
            "role": "user",
            "content": "[inspection-only example of the next user message]",
        }
    ]
    model_input = keep_recent_history(stored, next_input)

    print(f"Session ID: {get_session_id()}")
    print(f"Database: {SESSION_DB_PATH}")
    print(f"Stored history items: {len(stored)}")
    print(
        "History items kept for the next model call: "
        f"{min(len(stored), KEEP_HISTORY_ITEMS)}"
    )
    print(
        "Total next model-input items including the example input: "
        f"{len(model_input)}"
    )
    print("Counts above are message/item counts, not token counts.")

    if len(stored) > KEEP_HISTORY_ITEMS:
        print("Verified: full storage is larger than the trimmed history window.")
    else:
        print(
            "The session has not exceeded the window yet. Continue the CLI "
            f"until it contains more than {KEEP_HISTORY_ITEMS} items, then rerun."
        )


if __name__ == "__main__":
    asyncio.run(main())
