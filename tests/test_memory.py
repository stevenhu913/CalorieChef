"""Deterministic regression test for short- and long-term memory behavior."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

from agents import SQLiteSession

from long_term_memory import (
    clear_user_memories,
    get_memory,
    list_memories,
    retrieve_context_memories,
    retrieve_memories,
    upsert_memory,
)
from memory import KEEP_HISTORY_ITEMS, keep_recent_history
from memory_router import route_memory_write


TEST_USER = "memory_regression_user"


def deterministic_embeddings(texts) -> list[list[float]]:
    """Return stable lexical vectors so this test never calls Ollama."""
    vocabulary = ("high", "protein", "lunch", "calorie", "target", "broccoli", "quantum")
    return [
        [float(text.lower().count(term)) for term in vocabulary]
        for text in texts
    ]


def store_routed(message: str, turn: str) -> None:
    decision = route_memory_write(message)
    assert decision.action == "keep", (message, decision)
    upsert_memory(
        topic=decision.topic or "general",
        value=decision.value or message,
        kind=decision.kind or "meal_preference",
        source_turn=turn,
        user_id=TEST_USER,
    )


async def verify_short_term() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "session.db"
        first = SQLiteSession("session-a", path)
        await first.add_items([{"role": "user", "content": "My nickname is Sam."}])
        reopened = SQLiteSession("session-a", path)
        stored = await reopened.get_items()
        assert stored[-1]["content"] == "My nickname is Sam."
        history = [{"role": "user", "content": str(index)} for index in range(12)]
        current = [{"role": "user", "content": "current"}]
        assert len(keep_recent_history(history, current)) == KEEP_HISTORY_ITEMS + 1


def verify_memory_behavior() -> None:
    clear_user_memories(TEST_USER)

    routing = {
        "Thanks.": "drop",
        "How many calories are in 42g protein, 38g carbs, and 15g fat?": "drop",
        "Make me lunch today.": "drop",
        "I usually prefer high-protein lunches.": "keep",
    }
    for message, expected in routing.items():
        assert route_memory_write(message).action == expected

    store_routed("I usually prefer high-protein lunches.", "session-1:turn-1")
    store_routed("My usual lunch target is 700 calories.", "session-1:turn-2")
    store_routed("I dislike broccoli.", "session-1:turn-3")

    records = list_memories(TEST_USER)
    assert len(records) == 3
    assert {record["kind"] for record in records} == {
        "meal_preference",
        "calorie_target",
        "disliked_ingredient",
    }

    recalled = retrieve_context_memories(
        "Suggest my usual high-protein lunch near my calorie target without foods I dislike.",
        user_id=TEST_USER,
    )
    assert len(recalled) == 3, recalled

    store_routed(
        "Actually, make my usual lunch target 600 calories from now on.",
        "session-2:turn-1",
    )
    target = get_memory("lunch_calorie_target", TEST_USER)
    assert target is not None
    assert target["value"] == "600 calories"
    assert target["version"] == 2
    assert "700 calories" not in " ".join(record["document"] for record in list_memories(TEST_USER))

    irrelevant = retrieve_memories("Explain quantum entanglement.", user_id=TEST_USER)
    assert irrelevant == [], irrelevant

    asyncio.run(verify_short_term())
    print("PASS A: three durable memory categories persisted")
    print("PASS B: a new retrieval context recalled all three categories")
    print("PASS C: lunch target updated from 700 to 600 at version 2")
    print("PASS D: transient messages were not routed to long-term memory")
    print("PASS E: irrelevant semantic retrieval returned no accepted memory")
    print("PASS F: SQLite persistence and the sliding window remain functional")
    print("PASS G: persistent vector retrieval remains functional")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with (
            patch("long_term_memory.VECTOR_STORE_PATH", Path(temp_dir) / "chroma"),
            patch("long_term_memory.embed_texts", side_effect=deterministic_embeddings),
        ):
            verify_memory_behavior()


if __name__ == "__main__":
    main()
