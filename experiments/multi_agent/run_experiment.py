"""
main.py — command-line entry point for CalorieChef.

Run:
    python main.py

This experimental entry point contains:
- The multi-turn conversation loop
- Persistent SQLite short-term memory with a sliding model-input window
- Basic command-line interaction
- Programmatic USDA MCP server startup and shutdown
- Manager-owned execution with bounded specialist Agents

The production Single-Agent prompt remains in prompts.py for comparison.
Manager and specialist boundaries are defined in this package.
"""

from __future__ import annotations

import asyncio
import sys
from uuid import uuid4
from pathlib import Path

import _local  # noqa: F401
from agents.mcp import MCPServerStdio

from long_term_memory import (
    get_user_id,
    list_memories,
    retrieve_context_memories,
    upsert_memory,
)
from memory import create_session, get_session_id
from memory_router import route_memory_write
from experiments.multi_agent.orchestrator import CalorieChefManager
from observability import configure_local_tracing, summarize_query


ROOT = Path(__file__).resolve().parent
NUTRITION_SERVER = ROOT.parents[1] / "nutrition_mcp_server.py"
REQUIRED_MCP_TOOLS = {"search_food", "get_food_nutrition"}


configure_local_tracing()


async def _chat(nutrition_server) -> None:
    """Run the user-facing conversation loop without internal trace output."""
    session = create_session()
    manager = CalorieChefManager(nutrition_server)

    print("=== CalorieChef is ready ===")
    print(f"Session: {get_session_id()}")
    print("Type 'exit' or 'quit' to stop.")
    print(
        "Try: 'How many calories and how much protein are in "
        "chicken breast?'"
    )

    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if user.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        if not user:
            continue

        try:
            decision = route_memory_write(user)
            saved_memory = None
            if decision.action == "keep":
                saved_memory = upsert_memory(
                    topic=decision.topic or "general",
                    value=decision.value or user,
                    kind=decision.kind or "meal_preference",
                    source_turn=f"{get_session_id()}:{uuid4().hex[:12]}",
                )

            stored_items = await session.get_items()
            recent_text = " ".join(
                str(item.get("content", "")) for item in stored_items[-8:]
            )
            memories = retrieve_context_memories(
                user,
                recent_text=f"{recent_text} {user}",
            )
            result = await manager.run(
                user,
                memories,
                session=session,
                preprocessing_trace_data={
                    "memory_write_routing": {
                        "action": decision.action,
                        "kind": decision.kind,
                        "topic": decision.topic,
                    },
                    "short_term_memory": {"stored_item_count": len(stored_items)},
                    "long_term_retrieval": {
                        "threshold": 0.38,
                        "max_results": 3,
                        "accepted_count": len(memories),
                        "memory_ids": [memory["id"] for memory in memories],
                    },
                    "request_metadata": {
                        "query_type": summarize_query(user),
                        "session_id": get_session_id(),
                    },
                },
            )
        except Exception as exc:
            print(
                "\nCalorieChef could not complete that request. "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "Please try again. Exact nutrition values will not be "
                "guessed when specialist evidence is unavailable."
            )
            continue

        print("\nCalorieChef:", result.answer)
        if saved_memory:
            print(
                "Memory saved: "
                f"topic={saved_memory['topic']} version={saved_memory['version']}"
            )
        if memories:
            source_ids = ", ".join(memory["id"] for memory in memories)
            print(f"Long-term memory sources: {source_ids}")


async def async_main() -> int:
    """Start the USDA MCP subprocess, run the CLI, and close it cleanly."""
    try:
        memory_count = len(list_memories(get_user_id()))
    except Exception as exc:
        print(
            "CalorieChef could not open its long-term memory index. "
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "Check ChromaDB, Ollama, and the nomic-embed-text model, "
            "then try again."
        )
        return 1

    print(f"Long-term user memory: {memory_count} active records ready.")

    try:
        async with MCPServerStdio(
            name="CalorieChef USDA Nutrition Server",
            params={
                "command": sys.executable,
                "args": [str(NUTRITION_SERVER)],
            },
            cache_tools_list=True,
            client_session_timeout_seconds=30,
        ) as nutrition_server:
            available_tools = {
                tool.name for tool in await nutrition_server.list_tools()
            }
            missing_tools = REQUIRED_MCP_TOOLS - available_tools

            if missing_tools:
                names = ", ".join(sorted(missing_tools))
                print(
                    "CalorieChef could not start because the USDA MCP "
                    f"server is missing required tools: {names}."
                )
                return 1

            await _chat(nutrition_server)
            return 0

    except Exception as exc:
        print(
            "CalorieChef could not connect to its USDA nutrition service. "
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "Check the Python dependencies, USDA_API_KEY, and network "
            "connection, then try again."
        )
        return 1


def main() -> None:
    """Run the asynchronous experimental application entry point."""
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
