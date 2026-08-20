"""Local CLI using the shared Single-Agent Core."""

from __future__ import annotations

import asyncio

from agent_core import CalorieChefService
from memory import get_session_id


async def async_main() -> int:
    """Start one service, run the local conversation loop, and close it."""
    service = CalorieChefService()
    await service.start()
    if not service.ready:
        print(f"CalorieChef is not ready: {service.readiness_error}")
        return 1
    print("=== CalorieChef is ready ===")
    print(f"Session: {get_session_id()}")
    print(f"Model backend: {service.model_backend}")
    print(f"Memory mode: {service.memory_mode}")
    print("Type 'exit' or 'quit' to stop.")
    try:
        while True:
            try:
                message = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0
            if message.lower() in {"exit", "quit"}:
                print("Goodbye!")
                return 0
            if not message:
                continue
            try:
                result = await service.answer(message, get_session_id())
                print("\nCalorieChef:", result.answer)
            except Exception as exc:
                print(
                    "\nCalorieChef could not complete that request. "
                    f"{type(exc).__name__}. Please retry; exact nutrition was not guessed."
                )
    finally:
        await service.close()


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
