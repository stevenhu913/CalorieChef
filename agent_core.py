"""FastAPI-independent stable Single-Agent CalorieChef service."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import _local
from agents import Runner, custom_span, trace
from agents.mcp import MCPServerStdio
from pydantic import BaseModel, Field

from agent import create_agent
from backend_config import resolve_long_term_memory_enabled
from memory import RUN_CONFIG, create_session
from observability import configure_local_tracing, summarize_query
from presentation import format_meal_plan


ROOT = Path(__file__).resolve().parent
NUTRITION_SERVER = ROOT / "nutrition_mcp_server.py"
REQUIRED_MCP_TOOLS = {"search_food", "get_food_nutrition"}
MAX_TURNS = 12
MEAL_TOOL_NAME = "calculate_meal_nutrition"
MEAL_FORMAT_FAILURE = (
    "Meal nutrition was calculated, but the structured result could not be "
    "formatted safely. Please retry."
)
MEAL_ORCHESTRATION_FAILURE = (
    "A single authoritative meal plan could not be produced because the meal "
    "calculator ran more than once. Please retry."
)


class AgentAnswer(BaseModel):
    """Stable structured result shared by CLI, Web, and tests."""

    answer: str
    thread_id: str
    mode: str
    architecture: str = "single"
    memory_mode: str
    trace_id: str | None = None
    tools_called: list[str] = Field(default_factory=list)


class AgentCoreUnavailable(RuntimeError):
    """Raised when startup prerequisites are not ready."""


def _tool_names(items: list[Any]) -> list[str]:
    names: list[str] = []
    for item in items:
        name = getattr(item, "tool_name", None)
        raw = getattr(item, "raw_item", None)
        if name is None:
            name = (
                raw.get("name")
                if isinstance(raw, Mapping)
                else getattr(raw, "name", None)
            )
        if name and name not in names:
            names.append(str(name))
    return names


def _decode_tool_output(value: Any) -> dict[str, Any] | None:
    """Decode only mapping or JSON-object tool outputs."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(decoded) if isinstance(decoded, Mapping) else None
    return None


def _meal_tool_evidence(
    items: list[Any],
) -> tuple[int, int, dict[str, Any] | None, bool]:
    """Return meal call counts, the sole output, and parse failure state."""
    calls: dict[str, str] = {}
    meal_call_count = 0
    successful_count = 0
    meal_output: dict[str, Any] | None = None
    malformed = False

    for item in items:
        item_type = getattr(item, "type", None)
        if item_type == "tool_call_item":
            name = getattr(item, "tool_name", None)
            call_id = getattr(item, "call_id", None)
            if name and call_id:
                calls[str(call_id)] = str(name)
                if name == MEAL_TOOL_NAME:
                    meal_call_count += 1
        elif item_type == "tool_call_output_item":
            call_id = getattr(item, "call_id", None)
            if call_id and calls.get(str(call_id)) == MEAL_TOOL_NAME:
                decoded = _decode_tool_output(getattr(item, "output", None))
                if decoded is None:
                    malformed = True
                else:
                    meal_output = decoded
                    if decoded.get("status") == "ok":
                        successful_count += 1

    if meal_call_count and meal_output is None:
        malformed = True
    return meal_call_count, successful_count, meal_output, malformed


def _authoritative_meal_output(
    model_output: str,
    items: list[Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve one final response while treating a sole meal result as final."""
    call_count, successful_count, meal_result, malformed = _meal_tool_evidence(items)
    metadata = {
        "meal_tool_call_count": call_count,
        "successful_meal_tool_call_count": successful_count,
        "orchestration_violation": call_count > 1,
        "model_final_output_discarded": False,
        "presentation_status": "model_output",
    }
    if call_count > 1:
        metadata["presentation_status"] = "duplicate_meal_tool_calls"
        metadata["model_final_output_discarded"] = True
        return MEAL_ORCHESTRATION_FAILURE, metadata
    if malformed:
        metadata["presentation_status"] = "unsafe_tool_output"
        metadata["model_final_output_discarded"] = True
        return MEAL_FORMAT_FAILURE, metadata
    if call_count == 1 and meal_result and meal_result.get("status") == "ok":
        metadata["presentation_status"] = "formatted"
        metadata["model_final_output_discarded"] = True
        return format_meal_plan(meal_result), metadata
    return model_output, metadata


class CalorieChefService:
    """Own one bounded MCP lifecycle and serve stable Single-Agent requests."""

    def __init__(self, *, memory_enabled: bool | None = None) -> None:
        self.model_backend = _local.backend_mode()
        self.memory_enabled = (
            resolve_long_term_memory_enabled(self.model_backend)
            if memory_enabled is None
            else memory_enabled
        )
        self.memory_mode = "full_local" if self.memory_enabled else "limited"
        self.architecture = "single"
        self.ready = False
        self.readiness_error: str | None = None
        self._mcp_context: MCPServerStdio | None = None
        self.nutrition_server: Any | None = None

    async def start(self) -> None:
        """Validate configuration and start the USDA MCP subprocess once."""
        configure_local_tracing()
        if os.getenv("CALORIECHEF_ARCHITECTURE", "single").strip() not in {"", "single"}:
            self.readiness_error = "Web deployment supports only the single architecture."
            return
        if not _local.backend_ready():
            self.readiness_error = _local.backend_readiness_error() or "Model backend is unavailable."
            return
        if self.model_backend == "hosted" and self.memory_enabled:
            self.readiness_error = "Hosted long-term memory configuration is unavailable."
            return
        if not os.getenv("USDA_API_KEY", "").strip():
            self.readiness_error = "USDA service credential is missing."
            return
        try:
            # Reuse one USDA MCP subprocess for the application lifetime.
            self._mcp_context = MCPServerStdio(
                name="CalorieChef USDA Nutrition Server",
                params={"command": sys.executable, "args": [str(NUTRITION_SERVER)]},
                cache_tools_list=True,
                client_session_timeout_seconds=30,
            )
            self.nutrition_server = await self._mcp_context.__aenter__()
            available = {tool.name for tool in await self.nutrition_server.list_tools()}
            missing = REQUIRED_MCP_TOOLS - available
            if missing:
                raise RuntimeError("Required USDA MCP tools are unavailable.")
            self.ready = True
        except Exception as exc:
            self.readiness_error = f"USDA MCP startup failed: {type(exc).__name__}."
            await self.close()

    async def close(self) -> None:
        """Close the one managed MCP subprocess cleanly."""
        context = self._mcp_context
        self._mcp_context = None
        self.nutrition_server = None
        self.ready = False
        if context is not None:
            await context.__aexit__(None, None, None)

    async def _memory_context(
        self,
        message: str,
        thread_id: str,
        recent_text: str,
    ) -> tuple[str, int]:
        """Route and retrieve memory only when the configured mode enables it."""
        if not self.memory_enabled:
            return "Long-term memory is disabled for this runtime.", 0
        from long_term_memory import format_evidence, retrieve_context_memories, upsert_memory
        from memory_router import route_memory_write

        decision = route_memory_write(message)
        if decision.action == "keep":
            upsert_memory(
                topic=decision.topic or "general",
                value=decision.value or message,
                kind=decision.kind or "meal_preference",
                source_turn=f"web:{thread_id}:{uuid4().hex[:12]}",
                user_id=f"thread:{thread_id}",
            )
        memories = retrieve_context_memories(
            message,
            recent_text=f"{recent_text} {message}",
            user_id=f"thread:{thread_id}",
        )
        return format_evidence(memories), len(memories)

    async def answer(self, message: str, thread_id: str) -> AgentAnswer:
        """Run one bounded stable Single-Agent request."""
        message = message.strip()
        if not message:
            raise ValueError("Message must not be empty.")
        if not self.ready or self.nutrition_server is None:
            raise AgentCoreUnavailable(self.readiness_error or "Agent Core is not ready.")

        session = create_session(thread_id)
        stored = await session.get_items()
        recent_text = " ".join(str(item.get("content", "")) for item in stored[-8:])
        evidence, memory_count = await self._memory_context(message, thread_id, recent_text)
        output = ""
        tools_called: list[str] = []

        with trace(
            "caloriechef_web_request",
            metadata={
                "architecture": "single",
                "model_backend": self.model_backend,
                "memory_mode": self.memory_mode,
                "query_type": summarize_query(message),
                "raw_user_text_recorded": False,
            },
        ) as workflow:
            try:
                agent = create_agent(self.nutrition_server, evidence)
                with custom_span(
                    "agent_core_run",
                    data={"max_turns": MAX_TURNS, "memory_source_count": memory_count},
                ):
                    result = await Runner.run(
                        agent,
                        message,
                        session=session,
                        run_config=RUN_CONFIG,
                        max_turns=MAX_TURNS,
                    )
                    model_output = str(result.final_output)
                    tools_called = _tool_names(result.new_items)
                    try:
                        output, presentation_data = _authoritative_meal_output(
                            model_output,
                            result.new_items,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        output = MEAL_FORMAT_FAILURE
                        presentation_data = {
                            "meal_tool_call_count": 1,
                            "successful_meal_tool_call_count": 1,
                            "orchestration_violation": False,
                            "model_final_output_discarded": True,
                            "presentation_status": "invalid_tool_schema",
                            "error_type": type(exc).__name__,
                        }
                    with custom_span(
                        "meal_presentation",
                        data=presentation_data,
                    ) as presentation_span:
                        if presentation_data["presentation_status"] not in {
                            "formatted",
                            "model_output",
                        }:
                            presentation_span.set_error(
                                {
                                    "message": "A safe authoritative meal response could not be produced.",
                                    "data": {},
                                }
                            )
                with custom_span(
                    "final_response",
                    data={"response_character_count": len(output)},
                ):
                    pass
            except Exception as exc:
                with custom_span(
                    "request_error",
                    data={"error_type": type(exc).__name__, "recovered": False},
                ) as error_span:
                    error_span.set_error(
                        {"message": f"{type(exc).__name__}: {exc}", "data": {}}
                    )
                raise

        return AgentAnswer(
            answer=output,
            thread_id=thread_id,
            mode=self.model_backend,
            memory_mode=self.memory_mode,
            trace_id=workflow.trace_id,
            tools_called=tools_called,
        )


async def answer(message: str, thread_id: str, service: CalorieChefService) -> AgentAnswer:
    """Call the shared Agent Core through an explicitly managed service."""
    return await service.answer(message, thread_id)


async def _example() -> int:
    service = CalorieChefService()
    await service.start()
    try:
        if not service.ready:
            print(f"Agent Core unavailable: {service.readiness_error}")
            return 1
        result = await service.answer(
            "I have 42 g of protein, 38 g of carbohydrates, and 15 g of fat. "
            "How many calories is that?",
            "caloriechef_core_example",
        )
        print(f"architecture={result.architecture}")
        print(f"mode={result.mode}")
        print(f"memory_mode={result.memory_mode}")
        print(f"tools_called={','.join(result.tools_called) or 'none'}")
        print(f"answer={result.answer}")
        print(f"trace_id={result.trace_id}")
        return 0
    finally:
        await service.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_example()))
