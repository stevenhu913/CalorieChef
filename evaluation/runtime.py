"""Isolated Agent runtime used only by the offline evaluator."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

from agents import Runner, custom_span, trace

from agent import create_agent
from evaluation.models import EvalRunResult
from long_term_memory import (
    format_evidence,
    list_memories,
    retrieve_context_memories,
    upsert_memory,
)
from memory import RUN_CONFIG, create_session
from memory_router import route_memory_write
from evaluation.trace_evidence import extract_trace


MAX_TURNS = 12


def configure_case_scope(user_id: str, session_id: str) -> None:
    os.environ["CALORIECHEF_USER_ID"] = user_id
    os.environ["CALORIECHEF_SESSION_ID"] = session_id


def apply_memory_statement(message: str, source_turn: str, user_id: str) -> dict[str, Any] | None:
    """Apply the production deterministic router/upsert without an Agent call."""
    decision = route_memory_write(message)
    if decision.action != "keep":
        return None
    return upsert_memory(
        topic=decision.topic or "general",
        value=decision.value or message,
        kind=decision.kind or "meal_preference",
        source_turn=source_turn,
        user_id=user_id,
    )


def _decode_output(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if isinstance(value.get("structuredContent"), dict):
            return value["structuredContent"]
        if "content" in value:
            return _decode_output(value["content"])
        if isinstance(value.get("text"), str):
            return _decode_output(value["text"])
        return value
    if isinstance(value, list):
        for item in value:
            decoded = _decode_output(item)
            if decoded:
                return decoded
        return {}
    if isinstance(value, str):
        try:
            return _decode_output(json.loads(value))
        except json.JSONDecodeError:
            try:
                return _decode_output(ast.literal_eval(value))
            except (SyntaxError, ValueError):
                return {}
    return {}


def _sanitize_tool_result(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "search_food": ("status", "count", "query"),
        "get_food_nutrition": (
            "status",
            "fdc_id",
            "description",
            "basis",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
        ),
        "calculate_macro_calories": (
            "status",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
        ),
        "calculate_meal_nutrition": (
            "status",
            "target_calories",
            "actual_calories",
            "target_met",
            "final_target_met",
            "calorie_difference",
            "absolute_calorie_gap",
            "calorie_gap_percentage",
            "tolerance_calories",
            "target_status",
            "preferred_range",
            "expanded_range",
            "preferred_target_met",
            "preferred_actual_calories",
            "preferred_calorie_difference",
            "preferred_absolute_calorie_gap",
            "preferred_minimum_possible_calories",
            "preferred_maximum_possible_calories",
            "expanded_search_attempted",
            "expanded_actual_calories",
            "expanded_calorie_difference",
            "expanded_absolute_calorie_gap",
            "expanded_target_met",
            "expanded_result_improved_gap",
            "used_expanded_range",
            "search_range_used",
            "target_feasible_within_bounds",
            "minimum_possible_calories",
            "maximum_possible_calories",
            "expanded_minimum_possible_calories",
            "expanded_maximum_possible_calories",
            "adjustment_direction",
            "boundary_hits",
            "feasibility_message",
            "items",
            "totals",
        ),
    }
    result = {"tool": tool}
    for key in allowed.get(tool, ("status",)):
        if key in payload:
            result[key] = payload[key]
    return result


def _extract_tool_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for item in items:
        if item.get("type") == "function_call":
            calls[str(item.get("call_id", ""))] = str(item.get("name", ""))
        elif item.get("type") == "function_call_output":
            tool = calls.get(str(item.get("call_id", "")), "unknown")
            results.append(_sanitize_tool_result(tool, _decode_output(item.get("output"))))
    return results


async def run_live_turn(
    case: dict[str, Any],
    nutrition_server: Any,
    trace_path: Path,
    user_id: str,
    session_id: str,
) -> EvalRunResult:
    """Run one isolated production-equivalent turn and return sanitized evidence."""
    configure_case_scope(user_id, session_id)
    session = create_session()
    memory_before = list_memories(user_id)
    answer = ""
    error: str | None = None
    decision = route_memory_write(case["user_input"])
    retrieved: list[dict[str, Any]] = []

    with trace(
        "caloriechef_request",
        metadata={
            "query_type": "offline_evaluation",
            "session_id": session_id,
            "case_id": case["case_id"],
            "raw_user_text_recorded": False,
        },
    ) as workflow:
        try:
            with custom_span(
                "memory_write_routing",
                data={"action": decision.action, "kind": decision.kind, "topic": decision.topic},
            ):
                pass
            if decision.action == "keep":
                with custom_span(
                    "long_term_memory_write",
                    data={"kind": decision.kind, "topic": decision.topic},
                ) as write_span:
                    saved = upsert_memory(
                        topic=decision.topic or "general",
                        value=decision.value or case["user_input"],
                        kind=decision.kind or "meal_preference",
                        source_turn=f"{session_id}:evaluated-turn",
                        user_id=user_id,
                    )
                    write_span.span_data.data["version"] = saved["version"]
            with custom_span("short_term_memory", data={}) as short_span:
                stored = await session.get_items()
                short_span.span_data.data["stored_item_count"] = len(stored)
                recent = " ".join(str(item.get("content", "")) for item in stored[-8:])
            with custom_span("long_term_retrieval", data={"threshold": 0.38, "max_results": 3}) as retrieval_span:
                retrieved = retrieve_context_memories(
                    case["user_input"],
                    recent_text=f"{recent} {case['user_input']}",
                    user_id=user_id,
                )
                retrieval_span.span_data.data.update(
                    {
                        "accepted_count": len(retrieved),
                        "memory_ids": [item["id"] for item in retrieved],
                    }
                )
            agent = create_agent(nutrition_server, format_evidence(retrieved))
            with custom_span("agent_run", data={"session_id": session_id, "max_turns": MAX_TURNS}) as agent_span:
                result = await Runner.run(
                    agent,
                    case["user_input"],
                    session=session,
                    run_config=RUN_CONFIG,
                    max_turns=MAX_TURNS,
                )
                answer = str(result.final_output)
                agent_span.span_data.data["success"] = True
            with custom_span("final_response", data={"response_character_count": len(answer)}):
                pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with custom_span(
                "request_error",
                data={"error_type": type(exc).__name__, "recovered": True},
            ) as error_span:
                error_span.set_error({"message": error, "data": {}})

    stored_after = await session.get_items()
    trace_evidence = extract_trace(trace_path, workflow.trace_id)
    tool_names = trace_evidence.tool_names
    return EvalRunResult(
        case_id=case["case_id"],
        answer=answer,
        router_action=decision.action,
        tool_names=tool_names,
        tool_results=_extract_tool_results(stored_after),
        retrieved_memory_ids=[item["id"] for item in retrieved],
        retrieved_memory_kinds=[str(item["metadata"].get("kind")) for item in retrieved],
        memory_before=memory_before,
        memory_after=list_memories(user_id),
        trace=trace_evidence,
        error=error,
    )
