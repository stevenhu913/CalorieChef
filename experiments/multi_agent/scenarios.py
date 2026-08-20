"""Run controlled scenarios for the experimental multi-Agent architecture."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import _local  # noqa: F401
from agents.mcp import MCPServerStdio

from long_term_memory import clear_user_memories, retrieve_context_memories, upsert_memory
from evaluation.trace_evidence import extract_trace
from experiments.multi_agent.orchestrator import CalorieChefManager, _food_conflict
from observability import configure_local_tracing


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
TRACE_PATH = ARTIFACT_DIR / "scenario_traces.jsonl"
RESULT_PATH = ARTIFACT_DIR / "scenario_results.json"
NUTRITION_SERVER = PROJECT_ROOT / "nutrition_mcp_server.py"


def _setup_memory(user_id: str) -> None:
    records = (
        ("allergy:peanut", "peanut", "allergy"),
        ("disliked_ingredient:broccoli", "broccoli", "disliked_ingredient"),
        ("lunch_calorie_target", "600 calories", "calorie_target"),
        ("lunch_style", "high-protein lunches", "meal_preference"),
    )
    for index, (topic, value, kind) in enumerate(records, start=1):
        upsert_memory(
            topic=topic,
            value=value,
            kind=kind,
            source_turn=f"experiment-setup:{index}",
            user_id=user_id,
        )


def _execution_summary(result) -> list[dict[str, Any]]:
    return [
        {
            "specialist": execution.specialist_name,
            "status": execution.status,
            "timeout": execution.timeout,
            "fallback_used": execution.fallback_used,
            "result_used": execution.result_used,
            "latency_ms": execution.latency_ms,
            "limitations": execution.result.limitations,
        }
        for execution in result.executions
    ]


async def _run_case(
    *,
    case_id: str,
    request: str,
    nutrition_server: Any,
    setup_personalization: bool,
    force_timeout_specialist: str | None = None,
) -> dict[str, Any]:
    scope = uuid.uuid4().hex[:10]
    user_id = f"experiment_{case_id}_{scope}"
    os.environ["CALORIECHEF_USER_ID"] = user_id
    os.environ["CALORIECHEF_SESSION_ID"] = f"{user_id}_session"
    clear_user_memories(user_id)
    if setup_personalization:
        _setup_memory(user_id)
    memories = retrieve_context_memories(request, user_id=user_id, k=4)
    manager = CalorieChefManager(
        nutrition_server,
        force_timeout_specialist=force_timeout_specialist,
    )
    result = await manager.run(request, memories)
    trace_evidence = extract_trace(TRACE_PATH, result.trace_id)
    specialist_names = [item.specialist_name for item in result.executions]
    row = {
        "case_id": case_id,
        "request": request,
        "route": result.route,
        "manager_owned_final": result.active_agent == "CalorieChefManager",
        "answer": result.answer,
        "executions": _execution_summary(result),
        "constraints_preserved": not any(
            _food_conflict(result.answer, food) for food in ("peanut", "broccoli")
        ),
        "preference_called": "PreferenceSafetySpecialist" in specialist_names,
        "nutrition_called": "NutritionSpecialist" in specialist_names,
        "usda_search_called": "search_food" in trace_evidence.tool_names,
        "usda_lookup_called": "get_food_nutrition" in trace_evidence.tool_names,
        "calculator_called": "calculate_meal_nutrition" in trace_evidence.tool_names,
        "trace_id": result.trace_id,
        "trace_status": trace_evidence.status,
        "trace_tools": trace_evidence.tool_names,
        "trace_custom_spans": sorted(trace_evidence.custom_spans),
        "limitations": result.limitations,
        "conflict_detected": result.conflict_detected,
        "error": result.error,
    }
    clear_user_memories(user_id)
    return row


async def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_tracing(TRACE_PATH)
    scenarios = []
    async with MCPServerStdio(
        name="CalorieChef USDA Nutrition Server",
        params={"command": sys.executable, "args": [str(NUTRITION_SERVER)]},
        cache_tools_list=True,
        client_session_timeout_seconds=30,
    ) as nutrition_server:
        scenarios.append(
            await _run_case(
                case_id="normal_personalized_meal",
                request="Recommend my final high-protein lunch.",
                nutrition_server=nutrition_server,
                setup_personalization=True,
            )
        )
        scenarios.append(
            await _run_case(
                case_id="nutrition_only",
                request="How many calories and how much protein are in chicken breast?",
                nutrition_server=nutrition_server,
                setup_personalization=False,
            )
        )
        scenarios.append(
            await _run_case(
                case_id="controlled_nutrition_timeout",
                request="Recommend my final high-protein lunch.",
                nutrition_server=nutrition_server,
                setup_personalization=True,
                force_timeout_specialist="NutritionSpecialist",
            )
        )
    RESULT_PATH.write_text(
        json.dumps({"scenarios": scenarios}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in scenarios:
        print(
            f"{row['case_id']}: route={row['route']} "
            f"manager_owned={row['manager_owned_final']} "
            f"executions={[item['specialist'] + ':' + item['status'] for item in row['executions']]}"
        )
    print(f"Results: {RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
