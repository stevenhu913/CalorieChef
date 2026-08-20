"""Run one controlled Single-Agent versus Multi-Agent comparison."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import _local  # noqa: F401
from agents import Runner, custom_span, trace
from agents.mcp import MCPServerStdio

from agent import create_agent as create_single_agent
from long_term_memory import (
    clear_user_memories,
    format_evidence,
    retrieve_context_memories,
    upsert_memory,
)
from memory import RUN_CONFIG
from evaluation.trace_evidence import extract_trace
from experiments.multi_agent.models import ArchitectureMetrics
from experiments.multi_agent.orchestrator import CalorieChefManager
from observability import configure_local_tracing


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
TRACE_PATH = ARTIFACT_DIR / "architecture_comparison_traces.jsonl"
RESULT_PATH = ARTIFACT_DIR / "architecture_comparison.json"
SUMMARY_PATH = ARTIFACT_DIR / "architecture_comparison.md"
NUTRITION_SERVER = PROJECT_ROOT / "nutrition_mcp_server.py"
REQUEST = "Recommend my final high-protein lunch."


def _setup(user_id: str) -> None:
    for index, (topic, value, kind) in enumerate(
        (
            ("allergy:peanut", "peanut", "allergy"),
            ("disliked_ingredient:broccoli", "broccoli", "disliked_ingredient"),
            ("lunch_calorie_target", "600 calories", "calorie_target"),
            ("lunch_style", "high-protein lunches", "meal_preference"),
        ),
        start=1,
    ):
        upsert_memory(
            topic=topic,
            value=value,
            kind=kind,
            source_turn=f"architecture-comparison:{index}",
            user_id=user_id,
        )


def _prohibited_recommendation(answer: str, food: str) -> bool:
    lower = answer.lower()
    escaped = re.escape(food)
    for safe in (
        rf"\b(?:avoid|avoiding|without|no)\s+{escaped}s?\b",
        rf"\b{escaped}-free\b",
        rf"\b{escaped}\s+(?:allergy|dislike)\b",
        rf"\b(?:allergic to|dislikes?)\s+{escaped}\b",
    ):
        lower = re.sub(safe, " ", lower)
    return bool(re.search(rf"\b{escaped}s?\b", lower))


def _metrics(
    architecture: str,
    answer: str,
    evidence,
    latency_ms: float,
    *,
    limitations: list[str] | None = None,
) -> ArchitectureMetrics:
    exact_claim = bool(
        re.search(
            r"\b(?:contains?|provides?|totals?|approximately|about)\s+\d+(?:\.\d+)?\s*(?:calories|kcal|g)\b",
            answer.lower(),
        )
    )
    tool_names = evidence.tool_names
    verified = "search_food" in tool_names and "get_food_nutrition" in tool_names
    calculated = "calculate_meal_nutrition" in tool_names
    raw_tool_markup = "<tool_call>" in answer or '"name": "nutrition_specialist"' in answer
    final_meal = not raw_tool_markup and any(
        marker in answer.lower()
        for marker in ("ingredients:", "portions:", "serving_g", "meal:", "### meal")
    )
    limits = limitations or []
    limitation_visible = not limits or any(
        limitation.lower() in answer.lower() for limitation in limits
    )
    return ArchitectureMetrics(
        architecture=architecture,
        hard_constraints_preserved=not any(
            _prohibited_recommendation(answer, food) for food in ("peanut", "broccoli")
        ),
        required_tool_evidence_present=verified and calculated,
        final_meal_artifact_present=final_meal,
        unsupported_exact_claims=exact_claim and not verified,
        partial_failure_honest=limitation_visible,
        observed_latency_ms=round(latency_ms, 2),
        generation_count=evidence.generation_count,
        tool_call_count=len(tool_names),
        token_usage=evidence.token_usage,
        trace_clarity=sorted(evidence.custom_spans),
        trace_id=evidence.trace_id,
        details={"tool_names": tool_names, "answer": answer},
    )


async def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_tracing(TRACE_PATH)
    user_id = f"architecture_compare_{uuid.uuid4().hex[:10]}"
    os.environ["CALORIECHEF_USER_ID"] = user_id
    clear_user_memories(user_id)
    _setup(user_id)
    memories = retrieve_context_memories(REQUEST, user_id=user_id, k=4)

    async with MCPServerStdio(
        name="CalorieChef USDA Nutrition Server",
        params={"command": sys.executable, "args": [str(NUTRITION_SERVER)]},
        cache_tools_list=True,
        client_session_timeout_seconds=30,
    ) as nutrition_server:
        single_started = time.perf_counter()
        with trace(
            "caloriechef_single_agent_comparison",
            metadata={"architecture": "single_agent", "raw_user_text_recorded": False},
        ) as single_trace:
            with custom_span("architecture", data={"architecture": "single_agent"}):
                pass
            single_agent = create_single_agent(nutrition_server, format_evidence(memories))
            single_result = await Runner.run(
                single_agent,
                REQUEST,
                run_config=RUN_CONFIG,
                max_turns=12,
            )
            single_answer = str(single_result.final_output)
        single_latency = (time.perf_counter() - single_started) * 1000
        single_evidence = extract_trace(TRACE_PATH, single_trace.trace_id)

        manager = CalorieChefManager(nutrition_server)
        multi_result = await manager.run(REQUEST, memories)
        multi_evidence = extract_trace(TRACE_PATH, multi_result.trace_id)

    single_metrics = _metrics(
        "single_agent",
        single_answer,
        single_evidence,
        single_latency,
    )
    multi_metrics = _metrics(
        "multi_agent",
        multi_result.answer,
        multi_evidence,
        multi_result.observed_latency_ms,
        limitations=multi_result.limitations,
    )
    quality_fields = (
        "hard_constraints_preserved",
        "required_tool_evidence_present",
        "final_meal_artifact_present",
        "partial_failure_honest",
    )
    single_quality = sum(bool(getattr(single_metrics, field)) for field in quality_fields) - int(
        single_metrics.unsupported_exact_claims
    )
    multi_quality = sum(bool(getattr(multi_metrics, field)) for field in quality_fields) - int(
        multi_metrics.unsupported_exact_claims
    )
    conclusion = (
        "retain_multi_agent"
        if multi_quality > single_quality
        else "experimental_quality_gain_not_proven"
    )
    payload: dict[str, Any] = {
        "request": REQUEST,
        "single_agent": single_metrics.model_dump(),
        "multi_agent": multi_metrics.model_dump(),
        "quality_check_score": {"single_agent": single_quality, "multi_agent": multi_quality},
        "conclusion": conclusion,
        "warning": "One observed run is not a statistical benchmark or P50/P95 measurement.",
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                "# Single-Agent vs Multi-Agent Comparison",
                "",
                f"- Conclusion: {conclusion}",
                f"- Single observed latency: {single_metrics.observed_latency_ms:.2f} ms",
                f"- Multi observed latency: {multi_metrics.observed_latency_ms:.2f} ms",
                f"- Single quality checks: {single_quality}",
                f"- Multi quality checks: {multi_quality}",
                f"- Single generations/tools: {single_metrics.generation_count}/{single_metrics.tool_call_count}",
                f"- Multi generations/tools: {multi_metrics.generation_count}/{multi_metrics.tool_call_count}",
                "- This is one observed run, not a statistical latency or quality claim.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    clear_user_memories(user_id)
    print(json.dumps(payload, indent=2))
    print(f"Results: {RESULT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
