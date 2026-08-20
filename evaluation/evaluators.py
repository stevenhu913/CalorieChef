"""Deterministic turn/conversation evaluators composed with semantic Judge output."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from evaluation.models import (
    ConversationVerdict,
    CriterionResult,
    EvalRunResult,
    JudgeVerdict,
    TurnVerdict,
)


def _criterion(name: str, passed: bool, reason: str, evidence: dict[str, Any]) -> CriterionResult:
    return CriterionResult(
        name=name,
        passed=passed,
        reason=reason,
        evaluator_type="code",
        evidence=evidence,
    )


def _number_present(text: str, expected: float, tolerance: float) -> bool:
    values = [float(value) for value in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)]
    return any(abs(value - expected) <= tolerance for value in values)


def _positive_forbidden_food(text: str, food: str) -> bool:
    lower = text.lower()
    escaped = re.escape(food)
    safe_patterns = (
        rf"\b(?:avoid|without|no)\s+{escaped}s?\b(?:\s+(?:butter|sauce|dressing))?",
        rf"\b{escaped}-free\b",
    )
    for pattern in safe_patterns:
        lower = re.sub(pattern, " ", lower)
    patterns = [
        rf"\b{escaped}\s+(?:butter|sauce|dressing)\b",
        rf"\bwith\s+{escaped}s?\b",
        rf"\badd\s+{escaped}s?\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def _tool_order_passed(actual: list[str], required_order: list[str]) -> bool:
    cursor = -1
    for name in required_order:
        try:
            cursor = actual.index(name, cursor + 1)
        except ValueError:
            return False
    return True


def evaluate_turn(
    case: dict[str, Any],
    run: EvalRunResult,
    judge: JudgeVerdict | None = None,
) -> TurnVerdict:
    """Evaluate one turn with hard checks first and Judge only for semantics."""
    criteria: list[CriterionResult] = []
    counts = Counter(run.tool_names)

    expected = case.get("expected_tools", [])
    forbidden = case.get("forbidden_tools", [])
    minimums = case.get("minimum_tool_counts", {})
    tools_ok = all(name in run.tool_names for name in expected)
    tools_ok = tools_ok and all(name not in run.tool_names for name in forbidden)
    tools_ok = tools_ok and all(counts[name] >= count for name, count in minimums.items())
    if expected or forbidden or minimums:
        criteria.append(
            _criterion(
                "tool_correct",
                tools_ok,
                "Required/forbidden tool checks match the trace." if tools_ok else "Tool trace does not match requirements.",
                {"actual": run.tool_names, "expected": expected, "forbidden": forbidden, "minimums": minimums},
            )
        )

    order = case.get("tool_order", [])
    if order:
        order_ok = _tool_order_passed(run.tool_names, order)
        if "calculate_meal_nutrition" in order and "calculate_meal_nutrition" in run.tool_names:
            calc_index = run.tool_names.index("calculate_meal_nutrition")
            order_ok = order_ok and all(
                index < calc_index
                for index, name in enumerate(run.tool_names)
                if name == "get_food_nutrition"
            )
        criteria.append(_criterion("tool_order_correct", order_ok, "Tool dependency order checked.", {"actual": run.tool_names, "required": order}))

    if "expected_numeric" in case:
        expected_value = float(case["expected_numeric"])
        tolerance = float(case.get("numeric_tolerance", 0.1))
        numeric_ok = _number_present(run.answer, expected_value, tolerance)
        criteria.append(_criterion("numerically_grounded", numeric_ok, "Expected numeric value appears within tolerance." if numeric_ok else "Expected numeric value is missing or wrong.", {"expected": expected_value, "tolerance": tolerance}))

    grounding = case.get("grounding")
    if grounding:
        if grounding == "usda":
            results = [item for item in run.tool_results if item.get("tool") == "get_food_nutrition" and item.get("status") == "ok"]
            grounded = bool(results) and all(
                _number_present(run.answer, float(item[field]), 0.11)
                for item in results[:1]
                for field in ("calories", "protein_g")
                if item.get(field) is not None
            )
            grounded = grounded and "usda" in run.answer.lower()
        elif grounding == "no_exact_without_usda":
            has_lookup = "get_food_nutrition" in run.tool_names
            exact_claim = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:calories|kcal|g(?:rams?)?)\b", run.answer.lower()))
            grounded = has_lookup or not exact_claim
        else:
            grounded = True
        criteria.append(_criterion("grounded", grounded, "Grounding checked against sanitized tool evidence.", {"tool_results": run.tool_results}))

    expected_router = case.get("expected_router_action")
    if expected_router:
        criteria.append(_criterion("write_routing", run.router_action == expected_router, "Deterministic router action checked.", {"actual": run.router_action, "expected": expected_router}))

    expected_memory = case.get("expected_memory")
    if expected_memory:
        matching = [item for item in run.memory_after if item.get("topic") == expected_memory["topic"]]
        memory_ok = bool(matching) and matching[0].get("value") == expected_memory["value"]
        memory_ok = memory_ok and int(matching[0].get("version", 0)) >= int(expected_memory.get("minimum_version", 1))
        if expected_memory.get("old_value"):
            memory_ok = memory_ok and all(item.get("value") != expected_memory["old_value"] for item in run.memory_after)
        criteria.append(_criterion("memory_update", memory_ok, "Current topic value/version checked.", {"matching": matching}))

    if "retrieval_min" in case or "retrieval_max" in case:
        count = len(run.retrieved_memory_ids)
        retrieval_ok = count >= int(case.get("retrieval_min", 0)) and count <= int(case.get("retrieval_max", 10**6))
        criteria.append(_criterion("memory_grounded", retrieval_ok, "Accepted retrieval count checked.", {"accepted_count": count, "memory_ids": run.retrieved_memory_ids}))
        if case.get("require_memory_citations") and run.retrieved_memory_ids:
            cited = all(memory_id in run.answer for memory_id in run.retrieved_memory_ids)
            criteria.append(_criterion("memory_citations", cited, "Retrieved memory IDs must be cited.", {"expected_ids": run.retrieved_memory_ids}))

    for food in case.get("safety_forbidden_foods", []):
        safe = not _positive_forbidden_food(run.answer, food)
        criteria.append(_criterion("safety_preserved", safe, f"No positive recommendation of {food}." if safe else f"Response recommends forbidden {food}.", {"forbidden_food": food}))

    if case.get("clarification_required"):
        clarification = "?" in run.answer or "？" in run.answer or any(word in run.answer.lower() for word in ("please", "could you", "what", "which"))
        criteria.append(_criterion("clarification_present", clarification, "A clarification question is present." if clarification else "No useful clarification was asked.", {}))

    required_spans = case.get("required_trace_spans", [])
    if required_spans:
        actual_spans = set(run.trace.custom_spans) | set(run.trace.tool_names)
        span_ok = all(name in actual_spans for name in required_spans)
        criteria.append(_criterion("trace_evidence_present", span_ok, "Required trace spans checked.", {"required": required_spans, "actual": sorted(actual_spans)}))

    hard_pass = all(item.passed for item in criteria)
    semantic_required = bool(case.get("semantic_rubric"))
    judge_available = judge is not None
    disagreement = judge_available and judge.passed != hard_pass
    if semantic_required:
        criteria.append(
            CriterionResult(
                name="semantic_quality",
                passed=judge.passed if judge else False,
                reason=judge.reason if judge else "Judge unavailable; deterministic results preserved.",
                evaluator_type="judge" if judge else "human_required",
                evidence={"score": judge.score if judge else None},
            )
        )

    safety_failed = any(item.name == "safety_preserved" and not item.passed for item in criteria)
    gray = bool(judge and 4 <= judge.score <= 7)
    human = safety_failed or disagreement or (semantic_required and not judge_available) or gray
    semantic_pass = not semantic_required or bool(judge and judge.passed)
    passed = hard_pass and semantic_pass
    failed = [item.name for item in criteria if not item.passed]
    score = round(sum(item.passed for item in criteria) / max(1, len(criteria)) * 10)
    return TurnVerdict(
        case_id=case["case_id"],
        passed=passed,
        score=score,
        criteria=criteria,
        failed_criteria=failed,
        reason="All required criteria passed." if passed else "Failed: " + ", ".join(failed),
        requires_human_review=human,
        trace_id=run.trace.trace_id,
        judge_available=judge_available,
        code_judge_disagreement=disagreement,
    )


def evaluate_conversation(
    case: dict[str, Any],
    history: list[dict[str, Any]],
    final_run: EvalRunResult,
    judge: JudgeVerdict | None = None,
) -> ConversationVerdict:
    """Evaluate final goal completion and cross-turn constraints."""
    answer = final_run.answer
    criteria: list[CriterionResult] = []
    goal_completed = bool(answer.strip()) and any(word in answer.lower() for word in case.get("goal_signals", []))
    criteria.append(_criterion("goal_completed", goal_completed, "Final answer contains a goal artifact." if goal_completed else "Final artifact is missing.", {"signals": case.get("goal_signals", [])}))

    constraints_ok = True
    for food in case.get("safety_forbidden_foods", []):
        constraints_ok = constraints_ok and not _positive_forbidden_food(answer, food)
    current = case.get("current_value")
    superseded = case.get("superseded_value")
    if current:
        constraints_ok = constraints_ok and current in answer
    if superseded:
        constraints_ok = constraints_ok and superseded not in answer
    criteria.append(_criterion("constraints_preserved", constraints_ok, "Cross-turn constraints and current values checked.", {"current": current, "superseded": superseded}))

    final_complete = all(term.lower() in answer.lower() for term in case.get("required_final_terms", []))
    criteria.append(_criterion("final_artifact_complete", final_complete, "Required final artifact terms checked.", {"required_terms": case.get("required_final_terms", [])}))

    expected_tools = case.get("expected_tools", [])
    grounded = all(tool in final_run.tool_names for tool in expected_tools)
    criteria.append(_criterion("grounded", grounded, "Final artifact tool evidence checked.", {"actual": final_run.tool_names, "expected": expected_tools}))

    hard_pass = all(item.passed for item in criteria)
    semantic_required = bool(case.get("semantic_rubric"))
    judge_available = judge is not None
    disagreement = judge_available and judge.passed != hard_pass
    if semantic_required:
        criteria.append(CriterionResult(
            name="semantic_quality",
            passed=judge.passed if judge else False,
            reason=judge.reason if judge else "Judge unavailable; human review required.",
            evaluator_type="judge" if judge else "human_required",
            evidence={"score": judge.score if judge else None, "history_turns": len(history)},
        ))
    gray = bool(judge and 4 <= judge.score <= 7)
    human = disagreement or (semantic_required and not judge_available) or gray or not constraints_ok
    semantic_pass = not semantic_required or bool(judge and judge.passed)
    passed = hard_pass and semantic_pass
    failed = [item.name for item in criteria if not item.passed]
    score = round(sum(item.passed for item in criteria) / max(1, len(criteria)) * 10)
    return ConversationVerdict(
        case_id=case["case_id"],
        passed=passed,
        score=score,
        goal_completed=goal_completed,
        constraints_preserved=constraints_ok,
        final_artifact_complete=final_complete,
        criteria=criteria,
        failed_criteria=failed,
        reason="Conversation goal completed." if passed else "Failed: " + ", ".join(failed),
        requires_human_review=human,
        judge_available=judge_available,
        code_judge_disagreement=disagreement,
    )
