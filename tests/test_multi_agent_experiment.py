"""Deterministic tests for experimental contracts, boundaries, and failure."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agents import ModelBehaviorError, custom_span, function_tool, trace
from pydantic import ValidationError

from evaluation.trace_evidence import extract_trace
from experiments.multi_agent.models import (
    ExpertResult,
    ManagerRunResult,
    NutritionResult,
    PreferenceSafetyResult,
    SpecialistExecution,
)
from experiments.multi_agent.orchestrator import (
    add_missing_specialist_failures,
    bounded_expert_call,
    bounded_structured_call,
    create_manager_agent,
    manager_tool_names,
    merge_specialist_results,
    nutrition_failure,
    route_request,
)
from observability import configure_local_tracing
from experiments.multi_agent.specialists import (
    create_nutrition_specialist,
    create_preference_safety_specialist,
)


def execution(name: str, result) -> SpecialistExecution:
    return SpecialistExecution(
        specialist_name=name,
        status=result.status,
        latency_ms=1.0,
        confidence=result.confidence,
        finding_count=len(result.findings),
        limitation_count=len(result.limitations),
        result=result,
    )


class MultiAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_expert_contract_validation(self):
        for status in ("ok", "partial", "failed"):
            with self.subTest(status=status):
                self.assertEqual(
                    ExpertResult(status=status, findings=[], confidence=0.5).status,
                    status,
                )
        with self.assertRaises(ValidationError):
            ExpertResult(status="unknown", findings=[], confidence=0.5)
        with self.assertRaises(ValidationError):
            ExpertResult(status="ok", findings=[], confidence=1.1)

    def test_tool_and_context_boundaries(self):
        preference = create_preference_safety_specialist()
        fake_mcp = object()
        nutrition = create_nutrition_specialist(fake_mcp)
        self.assertEqual(preference.tools, [])
        self.assertEqual(preference.mcp_servers, [])
        self.assertEqual(
            {tool.name for tool in nutrition.tools},
            {"calculate_macro_calories", "calculate_meal_nutrition"},
        )
        self.assertEqual(nutrition.mcp_servers, [fake_mcp])
        self.assertNotIn("memory", " ".join(tool.name for tool in nutrition.tools))

        async def preference_tool(value: str) -> str:
            return value

        async def nutrition_tool(value: str) -> str:
            return value

        preference_wrapper = function_tool(
            preference_tool,
            name_override="preference_safety_specialist",
            description_override="Preference specialist.",
        )
        nutrition_wrapper = function_tool(
            nutrition_tool,
            name_override="nutrition_specialist",
            description_override="Nutrition specialist.",
        )
        manager = create_manager_agent(
            "personalized_meal",
            [preference_wrapper, nutrition_wrapper],
        )
        self.assertEqual(
            {tool.name for tool in manager.tools},
            {"preference_safety_specialist", "nutrition_specialist"},
        )

    def test_manager_owns_final_response(self):
        result = ManagerRunResult(
            answer="Manager final answer",
            route="nutrition_only",
            observed_latency_ms=1.0,
        )
        self.assertEqual(result.active_agent, "CalorieChefManager")

    def test_status_first_merge_discards_failed_findings(self):
        failed = NutritionResult(
            status="failed",
            findings=["Invented 999 calorie claim"],
            confidence=0.0,
            limitations=["USDA failed."],
        )
        answer, limitations, _ = merge_specialist_results(
            "Invented 999 calorie claim",
            [execution("NutritionSpecialist", failed)],
        )
        self.assertNotIn("999", answer)
        self.assertIn("USDA failed.", limitations)

    def test_raw_tool_markup_is_not_user_facing(self):
        preference = PreferenceSafetyResult(
            status="partial",
            findings=["Target is 600 calories"],
            confidence=0.5,
            limitations=["Nutrition is missing."],
        )
        answer, _, _ = merge_specialist_results(
            '<tool_call>{"name": "nutrition_specialist"}</tool_call>',
            [execution("PreferenceSafetySpecialist", preference)],
        )
        self.assertNotIn("<tool_call>", answer)

    def test_missing_required_specialist_becomes_failed_contract(self):
        preference = PreferenceSafetyResult(
            status="ok",
            findings=["Target is 600 calories"],
            confidence=1.0,
        )
        executions = [execution("PreferenceSafetySpecialist", preference)]
        missing = add_missing_specialist_failures("personalized_meal", executions)
        self.assertEqual(missing, ["NutritionSpecialist"])
        self.assertEqual(executions[-1].status, "failed")
        self.assertIn("did not return", executions[-1].result.limitations[0])

    def test_partial_limitations_are_visible(self):
        partial = PreferenceSafetyResult(
            status="partial",
            findings=["Peanut allergy"],
            confidence=0.5,
            limitations=["Calorie target is unknown."],
            allergies=["peanut"],
        )
        answer, _, _ = merge_specialist_results(
            "I can preserve the peanut allergy.",
            [execution("PreferenceSafetySpecialist", partial)],
        )
        self.assertIn("Calorie target is unknown.", answer)

    def test_safety_conflict_rejects_nutrition_proposal(self):
        preference = PreferenceSafetyResult(
            status="ok",
            findings=["Peanut allergy"],
            confidence=1.0,
            allergies=["peanut"],
            hard_constraints=["allergy: peanut"],
        )
        nutrition = NutritionResult(
            status="ok",
            findings=["Use peanut sauce"],
            confidence=0.9,
            selected_food_descriptions=["Chicken with peanut sauce"],
        )
        executions = [
            execution("PreferenceSafetySpecialist", preference),
            execution("NutritionSpecialist", nutrition),
        ]
        answer, limitations, conflict = merge_specialist_results(
            "Serve chicken with peanut sauce.",
            executions,
        )
        self.assertTrue(conflict)
        self.assertIn("conflicts", answer)
        self.assertFalse(executions[1].result_used)
        self.assertTrue(any("conflicts" in item for item in limitations))

    def test_safety_constraint_mention_is_not_a_food_proposal(self):
        preference = PreferenceSafetyResult(
            status="ok",
            findings=["Peanut allergy"],
            confidence=1.0,
            allergies=["peanut"],
        )
        nutrition = NutritionResult(
            status="partial",
            findings=["Peanut is in foods_to_avoid."],
            confidence=0.5,
            limitations=["A safe meal was not verified."],
        )
        _, _, conflict = merge_specialist_results(
            "Avoiding peanut; no verified meal is available.",
            [
                execution("PreferenceSafetySpecialist", preference),
                execution("NutritionSpecialist", nutrition),
            ],
        )
        self.assertFalse(conflict)

    def test_empty_partial_nutrition_is_not_used(self):
        preference = PreferenceSafetyResult(
            status="ok",
            findings=["Target is 600 calories"],
            confidence=1.0,
            calorie_target=600,
        )
        nutrition = NutritionResult(
            status="partial",
            findings=[],
            confidence=0.0,
            limitations=["Nutrition Specialist timed out."],
        )
        executions = [
            execution("PreferenceSafetySpecialist", preference),
            execution("NutritionSpecialist", nutrition),
        ]
        answer, _, _ = merge_specialist_results(
            "Serve a verified 600 calorie chicken meal.",
            executions,
        )
        self.assertNotIn("verified 600 calorie chicken meal", answer)
        self.assertFalse(executions[1].result_used)
        self.assertIn("Nutrition Specialist timed out.", answer)

    async def test_timeout_returns_partial_contract_without_crash(self):
        fallback = NutritionResult(
            status="partial",
            findings=[],
            confidence=0.0,
            limitations=["Nutrition Specialist timed out."],
        )

        async def slow_result():
            await asyncio.sleep(0.05)
            return NutritionResult(status="ok", findings=["late"], confidence=1.0)

        result, timed_out = await bounded_expert_call(
            slow_result(),
            timeout_seconds=0.001,
            timeout_result=fallback,
        )
        self.assertTrue(timed_out)
        self.assertEqual(result.status, "partial")

    async def test_structured_call_retries_model_behavior_error_once(self):
        attempts = 0
        expected = NutritionResult(
            status="ok",
            findings=["USDA meal calculated."],
            confidence=1.0,
            selected_food_descriptions=["Food A", "Food B"],
            portion_grams=[100.0, 150.0],
            calories=600.0,
            protein_g=50.0,
            carbs_g=60.0,
            fat_g=17.8,
            source="USDA FoodData Central",
        )

        async def flaky_result():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelBehaviorError("Invalid JSON provided")
            return expected

        result, timed_out, retry_count = await bounded_structured_call(
            flaky_result,
            timeout_seconds=1.0,
            timeout_result=nutrition_failure("partial", "Timed out."),
        )
        self.assertEqual(result, expected)
        self.assertFalse(timed_out)
        self.assertEqual(retry_count, 1)
        self.assertEqual(attempts, 2)

    async def test_structured_call_does_not_retry_unrelated_errors(self):
        attempts = 0

        async def broken_result():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("unrelated")

        with self.assertRaises(RuntimeError):
            await bounded_structured_call(
                broken_result,
                timeout_seconds=1.0,
                timeout_result=nutrition_failure("partial", "Timed out."),
            )
        self.assertEqual(attempts, 1)

    def test_partial_failure_preserves_usable_findings(self):
        preference = PreferenceSafetyResult(
            status="ok",
            findings=["Target is 600 calories"],
            confidence=1.0,
            calorie_target=600,
        )
        nutrition = NutritionResult(
            status="failed",
            findings=["Unsupported nutrition"],
            confidence=0.0,
            limitations=["USDA unavailable."],
        )
        answer, limitations, _ = merge_specialist_results(
            "Unsupported nutrition",
            [
                execution("PreferenceSafetySpecialist", preference),
                execution("NutritionSpecialist", nutrition),
            ],
        )
        self.assertIn("Target is 600 calories", answer)
        self.assertNotIn("Unsupported nutrition", answer)
        self.assertIn("USDA unavailable.", limitations)

    def test_no_unnecessary_specialist(self):
        self.assertEqual(
            route_request("How many calories are in chicken breast?"),
            "nutrition_only",
        )
        self.assertEqual(manager_tool_names("nutrition_only"), ["nutrition_specialist"])
        self.assertEqual(
            route_request("What preferences do you remember?"),
            "preference_only",
        )
        self.assertEqual(
            manager_tool_names("preference_only"),
            ["preference_safety_specialist"],
        )

    def test_trace_identifies_manager_specialist_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            configure_local_tracing(path)
            with trace("multi_agent_experiment_test") as workflow:
                with custom_span("manager_request", data={"active_agent": "CalorieChefManager"}):
                    pass
                with custom_span(
                    "specialist_timeout",
                    data={
                        "specialist_name": "NutritionSpecialist",
                        "status": "partial",
                        "timeout": True,
                        "fallback_used": True,
                    },
                ):
                    pass
                with custom_span("manager_merge", data={"fallback_used": True}):
                    pass
            evidence = extract_trace(path, workflow.trace_id)
            self.assertIn("manager_request", evidence.custom_spans)
            self.assertIn("specialist_timeout", evidence.custom_spans)
            self.assertTrue(evidence.custom_spans["specialist_timeout"]["fallback_used"])
            self.assertIn("manager_merge", evidence.custom_spans)


if __name__ == "__main__":
    unittest.main(verbosity=2)
