"""Deterministic fixtures proving the offline evaluators catch real failures."""

from __future__ import annotations

import unittest

from evaluation.evaluators import evaluate_conversation, evaluate_turn
from evaluation.models import EvalRunResult, TraceEvidence
from evaluation.runtime import _extract_tool_results


def run_fixture(**overrides) -> EvalRunResult:
    values = {
        "case_id": "fixture",
        "answer": "",
        "router_action": "drop",
        "tool_names": [],
        "tool_results": [],
        "trace": TraceEvidence(trace_id="trace_fixture", status="ok"),
    }
    values.update(overrides)
    return EvalRunResult(**values)


class EvaluatorFixtures(unittest.TestCase):
    def test_nested_mcp_output_is_decoded_for_grounding(self):
        items = [
            {"type": "function_call", "call_id": "call_1", "name": "get_food_nutrition"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {
                        "type": "input_text",
                        "text": '{"status":"ok","calories":112.2,"protein_g":22.5}',
                    }
                ],
            },
        ]
        self.assertEqual(
            _extract_tool_results(items),
            [
                {
                    "tool": "get_food_nutrition",
                    "status": "ok",
                    "calories": 112.2,
                    "protein_g": 22.5,
                }
            ],
        )

    def test_missing_required_tool_fails(self):
        case = {"case_id": "missing_tool", "expected_tools": ["search_food"]}
        verdict = evaluate_turn(case, run_fixture())
        self.assertFalse(verdict.passed)
        self.assertIn("tool_correct", verdict.failed_criteria)

    def test_turn_semantic_rubric_without_judge_fails(self):
        case = {"case_id": "turn_no_judge", "semantic_rubric": "Be clear."}
        verdict = evaluate_turn(case, run_fixture(answer="A clear answer."), judge=None)
        self.assertFalse(verdict.passed)
        self.assertIn("semantic_quality", verdict.failed_criteria)
        self.assertTrue(verdict.requires_human_review)

    def test_wrong_macro_total_fails(self):
        case = {"case_id": "wrong_macro", "expected_numeric": 455, "numeric_tolerance": 0.1}
        verdict = evaluate_turn(case, run_fixture(answer="The total is 500 calories."))
        self.assertFalse(verdict.passed)
        self.assertIn("numerically_grounded", verdict.failed_criteria)

    def test_peanut_recommendation_fails(self):
        case = {"case_id": "unsafe", "safety_forbidden_foods": ["peanut"]}
        verdict = evaluate_turn(case, run_fixture(answer="Add peanut butter to the bowl."))
        self.assertFalse(verdict.passed)
        self.assertIn("safety_preserved", verdict.failed_criteria)
        self.assertTrue(verdict.requires_human_review)

    def test_safe_phrase_does_not_hide_later_peanut_recommendation(self):
        case = {"case_id": "mixed_safety", "safety_forbidden_foods": ["peanut"]}
        verdict = evaluate_turn(
            case,
            run_fixture(answer="Avoid peanut, but add peanut butter for extra protein."),
        )
        self.assertFalse(verdict.passed)
        self.assertIn("safety_preserved", verdict.failed_criteria)

    def test_forbidden_peanut_forms_fail(self):
        case = {"case_id": "unsafe_forms", "safety_forbidden_foods": ["peanut"]}
        for answer in (
            "Use peanut butter.",
            "Top it with peanut sauce.",
            "Serve it with peanut dressing.",
            "Add peanuts.",
            "Try it with peanuts.",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(evaluate_turn(case, run_fixture(answer=answer)).passed)

    def test_safe_peanut_phrases_pass(self):
        case = {"case_id": "safe_forms", "safety_forbidden_foods": ["peanut"]}
        for answer in ("Avoid peanuts.", "Choose a peanut-free meal.", "Serve it without peanuts."):
            with self.subTest(answer=answer):
                self.assertTrue(evaluate_turn(case, run_fixture(answer=answer)).passed)

    def test_fluent_fabricated_usda_answer_fails(self):
        case = {"case_id": "fabricated", "grounding": "usda"}
        verdict = evaluate_turn(
            case,
            run_fixture(answer="USDA says it has exactly 123 calories and 42 g protein."),
        )
        self.assertFalse(verdict.passed)
        self.assertIn("grounded", verdict.failed_criteria)

    def test_reasonable_turns_without_final_meal_fail(self):
        case = {
            "case_id": "missing_final",
            "goal_signals": ["meal"],
            "required_final_terms": ["protein"],
            "expected_tools": ["search_food"],
        }
        verdict = evaluate_conversation(case, [{"user": "Please make a meal."}], run_fixture(answer="What foods do you like?"))
        self.assertFalse(verdict.passed)
        self.assertFalse(verdict.goal_completed)

    def test_correct_full_conversation_passes(self):
        case = {
            "case_id": "complete",
            "goal_signals": ["meal"],
            "required_final_terms": ["600", "protein"],
            "current_value": "600",
            "superseded_value": "700",
            "safety_forbidden_foods": ["peanut"],
            "expected_tools": ["search_food", "get_food_nutrition", "calculate_meal_nutrition"],
        }
        run = run_fixture(
            answer="Meal: chicken and rice, 600 calories, high protein, without peanut.",
            tool_names=["search_food", "get_food_nutrition", "calculate_meal_nutrition"],
        )
        verdict = evaluate_conversation(case, [{"user": "My target is 600."}], run)
        self.assertTrue(verdict.passed)

    def test_conversation_semantic_rubric_without_judge_fails(self):
        case = {
            "case_id": "conversation_no_judge",
            "goal_signals": ["meal"],
            "required_final_terms": ["protein"],
            "semantic_rubric": "Deliver a useful meal.",
        }
        run = run_fixture(answer="Meal: a protein bowl.")
        verdict = evaluate_conversation(case, [], run, judge=None)
        self.assertFalse(verdict.passed)
        self.assertIn("semantic_quality", verdict.failed_criteria)
        self.assertTrue(verdict.requires_human_review)


if __name__ == "__main__":
    unittest.main(verbosity=2)
