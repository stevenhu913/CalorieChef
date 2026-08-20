"""Deterministic meal-presentation and SDK evidence tests."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent_core import (
    MEAL_FORMAT_FAILURE,
    MEAL_ORCHESTRATION_FAILURE,
    _authoritative_meal_output,
    _meal_tool_evidence,
)
from presentation import format_meal_plan


def regression_result():
    return {
        "status": "ok",
        "target_calories": 1000.0,
        "actual_calories": 994.2,
        "calorie_difference": -5.8,
        "absolute_calorie_gap": 5.8,
        "calorie_gap_percentage": 0.58,
        "tolerance_calories": 50.0,
        "preferred_range": {"min_serving_g": 50, "max_serving_g": 300},
        "expanded_range": {"min_serving_g": 50, "max_serving_g": 600},
        "preferred_target_met": False,
        "final_target_met": True,
        "target_met": True,
        "target_status": "met_expanded",
        "search_range_used": "expanded",
        "used_expanded_range": True,
        "adjustment_direction": "none",
        "boundary_hits": [
            {"food_name": "Chicken breast", "boundary": "preferred_max", "serving_g": 300},
        ],
        "feasibility_message": (
            "The 1,000 kcal target is achievable within the expanded practical "
            "range. The final combination is 994.2 kcal, 5.8 kcal below target. "
            "Rice pilaf requires a serving above the preferred 300 g range."
        ),
        "items": [
            {
                "food_name": "Chicken breast",
                "serving_g": 300,
                "calories": 336.6,
                "protein_g": 67.5,
                "carbs_g": 0.0,
                "fat_g": 5.8,
                "at_preferred_min": False,
                "at_preferred_max": True,
                "at_expanded_max": False,
            },
            {
                "food_name": "Rice pilaf",
                "serving_g": 480,
                "calories": 657.6,
                "protein_g": 15.8,
                "carbs_g": 115.7,
                "fat_g": 14.6,
                "at_preferred_min": False,
                "at_preferred_max": False,
                "at_expanded_max": False,
            },
        ],
        "totals": {
            "calories": 994.2,
            "protein_g": 83.3,
            "carbs_g": 115.7,
            "fat_g": 20.4,
        },
    }


def preferred_result():
    result = regression_result()
    result.update(
        {
            "target_calories": 200.0,
            "actual_calories": 200.0,
            "calorie_difference": 0.0,
            "absolute_calorie_gap": 0.0,
            "preferred_target_met": True,
            "final_target_met": True,
            "target_status": "met_preferred",
            "search_range_used": "preferred",
            "used_expanded_range": False,
            "feasibility_message": (
                "Meal is within the target tolerance using preferred serving sizes. "
                "The final combination is 200 kcal, exactly on target."
            ),
            "items": [
                {
                    "food_name": "Preferred food",
                    "serving_g": 200,
                    "calories": 200.0,
                    "protein_g": 20.0,
                    "carbs_g": 10.0,
                    "fat_g": 4.0,
                }
            ],
            "totals": {
                "calories": 200.0,
                "protein_g": 20.0,
                "carbs_g": 10.0,
                "fat_g": 4.0,
            },
        }
    )
    return result


class PresentationTests(unittest.TestCase):
    def test_formatter_uses_only_structured_meal_values(self):
        text = format_meal_plan(regression_result())
        for expected in (
            "### Meal plan",
            "### Ingredients",
            "### Totals",
            "### Target check",
            "**Target:** 1,000 kcal",
            "**Calculated:** 994.2 kcal",
            "**Gap:** 5.8 kcal below target",
            "**Chicken breast** — 300 g",
            "336.6 kcal",
            "**Rice pilaf** — 480 g · expanded portion",
            "**Protein:** 83.3 g",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("999", text)

    def test_formatter_explains_expanded_success(self):
        text = format_meal_plan(regression_result())
        lower = text.lower()
        self.assertIn("expanded portion", lower)
        self.assertIn("expanded serving range was required", lower)
        self.assertNotIn("cannot be reached", lower)
        self.assertNotIn("very close", lower)
        self.assertNotIn("slight shortfall", lower)
        self.assertNotIn("cutting back", lower)

    def test_preferred_success_does_not_mention_expansion(self):
        text = format_meal_plan(preferred_result()).lower()
        self.assertIn("target met within tolerance", text)
        self.assertNotIn("expanded", text)

    def test_unreachable_language_requires_expanded_failure(self):
        result = regression_result()
        result.update(
            {
                "target_calories": 2000.0,
                "actual_calories": 1495.2,
                "calorie_difference": -504.8,
                "absolute_calorie_gap": 504.8,
                "final_target_met": False,
                "target_met": False,
                "target_status": "above_expanded_maximum",
                "adjustment_direction": "increase",
                "feasibility_message": (
                    "The requested target cannot be reached with these foods "
                    "within the practical 50–600 g serving range. The closest "
                    "combination is 1,495.2 kcal, 504.8 kcal below target."
                ),
            }
        )
        text = format_meal_plan(result).lower()
        self.assertIn("cannot be reached", text)
        self.assertIn("50–600 g", text)
        self.assertNotIn("reduce", text)

    def test_sdk_items_are_paired_by_call_id(self):
        payload = regression_result()
        items = [
            SimpleNamespace(
                type="tool_call_item",
                tool_name="calculate_meal_nutrition",
                call_id="call_meal",
            ),
            SimpleNamespace(
                type="tool_call_output_item",
                call_id="call_meal",
                output=payload,
            ),
        ]
        call_count, successful_count, result, malformed = _meal_tool_evidence(items)
        self.assertEqual(call_count, 1)
        self.assertEqual(successful_count, 1)
        self.assertEqual(result, payload)
        self.assertFalse(malformed)

    def test_malformed_meal_output_cannot_use_model_fallback(self):
        items = [
            SimpleNamespace(
                type="tool_call_item",
                tool_name="calculate_meal_nutrition",
                call_id="call_meal",
            ),
            SimpleNamespace(
                type="tool_call_output_item",
                call_id="call_meal",
                output="not structured JSON",
            ),
        ]
        call_count, successful_count, result, malformed = _meal_tool_evidence(items)
        self.assertEqual(call_count, 1)
        self.assertEqual(successful_count, 0)
        self.assertIsNone(result)
        self.assertTrue(malformed)
        self.assertIn("could not be formatted safely", MEAL_FORMAT_FAILURE)

    def test_invalid_serving_range_is_rejected(self):
        result = regression_result()
        result["preferred_range"] = "50 to 300"
        with self.assertRaises(ValueError):
            format_meal_plan(result)

    def test_successful_meal_result_discards_verbose_model_refinements(self):
        items = [
            SimpleNamespace(
                type="tool_call_item",
                tool_name="search_food",
                call_id="search_chicken",
            ),
            SimpleNamespace(
                type="tool_call_item",
                tool_name="get_food_nutrition",
                call_id="nutrition_chicken",
            ),
            SimpleNamespace(
                type="tool_call_item",
                tool_name="get_food_nutrition",
                call_id="nutrition_rice",
            ),
            SimpleNamespace(
                type="tool_call_item",
                tool_name="calculate_meal_nutrition",
                call_id="meal_once",
            ),
            SimpleNamespace(
                type="tool_call_output_item",
                call_id="meal_once",
                output=regression_result(),
            ),
        ]
        wrong_model_output = (
            "Let's refine. Updated servings: 999 g chicken. "
            "Let's refine further. Final Meal Totals: 9,999 kcal."
        )
        output, metadata = _authoritative_meal_output(wrong_model_output, items)
        self.assertNotIn(wrong_model_output, output)
        self.assertNotIn("let's refine", output.lower())
        self.assertNotIn("999 g chicken", output.lower())
        self.assertEqual(output.count("### Meal plan"), 1)
        self.assertEqual(output.count("### Totals"), 1)
        self.assertEqual(output.count("994.2 kcal"), 2)
        self.assertTrue(metadata["model_final_output_discarded"])
        self.assertEqual(metadata["meal_tool_call_count"], 1)

    def test_duplicate_meal_calls_are_an_orchestration_violation(self):
        payload = regression_result()
        items = []
        for call_id in ("meal_one", "meal_two"):
            items.extend(
                [
                    SimpleNamespace(
                        type="tool_call_item",
                        tool_name="calculate_meal_nutrition",
                        call_id=call_id,
                    ),
                    SimpleNamespace(
                        type="tool_call_output_item",
                        call_id=call_id,
                        output=payload,
                    ),
                ]
            )
        output, metadata = _authoritative_meal_output("multiple plans", items)
        self.assertEqual(output, MEAL_ORCHESTRATION_FAILURE)
        self.assertTrue(metadata["orchestration_violation"])
        self.assertEqual(metadata["meal_tool_call_count"], 2)
        self.assertEqual(metadata["successful_meal_tool_call_count"], 2)

    def test_normal_meal_response_is_compact(self):
        meaningful_lines = [
            line for line in format_meal_plan(regression_result()).splitlines() if line
        ]
        self.assertGreaterEqual(len(meaningful_lines), 10)
        self.assertLessEqual(len(meaningful_lines), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
