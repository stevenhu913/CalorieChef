"""Deterministic tests for bounded nutrition calculations."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from tools import calculate_meal_nutrition


def calculate(**values):
    """Invoke the exported Agents SDK function tool without a model call."""
    context = SimpleNamespace(tool_name=calculate_meal_nutrition.name)
    return asyncio.run(
        calculate_meal_nutrition.on_invoke_tool(context, json.dumps(values))
    )


def meal_input(
    *,
    food_names,
    target_calories,
    calories_per_100g,
    protein_per_100g=None,
):
    count = len(food_names)
    return {
        "food_names": food_names,
        "target_calories": target_calories,
        "calories_per_100g": calories_per_100g,
        "protein_per_100g": protein_per_100g or [10.0] * count,
        "carbs_per_100g": [5.0] * count,
        "fat_per_100g": [2.0] * count,
    }


class MealToolTests(unittest.TestCase):
    def test_1000_kcal_regression_uses_expanded_range(self):
        result = calculate(
            **meal_input(
                food_names=["Chicken breast", "Rice pilaf"],
                target_calories=1000,
                calories_per_100g=[112.2, 137.0],
                protein_per_100g=[22.5, 3.3],
            )
        )

        self.assertFalse(result["preferred_target_met"])
        self.assertEqual(result["preferred_actual_calories"], 747.6)
        self.assertEqual(result["preferred_calorie_difference"], -252.4)
        self.assertEqual(result["preferred_maximum_possible_calories"], 747.6)
        self.assertTrue(result["expanded_search_attempted"])
        self.assertTrue(result["expanded_result_improved_gap"])
        self.assertTrue(result["used_expanded_range"])
        self.assertTrue(result["final_target_met"])
        self.assertTrue(result["target_met"])
        self.assertEqual(result["target_status"], "met_expanded")
        self.assertEqual(result["search_range_used"], "expanded")
        self.assertLessEqual(
            result["absolute_calorie_gap"],
            result["tolerance_calories"],
        )
        self.assertEqual(result["tolerance_calories"], 50.0)
        servings = [item["serving_g"] for item in result["items"]]
        self.assertTrue(any(grams > 300 for grams in servings))
        self.assertTrue(all(grams <= 600 for grams in servings))
        self.assertEqual(result["adjustment_direction"], "none")
        message = result["feasibility_message"].lower()
        self.assertIn("expanded practical range", message)
        self.assertIn("above the preferred 300 g range", message)
        self.assertNotIn("cannot be reached", message)
        self.assertNotIn("very close", message)
        self.assertNotIn("slight shortfall", message)

    def test_target_within_tolerance_is_met(self):
        result = calculate(
            **meal_input(
                food_names=["Food"],
                target_calories=200,
                calories_per_100g=[100.0],
            )
        )
        self.assertTrue(result["preferred_target_met"])
        self.assertFalse(result["expanded_search_attempted"])
        self.assertFalse(result["used_expanded_range"])
        self.assertTrue(result["final_target_met"])
        self.assertEqual(result["target_status"], "met_preferred")
        self.assertEqual(result["adjustment_direction"], "none")
        self.assertTrue(all(item["serving_g"] <= 300 for item in result["items"]))

    def test_target_below_minimum_is_not_feasible(self):
        result = calculate(
            **meal_input(
                food_names=["Food"],
                target_calories=20,
                calories_per_100g=[100.0],
            )
        )
        self.assertFalse(result["expanded_search_attempted"])
        self.assertFalse(result["final_target_met"])
        self.assertEqual(result["target_status"], "below_preferred_minimum")
        self.assertFalse(result["target_feasible_within_bounds"])
        self.assertEqual(result["minimum_possible_calories"], 50.0)
        self.assertEqual(result["adjustment_direction"], "decrease")
        self.assertTrue(result["items"][0]["at_preferred_min"])

    def test_grid_miss_inside_bounds_remains_theoretically_feasible(self):
        result = calculate(
            **meal_input(
                food_names=["Energy-dense food"],
                target_calories=550,
                calories_per_100g=[1000.0],
            )
        )
        self.assertFalse(result["final_target_met"])
        self.assertFalse(result["expanded_search_attempted"])
        self.assertEqual(result["target_status"], "closest_preferred")
        self.assertTrue(result["target_feasible_within_bounds"])
        self.assertEqual(result["absolute_calorie_gap"], 50.0)

    def test_target_above_expanded_maximum_is_truly_unreachable(self):
        result = calculate(
            **meal_input(
                food_names=["Food A", "Food B"],
                target_calories=5000,
                calories_per_100g=[100.0, 100.0],
            )
        )
        self.assertTrue(result["expanded_search_attempted"])
        self.assertTrue(result["used_expanded_range"])
        self.assertFalse(result["final_target_met"])
        self.assertEqual(result["target_status"], "above_expanded_maximum")
        self.assertEqual(result["expanded_maximum_possible_calories"], 1200.0)
        self.assertEqual(result["maximum_possible_calories"], 1200.0)
        self.assertTrue(all(item["serving_g"] == 600 for item in result["items"]))
        self.assertIn("cannot be reached", result["feasibility_message"].lower())
        self.assertIn("50–600 g", result["feasibility_message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
