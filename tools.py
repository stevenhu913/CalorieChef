"""
tools.py — deterministic tools for CalorieChef.

This file contains local function tools that provide reliable calculations
for both single macro totals and multi-food meal portions.
"""

from itertools import product

from agents import function_tool


@function_tool
def calculate_macro_calories(
    protein_g: float,
    carbs_g: float,
    fat_g: float,
) -> dict:
    """
    Calculate total calories from protein, carbohydrates, and fat.

    Use this tool when macronutrient amounts are already known and a calorie
    total must be calculated deterministically instead of estimated by the model.

    Args:
        protein_g: Protein amount in grams. Must be non-negative.
        carbs_g: Carbohydrate amount in grams. Must be non-negative.
        fat_g: Fat amount in grams. Must be non-negative.

    Returns:
        A structured result containing the original macronutrient values and
        the calculated calorie total, or an error message for invalid input.
    """

    if protein_g < 0 or carbs_g < 0 or fat_g < 0:
        return {
            "status": "error",
            "message": "Macronutrient values cannot be negative.",
        }

    calories = (
        protein_g * 4
        + carbs_g * 4
        + fat_g * 9
    )

    return {
        "status": "ok",
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "calories": round(calories, 1),
    }


@function_tool
def calculate_meal_nutrition(
    food_names: list[str],
    target_calories: float,
    calories_per_100g: list[float],
    protein_per_100g: list[float],
    carbs_per_100g: list[float],
    fat_per_100g: list[float],
) -> dict:
    """Choose bounded portions and total a meal using verified nutrition.

    Use this tool after USDA lookups return nutrition for every food and the
    user provides a calorie target. It searches serving sizes from 50 to 300
    grams in 10-gram steps, chooses the closest calorie total, and favors more
    protein when alternatives are equally close. Values at the same list index
    must describe the same food.

    Args:
        food_names: USDA food descriptions in meal order.
        target_calories: Requested total meal calories.
        calories_per_100g: USDA calories for each food per 100 grams.
        protein_per_100g: USDA protein grams for each food per 100 grams.
        carbs_per_100g: USDA carbohydrate grams for each food per 100 grams.
        fat_per_100g: USDA fat grams for each food per 100 grams.

    Returns:
        A structured result with selected portions and deterministic totals,
        or a structured error if fields are missing or invalid. Supports one
        to three foods to keep execution bounded.
    """
    fields = {
        "food_names": food_names,
        "calories_per_100g": calories_per_100g,
        "protein_per_100g": protein_per_100g,
        "carbs_per_100g": carbs_per_100g,
        "fat_per_100g": fat_per_100g,
    }
    lengths = {len(values) for values in fields.values()}

    if not food_names:
        return {
            "status": "error",
            "message": "At least one food is required.",
        }

    if len(food_names) > 3:
        return {
            "status": "error",
            "message": "Meal planning supports at most three foods at a time.",
        }

    if target_calories <= 0:
        return {
            "status": "error",
            "message": "target_calories must be greater than zero.",
        }

    if len(lengths) != 1:
        return {
            "status": "error",
            "message": (
                "All meal nutrition lists must contain the same number "
                "of items."
            ),
        }

    numeric_fields = (
        calories_per_100g,
        protein_per_100g,
        carbs_per_100g,
        fat_per_100g,
    )

    if any(value < 0 for values in numeric_fields for value in values):
        return {
            "status": "error",
            "message": "Nutrition values cannot be negative.",
        }

    serving_options = range(50, 301, 10)
    best_servings = None
    best_totals = None
    best_score = None

    for servings in product(serving_options, repeat=len(food_names)):
        calories = sum(
            calories_per_100g[index] * grams / 100
            for index, grams in enumerate(servings)
        )
        protein = sum(
            protein_per_100g[index] * grams / 100
            for index, grams in enumerate(servings)
        )
        score = (abs(calories - target_calories), -protein)

        if best_score is None or score < best_score:
            best_score = score
            best_servings = servings
            best_totals = {
                "calories": calories,
                "protein_g": protein,
                "carbs_g": sum(
                    carbs_per_100g[index] * grams / 100
                    for index, grams in enumerate(servings)
                ),
                "fat_g": sum(
                    fat_per_100g[index] * grams / 100
                    for index, grams in enumerate(servings)
                ),
            }

    items = []

    for index, food_name in enumerate(food_names):
        grams = best_servings[index]
        scale = grams / 100
        items.append({
            "food_name": food_name,
            "serving_g": grams,
            "calories": round(calories_per_100g[index] * scale, 1),
            "protein_g": round(protein_per_100g[index] * scale, 1),
            "carbs_g": round(carbs_per_100g[index] * scale, 1),
            "fat_g": round(fat_per_100g[index] * scale, 1),
        })

    totals = {
        field: round(value, 1) for field, value in best_totals.items()
    }
    difference = round(totals["calories"] - target_calories, 1)
    tolerance = max(25, target_calories * 0.05)

    return {
        "status": "ok",
        "basis": "USDA per-100-g values scaled to selected serving grams",
        "target_calories": round(target_calories, 1),
        "target_met": abs(difference) <= tolerance,
        "calorie_difference": difference,
        "items": items,
        "totals": totals,
    }
