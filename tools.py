"""
tools.py — deterministic tools for CalorieChef.

This file contains local function tools that provide reliable calculations
for both single macro totals and multi-food meal portions.
"""

from itertools import product

from agents import function_tool


PREFERRED_MIN_SERVING_G = 50
PREFERRED_MAX_SERVING_G = 300
EXPANDED_MIN_SERVING_G = 50
EXPANDED_MAX_SERVING_G = 600
SERVING_STEP_G = 10
MAX_MEAL_FOODS = 3
TARGET_TOLERANCE_RATIO = 0.05
MIN_TARGET_TOLERANCE_KCAL = 25


def _number(value: float) -> str:
    """Format one calculated value without subjective qualifiers."""
    rounded = round(value, 1)
    return f"{rounded:,.1f}".rstrip("0").rstrip(".")


def _search_meal_portions(
    food_names: list[str],
    calories_per_100g: list[float],
    protein_per_100g: list[float],
    carbs_per_100g: list[float],
    fat_per_100g: list[float],
    target_calories: float,
    min_serving_g: int,
    max_serving_g: int,
    step_g: int,
) -> dict:
    """Return the best exhaustive-search candidate for one bounded range."""
    serving_options = range(min_serving_g, max_serving_g + 1, step_g)
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

    return {
        "servings": best_servings,
        "totals": {
            field: round(value, 1) for field, value in best_totals.items()
        },
    }


def _calorie_bounds(
    calories_per_100g: list[float],
    min_serving_g: int,
    max_serving_g: int,
) -> tuple[float, float]:
    return (
        sum(value * min_serving_g / 100 for value in calories_per_100g),
        sum(value * max_serving_g / 100 for value in calories_per_100g),
    )


def _gap_clause(difference: float, absolute_gap: float) -> str:
    if difference < 0:
        return f"{_number(absolute_gap)} kcal below target"
    if difference > 0:
        return f"{_number(absolute_gap)} kcal above target"
    return "exactly on target"


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
    user provides a calorie target. It first searches the preferred 50–300 g
    range in 10 g steps. If that result is meaningfully below target at the
    upper boundary, it performs one expanded 50–600 g search. Both stages
    minimize calorie error and use protein as the tie-break. Values at the same
    list index must describe the same food.

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

    if len(food_names) > MAX_MEAL_FOODS:
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

    tolerance = max(
        MIN_TARGET_TOLERANCE_KCAL,
        target_calories * TARGET_TOLERANCE_RATIO,
    )
    preferred = _search_meal_portions(
        food_names,
        calories_per_100g,
        protein_per_100g,
        carbs_per_100g,
        fat_per_100g,
        target_calories,
        PREFERRED_MIN_SERVING_G,
        PREFERRED_MAX_SERVING_G,
        SERVING_STEP_G,
    )
    preferred_actual = preferred["totals"]["calories"]
    preferred_difference = round(preferred_actual - target_calories, 1)
    preferred_absolute_gap = round(abs(preferred_difference), 1)
    preferred_target_met = preferred_absolute_gap <= tolerance
    preferred_minimum, preferred_maximum = _calorie_bounds(
        calories_per_100g,
        PREFERRED_MIN_SERVING_G,
        PREFERRED_MAX_SERVING_G,
    )

    preferred_at_maximum = any(
        grams == PREFERRED_MAX_SERVING_G for grams in preferred["servings"]
    )
    expanded_search_attempted = (
        not preferred_target_met
        and preferred_actual < target_calories
        and (
            target_calories > preferred_maximum
            or preferred_at_maximum
        )
    )
    expanded = None
    expanded_actual = None
    expanded_difference = None
    expanded_absolute_gap = None
    expanded_target_met = None
    expanded_result_improved_gap = False
    if expanded_search_attempted:
        expanded = _search_meal_portions(
            food_names,
            calories_per_100g,
            protein_per_100g,
            carbs_per_100g,
            fat_per_100g,
            target_calories,
            EXPANDED_MIN_SERVING_G,
            EXPANDED_MAX_SERVING_G,
            SERVING_STEP_G,
        )
        expanded_actual = expanded["totals"]["calories"]
        expanded_difference = round(expanded_actual - target_calories, 1)
        expanded_absolute_gap = round(abs(expanded_difference), 1)
        expanded_target_met = expanded_absolute_gap <= tolerance
        expanded_result_improved_gap = (
            expanded_absolute_gap < preferred_absolute_gap
        )

    used_expanded_range = expanded_result_improved_gap
    selected = expanded if used_expanded_range else preferred
    search_range_used = "expanded" if used_expanded_range else "preferred"
    totals = selected["totals"]
    actual_calories = totals["calories"]
    difference = round(actual_calories - target_calories, 1)
    absolute_gap = round(abs(difference), 1)
    final_target_met = absolute_gap <= tolerance
    expanded_minimum, expanded_maximum = _calorie_bounds(
        calories_per_100g,
        EXPANDED_MIN_SERVING_G,
        EXPANDED_MAX_SERVING_G,
    )

    items = []
    for index, food_name in enumerate(food_names):
        grams = selected["servings"][index]
        scale = grams / 100
        items.append(
            {
                "food_name": food_name,
                "serving_g": grams,
                "calories": round(calories_per_100g[index] * scale, 1),
                "protein_g": round(protein_per_100g[index] * scale, 1),
                "carbs_g": round(carbs_per_100g[index] * scale, 1),
                "fat_g": round(fat_per_100g[index] * scale, 1),
                "at_preferred_min": grams == PREFERRED_MIN_SERVING_G,
                "at_preferred_max": grams == PREFERRED_MAX_SERVING_G,
                "at_expanded_max": grams == EXPANDED_MAX_SERVING_G,
                "at_min_serving": grams == PREFERRED_MIN_SERVING_G,
                "at_max_serving": grams
                == (
                    EXPANDED_MAX_SERVING_G
                    if used_expanded_range
                    else PREFERRED_MAX_SERVING_G
                ),
            }
        )

    boundary_hits = []
    for item in items:
        if item["at_preferred_min"]:
            boundary_hits.append(
                {
                    "food_name": item["food_name"],
                    "boundary": "preferred_min",
                    "serving_g": item["serving_g"],
                }
            )
        if item["at_preferred_max"]:
            boundary_hits.append(
                {
                    "food_name": item["food_name"],
                    "boundary": "preferred_max",
                    "serving_g": item["serving_g"],
                }
            )
        if item["at_expanded_max"]:
            boundary_hits.append(
                {
                    "food_name": item["food_name"],
                    "boundary": "expanded_max",
                    "serving_g": item["serving_g"],
                }
            )

    if preferred_target_met:
        target_status = "met_preferred"
    elif final_target_met and used_expanded_range:
        target_status = "met_expanded"
    elif target_calories < preferred_minimum:
        target_status = "below_preferred_minimum"
    elif expanded_search_attempted and target_calories > expanded_maximum:
        target_status = "above_expanded_maximum"
    elif used_expanded_range:
        target_status = "closest_expanded"
    else:
        target_status = "closest_preferred"

    adjustment_direction = (
        "none"
        if final_target_met
        else ("increase" if difference < 0 else "decrease")
    )
    gap_clause = _gap_clause(difference, absolute_gap)
    expanded_foods = [
        item["food_name"]
        for item in items
        if item["serving_g"] > PREFERRED_MAX_SERVING_G
    ]
    if target_status == "met_preferred":
        feasibility_message = (
            "Meal is within the target tolerance using preferred serving sizes. "
            f"The final combination is {_number(actual_calories)} kcal, "
            f"{gap_clause}."
        )
    elif target_status == "met_expanded":
        names = ", ".join(expanded_foods)
        verb = "requires" if len(expanded_foods) == 1 else "require"
        feasibility_message = (
            f"The {_number(target_calories)} kcal target is achievable within "
            "the expanded practical range. The final combination is "
            f"{_number(actual_calories)} kcal, {gap_clause}. {names} {verb} a "
            f"serving above the preferred {PREFERRED_MAX_SERVING_G} g range."
        )
    elif target_status == "above_expanded_maximum":
        feasibility_message = (
            "The requested target cannot be reached with these foods within "
            f"the practical {EXPANDED_MIN_SERVING_G}–{EXPANDED_MAX_SERVING_G} g "
            f"serving range. The closest combination is "
            f"{_number(actual_calories)} kcal, {gap_clause}."
        )
    elif target_status == "below_preferred_minimum":
        feasibility_message = (
            "The requested target is below the minimum achievable calories "
            f"in the preferred {PREFERRED_MIN_SERVING_G}–"
            f"{PREFERRED_MAX_SERVING_G} g serving range. The closest "
            f"combination is {_number(actual_calories)} kcal, {gap_clause}."
        )
    elif target_status == "closest_expanded":
        feasibility_message = (
            f"The expanded {EXPANDED_MIN_SERVING_G}–{EXPANDED_MAX_SERVING_G} g "
            "search produced the closest practical combination, but the meal "
            f"remains {gap_clause}."
        )
    else:
        feasibility_message = (
            "The preferred serving search produced the closest combination, "
            f"but the meal remains {gap_clause}."
        )

    final_minimum = expanded_minimum if used_expanded_range else preferred_minimum
    final_maximum = expanded_maximum if used_expanded_range else preferred_maximum

    return {
        "status": "ok",
        "basis": "USDA per-100-g values scaled to selected serving grams",
        "target_calories": round(target_calories, 1),
        "actual_calories": actual_calories,
        "calorie_difference": difference,
        "absolute_calorie_gap": absolute_gap,
        "calorie_gap_percentage": round(
            absolute_gap / target_calories * 100,
            2,
        ),
        "tolerance_calories": round(tolerance, 1),
        "preferred_range": {
            "min_serving_g": PREFERRED_MIN_SERVING_G,
            "max_serving_g": PREFERRED_MAX_SERVING_G,
        },
        "expanded_range": {
            "min_serving_g": EXPANDED_MIN_SERVING_G,
            "max_serving_g": EXPANDED_MAX_SERVING_G,
        },
        "preferred_target_met": preferred_target_met,
        "preferred_actual_calories": preferred_actual,
        "preferred_calorie_difference": preferred_difference,
        "preferred_absolute_calorie_gap": preferred_absolute_gap,
        "preferred_minimum_possible_calories": round(preferred_minimum, 1),
        "preferred_maximum_possible_calories": round(preferred_maximum, 1),
        "expanded_search_attempted": expanded_search_attempted,
        "expanded_actual_calories": expanded_actual,
        "expanded_calorie_difference": expanded_difference,
        "expanded_absolute_calorie_gap": expanded_absolute_gap,
        "expanded_target_met": expanded_target_met,
        "expanded_result_improved_gap": expanded_result_improved_gap,
        "used_expanded_range": used_expanded_range,
        "final_target_met": final_target_met,
        "target_met": final_target_met,
        "target_status": target_status,
        "search_range_used": search_range_used,
        "target_feasible_within_bounds": (
            expanded_minimum <= target_calories <= expanded_maximum
        ),
        "minimum_possible_calories": round(final_minimum, 1),
        "maximum_possible_calories": round(final_maximum, 1),
        "expanded_minimum_possible_calories": round(expanded_minimum, 1),
        "expanded_maximum_possible_calories": round(expanded_maximum, 1),
        "adjustment_direction": adjustment_direction,
        "boundary_hits": boundary_hits,
        "feasibility_message": feasibility_message,
        "items": items,
        "totals": totals,
    }
