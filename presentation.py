"""Deterministic presentation for verified meal-calculation results."""

from __future__ import annotations

from typing import Any


REQUIRED_MEAL_FIELDS = {
    "target_calories",
    "actual_calories",
    "calorie_difference",
    "absolute_calorie_gap",
    "preferred_target_met",
    "final_target_met",
    "target_status",
    "search_range_used",
    "used_expanded_range",
    "preferred_range",
    "expanded_range",
    "adjustment_direction",
    "feasibility_message",
    "items",
    "totals",
}


def _number(value: Any) -> str:
    """Format a numeric tool value while rejecting booleans and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Meal presentation requires numeric tool fields.")
    return f"{float(value):,.1f}".rstrip("0").rstrip(".")


def _food_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Meal presentation requires a food name.")
    return value.strip()


def _gap_text(result: dict[str, Any]) -> str:
    gap = _number(result["absolute_calorie_gap"])
    difference = result["calorie_difference"]
    if isinstance(difference, bool) or not isinstance(difference, (int, float)):
        raise ValueError("Meal presentation requires a numeric calorie difference.")
    if difference < 0:
        return f"{gap} kcal below target"
    if difference > 0:
        return f"{gap} kcal above target"
    return "0 kcal"


def _target_check(result: dict[str, Any]) -> str:
    """Summarize final feasibility without exposing search candidates."""
    status = result["target_status"]
    if result["final_target_met"] is True:
        if status == "met_expanded":
            return (
                "Target met within tolerance. An expanded serving range was "
                "required to reach the target."
            )
        if status == "met_preferred":
            return "Target met within tolerance."
        raise ValueError("Meal presentation received an inconsistent target status.")
    message = result["feasibility_message"]
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Meal presentation requires a feasibility message.")
    return message.strip()


def format_meal_plan(tool_result: dict[str, Any]) -> str:
    """Format one successful meal-tool result as factual Markdown."""
    if not isinstance(tool_result, dict) or tool_result.get("status") != "ok":
        raise ValueError("Meal presentation requires a successful tool result.")
    missing = REQUIRED_MEAL_FIELDS - tool_result.keys()
    if missing:
        raise ValueError("Meal presentation is missing required tool fields.")
    items = tool_result["items"]
    totals = tool_result["totals"]
    preferred_range = tool_result["preferred_range"]
    expanded_range = tool_result["expanded_range"]
    if (
        not isinstance(items, list)
        or not items
        or not isinstance(totals, dict)
        or not isinstance(preferred_range, dict)
        or not isinstance(expanded_range, dict)
    ):
        raise ValueError("Meal presentation requires items and totals.")
    preferred_max = preferred_range.get("max_serving_g")
    expanded_max = expanded_range.get("max_serving_g")
    if (
        isinstance(preferred_max, bool)
        or not isinstance(preferred_max, (int, float))
        or isinstance(expanded_max, bool)
        or not isinstance(expanded_max, (int, float))
        or preferred_max >= expanded_max
    ):
        raise ValueError("Meal presentation requires valid serving ranges.")

    lines = [
        "### Meal plan",
        "",
        f"**Target:** {_number(tool_result['target_calories'])} kcal",
        f"**Calculated:** {_number(tool_result['actual_calories'])} kcal",
        f"**Gap:** {_gap_text(tool_result)}",
        "",
        "### Ingredients",
        "",
    ]
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Meal presentation received an invalid item.")
        serving = item.get("serving_g")
        expanded_label = (
            " · expanded portion"
            if isinstance(serving, (int, float)) and serving > preferred_max
            else ""
        )
        lines.append(
            f"- **{_food_name(item.get('food_name'))}** — {_number(serving)} g"
            f"{expanded_label}; {_number(item.get('calories'))} kcal, "
            f"{_number(item.get('protein_g'))} g protein, "
            f"{_number(item.get('carbs_g'))} g carbohydrates, "
            f"{_number(item.get('fat_g'))} g fat"
        )

    lines.extend(
        [
            "### Totals",
            "",
            f"- **Calories:** {_number(totals.get('calories'))} kcal",
            f"- **Protein:** {_number(totals.get('protein_g'))} g",
            f"- **Carbohydrates:** {_number(totals.get('carbs_g'))} g",
            f"- **Fat:** {_number(totals.get('fat_g'))} g",
            "",
            "### Target check",
            "",
            _target_check(tool_result),
        ]
    )
    return "\n".join(lines)
