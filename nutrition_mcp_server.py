"""
nutrition_mcp_server.py — USDA FoodData Central MCP Server for CalorieChef.

This MCP server exposes nutrition lookup tools backed by the USDA
FoodData Central API.

Run:
    python nutrition_mcp_server.py
"""

from __future__ import annotations

import os
import warnings

import httpx
from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message="Field 'lifespan' has an incomplete definition.*",
)

from mcp.server.fastmcp import FastMCP


load_dotenv()

USDA_API_KEY = os.getenv("USDA_API_KEY", "").strip()
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

mcp = FastMCP(
    "CalorieChef Nutrition Server",
    log_level="WARNING",
)


def _require_api_key() -> str:
    if not USDA_API_KEY:
        raise RuntimeError(
            "USDA_API_KEY is not configured. Add it to the local .env file."
        )

    return USDA_API_KEY


def _food_priority(food: dict) -> int:
    """
    Rank USDA food data types for generic food searches.

    Lower values have higher priority:
    Foundation > Survey/FNDDS > SR Legacy > Branded > Other
    """
    data_type = str(food.get("dataType", "")).lower()

    if "foundation" in data_type:
        return 0

    if "survey" in data_type or "fndds" in data_type:
        return 1

    if "sr legacy" in data_type:
        return 2

    if "branded" in data_type:
        return 3

    return 4


def _has_core_nutrients(food: dict) -> bool:
    """Return whether a USDA search result contains the required nutrients."""
    nutrient_ids = {
        item.get("nutrientId")
        for item in food.get("foodNutrients", [])
        if item.get("value") is not None
    }

    return (
        {1003, 1004, 1005}.issubset(nutrient_ids)
        and bool({1008, 2047, 2048} & nutrient_ids)
    )


@mcp.tool()
def search_food(food_query: str, page_size: int = 5) -> dict:
    """
    Search USDA FoodData Central for foods matching a user-provided query.

    Use this tool when CalorieChef needs verified nutrition data for a food
    but does not yet have a specific USDA FDC ID.

    For generic foods, Foundation and Survey/FNDDS records are prioritized
    over branded products.

    Args:
        food_query: Food name or description, such as "chicken breast".
        page_size: Maximum number of candidate foods to return. Defaults to 5.

    Returns:
        A structured list of USDA food candidates with FDC IDs, descriptions,
        data types, and brand names when available.
    """

    food_query = food_query.strip()

    if not food_query:
        return {
            "status": "error",
            "message": "food_query cannot be empty.",
        }

    page_size = max(1, min(page_size, 10))

    try:
        response = httpx.get(
            f"{USDA_BASE_URL}/foods/search",
            headers={
                "X-Api-Key": _require_api_key(),
            },
            params={
                "query": food_query,
                "pageSize": min(page_size * 5, 50),
            },
            timeout=15.0,
        )

        response.raise_for_status()
        payload = response.json()

    except Exception as exc:
        return {
            "status": "error",
            "query": food_query,
            "count": 0,
            "foods": [],
            "message": (
                f"USDA food search failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    foods = payload.get("foods", [])

    if not foods:
        return {
            "status": "ok",
            "query": food_query,
            "count": 0,
            "foods": [],
            "message": "No matching foods were found in USDA FoodData Central.",
        }

    foods = sorted(
        foods,
        key=lambda food: (
            not _has_core_nutrients(food),
            _food_priority(food),
        ),
    )

    results = [
        {
            "fdc_id": food.get("fdcId"),
            "description": food.get("description"),
            "data_type": food.get("dataType"),
            "brand_name": food.get("brandName"),
        }
        for food in foods[:page_size]
    ]

    return {
        "status": "ok",
        "query": food_query,
        "count": len(results),
        "foods": results,
    }


@mcp.tool()
def get_food_nutrition(fdc_id: int) -> dict:
    """
    Retrieve verified nutrition data for a specific USDA FoodData Central food.

    Use this tool after search_food identifies the desired food and provides
    its FDC ID.

    Args:
        fdc_id: USDA FoodData Central food identifier.

    Returns:
        A structured nutrition result containing USDA-reported calories,
        protein, carbohydrates, and fat when available.
    """

    if fdc_id <= 0:
        return {
            "status": "error",
            "fdc_id": fdc_id,
            "message": "fdc_id must be a positive integer.",
        }

    try:
        response = httpx.get(
            f"{USDA_BASE_URL}/food/{fdc_id}",
            headers={
                "X-Api-Key": _require_api_key(),
            },
            timeout=15.0,
        )

        response.raise_for_status()
        payload = response.json()

    except Exception as exc:
        return {
            "status": "error",
            "fdc_id": fdc_id,
            "message": (
                f"USDA nutrition lookup failed for FDC ID {fdc_id}: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    nutrients = payload.get("foodNutrients", [])

    extracted = {
        "calories": None,
        "protein_g": None,
        "carbs_g": None,
        "fat_g": None,
    }
    energy_priority = 0

    for item in nutrients:
        nutrient = item.get("nutrient", {})
        nutrient_id = nutrient.get("id", item.get("nutrientId"))
        name = str(
            nutrient.get("name", item.get("nutrientName", ""))
        ).lower()
        amount = item.get("amount", item.get("value"))

        if amount is None:
            continue

        if nutrient_id in {1008, 2047, 2048} or name.startswith("energy"):
            priority = {1008: 3, 2048: 2, 2047: 1}.get(nutrient_id, 1)

            if priority > energy_priority:
                extracted["calories"] = amount
                energy_priority = priority

        elif nutrient_id == 1003 or name == "protein":
            extracted["protein_g"] = amount

        elif nutrient_id == 1005 or name == "carbohydrate, by difference":
            extracted["carbs_g"] = amount

        elif nutrient_id == 1004 or name == "total lipid (fat)":
            extracted["fat_g"] = amount

    missing_fields = [
        field for field, value in extracted.items() if value is None
    ]
    result = {
        "status": "partial" if missing_fields else "ok",
        "source": "USDA FoodData Central",
        "fdc_id": payload.get("fdcId", fdc_id),
        "description": payload.get("description"),
        "data_type": payload.get("dataType"),
        "basis": "per 100 g",
        **extracted,
    }

    if missing_fields:
        result["missing_fields"] = missing_fields
        result["message"] = (
            "USDA does not provide all requested nutrient fields for this "
            "food record. Do not infer or invent the missing values."
        )

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
