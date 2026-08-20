"""
test_nutrition_mcp.py — end-to-end test for the CalorieChef USDA MCP server.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


NUTRITION_SERVER = Path(__file__).resolve().parents[1] / "nutrition_mcp_server.py"


def parse_tool_result(result) -> dict:
    for content in result.content:
        text = getattr(content, "text", None)

        if not text:
            continue

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue

    return {}


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(NUTRITION_SERVER)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}")

            print("\nSearching USDA for chicken breast...")

            search_result = await session.call_tool(
                "search_food",
                {
                    "food_query": "chicken breast",
                    "page_size": 3,
                },
            )

            search_data = parse_tool_result(search_result)

            print("\nSearch result:")
            print(json.dumps(search_data, indent=2))

            foods = search_data.get("foods", [])

            if not foods:
                print("\nNo USDA food candidates found.")
                return

            selected_food = foods[0]
            fdc_id = selected_food["fdc_id"]

            print(
                f"\nSelected food: {selected_food['description']} "
                f"(FDC ID: {fdc_id})"
            )

            nutrition_result = await session.call_tool(
                "get_food_nutrition",
                {
                    "fdc_id": fdc_id,
                },
            )

            nutrition_data = parse_tool_result(nutrition_result)

            print("\nNutrition result:")
            print(json.dumps(nutrition_data, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
