"""
agent.py — the Agent definition.

The system prompt is imported from prompts.py.
This file assembles the CalorieChef Agent with local deterministic tools and
one MCP server connection supplied by the application lifecycle.
"""

from agents import Agent
from agents.mcp import MCPServer

from prompts import SYSTEM_PROMPT
from tools import calculate_macro_calories, calculate_meal_nutrition


def create_agent(nutrition_server: MCPServer, memory_evidence: str) -> Agent:
    """Create CalorieChef with local tools, USDA MCP, and retrieved evidence."""
    return Agent(
        name="CalorieChef",
        instructions=(
            SYSTEM_PROMPT
            + "\n\n[LONG-TERM MEMORY RETRIEVAL]\n"
            + memory_evidence
        ),
        tools=[calculate_macro_calories, calculate_meal_nutrition],
        mcp_servers=[nutrition_server],
    )
