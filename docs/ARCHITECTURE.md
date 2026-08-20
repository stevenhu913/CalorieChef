# Architecture

## Product boundary

CalorieChef provides grounded nutrition information and meal-planning support.
It is not a medical diagnosis system. Exact food values come from USDA
FoodData Central, arithmetic comes from deterministic Python tools, and the
language model coordinates those capabilities within explicit safety rules.

## Stable Single-Agent runtime

`agent_core.py` owns one bounded request lifecycle and one USDA MCP subprocess
for the FastAPI application lifetime. `agent.py` defines the Agent and exposes
only the local calculators plus the MCP server. `prompts.py` defines grounding,
failure, privacy, and allergy boundaries. Neither the browser nor FastAPI
duplicates Agent logic.

```text
Browser / CLI
      |
      v
FastAPI protocol or terminal loop
      |
      v
Agent Core ----- SQLite short-term session
      |
      +----- accepted long-term memory evidence
      |
      v
Single Agent
  |                    |
  v                    v
USDA MCP tools      deterministic calculators
  |                    |
  +---------+----------+
            v
       grounded answer
            |
            v
   sanitized local trace
```

## Tools and source of truth

`search_food` must precede `get_food_nutrition`. The resulting USDA record is
the source of truth for food nutrition. `calculate_macro_calories` applies the
4/4/9 rule to supplied macro grams. `calculate_meal_nutrition` derives bounded
portions and totals from verified per-100-gram records. The Agent must disclose
missing evidence rather than fabricate exact values.

## Memory

SQLite stores full short-term session history while a bounded window is sent to
the model. Long-term memory uses explicit user scopes, conservative
deterministic write routing, topic-stable IDs, versioned upserts, cosine
distance, and a small evidence budget. Allergy and dietary records receive
priority. Current user text supersedes retrieved preference context.

Local mode can use Chroma and an Ollama embedding model. Hosted mode disables
long-term retrieval by default because the deployment package has no hosted
embedding service or durable vector store.

## Structured observability

The local trace processor records request and span status, latency, tool names,
safe attributes, and token usage. It redacts common credential forms and omits
raw prompts, tool payloads, embeddings, and private history.

## Experimental Multi-Agent decision

The isolated experiment uses a Manager, a tool-free Preference/Safety
Specialist, and a Nutrition Specialist with USDA and calculator access. The
Manager remains the only final-response writer. Personalized meal planning is
sequential because nutrition depends on extracted constraints.

In one controlled observation, the Single-Agent path took 11.65 seconds with
one generation and 1,957 tokens; the Multi-Agent path took 16.76 seconds with
three generations and 2,145 tokens. Neither produced a fully grounded final
meal in that run. This is not a benchmark or statistical superiority claim.
The experiment remains separated from production until repeated evidence shows
a reliability or quality benefit.
