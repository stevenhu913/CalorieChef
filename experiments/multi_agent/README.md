# Experimental Multi-Agent Architecture

This package preserves a controlled Manager/Specialist exploration. It is not
imported by the production Agent Core or FastAPI application.

## Why it was explored

Preference and safety interpretation have different tool and context needs
from verified nutrition lookup. The experiment tested whether separating those
responsibilities improved grounding, failure containment, or trace clarity.

## Boundaries

- `CalorieChefManager` owns routing, conflict handling, merge policy, and the
  only final user-facing response.
- `PreferenceSafetySpecialist` receives accepted constraints and preferences.
  It has no USDA, calculator, memory, or write tools.
- `NutritionSpecialist` receives a bounded nutrition task. It can use USDA MCP
  and deterministic calculators but cannot read or write user memory.

Every specialist returns a structured result with status, findings,
confidence, and limitations. The Manager reads status before findings,
discards failed evidence, surfaces partial limitations, and rejects nutrition
that conflicts with hard constraints.

## Timeout and partial failure

Specialist calls have bounded turn counts and timeouts. Structured-output
failure permits one bounded retry. A timeout or parse failure becomes a
structured partial or failed result; nutrition values are never fabricated.

## Retention decision

The experiment was not selected as the production default. One controlled run
showed higher orchestration cost and no demonstrated grounded-quality
improvement. A successful Multi-Agent nutrition happy path was not observed
because the local model failed structured tool-call output before USDA was
reached. One run is not statistical evidence and does not establish that
Multi-Agent systems are universally worse.

Run experimental scripts explicitly from the repository root:

```bash
python -m experiments.multi_agent.run_experiment
python -m experiments.multi_agent.compare
python -m experiments.multi_agent.scenarios
```

These commands can invoke local models, embeddings, and external nutrition
services. They are not part of deterministic test execution.
