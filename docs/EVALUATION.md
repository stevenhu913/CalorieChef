# Evaluation

## Scope

`evaluation/cases.json` defines fixed turn and conversation cases. The runner
isolates user and session identifiers, records sanitized evidence, and writes
generated results under an ignored `evaluation/artifacts/` directory.

## Deterministic checks

The evaluators check:

- required, forbidden, minimum-count, and ordered tool calls
- numeric results and USDA grounding evidence
- prohibited-food recommendations and allergy preservation
- deterministic memory routing and versioned updates
- final-answer goal and artifact completeness
- trace availability and reported runtime errors

These checks are authoritative for hard facts. A semantic Judge cannot convert
a deterministic failure into a pass.

## Optional semantic Judge

The Judge uses structured verdicts for case-specific semantic rubrics. If the
Judge is unavailable when a rubric is required, semantic quality fails and the
case is flagged for human review. The Judge is tool-free and receives sanitized
evidence rather than raw tool payloads.

Judge output can be unstable. A local Agent and Judge using similar models may
share correlated errors, so calibration and code/Judge disagreement flags are
retained. Human review remains appropriate for ambiguous semantic cases.

## Turn and conversation evaluation

Turn evaluation measures a single response and its tool evidence. Conversation
evaluation additionally checks whether a final goal was completed, current
constraints superseded stale values, and the final artifact is usable.

## Running evaluation

Deterministic evaluator tests require no model or external API:

```bash
python -m unittest tests.test_evaluation
```

The fixed evaluation runner can invoke model, USDA, embedding, and Judge
services. Run it only in an explicitly configured environment:

```bash
python -m evaluation.run
```

Generated evaluation artifacts are ignored and must not be committed.
