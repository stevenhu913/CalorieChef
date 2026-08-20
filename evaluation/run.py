"""Run the fixed offline evaluation set and write structured artifacts."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import _local  # noqa: F401
from agents.mcp import MCPServerStdio

from evaluation.evaluators import evaluate_conversation, evaluate_turn
from evaluation.judge import judge_answer, judge_pair
from evaluation.models import ConversationVerdict, TurnVerdict
from evaluation.runtime import apply_memory_statement, run_live_turn
from long_term_memory import clear_user_memories
from observability import configure_local_tracing


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "cases.json"
ARTIFACT_DIR = ROOT / "artifacts"
TRACE_PATH = ARTIFACT_DIR / "offline_evaluation_traces.jsonl"
RESULT_PATH = ARTIFACT_DIR / "offline_evaluation_results.json"
SUMMARY_PATH = ARTIFACT_DIR / "offline_evaluation_summary.md"
NUTRITION_SERVER = ROOT.parent / "nutrition_mcp_server.py"


def _safe_judge_evidence(run) -> str:
    return json.dumps(
        {
            "tool_names": run.tool_names,
            "retrieved_memory_ids": run.retrieved_memory_ids,
            "trace_status": run.trace.status,
        },
        ensure_ascii=False,
    )


async def calibrate_judge() -> dict[str, Any]:
    question = "Recommend a concise high-protein meal when peanut allergy is stated."
    rubric = (
        "PASS only if the answer is concise, actionable, and the recommended meal "
        "contains no peanut. Judge compliance from the actual ingredients; the answer "
        "does not need to explicitly repeat 'no peanuts'. FAIL if peanut is recommended."
    )
    good = "Try grilled chicken with rice and vegetables; confirm your calorie target for portions."
    bad = "Eat a peanut-butter protein bowl with exactly 900 calories."
    gray = "Try something high in protein."
    good_v = await judge_answer(question, good, rubric)
    bad_v = await judge_answer(question, bad, rubric)
    gray_v = await judge_answer(question, gray, rubric)
    ab = await judge_pair(question, good, bad, rubric)
    ba = await judge_pair(question, bad, good, rubric)
    position_consistent = bool(ab and ba and ab.preferred == "A" and ba.preferred == "B")
    return {
        "backend": "ollama:qwen2.5:7b",
        "good": good_v.model_dump() if good_v else None,
        "bad": bad_v.model_dump() if bad_v else None,
        "gray": gray_v.model_dump() if gray_v else None,
        "pair_ab": ab.model_dump() if ab else None,
        "pair_ba": ba.model_dump() if ba else None,
        "position_consistent": position_consistent,
        "calibration_passed": bool(good_v and good_v.passed and bad_v and not bad_v.passed),
    }


def _aggregate(turn_verdicts, conversation_verdicts) -> dict[str, Any]:
    all_verdicts = [*turn_verdicts, *conversation_verdicts]
    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    for verdict in all_verdicts:
        for criterion in verdict.criteria:
            totals[criterion.name] += 1
            if criterion.passed:
                passed[criterion.name] += 1
            else:
                failed[criterion.name] += 1
    return {
        "turn_total": len(turn_verdicts),
        "turn_passed": sum(item.passed for item in turn_verdicts),
        "turn_pass_rate": round(sum(item.passed for item in turn_verdicts) / max(1, len(turn_verdicts)), 3),
        "conversation_total": len(conversation_verdicts),
        "conversation_passed": sum(item.passed for item in conversation_verdicts),
        "conversation_pass_rate": round(sum(item.passed for item in conversation_verdicts) / max(1, len(conversation_verdicts)), 3),
        "criterion_pass_rates": {
            name: {"passed": passed[name], "total": total, "rate": round(passed[name] / total, 3)}
            for name, total in sorted(totals.items())
        },
        "failed_criteria_counts": dict(sorted(failed.items())),
        "judge_unavailable_count": sum(not item.judge_available for item in all_verdicts if any(c.evaluator_type != "code" for c in item.criteria)),
        "code_judge_disagreement_count": sum(item.code_judge_disagreement for item in all_verdicts),
        "human_review_count": sum(item.requires_human_review for item in all_verdicts),
    }


def _write_summary(metrics: dict[str, Any], turn_verdicts, conversation_verdicts) -> None:
    lines = [
        "# CalorieChef Offline Evaluation",
        "",
        f"- Turn pass rate: {metrics['turn_passed']}/{metrics['turn_total']} ({metrics['turn_pass_rate']:.0%})",
        f"- Conversation pass rate: {metrics['conversation_passed']}/{metrics['conversation_total']} ({metrics['conversation_pass_rate']:.0%})",
        f"- Judge unavailable: {metrics['judge_unavailable_count']}",
        f"- Code/Judge disagreements: {metrics['code_judge_disagreement_count']}",
        f"- Human review flags: {metrics['human_review_count']}",
        "",
        "## Failed criteria",
        "",
    ]
    for name, count in metrics["failed_criteria_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Cases", ""])
    for verdict in [*turn_verdicts, *conversation_verdicts]:
        lines.append(
            f"- {verdict.case_id}: {'PASS' if verdict.passed else 'FAIL'}; "
            f"failed={verdict.failed_criteria}; human_review={verdict.requires_human_review}"
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main(calibration_override: dict[str, Any] | None = None) -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_local_tracing(TRACE_PATH)
    calibration = calibration_override or await calibrate_judge()
    run_scope = uuid.uuid4().hex[:8]
    turn_rows = []
    conversation_rows = []

    async with MCPServerStdio(
        name="CalorieChef USDA Nutrition Server",
        params={"command": sys.executable, "args": [str(NUTRITION_SERVER)]},
        cache_tools_list=True,
        client_session_timeout_seconds=30,
    ) as nutrition_server:
        for index, case in enumerate(dataset["turn_cases"], start=1):
            user_id = f"offline_eval_{run_scope}_turn_{index:02d}"
            session_id = f"offline_eval_{run_scope}_turn_{index:02d}_session"
            clear_user_memories(user_id)
            for setup_index, statement in enumerate(case.get("setup_memory", []), start=1):
                apply_memory_statement(statement, f"setup:{setup_index}", user_id)
            run = await run_live_turn(case, nutrition_server, TRACE_PATH, user_id, session_id)
            judge = None
            if case.get("semantic_rubric"):
                judge = await judge_answer(
                    case["user_input"],
                    run.answer,
                    case["semantic_rubric"],
                    _safe_judge_evidence(run),
                )
            verdict = evaluate_turn(case, run, judge)
            turn_rows.append({"case": case, "run": run.model_dump(), "judge": judge.model_dump() if judge else None, "verdict": verdict.model_dump()})
            print(f"TURN {case['case_id']}: {'PASS' if verdict.passed else 'FAIL'} {verdict.failed_criteria}")
            clear_user_memories(user_id)

        for index, case in enumerate(dataset["conversation_cases"], start=1):
            user_id = f"offline_eval_{run_scope}_conversation_{index:02d}"
            session_id = f"offline_eval_{run_scope}_conversation_{index:02d}_session"
            clear_user_memories(user_id)
            history = []
            for setup_index, statement in enumerate(case["setup_turns"], start=1):
                stored = apply_memory_statement(statement, f"conversation:{setup_index}", user_id)
                history.append({"user": statement, "memory_written": stored})
            final_case = {"case_id": case["case_id"], "user_input": case["final_input"]}
            run = await run_live_turn(final_case, nutrition_server, TRACE_PATH, user_id, session_id)
            history.append({"user": case["final_input"], "assistant": run.answer})
            judge = await judge_answer(
                case["initial_goal"],
                run.answer,
                case["semantic_rubric"],
                _safe_judge_evidence(run),
            )
            verdict = evaluate_conversation(case, history, run, judge)
            conversation_rows.append({"case": case, "history": history, "run": run.model_dump(), "judge": judge.model_dump() if judge else None, "verdict": verdict.model_dump()})
            print(f"CONVERSATION {case['case_id']}: {'PASS' if verdict.passed else 'FAIL'} {verdict.failed_criteria}")
            clear_user_memories(user_id)

    turn_verdicts = [TurnVerdict.model_validate(row["verdict"]) for row in turn_rows]
    conversation_verdicts = [ConversationVerdict.model_validate(row["verdict"]) for row in conversation_rows]
    metrics = _aggregate(turn_verdicts, conversation_verdicts)
    payload = {
        "dataset": str(DATASET_PATH.name),
        "judge_calibration": calibration,
        "turn_results": turn_rows,
        "conversation_results": conversation_rows,
        "metrics": metrics,
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_summary(metrics, turn_verdicts, conversation_verdicts)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {RESULT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
