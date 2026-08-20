"""Structured models shared by the offline evaluation system."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CriterionResult(BaseModel):
    name: str
    passed: bool
    reason: str
    evaluator_type: Literal["code", "judge", "human_required"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class JudgeVerdict(BaseModel):
    passed: bool
    score: int = Field(ge=0, le=10)
    reason: str
    failed_criteria: list[str] = Field(default_factory=list)


class PairwiseVerdict(BaseModel):
    preferred: Literal["A", "B", "tie"]
    reason: str


class TraceEvidence(BaseModel):
    trace_id: str | None = None
    status: str = "unavailable"
    tool_names: list[str] = Field(default_factory=list)
    generation_count: int = 0
    custom_spans: dict[str, dict[str, Any]] = Field(default_factory=dict)
    token_usage: dict[str, int] | None = None
    errors: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    case_id: str
    answer: str
    router_action: str
    tool_names: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    retrieved_memory_kinds: list[str] = Field(default_factory=list)
    memory_before: list[dict[str, Any]] = Field(default_factory=list)
    memory_after: list[dict[str, Any]] = Field(default_factory=list)
    trace: TraceEvidence = Field(default_factory=TraceEvidence)
    error: str | None = None


class TurnVerdict(BaseModel):
    case_id: str
    passed: bool
    score: int = Field(ge=0, le=10)
    criteria: list[CriterionResult]
    failed_criteria: list[str]
    reason: str
    requires_human_review: bool
    trace_id: str | None = None
    judge_available: bool
    code_judge_disagreement: bool


class ConversationVerdict(BaseModel):
    case_id: str
    passed: bool
    score: int = Field(ge=0, le=10)
    goal_completed: bool
    constraints_preserved: bool
    final_artifact_complete: bool
    criteria: list[CriterionResult]
    failed_criteria: list[str]
    reason: str
    requires_human_review: bool
    judge_available: bool
    code_judge_disagreement: bool
