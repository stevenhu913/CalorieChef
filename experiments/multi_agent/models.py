"""Structured contracts for experimental multi-Agent collaboration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ExpertStatus = Literal["ok", "partial", "failed"]
RouteName = Literal["preference_only", "nutrition_only", "personalized_meal"]


class ExpertResult(BaseModel):
    """Base contract every specialist must return to the Manager."""

    status: ExpertStatus
    findings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class PreferenceSafetyTask(BaseModel):
    """Minimal context allowed for preference and safety interpretation."""

    current_user_request: str
    accepted_memory_facts: list[str] = Field(default_factory=list)
    accepted_memory_ids: list[str] = Field(default_factory=list)
    explicit_current_constraints: list[str] = Field(default_factory=list)


class PreferenceSafetyResult(ExpertResult):
    hard_constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    calorie_target: float | None = Field(default=None, gt=0)
    disliked_ingredients: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    dietary_pattern: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)


class NutritionTask(BaseModel):
    """Minimal context allowed for USDA verification and calculations."""

    requested_foods: list[str] = Field(default_factory=list)
    calorie_target: float | None = Field(default=None, gt=0)
    protein_goal: float | None = Field(default=None, ge=0)
    hard_constraints: list[str] = Field(default_factory=list)
    dietary_pattern: str | None = None
    foods_to_avoid: list[str] = Field(default_factory=list)
    preparation_assumptions: list[str] = Field(default_factory=list)
    current_user_request_summary: str


class NutritionResult(ExpertResult):
    """Flat, structured nutrition evidence returned to the Manager."""

    selected_food_descriptions: list[str] = Field(default_factory=list)
    portion_grams: list[float] = Field(default_factory=list)
    calories: float | None = Field(default=None, ge=0)
    protein_g: float | None = Field(default=None, ge=0)
    carbs_g: float | None = Field(default=None, ge=0)
    fat_g: float | None = Field(default=None, ge=0)
    target_calories: float | None = Field(default=None, gt=0)
    target_met: bool | None = None
    source: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class SpecialistExecution(BaseModel):
    specialist_name: str
    status: ExpertStatus
    latency_ms: float = Field(ge=0)
    timeout: bool = False
    fallback_used: bool = False
    retry_count: int = Field(default=0, ge=0, le=1)
    result_used: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    finding_count: int = Field(ge=0)
    limitation_count: int = Field(ge=0)
    result: PreferenceSafetyResult | NutritionResult


class ManagerRunResult(BaseModel):
    answer: str
    active_agent: Literal["CalorieChefManager"] = "CalorieChefManager"
    route: RouteName
    executions: list[SpecialistExecution] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    conflict_detected: bool = False
    trace_id: str | None = None
    observed_latency_ms: float = Field(ge=0)
    error: str | None = None


class ArchitectureMetrics(BaseModel):
    architecture: Literal["single_agent", "multi_agent"]
    hard_constraints_preserved: bool
    required_tool_evidence_present: bool
    final_meal_artifact_present: bool
    unsupported_exact_claims: bool
    partial_failure_honest: bool
    observed_latency_ms: float = Field(ge=0)
    generation_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    token_usage: dict[str, int] | None = None
    trace_clarity: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
