"""Local, privacy-conscious structured tracing for CalorieChef."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import set_trace_processors, set_tracing_disabled
from agents.tracing import TracingProcessor


ROOT = Path(__file__).resolve().parent
TRACE_PATH = Path(
    os.getenv("CALORIECHEF_TRACE_PATH", str(ROOT / "traces" / "caloriechef_traces.jsonl"))
)
USDA_TOOLS = {"search_food", "get_food_nutrition"}
LOCAL_TOOLS = {"calculate_macro_calories", "calculate_meal_nutrition"}
SPECIALIST_TOOLS = {"preference_safety_specialist", "nutrition_specialist"}


def summarize_query(text: str) -> str:
    """Classify a request without storing its raw text."""
    lower = text.lower()
    macro_terms = sum(word in lower for word in ("protein", "carb", "fat"))
    if re.search(r"\d+(?:\.\d+)?\s*g\b", lower) and macro_terms >= 2:
        return "macro_calculation"
    if any(word in lower for word in ("chicken", "rice", "nutrition", "protein")) and any(
        word in lower for word in ("calorie", "nutrition", "meal", "lunch")
    ):
        return "nutrition_or_meal"
    if any(word in lower for word in ("meal", "lunch", "dinner", "breakfast", "recommend")):
        return "meal_recommendation"
    if any(word in lower for word in ("usually", "allergy", "allergic", "prefer", "target")):
        return "user_memory_statement"
    return "general_request"


def _duration_ms(span: Any) -> float | None:
    if not span.started_at or not span.ended_at:
        return None
    started = datetime.fromisoformat(span.started_at)
    ended = datetime.fromisoformat(span.ended_at)
    return round((ended - started).total_seconds() * 1000, 2)


def _redact_error(value: Any) -> str:
    """Keep a concise error while masking common credential shapes."""
    text = str(value or "")
    text = re.sub(r"(?i)(api[_-]?key|token|password)=([^\s&]+)", r"\1=<redacted>", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "sk-<redacted>", text)
    return text[:240]


def _safe_custom_attributes(data: dict[str, Any] | None) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return safe


def _span_record(span: Any) -> dict[str, Any]:
    data = span.span_data
    span_type = data.type
    name = getattr(data, "name", None) or span_type
    attributes: dict[str, Any] = {}

    if span_type == "custom":
        attributes = _safe_custom_attributes(getattr(data, "data", None))
    elif span_type == "function":
        tool_name = str(getattr(data, "name", "function"))
        attributes["tool_name"] = tool_name
        if tool_name in USDA_TOOLS:
            attributes["tool_backend"] = "usda_mcp"
        elif tool_name in LOCAL_TOOLS:
            attributes["tool_backend"] = "local_deterministic"
        elif tool_name in SPECIALIST_TOOLS:
            attributes["tool_backend"] = "specialist_agent"
        else:
            attributes["tool_backend"] = "other"
    elif span_type == "generation":
        attributes["model"] = getattr(data, "model", None)
        usage = getattr(data, "usage", None) or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if usage.get(key) is not None:
                attributes[key] = usage[key]
    elif span_type == "agent":
        attributes["agent_name"] = getattr(data, "name", None)
        attributes["available_tool_count"] = len(getattr(data, "tools", None) or [])

    error = span.error or None
    return {
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": name,
        "type": span_type,
        "status": "error" if error else "ok",
        "started_at": span.started_at,
        "ended_at": span.ended_at,
        "latency_ms": _duration_ms(span),
        "attributes": attributes,
        "error": _redact_error(error.get("message")) if isinstance(error, dict) else _redact_error(error) if error else None,
    }


class LocalTraceProcessor(TracingProcessor):
    """Write one sanitized JSON object per completed request trace."""

    def __init__(self, path: Path = TRACE_PATH):
        self.path = path
        self._active: dict[str, dict[str, Any]] = {}

    def on_trace_start(self, trace: Any) -> None:
        self._active[trace.trace_id] = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "metadata": _safe_custom_attributes(trace.metadata),
            "spans": [],
        }

    def on_trace_end(self, trace: Any) -> None:
        record = self._active.pop(trace.trace_id, None)
        if record is None:
            return
        spans = record["spans"]
        durations = [span["latency_ms"] for span in spans if span["latency_ms"] is not None]
        top_level = [span for span in spans if span["parent_id"] is None]
        record["status"] = "error" if any(span["status"] == "error" for span in spans) else "ok"
        record["latency_ms"] = round(sum(span["latency_ms"] or 0 for span in top_level), 2)
        record["span_count"] = len(spans)
        record["max_span_latency_ms"] = max(durations, default=None)
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        usage_seen = False
        for span in spans:
            if span["type"] != "generation":
                continue
            for key in usage:
                value = span["attributes"].get(key)
                if isinstance(value, (int, float)):
                    usage[key] += value
                    usage_seen = True
        record["token_usage"] = usage if usage_seen else None
        api_key = os.getenv("OPENAI_API_KEY", "")
        record["cost"] = "local_not_directly_billed" if api_key in {"", "ollama"} else "unavailable"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_span_start(self, span: Any) -> None:
        del span

    def on_span_end(self, span: Any) -> None:
        record = self._active.get(span.trace_id)
        if record is not None:
            record["spans"].append(_span_record(span))

    def force_flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def configure_local_tracing(path: Path = TRACE_PATH) -> None:
    """Disable remote trace upload and enable the local sanitized processor."""
    set_tracing_disabled(False)
    set_trace_processors([LocalTraceProcessor(path)])
