"""Extract sanitized evidence from structured local traces."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.models import TraceEvidence


def extract_trace(path: Path, trace_id: str) -> TraceEvidence:
    """Read function order, generations, custom spans, status, and errors."""
    if not path.exists():
        return TraceEvidence(trace_id=trace_id)
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        if record.get("trace_id") != trace_id:
            continue
        spans = record.get("spans", [])
        return TraceEvidence(
            trace_id=trace_id,
            status=record.get("status", "unavailable"),
            tool_names=[span["name"] for span in spans if span.get("type") == "function"],
            generation_count=sum(span.get("type") == "generation" for span in spans),
            custom_spans={
                span["name"]: span.get("attributes", {})
                for span in spans
                if span.get("type") == "custom"
            },
            token_usage=record.get("token_usage"),
            errors=[span["error"] for span in spans if span.get("error")],
        )
    return TraceEvidence(trace_id=trace_id)
