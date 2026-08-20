"""Verify local trace persistence, error status, and credential redaction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agents import custom_span, trace

from observability import configure_local_tracing


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "trace.jsonl"
        configure_local_tracing(path)
        with trace(
            "caloriechef_request",
            metadata={"query_type": "error_test", "raw_user_text_recorded": False},
        ):
            with custom_span(
                "request_error",
                data={"error_type": "TimeoutError", "recovered": True},
            ) as span:
                span.set_error(
                    {
                        "message": "Timeout token=secret-value sk-secret-value",
                        "data": {},
                    }
                )
        record = json.loads(path.read_text(encoding="utf-8").strip())
        failed = next(span for span in record["spans"] if span["name"] == "request_error")
        assert record["status"] == "error"
        assert failed["status"] == "error"
        assert failed["attributes"]["error_type"] == "TimeoutError"
        assert failed["attributes"]["recovered"] is True
        assert "secret-value" not in failed["error"]
        assert "<redacted>" in failed["error"]
        print("PASS: local trace persisted a redacted, attributable error span")


if __name__ == "__main__":
    main()
