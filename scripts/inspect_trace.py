"""Print the latest local CalorieChef trace as a compact span tree."""

from __future__ import annotations

import argparse
import json

from observability import TRACE_PATH


def load_trace(trace_id: str | None = None) -> dict:
    if not TRACE_PATH.exists():
        raise SystemExit(f"No trace file exists at {TRACE_PATH}.")
    records = [json.loads(line) for line in TRACE_PATH.read_text(encoding="utf-8").splitlines() if line]
    if trace_id:
        for record in reversed(records):
            if record["trace_id"] == trace_id:
                return record
        raise SystemExit(f"Trace '{trace_id}' was not found.")
    if not records:
        raise SystemExit("The trace file is empty.")
    return records[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-id", help="Show a specific trace instead of the latest one.")
    args = parser.parse_args()
    record = load_trace(args.trace_id)
    print(
        f"TRACE {record['trace_id']} name={record['name']} status={record['status']} "
        f"latency_ms={record['latency_ms']}"
    )
    print(f"  metadata={json.dumps(record['metadata'], ensure_ascii=False, sort_keys=True)}")
    children: dict[str | None, list[dict]] = {}
    for span in record["spans"]:
        children.setdefault(span["parent_id"], []).append(span)

    def walk(parent_id: str | None, depth: int) -> None:
        for span in children.get(parent_id, []):
            attributes = json.dumps(span["attributes"], ensure_ascii=False, sort_keys=True)
            print(
                f"{'  ' * (depth + 1)}- {span['name']} status={span['status']} "
                f"latency_ms={span['latency_ms']} attributes={attributes}"
            )
            if span["error"]:
                print(f"{'  ' * (depth + 2)}error={span['error']}")
            walk(span["span_id"], depth + 1)

    walk(None, 0)
    print(f"  token_usage={record['token_usage']}")
    print(f"  cost={record['cost']}")


if __name__ == "__main__":
    main()
