"""Verify the three automated public CalorieChef deployment checks."""

from __future__ import annotations

import argparse

import httpx


MACRO_REQUEST = (
    "I have 42 g of protein, 38 g of carbohydrates, and 15 g of fat. "
    "How many calories is that?"
)


def verify(base_url: str) -> bool:
    """Perform exactly root, health, and chat HTTP requests."""
    base_url = base_url.rstrip("/")
    checks: list[tuple[str, bool, str]] = []
    try:
        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            root = client.get(base_url + "/")
            checks.append(
                ("GET /", root.status_code == 200 and "text/html" in root.headers.get("content-type", ""), str(root.status_code))
            )
            health = client.get(base_url + "/healthz")
            health_json = health.json() if health.status_code == 200 else {}
            checks.append(
                ("GET /healthz", health.status_code == 200 and health_json.get("agent_ready") is True, str(health.status_code))
            )
            chat = client.post(base_url + "/chat", json={"message": MACRO_REQUEST})
            chat_json = chat.json() if chat.status_code == 200 else {}
            chat_ok = (
                chat.status_code == 200
                and bool(chat_json.get("answer"))
                and "455" in str(chat_json.get("answer"))
                and bool(chat_json.get("thread_id"))
                and chat_json.get("architecture") == "single"
                and "calculate_macro_calories" in chat_json.get("tools_called", [])
            )
            checks.append(("POST /chat", chat_ok, str(chat.status_code)))
    except Exception as exc:
        print(f"FAIL connection: {type(exc).__name__}")
        print("MANUAL browser interaction: PENDING")
        return False

    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name} ({detail})")
    print("MANUAL browser interaction: PENDING")
    return all(passed for _, passed, _ in checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.base_url) else 1)


if __name__ == "__main__":
    main()
