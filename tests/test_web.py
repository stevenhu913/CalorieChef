"""Deterministic Web and deployment-policy tests."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent_core import AgentAnswer, CalorieChefService
from backend_config import resolve_backend_policy
from web_app import create_app


ROOT = Path(__file__).resolve().parents[1]


class FakeService:
    def __init__(self, *, ready: bool = True, behavior: str = "ok") -> None:
        self.ready = ready
        self.behavior = behavior
        self.readiness_error = None if ready else "configuration missing"
        self.model_backend = "hosted"
        self.memory_mode = "limited"
        self.answer_calls = 0

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def answer(self, message: str, thread_id: str) -> AgentAnswer:
        self.answer_calls += 1
        if self.behavior == "slow":
            await asyncio.sleep(0.05)
        if self.behavior == "error":
            raise RuntimeError("OPENAI_API_KEY=secret-value")
        return AgentAnswer(
            answer="The total is 455 calories.",
            thread_id=thread_id,
            mode="hosted",
            memory_mode="limited",
            trace_id="trace_test",
            tools_called=["calculate_macro_calories"],
        )


def client_for(service: FakeService, *, timeout: float = 1.0) -> TestClient:
    return TestClient(
        create_app(
            service,
            manage_service_lifecycle=False,
            timeout_seconds=timeout,
        )
    )


class WebTests(unittest.TestCase):
    def test_health_returns_200_without_calling_agent(self):
        service = FakeService()
        with client_for(service) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIsNone(response.json()["reason"])
        self.assertEqual(service.answer_calls, 0)

    def test_health_degraded_returns_503_with_sanitized_metadata(self):
        service = FakeService(ready=False)
        with client_for(service) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {
                "status": "degraded",
                "service": "caloriechef",
                "architecture": "single",
                "model_backend": "hosted",
                "agent_ready": False,
                "memory_mode": "limited",
                "reason": "configuration missing",
            },
        )
        self.assertEqual(service.answer_calls, 0)

    def test_chat_accepts_valid_message_and_returns_answer(self):
        with client_for(FakeService()) as client:
            response = client.post("/chat", json={"message": "Calculate my macros."})
        self.assertEqual(response.status_code, 200)
        self.assertIn("455", response.json()["answer"])
        self.assertIn("calculate_macro_calories", response.json()["tools_called"])

    def test_missing_thread_id_generates_one(self):
        with client_for(FakeService()) as client:
            thread_id = client.post("/chat", json={"message": "Hello"}).json()["thread_id"]
        self.assertRegex(thread_id, r"^thread_[0-9a-f]{32}$")

    def test_supplied_valid_thread_id_is_preserved(self):
        with client_for(FakeService()) as client:
            response = client.post(
                "/chat",
                json={"message": "Hello", "thread_id": "valid-thread"},
            )
        self.assertEqual(response.json()["thread_id"], "valid-thread")

    def test_empty_message_is_rejected(self):
        with client_for(FakeService()) as client:
            response = client.post("/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_oversized_message_is_rejected(self):
        with client_for(FakeService()) as client:
            response = client.post("/chat", json={"message": "x" * 4001})
        self.assertEqual(response.status_code, 422)

    def test_path_like_thread_id_is_rejected(self):
        with client_for(FakeService()) as client:
            response = client.post(
                "/chat",
                json={"message": "Hello", "thread_id": "../private"},
            )
        self.assertEqual(response.status_code, 422)

    def test_agent_unavailable_returns_503(self):
        with client_for(FakeService(ready=False)) as client:
            response = client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 503)

    def test_request_timeout_maps_to_504(self):
        with client_for(FakeService(behavior="slow"), timeout=0.001) as client:
            response = client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 504)

    def test_internal_exception_is_sanitized(self):
        with client_for(FakeService(behavior="error")) as client:
            response = client.post("/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Agent request failed."})
        self.assertNotIn("secret-value", response.text)

    def test_root_returns_html(self):
        with client_for(FakeService()) as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_frontend_uses_same_origin_chat(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertIn("fetch('/chat'", html)
        self.assertNotIn("localhost", html)
        self.assertNotIn("127.0.0.1", html)

    def test_frontend_uses_safe_markdown_subset_renderer(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertIn("function renderAnswer(markdownText)", html)
        self.assertIn("renderAnswer(data.answer)", html)
        self.assertIn("document.createElement('h3')", html)
        self.assertIn("document.createTextNode", html)
        self.assertIn("strong.textContent =", html)
        self.assertNotIn("answer.textContent = data.answer", html)

    def test_frontend_never_inserts_model_html(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertNotRegex(html, r"\.innerHTML\s*=")
        self.assertNotIn("insertAdjacentHTML", html)
        self.assertNotIn("DOMParser", html)
        self.assertIn("String(markdownText)", html)

    def test_frontend_badges_use_only_fixed_safe_labels(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertIn("renderBadges(data.tools_called, data.answer)", html)
        self.assertIn("labels.push('USDA verified')", html)
        self.assertIn("labels.push('Deterministic calculation')", html)
        self.assertIn("labels.push('Expanded portion range')", html)
        self.assertNotIn("JSON.stringify(data.tools_called)", html)
        self.assertNotIn("tool_output", html)
        self.assertNotIn("tool_arguments", html)

    def test_frontend_prevents_empty_and_duplicate_submission(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertIn("if (!message || requestInFlight)", html)
        self.assertIn("requestInFlight = true", html)
        self.assertIn("requestInFlight = false", html)
        self.assertIn("send.disabled = true", html)
        self.assertIn("send.disabled = false", html)
        self.assertEqual(html.count("form.addEventListener('submit'"), 1)
        self.assertEqual(html.count("fetch('/chat'"), 1)
        self.assertIn("finally", html)

    def test_frontend_replaces_one_answer_per_completed_request(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertIn("answer.replaceChildren()", html)
        self.assertIn("answer.classList.remove('visible')", html)
        self.assertEqual(html.count("renderAnswer(data.answer)"), 1)
        self.assertNotIn("answer.append(data.answer)", html)

    def test_env_remains_ignored(self):
        ignored = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn(".env", ignored)

    def test_default_public_architecture_is_single(self):
        with client_for(FakeService()) as client:
            response = client.get("/healthz")
        self.assertEqual(response.json()["architecture"], "single")
        self.assertNotIn("multi_experimental", (ROOT / "static" / "index.html").read_text())

    def test_long_term_memory_disabled_does_not_initialize_chroma(self):
        service = CalorieChefService(memory_enabled=False)
        with patch.dict(sys.modules, {"long_term_memory": None}):
            evidence, count = asyncio.run(service._memory_context("hello", "thread", ""))
        self.assertEqual(count, 0)
        self.assertIn("disabled", evidence)

    def test_hosted_policy_never_probes_or_configures_localhost(self):
        probed = False

        def probe() -> bool:
            nonlocal probed
            probed = True
            return True

        policy = resolve_backend_policy(
            {
                "CALORIECHEF_MODEL_BACKEND": "hosted",
                "CALORIECHEF_HOSTED_MODEL": "established-model",
                "OPENAI_API_KEY": "configured",
            },
            ollama_probe=probe,
        )
        self.assertTrue(policy.ready)
        self.assertFalse(probed)
        self.assertIsNone(policy.base_url)

    def test_hosted_missing_configuration_is_not_ready(self):
        policy = resolve_backend_policy({"CALORIECHEF_MODEL_BACKEND": "hosted"})
        self.assertFalse(policy.ready)
        self.assertNotIn("localhost", policy.error or "")

    def test_health_response_contains_no_secret(self):
        service = FakeService(ready=False)
        service.readiness_error = "Hosted model credential is missing."
        with client_for(service) as client:
            response = client.get("/healthz")
        self.assertNotIn("API_KEY", response.text)
        self.assertNotIn("secret", response.text.lower())

    def test_agent_core_does_not_import_fastapi(self):
        source = (ROOT / "agent_core.py").read_text()
        self.assertNotIn("from fastapi", source)
        self.assertNotIn("import fastapi", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
