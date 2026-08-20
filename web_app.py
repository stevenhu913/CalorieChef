"""FastAPI protocol layer for the stable CalorieChef Agent Core."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_core import AgentAnswer, AgentCoreUnavailable, CalorieChefService


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "static" / "index.html"
MAX_MESSAGE_LENGTH = 4000
THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOGGER = logging.getLogger("caloriechef.web")


def request_timeout_seconds() -> float:
    """Return the bounded Web request timeout."""
    raw = os.getenv("CALORIECHEF_REQUEST_TIMEOUT_SECONDS", "120").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 120.0


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    thread_id: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be empty.")
        return value.strip()

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if not THREAD_ID_PATTERN.fullmatch(value):
            raise ValueError("thread_id must contain only letters, numbers, hyphens, or underscores.")
        return value


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    mode: str
    architecture: str
    memory_mode: str
    trace_id: str | None = None
    tools_called: list[str] = Field(default_factory=list)


def create_app(
    service: Any | None = None,
    *,
    manage_service_lifecycle: bool = True,
    timeout_seconds: float | None = None,
) -> FastAPI:
    """Create the Web app with injectable Agent Core for deterministic tests."""
    selected_service = service or CalorieChefService()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = selected_service
        if manage_service_lifecycle:
            await selected_service.start()
        try:
            yield
        finally:
            if manage_service_lifecycle:
                await selected_service.close()

    app = FastAPI(
        title="CalorieChef Web",
        version="1.0.0",
        description="Stable Single-Agent nutrition assistant Web service.",
        lifespan=lifespan,
    )
    app.state.service = selected_service
    bounded_timeout = timeout_seconds or request_timeout_seconds()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND)

    @app.get("/healthz")
    async def health(request: Request) -> JSONResponse:
        core = request.app.state.service
        payload: dict[str, Any] = {
            "status": "ok" if core.ready else "degraded",
            "service": "caloriechef",
            "architecture": "single",
            "model_backend": core.model_backend,
            "agent_ready": bool(core.ready),
            "memory_mode": core.memory_mode,
            "reason": None if core.ready else (core.readiness_error or "Agent Core is unavailable."),
        }
        return JSONResponse(status_code=200 if core.ready else 503, content=payload)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        core = request.app.state.service
        if not core.ready:
            raise HTTPException(status_code=503, detail="Agent Core is unavailable.")
        thread_id = payload.thread_id or f"thread_{uuid4().hex}"
        try:
            result: AgentAnswer = await asyncio.wait_for(
                core.answer(payload.message, thread_id),
                timeout=bounded_timeout,
            )
            return ChatResponse.model_validate(result.model_dump())
        except AgentCoreUnavailable:
            raise HTTPException(status_code=503, detail="Agent Core is unavailable.") from None
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Agent request timed out.") from None
        except Exception as exc:
            LOGGER.error("Agent Core request failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Agent request failed.") from None

    return app


app = create_app()
