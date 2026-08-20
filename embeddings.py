"""Central embedding configuration for CalorieChef long-term memory."""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence


EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_BACKEND = f"ollama:{EMBEDDING_MODEL}"


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding backend is unavailable."""


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed documents or queries with the single configured local model."""
    vectors: list[list[float]] = []

    for text in texts:
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            data=json.dumps(
                {
                    "model": EMBEDDING_MODEL,
                    "prompt": text,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise EmbeddingError(
                f"Ollama embedding failed: {type(exc).__name__}: {exc}"
            ) from exc

        vector = payload.get("embedding")
        if not vector:
            raise EmbeddingError(
                f"Ollama returned no embedding for model {EMBEDDING_MODEL}."
            )
        vectors.append(vector)

    return vectors
