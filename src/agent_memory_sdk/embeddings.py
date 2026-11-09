"""Local embedding provider for the dense-retrieval path.

DeepSeek's API offers no embedding endpoint, so dense vectors come from a
locally running Ollama instance (default model: bge-m3 — 1024-dim,
multilingual, strong on Chinese). The embedder degrades gracefully: if
Ollama is unreachable or the model is missing, callers fall back to
FTS-only retrieval.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-m3"
DEFAULT_BASE_URL = "http://localhost:11434"


class Embedder:
    """Minimal client for Ollama's /api/embed endpoint."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        # trust_env=False: never route the local embedding service through an
        # HTTP(S)_PROXY — a proxy that can't reach Ollama breaks dense retrieval.
        self._client = httpx.Client(timeout=timeout, trust_env=False)
        self.dim: int = len(self.embed(["dim probe"])[0])  # fails fast if down
        logger.info("embedder ready: %s @ %s (dim=%d)",
                    model_name, self.base_url, self.dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    @staticmethod
    def try_create(model_name: str | None = None,
                   base_url: str | None = None) -> "Embedder | None":
        """Create an embedder, returning None (with a warning) on any failure."""
        name = model_name or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        try:
            return Embedder(name, url)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash retrieval
            logger.warning("dense retrieval disabled, embedder unavailable: %s", exc)
            return None
