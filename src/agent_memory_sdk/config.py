"""Top-level SDK configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryConfig:
    """Everything a MemorySystem needs, in one immutable object."""

    root: Path = Path("./memory_data")
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # L4 injected-index budget (mirrors Claude Code's auto-memory limits)
    index_max_lines: int = 200
    index_max_bytes: int = 25 * 1024
    # Lifecycle: unverified notes unused for this many dream cycles go stale;
    # one more idle cycle after that deletes them.
    stale_after_dreams: int = 2
    # Agentic tool loop: max tool round-trips per question
    max_tool_rounds: int = 6

    @staticmethod
    def from_env(root: str | Path | None = None) -> "MemoryConfig":
        return MemoryConfig(
            root=Path(root or os.environ.get("MEMORY_ROOT", "./memory_data")),
            retrieval=RetrievalConfig.from_env(),
            index_max_lines=_int_env("MEMORY_INDEX_MAX_LINES", 200),
            index_max_bytes=_int_env("MEMORY_INDEX_MAX_BYTES", 25 * 1024),
            stale_after_dreams=_int_env("MEMORY_STALE_AFTER_DREAMS", 2),
            max_tool_rounds=_int_env("MEMORY_MAX_TOOL_ROUNDS", 6),
        )


def _int_env(name: str, default: int) -> int:
    """Env var to int with a sane fallback instead of a bare ValueError."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, raw, default)
        return default
