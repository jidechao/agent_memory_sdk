"""Top-level SDK configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import RetrievalConfig


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
            index_max_lines=int(os.environ.get("MEMORY_INDEX_MAX_LINES", "200")),
            index_max_bytes=int(os.environ.get("MEMORY_INDEX_MAX_BYTES",
                                               str(25 * 1024))),
            stale_after_dreams=int(os.environ.get("MEMORY_STALE_AFTER_DREAMS",
                                                  "2")),
            max_tool_rounds=int(os.environ.get("MEMORY_MAX_TOOL_ROUNDS", "6")),
        )
