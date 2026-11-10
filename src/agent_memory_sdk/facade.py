"""MemorySystem: the SDK facade that wires all four layers together.

This is the single entry point for host agents:

    from agent_memory_sdk import MemorySystem, MemoryConfig

    mem = MemorySystem(root="./memory_data")          # or inject your own
    ctx = mem.session_start()                          # chat client / callable
    ... your agent loop, optionally with mem.tool_schemas() + mem.call_tool()
    mem.record("user", "...")
    mem.session_end()
    mem.dream()
    print(mem.metrics().render())
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from .agent import MemoryAgent
from .config import MemoryConfig
from .dreaming import Dreaming
from .embeddings import Embedder
from .layers import InjectedContext, NotesStore, TranscriptStore, WikiStore
from .lifecycle import Lifecycle
from .llm import LLMClient, TokenLedger
from .metrics import MemoryMetrics
from .retrieval import ZvecMemoryIndex
from .tools import ToolCounters, ToolExecutor

logger = logging.getLogger(__name__)


class MemorySystem:
    """Facade over the four layers + Dreaming + retrieval + metrics."""

    def __init__(self, root: str | Path | None = None,
                 config: MemoryConfig | None = None, *,
                 chat_client=None, chat_model: str | None = None,
                 chat_fn: Callable[[str, str], str] | None = None,
                 env_path: str | None = None,
                 wipe: bool = False):
        self.config = config or MemoryConfig.from_env(root)
        if wipe:
            self._guarded_wipe()
        self.config.root.mkdir(parents=True, exist_ok=True)

        # LLM backend: injected client / callable, or .env default
        self.ledger = TokenLedger()
        self.llm = LLMClient(env_path, client=chat_client, model=chat_model,
                             chat_fn=chat_fn, ledger=self.ledger)

        # Retrieval index (degrades to FTS when no embedder is available)
        rc = self.config.retrieval
        self.embedder = Embedder.try_create(
            rc.embedding_model, rc.ollama_base_url) if rc.mode != "fts" else None
        self.index = ZvecMemoryIndex(self.config.root, rc, self.embedder)

        # The four layers
        self.transcripts = TranscriptStore(self.config.root)
        self.notes = NotesStore(self.config.root, index=self.index)
        self.wiki = WikiStore(self.config.root, index=self.index)
        self.injected = InjectedContext(
            self.config.root,
            max_lines=self.config.index_max_lines,
            max_bytes=self.config.index_max_bytes)
        if self.index.needs_resync:
            self.index.sync(self.notes.list(), self.wiki.list_pages())

        # Engines
        self.lifecycle = Lifecycle(self.notes, self.wiki, self.transcripts)
        self.tool_counters = ToolCounters()
        self.tools = ToolExecutor(self.notes, self.wiki, self.transcripts,
                                  self.lifecycle, index=self.index,
                                  counters=self.tool_counters)
        self.metrics = MemoryMetrics(self.config.root, self.ledger,
                                     self.notes, self.wiki, self.injected,
                                     tool_counters=self.tool_counters)
        self._agent: MemoryAgent | None = None

    # ------------------------------------------------------------------ #
    # Session lifecycle (host integration points)
    # ------------------------------------------------------------------ #
    def session_start(self, mode: str = "agentic") -> str:
        """Begin a session; returns the injected context for the system prompt."""
        self._agent = MemoryAgent(
            self.llm, self.transcripts, self.notes, self.wiki, self.injected,
            tools=self.tools, mode=mode,  # type: ignore[arg-type]
            max_tool_rounds=self.config.max_tool_rounds)
        self.metrics.snapshot("session",
                              {"event": "start", "mode": mode,
                               "session_id": self._agent.session_id})
        return self.injected.load()

    @property
    def agent(self) -> MemoryAgent:
        if self._agent is None:
            self.session_start()
        assert self._agent is not None
        return self._agent

    def record(self, kind: str, content: str, meta: dict | None = None) -> None:
        """Append a raw entry to L1 (evidence anchor)."""
        self.transcripts.append(self.agent.session_id, kind, content, meta)

    def session_end(self) -> list[str]:
        """Reflect on the session and write memory notes (L2)."""
        created = self.agent.reflect_and_write_notes()
        self.metrics.snapshot("session",
                              {"event": "end", "notes_created": created})
        return created

    # ------------------------------------------------------------------ #
    # Tools (agentic retrieval)
    # ------------------------------------------------------------------ #
    def tool_schemas(self) -> list[dict]:
        return self.tools.schemas()

    def call_tool(self, name: str, args: dict) -> str:
        return self.tools.call(name, args)

    # ------------------------------------------------------------------ #
    # Offline maintenance + feedback + metrics
    # ------------------------------------------------------------------ #
    def dream(self) -> dict:
        """Run one Dreaming cycle (verify, consolidate, wiki, promote, demote)."""
        report = Dreaming(
            self.llm, self.transcripts, self.notes, self.wiki, self.injected,
            lifecycle=self.lifecycle,
            stale_after=self.config.stale_after_dreams,
        ).run()
        self.metrics.snapshot("dream", {"report": _summarize(report)})
        return report

    def feedback(self, memory_id: str, helpful: bool,
                 reason: str = "") -> dict:
        return self.lifecycle.feedback(memory_id, helpful, reason)

    def health(self) -> dict:
        return self.metrics.report()

    def _guarded_wipe(self) -> None:
        """wipe=True deletes the memory root — guard against catastrophic
        misconfiguration (root pointing at CWD, home, or a filesystem root)."""
        root = self.config.root.resolve()
        forbidden = {Path.cwd().resolve(), Path.home().resolve(),
                     Path(root.anchor)}
        if root in forbidden or len(root.parts) < 3:
            raise ValueError(
                f"refusing to wipe suspicious memory root: {root} "
                f"(must be a dedicated directory at least 2 levels deep)")
        if root.exists():
            shutil.rmtree(root)

    def close(self) -> None:
        self.index.close()
        if self.embedder is not None:
            self.embedder.close()


def _summarize(report: dict) -> dict:
    """Compact a dream report for metrics history (drop bulky reasons)."""
    return {k: (len(v) if isinstance(v, list) else v)
            for k, v in report.items()}
