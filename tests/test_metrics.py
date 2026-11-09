"""Health-metrics tests: token accounting, snapshots, reporting (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory_sdk.layers import (InjectedContext, NotesStore,
                                     TranscriptStore, WikiStore)
from agent_memory_sdk.llm import LLMClient, TokenLedger
from agent_memory_sdk.metrics import MemoryMetrics
from agent_memory_sdk.tools import ToolCounters


@pytest.fixture()
def metrics_env(tmp_path: Path):
    ledger = TokenLedger()
    llm = LLMClient(chat_fn=lambda s, u: "stub", ledger=ledger)
    notes = NotesStore(tmp_path)
    wiki = WikiStore(tmp_path)
    injected = InjectedContext(tmp_path)
    TranscriptStore(tmp_path).new_session()
    counters = ToolCounters()
    metrics = MemoryMetrics(tmp_path, ledger, notes, wiki, injected,
                            tool_counters=counters)
    return llm, ledger, notes, injected, counters, metrics


class TestTokenLedger:
    def test_calls_are_recorded_with_tags(self, metrics_env):
        llm, ledger, *_ = metrics_env
        llm.chat("s", "u", tag="agent")
        llm.chat("s", "u", tag="verify")
        llm.chat("s", "u", tag="verify")
        totals = ledger.totals()
        assert totals["agent"]["calls"] == 1
        assert totals["verify"]["calls"] == 2
        # callable backends report no token usage but must still count calls
        assert totals["verify"]["prompt"] == 0


class TestSnapshots:
    def test_snapshot_persists_and_has_expected_shape(self, metrics_env):
        _, _, notes, injected, counters, metrics = metrics_env
        notes.create("笔记A", "内容A", "rule")
        injected.write_index(["# index", "- [笔记A](notes/x.md) — hook"])
        counters.record("search_memory")

        snap = metrics.snapshot("session")
        assert snap["kind"] == "session"
        assert snap["layers"]["notes"]["count"] == 1
        assert snap["layers"]["index"]["lines"] == 2
        assert snap["layers"]["index"]["budget_lines_pct"] == 1.0
        assert snap["quality"]["unverified"] == 1
        assert snap["retrieval"]["tool_calls"] == {"search_memory": 1}
        assert metrics.history_path.exists()
        persisted = json.loads(
            metrics.history_path.read_text(encoding="utf-8").splitlines()[0])
        assert persisted["kind"] == "session"

    def test_delta_cost_between_snapshots(self, metrics_env):
        llm, _, _, _, _, metrics = metrics_env
        metrics.snapshot("session")
        llm.chat("s", "u", tag="verify")
        snap = metrics.snapshot("dream")
        delta = snap["cost"]["since_last_snapshot"]
        assert delta["verify"]["calls"] == 1
        assert "cumulative" in snap["cost"]


class TestReport:
    def test_report_and_render(self, metrics_env):
        llm, _, notes, _, _, metrics = metrics_env
        notes.create("笔记A", "内容", "rule")
        llm.chat("s", "u", tag="agent")
        metrics.snapshot("session")
        metrics.snapshot("dream")
        report = metrics.report()
        assert set(report) == {"latest", "trend"}
        assert len(report["trend"]["notes_count"]) == 2
        assert len(report["trend"]["dream_cost_calls"]) == 1
        rendered = metrics.render()
        assert "健康度面板" in rendered
        assert "分阶段成本" in rendered
