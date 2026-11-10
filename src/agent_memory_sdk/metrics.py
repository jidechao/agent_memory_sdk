"""Health metrics for the memory system.

The article names the real metrics of a memory system: not just hit rate,
but whether memories are still correct, whether they reduce repeated work,
maintenance cost, retrieval tokens, error propagation, and final task gain.

MemoryMetrics snapshots the four layers plus the token ledger after every
session / dream cycle, persists history.jsonl, and renders a health panel.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .layers import InjectedContext, NotesStore, WikiStore
from .llm import TokenLedger
from .tools import ToolCounters

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    root: Path
    ledger: TokenLedger
    notes: NotesStore
    wiki: WikiStore
    injected: InjectedContext
    tool_counters: ToolCounters | None = None
    _last_ledger_size: int = 0
    _last_dream_cost: dict = field(default_factory=dict)

    @property
    def history_path(self) -> Path:
        path = self.root / "metrics"
        path.mkdir(parents=True, exist_ok=True)
        return path / "history.jsonl"

    # ------------------------------------------------------------------ #
    def snapshot(self, kind: str, extra: dict | None = None,
                 persist: bool = True) -> dict:
        """Take a snapshot after a session or dream cycle and persist it."""
        notes = self.notes.list()
        pages = self.wiki.list_pages()
        verified = sum(1 for n in notes if n.status == "verified")
        refuted = sum(1 for n in notes if n.status == "refuted")
        index_text = self.injected.load()

        ledger_totals = self.ledger.totals()
        delta_entries = self.ledger.entries[self._last_ledger_size:]
        if persist:
            self._last_ledger_size = len(self.ledger.entries)
        delta: dict[str, dict[str, float]] = {}
        for e in delta_entries:
            t = delta.setdefault(e.tag, {"calls": 0, "prompt": 0,
                                         "completion": 0, "elapsed": 0.0})
            t["calls"] += 1
            t["prompt"] += e.prompt_tokens or 0
            t["completion"] += e.completion_tokens or 0
            t["elapsed"] += e.elapsed

        snap = {
            "ts": round(time.time(), 3),
            "kind": kind,                      # "session" | "dream"
            "layers": {
                "transcripts": self._dir_stats("transcripts"),
                "notes": {"count": len(notes),
                          "bytes": self._dir_stats("notes")["bytes"]},
                "wiki": {"count": len(pages),
                         "bytes": self._dir_stats("wiki")["bytes"]},
                "index": {"lines": len(index_text.splitlines()),
                          "bytes": len(index_text.encode("utf-8")),
                          "budget_lines_pct": round(
                              100 * len(index_text.splitlines())
                              / self.injected.max_lines, 1),
                          "budget_bytes_pct": round(
                              100 * len(index_text.encode("utf-8"))
                              / self.injected.max_bytes, 1)},
            },
            "quality": {
                "verified": verified,
                "refuted": refuted,
                "unverified": sum(1 for n in notes if n.status == "unverified"),
                "stale": sum(1 for n in notes if n.status == "stale"),
                "verified_ratio": (round(verified / (verified + refuted), 3)
                                   if verified + refuted else None),
                "helpful": sum(n.helpful for n in notes),
                "harmful": sum(n.harmful for n in notes),
            },
            "retrieval": {
                "tool_calls": dict(self.tool_counters.calls)
                if self.tool_counters else {},
            },
            "cost": {"since_last_snapshot": delta,
                     "cumulative": ledger_totals},
        }
        if extra:
            snap["extra"] = extra
        if persist:
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")
        return snap

    def _dir_stats(self, name: str) -> dict:
        d = self.root / name
        if not d.exists():
            return {"count": 0, "bytes": 0}
        files = [p for p in d.iterdir() if p.is_file()]
        return {"count": len(files),
                "bytes": sum(p.stat().st_size for p in files)}

    # ------------------------------------------------------------------ #
    def report(self) -> dict:
        history = self.read_history()
        # No write side effects in a read path: when there is no history yet,
        # compute a transient snapshot without persisting it.
        latest = (history[-1] if history
                  else self.snapshot("report", persist=False))
        return {
            "latest": latest,
            "trend": {
                "verified_ratio": [s["quality"]["verified_ratio"]
                                   for s in history],
                "notes_count": [s["layers"]["notes"]["count"] for s in history],
                "wiki_bytes": [s["layers"]["wiki"]["bytes"] for s in history],
                "dream_cost_calls": [
                    s["cost"]["since_last_snapshot"].get("verify", {}).get("calls", 0)
                    + s["cost"]["since_last_snapshot"].get("consolidate", {}).get("calls", 0)
                    + s["cost"]["since_last_snapshot"].get("wiki", {}).get("calls", 0)
                    + s["cost"]["since_last_snapshot"].get("index", {}).get("calls", 0)
                    for s in history if s["kind"] == "dream"],
            },
        }

    def read_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in
                self.history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    # ------------------------------------------------------------------ #
    def render(self) -> str:
        r = self.report()
        s = r["latest"]
        L, q, c = s["layers"], s["quality"], s["cost"]["cumulative"]
        total_calls = int(sum(v["calls"] for v in c.values()))
        total_tokens = int(sum(v["prompt"] + v["completion"]
                               for v in c.values()))
        lines = [
            "=" * 56,
            "记忆系统健康度面板",
            "-" * 56,
            f"  正确率(verified/(v+r)) : {q['verified_ratio']}",
            f"  反馈 helpful/harmful   : {q['helpful']} / {q['harmful']}",
            f"  笔记 v/u/r/stale       : {q['verified']}/{q['unverified']}"
            f"/{q['refuted']}/{q['stale']}",
            f"  顶层预算占用           : 行 {L['index']['budget_lines_pct']}%"
            f" / 字节 {L['index']['budget_bytes_pct']}%",
            f"  各层规模 (count/bytes) : notes {L['notes']['count']}"
            f"/{L['notes']['bytes']}  wiki {L['wiki']['count']}"
            f"/{L['wiki']['bytes']}  transcripts {L['transcripts']['count']}",
            f"  检索工具调用           : {s['retrieval']['tool_calls']}",
            f"  累计 LLM 成本          : {total_calls} 次调用, "
            f"{total_tokens} tokens",
            "-" * 56,
            "  分阶段成本 (calls/prompt+completion tokens):",
        ]
        for tag, v in sorted(c.items()):
            lines.append(f"    {tag:12s}: {int(v['calls']):3d} 次, "
                         f"{int(v['prompt'] + v['completion'])} tok")
        lines.append(f"  正确率趋势             : {r['trend']['verified_ratio']}")
        lines.append(f"  笔记规模趋势           : {r['trend']['notes_count']}")
        lines.append("=" * 56)
        return "\n".join(lines)
