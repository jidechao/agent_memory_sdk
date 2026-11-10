"""Agentic retrieval tools: let the host agent decide what to look up.

Instead of a fixed top-k injection, the agent calls tools on demand —
"the top layer is a directory, details are pulled when needed". Tools are
defined once and exported in OpenAI function-calling format, so any host
framework can register them verbatim; ToolExecutor executes them against
the four memory layers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .layers import NotesStore, TranscriptStore, WikiStore
from .lifecycle import Lifecycle
from .retrieval import RetrievalHit, ZvecMemoryIndex

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "跨会话记忆检索（笔记 + Wiki）。返回命中目录"
                "（id/层级/标题/相关度/摘要），需要全文再用 read_note 或 "
                "read_wiki_page 按需读取。适用于回忆规则、经验教训、已验证事实。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自然语言查询"},
                    "layer": {"type": "string", "enum": ["note", "wiki"],
                              "description": "可选，限定检索层"},
                    "top_k": {"type": "integer", "default": 5,
                              "description": "返回条数"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "按 id 读取一条记忆笔记的完整内容与元数据（验证状态/来源/反馈计数）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string", "description": "笔记 id"},
                },
                "required": ["note_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_wiki_page",
            "description": "按 slug 读取一页 Wiki 的完整内容（稳定、结构化的知识）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Wiki 页面 slug"},
                },
                "required": ["slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_wiki",
            "description": "列出全部 Wiki 页面目录（slug + 标题）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": (
                "在原始会话记录（L1 证据锚点）中做关键词检索。"
                "当记忆笔记与实际情况矛盾、需要核对原始证据时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "关键词"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "feedback_memory",
            "description": (
                "对一条记忆的使用效果反馈：它帮上忙（helpful=true）还是误导了你"
                "（helpful=false）。负面反馈会让该记忆停止被检索，并触发生命周期降级。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string",
                                  "description": "笔记 id 或 Wiki slug"},
                    "helpful": {"type": "boolean"},
                    "reason": {"type": "string", "description": "可选说明"},
                },
                "required": ["memory_id", "helpful"],
            },
        },
    },
]


MAX_TOP_K = 20


@dataclass
class ToolCounters:
    """Usage counters consumed by the health-metrics layer."""
    calls: dict[str, int] = field(default_factory=dict)

    def record(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1


class ToolExecutor:
    """Executes the memory tools against the four layers."""

    def __init__(self, notes: NotesStore, wiki: WikiStore,
                 transcripts: TranscriptStore, lifecycle: Lifecycle,
                 index: ZvecMemoryIndex | None = None,
                 counters: ToolCounters | None = None):
        self.notes = notes
        self.wiki = wiki
        self.transcripts = transcripts
        self.lifecycle = lifecycle
        self.index = index
        self.counters = counters or ToolCounters()

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def call(self, name: str, args: dict[str, Any]) -> str:
        self.counters.record(name)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name!r}"},
                              ensure_ascii=False)
        try:
            if "top_k" in args:   # clamp untrusted (model-supplied) bounds
                args = {**args,
                        "top_k": max(1, min(int(args["top_k"]), MAX_TOP_K))}
            return json.dumps(handler(**args), ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - tool errors go back to the LLM
            logger.warning("tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def _tool_search_memory(self, query: str, layer: str | None = None,
                            top_k: int = 5) -> list[dict]:
        layers = (layer,) if layer in ("note", "wiki") else ("note", "wiki")
        hits: list[RetrievalHit] = []
        if self.index:
            hits = self.index.search(query, top_k=top_k, layers=layers)
        else:
            hits = [RetrievalHit(ref_id=n.note_id, layer="note",
                                 title=n.title, score=0.0)
                    for n in self.notes.search(query, top_k=top_k)]
        out = []
        for h in hits:
            snippet = ""
            if h.layer == "note":
                for n in self.notes.list():
                    if n.note_id == h.ref_id:
                        if n.is_penalized:
                            continue
                        snippet = n.body[:200]
                        break
            else:
                page = self.wiki.get_page(h.ref_id)
                snippet = (page.body[:200] if page else "")
            out.append({"id": h.ref_id, "layer": h.layer, "title": h.title,
                        "score": round(h.score, 4), "snippet": snippet})
        return out

    def _tool_read_note(self, note_id: str) -> dict:
        for n in self.notes.list():
            if n.note_id == note_id:
                self.notes.mark_used(n.note_id)
                return {"note_id": n.note_id, "title": n.title, "body": n.body,
                        "type": n.note_type, "status": n.status,
                        "sources": list(n.sources), "uses": n.uses,
                        "helpful": n.helpful, "harmful": n.harmful}
        return {"error": f"note {note_id!r} not found"}

    def _tool_read_wiki_page(self, slug: str) -> dict:
        page = self.wiki.get_page(slug)
        if not page:
            return {"error": f"wiki page {slug!r} not found"}
        return {"slug": page.slug, "title": page.title, "body": page.body,
                "sources": list(page.sources)}

    def _tool_list_wiki(self) -> list[dict]:
        return [{"slug": p.slug, "title": p.title}
                for p in self.wiki.list_pages()]

    def _tool_search_transcripts(self, query: str, top_k: int = 10) -> list[dict]:
        """Most recent matching raw entries, newest first."""
        tokens = [t for t in re.findall(r"\w+", query) if len(t) > 1]
        matches = []
        sessions = sorted(self.transcripts.read_all().items(), reverse=True)
        for session_id, entries in sessions:
            for e in reversed(entries):
                if any(t in e["content"] for t in tokens):
                    matches.append({"session": session_id, "kind": e["kind"],
                                    "content": e["content"][:300]})
                    if len(matches) >= top_k:
                        return matches
        return matches

    def _tool_feedback_memory(self, memory_id: str, helpful: bool,
                              reason: str = "") -> dict:
        return self.lifecycle.feedback(memory_id, helpful, reason)
