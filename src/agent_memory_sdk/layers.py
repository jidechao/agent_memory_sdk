"""The four memory layers.

L1 TranscriptStore  - immutable JSONL raw records (evidence anchor, never overwritten)
L2 NotesStore       - markdown memory notes with JSON frontmatter (source/status/uses)
L3 WikiStore        - interlinked wiki pages ([[wiki-links]])
L4 InjectedContext  - MEMORY.md index, hard-capped at 200 lines / 25 KB
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import ZvecMemoryIndex

logger = logging.getLogger(__name__)

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


# --------------------------------------------------------------------------- #
# L1: Transcripts — the immutable evidence anchor
# --------------------------------------------------------------------------- #
class TranscriptStore:
    """Append-only raw record of every session: messages, tool calls, grades.

    Raw transcripts are NEVER rewritten by later maintenance — they are the
    evidence anchor that Dreaming uses to validate compressed memories.
    """

    def __init__(self, root: Path):
        self.root = root / "transcripts"
        self.root.mkdir(parents=True, exist_ok=True)

    def new_session(self) -> str:
        return uuid.uuid4().hex[:12]

    def append(self, session_id: str, kind: str, content: str,
               meta: dict | None = None) -> dict:
        entry = {
            "ts": round(time.time(), 3),
            "kind": kind,          # user | assistant | tool | grade | feedback
            "content": content,
            "meta": meta or {},
        }
        with (self.root / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def read_session(self, session_id: str) -> list[dict]:
        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def read_all(self) -> dict[str, list[dict]]:
        return {p.stem: self.read_session(p.stem)
                for p in sorted(self.root.glob("*.jsonl"))}


# --------------------------------------------------------------------------- #
# L2: Memory Notes — actively written, compressed working material
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemoryNote:
    """One memory note. Immutable value object; updates go through replace().

    Lifecycle: unverified -> verified | refuted | stale -> (deleted)
    Promotion path: note -> wiki -> injected index; demotion runs in reverse.
    """
    note_id: str
    title: str
    body: str
    note_type: str = "rule"            # rule | failure | fact | plan
    status: str = "unverified"         # unverified | verified | refuted | stale
    sources: tuple[str, ...] = ()      # transcript session ids (traceability)
    uses: int = 0                      # how often it was retrieved
    helpful: int = 0                   # positive usage feedback
    harmful: int = 0                   # negative usage feedback
    idle_dreams: int = 0               # consecutive dream cycles without use
    uses_snapshot: int = 0             # uses at the last dream cycle
    updated: float = field(default_factory=time.time)

    @property
    def is_penalized(self) -> bool:
        """More negative than positive usage feedback — demoted at retrieval."""
        return self.harmful > self.helpful

    def to_markdown(self) -> str:
        fm = {
            "id": self.note_id, "title": self.title, "type": self.note_type,
            "status": self.status, "sources": list(self.sources),
            "uses": self.uses, "helpful": self.helpful,
            "harmful": self.harmful, "idle_dreams": self.idle_dreams,
            "uses_snapshot": self.uses_snapshot, "updated": self.updated,
        }
        return f"---\n{json.dumps(fm, ensure_ascii=False, indent=2)}\n---\n{self.body}\n"

    @staticmethod
    def from_markdown(text: str) -> "MemoryNote":
        m = FRONTMATTER_RE.match(text)
        if not m:
            raise ValueError("note file missing frontmatter")
        fm = json.loads(m.group(1))
        return MemoryNote(
            note_id=fm["id"], title=fm["title"], body=text[m.end():].strip(),
            note_type=fm.get("type", "rule"), status=fm.get("status", "unverified"),
            sources=tuple(fm.get("sources", [])), uses=fm.get("uses", 0),
            helpful=fm.get("helpful", 0), harmful=fm.get("harmful", 0),
            idle_dreams=fm.get("idle_dreams", 0),
            uses_snapshot=fm.get("uses_snapshot", 0),
            updated=fm.get("updated", 0.0),
        )


class NotesStore:
    def __init__(self, root: Path, index: "ZvecMemoryIndex | None" = None):
        self.root = root / "notes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index

    def _path(self, note_id: str) -> Path:
        return self.root / f"{note_id}.md"

    def create(self, title: str, body: str, note_type: str = "rule",
               sources: tuple[str, ...] = ()) -> MemoryNote:
        note = MemoryNote(note_id=uuid.uuid4().hex[:8], title=title, body=body,
                          note_type=note_type, sources=sources)
        self._path(note.note_id).write_text(note.to_markdown(), encoding="utf-8")
        if self.index:
            self.index.upsert_note(note)
        logger.info("note created: %s (%s)", note.note_id, title)
        return note

    def save(self, note: MemoryNote) -> MemoryNote:
        note = replace(note, updated=time.time())
        self._path(note.note_id).write_text(note.to_markdown(), encoding="utf-8")
        if self.index:
            self.index.upsert_note(note)
        return note

    def delete(self, note_id: str) -> None:
        self._path(note_id).unlink(missing_ok=True)
        if self.index:
            self.index.delete_note(note_id)

    def list(self) -> list[MemoryNote]:
        return [MemoryNote.from_markdown(p.read_text(encoding="utf-8"))
                for p in sorted(self.root.glob("*.md"))]

    def mark_used(self, note_id: str) -> None:
        for n in self.list():
            if n.note_id == note_id:
                self.save(replace(n, uses=n.uses + 1))

    def search(self, query: str, top_k: int = 5) -> list[MemoryNote]:
        """Semantic/hybrid retrieval via zvec when an index is attached.

        Notes with more negative than positive usage feedback are excluded
        immediately — the negative-feedback loop closes at retrieval time.
        """
        if self.index:
            hits = self.index.search(query, top_k=top_k * 2, layers=("note",))
            by_id = {n.note_id: n for n in self.list()}
            notes = [by_id[h.ref_id] for h in hits
                     if h.ref_id in by_id and not by_id[h.ref_id].is_penalized]
            return notes[:top_k]
        return self._search_keywords(query, top_k)

    def _search_keywords(self, query: str, top_k: int = 5) -> list[MemoryNote]:
        """Legacy lexical fallback (used only when no index is attached)."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for n in self.list():
            if n.is_penalized:
                continue
            text = f"{n.title} {n.body}".lower()
            score = sum(text.count(t) for t in q_tokens if len(t) > 1)
            if score:
                scored.append((score, n))
        scored.sort(key=lambda x: (-x[0], -x[1].uses))
        return [n for _, n in scored[:top_k]]


# --------------------------------------------------------------------------- #
# L3: Wiki — stable, structured, interlinked knowledge
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WikiPage:
    slug: str
    title: str
    body: str
    sources: tuple[str, ...] = ()      # note ids this page was built from
    updated: float = field(default_factory=time.time)


class WikiStore:
    def __init__(self, root: Path, index: "ZvecMemoryIndex | None" = None):
        self.root = root / "wiki"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = index

    def save_page(self, page: WikiPage) -> None:
        fm = {"title": page.title, "sources": list(page.sources),
              "updated": page.updated}
        text = f"---\n{json.dumps(fm, ensure_ascii=False, indent=2)}\n---\n{page.body}\n"
        (self.root / f"{page.slug}.md").write_text(text, encoding="utf-8")
        if self.index:
            self.index.upsert_page(page)

    def delete_page(self, slug: str) -> None:
        (self.root / f"{slug}.md").unlink(missing_ok=True)
        if self.index:
            self.index.delete_page(slug)

    def get_page(self, slug: str) -> WikiPage | None:
        path = self.root / f"{slug}.md"
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        fm = json.loads(m.group(1)) if m else {}
        return WikiPage(slug=slug, title=fm.get("title", slug),
                        body=text[m.end():].strip() if m else text,
                        sources=tuple(fm.get("sources", [])),
                        updated=fm.get("updated", 0.0))

    def list_pages(self) -> list[WikiPage]:
        return [p for p in (self.get_page(f.stem)
                            for f in sorted(self.root.glob("*.md"))) if p]

    def broken_links(self) -> dict[str, list[str]]:
        """{page_slug: [linked slugs that do not exist]}"""
        existing = {p.slug for p in self.list_pages()}
        broken: dict[str, list[str]] = {}
        for p in self.list_pages():
            bad = [s for s in WIKILINK_RE.findall(p.body) if s not in existing]
            if bad:
                broken[p.slug] = bad
        return broken

    def retarget_links(self, slug: str, mapping: dict[str, str]) -> None:
        """Fix broken links by retargeting them to existing pages."""
        page = self.get_page(slug)
        if not page:
            return
        body = page.body
        for old, new in mapping.items():
            body = body.replace(f"[[{old}]]", f"[[{new}]]")
        self.save_page(replace(page, body=body, updated=time.time()))


# --------------------------------------------------------------------------- #
# L4: Injected Context — a tiny index, not a warehouse
# --------------------------------------------------------------------------- #
class InjectedContext:
    """MEMORY.md: the only layer auto-loaded at session start.

    Hard budget (mirrors Claude Code's auto-memory): max 200 lines / 25 KB.
    The top level is a directory of pointers, not a warehouse — details stay
    in the lower layers and are pulled on demand.
    """

    MAX_LINES = 200
    MAX_BYTES = 25 * 1024

    def __init__(self, root: Path):
        self.path = root / "MEMORY.md"

    def write_index(self, lines: list[str]) -> None:
        text = "\n".join(lines).strip() + "\n"
        # Enforce the budget exactly like the real design does.
        raw = text.encode("utf-8")[: self.MAX_BYTES]
        text = raw.decode("utf-8", errors="ignore")
        text = "\n".join(text.splitlines()[: self.MAX_LINES]) + "\n"
        self.path.write_text(text, encoding="utf-8")
        logger.info("injected index written: %d lines, %d bytes",
                    len(text.splitlines()), len(text.encode("utf-8")))

    def load(self) -> str:
        if not self.path.exists():
            return ""
        text = self.path.read_text(encoding="utf-8")
        raw = text.encode("utf-8")[: self.MAX_BYTES]
        text = raw.decode("utf-8", errors="ignore")
        return "\n".join(text.splitlines()[: self.MAX_LINES])
