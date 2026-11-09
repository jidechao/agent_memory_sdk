"""Lifecycle engine: usage feedback and demotion.

Closes the memory lifecycle into a full loop:

    note(unverified -> verified/refuted) -> wiki -> injected index
      ^                                                    |
      |____________ demotion runs in reverse ______________|
             index -> wiki -> note -> stale -> deleted

Two mechanisms live here:

  feedback()  - the host agent (or Dreaming, or a human) reports whether a
                memory actually helped. Feedback is written to L1 (audit
                trail) and updates helpful/harmful counters; notes with
                harmful > helpful stop being retrieved immediately.

  demote()    - runs inside the Dreaming cycle. Moves memories DOWN the
                pyramid: orphaned wiki pages fall back to notes, idle
                unverified notes go stale then get deleted, and harmful
                refuted notes are removed. Every action is logged to L1.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from .layers import MemoryNote, NotesStore, TranscriptStore, WikiStore

logger = logging.getLogger(__name__)

LIFECYCLE_SESSION = "memory-lifecycle"     # pseudo-session for L1 audit entries


class Lifecycle:
    def __init__(self, notes: NotesStore, wiki: WikiStore,
                 transcripts: TranscriptStore):
        self.notes = notes
        self.wiki = wiki
        self.transcripts = transcripts

    # ------------------------------------------------------------------ #
    # Usage feedback (the negative-feedback loop)
    # ------------------------------------------------------------------ #
    def feedback(self, memory_id: str, helpful: bool,
                 reason: str = "") -> dict:
        """Record usage feedback for a note id or wiki slug.

        Feedback on a wiki page propagates to its source notes — the wiki is
        derived knowledge, notes carry the lifecycle.
        """
        targets = self._resolve(memory_id)
        if not targets:
            return {"ok": False, "error": f"memory {memory_id!r} not found"}

        updated = []
        for note in targets:
            note = replace(
                note,
                helpful=note.helpful + (1 if helpful else 0),
                harmful=note.harmful + (0 if helpful else 1),
            )
            self.notes.save(note)
            updated.append({"note_id": note.note_id, "title": note.title,
                            "helpful": note.helpful, "harmful": note.harmful,
                            "penalized": note.is_penalized})

        self.transcripts.append(
            LIFECYCLE_SESSION, "memory_feedback",
            f"feedback on {memory_id}: helpful={helpful}"
            + (f", reason: {reason}" if reason else ""),
            meta={"memory_id": memory_id, "helpful": helpful, "reason": reason},
        )
        logger.info("feedback recorded: %s helpful=%s -> %d note(s)",
                    memory_id, helpful, len(updated))
        return {"ok": True, "updated": updated}

    def _resolve(self, memory_id: str) -> list[MemoryNote]:
        for n in self.notes.list():
            if n.note_id == memory_id:
                return [n]
        page = self.wiki.get_page(memory_id)
        if page:
            by_id = {n.note_id: n for n in self.notes.list()}
            return [by_id[s] for s in page.sources if s in by_id]
        return []

    # ------------------------------------------------------------------ #
    # Demotion (runs inside Dreaming)
    # ------------------------------------------------------------------ #
    def demote(self, stale_after: int = 2) -> list[dict]:
        """Move memories down the pyramid. Returns an action report."""
        report: list[dict] = []
        report.extend(self._demote_orphaned_wiki())
        report.extend(self._demote_notes(stale_after))
        return report

    def _demote_orphaned_wiki(self) -> list[dict]:
        """Wiki pages whose source notes are all gone fall back to notes."""
        existing = {n.note_id for n in self.notes.list()}
        report = []
        for page in self.wiki.list_pages():
            if not page.sources or any(s in existing for s in page.sources):
                continue
            note = self.notes.create(
                title=f"[降级] {page.title}",
                body=page.body,
                note_type="fact",
                sources=(),
            )
            self.notes.save(replace(note, status="unverified"))
            self.wiki.delete_page(page.slug)
            self._log("wiki_demoted",
                      f"wiki page {page.slug} demoted to note {note.note_id} "
                      f"(all source notes lost)")
            report.append({"action": "wiki->note", "page": page.slug,
                           "note_id": note.note_id})
        return report

    def _demote_notes(self, stale_after: int) -> list[dict]:
        """Harmful refuted notes are deleted; idle unverified ones go stale."""
        report = []
        for note in self.notes.list():
            if note.harmful > note.helpful and note.status == "refuted":
                self.notes.delete(note.note_id)
                self._log("note_deleted",
                          f"refuted+penalized note {note.note_id} "
                          f"({note.title}) deleted")
                report.append({"action": "note->deleted",
                               "note": note.title, "reason": "refuted+penalized"})
                continue

            idle = note.uses == note.uses_snapshot
            idle_dreams = note.idle_dreams + 1 if idle else 0
            updated = replace(note, idle_dreams=idle_dreams,
                              uses_snapshot=note.uses)

            if (note.status == "stale" and idle_dreams > stale_after) or \
               (note.status == "unverified" and idle_dreams > stale_after + 1):
                self.notes.delete(note.note_id)
                self._log("note_deleted",
                          f"stale note {note.note_id} ({note.title}) deleted "
                          f"after {idle_dreams} idle dream cycles")
                report.append({"action": "note->deleted", "note": note.title,
                               "reason": f"idle {idle_dreams} cycles"})
            elif note.status == "unverified" and idle_dreams >= stale_after:
                self.notes.save(replace(updated, status="stale"))
                self._log("note_staled",
                          f"note {note.note_id} ({note.title}) marked stale")
                report.append({"action": "note->stale", "note": note.title})
            else:
                self.notes.save(updated)
        return report

    def _log(self, kind: str, content: str) -> None:
        self.transcripts.append(LIFECYCLE_SESSION, kind, content)
        logger.info("lifecycle: %s", content)
