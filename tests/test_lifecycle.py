"""Lifecycle tests: feedback loop and demotion rules (no LLM required)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory_sdk.layers import (NotesStore, TranscriptStore, WikiPage,
                                     WikiStore)
from agent_memory_sdk.lifecycle import LIFECYCLE_SESSION, Lifecycle


@pytest.fixture()
def stores(tmp_path: Path):
    transcripts = TranscriptStore(tmp_path)
    notes = NotesStore(tmp_path)
    wiki = WikiStore(tmp_path)
    return transcripts, notes, wiki, Lifecycle(notes, wiki, transcripts)


class TestFeedback:
    def test_feedback_updates_counters_and_audit_log(self, stores):
        transcripts, notes, _, lc = stores
        note = notes.create("部署窗口", "每周四凌晨", "rule")
        result = lc.feedback(note.note_id, helpful=False, reason="误导了我")
        assert result["ok"] and result["updated"][0]["harmful"] == 1
        reloaded = [n for n in notes.list() if n.note_id == note.note_id][0]
        assert reloaded.harmful == 1 and reloaded.helpful == 0
        audit = transcripts.read_session(LIFECYCLE_SESSION)
        assert any(e["kind"] == "memory_feedback" for e in audit)

    def test_negative_feedback_excludes_note_from_search(self, stores):
        _, notes, _, lc = stores
        good = notes.create("回滚超时", "15 分钟超时", "rule")
        bad = notes.create("回滚窗口", "回滚随时可以", "rule")
        assert {n.title for n in notes.search("回滚")} == {"回滚超时", "回滚窗口"}
        lc.feedback(bad.note_id, helpful=False)
        assert {n.title for n in notes.search("回滚")} == {good.title}

    def test_wiki_feedback_propagates_to_source_notes(self, stores):
        _, notes, wiki, lc = stores
        src = notes.create("源笔记", "内容", "rule")
        wiki.save_page(WikiPage(slug="p1", title="页面", body="正文",
                                sources=(src.note_id,)))
        result = lc.feedback("p1", helpful=False)
        assert result["ok"]
        reloaded = [n for n in notes.list() if n.note_id == src.note_id][0]
        assert reloaded.harmful == 1

    def test_unknown_memory_id(self, stores):
        _, _, _, lc = stores
        assert lc.feedback("nope", helpful=True)["ok"] is False


class TestDemotion:
    def test_orphaned_wiki_falls_back_to_note(self, stores):
        transcripts, notes, wiki, lc = stores
        src = notes.create("将被删除的笔记", "内容", "rule")
        wiki.save_page(WikiPage(slug="orphan", title="孤儿页",
                                body="维基正文", sources=(src.note_id,)))
        notes.delete(src.note_id)
        report = lc.demote(stale_after=2)
        assert any(a["action"] == "wiki->note" and a["page"] == "orphan"
                   for a in report)
        assert wiki.get_page("orphan") is None
        demoted = [n for n in notes.list() if n.title.startswith("[降级]")]
        assert demoted and demoted[0].status == "unverified"
        kinds = {e["kind"] for e in transcripts.read_session(LIFECYCLE_SESSION)}
        assert "wiki_demoted" in kinds

    def test_idle_unverified_note_goes_stale_then_deleted(self, stores):
        _, notes, _, lc = stores
        note = notes.create("闲置笔记", "无人使用", "rule")
        r1 = lc.demote(stale_after=1)
        assert any(a["action"] == "note->stale" for a in r1)
        assert [n for n in notes.list() if n.note_id == note.note_id][0] \
            .status == "stale"
        r2 = lc.demote(stale_after=1)
        assert any(a["action"] == "note->deleted" for a in r2)
        assert not any(n.note_id == note.note_id for n in notes.list())

    def test_used_note_does_not_go_stale(self, stores):
        _, notes, _, lc = stores
        note = notes.create("活跃笔记", "经常被用", "rule")
        notes.mark_used(note.note_id)
        r = lc.demote(stale_after=1)
        assert not any(a["action"] in ("note->stale", "note->deleted")
                       and a.get("note") == "活跃笔记" for a in r)

    def test_refuted_and_penalized_note_deleted(self, stores):
        _, notes, _, lc = stores
        note = notes.create("错误笔记", "错误内容", "rule")
        notes.save(replace(note, status="refuted", harmful=2))
        r = lc.demote(stale_after=2)
        assert any(a["action"] == "note->deleted"
                   and a.get("reason") == "refuted+penalized" for a in r)
