"""Regression tests for the 2026-11-09 code-review hardening fixes.

Covers: P1-1 wiki slug path-traversal guard, P1-2 refuted state persistence,
P2-3 facade embedder config, P2-5 corrupt-file tolerance, P2-6 wipe guard,
P2-7 missing-env error, P2-8 consolidate feedback sums, P3 tool top_k clamp.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory_sdk import MemoryConfig, MemorySystem
from agent_memory_sdk.config import _int_env
from agent_memory_sdk.layers import (NotesStore, TranscriptStore,
                                     WikiPage, WikiStore)
from agent_memory_sdk.lifecycle import Lifecycle
from agent_memory_sdk.llm import LLMClient
from agent_memory_sdk.retrieval import RetrievalConfig, RetrievalHit
from agent_memory_sdk.tools import MAX_TOP_K, ToolExecutor


class TestSlugGuard:  # P1-1
    @pytest.mark.parametrize("bad", ["../evil", "..", "a/b", "C:\\evil",
                                     "a" * 100, "UPPER", "with space", ""])
    def test_save_page_rejects_malicious_slug(self, tmp_path, bad):
        wiki = WikiStore(tmp_path)
        with pytest.raises(ValueError, match="invalid wiki slug"):
            wiki.save_page(WikiPage(slug=bad, title="t", body="b"))

    def test_get_page_returns_none_for_invalid_slug(self, tmp_path):
        wiki = WikiStore(tmp_path)
        assert wiki.get_page("../etc/passwd") is None

    def test_sanitize_slug(self):
        assert WikiStore.sanitize_slug("Nova Deploy!!") == "nova-deploy"
        assert WikiStore.sanitize_slug("../evil") == "evil"
        assert WikiStore.sanitize_slug("...") == ""
        assert len(WikiStore.sanitize_slug("a" * 200)) <= 64

    def test_traversal_cannot_escape_root(self, tmp_path):
        wiki = WikiStore(tmp_path)
        outside = tmp_path.parent / "evil.md"
        with pytest.raises(ValueError):
            wiki.save_page(WikiPage(slug="../evil", title="t", body="b"))
        assert not outside.exists()


class TestCorruptionTolerance:  # P2-5
    def test_corrupt_note_file_skipped(self, tmp_path):
        notes = NotesStore(tmp_path)
        good = notes.create("好笔记", "内容", "rule")
        (tmp_path / "notes" / "bad1.md").write_text("没有 frontmatter",
                                                    encoding="utf-8")
        (tmp_path / "notes" / "bad2.md").write_text(
            "---\n{损坏的json\n---\n正文", encoding="utf-8")
        titles = [n.title for n in notes.list()]
        assert titles == [good.title]

    def test_corrupt_transcript_line_skipped(self, tmp_path):
        ts = TranscriptStore(tmp_path)
        sid = ts.new_session()
        ts.append(sid, "user", "正常条目")
        with (tmp_path / "transcripts" / f"{sid}.jsonl").open(
                "a", encoding="utf-8") as f:
            f.write("{这不是json\n")
        entries = ts.read_session(sid)
        assert len(entries) == 1 and entries[0]["content"] == "正常条目"


class TestWipeGuard:  # P2-6
    def test_wipe_refuses_cwd(self):
        with pytest.raises(ValueError, match="refusing to wipe"):
            MemorySystem(config=MemoryConfig(root=Path(".")),
                         chat_fn=lambda s, u: "x", wipe=True)

    def test_wipe_refuses_shallow_path(self):
        # 盘符根目录（C:\ 或 /）：parts 只有 1-2 级，必须拒绝
        drive_root = Path(Path.cwd().anchor)
        with pytest.raises(ValueError, match="refusing to wipe"):
            MemorySystem(config=MemoryConfig(root=drive_root),
                         chat_fn=lambda s, u: "x", wipe=True)

    def test_wipe_allows_dedicated_dir(self, tmp_path):
        root = tmp_path / "memory" / "data"
        root.mkdir(parents=True)
        (root / "keep.txt").write_text("x")
        mem = MemorySystem(config=MemoryConfig(root=root),
                           chat_fn=lambda s, u: "x", wipe=True)
        assert not (root / "keep.txt").exists()
        mem.close()


class TestEnvHandling:  # P2-7 / P3-18
    def test_missing_env_vars_raise_clear_error(self, monkeypatch):
        for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(ValueError, match="缺少环境变量"):
            LLMClient(env_path="/nonexistent/.env")

    def test_int_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MEMORY_MAX_TOOL_ROUNDS", "abc")
        assert _int_env("MEMORY_MAX_TOOL_ROUNDS", 6) == 6
        monkeypatch.setenv("MEMORY_MAX_TOOL_ROUNDS", "9")
        assert _int_env("MEMORY_MAX_TOOL_ROUNDS", 6) == 9


class TestFacadeEmbedderConfig:  # P2-3
    def test_custom_embedding_config_is_used(self, tmp_path):
        config = MemoryConfig(
            root=tmp_path,
            retrieval=RetrievalConfig(mode="dense",
                                      embedding_model="no-such-model-xyz"),
        )
        mem = MemorySystem(config=config, chat_fn=lambda s, u: "x")
        # 自定义模型名被尊重：try_create 用它（并因不存在而优雅降级为 None）
        assert mem.embedder is None
        assert mem.index.effective_mode == "fts"
        mem.close()


class TestRefutedPersistence:  # P1-2
    def test_refuted_state_is_reachable_and_metric_meaningful(self, tmp_path):
        """Simulate verify_notes' no-correction branch: refuted persists."""
        notes = NotesStore(tmp_path)
        note = notes.create("错误规则", "错误内容", "rule")
        notes.save(replace(note, status="refuted"))
        stored = [n for n in notes.list()][0]
        assert stored.status == "refuted"
        # verified_ratio 不再恒等于 1.0
        verified = sum(1 for n in notes.list() if n.status == "verified")
        refuted = sum(1 for n in notes.list() if n.status == "refuted")
        assert verified / (verified + refuted) == 0.0
        # 负反馈后 lifecycle 删除（死分支复活）
        transcripts = TranscriptStore(tmp_path)
        wiki = WikiStore(tmp_path)
        lc = Lifecycle(notes, wiki, transcripts)
        lc.feedback(stored.note_id, helpful=False)
        report = lc.demote()
        assert any(a["action"] == "note->deleted" for a in report)
        assert notes.list() == []


class TestRefutedRetrievalFiltering:
    def test_refuted_note_excluded_from_search_promotion_and_wiki(self, tmp_path):
        from agent_memory_sdk.dreaming import Dreaming
        from agent_memory_sdk.layers import InjectedContext

        notes = NotesStore(tmp_path)
        good = notes.create("正确规则", "正确内容", "rule")
        refuted = notes.create("错误规则", "错误内容", "rule")
        refuted = notes.save(replace(refuted, status="refuted"))

        # Both the lexical fallback and an index-backed search must filter it.
        assert [n.note_id for n in notes.search("内容")] == [good.note_id]

        class StubIndex:
            def search(self, query, top_k=5, layers=("note", "wiki")):
                return [RetrievalHit(n.note_id, "note", n.title, 1.0)
                        for n in notes.list()][:top_k]

        index = StubIndex()
        notes.index = index
        assert [n.note_id for n in notes.search("内容")] == [good.note_id]

        transcripts = TranscriptStore(tmp_path)
        wiki = WikiStore(tmp_path)
        executor = ToolExecutor(
            notes, wiki, transcripts,
            Lifecycle(notes, wiki, transcripts), index=index,
        )
        hits = executor._tool_search_memory("内容", layer="note")
        assert [h["id"] for h in hits] == [good.note_id]

        wiki_catalogs = []

        def chat(system, user):
            if "ranked_ids" in system:
                # Even a stale/invalid LLM ranking cannot reintroduce refuted.
                return json.dumps(
                    {"ranked_ids": [refuted.note_id, good.note_id]})
            wiki_catalogs.append(user)
            return '{"pages": []}'

        injected = InjectedContext(tmp_path)
        dreaming = Dreaming(LLMClient(chat_fn=chat), transcripts, notes, wiki,
                            injected)
        dreaming.promote_index()
        memory_index = injected.load()
        assert good.title in memory_index
        assert refuted.title not in memory_index
        assert refuted.note_id not in memory_index

        dreaming.build_wiki()
        assert len(wiki_catalogs) == 1
        assert good.title in wiki_catalogs[0]
        assert refuted.title not in wiki_catalogs[0]
        assert refuted.note_id not in wiki_catalogs[0]


class TestConsolidateKeeperProtection:  # demo 回归暴露的缺陷
    def test_keeper_survives_delete_ids(self, tmp_path):
        """LLM 把 merge keeper 同时列进 delete_ids 时，keeper 必须存活。"""
        from agent_memory_sdk.dreaming import Dreaming
        from agent_memory_sdk.layers import InjectedContext
        from agent_memory_sdk.llm import LLMClient

        notes = NotesStore(tmp_path)
        a = notes.create("规则A", "内容A", "rule")
        b = notes.create("规则A重复", "内容A", "rule")
        transcripts = TranscriptStore(tmp_path)
        wiki = WikiStore(tmp_path)
        injected = InjectedContext(tmp_path)

        # stub LLM：提议合并 a+b（keeper=uses 高者），同时把 keeper 列入删除
        notes.mark_used(a.note_id)  # a.uses=1 > b.uses=0 → keeper 是 a
        llm = LLMClient(chat_fn=lambda s, u: (
            '{"merge_groups": [["%s", "%s"]], "delete_ids": ["%s"]}'
            % (a.note_id, b.note_id, a.note_id)
            if "merge_groups" in s else "合并后的正文"))
        dreaming = Dreaming(llm, transcripts, notes, wiki, injected)
        dreaming.consolidate()
        remaining = [n.title for n in notes.list()]
        assert "规则A" in remaining and "规则A重复" not in remaining


class TestToolInputClamp:  # P3-16
    def test_top_k_clamped(self, tmp_path):
        notes = NotesStore(tmp_path)
        wiki = WikiStore(tmp_path)
        transcripts = TranscriptStore(tmp_path)
        for i in range(30):
            notes.create(f"笔记{i}", f"内容{i}", "rule")
        ex = ToolExecutor(notes, wiki, transcripts,
                          Lifecycle(notes, wiki, transcripts))
        hits = json.loads(ex.call("search_memory",
                                  {"query": "内容", "top_k": 10 ** 6}))
        assert len(hits) <= MAX_TOP_K
