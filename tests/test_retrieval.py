"""Retrieval tests: fts / dense / hybrid modes against the zvec index.

dense and hybrid tests require a local Ollama embedding model; they are
skipped automatically when no embedder is available. fts tests always run.
Run directly for verbose output: python tests/test_retrieval.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory_sdk.embeddings import Embedder
from agent_memory_sdk.layers import NotesStore, WikiPage, WikiStore
from agent_memory_sdk.retrieval import RetrievalConfig, ZvecMemoryIndex

NOTES = [
    ("部署窗口规则", "Nova 系统每周四凌晨 02:00-04:00 是标准部署窗口，窗口外禁止部署", "verified"),
    ("日志保留策略", "热存储保留 7 天，冷存储 180 天，之后归档到对象存储", "verified"),
    ("数据库连接池", "单实例上限 200，集群硬上限 1200，超过触发自动熔断", "unverified"),
    ("回滚超时", "回滚必须在 15 分钟内完成，超时自动升级为 P1 事故", "verified"),
]
TARGET = "部署窗口规则"
SEMANTIC_QUERY = "什么时候可以发布新版本到生产环境"   # 与目标零词汇重叠
OVERLAP_QUERY = "紧急部署的时间规定"                  # 部分重叠

EMBEDDER = Embedder.try_create()
needs_embedder = pytest.mark.skipif(EMBEDDER is None,
                                    reason="no Ollama embedding model available")


def build(tmp_path: Path, mode: str, embedder: Embedder | None):
    config = RetrievalConfig(mode=mode, rrf_rank_constant=60)  # type: ignore[arg-type]
    index = ZvecMemoryIndex(tmp_path, config, embedder)
    notes = NotesStore(tmp_path, index=index)
    wiki = WikiStore(tmp_path, index=index)
    for title, body, status in NOTES:
        note = notes.create(title=title, body=body, note_type="rule")
        if status != "unverified":
            notes.save(replace(note, status=status))
    wiki.save_page(WikiPage(slug="oncall-rotation", title="值班轮换",
                            body="值班每周一上午十点交接，节假日顺延。"))
    return index, notes, wiki


class TestFts:
    def test_zero_overlap_semantic_query_misses(self, tmp_path):
        index, _, _ = build(tmp_path, "fts", None)
        titles = [h.title for h in index.search(SEMANTIC_QUERY, top_k=3)]
        assert TARGET not in titles          # 关键词检索的固有盲区
        index.close()

    def test_overlap_query_hits(self, tmp_path):
        index, _, _ = build(tmp_path, "fts", None)
        hits = index.search(OVERLAP_QUERY, top_k=3)
        assert hits and hits[0].title == TARGET
        index.close()


@needs_embedder
class TestDense:
    def test_semantic_recall_fills_keyword_blind_spot(self, tmp_path):
        index, _, _ = build(tmp_path, "dense", EMBEDDER)
        hits = index.search(SEMANTIC_QUERY, top_k=3)
        assert hits and hits[0].title == TARGET
        index.close()


@needs_embedder
class TestHybrid:
    def test_rrf_fusion_hits(self, tmp_path):
        index, _, _ = build(tmp_path, "hybrid", EMBEDDER)
        hits = index.search(OVERLAP_QUERY, top_k=3)
        assert hits and hits[0].title == TARGET
        index.close()

    def test_model_switch_rebuilds_index(self, tmp_path):
        other = {"bge-m3": "qwen3-embedding:0.6b",
                 "qwen3-embedding:0.6b": "bge-m3"}.get(
                     EMBEDDER.model_name, "qwen3-embedding:0.6b")
        try:
            alt = Embedder(other)
        except Exception:
            pytest.skip(f"alternate model {other} not installed")
        index, notes, _ = build(tmp_path, "hybrid", EMBEDDER)
        assert not index.needs_resync
        index.close()
        index2, notes2, _ = build(tmp_path, "hybrid", alt)
        assert index2.needs_resync
        index2.sync(notes2.list(), [])
        hits = index2.search(OVERLAP_QUERY, top_k=1)
        assert hits and hits[0].title == TARGET
        index2.close()


class TestIndexSync:
    def test_delete_propagates(self, tmp_path):
        index, notes, _ = build(tmp_path, "fts", None)
        victim = notes.list()[0]
        notes.delete(victim.note_id)
        hits = index.search(victim.title, top_k=5, layers=("note",))
        assert all(h.ref_id != victim.note_id for h in hits)
        index.close()

    def test_wiki_layer_filter(self, tmp_path):
        index, _, _ = build(tmp_path, "fts", None)
        hits = index.search("值班什么时候交接", top_k=3, layers=("wiki",))
        assert hits and hits[0].title == "值班轮换"
        assert all(h.layer == "wiki" for h in hits)
        index.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
