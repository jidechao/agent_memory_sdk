"""Tool-layer tests: schema validity and executor behavior (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory_sdk.layers import (InjectedContext, NotesStore,
                                     TranscriptStore, WikiPage, WikiStore)
from agent_memory_sdk.lifecycle import Lifecycle
from agent_memory_sdk.tools import TOOL_SCHEMAS, ToolExecutor


@pytest.fixture()
def executor(tmp_path: Path):
    transcripts = TranscriptStore(tmp_path)
    notes = NotesStore(tmp_path)
    wiki = WikiStore(tmp_path)
    lifecycle = Lifecycle(notes, wiki, transcripts)
    note = notes.create("部署窗口", "每周四凌晨 02:00-04:00", "rule")
    wiki.save_page(WikiPage(slug="deploy", title="部署", body="部署知识页",
                            sources=(note.note_id,)))
    sid = transcripts.new_session()
    transcripts.append(sid, "user", "部署窗口是什么时候？")
    return ToolExecutor(notes, wiki, transcripts, lifecycle), note


class TestSchemas:
    def test_valid_openai_function_calling_format(self):
        assert len(TOOL_SCHEMAS) == 6
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert set(fn) >= {"name", "description", "parameters"}
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]

    def test_expected_tool_names(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert names == {"search_memory", "read_note", "read_wiki_page",
                         "list_wiki", "search_transcripts", "feedback_memory"}


class TestExecutor:
    def _call(self, executor: ToolExecutor, name: str, **args) -> object:
        return json.loads(executor.call(name, args))

    def test_search_memory_keyword_fallback(self, executor):
        ex, note = executor
        hits = self._call(ex, "search_memory", query="部署窗口")
        assert hits and hits[0]["id"] == note.note_id
        assert hits[0]["layer"] == "note"

    def test_read_note_marks_used(self, executor):
        ex, note = executor
        full = self._call(ex, "read_note", note_id=note.note_id)
        assert full["body"] == "每周四凌晨 02:00-04:00"
        assert full["uses"] == 0          # read returns the pre-increment view
        reloaded = self._call(ex, "read_note", note_id=note.note_id)
        assert reloaded["uses"] == 1

    def test_read_wiki_page_and_list(self, executor):
        ex, _ = executor
        listing = self._call(ex, "list_wiki")
        assert listing == [{"slug": "deploy", "title": "部署"}]
        page = self._call(ex, "read_wiki_page", slug="deploy")
        assert page["body"] == "部署知识页"

    def test_search_transcripts(self, executor):
        ex, _ = executor
        hits = self._call(ex, "search_transcripts", query="部署窗口")
        assert hits and hits[0]["kind"] == "user"

    def test_feedback_memory(self, executor):
        ex, note = executor
        result = self._call(ex, "feedback_memory", memory_id=note.note_id,
                            helpful=False, reason="过时了")
        assert result["ok"] and result["updated"][0]["harmful"] == 1

    def test_unknown_tool_returns_error(self, executor):
        ex, _ = executor
        result = self._call(ex, "no_such_tool")
        assert "error" in result

    def test_counters_recorded(self, executor):
        ex, note = executor
        ex.call("read_note", {"note_id": note.note_id})
        ex.call("search_memory", {"query": "x"})
        assert ex.counters.calls == {"read_note": 1, "search_memory": 1}
