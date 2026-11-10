"""MemoryAgent: a reference host agent living on top of the four layers.

Two retrieval modes:
  - "agentic"      - the LLM drives memory tools itself (search_memory /
                     read_note / read_wiki_page / ...) in a tool-call loop,
                     deciding what to look up and what to read in full.
                     This is the agentic retrieval the article describes.
  - "auto_inject"  - the old fixed top-k injection path, kept for
                     lightweight hosts that cannot run a tool loop.

Per session the agent:
  - loads ONLY the injected index (L4) into its system prompt
  - appends everything to the raw transcript (L1)
  - at session end, reflects and writes memory notes (L2) with source
    pointers back into the transcript
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from .layers import InjectedContext, NotesStore, TranscriptStore, WikiStore
from .llm import LLMClient
from .tools import ToolExecutor

logger = logging.getLogger(__name__)

BASE_SYSTEM = (
    "你是部署在企业内部的运维问答 Agent。回答必须简洁、确定。"
    "不知道的内部规则就明确说不知道并指出信息缺口，不要编造。"
)

AGENTIC_SYSTEM_SUFFIX = (
    "你可以使用记忆工具：search_memory 检索目录，read_note / read_wiki_page "
    "读取全文，search_transcripts 核对原始记录，feedback_memory 反馈记忆质量。"
    "回答涉及内部规则的问题前，先检索记忆；发现记忆有误或被误导时，"
    "调用 feedback_memory 给出负面反馈。"
)


class MemoryAgent:
    def __init__(self, llm: LLMClient, transcripts: TranscriptStore,
                 notes: NotesStore, wiki: WikiStore, injected: InjectedContext,
                 tools: ToolExecutor | None = None,
                 mode: Literal["agentic", "auto_inject"] = "agentic",
                 max_tool_rounds: int = 6):
        self.llm = llm
        self.transcripts = transcripts
        self.notes = notes
        self.wiki = wiki
        self.injected = injected
        self.tools = tools
        self.mode = mode if tools is not None else "auto_inject"
        self.max_tool_rounds = max_tool_rounds
        self.session_id = transcripts.new_session()

    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        """Injected context = the tiny top-of-pyramid index only."""
        prompt = BASE_SYSTEM
        if self.mode == "agentic":
            prompt += AGENTIC_SYSTEM_SUFFIX
        index = self.injected.load()
        if index:
            prompt += ("\n\n【自动加载的记忆索引】\n" + index +
                       "\n（需要细节时用工具按需读取。）")
        return prompt

    def _retrieve(self, question: str, top_k: int = 3) -> str:
        """auto_inject path: fixed top-k pull from the lower layers."""
        parts = []
        if self.notes.index:
            hits = self.notes.index.search(question, top_k=top_k * 2)
            notes_by_id = {n.note_id: n for n in self.notes.list()}
            note_hits = 0
            for h in hits:
                if h.layer == "note" and h.ref_id in notes_by_id and note_hits < top_k:
                    n = notes_by_id[h.ref_id]
                    if n.is_penalized or n.status == "refuted":
                        continue
                    self.notes.mark_used(n.note_id)
                    parts.append(f"[笔记:{n.title} / {n.status}] {n.body}")
                    note_hits += 1
                elif h.layer == "wiki":
                    page = self.wiki.get_page(h.ref_id)
                    if page:
                        parts.append(f"[Wiki:{page.title}] {page.body[:400]}")
            return "\n".join(parts)

        hits = self.notes.search(question, top_k=top_k)
        for n in hits:
            self.notes.mark_used(n.note_id)
            parts.append(f"[笔记:{n.title} / {n.status}] {n.body}")
        for page in self.wiki.list_pages():
            if any(tok and tok in page.body for tok in re.findall(r"\w+", question)):
                parts.append(f"[Wiki:{page.title}] {page.body[:400]}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> str:
        self.transcripts.append(self.session_id, "user", question)
        if self.mode == "agentic":
            reply = self._answer_agentic(question)
        else:
            memory = self._retrieve(question)
            user = question if not memory else (
                f"【检索到的记忆】\n{memory}\n\n【问题】{question}")
            reply = self.llm.chat(self._system_prompt(), user,
                                  max_tokens=512, tag="agent")
        self.transcripts.append(self.session_id, "assistant", reply)
        return reply

    def _answer_agentic(self, question: str) -> str:
        """Tool-call loop: the LLM decides what to search and read."""
        assert self.tools is not None
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        schemas = self.tools.schemas()
        for round_no in range(self.max_tool_rounds):
            result = self.llm.chat_with_tools(
                messages, schemas, max_tokens=1024, tag="agent")
            if not result.tool_calls:
                return result.content
            messages.append({
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in result.tool_calls
                ],
            })
            for tc in result.tool_calls:
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    args = {}
                output = self.tools.call(tc.name, args)
                logger.info("tool round %d: %s(%s) -> %d chars",
                            round_no, tc.name, tc.arguments[:60], len(output))
                self.transcripts.append(
                    self.session_id, "tool",
                    f"{tc.name}({tc.arguments}) -> {output[:500]}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": output})
        # Out of rounds: force a final answer without tools.
        result = self.llm.chat_with_tools(messages, [], max_tokens=1024,
                                          tag="agent")
        return result.content

    def receive_grade(self, question: str, answer: str, correct: bool,
                      ground_truth: str) -> None:
        """Grader feedback is recorded as raw evidence, including corrections."""
        self.transcripts.append(
            self.session_id, "grade",
            f"Q: {question}\nA: {answer}\n结果: {'正确' if correct else '错误'}",
            meta={"correct": correct},
        )
        if not correct:
            self.transcripts.append(
                self.session_id, "feedback",
                f"针对问题「{question}」，错误回答「{answer}」；"
                f"已验证的正确答案是：{ground_truth}",
            )

    # ------------------------------------------------------------------ #
    def reflect_and_write_notes(self) -> list[str]:
        """Online note-writing at session end (rules + failures, both matter)."""
        transcript = self.transcripts.read_session(self.session_id)
        graded = [e for e in transcript if e["kind"] in ("user", "grade", "feedback")]
        if not graded:
            return []
        raw = json.dumps(graded, ensure_ascii=False)
        out = self.llm.chat(
            system=(
                "复盘本次会话，把值得跨会话保留的内容写成记忆笔记。"
                "成功经验和已验证事实与失败教训同样重要，不要只记失败。"
                '只输出 JSON：{"notes": [{"title": "...", "type": "rule|failure|fact", '
                '"body": "精炼、可操作的内容"}]}。没有值得记的就返回空数组。'
            ),
            user=f"【会话记录】\n{raw[:6000]}",
            tag="reflect",
        )
        try:
            specs = json.loads(re.search(r"\{.*\}", out, re.S).group(0)).get("notes", [])
        except (AttributeError, json.JSONDecodeError):
            specs = []
        created = []
        for s in specs[:6]:
            note = self.notes.create(
                title=s.get("title", "untitled")[:60],
                body=s.get("body", ""),
                note_type=s.get("type", "rule"),
                sources=(self.session_id,),   # traceability back to raw evidence
            )
            created.append(note.title)
        return created
