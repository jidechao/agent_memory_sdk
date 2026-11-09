"""Minimal host-agent integration example.

Shows how to embed agent-memory-sdk into YOUR OWN agent loop:
  1. create MemorySystem (optionally injecting your own chat client)
  2. prepend mem.session_start() output to your system prompt
  3. register mem.tool_schemas() with your LLM and route tool calls
     to mem.call_tool()
  4. mem.record() every message into the L1 transcript
  5. mem.session_end() + mem.dream() between sessions

The host agent here is a plain hand-written tool loop — no SDK agent
class involved. That is the point: the SDK serves ANY host.

Run:  ./.venv/Scripts/python examples/integrate_host_agent.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_memory_sdk import MemorySystem  # noqa: E402

# Simulate a host application that learned something in a previous session.
SEED_NOTES = [
    ("部署窗口规则", "Nova 系统每周四凌晨 02:00-04:00 是标准部署窗口，窗口外禁止部署。"),
    ("回滚超时", "回滚必须在 15 分钟内完成，超时自动升级为 P1 事故。"),
]

QUESTION = "我们今晚 23 点能把新版本发上生产吗？另外回滚最多能拖多久？"


def seed_memory(mem: MemorySystem) -> None:
    """Pretend a previous session/dream cycle produced verified notes."""
    from dataclasses import replace
    lines = ["# Memory Index", ""]
    for title, body in SEED_NOTES:
        note = mem.notes.create(title=title, body=body, note_type="rule")
        note = mem.notes.save(replace(note, status="verified"))
        lines.append(f"- [{title}](notes/{note.note_id}.md) — {body[:60]}")
    mem.injected.write_index(lines)


def host_agent_loop(mem: MemorySystem, question: str) -> str:
    """A hand-written tool loop, exactly what your own agent would do."""
    system = (
        "你是一个谨慎的运维助手。回答涉及内部规则的问题前，"
        "必须先调用记忆工具查证；不要凭印象回答。\n\n"
        "【自动加载的记忆索引】\n" + mem.session_start(mode="agentic")
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    mem.record("user", question)

    for round_no in range(6):
        result = mem.llm.chat_with_tools(messages, mem.tool_schemas(),
                                         max_tokens=1024, tag="agent")
        if not result.tool_calls:
            mem.record("assistant", result.content)
            return result.content
        messages.append({
            "role": "assistant",
            "content": result.content or None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in result.tool_calls],
        })
        for tc in result.tool_calls:
            output = mem.call_tool(tc.name, json.loads(tc.arguments or "{}"))
            print(f"  [host 工具调用] {tc.name}({tc.arguments[:50]}...) "
                  f"-> {len(output)} chars")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": output})
    return "(达到工具轮次上限)"


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="host_agent_demo_"))
    mem = MemorySystem(root=tmp)
    seed_memory(mem)

    print(f"宿主 Agent 提问: {QUESTION}\n")
    answer = host_agent_loop(mem, QUESTION)
    print(f"\n宿主 Agent 回答:\n{answer}")

    # The host can also report memory quality back (lifecycle loop):
    hits = json.loads(mem.call_tool("search_memory", {"query": "部署窗口"}))
    if hits:
        fb = mem.feedback(hits[0]["id"], helpful=True, reason="规则命中且正确")
        print(f"\n反馈回流: {fb['updated'][0]}")

    created = mem.session_end()
    print(f"会话结束，复盘写笔记: {created}")
    print()
    print(mem.metrics.render())
    mem.close()


if __name__ == "__main__":
    main()
