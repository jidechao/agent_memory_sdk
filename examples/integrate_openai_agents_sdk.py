"""Integration demo: agent-memory-sdk + OpenAI Agents SDK (openai-agents).

Shows the canonical wiring:
  1. Point the Agents SDK at any OpenAI-compatible endpoint (here: DeepSeek
     from .env) via a custom AsyncOpenAI client + chat_completions API.
  2. Wrap MemorySystem's tools as Agents-SDK @function_tool's.
  3. Prepend mem.session_start() (the L4 injected index) to instructions.
  4. Runner.run() drives the tool loop by itself — the agent searches and
     reads memory on its own.
  5. Afterwards: record the exchange into L1, session_end(), dream(),
     and print the health panel.

Run:  ./.venv/Scripts/python examples/integrate_openai_agents_sdk.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents import (Agent, Runner, function_tool, set_default_openai_api,
                    set_default_openai_client, set_tracing_disabled)
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent_memory_sdk import MemorySystem  # noqa: E402

# --------------------------------------------------------------------------- #
# 0. Agents SDK -> DeepSeek (or any OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
load_dotenv()
set_default_openai_client(AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
))
set_default_openai_api("chat_completions")   # DeepSeek has no Responses API
set_tracing_disabled(True)
MODEL = os.environ["DEEPSEEK_MODEL"]

# --------------------------------------------------------------------------- #
# 1. Memory system + seed data (as if learned in previous sessions)
# --------------------------------------------------------------------------- #
_tmp = Path(tempfile.mkdtemp(prefix="agents_sdk_memory_"))
mem = MemorySystem(root=_tmp)


def _seed() -> None:
    from dataclasses import replace
    seeds = [
        ("部署窗口规则",
         "Nova 系统每周四凌晨 02:00-04:00 是标准部署窗口，窗口外禁止部署。"),
        ("回滚超时", "回滚必须在 15 分钟内完成，超时自动升级为 P1 事故。"),
    ]
    lines = ["# Memory Index", ""]
    for title, body in seeds:
        note = mem.notes.create(title=title, body=body, note_type="rule")
        note = mem.notes.save(replace(note, status="verified"))
        lines.append(f"- [{title}](notes/{note.note_id}.md) — {body[:60]}")
    mem.injected.write_index(lines)


_seed()

# --------------------------------------------------------------------------- #
# 2. Memory tools as Agents-SDK function tools (thin wrappers over the SDK)
# --------------------------------------------------------------------------- #
@function_tool
def search_memory(query: str, top_k: int = 5) -> str:
    """跨会话记忆检索（笔记 + Wiki）。返回命中目录；需要全文再用 read_note。"""
    return mem.call_tool("search_memory", {"query": query, "top_k": top_k})


@function_tool
def read_note(note_id: str) -> str:
    """按 id 读取一条记忆笔记的完整内容与元数据（验证状态/来源/反馈计数）。"""
    return mem.call_tool("read_note", {"note_id": note_id})


@function_tool
def read_wiki_page(slug: str) -> str:
    """按 slug 读取一页 Wiki 的完整内容（稳定、结构化的知识）。"""
    return mem.call_tool("read_wiki_page", {"slug": slug})


@function_tool
def list_wiki() -> str:
    """列出全部 Wiki 页面目录（slug + 标题）。"""
    return mem.call_tool("list_wiki", {})


@function_tool
def feedback_memory(memory_id: str, helpful: bool, reason: str = "") -> str:
    """反馈一条记忆是否有用；负面反馈会让它停止被检索并触发降级。"""
    return mem.call_tool("feedback_memory",
                         {"memory_id": memory_id, "helpful": helpful,
                          "reason": reason})


# --------------------------------------------------------------------------- #
# 3. The agent: instructions carry the injected index (L4) only
# --------------------------------------------------------------------------- #
INSTRUCTIONS = (
    "你是 Nova 系统的运维值班助手。规则：\n"
    "1. 涉及内部规则的问题，必须先调用记忆工具查证，不要凭印象回答；\n"
    "2. 引用记忆时注明其验证状态；记忆里没有的流程，明确说不知道；\n"
    "3. 发现记忆有误或被误导时，调用 feedback_memory 反馈。\n\n"
    "【自动加载的记忆索引】\n" + mem.session_start(mode="agentic")
)

agent = Agent(
    name="Nova 运维值班助手",
    instructions=INSTRUCTIONS,
    tools=[search_memory, read_note, read_wiki_page, list_wiki,
           feedback_memory],
    model=MODEL,
)

QUESTION = "今晚 23 点能把新版本发上生产吗？回滚最多能拖多久？超时会怎样？"


async def main() -> None:
    print(f"Agents SDK 模型: {MODEL}")
    print(f"提问: {QUESTION}\n")

    result = await Runner.run(agent, QUESTION, max_turns=10)
    answer = result.final_output
    print(f"回答:\n{answer}\n")

    # 4. Persist the exchange into L1, then close the learning loop
    mem.record("user", QUESTION)
    mem.record("assistant", answer)
    created = mem.session_end()
    print(f"会话结束，复盘写笔记: {created}\n")

    print("Dreaming 离线整理...")
    report = mem.dream()
    print(f"  verify: {len(report['verify'])} 条, demote: {len(report['demote'])} 条, "
          f"index: {report['index']}\n")

    print(mem.metrics.render())
    mem.close()


if __name__ == "__main__":
    asyncio.run(main())
