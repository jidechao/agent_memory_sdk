"""End-to-end demo of the agent-memory-sdk four-layer memory architecture.

Scenario (mirrors the CL-Bench setup from the article):
  A hidden rule base about a fictional internal system "Nova". The model
  cannot possibly know these rules — it can only learn them across sessions.

  Session 1 : fresh agent, no memory -> guesses, fails, gets graded,
              writes memory notes at session end (online write path).
  Dreaming  : offline maintenance -> verifies notes against raw transcripts,
              corrects refuted ones, dedupes, builds the wiki, compresses
              the top-level index, and demotes stale/harmful memories.
  Session 2 : brand-new agent in "agentic" mode — it loads only the injected
              index and pulls details via memory tools (search_memory /
              read_note / ...) on demand.

One deliberately WRONG note is planted before Dreaming to reproduce the
"Pokemon Mt. Moon" case: an online memory error that Dreaming must fix by
re-checking the raw transcripts.

Run:  ./.venv/Scripts/python examples/demo.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent_memory_sdk import MemoryConfig, MemorySystem  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

DATA = Path(__file__).parent.parent / "data"

# Hidden ground truth — the model cannot know these rules a priori.
RULES: dict[str, str] = {
    "Nova 系统的每周部署窗口是什么时候？":
        "每周四 02:00-04:00（UTC+8），窗口外禁止部署。",
    "Nova 系统生产环境配置变更需要什么审批？":
        "需要变更经理和值班架构师双重审批（two-person rule）。",
    "Nova 系统的数据库连接池上限是多少？":
        "单实例 200，集群级硬上限 1200，超过会触发自动熔断。",
    "Nova 系统的日志保留策略是什么？":
        "热存储 7 天，冷存储 180 天，之后归档到对象存储。",
    "Nova 系统回滚操作的超时时间是多少？":
        "回滚必须在 15 分钟内完成，超时自动升级到 P1 事故。",
    "Nova 系统的 API 限流规则是什么？":
        "默认每租户 100 QPS，突发允许 2 倍但持续不超过 30 秒。",
}

KEYWORDS: dict[str, tuple[str, ...]] = {
    RULES["Nova 系统的每周部署窗口是什么时候？"]: ("周四", "02:00"),
    RULES["Nova 系统生产环境配置变更需要什么审批？"]: ("双", "审批"),
    RULES["Nova 系统的数据库连接池上限是多少？"]: ("200", "1200"),
    RULES["Nova 系统的日志保留策略是什么？"]: ("7 天", "180 天"),
    RULES["Nova 系统回滚操作的超时时间是多少？"]: ("15 分钟",),
    RULES["Nova 系统的 API 限流规则是什么？"]: ("100 QPS",),
}


def grade(question: str, answer: str) -> tuple[bool, str]:
    truth = RULES[question]
    return all(k in answer for k in KEYWORDS[truth]), truth


def run_session(mem: MemorySystem, label: str) -> float:
    mem.session_start(mode="agentic")
    agent = mem.agent
    print(f"\n{'=' * 70}\n{label} (session={agent.session_id})")
    print(f"  注入的顶层索引: {len(mem.injected.load().splitlines())} 行")
    score = 0
    for q in RULES:
        ans = agent.answer(q)
        ok, truth = grade(q, ans)
        agent.receive_grade(q, ans, ok, truth)
        score += ok
        print(f"  [{'✓' if ok else '✗'}] {q}\n      → {ans[:90]}")
    created = mem.session_end()
    print(f"  得分: {score}/{len(RULES)}；在线写下 {len(created)} 条笔记: {created}")
    return score / len(RULES)


def main() -> None:
    config = MemoryConfig.from_env(root=DATA)
    mem = MemorySystem(config=config, wipe=True)
    print(f"检索模式: {mem.index.effective_mode}"
          f"（配置: {config.retrieval.mode}，"
          f"RRF k={config.retrieval.rrf_rank_constant}）")

    # ---------------- Session 1: cold start, no memory ---------------- #
    s1 = run_session(mem, "会话 1：无记忆冷启动")

    # Plant one deliberately WRONG note (the 'Mt. Moon' failure mode).
    wrong = mem.notes.create(
        title="Nova 部署窗口（错误记忆）",
        body="部署窗口是每周二 22:00-24:00，窗口外随时可以紧急部署。",
        note_type="rule",
    )
    print(f"\n[注入错误笔记] {wrong.title}: {wrong.body}")

    # ---------------- Dreaming: offline maintenance cycle -------------- #
    print(f"\n{'=' * 70}\nDreaming：离线整理（校验→合并→建 Wiki→压缩索引→下沉）")
    report = mem.dream()
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # ---------------- Session 2: fresh agent, memory loaded ------------ #
    s2 = run_session(mem, "会话 2：新会话，仅加载注入索引（agentic 检索）")

    # ---------------- Summary + health panel --------------------------- #
    print(f"\n{'=' * 70}\n结果汇总")
    print(f"  无记忆会话得分: {s1:.2f}")
    print(f"  记忆加持得分:   {s2:.2f}")
    print()
    print(mem.metrics.render())
    mem.close()


if __name__ == "__main__":
    main()
