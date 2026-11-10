# agent-memory-sdk — Claude 分层记忆架构的 Python 复现与 SDK

复现 Anthropic 公开的 Claude 内部记忆架构设计：四层金字塔 + Dreaming 离线维护引擎 + zvec 混合检索 + 完整生命周期闭环 + agentic 检索工具 + 健康度度量。可作为 SDK 集成到你自己的 Agent。

## 架构

```
        ┌─────────────────────────────┐
   L4   │  Injected Context (MEMORY.md)│  顶层索引：≤200 行 / 25KB，新会话自动加载
        ├─────────────────────────────┤
   L3   │  Wiki (wiki/*.md)            │  稳定、结构化、[[互链]] 的知识地图
        ├─────────────────────────────┤
   L2   │  Memory Notes (notes/*.md)   │  在线写下的规则/失败/事实，带来源与验证状态
        ├─────────────────────────────┤
   L1   │  Transcripts (*.jsonl)       │  不可变原始记录，证据锚点，永不覆写
        └─────────────────────────────┘
              ↑ Dreaming 贯穿四层的离线维护引擎（不是第五层存储）
```

**生命周期闭环**：`note(unverified→verified/refuted) → wiki → index`，反向 `index→wiki→note→stale→deleted`。
负反馈（`harmful > helpful`）的记忆立即从检索中消失，并在下一轮 Dreaming 被降级或删除；
失去全部来源笔记的 Wiki 页自动降级回笔记。所有升降级动作写入 L1 审计。

**Dreaming 五步**：`verify_notes`（拿笔记回查原始记录，推翻则改写，或落盘 `refuted` 保留证据）→ `consolidate`（去重合并）→ `build_wiki`（组织互链知识页 + 修链）→ `promote_index`（压缩顶层索引，penalized/stale/refuted 不得入顶）→ `demote`（下沉与清理）。

📖 **时序详解**：[docs/memory-lifecycle.md](docs/memory-lifecycle.md) —— 每层什么时候写入、什么时候加载、由谁触发（含矩阵表与状态机）。

📖 **冷启动集成**：[docs/cold-start-integration.md](docs/cold-start-integration.md) —— 零记忆起步的生产集成骨架、dream 调度策略、历史资料 bootstrap 与健康检查。

📖 **质量记录**：[Codex 审查报告](docs/code-review-agent-memory-sdk-2026-11-09.md) + [修复总结](docs/code-review-fixes-2026-11-09.md)（21 项全修 + 修复过程新发现的 keeper 误删缺陷）。

## 安装

```bash
pip install -e .          # 本仓库 src-layout
# 依赖：openai / python-dotenv / zvec / httpx
# 可选：本地 Ollama（dense/hybrid 检索的 embedding：bge-m3 或 qwen3-embedding:0.6b）
```

`.env` 配置（默认值即可工作）：

```ini
DEEPSEEK_API_KEY=...              # 任意 OpenAI 兼容端点
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
MEMORY_SEARCH_MODE=hybrid         # hybrid(默认) | fts | dense
EMBEDDING_MODEL=bge-m3            # 换模型后索引自动重建重同步
OLLAMA_BASE_URL=http://localhost:11434
RRF_RANK_CONSTANT=60
```

## 集成到你的 Agent（SDK 用法）

### ① 核心挂点（任何宿主通用）

```python
from agent_memory_sdk import MemorySystem, MemoryConfig

mem = MemorySystem(root="./memory_data")   # LLM 默认读 .env；
# 也可注入你自己的客户端：MemorySystem(chat_client=your_openai_client, chat_model="gpt-x")
# 或注入回调：MemorySystem(chat_fn=lambda system, user: "...")

# 1. 会话开始：把返回的注入索引拼进你的 system prompt（零记忆时返回 ""，无需特判）
ctx = mem.session_start(mode="agentic")

# 2. 注册记忆工具（OpenAI function-calling 格式），在你的 agent loop 里转发调用
tools = mem.tool_schemas()                       # search_memory / read_note /
                                                 # read_wiki_page / list_wiki /
                                                 # search_transcripts / feedback_memory
result = mem.call_tool("search_memory", {"query": "部署窗口"})

# 3. 把每条消息写进 L1 原始记录（证据锚点）
mem.record("user", question)
mem.record("assistant", answer)

# 4. 使用效果反馈（负反馈会让误导性记忆停止被检索并触发降级）
mem.feedback(memory_id, helpful=False, reason="规则已过期")

# 5. 会话结束：复盘写笔记；之后离线跑 Dreaming（调度策略见 docs/cold-start-integration.md）
mem.session_end()
report = mem.dream()

# 6. 健康度：正确率 / 反馈 / 顶层预算 / 分阶段 token 成本 / 趋势
print(mem.metrics.render())
```

### ② 与 OpenAI Agents SDK 集成（`pip install -e ".[agents]"`）

三个关键动作：**自定义客户端指向你的 OpenAI 兼容端点 → `@function_tool` 薄包装记忆工具 → `Runner` 自动驱动工具循环**。

```python
import asyncio, os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (Agent, Runner, function_tool, set_default_openai_api,
                    set_default_openai_client, set_tracing_disabled)
from agent_memory_sdk import MemorySystem

load_dotenv()
# 1) 指向任意 OpenAI 兼容端点（此处为 DeepSeek；官方 OpenAI 可跳过这两行）
set_default_openai_client(AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
))
set_default_openai_api("chat_completions")   # 非 OpenAI 端点必须：它们没有 Responses API
set_tracing_disabled(True)

# 2) 记忆系统（固定持久目录；零记忆开箱即用）
mem = MemorySystem(root="./memory_data")

# 3) 记忆工具薄包装：docstring 就是给 LLM 看的工具说明书
@function_tool
def search_memory(query: str, top_k: int = 5) -> str:
    """跨会话记忆检索（笔记 + Wiki）。返回命中目录；需要全文再用 read_note。"""
    return mem.call_tool("search_memory", {"query": query, "top_k": top_k})

@function_tool
def read_note(note_id: str) -> str:
    """按 id 读取一条记忆笔记的完整内容与元数据（验证状态/来源/反馈计数）。"""
    return mem.call_tool("read_note", {"note_id": note_id})

@function_tool
def feedback_memory(memory_id: str, helpful: bool, reason: str = "") -> str:
    """反馈一条记忆是否有用；负面反馈会让它停止被检索并触发生命周期降级。"""
    return mem.call_tool("feedback_memory",
                         {"memory_id": memory_id, "helpful": helpful,
                          "reason": reason})

# 4) 注入顶层索引（L4）进 instructions；细节由 agent 按需用工具拉取
agent = Agent(
    name="运维值班助手",
    instructions=(
        "涉及内部规则的问题，必须先调用记忆工具查证，不要凭印象回答；"
        "记忆里没有的流程，明确说不知道。\n\n"
        "【自动加载的记忆索引】\n" + mem.session_start(mode="agentic")
    ),
    tools=[search_memory, read_note, feedback_memory],  # 其余工具照此包装
    model=os.environ["DEEPSEEK_MODEL"],
)

async def main():
    result = await Runner.run(agent, "今晚 23 点能把新版本发上生产吗？")
    mem.record("user", "...")                       # 问答落 L1 证据锚点
    mem.record("assistant", result.final_output)
    mem.session_end()                               # 复盘写 L2 笔记
    mem.dream()                                     # 离线整理（生产按策略调度）
    print(mem.metrics.render())                     # 健康度面板

asyncio.run(main())
```

> 完整可运行版（含 `read_wiki_page` / `list_wiki` 全部五个工具与输出示例）见
> [`examples/integrate_openai_agents_sdk.py`](examples/integrate_openai_agents_sdk.py)。
> 注意：Runner 驱动回答用的是 Agents SDK 自己的 client，那部分 token 不进 SDK 账本——
> TokenLedger 计量的是**记忆系统自身**的 LLM 成本（verify/consolidate/wiki/index/reflect）。

### ③ 其他宿主形态

- **手写 tool loop**（不依赖任何框架）：见 [`examples/integrate_host_agent.py`](examples/integrate_host_agent.py)——展示 `tool_schemas()` + `call_tool()` 的最小转发循环。
- **轻量宿主 / 无工具循环能力**：`mem.session_start(mode="auto_inject")` 后用 `mem.agent.answer(q)`，回答前固定 top-k 检索自动注入（一次 LLM 调用，无工具循环）。
- **冷启动（零记忆起步）**：无需任何特判，所有 API 有空态行为；生产部署骨架见 [docs/cold-start-integration.md](docs/cold-start-integration.md)。

### 完整示例清单

- `examples/demo.py` —— 跨会话学习循环（无记忆 0.00（诚实说不知道）→ 记忆加持 0.83，含错误记忆纠正）
- `examples/integrate_host_agent.py` —— 手写宿主 agent loop 的最小集成
- `examples/integrate_openai_agents_sdk.py` —— OpenAI Agents SDK（openai-agents）集成：`@function_tool` 包装记忆工具 + `Runner` 驱动工具循环

## 检索层（zvec）

笔记与 Wiki 的检索由 [zvec](https://github.com/alibaba/zvec)（阿里开源进程内向量数据库）承担，镜像同步 L2/L3 的每次写入：

| `MEMORY_SEARCH_MODE` | 行为 |
|---|---|
| `hybrid`（默认） | dense 向量 + 中文 FTS（内置 Jieba 分词）双路查询，zvec 原生 **RrfReRanker** 融合 |
| `fts` | 仅全文检索（零外部依赖，不需要 embedding 模型） |
| `dense` | 仅向量语义检索 |

Embedding 走本地 Ollama（`/api/embed`，`trust_env=False` 不受系统代理影响）。换 `EMBEDDING_MODEL` 后索引自动检测维度变化、重建并从 notes/wiki 重同步（索引只是派生缓存，markdown 文件才是事实源）。

## 测试与验证

```bash
./.venv/Scripts/python -m pytest tests/ -v   # tests/ 下 51 个单元测试（无需 LLM；dense/hybrid 需 Ollama）
./.venv/Scripts/python examples/demo.py       # 端到端学习循环（调用 LLM，~3-5 分钟）
```
