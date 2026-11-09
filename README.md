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

**Dreaming 五步**：`verify_notes`（拿笔记回查原始记录，推翻则改写）→ `consolidate`（去重合并）→ `build_wiki`（组织互链知识页 + 修链）→ `promote_index`（压缩顶层索引，penalized/stale 不得入顶）→ `demote`（下沉与清理）。

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

```python
from agent_memory_sdk import MemorySystem, MemoryConfig

mem = MemorySystem(root="./memory_data")   # LLM 默认读 .env；
# 也可注入你自己的客户端：MemorySystem(chat_client=your_openai_client, chat_model="gpt-x")
# 或注入回调：MemorySystem(chat_fn=lambda system, user: "...")

# 1. 会话开始：把返回的注入索引拼进你的 system prompt
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

# 5. 会话结束：复盘写笔记；之后离线跑 Dreaming
mem.session_end()
report = mem.dream()

# 6. 健康度：正确率 / 反馈 / 顶层预算 / 分阶段 token 成本 / 趋势
print(mem.metrics.render())
```

不想跑工具循环的轻量宿主：`mem.session_start(mode="auto_inject")` 后用 `mem.agent.answer(q)`（固定 top-k 注入检索结果）。

完整示例：
- `examples/demo.py` —— 跨会话学习循环（CL-Bench 复现：0.33 → 1.00，含错误记忆纠正）
- `examples/integrate_host_agent.py` —— 手写宿主 agent loop 的最小集成
- `examples/integrate_openai_agents_sdk.py` —— OpenAI Agents SDK（openai-agents）集成：`@function_tool` 包装记忆工具 + `Runner` 驱动工具循环（`pip install -e ".[agents]"`）

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
./.venv/Scripts/python -m pytest tests/ -v   # 28 个单元测试（无需 LLM；dense/hybrid 需 Ollama）
./.venv/Scripts/python examples/demo.py       # 端到端学习循环（调用 LLM，~3-5 分钟）
```
