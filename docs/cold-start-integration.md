# 冷启动集成指南：Agent 刚启动、没有任何记忆时怎么接

> 面向真实部署场景。核心结论：**零记忆是本架构的默认设计场景**（文章中的 CL-Bench 就是每次从新会话开始），集成代码在"零记忆"和"有记忆"两种状态下**完全相同**，不需要任何特判。真正需要决策的只有三件事：**持久目录、不 seed、dream 调度策略**。

## 1. 零记忆时各 API 的空态行为（内置，无需判空）

| 调用 | 零记忆时的行为 |
|---|---|
| `mem.session_start()` | 返回 `""`（MEMORY.md 不存在） |
| `search_memory` / `list_wiki` | 返回 `[]`——agent 查不到就靠自身知识回答，并在指令约束下声明"我不知道内部规则" |
| `mem.record()` | 正常写入，**第一条 transcript 从此刻开始积累** |
| `mem.session_end()` | 正常复盘——**第一批 L2 笔记在这里诞生** |
| `mem.dream()` | 正常执行——**第一个 Wiki、第一版索引在这里形成** |

唯一建议做的判空是 **prompt 卫生**：注入字符串为空时，不要把空的"记忆索引"标题拼进系统提示（见 §2 代码）。

## 2. 标准集成骨架（生产形态）

```python
from agent_memory_sdk import MemorySystem, MemoryConfig

# ① root 必须是固定持久目录，不是 tempfile —— 记忆的价值全在跨重启存活
# ② 生产环境绝不传 wipe=True（那是 demo 用的）
mem = MemorySystem(root="/var/lib/myagent/memory")

BASE_PROMPT = (
    "你是运维值班助手。涉及内部规则的问题，先调用记忆工具查证；"
    "记忆里没有的，明确说不知道，不要凭印象回答。"
)

def handle_session(question: str) -> str:
    # ③ 冷启动时 ctx 是 ""，有记忆后是索引——业务代码两种情形完全一样
    ctx = mem.session_start(mode="agentic")
    system = BASE_PROMPT + (f"\n\n【记忆索引】\n{ctx}" if ctx else "")

    # 你的 agent loop（OpenAI Agents SDK / LangGraph / 手写均可）
    answer = run_my_agent(
        system=system,
        user=question,
        tools=mem.tool_schemas(),
        tool_handler=mem.call_tool,
    )

    # ④ 无论答对答错都落 L1 —— 错误答案 + 评分反馈是 Dreaming 纠错的证据
    mem.record("user", question)
    mem.record("assistant", answer)

    # ⑤ 会话末复盘：第一批笔记在此诞生（冷启动最重要的一个调用）
    mem.session_end()
    return answer
```

与 demo 的三处差异：**固定 root、不 seed 不手写 MEMORY.md、不 wipe**。其余一模一样——这就是"零记忆是默认场景"的含义。

## 3. `dream()` 的调度策略（生产环境唯一需要真想清楚的事）

Dreaming 有 LLM 成本（健康面板中 `verify / consolidate / wiki / index` 四段 token 统计就是为成本监控准备的），**不要每个会话都跑**。常见三种策略：

```python
# 策略 A：按会话计数（简单直接）
sessions_since_dream += 1
if sessions_since_dream >= 10:
    mem.dream()
    sessions_since_dream = 0

# 策略 B：按时间/空闲（服务型 agent 推荐）
#   用 cron / APScheduler 在低谷期跑，比如每天凌晨
scheduler.add_job(mem.dream, "cron", hour=3)

# 策略 C：事件触发（有新笔记才整理，最省成本）
created = mem.session_end()
if len(created) >= 5:          # 攒够一批新经验再整理
    mem.dream()
```

**冷启动初期建议前几次会话后各跑一次 dream**——让 Wiki 和索引尽快建立（第一次 dream 后 agent 就有"地图"了），之后转入策略 B/C 节流。

## 4. 可选：用历史资料引导冷启动（一次性 bootstrap）

如果上线时手里已有运维文档、历史工单、FAQ——不必让 agent 从零踩坑，把存量知识灌进去再整理一次：

```python
def bootstrap_if_first_run(mem: MemorySystem) -> None:
    if mem.notes.list():            # 已有记忆 → 不是首次启动，跳过
        return
    for doc in load_runbooks():     # 你的历史文档
        # 方式一（推荐）：当作"历史会话证据"写 L1，让 Dreaming 自己提炼
        #   —— 笔记自动带 sources 证据链和 verified 状态，与自然生长的记忆质量一致
        mem.record("user", doc["question"])
        mem.record("feedback", f"已验证的正确答案是：{doc['answer']}")
        # 方式二：直接 mem.notes.create(...) 写 L2
        #   —— 更快，但失去 L1 证据锚点，status 停留在 unverified
    mem.dream()                     # 一次性整理出 Wiki + 索引
```

**推荐方式一**：走 L1 → Dreaming 的标准通道，保留完整证据链（对应文章"笔记是压缩后的工作材料，原始记录是证据锚点"的原则）。

## 5. 冷启动后的生长节奏（预期管理）

```
会话 1-2    : 四层基本全空，agent 靠自身知识 + 对内部规则说"不知道"
              → 每次 session_end 攒下第一批笔记
第 1 次 dream: Wiki / 顶层索引首次形成（L3/L4 诞生）
会话 3+      : agent 开始有索引可用，search_memory / read_note 命中率上升
第 N 次 dream: 错误笔记被纠正(corrected)、重复被合并、stale 被清理
              —— 生命周期闭环开始运转
```

## 6. 上线后的健康检查

定期查看 `mem.metrics.render()`，健康的生长曲线应满足：

| 指标 | 健康表现 |
|---|---|
| `笔记规模趋势` | 稳步上升后趋平（增长放缓说明 consolidate/demote 在控制膨胀） |
| `正确率趋势 (verified/(v+r))` | 稳定在高位（接近 1.0） |
| `顶层预算占用` | 远低于 100%（长期贴近预算上限说明 promote 策略太松） |
| `反馈 helpful/harmful` | helpful 持续占优；harmful 增长快说明知识过期，应检查 dream 频率 |
| `分阶段成本` | dream 四阶段 token 与任务收益匹配；成本过高时降低 dream 频率 |

## 7. 重启与索引行为速查

- 进程重启后 `MemorySystem(root=同一目录)` 会复用全部四层数据与 zvec 索引；
- zvec 索引是**派生缓存**：只有 `EMBEDDING_MODEL` 变更（维度变化）时才会自动重建并从 notes/wiki 重同步（`needs_resync`），日常重启零成本；
- `MEMORY_SEARCH_MODE=fts` 时完全不依赖 embedding 服务，Ollama 不可用也会自动降级为 fts（记 warning 日志），检索不中断。

相关文档：[四层记忆的存储与加载时机详解](memory-lifecycle.md)
