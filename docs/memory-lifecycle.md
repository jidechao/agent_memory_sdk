# 四层记忆的存储与加载时机详解

> 本文是 `agent-memory-sdk` 的架构时序文档：每一层**什么时候写入（形成）**、**什么时候读取（加载）**、**由谁触发**。
> 核心原则来自 Anthropic 公开的 Claude 记忆设计：*在线记录 + 离线整理形成闭环；顶层是目录不是仓库；只有顶层自动加载，其余全部按需拉取。*

## 0. 总览：两条时间线

```
┌─ 在线时间线（会话进行中，在线写入） ─────────────────────────────┐
│ session_start → [agent 工作: 问答/工具调用] → session_end         │
│      ↑                                        ↓                  │
│   加载 L4                                  写入 L1、L2           │
└─────────────────────────────────────────────────────────────────┘
                          ↓ 会话之间
┌─ 离线时间线（Dreaming，维护与整理） ─────────────────────────────┐
│ verify_notes → consolidate → build_wiki → promote_index → demote │
│   回查 L1        合并 L2      形成 L3      形成/重写 L4   下沉   │
└─────────────────────────────────────────────────────────────────┘
```

**关键认知**：L3（Wiki）和 L4（注入索引）**不在会话进行中形成**——它们是 Dreaming 离线整理的产物。第一次运行一个全新系统时，会话 1 的 agent 是"裸奔"的（四层全空）；记忆是**跨会话逐步长出来的**。

## 1. 各层详解

### L1 Transcripts —— 证据锚点（不可变）

| | 说明 |
|---|---|
| **存储形式** | `data/transcripts/{session_id}.jsonl`，每行一条 JSON： `{ts, kind, content, meta}` |
| **写入时机** | 会话进行中**实时追加**：`session_start` 建立 session → 每条用户消息（`record("user", ...)`）、助手回答、工具调用与结果、评分反馈（`grade`/`feedback`）立即写入；**追加式，任何组件都不得回写或删除** |
| **谁写入** | `MemoryAgent.answer()` / `MemorySystem.record()` / `receive_grade()`；生命周期事件（feedback、demote）写入伪会话 `memory-lifecycle` |
| **加载时机** | ① **Dreaming `verify_notes`**：全文回查，作为校验笔记真伪的证据（这是 L1 最核心的用途——笔记是缓存，L1 才是事实源）；② **`reflect_and_write_notes`**：会话末复盘时读取本会话记录；③ **`search_transcripts` 工具**：agent 怀疑记忆有误时主动核对原始记录 |
| **从不发生** | 永不修改、永不删除、不进入任何 prompt 的自动注入 |

### L2 Memory Notes —— 压缩后的工作材料

| | 说明 |
|---|---|
| **存储形式** | `data/notes/{note_id}.md`，JSON frontmatter（`id/title/type/status/sources/uses/helpful/harmful/idle_dreams/uses_snapshot/updated`）+ 正文 |
| **写入时机** | ① **在线**：`session_end()` → LLM 复盘本会话 L1，写成笔记（成功经验和失败教训都记，`sources` 指回 L1 session id）；② **手动**：宿主直接 `notes.create()`（如示例的 seed）；③ **离线改写**：Dreaming `verify_notes` 发现笔记被证据推翻时**改写正文**（`corrected`）；④ **降级回流**：Wiki 页失去全部来源笔记时降级回笔记 |
| **谁写入** | `NotesStore.create/save`；每次写入自动镜像到 zvec 检索索引 |
| **加载时机** | **从不全量加载**。两个按需入口：① **检索**：`search_memory` 工具（zvec 混合检索返回目录 snippet）→ 看中后 `read_note` 读全文（同时 `mark_used` 计入使用统计）；② **auto_inject 模式**：回答前固定 top-k 检索结果拼进 prompt。`harmful > helpful` 的笔记在检索时被即时过滤 |
| **状态流转** | `unverified` →（Dreaming 校验）→ `verified` / `refuted`；闲置 N 轮 → `stale` → 删除 |

### L3 Wiki —— 稳定、互链的知识地图

| | 说明 |
|---|---|
| **存储形式** | `data/wiki/{slug}.md`，frontmatter（`title/sources/updated`）+ 正文（含 `[[wiki-link]]`） |
| **写入时机** | **只在 Dreaming 离线整理时形成**：`build_wiki` 步骤由 LLM 把已验证笔记聚类、提炼成页面（提取稳定概念而非照抄），并自动修复指向不存在页面的 `[[死链]]`。会话进行中**没有任何组件写 Wiki** |
| **谁写入** | `Dreaming.build_wiki()` → `WikiStore.save_page()`；降级路径：`Lifecycle._demote_orphaned_wiki()` 把失源页面降级回 L2 并删除页面 |
| **加载时机** | **按需拉取，从不自动注入**：① `list_wiki` 工具看目录；② `read_wiki_page` 读全文；③ `search_memory(layer="wiki")` 定向检索；④ auto_inject 模式的 top-k 检索会同时命中 wiki 页 |
| **设计意图** | 文章的核心论据："Wiki 先给地图，再减少盲目搜索"——跨文档组合信息时 Wiki+搜索（61%）优于纯搜索（53%），且省约 2 万 token/题 |

### L4 Injected Context —— 顶层索引（唯一自动加载层）

| | 说明 |
|---|---|
| **存储形式** | `data/MEMORY.md`，纯文本索引行：`- [标题](notes/{id}.md) — 一句话钩子` |
| **写入时机** | **只在 Dreaming `promote_index` 时重写**：LLM 按"下次会话最值得自动加载"对候选笔记排序（高频、通用、已验证优先；**penalized 和 stale 笔记被预过滤，永不可入顶**），取 top-k 生成索引行。写入时**强制预算截断：200 行 / 25KB**（对齐 Claude Code 自动记忆）。也可由宿主手写（如示例的 `_seed()`，模拟上一轮产物） |
| **谁写入** | `Dreaming.promote_index()` → `InjectedContext.write_index()` |
| **加载时机** | ✅ **唯一自动加载层**：`session_start()` → `InjectedContext.load()`（加载时再次执行预算截断）→ 返回字符串**拼进系统 prompt**。此后 agent 看到的只是目录——细节靠工具按需读取 |
| **设计意图** | "顶层是目录不是仓库"：注入内容占用每次会话的 token，错误也会被重复带入，所以必须小；加载行数/字节数及预算占比纳入健康度面板监控 |

## 2. 存储 × 加载矩阵

| 层 | 形成（写）时机 | 写入方 | 加载（读）时机 | 加载方式 |
|---|---|---|---|---|
| L1 Transcripts | 会话中实时追加 | Agent / 生命周期事件 | Dreaming 校验、复盘、`search_transcripts` | 按需（工具/引擎） |
| L2 Notes | 会话末复盘、手动、Dreaming 改写、Wiki 降级 | reflect / 宿主 / Dreaming / Lifecycle | `search_memory`→`read_note`；auto_inject top-k | 按需 |
| L3 Wiki | **仅** Dreaming `build_wiki` | Dreaming | `list_wiki` / `read_wiki_page` / 检索 | 按需 |
| L4 Index | **仅** Dreaming `promote_index`（或手写 seed） | Dreaming / 宿主 | **`session_start` 拼进系统 prompt** | ✅ **自动（唯一）** |

## 3. 生命周期状态机（闭环）

```
                     ┌──────────────── Dreaming 上升路径 ───────────────┐
                     ▼                                                   │
  note(unverified) ──verify──▶ note(verified) ──build_wiki──▶ Wiki ──promote──▶ Index(L4)
      │  ▲                       │                            │                 │
      │  │                       │                            │ 失源            │ 重算轮换
      │  └── wiki 失源降级回笔记 ◀┴────────────────────────────┘                 │
      │                                                                          │
      ├─ 闲置 N 轮 ─▶ note(stale) ── 再闲置 ─▶ 删除                               │
      ├─ verify 推翻 ─▶ refuted（落盘保留证据，不再直接删除）                 │
      │                 ├─ 排除出检索 / 索引提升 / Wiki 组织                    │
      │                 └─ lifecycle：refuted 且 penalized(harmful>helpful) 时删除 │
      └─ feedback 负反馈 harmful>helpful ─▶ 检索即时过滤 ─▶ refuted 时删除 ◀─────┘
```

所有升降级与反馈动作都写入 L1（伪会话 `memory-lifecycle`）作为审计日志，保证"记忆为什么变成现在这样"永远可追溯。

## 4. 两个示例的实际时序对照

### `examples/demo.py`（完整学习循环）

```
会话 1（四层全空，裸奔）: 答题 0.33 → 复盘写 6 条 L2
[手动植入 1 条错误笔记]
Dreaming: verify 纠正错误笔记(corrected) → consolidate 合并去重
          → build_wiki 形成 L3(3 页) → promote_index 形成 L4(8 行索引)
          → demote 清理
会话 2（session_start 自动加载 L4）: agentic 工具按需读 L2/L3 → 答题 1.00
```

### `examples/integrate_openai_agents_sdk.py`

```
_seed(): 手写 L2(2 条 verified) + L4(MEMORY.md)   ← 模拟"上一轮 Dreaming 产物"
session_start(): L4 加载进 INSTRUCTIONS            ← 本文件唯一一次 L4 读取
Runner.run(): search_memory×3 / read_note×4 / list_wiki×1(返回空，L3 尚未形成)
session_end(): 无新笔记
dream(): build_wiki 此刻才形成 L3(1 页) + promote_index 重写 L4(4 行)
```

## 5. API 速查

| 时机 | API | 作用层 |
|---|---|---|
| 会话开始 | `mem.session_start(mode=...)` → 返回注入字符串 | L4 加载 |
| 会话中记录 | `mem.record(kind, content)` | L1 追加 |
| 会话中检索 | `mem.call_tool("search_memory"/"read_note"/"read_wiki_page"/"list_wiki"/"search_transcripts", args)` | L2/L3/L1 按需读 |
| 使用效果反馈 | `mem.feedback(memory_id, helpful, reason)` | L2 计数 + 检索过滤 |
| 会话结束 | `mem.session_end()` | L1→L2（复盘写笔记） |
| 会话之间 | `mem.dream()` | L2 校验合并、形成 L3、重写 L4、下沉清理 |
| 健康度 | `mem.metrics().report()` / `.render()` | 全层监控 |
