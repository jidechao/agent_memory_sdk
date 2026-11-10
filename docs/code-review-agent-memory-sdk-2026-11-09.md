# 代码审查报告：`src/agent_memory_sdk`

- 审查日期：2026-11-09
- 审查范围：`src/agent_memory_sdk` 全部 12 个源文件（约 83KB）
- 审查方式：人工全量通读（无静态分析工具）

---

## 一、问题列表

### 严重（P1）

**1. Wiki slug 未校验 → 路径穿越（任意文件读写）**
- 位置：`layers.py:212-230`（`save_page`/`get_page`/`delete_page`）+ `dreaming.py:192`（slug 直接取自 LLM JSON）
- 描述：`WikiPage.slug` 由 LLM 输出的 JSON 提供，未做任何校验就拼入文件路径 `(self.root / f"{page.slug}.md")`。slug 为 `../evil` 可写到记忆根目录之外；Windows 下 slug 如 `C:\evil` 会被 `Path` 解析为绝对路径，完全脱离 root。`get_page`/`delete_page` 同样可被 `read_wiki_page` / `feedback_memory` 工具参数穿越（任意读任意删）。这不是假设性攻击面：slug 是不可信的模型输出，且 `search_memory`/`feedback_memory` 也是直接暴露给宿主 Agent 的工具入口。

**2. `refuted` 生命周期状态不可达 → 核心指标失真 + 死代码**
- 位置：`dreaming.py:75-88`、`lifecycle.py:124`、`metrics.py:52,90`
- 描述：全库搜索确认 **没有任何代码把 note 状态设为 `refuted`**（`rg 'status='` 结果仅有 verified/unverified/stale 三种写入）。`verify_notes` 对 refuted 结论要么改写为 verified、要么直接删除，从不落 `refuted` 状态。后果：
  - `lifecycle.py:124` 的 `note.status == "refuted"` 分支永远不触发（有害但未删的笔记永远留在库里，只是检索时被过滤）；
  - `metrics.py:90` 的 `verified_ratio = verified/(verified+refuted)` 恒等于 1.0，健康面板的"正确率"指标失效——而这个指标正是设计文档里强调的核心健康度。

### 中等（P2）

**3. Facade 忽略自定义配置的 embedding 参数**
- 位置：`facade.py:58`
- 描述：`Embedder.try_create()` 不传参，直接读环境变量。用户传自定义 `MemoryConfig`（其 `retrieval.embedding_model` / `ollama_base_url` 已解析好）时会被环境变量覆盖——构造器已解析配置却不用，属明显不一致的 bug。

**4. `build_wiki` 对 LLM JSON 无防御**
- 位置：`dreaming.py:178,192,195`
- 描述：`spec["slug"]`、`spec["title"]` 直接下标访问，模型漏字段时整个 Dreaming 周期 KeyError 崩溃；其他同类函数都用了 `.get()`，这里不一致。

**5. 单个损坏文件击穿整个存储层**
- 位置：`layers.py:156-157`（`NotesStore.list`）、`layers.py:238-243`（`WikiStore.get_page`）
- 描述：`NotesStore.list()` 对每个文件调 `MemoryNote.from_markdown`，缺 frontmatter 抛 ValueError、frontmatter JSON 损坏抛 JSONDecodeError——任何一个坏文件会让 `list()`/`search()`/metrics/lifecycle/Dreaming 全部不可用（进程崩溃或写入过程中断电即触发）。Wiki 的 `get_page` 对 frontmatter JSON 损坏同样不设防。注释声称"markdown 文件是 source of truth"，source of truth 反而最脆。

**6. `wipe=True` 可删除任意环境变量指定的目录**
- 位置：`facade.py:47-48`
- 描述：`MEMORY_ROOT` 被 env 覆盖（如误设为 `.` 或用户目录）后 `wipe=True` 直接 `shutil.rmtree`，无任何合理性守卫。

**7. LLM 默认配置缺失时裸 KeyError**
- 位置：`llm.py:151-153`
- 描述：`os.environ["DEEPSEEK_MODEL"]` 等三处直接下标，缺一个变量就抛不带上下文的 `KeyError`，用户很难定位。`load_dotenv` 后仍可能为空。

**8. `consolidate` 合并时丢弃反馈计数**
- 位置：`dreaming.py:117-124`
- 描述：合并 keeper 时 `uses` 做了求和，但 `helpful`/`harmful`/`idle_dreams`/`status` 全部丢失——负面反馈累积的"毒笔记"合并后毒信号清零，负面反馈闭环在合并处漏了。

**9. Dreaming 每 cycle 全量重建 Wiki**
- 位置：`dreaming.py:143-203`
- 描述：`build_wiki` 不区分已存在页面，每个 dream cycle 对全部笔记重新生成 slug/正文并覆盖写入（churn + LLM 成本 + 同名 slug 直接覆盖旧页）；另外笔记升级为 wiki 后**不删除原笔记**（pyramid "提升"实为复制），orphaned-wiki 降级逻辑（`lifecycle.py:98`）只有在 consolidate 删掉源笔记时才可能触发，整体状态机有振荡隐患。

**10. 系统提示鼓励幻觉**
- 位置：`agent.py:23`
- 描述：`BASE_SYSTEM` 写明"如果你不知道某条内部规则，直接给出你的最佳猜测，不要回避"。对一个强调"回答必须简洁、确定"的运维问答 Agent，把"不知道就猜"写进 system prompt 与整个记忆系统"宁缺毋滥、验证后才能入库"的设计哲学直接矛盾。

### 轻微（P3）

| # | 位置 | 问题 |
|---|------|------|
| 11 | `dreaming.py:45-47` | `read_all()` 连续调用两次，全量读两遍 transcript |
| 12 | `layers.py:160-162` | `mark_used` 每次 `list()` 全量 + 单条 save，O(N)；`tools.py:177-183` `_tool_search_memory` 对每个 hit 再全量 `notes.list()`，O(N·H) |
| 13 | `config.py:30-45` | `index_max_lines/bytes`、`stale_after_dreams` 等配置字段与 `metrics.py:25-26`、`layers.py:273-274` 硬编码常量三处并存，配置字段实际是死的 |
| 14 | `metrics.py:107` | `report()` 在无历史时顺手写一条 snapshot（读操作带写副作用） |
| 15 | `llm.py:24-27` | 模块导入时全局 `sys.stdout.reconfigure`，对宿主进程有隐式全局副作用 |
| 16 | `tools.py:165-220` | 工具入参 `top_k` 无上界，模型可传 10^6；`_tool_search_transcripts` 全量读 transcript 后取 `matches[-top_k:]`（顺序无意义） |
| 17 | `lifecycle.py:108-111` | `create` 已写入 unverified 状态，紧接的 `save(replace(status="unverified"))` 是冗余重写（还顺带刷新了时间戳） |
| 18 | `config.py:38-45` | env 数值解析 `int(...)` 无容错，坏值直接 ValueError |
| 19 | `dreaming.py:259` | `promoted` 统计的是模型返回的 id 数，不是实际写入 index 的行数 |
| 20 | `facade.py:117` | `close()` 只关 index，不关 `Embedder` 的 httpx client |
| 21 | `layers.py:42-47` | `read_session` 对损坏 JSONL 行无容错 |

---

## 二、修改建议

**P1-1（slug 校验，优先修）：**

```python
# layers.py，加一个统一入口
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

def _slug_path(self, slug: str) -> Path:
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"invalid wiki slug: {slug!r}")
    return self.root / f"{slug}.md"
```

`save_page`/`get_page`/`delete_page` 全部走 `_slug_path`；`dreaming.build_wiki` 里生成前先清洗（非法字符替换 + 长度截断），失败跳过该页面并记入 report。

**P1-2（refuted 状态）：** 在 `verify_notes` 的 refuted 分支，当无法给出修正时不要直接删除，先 `save(replace(note, status="refuted"))` 并保留证据引用；只有已 refuted 且被惩罚的笔记才在 lifecycle 里删除（`lifecycle.py:124` 现有逻辑就接通了），`verified_ratio` 由此恢复意义。

**P2-3：** `facade.py:58` 改为 `Embedder.try_create(rc.embedding_model, rc.ollama_base_url)`。

**P2-4：** `spec.get("slug")`/`spec.get("title")` 判空跳过，与同文件其他 try/except 风格对齐。

**P2-5：** `NotesStore.list` 用 try/except 逐文件兜底，坏文件重命名为 `.corrupt` 或跳过并 `logger.error`；Wiki frontmatter 解析同样包一层。

**P2-6：** `wipe` 生效前校验 root 非根目录、非当前目录、路径深度 ≥ 2，或干脆要求显式传参 `wipe=Path(...)` 并文档标注危险。

**P2-7：**

```python
missing = [k for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL") if not os.environ.get(k)]
if missing:
    raise ValueError(f"LLM 缺少环境变量: {', '.join(missing)}（或改用 client=/chat_fn= 注入）")
```

**P2-8：** keeper 合并时 `helpful=sum(m.helpful)`、`harmful=sum(m.harmful)`。

**P2-9：** `build_wiki` 前 `self.wiki.list_pages()` 取现有 slug，模型输出与现有页同 slug 时跳过或要求 diff 更新；wiki 提升后是否删原笔记应与 lifecycle 状态机一起明确（当前"复制式提升"配合 orphaned-wiki 降级会来回震荡）。

**P2-10：** 改为"不知道就说不知道，并说明信息缺口"，把"猜"的权力留给上层 Agent 的明确配置。

P3 项多为性能与整洁度，可随手修：合并两次 `read_all`、给 `mark_used` 直接按 id 定位文件、把 index 预算统一走 `MemoryConfig`、给工具入参 `top_k` 上限（如 ≤20）等。

---

## 三、总结评价

**整体质量：良好偏上（B+）。** 架构清晰（四层 + Dreaming + 生命周期 + 度量，职责划分准确）、注释密度高且大多与实现一致、可注入性设计（chat_fn/client/backend、可禁用 dense）到位、错误兜底意识不错（tool 异常回传 LLM、embedder 优雅降级、索引重建自愈）。中文 prompt 与领域匹配。

主要短板集中在两条线：**（1）不可信数据进文件系统的防御缺失**——slug 是唯一的 LLM→文件路径直通口，必须加白名单校验；**（2）状态机与指标的"设计-实现"脱节**——`refuted` 状态不可达导致核心正确率指标恒为 1.0、有害笔记删除分支成死代码、wiki 提升与降级构成复制/震荡回路。这批问题不属于风格问题，建议在下一个版本前修复 P1 两项和 P2 的 3/5/7/8 四项；P3 可作为 cleanup PR 批量处理。
