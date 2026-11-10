"""Dreaming: the offline maintenance engine across all four layers.

It is NOT a fifth storage layer. It runs between sessions and:
  1. verify_notes   - re-check every unverified note against raw transcripts
                      (the evidence anchor) and rewrite refuted memories
  2. consolidate    - merge duplicate/overlapping notes, drop stale ones
  3. build_wiki     - organize verified notes into interlinked wiki pages,
                      then repair broken [[links]]
  4. promote_index  - compress the highest-value memories into MEMORY.md
                      (the tiny injected index), pushing details back down
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace

from .layers import InjectedContext, NotesStore, TranscriptStore, WikiPage, WikiStore
from .lifecycle import Lifecycle
from .llm import LLMClient

logger = logging.getLogger(__name__)


class Dreaming:
    def __init__(self, llm: LLMClient, transcripts: TranscriptStore,
                 notes: NotesStore, wiki: WikiStore, injected: InjectedContext,
                 lifecycle: Lifecycle | None = None, stale_after: int = 2):
        self.llm = llm
        self.transcripts = transcripts
        self.notes = notes
        self.wiki = wiki
        self.injected = injected
        self.lifecycle = lifecycle or Lifecycle(notes, wiki, transcripts)
        self.stale_after = stale_after

    # ------------------------------------------------------------------ #
    # 1. Notes are a cache, transcripts are the source of truth
    # ------------------------------------------------------------------ #
    def verify_notes(self) -> list[dict]:
        """Re-validate each unverified note against the raw transcripts."""
        report: list[dict] = []
        # Keep the prompt bounded: graded entries carry the strongest signal.
        all_sessions = self.transcripts.read_all()
        graded = [e for sess in all_sessions.values()
                  for e in sess if e["kind"] in ("grade", "feedback")]
        evidence_short = json.dumps(graded, ensure_ascii=False) or \
            json.dumps(all_sessions, ensure_ascii=False)

        for note in self.notes.list():
            if note.status == "verified":
                continue
            verdict = self.llm.chat(
                system=(
                    "你是记忆校验器。给定一条记忆笔记和原始会话记录（证据锚点），"
                    "判断笔记是否被证据支持。只输出 JSON："
                    '{"verdict": "supported|refuted|unclear", '
                    '"corrected_body": "若被推翻，给出依据证据改写后的正确笔记正文，否则为空字符串", '
                    '"reason": "一句话理由"}'
                ),
                user=(f"【笔记】标题：{note.title}\n正文：{note.body}\n\n"
                      f"【原始记录节选】\n{evidence_short[:6000]}"),
                tag="verify",
            )
            try:
                data = json.loads(re.search(r"\{.*\}", verdict, re.S).group(0))
            except (AttributeError, json.JSONDecodeError):
                data = {"verdict": "unclear", "reason": "unparseable judge output"}

            if data.get("verdict") == "supported":
                self.notes.save(replace(note, status="verified"))
                report.append({"note": note.title, "action": "verified",
                               "reason": data.get("reason", "")})
            elif data.get("verdict") == "refuted":
                corrected = (data.get("corrected_body") or "").strip()
                # Error-propagation metric: how often was this wrong memory
                # retrieved before Dreaming caught it?
                uses_before = note.uses
                if corrected:
                    self.notes.save(replace(note, body=corrected, status="verified"))
                    report.append({"note": note.title, "action": "corrected",
                                   "uses_before_correction": uses_before,
                                   "reason": data.get("reason", "")})
                else:
                    # No fix available: persist the refuted state (with its
                    # evidence trail) instead of silently deleting — the
                    # refuted count feeds the accuracy metric, and lifecycle
                    # deletes it only once it is also penalized by feedback.
                    self.notes.save(replace(note, status="refuted"))
                    report.append({"note": note.title, "action": "refuted",
                                   "uses_before_correction": uses_before,
                                   "reason": data.get("reason", "")})
            else:
                report.append({"note": note.title, "action": "kept-unverified",
                               "reason": data.get("reason", "")})
        return report

    # ------------------------------------------------------------------ #
    # 2. Consolidation: merge duplicates, keep the memory budget small
    # ------------------------------------------------------------------ #
    def consolidate(self) -> list[dict]:
        notes = self.notes.list()
        if len(notes) < 2:
            return []
        catalog = "\n".join(f"- id={n.note_id} | {n.title} | {n.body[:120]}"
                            for n in notes)
        plan = self.llm.chat(
            system=(
                "你在做记忆去重。找出语义重复或应合并的笔记分组。"
                '只输出 JSON：{"merge_groups": [["id1","id2"], ...], '
                '"delete_ids": ["idX", ...]}。没有可合并项就返回空数组。'
            ),
            user=f"【笔记目录】\n{catalog}",
            tag="consolidate",
        )
        try:
            data = json.loads(re.search(r"\{.*\}", plan, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            return []

        by_id = {n.note_id: n for n in notes}
        report: list[dict] = []
        merged_away: set[str] = set()
        keepers: set[str] = set()
        for group in data.get("merge_groups", []):
            members = [by_id[i] for i in group if i in by_id and i not in merged_away]
            if len(members) < 2:
                continue
            merged_body = self.llm.chat(
                system="把以下几条记忆合并为一条精炼笔记正文，只输出正文。",
                user="\n---\n".join(f"{m.title}: {m.body}" for m in members),
                tag="consolidate",
            )
            keeper = max(members, key=lambda m: m.uses)
            keepers.add(keeper.note_id)
            self.notes.save(replace(
                keeper, body=merged_body,
                uses=sum(m.uses for m in members),
                helpful=sum(m.helpful for m in members),
                harmful=sum(m.harmful for m in members),
                sources=tuple(sorted({s for m in members for s in m.sources})),
            ))
            for m in members:
                if m.note_id != keeper.note_id:
                    self.notes.delete(m.note_id)
                    merged_away.add(m.note_id)
            report.append({"action": "merged", "into": keeper.title,
                           "from": [m.title for m in members]})
        for nid in data.get("delete_ids", []):
            # A merge keeper must never be deleted in the same cycle — the LLM
            # occasionally proposes both, which would silently lose knowledge.
            if nid in by_id and nid not in merged_away and nid not in keepers:
                self.notes.delete(nid)
                report.append({"action": "deleted-stale", "note": by_id[nid].title})
        return report

    # ------------------------------------------------------------------ #
    # 3. Wiki: a knowledge map, then repair its links
    # ------------------------------------------------------------------ #
    def build_wiki(self) -> list[dict]:
        notes = [n for n in self.notes.list() if n.status != "refuted"]
        if not notes:
            return []
        catalog = "\n".join(
            f"- note_id={n.note_id} | 主题建议 | {n.title} | {n.body[:150]}"
            for n in notes)
        plan = self.llm.chat(
            system=(
                "把记忆笔记组织成少量稳定的 Wiki 页面。只输出 JSON："
                '{"pages": [{"slug": "英文短横线slug", "title": "页面标题", '
                '"note_ids": ["..."], "outline": "页面应涵盖的要点"}]}'
            ),
            user=f"【已验证/待整理笔记】\n{catalog}",
            tag="wiki",
        )
        try:
            data = json.loads(re.search(r"\{.*\}", plan, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            return []

        by_id = {n.note_id: n for n in notes}
        report: list[dict] = []
        # Sanitize every LLM-proposed slug before it touches the filesystem.
        specs = []
        for spec in data.get("pages", []):
            slug = WikiStore.sanitize_slug(str(spec.get("slug", "")))
            title = str(spec.get("title", "")).strip()
            if not slug or not title:
                report.append({"action": "wiki-page-skipped",
                               "reason": f"invalid slug/title in {spec!r}"})
                continue
            specs.append((slug, title, spec))
        all_slugs = [s for s, _, _ in specs]

        existing = {p.slug: p for p in self.wiki.list_pages()}
        for slug, title, spec in specs:
            members = [by_id[i] for i in spec.get("note_ids", []) if i in by_id]
            if not members:
                continue
            source_ids = tuple(sorted(m.note_id for m in members))
            # Incremental update: skip pages whose slug AND source set are
            # unchanged — rebuilding the whole wiki every cycle is churn.
            old = existing.get(slug)
            if old and tuple(sorted(old.sources)) == source_ids:
                report.append({"action": "wiki-page-kept", "slug": slug})
                continue
            other = [s for s in all_slugs if s != slug]
            body = self.llm.chat(
                system=(
                    "撰写一页 Wiki 正文（Markdown）：提炼稳定概念、规则和证据位置，"
                    "不要照抄。提及相关主题时使用 [[slug]] 链接，"
                    f"可链接的页面有：{other}。只输出正文。"
                ),
                user=(f"页面：{title}\n要点：{spec.get('outline', '')}\n"
                      f"素材：\n" + "\n---\n".join(
                          f"{m.title}（来源会话 {','.join(m.sources) or 'n/a'}）: {m.body}"
                          for m in members)),
                tag="wiki",
            )
            self.wiki.save_page(WikiPage(
                slug=slug, title=title, body=body, sources=source_ids,
            ))
            report.append({"action": "wiki-page", "slug": slug,
                           "from_notes": [m.title for m in members]})

        # Repair broken cross-page links (link-level hole fixing).
        for slug, bad_links in self.wiki.broken_links().items():
            existing = {p.slug for p in self.wiki.list_pages()}
            mapping: dict[str, str] = {}
            for bad in bad_links:
                best = self._closest_slug(bad, existing)
                if best:
                    mapping[bad] = best
            if mapping:
                self.wiki.retarget_links(slug, mapping)
                report.append({"action": "links-repaired", "page": slug,
                               "mapping": mapping})
        return report

    @staticmethod
    def _closest_slug(bad: str, existing: set[str]) -> str | None:
        bad_tokens = set(bad.split("-"))
        best, best_score = None, 0
        for slug in existing:
            score = len(bad_tokens & set(slug.split("-")))
            if score > best_score:
                best, best_score = slug, score
        return best

    # ------------------------------------------------------------------ #
    # 4. Promote the top layer: an index, not a warehouse
    # ------------------------------------------------------------------ #
    def promote_index(self, top_k: int = 10) -> dict:
        # Penalized (harmful>helpful), stale, and refuted notes never reach L4.
        notes = [n for n in self.notes.list()
                 if not n.is_penalized and n.status not in ("stale", "refuted")]
        if not notes:
            return {"index_lines": 0}
        catalog = "\n".join(
            f"- id={n.note_id} | 类型={n.note_type} | 状态={n.status} | "
            f"使用次数={n.uses} | {n.title} | {n.body[:100]}"
            for n in notes)
        ranked = self.llm.chat(
            system=(
                "为记忆笔记按'下次会话最值得自动加载'排序：高频、通用、已验证的规则优先；"
                '只输出 JSON：{"ranked_ids": ["id", ...]}，'
                f"最多 {top_k} 条。"
            ),
            user=f"【笔记目录】\n{catalog}",
            tag="index",
        )
        try:
            ids = json.loads(re.search(r"\{.*\}", ranked, re.S).group(0))["ranked_ids"]
        except (AttributeError, json.JSONDecodeError, KeyError):
            ids = [n.note_id for n in
                   sorted(notes, key=lambda n: (-n.uses, n.title))[:top_k]]

        by_id = {n.note_id: n for n in notes}
        lines = ["# Memory Index (auto-loaded; details live in notes/ and wiki/)",
                 ""]
        written = 0
        for nid in ids[:top_k]:
            n = by_id.get(nid)
            if n:
                hook = re.sub(r"\s+", " ", n.body)[:80]
                lines.append(f"- [{n.title}](notes/{n.note_id}.md) — {hook}")
                written += 1
        self.injected.write_index(lines)
        return {"index_lines": len(lines), "promoted": written}

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        """One full Dreaming cycle: verify, consolidate, wiki, promote, demote."""
        logger.info("Dreaming: verify_notes")
        verify = self.verify_notes()
        logger.info("Dreaming: consolidate")
        merged = self.consolidate()
        logger.info("Dreaming: build_wiki")
        wiki = self.build_wiki()
        logger.info("Dreaming: promote_index")
        index = self.promote_index()
        logger.info("Dreaming: demote")
        demoted = self.lifecycle.demote(stale_after=self.stale_after)
        return {"verify": verify, "consolidate": merged,
                "wiki": wiki, "index": index, "demote": demoted}
