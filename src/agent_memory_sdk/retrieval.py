"""zvec-backed retrieval index for the memory layers.

One zvec collection mirrors the lower memory layers (L2 notes, L3 wiki pages)
and replaces the naive keyword scan with real retrieval:

  - mode "hybrid" (default): dense vector query + full-text query fused by
    zvec's native RrfReRanker (Reciprocal Rank Fusion)
  - mode "fts":    full-text only (built-in Jieba Chinese tokenization)
  - mode "dense":  vector only (Ollama embeddings, e.g. bge-m3)

Configuration (via .env):
  MEMORY_SEARCH_MODE=hybrid|fts|dense   (default: hybrid)
  EMBEDDING_MODEL=bge-m3
  OLLAMA_BASE_URL=http://localhost:11434
  RRF_RANK_CONSTANT=60
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import zvec
from zvec import Fts

from .embeddings import Embedder

if TYPE_CHECKING:
    from .layers import MemoryNote, WikiPage

logger = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "fts", "dense"]
VALID_MODES: tuple[SearchMode, ...] = ("hybrid", "fts", "dense")


@dataclass(frozen=True)
class RetrievalHit:
    ref_id: str        # note_id or wiki slug
    layer: str         # "note" | "wiki"
    title: str
    score: float


@dataclass(frozen=True)
class RetrievalConfig:
    mode: SearchMode = "hybrid"
    embedding_model: str = "bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    rrf_rank_constant: int = 60

    @staticmethod
    def from_env() -> "RetrievalConfig":
        from dotenv import load_dotenv

        load_dotenv()  # make sure .env is visible regardless of import order
        mode = os.environ.get("MEMORY_SEARCH_MODE", "hybrid").lower()
        if mode not in VALID_MODES:
            logger.warning("unknown MEMORY_SEARCH_MODE=%r, using 'hybrid'", mode)
            mode = "hybrid"
        return RetrievalConfig(
            mode=mode,  # type: ignore[arg-type]
            embedding_model=os.environ.get("EMBEDDING_MODEL", "bge-m3"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL",
                                           "http://localhost:11434"),
            rrf_rank_constant=int(os.environ.get("RRF_RANK_CONSTANT", "60")),
        )


class ZvecMemoryIndex:
    """Mirror of the memory layers inside a single zvec collection."""

    def __init__(self, root: Path, config: RetrievalConfig,
                 embedder: Embedder | None):
        self.config = config
        self.embedder = embedder
        self.path = root / "index" / "memory"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.needs_resync = False
        self._col = self._open_or_create()

        if self.effective_mode != config.mode:
            logger.warning("MEMORY_SEARCH_MODE=%s requested but dense vectors "
                           "are unavailable; effective mode: %s",
                           config.mode, self.effective_mode)

    # ------------------------------------------------------------------ #
    @property
    def effective_mode(self) -> SearchMode:
        """Configured mode, degraded to FTS when no embedder is available."""
        if self.embedder is None:
            return "fts"
        return self.config.mode

    def _schema(self) -> zvec.CollectionSchema:
        fields = [
            zvec.FieldSchema("layer", zvec.DataType.STRING,
                             index_param=zvec.InvertIndexParam()),
            zvec.FieldSchema("ref_id", zvec.DataType.STRING),
            zvec.FieldSchema("title", zvec.DataType.STRING),
            zvec.FieldSchema("text", zvec.DataType.STRING,
                             index_param=zvec.FtsIndexParam()),
            zvec.FieldSchema("status", zvec.DataType.STRING,
                             index_param=zvec.InvertIndexParam()),
            zvec.FieldSchema("uses", zvec.DataType.INT64),
        ]
        vectors = None
        if self.embedder is not None:
            vectors = zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32,
                                        self.embedder.dim)
        return zvec.CollectionSchema(name="memory", vectors=vectors,
                                     fields=fields)

    def _open_or_create(self) -> zvec.Collection:
        """Open the existing collection, or (re)build it when stale.

        The index is a derived cache — notes/ and wiki/ markdown files are the
        source of truth. If the embedding model (and with it the vector
        dimension) changed since the index was built, the whole index is
        dropped and flagged for re-sync.
        """
        meta_path = self.path / "index_meta.json"
        want_meta = {
            "model": self.embedder.model_name if self.embedder else None,
            "dim": self.embedder.dim if self.embedder else None,
        }
        if self.path.exists() and any(self.path.iterdir()):
            stale = True
            if meta_path.exists():
                try:
                    stale = json.loads(meta_path.read_text(
                        encoding="utf-8")) != want_meta
                except (json.JSONDecodeError, OSError):
                    stale = True
            if stale:
                logger.info("embedding model changed (%s); rebuilding index",
                            want_meta)
                shutil.rmtree(self.path, ignore_errors=True)
                self.needs_resync = True
            else:
                try:
                    return zvec.open(str(self.path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("index corrupt (%s); rebuilding", exc)
                    shutil.rmtree(self.path, ignore_errors=True)
                    self.needs_resync = True
        # zvec.create_and_open expects a path that does not exist yet
        shutil.rmtree(self.path, ignore_errors=True)
        col = zvec.create_and_open(str(self.path), schema=self._schema())
        meta_path.write_text(json.dumps(want_meta), encoding="utf-8")
        return col

    # ------------------------------------------------------------------ #
    # Write path: stores call these on every mutation
    # ------------------------------------------------------------------ #
    def _doc_id(self, layer: str, ref_id: str) -> str:
        return f"{layer}:{ref_id}"

    def _upsert(self, layer: str, ref_id: str, title: str, text: str,
                status: str, uses: int) -> None:
        doc_id = self._doc_id(layer, ref_id)
        vectors: dict | None = None
        if self.embedder is not None:
            vectors = {"embedding": self.embedder.embed([f"{title}\n{text}"])[0]}
        doc = zvec.Doc(
            id=doc_id,
            vectors=vectors,
            fields={"layer": layer, "ref_id": ref_id, "title": title,
                    "text": f"{title}\n{text}", "status": status, "uses": uses},
        )
        self._col.upsert([doc])

    def upsert_note(self, note: "MemoryNote") -> None:
        self._upsert("note", note.note_id, note.title, note.body,
                     note.status, note.uses)

    def delete_note(self, note_id: str) -> None:
        self._col.delete([self._doc_id("note", note_id)])

    def upsert_page(self, page: "WikiPage") -> None:
        self._upsert("wiki", page.slug, page.title, page.body, "stable", 0)

    def delete_page(self, slug: str) -> None:
        self._col.delete([self._doc_id("wiki", slug)])

    def sync(self, notes: list["MemoryNote"], pages: list["WikiPage"]) -> None:
        """Bulk re-index everything (used on startup / after rebuild)."""
        for n in notes:
            self.upsert_note(n)
        for p in pages:
            self.upsert_page(p)
        self._col.flush()
        logger.info("index synced: %d notes, %d wiki pages", len(notes), len(pages))

    # ------------------------------------------------------------------ #
    # Read path
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 5,
               layers: tuple[str, ...] = ("note", "wiki")) -> list[RetrievalHit]:
        mode = self.effective_mode
        queries: list[zvec.Query] = []
        if mode in ("fts", "hybrid"):
            queries.append(zvec.Query(field_name="text",
                                      fts=Fts(match_string=query)))
        if mode in ("dense", "hybrid") and self.embedder is not None:
            queries.append(zvec.Query(
                field_name="embedding",
                vector=self.embedder.embed([query])[0]))

        reranker = None
        if mode == "hybrid" and len(queries) > 1:
            reranker = zvec.RrfReRanker(
                rank_constant=self.config.rrf_rank_constant)

        filter_expr = None
        if len(layers) == 1:
            filter_expr = f"layer = '{layers[0]}'"

        docs = self._col.query(queries, topk=top_k, filter=filter_expr,
                               reranker=reranker)
        return [RetrievalHit(ref_id=d.fields["ref_id"], layer=d.fields["layer"],
                             title=d.fields["title"], score=d.score)
                for d in docs]

    def close(self) -> None:
        self._col.close()
