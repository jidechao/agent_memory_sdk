"""agent-memory-sdk: Claude-style four-layer agent memory.

Layers: Transcripts (evidence anchor) -> Memory Notes -> Wiki -> Injected
Context (tiny auto-loaded index), maintained by the offline Dreaming engine,
with zvec hybrid retrieval, a full lifecycle (feedback + demotion), agentic
retrieval tools, and health metrics.

Quick start:
    from agent_memory_sdk import MemorySystem
    mem = MemorySystem(root="./memory_data")
"""

from .agent import MemoryAgent
from .config import MemoryConfig
from .dreaming import Dreaming
from .embeddings import Embedder
from .facade import MemorySystem
from .layers import (InjectedContext, MemoryNote, NotesStore,
                     TranscriptStore, WikiPage, WikiStore)
from .lifecycle import Lifecycle
from .llm import LLMClient, TokenLedger
from .metrics import MemoryMetrics
from .retrieval import RetrievalConfig, RetrievalHit, ZvecMemoryIndex
from .tools import ToolExecutor

__all__ = [
    "MemorySystem",
    "MemoryConfig",
    "MemoryAgent",
    "Dreaming",
    "Embedder",
    "InjectedContext",
    "MemoryNote",
    "NotesStore",
    "TranscriptStore",
    "WikiPage",
    "WikiStore",
    "Lifecycle",
    "LLMClient",
    "TokenLedger",
    "MemoryMetrics",
    "RetrievalConfig",
    "RetrievalHit",
    "ZvecMemoryIndex",
    "ToolExecutor",
]

__version__ = "0.1.0"
