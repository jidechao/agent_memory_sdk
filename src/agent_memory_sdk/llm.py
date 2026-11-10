"""LLM client wrapper with injectable backends and per-tag token accounting.

Three construction modes:
  1. default      - build an OpenAI-compatible client from .env
                    (DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL)
  2. client=...   - any object exposing .chat.completions.create(...)
                    (an OpenAI() instance, AzureOpenAI, a proxy, ...)
  3. chat_fn=...  - a plain Callable[[system, user], str] for full control

Every call is recorded in a TokenLedger (tag, tokens, elapsed) so the
health-metrics layer can report maintenance cost per pipeline stage.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class ChatModel(Protocol):
    """Anything that can answer a system+user chat call."""

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float) -> "tuple[str, tuple[int | None, int | None]]":
        ...


# --------------------------------------------------------------------------- #
# Token accounting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenEntry:
    tag: str                     # agent | reflect | verify | consolidate | wiki | index
    prompt_tokens: int | None    # None when the backend does not report usage
    completion_tokens: int | None
    elapsed: float


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str               # JSON string, as emitted by the model


@dataclass(frozen=True)
class ToolCallResult:
    content: str
    tool_calls: list["ToolCall"]


@dataclass
class TokenLedger:
    entries: list[TokenEntry] = field(default_factory=list)

    def record(self, tag: str, prompt_tokens: int | None,
               completion_tokens: int | None, elapsed: float) -> None:
        self.entries.append(TokenEntry(tag, prompt_tokens,
                                       completion_tokens, elapsed))

    def totals(self) -> dict[str, dict[str, float]]:
        """{tag: {calls, prompt, completion, elapsed}}"""
        out: dict[str, dict[str, float]] = {}
        for e in self.entries:
            t = out.setdefault(e.tag, {"calls": 0, "prompt": 0,
                                       "completion": 0, "elapsed": 0.0})
            t["calls"] += 1
            t["prompt"] += e.prompt_tokens or 0
            t["completion"] += e.completion_tokens or 0
            t["elapsed"] += e.elapsed
        return out

    def reset(self) -> None:
        self.entries.clear()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _OpenAICompatibleBackend:
    """Wraps any .chat.completions.create-style client."""

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def raw_create(self, messages: list[dict], tools: list[dict] | None,
                   max_tokens: int, temperature: float):
        kwargs: dict = {
            "model": self._model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float) -> tuple[str, tuple[int | None, int | None]]:
        resp = self.raw_create(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            None, max_tokens, temperature)
        usage = getattr(resp, "usage", None)
        tokens = (getattr(usage, "prompt_tokens", None),
                  getattr(usage, "completion_tokens", None))
        return (resp.choices[0].message.content or "").strip(), tokens


class _CallableBackend:
    def __init__(self, fn: Callable[[str, str], str]):
        self._fn = fn

    def chat(self, system: str, user: str, max_tokens: int,
             temperature: float) -> tuple[str, tuple[int | None, int | None]]:
        return self._fn(system, user).strip(), (None, None)


# --------------------------------------------------------------------------- #
class LLMClient:
    """Uniform chat interface over any backend, with token accounting."""

    def __init__(self, env_path: str | None = None, *, client=None,
                 model: str | None = None,
                 chat_fn: Callable[[str, str], str] | None = None,
                 ledger: TokenLedger | None = None):
        self.ledger = ledger or TokenLedger()
        if chat_fn is not None:
            self._backend: ChatModel = _CallableBackend(chat_fn)
            self.model = model or "custom-callable"
        elif client is not None:
            if model is None:
                raise ValueError("model is required when injecting a client")
            self._backend = _OpenAICompatibleBackend(client, model)
            self.model = model
        else:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv(env_path or ".env")
            missing = [k for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
                                   "DEEPSEEK_MODEL") if not os.environ.get(k)]
            if missing:
                raise ValueError(
                    f"LLM 默认后端缺少环境变量: {', '.join(missing)}。"
                    f"请在 .env 中配置，或改用 client=/chat_fn= 注入自己的后端。")
            self.model = model or os.environ["DEEPSEEK_MODEL"]
            self._backend = _OpenAICompatibleBackend(
                OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                       base_url=os.environ["DEEPSEEK_BASE_URL"]),
                self.model,
            )

    def chat(self, system: str, user: str, max_tokens: int = 4096,
             temperature: float = 0.3, tag: str = "agent") -> str:
        start = time.perf_counter()
        text, (ptok, ctok) = self._backend.chat(system, user, max_tokens,
                                                temperature)
        self.ledger.record(tag, ptok, ctok, time.perf_counter() - start)
        return text

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        max_tokens: int = 4096, temperature: float = 0.3,
                        tag: str = "agent") -> "ToolCallResult":
        """One round of function-calling. Only for OpenAI-style backends."""
        if not isinstance(self._backend, _OpenAICompatibleBackend):
            raise NotImplementedError(
                "tool calling requires an OpenAI-compatible backend")
        start = time.perf_counter()
        resp = self._backend.raw_create(messages, tools, max_tokens,
                                        temperature)
        usage = getattr(resp, "usage", None)
        self.ledger.record(tag, getattr(usage, "prompt_tokens", None),
                           getattr(usage, "completion_tokens", None),
                           time.perf_counter() - start)
        msg = resp.choices[0].message
        calls = [ToolCall(id=tc.id, name=tc.function.name,
                          arguments=tc.function.arguments or "{}")
                 for tc in (msg.tool_calls or [])]
        return ToolCallResult(content=(msg.content or "").strip(),
                              tool_calls=calls)
