"""Thin LLM client abstraction.

`parse_structured()` returns a Pydantic instance of `response_model`. Callers should
pass `context_text` for shared content that benefits from prompt caching (e.g. the
patient span dump — same across every section call in a session).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Protocol, Type, TypeVar

import httpx
from pydantic import BaseModel

from ..config import config


T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def parse_structured(
        self,
        *,
        system: str,
        context_text: str,
        user_text: str,
        response_model: Type[T],
    ) -> T: ...


# ---------- Claude (Anthropic SDK) ----------


class ClaudeClient:
    name = "claude"

    def __init__(self, model: Optional[str] = None) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY not set")
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise LLMUnavailable("anthropic SDK not installed") from e

        from anthropic import Anthropic

        self._client = Anthropic()
        self.model = model or config.claude_model

    def parse_structured(
        self,
        *,
        system: str,
        context_text: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        # Cache the shared context (patient spans) so subsequent section calls only
        # pay for the small per-section suffix. Skip the cached block when context is
        # empty (the plan-generation path).
        if context_text:
            content = [
                {
                    "type": "text",
                    "text": context_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            content = [{"type": "text", "text": user_text}]

        response = self._client.messages.parse(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=response_model,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise RuntimeError(
                f"Claude returned no parseable output (stop_reason={response.stop_reason})"
            )
        return parsed


# ---------- Ollama (local) ----------


class OllamaClient:
    name = "ollama"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.model = model or config.ollama_model
        self.base_url = (base_url or config.ollama_url).rstrip("/")
        # Probe; raises if unreachable.
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            r.raise_for_status()
        except Exception as e:
            raise LLMUnavailable(f"Ollama unreachable at {self.base_url}: {e}") from e

    def parse_structured(
        self,
        *,
        system: str,
        context_text: str,
        user_text: str,
        response_model: Type[T],
    ) -> T:
        schema = response_model.model_json_schema()
        prompt = f"{context_text}\n\n---\n\n{user_text}"
        r = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "format": schema,  # Ollama 0.5+ supports JSON schema-constrained output
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=600.0,
        )
        r.raise_for_status()
        content = r.json()["message"]["content"]
        return response_model.model_validate(json.loads(content))


# ---------- Factory ----------


def make_llm_client() -> Optional[LLMClient]:
    """Try the configured backend; return None if unavailable so the caller can
    fall back to deterministic-only output."""
    backend = config.llm_backend
    try:
        if backend == "claude":
            return ClaudeClient()
        if backend == "ollama":
            return OllamaClient()
        if backend == "none":
            return None
    except LLMUnavailable:
        return None
    return None
