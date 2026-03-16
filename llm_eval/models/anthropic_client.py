"""Anthropic Claude client with built-in latency tracking."""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import BaseModelClient, ModelResponse


class AnthropicClient(BaseModelClient):
    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_id)
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        kwargs.setdefault("max_tokens", 1024)

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_id,
            system=system or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            text=response.content[0].text,
            model_id=self.model_id,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
