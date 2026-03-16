"""OpenAI model client with built-in latency tracking."""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import BaseModelClient, ModelResponse


class OpenAIClient(BaseModelClient):
    def __init__(self, model_id: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_id)
        from openai import OpenAI  # lazy import so the module loads without the key

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs.setdefault("temperature", 0)  # deterministic evals by default

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return ModelResponse(
            text=response.choices[0].message.content or "",
            model_id=self.model_id,
            latency_ms=latency_ms,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
