"""
Base model client ABC.

Every model client wraps generate() with built-in latency tracking
so the EvalRunner always has timing data without extra instrumentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelResponse:
    text: str
    model_id: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class BaseModelClient(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response. Latency is measured inside the implementation."""
        ...
