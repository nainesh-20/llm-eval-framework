"""
Latency evaluator — tracks p50/p95/p99 response times.

Latency is measured at the model client level (time.perf_counter around the API call).
The EvalRunner collects latency_ms from each ModelResponse and passes it via EvalInput.

Score mapping (higher = faster is better):
  <500ms   → 10
  <1000ms  → 9
  <2000ms  → 8
  <3000ms  → 7
  <5000ms  → 6
  <8000ms  → 4
  <12000ms → 2
  ≥12000ms → 0
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import BaseEvaluator, EvalInput, EvalResult

logger = logging.getLogger(__name__)

# (upper_bound_ms, score) — first match wins
LATENCY_SCORE_BINS = [
    (500,   10.0),
    (1000,  9.0),
    (2000,  8.0),
    (3000,  7.0),
    (5000,  6.0),
    (8000,  4.0),
    (12000, 2.0),
]


def latency_to_score(latency_ms: float) -> float:
    for upper_ms, score in LATENCY_SCORE_BINS:
        if latency_ms < upper_ms:
            return score
    return 0.0


class LatencyEvaluator(BaseEvaluator):
    """
    Per-sample latency scorer. Collect results and call compute_statistics()
    at the end of a run for aggregate p50/p95/p99 metrics.
    """

    name = "latency"

    def __init__(self, threshold: float = 5.0):
        super().__init__(threshold=threshold, model_client=None)
        self._latency_samples: list[float] = []

    def evaluate(self, input: EvalInput) -> EvalResult:
        latency_ms = input.latency_ms
        self._latency_samples.append(latency_ms)

        score = latency_to_score(latency_ms)
        reason = f"Response time: {latency_ms:.0f}ms"

        return self._make_result(
            score=score,
            reason=reason,
            metadata={"latency_ms": latency_ms},
        )

    def compute_statistics(self) -> dict:
        """Call after a full run to get aggregate latency stats."""
        if not self._latency_samples:
            return {}
        arr = np.array(self._latency_samples)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "sample_count": len(arr),
        }

    def reset(self) -> None:
        self._latency_samples = []
