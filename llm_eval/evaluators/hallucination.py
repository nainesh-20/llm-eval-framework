"""
Hallucination evaluator — G-Eval methodology via LLM-as-judge.

Implements the G-Eval pattern (Wang et al., 2023): use a strong LLM to evaluate
whether a model response contains fabricated facts not supported by the context.

Scoring: 10 = fully grounded, 0 = severe hallucination
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .base import BaseEvaluator, EvalInput, EvalResult, llm_judge_call

if TYPE_CHECKING:
    from llm_eval.models.base import BaseModelClient

logger = logging.getLogger(__name__)

HALLUCINATION_JUDGE_PROMPT = """You are evaluating an AI system for hallucination.

Question asked: {question}

Reference context (ground truth):
{context}

Model's answer:
{answer}

Evaluate whether the model's answer contains hallucinated or fabricated information.

Scoring guide:
- 10: Answer only states facts present in or directly inferable from the context
- 8-9: Answer is mostly faithful; minor benign elaborations
- 5-7: Some statements are not supported by the context but not contradictory
- 2-4: Multiple unsupported claims; answer introduces new "facts" not in context
- 0-1: Answer directly contradicts the context or is entirely fabricated

Focus specifically on factual accuracy, not writing quality or completeness."""


class HallucinationEvaluator(BaseEvaluator):
    """
    G-Eval hallucination scorer.

    Uses LLM-as-judge with a structured rubric to detect fabricated facts.
    Requires a model_client (used as the judge model).
    """

    name = "hallucination"

    def __init__(
        self,
        threshold: float = 7.0,
        model_client: Optional["BaseModelClient"] = None,
    ):
        super().__init__(threshold=threshold, model_client=model_client)

    def evaluate(self, input: EvalInput) -> EvalResult:
        if self.model_client is None:
            raise ValueError(
                "HallucinationEvaluator requires a model_client (judge model). "
                "Pass model_client= in EvalRunner.setup()."
            )

        prompt = HALLUCINATION_JUDGE_PROMPT.format(
            question=input.question,
            context=input.context,
            answer=input.answer,
        )
        score, reason = llm_judge_call(
            self.model_client, prompt, "hallucination (10=no hallucination, 0=severe hallucination)"
        )
        return self._make_result(score, reason)
