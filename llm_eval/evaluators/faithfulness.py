"""
Faithfulness evaluator — measures whether the model answer is grounded in the context.

Primary:  RAGAS faithfulness metric (industry-standard RAG evaluation library)
Fallback: LLM-as-judge if RAGAS is unavailable or fails
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .base import BaseEvaluator, EvalInput, EvalResult, llm_judge_call

if TYPE_CHECKING:
    from llm_eval.models.base import BaseModelClient

logger = logging.getLogger(__name__)

FAITHFULNESS_JUDGE_PROMPT = """Question: {question}

Context provided to the model:
{context}

Model answer:
{answer}

Evaluate whether the model answer is faithful to and grounded in the provided context.
- Score 10: Every claim in the answer is directly supported by the context
- Score 7-9: Most claims are supported; minor extrapolations acceptable
- Score 4-6: Some claims are unsupported or go beyond the context
- Score 0-3: Answer contradicts the context or contains fabricated information"""


class FaithfulnessEvaluator(BaseEvaluator):
    """
    RAGAS faithfulness wrapper with LLM-as-judge fallback.

    RAGAS faithfulness measures: of all the claims in the answer,
    what fraction are supported by the retrieved context?
    """

    name = "faithfulness"

    def __init__(
        self,
        threshold: float = 7.0,
        model_client: Optional["BaseModelClient"] = None,
    ):
        super().__init__(threshold=threshold, model_client=model_client)

    def evaluate(self, input: EvalInput) -> EvalResult:
        score, reason = self._try_ragas(input)
        if score is None:
            score, reason = self._llm_judge(input)
        return self._make_result(score, reason)

    def _try_ragas(self, input: EvalInput) -> tuple[Optional[float], Optional[str]]:
        """Attempt RAGAS evaluation. Returns (None, None) if unavailable."""
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness
            from datasets import Dataset

            context = input.context if isinstance(input.context, list) else [input.context]
            dataset = Dataset.from_dict({
                "question": [input.question],
                "contexts": [context],
                "answer": [input.answer],
            })

            result = ragas_evaluate(dataset, metrics=[faithfulness])

            # RAGAS returns a Results object; score may be float or pandas Series
            raw = result["faithfulness"]
            score_val = float(raw[0]) if hasattr(raw, "__getitem__") else float(raw)
            # Normalize from 0–1 to 0–10
            return score_val * 10, f"RAGAS faithfulness: {score_val:.3f}"

        except ImportError:
            logger.debug("RAGAS not installed; using LLM-judge fallback")
            return None, None
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed ({e}); using LLM-judge fallback")
            return None, None

    def _llm_judge(self, input: EvalInput) -> tuple[float, str]:
        if self.model_client is None:
            logger.warning("No model_client for faithfulness fallback; returning neutral score")
            return 5.0, "No judge model available"

        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            question=input.question,
            context=input.context,
            answer=input.answer,
        )
        return llm_judge_call(self.model_client, prompt, "faithfulness to context")
