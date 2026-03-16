"""
EvalRunner — orchestrates the full evaluation pipeline.

Architecture:
  1. setup()  → instantiate model clients and evaluators from config
  2. run(model_name) → generate answers + score each evaluator + persist to SQLite
  3. run_red_team(model_name) → run adversarial eval suite

Evaluators are registered by name via a dict (plugin pattern).
Adding a new evaluator = subclass BaseEvaluator + register it in setup().
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .config import EvalConfig
from .evaluators.base import BaseEvaluator, EvalInput
from .evaluators.faithfulness import FaithfulnessEvaluator
from .evaluators.hallucination import HallucinationEvaluator
from .evaluators.latency import LatencyEvaluator
from .evaluators.pii import PIIEvaluator
from .evaluators.toxicity import ToxicityEvaluator
from .models.base import BaseModelClient
from .storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.store = SQLiteStore(config.output.sqlite_path)
        self._evaluators: dict[str, BaseEvaluator] = {}
        self._model_clients: dict[str, BaseModelClient] = {}
        self._judge_client: Optional[BaseModelClient] = None

    # ── Plugin registration ────────────────────────────────────────────────────

    def register_evaluator(self, name: str, evaluator: BaseEvaluator) -> None:
        self._evaluators[name] = evaluator

    def register_model(self, name: str, client: BaseModelClient) -> None:
        self._model_clients[name] = client

    # ── Setup (called once before run) ────────────────────────────────────────

    def setup(self) -> None:
        """Instantiate model clients and evaluators from config."""
        self._setup_models()
        # The first model in config acts as the LLM judge for evaluators
        self._judge_client = next(iter(self._model_clients.values()), None)
        self._setup_evaluators()

    def _setup_models(self) -> None:
        from .models.openai_client import OpenAIClient
        from .models.anthropic_client import AnthropicClient

        for model_cfg in self.config.models:
            if model_cfg.provider == "openai":
                client = OpenAIClient(
                    model_id=model_cfg.model_id,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
            elif model_cfg.provider == "anthropic":
                client = AnthropicClient(
                    model_id=model_cfg.model_id,
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            else:
                logger.warning(f"Unknown provider '{model_cfg.provider}'; skipping {model_cfg.name}")
                continue
            self.register_model(model_cfg.name, client)

    def _setup_evaluators(self) -> None:
        t = self.config.thresholds
        judge = self._judge_client

        registry = {
            "faithfulness": FaithfulnessEvaluator(threshold=t.faithfulness, model_client=judge),
            "hallucination": HallucinationEvaluator(threshold=t.hallucination, model_client=judge),
            "pii":           PIIEvaluator(threshold=t.pii, model_client=judge),
            "toxicity":      ToxicityEvaluator(threshold=t.toxicity, model_client=judge),
            "latency":       LatencyEvaluator(threshold=t.latency),
        }

        for name in self.config.evaluators:
            if name in registry:
                self.register_evaluator(name, registry[name])
            else:
                logger.warning(f"Unknown evaluator '{name}'; skipping")

    # ── Main evaluation run ────────────────────────────────────────────────────

    def run(self, model_name: str) -> str:
        """
        Run the full eval suite for one model.
        Returns the run_id stored in SQLite.
        """
        from .reporting.cli_report import CLIReporter

        client = self._model_clients.get(model_name)
        if client is None:
            raise ValueError(
                f"No model client registered for '{model_name}'. "
                f"Available: {list(self._model_clients.keys())}"
            )

        dataset = self._load_dataset()
        run_id = self.store.create_run(
            model_name=model_name,
            dataset=self.config.dataset,
            config_path=None,
        )

        reporter = CLIReporter()
        reporter.print_run_header(model_name, len(dataset))

        latency_evaluator: Optional[LatencyEvaluator] = self._evaluators.get("latency")  # type: ignore
        if latency_evaluator:
            latency_evaluator.reset()

        for idx, sample in enumerate(dataset):
            question = sample["question"]
            context = sample.get("context", "")

            # Generate answer and capture latency
            response = client.generate(prompt=question)

            eval_input = EvalInput(
                question=question,
                context=context,
                answer=response.text,
                sample_id=idx,
                latency_ms=response.latency_ms,
            )

            for eval_name, evaluator in self._evaluators.items():
                try:
                    result = evaluator.evaluate(eval_input)
                    self.store.save_eval_result(
                        run_id=run_id,
                        metric=result.metric,
                        score=result.score,
                        reason=result.reason,
                        passed=result.passed,
                        threshold=result.threshold,
                        sample_id=idx,
                        metadata=result.metadata,
                    )
                except Exception as e:
                    logger.error(f"Evaluator '{eval_name}' failed on sample {idx}: {e}")

            reporter.print_progress(idx + 1, len(dataset))

        # Compute latency stats and log
        if latency_evaluator:
            stats = latency_evaluator.compute_statistics()
            logger.info(f"Latency stats: {stats}")

        summary = self.store.get_run_summary(run_id)
        reporter.print_summary(summary)
        self._export_json(run_id, summary)

        return run_id

    # ── Red-team run ───────────────────────────────────────────────────────────

    def run_red_team(self, model_name: str, run_id: Optional[str] = None) -> str:
        """Run adversarial red-team evaluation. Returns run_id."""
        from .red_team.runner import RedTeamRunner
        from .reporting.cli_report import CLIReporter

        client = self._model_clients.get(model_name)
        if client is None:
            raise ValueError(f"No model client for '{model_name}'")

        if run_id is None:
            run_id = self.store.create_run(
                model_name=model_name,
                dataset="red_team",
            )

        rt_runner = RedTeamRunner(
            model_client=client,
            store=self.store,
            judge_client=self._judge_client,
        )

        reporter = CLIReporter()
        reporter.print_red_team_header(model_name)

        categories = self.config.red_team.categories
        results = rt_runner.run(run_id=run_id, categories=categories)

        summary = self.store.get_red_team_summary(run_id)
        reporter.print_red_team_summary(summary)

        return run_id

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_dataset(self) -> list[dict]:
        path = Path(self.config.dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _export_json(self, run_id: str, summary: dict) -> None:
        output_path = Path(self.config.output.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_id, **summary}, f, indent=2, default=str)
