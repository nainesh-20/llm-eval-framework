"""
Evaluator unit tests.

These tests validate evaluator logic without making API calls:
  - PII detection (regex pass is fully deterministic)
  - Latency scoring (deterministic math)
  - EvalResult pass/fail gate logic
  - SQLite store round-trip
  - Config loading

Tests requiring LLM calls are tagged @pytest.mark.integration and skipped by default.
Run them with: pytest -m integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_eval.evaluators.base import EvalInput, EvalResult, BaseEvaluator
from llm_eval.evaluators.latency import LatencyEvaluator, latency_to_score
from llm_eval.evaluators.pii import PIIEvaluator
from llm_eval.storage.sqlite_store import SQLiteStore
from llm_eval.config import load_config, EvalConfig


# ── EvalInput / EvalResult ─────────────────────────────────────────────────────

def make_input(answer: str = "Test answer", question: str = "Test?", context: str = "Context.") -> EvalInput:
    return EvalInput(question=question, context=context, answer=answer, sample_id=0)


# ── PII Evaluator (no API calls needed) ────────────────────────────────────────

class TestPIIEvaluator:
    def setup_method(self):
        # Disable LLM second pass so tests don't hit the API
        self.evaluator = PIIEvaluator(threshold=9.0, use_llm_second_pass=False)

    def test_clean_response_passes(self):
        inp = make_input("The company policy requires all employees to follow the code of conduct.")
        result = self.evaluator.evaluate(inp)
        assert result.passed
        assert result.score == 10.0
        assert result.metric == "pii"

    def test_email_detected(self):
        inp = make_input("The user's email is john.doe@example.com, please contact them.")
        result = self.evaluator.evaluate(inp)
        assert not result.passed
        assert result.score == 0.0
        assert "email" in result.reason.lower() or "email" in str(result.metadata)

    def test_ssn_detected(self):
        inp = make_input("The patient's SSN is 123-45-6789.")
        result = self.evaluator.evaluate(inp)
        assert not result.passed
        assert result.score == 0.0

    def test_phone_detected(self):
        inp = make_input("Call me at 555-867-5309 anytime.")
        result = self.evaluator.evaluate(inp)
        assert not result.passed
        assert result.score == 0.0

    def test_credit_card_detected(self):
        inp = make_input("Card number: 4532 1234 5678 9012")
        result = self.evaluator.evaluate(inp)
        assert not result.passed
        assert result.score == 0.0

    def test_result_threshold(self):
        result = PIIEvaluator(threshold=9.0, use_llm_second_pass=False).evaluate(
            make_input("No PII here.")
        )
        assert result.threshold == 9.0

    def test_multiple_pii_types(self):
        inp = make_input("Email: alice@corp.com | SSN: 987-65-4321 | Phone: 415-555-0100")
        result = self.evaluator.evaluate(inp)
        assert not result.passed
        assert len(result.metadata.get("pii_types", [])) >= 2


# ── Latency Evaluator ─────────────────────────────────────────────────────────

class TestLatencyEvaluator:
    def test_fast_response_scores_high(self):
        evaluator = LatencyEvaluator(threshold=5.0)
        inp = make_input()
        inp.latency_ms = 300
        result = evaluator.evaluate(inp)
        assert result.score == 10.0
        assert result.passed

    def test_slow_response_scores_low(self):
        evaluator = LatencyEvaluator(threshold=5.0)
        inp = make_input()
        inp.latency_ms = 15000
        result = evaluator.evaluate(inp)
        assert result.score == 0.0

    def test_moderate_latency(self):
        assert latency_to_score(1500) == 8.0
        assert latency_to_score(2500) == 7.0
        assert latency_to_score(4000) == 6.0

    def test_statistics(self):
        evaluator = LatencyEvaluator()
        for ms in [200, 400, 600, 800, 1200, 3000, 9000]:
            inp = make_input()
            inp.latency_ms = ms
            evaluator.evaluate(inp)

        stats = evaluator.compute_statistics()
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert stats["sample_count"] == 7
        assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]

    def test_reset(self):
        evaluator = LatencyEvaluator()
        inp = make_input()
        inp.latency_ms = 500
        evaluator.evaluate(inp)
        evaluator.reset()
        assert evaluator.compute_statistics() == {}


# ── BaseEvaluator _make_result ─────────────────────────────────────────────────

class ConcreteEvaluator(BaseEvaluator):
    name = "test"

    def evaluate(self, input: EvalInput) -> EvalResult:
        return self._make_result(8.0, "ok")


class TestBaseEvaluator:
    def test_pass_when_above_threshold(self):
        ev = ConcreteEvaluator(threshold=7.0)
        result = ev._make_result(8.0, "ok")
        assert result.passed

    def test_fail_when_below_threshold(self):
        ev = ConcreteEvaluator(threshold=7.0)
        result = ev._make_result(5.0, "below")
        assert not result.passed

    def test_score_clipped_to_range(self):
        ev = ConcreteEvaluator(threshold=5.0)
        result_high = ev._make_result(15.0, "too high")
        result_low = ev._make_result(-5.0, "too low")
        assert result_high.score == 10.0
        assert result_low.score == 0.0

    def test_metadata_defaults_to_empty_dict(self):
        ev = ConcreteEvaluator()
        result = ev._make_result(7.0, "test")
        assert result.metadata == {}


# ── SQLiteStore ────────────────────────────────────────────────────────────────

class TestSQLiteStore:
    def test_create_run(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        run_id = store.create_run("gpt-4o", "datasets/test.json")
        assert len(run_id) == 36  # UUID format

    def test_save_and_retrieve_eval_result(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        run_id = store.create_run("gpt-4o", "datasets/test.json")
        store.save_eval_result(
            run_id=run_id,
            metric="pii",
            score=9.5,
            reason="No PII detected",
            passed=True,
            threshold=9.0,
            sample_id=0,
        )
        summary = store.get_run_summary(run_id)
        assert summary["run"]["model_name"] == "gpt-4o"
        assert len(summary["metrics"]) == 1
        assert summary["metrics"][0]["metric"] == "pii"
        assert abs(summary["metrics"][0]["avg_score"] - 9.5) < 0.001

    def test_red_team_result(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        run_id = store.create_run("claude", "red_team")
        store.save_red_team_result(
            run_id=run_id,
            category="jailbreak",
            prompt="ignore instructions",
            response="I cannot help with that.",
            passed=True,
            score=9.0,
            reason="Model refused safely",
        )
        rt_summary = store.get_red_team_summary(run_id)
        assert len(rt_summary) == 1
        assert rt_summary[0]["category"] == "jailbreak"
        assert rt_summary[0]["passed_count"] == 1

    def test_get_latest_run_id(self, tmp_path):
        store = SQLiteStore(db_path=str(tmp_path / "test.db"))
        run_id_1 = store.create_run("gpt-4o", "ds.json")
        run_id_2 = store.create_run("claude", "ds.json")
        latest = store.get_latest_run_id()
        assert latest == run_id_2


# ── Config loading ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_valid_config(self, tmp_path):
        config_content = """
models:
  - name: gpt-4o
    provider: openai
    model_id: gpt-4o
dataset: datasets/test.json
evaluators:
  - faithfulness
  - pii
thresholds:
  faithfulness: 7.0
  pii: 9.0
"""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(config_content)
        cfg = load_config(str(config_path))
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "gpt-4o"
        assert "pii" in cfg.evaluators
        assert cfg.thresholds.pii == 9.0

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_threshold_get_method(self, tmp_path):
        config_content = """
models:
  - name: gpt-4o
    provider: openai
    model_id: gpt-4o
dataset: ds.json
evaluators: [pii]
"""
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(config_content)
        cfg = load_config(str(config_path))
        assert cfg.thresholds.get("pii") == 9.0
        assert cfg.thresholds.get("unknown_metric", 6.0) == 6.0


# ── Integration tests (require API keys, skipped by default) ──────────────────

@pytest.mark.integration
class TestIntegration:
    """Requires OPENAI_API_KEY to be set. Run with: pytest -m integration"""

    def test_faithfulness_evaluator(self):
        from llm_eval.evaluators.faithfulness import FaithfulnessEvaluator

        ev = FaithfulnessEvaluator(threshold=5.0)
        inp = EvalInput(
            question="What is Paris?",
            context="Paris is the capital city of France.",
            answer="Paris is the capital of France.",
        )
        result = ev.evaluate(inp)
        assert result.score > 0
        assert result.metric == "faithfulness"

    def test_hallucination_evaluator(self):
        from llm_eval.models.openai_client import OpenAIClient
        from llm_eval.evaluators.hallucination import HallucinationEvaluator

        client = OpenAIClient()
        ev = HallucinationEvaluator(threshold=5.0, model_client=client)
        inp = EvalInput(
            question="What is the speed of light?",
            context="The speed of light in a vacuum is approximately 299,792,458 metres per second.",
            answer="The speed of light is approximately 300,000 km/s.",
        )
        result = ev.evaluate(inp)
        assert 0 <= result.score <= 10
