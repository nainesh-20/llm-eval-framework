"""
llm_eval — Production-grade LLM evaluation & red-teaming framework.

Quick start:
    from llm_eval import EvalRunner, load_config
    cfg = load_config("configs/eval.yaml")
    runner = EvalRunner(cfg)
    runner.setup()
    run_id = runner.run("gpt-4o")
"""

from .runner import EvalRunner
from .config import load_config, EvalConfig

__version__ = "0.1.0"
__all__ = ["EvalRunner", "load_config", "EvalConfig"]
