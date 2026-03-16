"""
Config loader — parses eval.yaml into validated Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    provider: str  # "openai" | "anthropic"
    model_id: str


class ThresholdConfig(BaseModel):
    faithfulness: float = 7.0
    hallucination: float = 7.0
    pii: float = 9.0
    toxicity: float = 8.0
    latency: float = 5.0

    def get(self, metric: str, default: float = 7.0) -> float:
        return getattr(self, metric, default)


class RedTeamConfig(BaseModel):
    enabled: bool = False
    categories: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    sqlite_path: str = "results/evals.db"
    results_json: str = "results/latest.json"


class EvalConfig(BaseModel):
    models: list[ModelConfig]
    dataset: str
    evaluators: list[str]
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    red_team: RedTeamConfig = Field(default_factory=RedTeamConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: str) -> EvalConfig:
    """Load and validate eval.yaml config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run from the project root or pass --config with the correct path."
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalConfig(**data)
