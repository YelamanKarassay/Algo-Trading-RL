from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Data source configuration."""

    source: str
    path: str
    symbol: str
    start: date
    end: date
    train_split: float = 0.7


class MarketConfig(BaseModel):
    """Market and execution configuration."""

    capital: int
    fee_bps_one_side: int = 20
    decision_times: list[str]
    force_close_time: str
    lot_size: int = 500


class StateConfig(BaseModel):
    """State encoder configuration."""

    encoder: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(BaseModel):
    """Reward function configuration."""

    function: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent configuration."""

    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    walk_forward: dict[str, int]
    episodes_per_window: int


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    baselines: list[str] = Field(default_factory=list)
    compute_oracle: bool = False
    feature_signal_check: bool = False


class ArtifactsConfig(BaseModel):
    """Runtime artifact paths for live bot use."""

    q_state_path: str = "artifacts/q_state.pkl"
    log_dir: str = "artifacts/logs/"
    runtime_state_path: str = "artifacts/runtime_state.json"


class ExperimentConfig(BaseModel):
    """Resolved experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    seed: int = 42
    inherits: str | None = None
    data: DataConfig
    market: MarketConfig
    state: StateConfig
    reward: RewardConfig
    agent: AgentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load, inherit, deep-merge, and validate an experiment YAML."""
    path = Path(path).resolve()
    raw = _load_yaml(path)
    if "inherits" in raw and raw["inherits"]:
        parent_path = (path.parent / raw["inherits"]).resolve()
        parent_raw = _load_resolved_raw(parent_path)
        raw = _deep_merge(parent_raw, raw)
    return ExperimentConfig.model_validate(raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    """Write a resolved experiment config as YAML."""
    path.write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


def _load_resolved_raw(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    if "inherits" in raw and raw["inherits"]:
        parent_path = (path.parent / raw["inherits"]).resolve()
        raw = _deep_merge(_load_resolved_raw(parent_path), raw)
    return raw


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Config {path} must contain a YAML mapping."
        raise ValueError(msg)
    return payload


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
