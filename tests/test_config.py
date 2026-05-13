from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantphemes_rl.config import load_config


def test_baseline_loads() -> None:
    cfg = load_config(Path("experiments/_baseline.yaml"))

    assert cfg.name == "baseline_2800_1h_qtable"
    assert cfg.data.symbol == "2800.HK"
    assert cfg.market.capital == 1_000_000
    assert cfg.state.encoder == "three_feature"


def test_inherits_deep_merge(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        """
name: parent
description: parent config
seed: 42
data:
  source: csv
  path: tests/fixtures/synthetic_5days_1h.csv
  symbol: 2800.HK
  start: 2026-05-04
  end: 2026-05-08
market:
  capital: 1000000
  fee_bps_one_side: 20
  decision_times: ["10:30", "11:30"]
  force_close_time: "16:00"
state:
  encoder: three_feature
  kwargs:
    thresholds:
      delta_close: [-0.01, 0.01]
      delta_open: [-0.02, 0.02]
      delta_prev: [-0.03, 0.03]
reward:
  function: log_return
agent:
  type: qtable
  kwargs:
    num_decision_points: 2
    num_states: 27
training:
  walk_forward:
    initial_train_days: 3
    test_window_days: 2
  episodes_per_window: 1
evaluation:
  baselines: []
  compute_oracle: false
  feature_signal_check: false
""",
        encoding="utf-8",
    )
    child.write_text(
        """
inherits: parent.yaml
name: child
market:
  fee_bps_one_side: 10
agent:
  kwargs:
    epsilon_start: 0.0
""",
        encoding="utf-8",
    )

    cfg = load_config(child)

    assert cfg.name == "child"
    assert cfg.market.capital == 1_000_000
    assert cfg.market.fee_bps_one_side == 10
    assert cfg.agent.kwargs["num_decision_points"] == 2
    assert cfg.agent.kwargs["epsilon_start"] == 0.0


def test_invalid_yaml_raises_pydantic(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)
