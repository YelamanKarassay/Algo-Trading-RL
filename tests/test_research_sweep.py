from __future__ import annotations

import json
from pathlib import Path

import yaml

from apps import research_sweep
from apps.research_report import generate_research_report


def test_core_sweep_count_and_decision_times() -> None:
    specs = research_sweep._specs("core")

    assert len(specs) == 5 * 3 * 3 * 4 * 2
    assert research_sweep._decision_times("1h") == ["10:30", "11:30", "14:00", "15:00"]
    assert research_sweep._decision_times("30m")[-1] == "15:30"
    assert "12:00" not in research_sweep._decision_times("30m")
    assert len(research_sweep._decision_times("5m")) == 54
    assert research_sweep._training_params("1m")["initial_train_days"] == 60


def test_research_config_sets_action_space_and_scaled_thresholds() -> None:
    spec = research_sweep.ResearchSpec(
        asset="7299.HK",
        interval="30m",
        reward="fee_aware",
        encoder="with_rsi",
        action_space="ternary",
    )

    config = research_sweep._config(spec)

    assert config["agent"]["kwargs"]["num_actions"] == 3
    assert config["agent"]["kwargs"]["num_states"] == 81
    assert config["reward"]["function"] == "fee_aware"
    assert config["data"]["path"] == "data/raw/webull_7299_hk_m30.csv"
    assert config["state"]["kwargs"]["thresholds"]["delta_close"] == [-0.0192857143, 0.0192857143]


def test_research_report_renders_completed_results(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "metrics.json").write_text(
        json.dumps(
            {
                "final_nav": 1_010_000.0,
                "return": 0.01,
                "alpha_vs_bh": 0.002,
                "sharpe": 1.2,
                "num_trades": 4,
                "total_fees": 800.0,
                "pct_states_visited": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "research_smoke",
                "data": {"symbol": "2800.HK", "path": "data/raw/webull_2800_hk_m30.csv"},
                "market": {"capital": 1_000_000, "decision_times": ["10:00"]},
                "reward": {"function": "log_return"},
                "state": {"encoder": "three_feature"},
                "agent": {"kwargs": {"num_actions": 2}},
            }
        ),
        encoding="utf-8",
    )

    out = generate_research_report([result_dir], tmp_path / "research.html")

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "Quantphemes RL ETF Research Report" in html
    assert "research_smoke" in html
