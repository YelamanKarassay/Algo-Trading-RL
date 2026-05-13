from __future__ import annotations

import csv
import json
from pathlib import Path

from apps import run_experiment


def test_end_to_end_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", tmp_path / "results")
    config = _write_experiment(tmp_path, baselines=["always_long"], episodes=2)

    result_dir = run_experiment.run(config)

    assert (result_dir / "config.snapshot.yaml").exists()
    assert (result_dir / "git_commit.txt").exists()
    assert (result_dir / "train_log.csv").exists()
    assert (result_dir / "eval_log.csv").exists()
    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "q_state.pkl").exists()


def test_baselines_run_alongside(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", tmp_path / "results")
    config = _write_experiment(tmp_path, baselines=["always_long", "random_binary"], episodes=1)

    result_dir = run_experiment.run(config)
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))

    assert "always_long" in metrics["baselines"]
    assert "random_binary" in metrics["baselines"]
    assert "final_nav" in metrics["baselines"]["always_long"]


def test_results_index_appended(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", results_root)
    config = _write_experiment(tmp_path, baselines=["always_long"], episodes=1)

    run_experiment.run(config)
    run_experiment.run(config)

    rows = list(csv.DictReader((results_root / "_index.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["name"] == "smoke" for row in rows)


def test_same_seed_produces_identical_q_state(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", results_root)
    config = _write_experiment(tmp_path, baselines=["always_long"], episodes=2)

    first = run_experiment.run(config)
    second = run_experiment.run(config)

    assert (first / "q_state.pkl").read_bytes() == (second / "q_state.pkl").read_bytes()


def _write_experiment(tmp_path: Path, baselines: list[str], episodes: int) -> Path:
    path = tmp_path / "experiment.yaml"
    baseline_text = (Path("experiments/_baseline.yaml")).read_text(encoding="utf-8")
    baselines_yaml = "\n".join(f"    - {baseline}" for baseline in baselines)
    path.write_text(
        f"""
{baseline_text}
name: smoke
seed: 123
agent:
  type: qtable
  kwargs:
    num_decision_points: 5
    num_states: 27
    num_actions: 2
    epsilon_start: 0.2
    epsilon_decay_per_episode: 0.9
    epsilon_min: 0.01
    alpha_mode: per_cell
    optimistic_init: 0.0
    tie_break: random
training:
  walk_forward:
    initial_train_days: 3
    test_window_days: 2
  episodes_per_window: {episodes}
evaluation:
  baselines:
{baselines_yaml if baselines_yaml else "    []"}
  compute_oracle: false
  feature_signal_check: false
""",
        encoding="utf-8",
    )
    return path
