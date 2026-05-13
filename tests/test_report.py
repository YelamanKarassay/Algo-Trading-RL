from __future__ import annotations

from pathlib import Path

from apps import run_experiment


def test_report_renders_without_oracle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", tmp_path / "results")
    config = _write_config(tmp_path, compute_oracle=False)

    result_dir = run_experiment.run(config)
    report = result_dir / "report.html"

    html = report.read_text(encoding="utf-8")
    assert report.exists()
    assert "Oracle disabled for this run." in html
    assert "State Coverage Heatmap" in html


def test_report_renders_with_oracle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_experiment, "RESULTS_ROOT", tmp_path / "results")
    config = _write_config(tmp_path, compute_oracle=True)

    result_dir = run_experiment.run(config)

    html = (result_dir / "report.html").read_text(encoding="utf-8")
    assert "Oracle final NAV" in html


def _write_config(tmp_path: Path, compute_oracle: bool) -> Path:
    path = tmp_path / "report.yaml"
    text = Path("experiments/_baseline.yaml").read_text(encoding="utf-8")
    path.write_text(
        f"""
{text}
name: report_smoke
seed: 42
training:
  walk_forward:
    initial_train_days: 3
    test_window_days: 2
  episodes_per_window: 1
evaluation:
  baselines:
    - always_long
    - buy_and_hold
  compute_oracle: {str(compute_oracle).lower()}
  feature_signal_check: true
""",
        encoding="utf-8",
    )
    return path
