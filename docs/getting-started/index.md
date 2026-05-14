# Getting Started

This page gets a fresh checkout into a verified local development state.

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Verify The Repo

```bash
ruff check .
pytest -q
```

The test suite is hermetic: it should not call the broker or any real network endpoint.

## Run A Baseline Experiment

```bash
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
```

The runner creates a timestamped folder under `results/` with:

| File | Meaning |
|---|---|
| `config.snapshot.yaml` | Resolved experiment config for reproducibility |
| `git_commit.txt` | Source revision or dirty-worktree marker |
| `train_log.csv` | Per-episode training summaries |
| `eval_log.csv` | Greedy evaluation path |
| `metrics.json` | Summary metrics and baseline comparison |
| `q_state.pkl` | Trained Q-table artifact |
| `report.html` | Diagnostics report |

## Compare Runs

```bash
python -m apps.compare results/<run_a> results/<run_b> --csv --plot
```

Use comparison output to decide whether a model is actually better than simple baselines after fees.

## Build The Docs Locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open the local URL printed by MkDocs.
