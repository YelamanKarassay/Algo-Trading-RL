# Runbook

## Install And Verify

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

## Train An Experiment

```bash
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
```

The runner writes a folder under `results/` containing:

- `metrics.json`
- `q_state.pkl`
- `report.html`
- logs and config snapshot

## Compare Results

```bash
python -m apps.compare results/<run_a> results/<run_b> --csv --plot
```

## Promote A Model For Dry-Run

Only promote a model after reading its report and checking that it beats simple baselines net of fees.

```bash
mkdir -p artifacts
cp results/<chosen_run>/q_state.pkl artifacts/q_state.pkl
```

## Dry-Run Bot

```bash
python -m apps.bot --config experiments/production_2800.yaml --dry-run
```

## Live Bot

Do not run live until:

- `artifacts/q_state.pkl` exists and came from a real-data run.
- Broker strategy holding is initialized correctly.
- Price symbol and trading symbol are confirmed with Quantphemes.
- The bot has passed a dry-run on the same day.

Live command:

```bash
python -m apps.bot --config experiments/production_2800.yaml
```

## Current Known Live-API Follow-Ups

- First-time strategy bootstrap needs `POST /api/v1/strategy/{id}/holding`.
- PATCH payload must match Quantphemes holding schema.
- Live `today_open` should come from intraday history, not last price.
- End-of-day flattening should be explicit unless Quantphemes confirms broker-side flattening.
