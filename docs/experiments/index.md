# Experiment Workflow

Experiments are reproducible runs defined by YAML. The baseline config holds shared defaults, and individual experiments override only what they need.

## Config Inheritance

```yaml
inherits: _baseline.yaml
name: exp_002_2800_30min
description: Tier 2 grid cell for 2800.HK at 30-minute interval.
data:
  path: data/raw/webull_2800_hk_m30.csv
market:
  decision_times: ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00", "15:30"]
  force_close_time: "16:00"
```

The child config wins on conflicts. The resolved config is saved into every results folder as `config.snapshot.yaml`.

## Run Training

```bash
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
```

The training loop:

1. Loads bars and groups them into trading days.
2. Splits days into walk-forward windows.
3. Trains the configured agent over shuffled training days.
4. Evaluates greedily on the test window.
5. Runs baselines and diagnostics.
6. Writes metrics, logs, Q-table, and HTML report.

## Interpret Results

| Artifact | What To Check |
|---|---|
| `metrics.json` | Final NAV, return, trades, fees, alpha vs buy-and-hold |
| `train_log.csv` | Whether learning explores enough states |
| `eval_log.csv` | Day-by-day greedy policy behavior |
| `report.html` | NAV curve, action distribution, coverage, reward histogram |
| `q_state.pkl` | Model artifact used by the live bot |

!!! warning "Synthetic results are not deployment evidence"
    A run on fixture data proves the machinery works. Deployment needs real historical data, sane diagnostics, and a dry-run check.

## Promote A Model

Only promote a model after reading its report and comparing it against baselines.

```bash
mkdir -p artifacts
cp results/<chosen_run>/q_state.pkl artifacts/q_state.pkl
```

Then run:

```bash
python -m apps.bot --config experiments/production_2800.yaml --dry-run
```

For the multi-bot paper deployment, promote into the named deploy artifact instead:

```bash
mkdir -p artifacts/deploy
cp results/<chosen_run>/q_state.pkl artifacts/deploy/rl_broad_a_q_state.pkl
python -m apps.bot \
  --config experiments/production_rl_broad_a.yaml \
  --strategy-id "$STRATEGY_RL_BROAD_A_ID" \
  --portfolio-id "$PORTFOLIO_RL_BROAD_A_ID" \
  --dry-run
```

Each production config owns its artifact and runtime state path. This avoids one bot accidentally reusing another bot’s Q-table or daily open/close state.

## Compare Multiple Runs

```bash
python -m apps.compare results/<run_a> results/<run_b> --csv --plot
```

Prefer strategies that beat simple baselines after fees and do not rely on a tiny number of trades or sparse state visits.
