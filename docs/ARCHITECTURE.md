# Architecture

This project has two tracks that share the same core library:

- **Lab track:** run YAML-defined experiments against historical data, produce metrics and HTML diagnostics.
- **Production track:** load a trained Q-table and run the live bot against the Quantphemes broker API.

## Main Flow

```text
experiments/*.yaml
  -> quantphemes_rl.config.load_config
  -> plugin registry
      -> data_source
      -> state_encoder
      -> reward_function
      -> agent
  -> TradingEnv + Portfolio
  -> results/<run>/
      config.snapshot.yaml
      metrics.json
      train_log.csv
      eval_log.csv
      step_log.csv
      q_state.pkl
      report.html
```

## Key Modules

| Path | Purpose |
|---|---|
| `src/quantphemes_rl/data/` | Normalizes CSV, Futu, and Bloomberg data into `Bar` / `DayData`. |
| `src/quantphemes_rl/state/` | Converts market context into discrete state indices. |
| `src/quantphemes_rl/reward/` | Reward functions; baseline is log return. |
| `src/quantphemes_rl/agent/` | Q-table and baseline agents. |
| `src/quantphemes_rl/env/` | Single-day trading environment. |
| `src/quantphemes_rl/portfolio/` | Cash, shares, lot sizing, and fee accounting. |
| `src/quantphemes_rl/diagnostics/` | Coverage, oracle, signal diagnostics, and HTML report. |
| `apps/run_experiment.py` | YAML-driven training/evaluation runner. |
| `apps/bot.py` | Live/dry-run broker bot. |
| `apps/compare.py` | Compare result folders. |
| `apps/federate.py` | Merge Q-tables with Quantphemes team tables. |

## Safety Rules

- Fee is charged only when position changes.
- `V_t` is measured before the action; fee is captured in `V_{t+1}`.
- Q-table ties default to random, not cash.
- Live broker calls only happen when `--dry-run` is omitted.
- Secrets live in `.env`; `.env` is gitignored.
