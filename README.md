# Quantphemes RL

Q-learning research and live-trading workflow for Hong Kong ETFs through the Quantphemes broker API. Production targets `2800.HK`; the lab workflow can test other assets, intervals, state encoders, rewards, and agents through YAML configs.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

## Common Commands

```bash
ruff check .
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
python -m apps.compare results/<run_a> results/<run_b> --csv --plot
python -m apps.bot --config experiments/production_2800.yaml --dry-run
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data Guide](docs/DATA_GUIDE.md)
- [Runbook](docs/RUNBOOK.md)
- [Project HTML Report](docs/project_report.html)
- [Development Plan](quantphemes_development_plan_v2.md)

## Repository Layout

```text
src/quantphemes_rl/   importable library
apps/                 entry points
experiments/          YAML configs
tests/                pytest suite
data/raw/             local market data, gitignored
results/              experiment outputs, gitignored
artifacts/            live bot runtime artifacts, gitignored
```
