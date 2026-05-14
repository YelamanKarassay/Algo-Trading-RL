# Quantphemes RL

Q-learning research and live-trading workflow for Hong Kong ETFs through the Quantphemes broker API. Production targets `2800.HK`; the lab workflow can test other assets, intervals, state encoders, rewards, and agents through YAML configs.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

## Documentation

The full project knowledge base lives in `docs/` and is built with MkDocs Material.

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Start here:

- [Home](docs/index.md)
- [Getting Started](docs/getting-started/index.md)
- [Architecture](docs/architecture/index.md)
- [Live Trading Runbook](docs/operations/runbook.md)

## Common Commands

```bash
ruff check .
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
python -m apps.compare results/<run_a> results/<run_b> --csv --plot
python -m apps.bot --config experiments/production_2800.yaml --dry-run
```

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
