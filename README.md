# Quantphemes RL

We are building a Q-learning day-trading bot for the Hong Kong ETF `2800.HK`, deployed via the Quantphemes broker API. The same codebase supports a lab workflow for backtesting alternative hypotheses across assets, intervals, state encodings, and agents; production trades `2800.HK`, while the lab can test anything.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

The canonical specification, architecture, and phase plan live in [`quantphemes_development_plan_v2.md`](quantphemes_development_plan_v2.md).
