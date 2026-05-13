# Experiments

Each YAML file describes one lab or production experiment. `_baseline.yaml` is the canonical
configuration and holds the locked production spec values; other experiments may inherit from it
and override only the hypothesis-specific fields.

Expected top-level sections:

- `name`: Stable experiment identifier, used in result folder names.
- `description`: Short hypothesis or purpose.
- `seed`: Reproducibility seed.
- `data`: Source adapter, symbol, paths, date range, and timezone.
- `market`: Capital, fee model, interval, decision points, and day-trading rules.
- `state`: State encoder name and encoder parameters.
- `reward`: Reward function name and reward-specific parameters.
- `agent`: Agent plugin name and agent-specific parameters.
- `training`: Training windows, episode settings, and output location.
- `evaluation`: Evaluation data, metrics, and baseline comparisons.
