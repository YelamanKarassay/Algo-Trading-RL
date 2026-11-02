# Technical Architecture Review

This document explains the current Quantphemes RL codebase from a developer-maintainer perspective. It is intentionally candid: the project is research-oriented, so failed strategies, imperfect models, and rough operational edges are normal. The goal is to make the system understandable, auditable, and easier to improve.

## 1. Project Snapshot

| Item | Summary |
|---|---|
| Project type | Research-grade reinforcement-learning trading bot, experiment pipeline, CLI tool suite, and paper-trading deployment system |
| Main language | Python 3.11+ |
| Business purpose | Train and evaluate Q-learning day-trading strategies for Hong Kong ETFs, then deploy selected Q-tables to Quantphemes paper trading |
| Main runtime modes | Local research experiments, batch research sweeps, HTML/PDF reports, Azure `systemd` live paper bots |
| Production-like external service | Quantphemes broker API |
| Main data sources | Webull / Futu CSV workflows; Bloomberg XLSX adapter is supported but not the current main data path |

In one sentence: this repository is a reproducible lab-to-paper-trading system where YAML experiment configs produce trained Q-table artifacts, and the live bot loads those artifacts to make scheduled broker target-position updates.

## 2. Architecture In 5 Bullets

- The repo is a modular Python monolith: one importable package under `src/quantphemes_rl/`, with thin CLI entry points in `apps/`.
- The core abstraction is a plugin registry: data sources, state encoders, reward functions, and agents are selected by name from YAML.
- The offline training path and live bot share most concepts, but not all execution details. This is the biggest area to watch for backtest/live drift.
- Experiments are file-driven: `experiments/*.yaml` defines the asset, data file, market assumptions, encoder, reward, agent, training loop, and artifact paths.
- Results are artifact-driven: each run writes `config.snapshot.yaml`, `git_commit.txt`, CSV logs, `metrics.json`, `q_state.pkl`, and HTML reports under `results/`.

## 3. Major Frameworks, Libraries, And Services

### Python Packaging: `setuptools`, `pyproject.toml`

What it does:

- Defines the package metadata, dependencies, and editable installation behavior.
- Uses a `src/` layout so imports work like an installed package, not as accidental local path imports.

Where used:

- `pyproject.toml`
- `src/quantphemes_rl/`

Fit:

- Good fit. This project is not a web server or large distributed system; a standard Python package with CLI modules is enough.
- The `src/` layout is the right choice because `apps/`, `tests/`, and notebooks import `quantphemes_rl` consistently.

### Pydantic v2

What it does:

- Validates structured config objects.
- Converts YAML dates and nested dictionaries into typed Python models.

Where used:

- `src/quantphemes_rl/config.py`
- Main models: `ExperimentConfig`, `DataConfig`, `MarketConfig`, `StateConfig`, `RewardConfig`, `AgentConfig`, `TrainingConfig`, `EvaluationConfig`, `ArtifactsConfig`.

Fit:

- Very good fit. This project relies heavily on YAML configs, and invalid configs could create wrong experiments or live behavior.
- `extra="forbid"` on `ExperimentConfig` is useful because typoed config keys fail early.

Trade-off:

- Some nested fields are still loose dictionaries, especially `StateConfig.kwargs`, `RewardConfig.kwargs`, `AgentConfig.kwargs`, and `TrainingConfig.walk_forward`. This preserves plugin flexibility but weakens validation.

### PyYAML

What it does:

- Reads and writes experiment YAML files.

Where used:

- `src/quantphemes_rl/config.py`
- `apps/research_sweep.py`
- `apps/compare.py`
- `apps/research_report.py`

Fit:

- Good fit. YAML is human-readable and reviewable, which matters for research configurations.

Trade-off:

- YAML inheritance is custom. It is lightweight and understandable, but there is no schema-aware merge or config provenance beyond the final snapshot.

### NumPy

What it does:

- Stores Q-tables and visit-count tensors.
- Provides vectorized statistics and array operations.

Where used:

- `src/quantphemes_rl/agent/qtable.py`
- `src/quantphemes_rl/diagnostics/coverage.py`
- Tests and diagnostics.

Fit:

- Excellent fit. Q-tables are naturally dense arrays with shape `(decision_points, states, actions)`.

### Pandas

What it does:

- Reads CSV/XLSX data.
- Normalizes vendor data into common bar data.
- Writes and analyzes experiment logs.

Where used:

- `src/quantphemes_rl/data/csv_source.py`
- `src/quantphemes_rl/data/futu_source.py`
- `src/quantphemes_rl/data/bloomberg_source.py`
- `apps/download_webull_data.py`
- `apps/compare.py`
- `apps/research_report.py`
- `src/quantphemes_rl/diagnostics/report.py`
- `src/quantphemes_rl/diagnostics/signal.py`

Fit:

- Good fit for research. It makes CSV/XLSX workflows fast to build and easy to inspect.

Trade-off:

- Large files are loaded fully into memory. This is fine for current ETF bars but may become a bottleneck for many symbols at 1-minute resolution.

### Requests

What it does:

- Makes HTTP calls to the Quantphemes API.

Where used:

- `src/quantphemes_rl/api_client/quantphemes.py`

Fit:

- Good fit for a simple synchronous bot.

Trade-off:

- The live bot is single-threaded and blocking. A slow API call can delay one bot. For one to several paper bots this is acceptable; for a larger fleet, a more robust scheduler and async/concurrent execution may be needed.

### Matplotlib

What it does:

- Generates static PNG plots for per-experiment HTML diagnostics.

Where used:

- `src/quantphemes_rl/diagnostics/report.py`
- `src/quantphemes_rl/diagnostics/coverage.py`
- `apps/compare.py`

Fit:

- Good for deterministic static reports.

Trade-off:

- Static plots are less pleasant for large research comparisons. The repo also uses Plotly for interactive batch research reports, which is a better fit there.

### Plotly

What it does:

- Creates interactive HTML charts for research sweeps.

Where used:

- `apps/research_report.py`

Fit:

- Good fit for comparing many experiments across assets, intervals, rewards, encoders, and action spaces.

### Jinja2

What it does:

- Renders HTML reports from templates.

Where used:

- `src/quantphemes_rl/diagnostics/report.py`
- `src/quantphemes_rl/diagnostics/templates/report.html.j2`

Fit:

- Good fit. Report generation is template-driven rather than string-concatenated.

### Scikit-learn

What it does:

- Computes mutual information for feature-signal diagnostics.

Where used:

- `src/quantphemes_rl/diagnostics/signal.py`

Fit:

- Reasonable fit. Mutual information is a useful early diagnostic for whether state features contain any relationship to next-interval returns.

Trade-off:

- Feature diagnostics are still basic. They help identify whether features are useless, but they do not prove tradability after fees.

### Pytest

What it does:

- Runs unit and smoke tests.

Where used:

- `tests/`
- `pyproject.toml`

Fit:

- Excellent fit. The project has many small deterministic components that are easy to test.

### Ruff

What it does:

- Lints and formats Python code.

Where used:

- `pyproject.toml`
- CI workflow.

Fit:

- Good fit. Fast feedback matters in a research repo where many small files are generated and edited.

### MkDocs Material

What it does:

- Builds a searchable documentation portal.

Where used:

- `mkdocs.yml`
- `docs/`
- `requirements-docs.txt`

Fit:

- Very good fit. It gives the project a readable knowledge base without building a custom frontend.

### Webull SDK

What it does:

- Fetches historical market data from Webull.

Where used:

- `apps/download_webull_data.py`
- `notebooks/webull_*.ipynb`

Fit:

- Useful but imperfect. It is practical for gathering HK ETF bars, but it is an optional operational dependency and is not currently listed in `pyproject.toml`.

### Quantphemes API

What it does:

- Provides portfolio, strategy, holdings, cash/asset, price, and order-related endpoints.

Where used:

- `src/quantphemes_rl/api_client/quantphemes.py`
- `src/quantphemes_rl/api_client/_undocumented.py`
- `src/quantphemes_rl/api_client/sync.py`
- `apps/bot.py`

Fit:

- Central to live paper trading.

Important design choice:

- Documented endpoints live in `quantphemes.py`.
- Undocumented endpoints live in `_undocumented.py`.
- This separation is good because undocumented endpoints may change and should not be treated as stable contract.

## 4. Directory Structure

```text
src/quantphemes_rl/   Importable library: data, state, reward, agent, env, portfolio, diagnostics, API client
apps/                 CLI entry points: train, compare, download data, live bot, federate, research report
experiments/          YAML experiment and production configs
experiments/research/ Generated research-sweep configs
tests/                Pytest suite
docs/                 MkDocs documentation
data/raw/             Local market data; gitignored
results/              Experiment outputs; gitignored
artifacts/            Live bot artifacts, logs, Q-tables; gitignored
notebooks/            Exploratory notebooks
```

Rationale:

- `src/` prevents accidental imports from the working directory.
- `apps/` keeps executable workflows thin and separate from the reusable library.
- `experiments/` makes hypotheses reviewable in Git.
- `results/`, `data/raw/`, and `artifacts/` are separated because they have different lifecycles: input data, experiment outputs, and live runtime state.

## 5. Core Architecture

The codebase is a layered modular monolith.

```text
YAML config
  -> Pydantic config loader
  -> plugin registry builds concrete classes
  -> data source loads Bar objects
  -> bars group into DayData
  -> TradingEnv simulates one day
  -> Portfolio enforces cash/shares/fees/lots
  -> StateEncoder maps MarketContext to state ID
  -> Agent picks action and updates Q-table
  -> RewardFunction computes reward
  -> diagnostics write results and reports
```

Main layers:

- Configuration layer: `src/quantphemes_rl/config.py`
- Plugin layer: `src/quantphemes_rl/registry.py`
- Domain data layer: `src/quantphemes_rl/data/base.py`
- Market simulation layer: `src/quantphemes_rl/env/trading_env.py`
- Accounting layer: `src/quantphemes_rl/portfolio/portfolio.py`
- Model layer: `src/quantphemes_rl/agent/`
- Feature layer: `src/quantphemes_rl/state/`
- Reward layer: `src/quantphemes_rl/reward/`
- Diagnostics layer: `src/quantphemes_rl/diagnostics/`
- Application layer: `apps/`
- External integration layer: `src/quantphemes_rl/api_client/`

## 6. Plugin Registry Pattern

The registry in `src/quantphemes_rl/registry.py` stores classes by kind and name:

- `data_source`
- `state_encoder`
- `reward_function`
- `agent`

Example:

- `CSVDataSource` registers as `data_source: csv`.
- `ThreeFeatureEncoder` registers as `state_encoder: three_feature`.
- `LogReturnReward` registers as `reward_function: log_return`.
- `QTableAgent` registers as `agent: qtable`.

Why this design works:

- New experiments can choose components from YAML.
- Training and live code avoid long `if/elif` chains.
- Each plugin is independently testable.
- It keeps the repo lightweight; no Hydra, no setuptools entry points, no heavy framework.

Main risk:

- Decorators fire only when modules are imported. The package `__init__.py` must import plugin modules, or registry lookup will fail.

## 7. Key Domain Concepts

### Bar

Defined in `src/quantphemes_rl/data/base.py`.

A `Bar` is one OHLCV candle:

- timestamp
- open
- high
- low
- close
- volume

The timestamp must be timezone-aware and normalized to Hong Kong time.

### DayData

Also in `data/base.py`.

A `DayData` object groups all bars for one trading day and exposes:

- `open_price`
- `close_price`
- `prices_at_decision_times(...)`
- `previous_close`

`previous_close` matters because one state feature compares the current price to yesterday's close.

### MarketContext

Defined in `src/quantphemes_rl/state/base.py`.

This is the object state encoders consume. It includes:

- current price
- today's open
- yesterday's close
- previous decision price
- decision index
- decision count
- recent returns
- volume context

### Portfolio

Defined in `src/quantphemes_rl/portfolio/portfolio.py`.

This handles cash, shares, fee, lot size, and discrete position changes. It is a critical module because small accounting mistakes can dominate trading results.

Important invariant:

- Fees are charged only when position changes.
- Holding the same position costs `0`.
- Buying respects available cash and lot size.
- Selling removes shares and charges one-side fee.

### TradingEnv

Defined in `src/quantphemes_rl/env/trading_env.py`.

This simulates one trading day:

1. Start at first decision time.
2. Encode the current market context.
3. Agent chooses action.
4. Portfolio executes position change.
5. Environment advances to the next decision time.
6. Reward is computed from portfolio value before and after the step.
7. At the terminal step, force close happens and the closing fee is included in terminal reward.

Important invariant:

- `V_t` is measured before the trade.
- Fee lives in `V_{t+1}`.

This is a very important financial modeling point. If fee is measured on the wrong side of the transition, the agent learns from distorted reward.

## 8. Critical Flow 1: Config Loading

Entrypoint:

- `load_config(path)` in `src/quantphemes_rl/config.py`

Step-by-step:

1. Read YAML using `yaml.safe_load`.
2. If the YAML has `inherits`, recursively load the parent config.
3. Deep-merge parent and child, with child values winning.
4. Validate the final dictionary using `ExperimentConfig`.
5. Return a typed config object.

Why this matters:

- Most experiments inherit from `_baseline.yaml`.
- Small YAML files can override only name, data path, reward, encoder, or action count.
- Every result folder writes `config.snapshot.yaml`, so the exact resolved config is preserved.

Trade-off:

- Nested plugin kwargs are flexible but loosely typed.
- A plugin can fail later at runtime if its kwargs are wrong.

Improvement:

- Add optional plugin-specific config validators for known plugins.

## 9. Critical Flow 2: Training An Experiment

Entrypoint:

- `apps/run_experiment.py`
- Main function: `run(path)`

Step-by-step:

1. Load YAML config via `load_config`.
2. Set Python and NumPy random seeds.
3. Create a result folder under `results/`.
4. Write `config.snapshot.yaml`.
5. Write `git_commit.txt`.
6. Build plugins:
   - data source
   - state encoder
   - reward function
   - agent
7. Load bars from the configured data source.
8. Group bars into `DayData`.
9. Reject days missing required decision-time bars or force-close bar.
10. Build walk-forward train/test windows.
11. Train by looping:
    - windows
    - episodes
    - shuffled days
    - intraday environment steps
12. Evaluate main agent greedily.
13. Evaluate configured baselines.
14. Write `train_log.csv`, `eval_log.csv`, `step_log.csv`.
15. Compute `metrics.json`.
16. Save `q_state.pkl`.
17. Generate `report.html`.
18. Append summary row to `results/_index.csv`.

Important code:

- `apps/run_experiment.py::run`
- `apps/run_experiment.py::_train`
- `apps/run_experiment.py::_evaluate`
- `apps/run_experiment.py::_run_day`
- `apps/run_experiment.py::_metrics`

Important assumptions:

- Each simulated day starts with fresh capital.
- The model is day-trading, so positions are closed by end-of-day.
- The configured decision times must exist in the data file.

Critique:

- `data_source.load(..., "1h")` currently passes `"1h"` regardless of config interval. CSV/Futu/Bloomberg adapters ignore this, so current research works because the file itself already has the desired bars. But if a future API data source uses `interval`, this becomes a correctness issue.

## 10. Critical Flow 3: Q-table Learning

Main file:

- `src/quantphemes_rl/agent/qtable.py`

The Q-table shape is:

```text
(num_decision_points, num_states, num_actions)
```

Example:

- 30-minute binary model: `9 × 27 × 2`
- 1-hour binary model: often `5 × 27 × 2`
- Ternary action model: actions are cash, half position, full position, so `num_actions = 3`.

Action selection:

- During training, the agent uses epsilon-greedy exploration.
- With probability `epsilon`, it picks a random action.
- Otherwise, it picks the action with the highest Q-value.
- If Q-values tie, default tie-break is random.

Why random tie-break matters:

- Unvisited states often start with equal Q-values.
- If ties always default to cash, the agent becomes silently biased toward doing nothing.
- This was explicitly protected by tests.

Q-learning update:

```text
Q(t, state, action) <- Q + alpha * (reward + max(Q(next_t, next_state, *)) - Q)
```

Plain English:

- The agent updates the value of the action it just took.
- The target is the immediate reward plus the best estimated value of the next state.
- `alpha` controls how much the new experience changes the table.

The implementation supports:

- `alpha_mode="per_cell"`: learning rate is `1 / visit_count` for that cell.
- `alpha_mode="one_over_d"`: learning rate depends on global day counter.

Persistence:

- `save(path)` writes a version byte plus pickle payload.
- `load(path)` checks the version byte and restores tables, visits, epsilon, and config.

Security note:

- Pickle is unsafe for untrusted files. Only load `q_state.pkl` artifacts created by this project or trusted collaborators.

## 11. Critical Flow 4: Data Download And Normalization

Main entrypoint:

- `apps/download_webull_data.py`

Step-by-step:

1. Load `.env`.
2. Build Webull SDK client from `WEBULL_APP_KEY` and `WEBULL_APP_SECRET`.
3. Build one or more `DownloadSpec` records.
4. Fetch historical bars by paginating backward from the end date.
5. Normalize timestamps to Hong Kong time.
6. Shift bar timestamps by timespan so bars align to their closing time.
7. Save CSV under `data/raw/`.

Main internal format:

- CSV columns: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

Why this matters:

- The training system expects decision bars like `10:30`, `11:00`, `15:30`, and `16:00`.
- If bars are timestamped at the start of interval instead of the close, the model can learn from the wrong time.

Critique:

- Webull SDK is operationally required for download scripts but is not declared as a project dependency. That is acceptable if treated as optional, but it should be documented more explicitly.
- The script uses simple backward pagination and in-memory pandas concatenation. Fine for current assets, but not ideal for large-scale market data.

## 12. Critical Flow 5: Live Paper Bot

Main entrypoint:

- `apps/bot.py`

Step-by-step:

1. Load `.env`.
2. Parse CLI args.
3. Load production YAML config.
4. Build state encoder and agent from registry.
5. Load `q_state.pkl` if configured artifact exists.
6. Build Quantphemes client if API key exists.
7. Load runtime state JSON.
8. Wait for market open when running in real time.
9. Fetch or simulate today's open.
10. For each decision time:
    - fetch current price
    - build `MarketContext`
    - encode state
    - call `agent.act_at(...)` if available
    - compute target quantity
    - patch target holding unless dry-run
    - write one JSON log line
11. At force-close time:
    - patch target quantity to zero
    - confirm current quantity if possible
    - write close log line
12. Save runtime state.

Important files:

- `apps/bot.py`
- `src/quantphemes_rl/api_client/quantphemes.py`
- `src/quantphemes_rl/api_client/_undocumented.py`
- `src/quantphemes_rl/api_client/sync.py`

Important runtime files:

- `artifacts/runtime_state*.json`
- `artifacts/logs/*.jsonl`
- `artifacts/deploy/*.pkl` or configured Q-table paths

Critique:

- `portfolio_id` is accepted by CLI but currently deleted in `run_bot`. This may be okay if all operations are strategy-scoped, but it is confusing and should be either used or removed from the required runtime contract.
- `apps/bot.py::handle_decision` currently appears to pass `price` as `prev_decision_price`. That makes the `delta_prev` feature neutral in live trading, while backtesting uses the actual previous decision price. This is likely a real backtest/live drift bug and should be high priority to fix.
- The live bot is synchronous and blocking. A slow Quantphemes request can delay execution.
- There is no robust holiday calendar. It skips weekends, but HK public holidays are not modeled.
- The bot has operational retries through the API client, but decision-level recovery is still basic.

## 13. Diagnostics And Reporting

Main files:

- `src/quantphemes_rl/diagnostics/coverage.py`
- `src/quantphemes_rl/diagnostics/oracle.py`
- `src/quantphemes_rl/diagnostics/signal.py`
- `src/quantphemes_rl/diagnostics/report.py`
- `apps/research_report.py`

Diagnostics include:

- State coverage: which Q-table cells were visited.
- Oracle NAV: ex-post best binary policy with perfect hindsight.
- Mutual information: rough signal strength of state features.
- Per-run HTML report.
- Batch Plotly research report.

Why diagnostics are critical:

- A bad trading model can fail because the market has no signal, because fees dominate, because features are weak, because the agent did not explore enough, or because live execution differs from backtest.
- Diagnostics separate these failure modes.

Critique:

- The diagnostic reports explain experiments well, but they are not yet a full monitoring system.
- There is no automated alert when live logs show too many trades, missing prices, repeated API failures, or force-close pending.

## 14. Research Sweep System

Main files:

- `apps/research_sweep.py`
- `apps/research_report.py`
- `experiments/research/*.yaml`

The sweep varies:

- assets: `2800.HK`, `2828.HK`, `7299.HK`, `7568.HK`, `7226.HK`
- intervals: `1h`, `30m`, `15m`, `5m`, `1m`
- rewards: `log_return`, `log_return_long_bias`, `fee_aware`, `sparse_liquidation`, `drawdown_penalty`
- encoders: `three_feature`, `with_rsi`, `with_volatility`, `with_volume`, `zscore_bins`
- action spaces: binary and ternary

This is good research architecture because it encourages controlled comparison rather than one-off notebook experiments.

Trade-off:

- The sweep is combinatorial. Full grids can become expensive, noisy, and hard to interpret.
- The project needs clear promotion criteria for selecting live candidates.

## 15. Federation

Main file:

- `apps/federate.py`

Purpose:

- Merge two compatible Q-table agents using visit-weighted averaging.

Formula:

```text
merged_q = (n_ours * q_ours + n_theirs * q_theirs) / (n_ours + n_theirs)
merged_visits = n_ours + n_theirs
```

Why this makes sense:

- A Q-value backed by many visits should influence the merged table more than a Q-value backed by few visits.

Important assumption:

- Both teams must use identical state encoding, reward, fee model, decision times, and action semantics.

Critique:

- The merge is mathematically simple and auditable, but it is not the same as retraining on combined data.
- If reward definitions or state bins differ even slightly, the merge becomes invalid.

## 16. Testing Setup

Current test suite:

- Registry tests
- Data adapter tests
- Portfolio accounting tests
- Trading environment tests
- State encoder tests
- Reward tests
- Q-table tests
- Baseline agent tests
- Config tests
- Experiment runner smoke tests
- Diagnostics tests
- Live bot logic tests with mocked clients
- Runtime state tests
- Federated merge tests
- Webull download helper tests
- Research sweep tests

This is strong for a research project.

Most important coverage:

- Fee-on-change behavior.
- `V_t` measured before trade.
- Force-close fee included in terminal reward.
- Random tie-break behavior.
- Q-table save/load.
- Live bot dry-run and force-close logic.

Gaps:

- No true end-to-end mocked live trading day with multiple services.
- No test that compares live `MarketContext` construction against backtest `TradingEnv` context for the same prices.
- No tests around Azure/systemd deployment units.
- No contract tests against Quantphemes API schemas.

## 17. Observability

Current observability:

- Python logging in library modules.
- Structured JSONL decision logs in live bot.
- CSV logs for training and evaluation.
- HTML reports for experiments.
- `systemd` journal logs on server.

Good parts:

- Live decisions are auditable through JSON lines.
- Training artifacts are reproducible through snapshots and git commit markers.

Weak parts:

- No centralized metrics.
- No alerting.
- No daily bot health report.
- No automated detection of “bot running but no decision logs today.”
- No reconciliation between desired target quantity and visible broker position/order state beyond limited confirmation.

## 18. Security Review

Good practices:

- API keys are read from `.env` or environment variables.
- Secrets are not supposed to be committed.
- `QuantphemesClient.__repr__` does not expose API key.
- Live writes require `--dry-run` to be omitted.

Risks:

- Pickle loading is unsafe for untrusted files.
- `.env` handling is custom and simple; malformed lines or accidental logging should be treated carefully.
- Live bot logs contain prices, quantities, states, and status. That is fine for debugging but should be considered operationally sensitive.
- The repo has many generated configs and artifacts paths; accidental commit of `artifacts/` or `data/raw/` would be bad.

Recommended security improvement:

- Add a small startup check that refuses to run live mode if `.env` is world-readable on the server.
- Add explicit docs saying never to load external Q-table pickles without conversion/review.

## 19. Performance Review

Current performance is adequate because:

- Q-tables are tiny.
- ETF data is moderate.
- Experiments are simple loops.
- Live bot acts only at scheduled decision points.

Likely bottlenecks:

- Full pandas CSV loading for large 1-minute data across many assets.
- Sequential research sweep execution.
- Per-run report generation if running many experiments.
- Blocking API calls in live bot.
- Large generated YAML set under `experiments/research/`.

Improvements:

- Cache parsed `DayData` per data file.
- Add multiprocessing or a job queue for research sweeps.
- Add a small SQLite/DuckDB result index for large research grids.
- Use async or a supervisor process if running many live bots.

## 20. Maintainability Review

Strengths:

- Clear src-layout.
- Thin app entry points.
- Good test coverage.
- Small plugin interfaces.
- YAML-driven experiments.
- Good separation of data, state, reward, agent, portfolio, and diagnostics.

Pain points:

- `apps/bot.py` is doing scheduling, state building, broker sync, logging, and runtime state persistence in one file.
- Plugin kwargs are not strongly typed.
- Research config generation creates a very large number of YAML files.
- Production service management lives mostly outside Python and docs, not as generated service definitions.
- Some code uses CLI `print()` despite the general no-print guideline. For CLI output this is acceptable, but it is inconsistent.

## 21. Main Design Decisions And Rationale

### Why Q-table before DQN?

Q-tables are small, inspectable, and easy to debug. With 27 to 81 states and 2 to 3 actions, tabular learning is enough for a first research system.

Trade-off:

- Q-tables cannot learn rich nonlinear market structure.
- State design becomes extremely important.
- Sparse state coverage is a real problem.

### Why YAML configs?

YAML lets experiments be reviewed as hypotheses:

- which asset
- which data
- which reward
- which encoder
- which action space
- which training parameters

Trade-off:

- YAML can drift or multiply quickly.
- Config inheritance is custom.

### Why plugin registry?

It keeps the core runner and live bot stable while allowing new research components.

Trade-off:

- Plugins must be imported before registration.
- Plugin kwargs are less type-safe.

### Why no database?

Current artifacts are file-based:

- CSV logs
- JSON metrics
- YAML snapshots
- pickle Q-tables
- HTML reports

This is simple and appropriate for early research.

Trade-off:

- Comparing hundreds or thousands of runs becomes clumsy.
- Querying historical results is harder than with SQLite/DuckDB.

### Why Azure + systemd?

It is simple, cheap, and reliable enough for scheduled paper bots.

Trade-off:

- `systemd` services are operationally robust but not self-documenting.
- Multi-bot fleet management can become tedious.

## 22. Financial Terms Used In The Code

### NAV

Net asset value. In this project it means:

```text
cash + shares * current_price
```

### Bps

Basis points. `20 bps` means `0.20%`.

The model assumes:

- 20 bps one-side fee
- around 40 bps round-trip fee

That means a buy and later sell must overcome roughly 0.40% cost before making net profit.

### Buy-and-hold

A baseline strategy that buys and keeps holding. It is used as a benchmark.

### Alpha vs buy-and-hold

How much better or worse the strategy did compared to buy-and-hold over the same evaluation window.

### Sharpe

A return-to-volatility ratio. Higher is better, but in this project Sharpe should be interpreted carefully because intraday/day-level sampling and paper-trading assumptions can distort it.

### Drawdown

Drop from a previous peak NAV. A drawdown penalty reward discourages unstable intraday paths.

## 23. Critique: Current Live Trading Design

This section is intentionally critical. The project is research-based, so discovering these issues is part of the process, not a failure.

### High-risk issue: live `delta_prev` drift

In backtesting, `TradingEnv._context` sets:

```text
prev_decision_price = price at previous decision time
```

In live code, `apps/bot.py::handle_decision` currently appears to set:

```text
prev_decision_price = current price
```

Impact:

- The `delta_prev` feature becomes close to zero in live trading.
- A model trained with real previous-decision movement may receive a different live state.
- This can cause action mismatch between backtest and live operation.

Recommendation:

- Store and update previous decision price in `RuntimeState`.
- Add a regression test comparing live context construction with `TradingEnv` for the same synthetic day.

### Medium-risk issue: production artifact/config mismatch

The live bot builds an agent from YAML and then loads `q_state.pkl`. If YAML says one action count or state count but the pickle has another, `load()` overwrites agent shape from the pickle.

Impact:

- Usually fine, but confusing.
- A wrong config may still appear to work while artifact content silently wins.

Recommendation:

- Add explicit compatibility checks after `agent.load`.

### Medium-risk issue: broker position reconciliation

The bot computes target quantity and patches holdings, but full broker reconciliation is limited.

Impact:

- If the broker accepts the PATCH but does not execute visibly, the bot may assume progress from API response.

Recommendation:

- Add post-patch reconciliation against holdings/orders.
- Log requested target, accepted payload, confirmed quantity, and dashboard-visible state if available.

### Medium-risk issue: no holiday calendar

Weekend skipping exists, but HK holidays are not modeled.

Impact:

- Bot may attempt to run on market holidays.

Recommendation:

- Add an HKEX calendar dependency or a simple maintained holiday file.

## 24. Prioritized Recommendations

### 1. Fix live previous-decision price state drift

What to change:

- Add `last_decision_price` to `RuntimeState`.
- In `handle_decision`, use previous stored decision price for `MarketContext.prev_decision_price`.
- After decision processing, update runtime with current price.
- Add regression tests.

Why it helps:

- Aligns live state encoding with backtest.
- Reduces the most dangerous kind of model failure: training/live feature mismatch.

Effort:

- S

### 2. Add artifact/config compatibility checks

What to change:

- After `agent.load(q_state_path)`, verify:
  - `num_decision_points`
  - `num_states`
  - `num_actions`
  - encoder name and expected state count
  - decision time count

Why it helps:

- Prevents deploying the wrong Q-table with the wrong production YAML.

Effort:

- S

### 3. Split `apps/bot.py` into smaller modules

What to change:

- Keep `apps/bot.py` as the CLI.
- Move runtime state to `src/quantphemes_rl/live/runtime.py`.
- Move target sizing to `src/quantphemes_rl/live/sizing.py`.
- Move scheduler helpers to `src/quantphemes_rl/live/scheduler.py`.
- Move broker sync logic to `src/quantphemes_rl/live/broker.py`.

Why it helps:

- Easier testing.
- Cleaner mental model.
- Less risk when changing live behavior.

Effort:

- M

### 4. Add a live/backtest parity test

What to change:

- Build a synthetic `DayData`.
- Build the equivalent live runtime state sequence.
- Assert both paths produce the same `MarketContext` and state IDs at every decision point.

Why it helps:

- Catches exactly the category of bug that matters most in trading: backtest says one thing, live code does another.

Effort:

- S

### 5. Improve Quantphemes API contract handling

What to change:

- Add schema-normalization functions for holdings, cash, orders, and price payloads.
- Add mocked contract tests using saved example payloads.
- Add better exception types for timeout, invalid symbol, no holding, and rejected patch.

Why it helps:

- Makes broker integration more robust and easier to debug.

Effort:

- M

### 6. Add operational health reporting

What to change:

- Daily script that reads JSONL logs and reports:
  - bot started
  - decisions made
  - price errors
  - target changes
  - force-close status
  - current quantity after close

Why it helps:

- Prevents silent failure on server.
- Makes paper-trading fleet easier to monitor.

Effort:

- M

### 7. Introduce a small results database

What to change:

- Keep current files, but also index results into SQLite or DuckDB.
- Store config summary, metrics, artifact path, and report path.

Why it helps:

- Research sweep comparisons become much easier.
- Enables richer dashboards and filtering.

Effort:

- M

### 8. Add optional typed plugin config models

What to change:

- Add validators for known plugins:
  - Q-table kwargs
  - three-feature thresholds
  - z-score settings
  - reward parameters

Why it helps:

- Catches bad YAML before experiments start.

Effort:

- M

### 9. Make Webull SDK dependency explicit

What to change:

- Add optional dependency group such as:

```text
pip install -e ".[data]"
```

or document separate installation clearly.

Why it helps:

- New developers can reproduce data download workflows.

Effort:

- S

### 10. Add promotion checklist tooling

What to change:

- A command like:

```text
python -m apps.promote results/<run> --bot RL_BROAD_A
```

It would copy `q_state.pkl`, update/check production config, run compatibility checks, and print restart commands.

Why it helps:

- Reduces manual deployment mistakes.

Effort:

- M

## 25. Suggested Refactor Roadmap

Short term:

1. Fix live previous-decision price parity.
2. Add artifact/config compatibility checks.
3. Add live/backtest parity tests.
4. Add daily bot health summary.

Medium term:

1. Split live bot internals into `quantphemes_rl.live`.
2. Add broker payload contract tests.
3. Add typed plugin config validators.
4. Improve result indexing with SQLite/DuckDB.

Longer term:

1. Add DQN only after the Q-table pipeline is stable and deeply understood.
2. Add richer execution simulation if Quantphemes exposes order/fill details.
3. Add portfolio-level multi-strategy risk controls.
4. Add centralized monitoring and alerts.

## 26. How To Explain The Project Simply

For a developer:

> The repo is a YAML-driven RL research and paper-trading platform. Historical ETF bars are normalized into day objects, a trading environment simulates intraday decisions with realistic fees and lot sizes, a Q-table agent learns state-action values, diagnostics explain what happened, and the live bot loads a chosen Q-table to send target holdings to Quantphemes.

For a finance audience:

> The system tests whether simple intraday signals can overcome trading costs in Hong Kong ETFs. It compares many controlled strategy variants, then paper-trades selected models while logging every decision for review.

For yourself as maintainer:

> Keep the backtest/live state definitions identical, keep fee accounting sacred, and never promote a model unless its config, artifact, data window, and live logs are all traceable.
