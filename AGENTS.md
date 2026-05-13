# AGENTS.md

> Instructions for AI coding agents (Codex CLI, Codex Cloud, Cursor,
> and any tool that reads the AGENTS.md convention) working in this
> repository. This file is loaded automatically — do not require the
> user to repeat its contents in every prompt.

---

## Project overview

We are building a Q-learning day-trading bot for the Hong Kong ETF
`2800.HK`, deployed via the Quantphemes broker API. The same codebase
supports a **lab workflow** for backtesting alternative hypotheses
(different assets including leveraged ETFs, different intervals,
different state encodings, different agents). Production trades
2800.HK; lab tests anything.

**The canonical specification, architecture, and phase plan live in
`quantphemes_development_plan_v2.md` at the repo root.** Read it
before writing any code. If a task contradicts the plan, stop and ask
for clarification.

---

## Setup commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest -q

# Lint and format
ruff check .
ruff format .

# Run a baseline experiment end-to-end (sanity check)
python -m apps.run_experiment experiments/_baseline.yaml

# Dry-run the live bot
python -m apps.bot --config experiments/production_2800.yaml \
    --strategy-id test --portfolio-id test --dry-run \
    --simulate-now 2026-05-13T09:25:00+08:00
```

---

## Locked specification values

These are non-negotiable unless the user explicitly says otherwise.
They are also reproduced in `experiments/_baseline.yaml`, which is
the runtime source of truth.

| Item | Value |
|---|---|
| Asset (production) | `2800.HK` |
| Asset (federated partner) | `2828.HK` |
| Starting capital | HKD 1,000,000 |
| Fee | 20 bps one-side = **40 bps round-trip**, applied **only when position changes** |
| Day-trading | flat by EOD, fresh state each morning |
| Decision interval (current) | **1 hour** (latest decision; was 30-min, was 5-min) |
| Decision points | 10:30, 11:30, 13:30, 14:30, 15:30 HK time → 5 per day |
| Action space | binary `{0, 1}` (cash / fully invested) |
| State features (baseline) | 3 features × 3 bins = 27 states |
| Q-tables | one per decision point: 5 × 27 × 2 = 270 cells |
| Reward | `r_t = log(V_{t+1} / V_t)` with fee inside `V_{t+1}` |

Lab experiments override the asset, interval, and state encoding via
YAML. They never override the fee model.

---

## Repository structure (src-layout)

```
src/quantphemes_rl/   # importable library
apps/                 # entry points (run_experiment, bot, federate)
experiments/          # YAML configs, one per hypothesis
tests/                # pytest, mirrors src structure
results/              # gitignored, auto-populated by run_experiment
data/raw/             # gitignored, contains input CSVs / xlsx
```

Imports are always absolute: `from quantphemes_rl.state import ...`,
never relative. The `src/` layout is enforced by `pyproject.toml`
(`[tool.setuptools.packages.find] where = ["src"]`).

---

## Plugin registry pattern

Four extension points have abstract base classes plus a registry:

| Kind | Base class | Module | Examples |
|---|---|---|---|
| `data_source` | `BarDataSource` | `data/base.py` | csv, futu, bloomberg, quantphemes_api |
| `state_encoder` | `StateEncoder` | `state/base.py` | three_feature, zscore_bins, with_rsi |
| `reward_function` | `RewardFunction` | `reward/base.py` | log_return, sharpe_adjusted, fee_aware |
| `agent` | `Agent` | `agent/base.py` | qtable, random_binary, always_long, buy_and_hold, dqn |

Every plugin is registered via:

```python
from quantphemes_rl.registry import register
from quantphemes_rl.state.base import StateEncoder

@register("state_encoder", "three_feature")
class ThreeFeatureEncoder(StateEncoder):
    ...
```

Adding a new plugin is **one file + one decorator + one test + one
line in `__init__.py` so the decorator fires on import.** Never an
`if/elif` chain in the training loop.

---

## Code style

- Python **3.11+**.
- **Type hints required on every function signature.** Use
  `from __future__ import annotations` at the top of every module.
- Use `dataclasses.dataclass` or `pydantic.BaseModel` for structured
  data; never bare dicts for typed records.
- `ruff` is the linter and formatter (config in `pyproject.toml`).
- Imports sorted by `ruff` (stdlib → third-party → local).
- No `print()` for logging. Use `log = logging.getLogger(__name__)`.
- Docstrings: one-line for trivial functions; full Google-style
  (Args, Returns, Raises) for public APIs.
- Keep modules small. If a file exceeds ~200 lines without good
  reason, split it.

---

## Testing instructions

- `pytest` with fixtures under `tests/fixtures/`.
- Run the full suite before opening a PR: `pytest -q && ruff check .`
- Every new plugin gets at least one unit test that:
  1. Instantiates it via the registry (`build("kind", "name", ...)`).
  2. Exercises its public interface on a synthetic input.
  3. Asserts a deterministic property (numerical equality, shape, etc.).
- Every bug fix gets a regression test in the same change.
- Tests must be hermetic: no network calls, no real broker API hits.
  Mock the `QuantphemesClient` in any test that involves API logic.
- Synthetic data fixtures go in `tests/fixtures/` with descriptive
  filenames (e.g. `synthetic_5days_1h_uptrend.csv`).

---

## Pull request conventions

PRs should be **one phase or one focused change**, not bundled work.
Each PR includes:

1. A title prefixed with the phase or area: `phase-3:`, `bot:`,
   `diagnostics:`, `fix:`.
2. A description with:
   - **What** was changed (1–3 bullets).
   - **Why** (link to plan section if applicable).
   - **Verification**: paste the output of the verification commands
     listed in the corresponding task / phase prompt.
3. All commits compile and pass tests; no broken intermediate states.
4. New plugins include the import line in
   `src/quantphemes_rl/__init__.py`.
5. New experiments include a YAML in `experiments/` and a one-line
   description of the hypothesis.

If a change touches `experiments/_baseline.yaml`, the fee model, or
the documented Quantphemes endpoints, **flag this prominently in the
PR description** — these affect the live bot.

---

## Hard rules (do NOT violate)

1. **Never break the tie rule.** When `Q[a=0] == Q[a=1]`, tie-break
   **randomly**. Never default to cash. The previous version had a
   bug that defaulted unvisited cells to cash, which silently destroyed
   evaluation. Use `random.choice([0, 1])` for ties.
2. **Never apply the fee on hold actions.** Fee is charged only when
   the position changes (0→1 or 1→0). `Portfolio.execute()` returns
   0.0 fee when `target == self.position`.
3. **Never add state features beyond the configured encoder.** New
   features live in new `StateEncoder` plugins, gated by config. Do
   not modify `ThreeFeatureEncoder` to add features.
4. **Never call undocumented Quantphemes endpoints from
   `api_client/quantphemes.py`.** They go in
   `api_client/_undocumented.py` with a file-level docstring noting
   they are not in the public docs and may break.
5. **Never measure `V_t` after a trade.** The reward is
   `log(V_{t+1} / V_t)` where `V_t` is the portfolio value **before**
   executing the action; the fee is captured inside `V_{t+1}`.
   Getting this backwards inverts the sign of the fee penalty.
6. **Never write production trading logic outside `apps/bot.py`.**
   Training code, evaluation code, and the live bot all use the same
   `TradingEnv` + `Agent.act()`. Divergent behaviour goes through
   config, not a parallel code path.
7. **Never commit secrets.** No API keys in code. Use environment
   variables (`QUANTPHEMES_API_KEY`) loaded via `os.environ`. Do
   not log API keys, not even partial.
8. **Never weaken a test to make it pass.** A failing test usually
   means the code is wrong. If you genuinely believe the test
   assertion is wrong, stop and ask.

---

## Soft rules (always do)

1. Always re-read `quantphemes_development_plan_v2.md` at the start
   of a new task.
2. Always add new plugin imports to `src/quantphemes_rl/__init__.py`
   so decorators fire on `import quantphemes_rl`.
3. Always include a regression test for any behaviour you fix.
4. Always write the pydantic schema before writing code that consumes
   the config.
5. Always log structured info (one JSON line per decision) in the live
   bot, never free-form strings.
6. Always preserve seeds: `random.seed(seed)`, `numpy.random.seed(seed)`,
   and any framework-specific seeding, at the start of every
   experiment.
7. Always include a `git_commit.txt` and `config.snapshot.yaml` in
   every results folder for reproducibility.

---

## Security and sandboxing notes (for Codex CLI)

- The repository contains no secrets. Do not request elevated network
  permissions unless a task explicitly involves calling the live
  Quantphemes API.
- Do not write outside the repo root.
- Do not execute the live bot (`apps/bot.py`) without `--dry-run`
  unless the user explicitly requests a real trading session.
- Tests must not require network access. If a test would, mock the
  network calls or skip with a clear reason.

---

## When in doubt

If a task is ambiguous or contradicts these instructions, **stop and
ask the user** rather than guess. Specifically ask if:

- A task asks for a state space larger than 27 (the baseline). Confirm
  it's a new plugin, not a modification of `three_feature`.
- A task asks to modify `_baseline.yaml`, the fee model, or the
  documented Quantphemes endpoints.
- The plan and the task disagree on a numerical value.
- A test is failing and the fix would require changing the test
  assertion (this usually means the code is wrong, not the test).
- A task implies real trading (no `--dry-run`, real API key, etc.)
  before Phase 7 is complete.

---

## Glossary

| Term | Meaning |
|---|---|
| Decision point | An intraday time at which the agent picks an action. With 1-hour interval, there are 5/day. |
| Q-table | The per-decision-point lookup table for state→action values. Shape: `(T, num_states, num_actions)`. |
| Tier 1 / Tier 2 / Tier 3 | Hypothesis-backlog tiers from plan §5. Tier 1 = bounds (random, always-long, buy-and-hold, oracle). Tier 2 = asset/interval grid. Tier 3 = signal engineering. |
| Oracle | Ex-post optimal binary policy with perfect foresight; the upper bound for any binary-action agent. |
| Federated merge | Combining our Q-table with Quantphemes' team's Q-table via visit-weighted cell-wise averaging. |
