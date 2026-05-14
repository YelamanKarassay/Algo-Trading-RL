# Developer Guide

The project is plugin-driven. Add new behavior through registered classes, not through `if/elif` chains in the runner.

## Add A Plugin

Every plugin needs:

1. A concrete class implementing the relevant ABC.
2. A `@register(kind, name)` decorator.
3. An import in `src/quantphemes_rl/__init__.py` so the decorator fires.
4. At least one unit test using `build(kind, name, **kwargs)`.

## Plugin Kinds

| Kind | Base Class | Examples |
|---|---|---|
| `data_source` | `BarDataSource` | `csv`, `futu`, `bloomberg` |
| `state_encoder` | `StateEncoder` | `three_feature` |
| `reward_function` | `RewardFunction` | `log_return`, `log_return_long_bias` |
| `agent` | `Agent` | `qtable`, `random_binary`, `always_long` |

## Testing Rules

Run before pushing:

```bash
ruff check .
pytest -q
```

Tests should be deterministic and hermetic. Mock broker/API behavior instead of calling live services.

## CI/CD

There are two GitHub Actions workflows:

| Workflow | Purpose |
|---|---|
| `ci.yml` | Install package, run Ruff, run Pytest on Python 3.11 and 3.12 |
| `docs.yml` | Build MkDocs with strict links and deploy GitHub Pages |

Docs checks:

```bash
pip install -r requirements-docs.txt
mkdocs build --strict
```

## Code Style

- Use absolute imports from `quantphemes_rl`.
- Add `from __future__ import annotations` to Python modules.
- Type every function signature.
- Use logging, not `print()`, except entrypoint summaries where already established.
- Keep secrets out of code, docs, tests, and logs.
