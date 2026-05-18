# Quantphemes RL Knowledge Base

Quantphemes RL is a Q-learning research and live-trading workflow for Hong Kong ETFs. The current production-style paper deployment runs five independent Q-table strategies through the Quantphemes broker API, while the lab path lets us test assets, intervals, state encoders, reward functions, and agents through YAML configuration.

This documentation is written for a mixed audience: someone should be able to understand the project goal in a few minutes, then keep reading into the engineering and operational details when needed.

## What The Project Solves

The project gives us one shared system for three jobs:

- **Research:** run reproducible experiments over historical ETF data.
- **Diagnostics:** explain whether a model failed because the policy is bad, the features are weak, or transaction costs dominate.
- **Production:** deploy the selected Q-table to a live paper-trading strategy with explicit safety checks.

## Two Tracks, One Core

```mermaid
flowchart LR
    YAML["Experiment YAML"] --> CFG["Config loader"]
    CFG --> REG["Plugin registry"]
    REG --> ENV["TradingEnv + Portfolio"]
    ENV --> RUN["Lab runner"]
    ENV --> BOT["Live bot"]
    RUN --> RESULTS["results/<run>/ report.html + q_state.pkl"]
    BOT --> QP["Quantphemes API"]
```

The same core abstractions drive both offline experiments and live execution. Lab experiments produce `q_state.pkl`; production loads that artifact and calls `Agent.act(..., training=False)`.

## Current Status

| Area | Status |
|---|---|
| Data adapters | CSV, Futu CSV, Bloomberg XLSX |
| Learning core | Q-table agent, baseline agents, plugin registry |
| Diagnostics | Coverage, oracle, signal checks, HTML report |
| Experiment runner | YAML-driven walk-forward training |
| Live bot | Azure/systemd paper deployment with five active services |
| Federation | Visit-weighted Q-table merge |

## Active Paper Deployment

As of the current stable deployment, Azure runs five paper-trading bots and the repo is prepared for two additional groupmate 1-hour bots. Each bot has its own Quantphemes portfolio/strategy IDs in `.env`, its own production YAML, its own Q-table artifact, and its own JSON log directory.

| Bot | Asset | Experiment Config | Status |
|---|---|---|---|
| `RL_CROSS` | `7226.HK` | `research_7226_30m_drawdown_penalty_three_feature_binary` | Active |
| `RL_VOL` | `7226.HK` | `research_7226_30m_drawdown_penalty_with_volatility_binary` | Active |
| `RL_FACOV` | `7226.HK` | `research_7226_30m_fee_aware_zscore_bins_ternary` | Active, watch fees |
| `RL_BROAD_A` | `2800.HK` | `research_2800_30m_fee_aware_zscore_bins_ternary` | Active |
| `RL_BROAD_B` | `2828.HK` | `research_2828_30m_log_return_with_volatility_ternary` | Active |
| `GROUPMATE_2800_1H` | `2800.HK` | `production_groupmate_2800_1h` | Ready once IDs are added |
| `GROUPMATE_7226_1H` | `7226.HK` | `production_groupmate_7226_1h` | Ready once IDs are added |

The older `RL_PRIME`, `RL_BIAS`, and `RL_FULLCOV` deployments were halted because Quantphemes rejected `7299.HK` as outside the tradable symbol list.

## Where To Go Next

<div class="grid cards" markdown>

-   **New user**

    Start with [Getting Started](getting-started/index.md), then read [Architecture](architecture/index.md).

-   **Running experiments**

    Use [Experiment Workflow](experiments/index.md) and [Data Guide](data/index.md).

-   **Operating the bot**

    Follow the [Live Trading Runbook](operations/runbook.md) before touching live mode.

-   **Extending the system**

    Read the [Developer Guide](developer/index.md) for plugin and testing rules.

</div>
