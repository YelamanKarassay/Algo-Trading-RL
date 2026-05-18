# Research Notes

The current model family is intentionally simple: tabular Q-learning agents with binary or ternary actions.

## Current Q-Table Model

| Item | Current Shape |
|---|---|
| Asset | Active paper set uses `7226.HK`, `2800.HK`, and `2828.HK` |
| Actions | Binary: `0` cash, `1` full. Ternary: `0` cash, `1` half, `2` full |
| State | Three-feature, z-score, and volatility-regime encoders |
| Tables | One table per decision point |
| Reward | Log return, fee-aware, and drawdown-penalty variants |
| Fee | 20 bps one-side, only on position changes |

The Q-table stores action values by decision point and state. During live trading, the bot loads the trained table and acts greedily with `training=False`.

## Current Paper Portfolio

| Bot | Research Hypothesis | Notes |
|---|---|---|
| `RL_CROSS` | `7226.HK` drawdown penalty + three-feature binary state | Leveraged ETF cross-check |
| `RL_VOL` | `7226.HK` drawdown penalty + volatility state | More exposure-seeking in observed live ticks |
| `RL_FACOV` | `7226.HK` fee-aware + z-score ternary state | Watch fees; backtest fee ratio was close to the kill threshold |
| `RL_BROAD_A` | `2800.HK` fee-aware + z-score ternary state | Broad HSI benchmark candidate |
| `RL_BROAD_B` | `2828.HK` log-return + volatility ternary state | Federated-partner asset candidate |

The `7299.HK` candidates were removed from live paper deployment because the broker API rejected the symbol for trading, even though price reads were available.

## Why Fees Matter

The round trip fee is 40 bps. Many intraday ETF moves are smaller than that, so frequent trading can lose money even when direction is sometimes right.

The diagnostics report should always be read with fees in mind:

- High trade count can destroy small gross edge.
- Oracle results show an upper bound after applying the same fee model.
- A model that beats random but loses to buy-and-hold may still be weak.

## Why The Model May Choose Cash

Cash can be rational when:

- The state has no learned edge.
- Fees dominate expected move.
- Historical windows are sideways or noisy.
- The Q-table has sparse visits for a state.

The hard rule is that exact Q-value ties must break randomly, not default to cash.

## Trade-Seeking Experiment

The `log_return_long_bias` reward adds a small bonus for being long after a step. It was introduced to test whether the agent was overly reluctant to take exposure.

Treat this as a research lever, not proof of profitability. It can make the model trade more, but it can also overpay fees.

## Future Work

- M5 and M15 decision-point experiments.
- Additional state encoders: RSI, realized volatility, volume regime, trend slope.
- Better walk-forward validation and market-regime tagging.
- DQN agent plugin once the tabular workflow is stable.
- More attractive, interactive experiment reports.
- Automated fee-ratio monitoring, especially for `RL_FACOV` where the warning threshold is 13% and the kill threshold is 15%.
