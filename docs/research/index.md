# Research Notes

The current model is intentionally simple: a tabular Q-learning agent with binary actions.

## Current Q-Table Model

| Item | Current Shape |
|---|---|
| Asset | Production uses `2800.HK` |
| Actions | `0` cash, `1` fully invested |
| State | Three-feature encoder, 27 states |
| Tables | One table per decision point |
| Reward | Log return, with optional long-bias experiment |
| Fee | 20 bps one-side, only on position changes |

The Q-table stores action values by decision point and state. During live trading, the bot loads the trained table and acts greedily with `training=False`.

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
