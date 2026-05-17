from __future__ import annotations

import math
from typing import Any

from quantphemes_rl.registry import register
from quantphemes_rl.reward.base import RewardFunction


@register("reward_function", "fee_aware")
class FeeAwareReward(RewardFunction):
    """Log return with an extra explicit penalty for fee spend."""

    def __init__(self, fee_penalty: float = 1.0) -> None:
        self.fee_penalty = fee_penalty

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Return log return minus extra normalized fee penalty."""
        fee = float(info.get("fee", 0.0))
        return math.log(v_t_plus_1 / v_t) - self.fee_penalty * fee / v_t


@register("reward_function", "sparse_liquidation")
class SparseLiquidationReward(RewardFunction):
    """Reward only at terminal liquidation, zero during intermediate holds."""

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Return terminal log return and zero otherwise."""
        if not bool(info.get("done", False)):
            return 0.0
        return math.log(v_t_plus_1 / v_t)


@register("reward_function", "drawdown_penalty")
class DrawdownPenaltyReward(RewardFunction):
    """Log return penalized by current intraday drawdown."""

    def __init__(self, drawdown_penalty: float = 1.0) -> None:
        self.drawdown_penalty = drawdown_penalty

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Return log return minus normalized drawdown penalty."""
        drawdown = float(info.get("drawdown", 0.0))
        return math.log(v_t_plus_1 / v_t) - self.drawdown_penalty * drawdown
