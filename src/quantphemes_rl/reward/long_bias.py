from __future__ import annotations

import math
from typing import Any

from quantphemes_rl.registry import register
from quantphemes_rl.reward.base import RewardFunction


@register("reward_function", "log_return_long_bias")
class LogReturnLongBiasReward(RewardFunction):
    """Log-return reward with a small bonus for ending an interval long."""

    def __init__(self, beta: float = 0.00003) -> None:
        self.beta = beta

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Return log portfolio return plus beta when the post-step position is long."""
        position = int(info.get("position", 0))
        return math.log(v_t_plus_1 / v_t) + self.beta * position
