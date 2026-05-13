from __future__ import annotations

import math
from typing import Any

from quantphemes_rl.registry import register
from quantphemes_rl.reward.base import RewardFunction


@register("reward_function", "log_return")
class LogReturnReward(RewardFunction):
    """Reward equal to log portfolio return."""

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Return log(V_t_plus_1 / V_t)."""
        del info
        return math.log(v_t_plus_1 / v_t)
