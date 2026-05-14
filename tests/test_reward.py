from __future__ import annotations

import math

from quantphemes_rl.reward.log_return import LogReturnReward
from quantphemes_rl.reward.long_bias import LogReturnLongBiasReward


def test_log_return_matches_math_log() -> None:
    reward = LogReturnReward()

    assert reward.compute(100.0, 105.0, {}) == math.log(105.0 / 100.0)


def test_log_return_long_bias_adds_bonus_when_long() -> None:
    reward = LogReturnLongBiasReward(beta=0.01)

    assert reward.compute(100.0, 105.0, {"position": 1}) == math.log(105.0 / 100.0) + 0.01
    assert reward.compute(100.0, 105.0, {"position": 0}) == math.log(105.0 / 100.0)
