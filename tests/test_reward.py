from __future__ import annotations

import math

from quantphemes_rl.reward.log_return import LogReturnReward


def test_log_return_matches_math_log() -> None:
    reward = LogReturnReward()

    assert reward.compute(100.0, 105.0, {}) == math.log(105.0 / 100.0)
