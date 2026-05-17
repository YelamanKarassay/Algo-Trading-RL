from __future__ import annotations

import math

import pytest

from quantphemes_rl.reward.log_return import LogReturnReward
from quantphemes_rl.reward.long_bias import LogReturnLongBiasReward
from quantphemes_rl.reward.research import (
    DrawdownPenaltyReward,
    FeeAwareReward,
    SparseLiquidationReward,
)


def test_log_return_matches_math_log() -> None:
    reward = LogReturnReward()

    assert reward.compute(100.0, 105.0, {}) == math.log(105.0 / 100.0)


def test_log_return_long_bias_adds_bonus_when_long() -> None:
    reward = LogReturnLongBiasReward(beta=0.01)

    assert reward.compute(100.0, 105.0, {"position": 1}) == math.log(105.0 / 100.0) + 0.01
    assert reward.compute(100.0, 105.0, {"position": 0}) == math.log(105.0 / 100.0)


def test_fee_aware_penalizes_fee_spend() -> None:
    reward = FeeAwareReward(fee_penalty=2.0)

    assert reward.compute(100.0, 105.0, {"fee": 1.5}) == pytest.approx(
        math.log(105.0 / 100.0) - 0.03
    )


def test_sparse_liquidation_rewards_only_terminal_step() -> None:
    reward = SparseLiquidationReward()

    assert reward.compute(100.0, 105.0, {"done": False}) == 0.0
    assert reward.compute(100.0, 105.0, {"done": True}) == math.log(105.0 / 100.0)


def test_drawdown_penalty_subtracts_drawdown() -> None:
    reward = DrawdownPenaltyReward(drawdown_penalty=0.5)

    assert reward.compute(100.0, 105.0, {"drawdown": 0.02}) == pytest.approx(
        math.log(105.0 / 100.0) - 0.01
    )
