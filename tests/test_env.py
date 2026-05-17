from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pytest

from quantphemes_rl.data.base import HK_TZ, Bar, DayData
from quantphemes_rl.env.trading_env import TradingEnv
from quantphemes_rl.portfolio.portfolio import Portfolio
from quantphemes_rl.reward.base import RewardFunction
from quantphemes_rl.state.base import MarketContext, StateEncoder

DECISION_TIMES = ["10:30", "11:30", "13:30", "14:30", "15:30"]
FORCE_CLOSE_TIME = "16:00"


class FixedStateEncoder(StateEncoder):
    @property
    def num_states(self) -> int:
        return len(DECISION_TIMES)

    def encode(self, ctx: MarketContext) -> int:
        return ctx.decision_index


class LogReturnReward(RewardFunction):
    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        del info
        return math.log(v_t_plus_1 / v_t)


class RecordingReward(RewardFunction):
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, dict[str, Any]]] = []

    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        self.calls.append((v_t, v_t_plus_1, info))
        return math.log(v_t_plus_1 / v_t)


class CapturingEncoder(StateEncoder):
    def __init__(self) -> None:
        self.contexts: list[MarketContext] = []

    @property
    def num_states(self) -> int:
        return len(DECISION_TIMES)

    def encode(self, ctx: MarketContext) -> int:
        self.contexts.append(ctx)
        return ctx.decision_index


def test_all_cash_day() -> None:
    env = _make_env(Portfolio(cash=1_000.0, shares=0, position=0, total_fees=0.0))

    rewards, infos = _run_actions(env, [0, 0, 0, 0, 0])

    assert infos[-1]["v_t_plus_1"] == 1_000.0
    assert env.portfolio.total_fees == 0.0
    assert sum(rewards) == pytest.approx(0.0, abs=1e-9)


def test_all_long_uptrend() -> None:
    capital = 1_002.0
    env = _make_env(
        Portfolio(
            cash=capital,
            shares=0,
            position=0,
            total_fees=0.0,
            fee_rate_one_side=0.002,
            lot_size=1,
        )
    )

    rewards, infos = _run_actions(env, [1, 1, 1, 1, 1])

    buy_fee = 10 * 100.0 * 0.002
    sell_fee = 10 * 110.0 * 0.002
    expected_final_nav = capital - buy_fee + 10 * (110.0 - 100.0) - sell_fee
    assert infos[-1]["v_t_plus_1"] == pytest.approx(expected_final_nav, abs=1e-9)
    assert sum(rewards) == pytest.approx(math.log(expected_final_nav / capital), abs=1e-9)


def test_alternating_actions() -> None:
    env = _make_env(
        Portfolio(
            cash=1_002.0,
            shares=0,
            position=0,
            total_fees=0.0,
            fee_rate_one_side=0.002,
            lot_size=1,
        )
    )

    _, infos = _run_actions(env, [0, 1, 0, 1, 0])

    fee_events = [info for info in infos if info["fee"] > 0.0]
    assert len(fee_events) == 4


def test_boundary_fee_captured() -> None:
    env = _make_env(
        Portfolio(
            cash=1_002.0,
            shares=0,
            position=0,
            total_fees=0.0,
            fee_rate_one_side=0.002,
            lot_size=1,
        )
    )

    rewards, infos = _run_actions(env, [1, 1, 1, 1, 1])
    terminal = infos[-1]

    assert terminal["force_close_fee"] == pytest.approx(2.2, abs=1e-9)
    assert terminal["position"] == 0
    assert terminal["shares"] == 0
    assert rewards[-1] == pytest.approx(
        math.log(terminal["v_t_plus_1"] / terminal["v_t"]),
        abs=1e-9,
    )


def test_v_t_measured_before_trade() -> None:
    reward = RecordingReward()
    env = _make_env(
        Portfolio(
            cash=1_002.0,
            shares=0,
            position=0,
            total_fees=0.0,
            fee_rate_one_side=0.002,
            lot_size=1,
        ),
        reward_fn=reward,
    )
    env.reset()

    _, _, _, info = env.step(1)

    assert reward.calls[0][0] == 1_002.0
    assert info["v_t"] == 1_002.0
    assert info["v_t_plus_1"] == 1_020.0


def test_context_uses_previous_close_when_available() -> None:
    encoder = CapturingEncoder()
    day = _make_uptrend_day(previous_close=99.5)
    env = TradingEnv(
        day=day,
        encoder=encoder,
        reward_fn=LogReturnReward(),
        portfolio=Portfolio(cash=1_000.0, shares=0, position=0, total_fees=0.0),
        decision_times=DECISION_TIMES,
        force_close_time=FORCE_CLOSE_TIME,
    )

    env.reset()

    assert encoder.contexts[0].yesterday_close == 99.5


def _make_env(portfolio: Portfolio, reward_fn: RewardFunction | None = None) -> TradingEnv:
    return TradingEnv(
        day=_make_uptrend_day(),
        encoder=FixedStateEncoder(),
        reward_fn=reward_fn or LogReturnReward(),
        portfolio=portfolio,
        decision_times=DECISION_TIMES,
        force_close_time=FORCE_CLOSE_TIME,
    )


def _run_actions(env: TradingEnv, actions: list[int]) -> tuple[list[float], list[dict[str, Any]]]:
    env.reset()
    rewards: list[float] = []
    infos: list[dict[str, Any]] = []
    done = False
    for action in actions:
        _, reward, done, info = env.step(action)
        rewards.append(reward)
        infos.append(info)
        if done:
            break
    assert done
    return rewards, infos


def _make_uptrend_day(previous_close: float | None = None) -> DayData:
    prices = {
        "10:30": 100.0,
        "11:30": 102.0,
        "13:30": 104.0,
        "14:30": 106.0,
        "15:30": 108.0,
        "16:00": 110.0,
    }
    bars = [
        Bar(
            timestamp=datetime(2026, 5, 4, int(time[:2]), int(time[3:]), tzinfo=HK_TZ),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000.0,
        )
        for time, price in prices.items()
    ]
    return DayData(
        date=date(2026, 5, 4),
        symbol="2800.HK",
        bars=bars,
        previous_close=previous_close,
    )
