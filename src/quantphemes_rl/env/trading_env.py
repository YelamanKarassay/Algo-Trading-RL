from __future__ import annotations

from typing import Any

from quantphemes_rl.data.base import DayData
from quantphemes_rl.portfolio.portfolio import Portfolio
from quantphemes_rl.reward.base import RewardFunction
from quantphemes_rl.state.base import MarketContext, StateEncoder


class TradingEnv:
    """Single-day binary-position trading environment."""

    def __init__(
        self,
        day: DayData,
        encoder: StateEncoder,
        reward_fn: RewardFunction,
        portfolio: Portfolio,
        decision_times: list[str],
        force_close_time: str,
    ) -> None:
        if not decision_times:
            msg = "decision_times must not be empty."
            raise ValueError(msg)
        self.day = day
        self.encoder = encoder
        self.reward_fn = reward_fn
        self.portfolio = portfolio
        self.decision_times = decision_times
        self.force_close_time = force_close_time
        self._prices = self.day.prices_at_decision_times([*decision_times, force_close_time])
        self._current_index = 0
        self.initial_value: float | None = None

    def reset(self) -> int:
        """Reset to the first decision point and return the initial state index."""
        self._current_index = 0
        first_price = self._price_at(self.decision_times[0])
        self.initial_value = self.portfolio.market_value(first_price)
        return self._encode(self.decision_times[0])

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Execute one decision and return next_state, reward, done, info."""
        current_time = self.decision_times[self._current_index]
        current_price = self._price_at(current_time)
        v_t = self.portfolio.market_value(current_price)
        context = self._context(current_time)
        state_tuple = (
            context.current_price,
            context.today_open,
            context.yesterday_close,
            context.prev_decision_price,
            context.decision_index,
        )

        action_fee = self.portfolio.execute(action, current_price)
        next_time, done = self._next_time()
        next_price = self._price_at(next_time)

        close_fee = 0.0
        if done:
            close_fee = self.portfolio.execute(0, next_price)

        v_t_plus_1 = self.portfolio.market_value(next_price)
        info = {
            "price": current_price,
            "next_price": next_price,
            "time": current_time,
            "next_time": next_time,
            "state_tuple": state_tuple,
            "fee": action_fee + close_fee,
            "action_fee": action_fee,
            "force_close_fee": close_fee,
            "position": self.portfolio.position,
            "cash": self.portfolio.cash,
            "shares": self.portfolio.shares,
            "v_t": v_t,
            "v_t_plus_1": v_t_plus_1,
        }
        reward = self.reward_fn.compute(v_t, v_t_plus_1, info)

        if done:
            next_state = -1
        else:
            self._current_index += 1
            next_state = self._encode(next_time)

        return next_state, reward, done, info

    def _next_time(self) -> tuple[str, bool]:
        next_index = self._current_index + 1
        if next_index >= len(self.decision_times):
            return self.force_close_time, True
        next_time = self.decision_times[next_index]
        if next_time == self.force_close_time:
            return next_time, True
        return next_time, False

    def _price_at(self, decision_time: str) -> float:
        return self._prices[decision_time]

    def _encode(self, decision_time: str) -> int:
        return self.encoder.encode(self._context(decision_time))

    def _context(self, decision_time: str) -> MarketContext:
        decision_index = self.decision_times.index(decision_time)
        prev_time = self.decision_times[max(0, decision_index - 1)]
        return MarketContext(
            current_price=self._price_at(decision_time),
            today_open=self.day.open_price,
            yesterday_close=self.day.open_price,
            prev_decision_price=self._price_at(prev_time),
            decision_index=decision_index,
            decision_count=len(self.decision_times),
            recent_returns=None,
        )
