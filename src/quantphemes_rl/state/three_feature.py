from __future__ import annotations

from quantphemes_rl.registry import register
from quantphemes_rl.state.base import MarketContext, StateEncoder


@register("state_encoder", "three_feature")
class ThreeFeatureEncoder(StateEncoder):
    """Baseline 3-feature, 3-bin encoder."""

    def __init__(self, thresholds: dict[str, tuple[float, float] | list[float]]) -> None:
        self.thresholds = {
            "delta_close": self._validate_threshold(thresholds, "delta_close"),
            "delta_open": self._validate_threshold(thresholds, "delta_open"),
            "delta_prev": self._validate_threshold(thresholds, "delta_prev"),
        }

    @property
    def num_states(self) -> int:
        """Return the number of possible encoded states."""
        return 27

    def encode(self, ctx: MarketContext) -> int:
        """Encode three relative price deltas as one state index."""
        s1 = _bin(ctx.current_price / ctx.yesterday_close - 1.0, self.thresholds["delta_close"])
        s2 = _bin(ctx.current_price / ctx.today_open - 1.0, self.thresholds["delta_open"])
        s3 = _bin(ctx.current_price / ctx.prev_decision_price - 1.0, self.thresholds["delta_prev"])
        return s1 * 9 + s2 * 3 + s3

    @staticmethod
    def _validate_threshold(
        thresholds: dict[str, tuple[float, float] | list[float]], key: str
    ) -> tuple[float, float]:
        if key not in thresholds:
            msg = f"Missing threshold '{key}'."
            raise ValueError(msg)
        value = thresholds[key]
        if len(value) != 2:
            msg = f"Threshold '{key}' must contain lower and upper bounds."
            raise ValueError(msg)
        lower, upper = float(value[0]), float(value[1])
        if lower > upper:
            msg = f"Threshold '{key}' lower bound must be <= upper bound."
            raise ValueError(msg)
        return lower, upper


def _bin(value: float, thresholds: tuple[float, float]) -> int:
    lower, upper = thresholds
    if value < lower:
        return 0
    if value > upper:
        return 2
    return 1
