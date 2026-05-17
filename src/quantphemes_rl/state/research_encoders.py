from __future__ import annotations

import math
from statistics import mean, pstdev

from quantphemes_rl.registry import register
from quantphemes_rl.state.base import MarketContext, StateEncoder
from quantphemes_rl.state.three_feature import ThreeFeatureEncoder, _bin


@register("state_encoder", "with_rsi")
class WithRsiEncoder(StateEncoder):
    """Three-feature encoder plus a 3-bin RSI regime."""

    def __init__(
        self,
        thresholds: dict[str, tuple[float, float] | list[float]],
        rsi_thresholds: tuple[float, float] | list[float] = (30.0, 70.0),
    ) -> None:
        self.base = ThreeFeatureEncoder(thresholds)
        self.rsi_thresholds = (float(rsi_thresholds[0]), float(rsi_thresholds[1]))

    @property
    def num_states(self) -> int:
        """Return number of combined states."""
        return self.base.num_states * 3

    def encode(self, ctx: MarketContext) -> int:
        """Encode price state and RSI regime."""
        return self.base.encode(ctx) * 3 + _bin(_rsi(ctx.recent_returns), self.rsi_thresholds)


@register("state_encoder", "with_volatility")
class WithVolatilityEncoder(StateEncoder):
    """Three-feature encoder plus a 3-bin realized-volatility regime."""

    def __init__(
        self,
        thresholds: dict[str, tuple[float, float] | list[float]],
        volatility_thresholds: tuple[float, float] | list[float] = (0.0015, 0.004),
    ) -> None:
        self.base = ThreeFeatureEncoder(thresholds)
        self.volatility_thresholds = (
            float(volatility_thresholds[0]),
            float(volatility_thresholds[1]),
        )

    @property
    def num_states(self) -> int:
        """Return number of combined states."""
        return self.base.num_states * 3

    def encode(self, ctx: MarketContext) -> int:
        """Encode price state and realized-volatility regime."""
        vol = _safe_std(ctx.recent_returns)
        return self.base.encode(ctx) * 3 + _bin(vol, self.volatility_thresholds)


@register("state_encoder", "with_volume")
class WithVolumeEncoder(StateEncoder):
    """Three-feature encoder plus a 3-bin relative-volume regime."""

    def __init__(
        self,
        thresholds: dict[str, tuple[float, float] | list[float]],
        volume_ratio_thresholds: tuple[float, float] | list[float] = (0.8, 1.2),
    ) -> None:
        self.base = ThreeFeatureEncoder(thresholds)
        self.volume_ratio_thresholds = (
            float(volume_ratio_thresholds[0]),
            float(volume_ratio_thresholds[1]),
        )

    @property
    def num_states(self) -> int:
        """Return number of combined states."""
        return self.base.num_states * 3

    def encode(self, ctx: MarketContext) -> int:
        """Encode price state and relative-volume regime."""
        if not ctx.current_volume or not ctx.average_volume:
            ratio = 1.0
        else:
            ratio = ctx.current_volume / ctx.average_volume
        return self.base.encode(ctx) * 3 + _bin(ratio, self.volume_ratio_thresholds)


@register("state_encoder", "zscore_bins")
class ZScoreBinsEncoder(StateEncoder):
    """Three z-scored price features with 3 bins each."""

    def __init__(
        self,
        thresholds: dict[str, tuple[float, float] | list[float]] | None = None,
        z_thresholds: tuple[float, float] | list[float] = (-1.0, 1.0),
        min_std: float = 1e-8,
    ) -> None:
        del thresholds
        self.z_thresholds = (float(z_thresholds[0]), float(z_thresholds[1]))
        self.min_std = min_std

    @property
    def num_states(self) -> int:
        """Return number of z-score states."""
        return 27

    def encode(self, ctx: MarketContext) -> int:
        """Encode z-scored close/open/previous-decision deltas."""
        center = mean(ctx.recent_returns or [0.0])
        scale = max(_safe_std(ctx.recent_returns), self.min_std)
        s1 = _bin(
            _z(ctx.current_price / ctx.yesterday_close - 1.0, center, scale),
            self.z_thresholds,
        )
        s2 = _bin(_z(ctx.current_price / ctx.today_open - 1.0, center, scale), self.z_thresholds)
        s3 = _bin(
            _z(ctx.current_price / ctx.prev_decision_price - 1.0, center, scale),
            self.z_thresholds,
        )
        return s1 * 9 + s2 * 3 + s3


def _rsi(returns: list[float] | None) -> float:
    if not returns:
        return 50.0
    gains = [item for item in returns if item > 0]
    losses = [-item for item in returns if item < 0]
    avg_gain = mean(gains) if gains else 0.0
    avg_loss = mean(losses) if losses else 0.0
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _safe_std(values: list[float] | None) -> float:
    if not values or len(values) < 2:
        return 0.0
    result = pstdev(values)
    return result if math.isfinite(result) else 0.0


def _z(value: float, center: float, scale: float) -> float:
    return (value - center) / scale
