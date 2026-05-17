from __future__ import annotations

from quantphemes_rl.state.base import MarketContext
from quantphemes_rl.state.research_encoders import (
    WithRsiEncoder,
    WithVolatilityEncoder,
    WithVolumeEncoder,
    ZScoreBinsEncoder,
)
from quantphemes_rl.state.three_feature import ThreeFeatureEncoder

THRESHOLDS = {
    "delta_close": [-0.01, 0.01],
    "delta_open": [-0.02, 0.02],
    "delta_prev": [-0.03, 0.03],
}


def test_boundary_inputs_map_to_correct_bins() -> None:
    encoder = ThreeFeatureEncoder(THRESHOLDS)

    low = encoder.encode(_ctx(current_price=95.0, today_open=100.0, yesterday_close=100.0))
    mid = encoder.encode(_ctx(current_price=100.0, today_open=100.0, yesterday_close=100.0))
    high = encoder.encode(_ctx(current_price=104.0, today_open=100.0, yesterday_close=100.0))

    assert low == 0
    assert mid == 13
    assert high == 26


def test_state_index_range_and_num_states() -> None:
    encoder = ThreeFeatureEncoder(THRESHOLDS)

    states = [
        encoder.encode(_ctx(current_price=95.0, today_open=100.0, yesterday_close=100.0)),
        encoder.encode(_ctx(current_price=100.0, today_open=100.0, yesterday_close=100.0)),
        encoder.encode(_ctx(current_price=104.0, today_open=100.0, yesterday_close=100.0)),
    ]

    assert encoder.num_states == 27
    assert all(0 <= state <= 26 for state in states)


def test_with_rsi_adds_three_regimes() -> None:
    encoder = WithRsiEncoder(THRESHOLDS)

    neutral = encoder.encode(_ctx(current_price=100.0, today_open=100.0, yesterday_close=100.0))
    high_rsi = encoder.encode(
        _ctx(
            current_price=100.0,
            today_open=100.0,
            yesterday_close=100.0,
            recent_returns=[0.01, 0.02, 0.01],
        )
    )

    assert encoder.num_states == 81
    assert neutral == 13 * 3 + 1
    assert high_rsi == 13 * 3 + 2


def test_with_volatility_adds_three_regimes() -> None:
    encoder = WithVolatilityEncoder(THRESHOLDS)

    low = encoder.encode(
        _ctx(
            current_price=100.0,
            today_open=100.0,
            yesterday_close=100.0,
            recent_returns=[0.0, 0.0, 0.0],
        )
    )
    high = encoder.encode(
        _ctx(
            current_price=100.0,
            today_open=100.0,
            yesterday_close=100.0,
            recent_returns=[-0.02, 0.02, -0.02, 0.02],
        )
    )

    assert encoder.num_states == 81
    assert low == 13 * 3
    assert high == 13 * 3 + 2


def test_with_volume_adds_relative_volume_regime() -> None:
    encoder = WithVolumeEncoder(THRESHOLDS)

    state = encoder.encode(
        _ctx(
            current_price=100.0,
            today_open=100.0,
            yesterday_close=100.0,
            current_volume=250.0,
            average_volume=100.0,
        )
    )

    assert encoder.num_states == 81
    assert state == 13 * 3 + 2


def test_zscore_bins_state_range() -> None:
    encoder = ZScoreBinsEncoder()

    state = encoder.encode(
        _ctx(
            current_price=102.0,
            today_open=100.0,
            yesterday_close=101.0,
            recent_returns=[-0.01, 0.0, 0.01, 0.02],
        )
    )

    assert encoder.num_states == 27
    assert 0 <= state <= 26


def _ctx(
    current_price: float,
    today_open: float,
    yesterday_close: float,
    recent_returns: list[float] | None = None,
    current_volume: float | None = None,
    average_volume: float | None = None,
) -> MarketContext:
    return MarketContext(
        current_price=current_price,
        today_open=today_open,
        yesterday_close=yesterday_close,
        prev_decision_price=100.0,
        decision_index=0,
        decision_count=5,
        recent_returns=recent_returns,
        current_volume=current_volume,
        average_volume=average_volume,
    )
