from __future__ import annotations

from quantphemes_rl.state.base import MarketContext
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


def _ctx(current_price: float, today_open: float, yesterday_close: float) -> MarketContext:
    return MarketContext(
        current_price=current_price,
        today_open=today_open,
        yesterday_close=yesterday_close,
        prev_decision_price=100.0,
        decision_index=0,
        decision_count=5,
    )
