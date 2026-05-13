from __future__ import annotations

from datetime import date, datetime, timedelta

from quantphemes_rl.data.base import HK_TZ, Bar, DayData
from quantphemes_rl.diagnostics.signal import mutual_info_per_feature
from quantphemes_rl.state.three_feature import ThreeFeatureEncoder

TIMES = ["10:30", "11:30", "13:30"]
THRESHOLDS = {
    "delta_close": [-0.01, 0.01],
    "delta_open": [-0.01, 0.01],
    "delta_prev": [-0.01, 0.01],
}


def test_mi_random_features_near_zero() -> None:
    encoder = ThreeFeatureEncoder(THRESHOLDS)
    days = [_flat_day(index) for index in range(30)]

    frame = mutual_info_per_feature(days, encoder, TIMES)

    assert frame["mi"].max() < 1e-9


def test_mi_deterministic_signal_exceeds_threshold() -> None:
    encoder = ThreeFeatureEncoder(THRESHOLDS)
    days = [_signal_day(index) for index in range(40)]

    frame = mutual_info_per_feature(days, encoder, TIMES)

    assert frame["mi"].max() > 0.1


def _flat_day(index: int) -> DayData:
    return _make_day(index, [100.0, 100.0, 100.0])


def _signal_day(index: int) -> DayData:
    if index % 2 == 0:
        return _make_day(index, [98.0, 104.0, 104.0])
    return _make_day(index, [102.0, 96.0, 96.0])


def _make_day(index: int, prices: list[float]) -> DayData:
    trading_date = date(2026, 5, 4) + timedelta(days=index)
    bars = [
        Bar(
            timestamp=datetime(
                trading_date.year,
                trading_date.month,
                trading_date.day,
                int(label[:2]),
                int(label[3:]),
                tzinfo=HK_TZ,
            ),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000.0,
        )
        for label, price in zip(TIMES, prices, strict=True)
    ]
    return DayData(date=trading_date, symbol="2800.HK", bars=bars)
