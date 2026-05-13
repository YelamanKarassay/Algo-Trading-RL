from __future__ import annotations

from datetime import date, datetime

import pytest

from quantphemes_rl.data.base import HK_TZ, Bar, DayData
from quantphemes_rl.diagnostics.oracle import oracle_nav
from quantphemes_rl.portfolio.portfolio import Portfolio

TIMES = ["10:30", "11:30", "13:30"]
CLOSE = "16:00"


def test_oracle_uptrend() -> None:
    day = _day([100.0, 104.0, 108.0, 110.0])

    final_nav, _ = oracle_nav([day], TIMES, CLOSE, 0.002, 1_002.0, 1)

    assert final_nav == pytest.approx(1_097.8, abs=1e-9)


def test_oracle_downtrend() -> None:
    day = _day([100.0, 98.0, 95.0, 90.0])

    final_nav, _ = oracle_nav([day], TIMES, CLOSE, 0.002, 1_002.0, 1)

    assert final_nav == 1_002.0


def test_oracle_zigzag() -> None:
    day = _day([100.0, 110.0, 90.0, 120.0])

    final_nav, _ = oracle_nav([day], TIMES, CLOSE, 0.002, 1_002.0, 1)

    assert final_nav == pytest.approx(1_452.76, abs=1e-9)


def test_oracle_fee_matches_portfolio() -> None:
    day = _day([100.0, 105.0, 108.0, 110.0])
    portfolio = Portfolio(1_002.0, 0, 0, 0.0, 0.002, 1)
    portfolio.execute(1, 100.0)
    portfolio.execute(1, 105.0)
    portfolio.execute(1, 108.0)
    portfolio.execute(0, 110.0)

    final_nav, _ = oracle_nav([day], TIMES, CLOSE, 0.002, 1_002.0, 1)

    assert final_nav == pytest.approx(portfolio.market_value(110.0), abs=1e-9)


def _day(prices: list[float]) -> DayData:
    labels = [*TIMES, CLOSE]
    bars = [
        Bar(
            timestamp=datetime(2026, 5, 4, int(label[:2]), int(label[3:]), tzinfo=HK_TZ),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1_000.0,
        )
        for label, price in zip(labels, prices, strict=True)
    ]
    return DayData(date=date(2026, 5, 4), symbol="2800.HK", bars=bars)
