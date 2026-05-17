from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import quantphemes_rl
from quantphemes_rl.data.base import HK_TZ, DayData, group_by_day, validate_day
from quantphemes_rl.data.csv_source import CSVDataSource
from quantphemes_rl.registry import build, list_registered

FIXTURE = Path("tests/fixtures/synthetic_5days_1h.csv")
DECISION_TIMES = ["10:30", "11:30", "13:30", "14:30", "15:30"]


def test_csv_source_loads_5_days() -> None:
    source = CSVDataSource(path=str(FIXTURE))

    bars = source.load(
        symbol="2800.HK",
        start=date(2026, 5, 4),
        end=date(2026, 5, 8),
        interval="1h",
    )

    assert len(bars) == 35
    assert bars[0].timestamp.tzinfo == HK_TZ
    assert bars[0].close == 100.1
    assert bars[-1].close == 105.3
    assert bars == sorted(bars, key=lambda bar: bar.timestamp)


def test_csv_source_filters_by_date_range() -> None:
    source = CSVDataSource(path=str(FIXTURE))

    bars = source.load(
        symbol="2800.HK",
        start=date(2026, 5, 5),
        end=date(2026, 5, 6),
        interval="1h",
    )

    assert len(bars) == 14
    assert {bar.timestamp.date() for bar in bars} == {date(2026, 5, 5), date(2026, 5, 6)}


def test_group_by_day_handles_lunch_break() -> None:
    source = CSVDataSource(path=str(FIXTURE))
    bars = source.load(
        symbol="2800.HK",
        start=date(2026, 5, 4),
        end=date(2026, 5, 8),
        interval="1h",
    )

    days = group_by_day(bars, symbol="2800.HK")

    assert len(days) == 5
    assert all(validate_day(day, DECISION_TIMES) for day in days)
    assert "12:30" not in days[0].prices_at_decision_times(DECISION_TIMES)
    assert days[0].open_price == 100.0
    assert days[0].close_price == 101.3
    assert days[0].previous_close is None
    assert days[1].previous_close == days[0].close_price


def test_validate_day_rejects_missing_bar() -> None:
    source = CSVDataSource(path=str(FIXTURE))
    bars = source.load(
        symbol="2800.HK",
        start=date(2026, 5, 4),
        end=date(2026, 5, 4),
        interval="1h",
    )
    incomplete_day = DayData(
        date=date(2026, 5, 4),
        symbol="2800.HK",
        bars=[bar for bar in bars if bar.timestamp.strftime("%H:%M") != "13:30"],
    )

    assert not validate_day(incomplete_day, DECISION_TIMES)


def test_data_source_registry_lookup() -> None:
    assert quantphemes_rl is not None

    source = build("data_source", "csv", path=str(FIXTURE))

    assert isinstance(source, CSVDataSource)
    assert list_registered("data_source") == ["bloomberg", "csv", "futu"]


def test_malformed_csv_raises_value_error(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.csv"
    malformed.write_text("timestamp,open\n2026-05-04 09:30:00,100\n")
    source = CSVDataSource(path=str(malformed))

    with pytest.raises(ValueError, match="missing columns"):
        source.load(
            symbol="2800.HK",
            start=date(2026, 5, 4),
            end=date(2026, 5, 4),
            interval="1h",
        )
