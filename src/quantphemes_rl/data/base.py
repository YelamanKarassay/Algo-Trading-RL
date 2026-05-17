from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

HK_TZ = ZoneInfo("Asia/Hong_Kong")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    """Single OHLCV bar with a timezone-aware Hong Kong timestamp."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            msg = "Bar timestamp must be timezone-aware."
            raise ValueError(msg)
        if self.timestamp.tzinfo != HK_TZ:
            object.__setattr__(self, "timestamp", self.timestamp.astimezone(HK_TZ))


@dataclass(frozen=True)
class DayData:
    """All bars for one trading day."""

    date: date
    symbol: str
    bars: list[Bar]
    previous_close: float | None = None

    @property
    def open_price(self) -> float:
        """Return the first bar open for the day."""
        if not self.bars:
            msg = "DayData has no bars."
            raise ValueError(msg)
        return self.bars[0].open

    @property
    def close_price(self) -> float:
        """Return the last bar close for the day."""
        if not self.bars:
            msg = "DayData has no bars."
            raise ValueError(msg)
        return self.bars[-1].close

    def prices_at_decision_times(self, times: list[str]) -> dict[str, float]:
        """Return close prices keyed by HH:MM decision time."""
        prices = {_format_time(bar.timestamp): bar.close for bar in self.bars}
        return {decision_time: prices[decision_time] for decision_time in times}


class BarDataSource(ABC):
    """Abstract interface for loading normalized market bars."""

    @abstractmethod
    def load(self, symbol: str, start: date, end: date, interval: str) -> list[Bar]:
        """Load bars for a symbol and date range."""


def group_by_day(bars: list[Bar], symbol: str = "") -> list[DayData]:
    """Group ascending bars into DayData records."""
    grouped: dict[date, list[Bar]] = {}
    for bar in sorted(bars, key=lambda item: item.timestamp):
        grouped.setdefault(bar.timestamp.date(), []).append(bar)
    days: list[DayData] = []
    previous_close: float | None = None
    for trading_day, day_bars in sorted(grouped.items()):
        day = DayData(
            date=trading_day,
            symbol=symbol,
            bars=day_bars,
            previous_close=previous_close,
        )
        days.append(day)
        previous_close = day.close_price
    return days


def validate_day(day: DayData, required_times: list[str], warn: bool = True) -> bool:
    """Validate that a day contains all required decision bars."""
    present = {_format_time(bar.timestamp) for bar in day.bars}
    missing = [decision_time for decision_time in required_times if decision_time not in present]
    if missing:
        if warn:
            log.warning(
                "Rejecting day with missing required bars",
                extra={
                    "date": day.date.isoformat(),
                    "symbol": day.symbol,
                    "missing_times": missing,
                },
            )
        return False
    return True


def ensure_hk_timestamp(value: object) -> datetime:
    """Parse a timestamp and return it as timezone-aware Hong Kong time."""
    timestamp = datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=HK_TZ)
    return timestamp.astimezone(HK_TZ)


def _format_time(timestamp: datetime) -> str:
    return timestamp.astimezone(HK_TZ).strftime("%H:%M")
