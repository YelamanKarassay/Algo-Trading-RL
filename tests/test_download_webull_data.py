from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from apps import download_webull_data


class FakeMarketData:
    def __init__(self) -> None:
        self.calls = 0

    def get_history_bar(
        self,
        symbol: str,
        category: str,
        timespan: str,
        *,
        count: str,
        end_time: int,
    ) -> object:
        del symbol, category, timespan, count, end_time
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("HTTP Status: 429, Code: TOO_MANY_REQUESTS")
        return object()


class FakeClient:
    def __init__(self) -> None:
        self.market_data = FakeMarketData()


def test_history_bar_retry_handles_throttle(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(download_webull_data.time, "sleep", sleeps.append)
    spec = download_webull_data.DownloadSpec(
        canonical_symbol="2800.HK",
        webull_symbol="02800",
        category="HK_ETF",
        timespan="M1",
        start=date(2026, 1, 1),
        end=date(2026, 5, 17),
        output=Path("unused.csv"),
    )

    result = download_webull_data._get_history_bar_with_retry(
        FakeClient(),
        spec,
        "1200",
        datetime(2026, 5, 17, tzinfo=ZoneInfo("Asia/Hong_Kong")),
    )

    assert result is not None
    assert sleeps == [5.0, 10.0]
