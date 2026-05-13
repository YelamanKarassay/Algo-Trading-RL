from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantphemes_rl.data.base import Bar, BarDataSource
from quantphemes_rl.data.csv_source import _bars_from_frame
from quantphemes_rl.registry import register

REQUIRED_COLUMNS = {"time_key", "open", "close", "high", "low", "volume", "turnover"}


@register("data_source", "futu")
class FutuDataSource(BarDataSource):
    """Load Futu-exported CSV bars into the common Bar interface."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"FutuDataSource(path={self.path!s})"

    def load(self, symbol: str, start: date, end: date, interval: str) -> list[Bar]:
        """Load bars from a Futu CSV export."""
        del interval
        try:
            frame = pd.read_csv(self.path)
        except Exception as exc:
            msg = f"Unable to read Futu CSV data from {self.path}: {exc}"
            raise ValueError(msg) from exc

        return _bars_from_frame(
            frame=frame,
            required_columns=REQUIRED_COLUMNS,
            timestamp_column="timestamp",
            symbol=symbol,
            start=start,
            end=end,
            source=str(self.path),
            column_map={"time_key": "timestamp"},
        )
