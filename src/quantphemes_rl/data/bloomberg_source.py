from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantphemes_rl.data.base import Bar, BarDataSource
from quantphemes_rl.data.csv_source import _bars_from_frame
from quantphemes_rl.registry import register

REQUIRED_COLUMNS = {"date", "open", "high", "low", "last_price", "volume"}


@register("data_source", "bloomberg")
class BloombergDataSource(BarDataSource):
    """Load Bloomberg Excel exports.

    Bloomberg field names vary across export templates. This adapter matches column names
    case-insensitively and expects the logical fields Date, Open, High, Low, Last_Price, and
    Volume on a sheet named "Data".
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"BloombergDataSource(path={self.path!s})"

    def load(self, symbol: str, start: date, end: date, interval: str) -> list[Bar]:
        """Load bars from a Bloomberg workbook."""
        del interval
        try:
            frame = pd.read_excel(self.path, sheet_name="Data")
        except Exception as exc:
            msg = f"Unable to read Bloomberg workbook from {self.path}: {exc}"
            raise ValueError(msg) from exc

        return _bars_from_frame(
            frame=frame,
            required_columns=REQUIRED_COLUMNS,
            timestamp_column="timestamp",
            symbol=symbol,
            start=start,
            end=end,
            source=str(self.path),
            column_map={"date": "timestamp", "last_price": "close"},
        )
