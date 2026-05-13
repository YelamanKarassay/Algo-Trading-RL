from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantphemes_rl.data.base import Bar, BarDataSource, ensure_hk_timestamp
from quantphemes_rl.registry import register

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


@register("data_source", "csv")
class CSVDataSource(BarDataSource):
    """Load normalized OHLCV bars from a CSV file."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"CSVDataSource(path={self.path!s})"

    def load(self, symbol: str, start: date, end: date, interval: str) -> list[Bar]:
        """Load bars from CSV, filtered by optional symbol column and date range."""
        del interval
        try:
            frame = pd.read_csv(self.path)
        except Exception as exc:
            msg = f"Unable to read CSV data from {self.path}: {exc}"
            raise ValueError(msg) from exc

        return _bars_from_frame(
            frame=frame,
            required_columns=REQUIRED_COLUMNS,
            timestamp_column="timestamp",
            symbol=symbol,
            start=start,
            end=end,
            source=str(self.path),
        )


def _bars_from_frame(
    *,
    frame: pd.DataFrame,
    required_columns: set[str],
    timestamp_column: str,
    symbol: str,
    start: date,
    end: date,
    source: str,
    column_map: dict[str, str] | None = None,
) -> list[Bar]:
    normalized = frame.rename(columns={column: column.strip().lower() for column in frame.columns})
    missing = required_columns - set(normalized.columns)
    if missing:
        msg = f"Malformed data in {source}: missing columns {sorted(missing)}."
        raise ValueError(msg)

    if column_map:
        normalized = normalized.rename(columns=column_map)

    if "symbol" in normalized.columns:
        normalized = normalized[normalized["symbol"].astype(str) == symbol]

    bars: list[Bar] = []
    for row_number, row in normalized.iterrows():
        try:
            timestamp = ensure_hk_timestamp(row[timestamp_column])
            if not start <= timestamp.date() <= end:
                continue
            volume_value = row["volume"]
            volume = None if pd.isna(volume_value) else float(volume_value)
            bars.append(
                Bar(
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=volume,
                )
            )
        except Exception as exc:
            msg = f"Malformed data in {source} at row {row_number}: {exc}"
            raise ValueError(msg) from exc

    return sorted(bars, key=lambda bar: bar.timestamp)
