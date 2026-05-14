
'''
Download historical bars for Webull HK stocks and ETFs, saving to data/raw.
python -m apps.download_webull_data --universe --timespan M30 --start 2018-01-01


Optional for research:
python -m apps.download_webull_data --universe --timespan M5 --start 2018-01-01
'''



from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore", message="Pandas requires version .*")

import pandas as pd  # noqa: E402

HK_TZ = ZoneInfo("Asia/Hong_Kong")
DEFAULT_UNIVERSE: dict[str, tuple[str, str]] = {
    "2800.HK": ("02800", "HK_ETF"),
    "2828.HK": ("02828", "HK_ETF"),
    "7299.HK": ("07299", "HK_ETF"),
    "7568.HK": ("07568", "HK_ETF"),
    "7226.HK": ("07226", "HK_ETF"),
}


@dataclass(frozen=True)
class DownloadSpec:
    canonical_symbol: str
    webull_symbol: str
    category: str
    timespan: str
    start: date
    end: date
    output: Path


def main(argv: list[str] | None = None) -> None:
    """Download Webull HK historical bars into data/raw."""
    _load_env_file(Path(".env"))
    args = _parse_args(argv)
    logging.disable(logging.CRITICAL)
    data_client = _build_client(args.endpoint)
    for spec in _build_specs(args):
        frame = download_bars(data_client, spec, args.count, args.sleep_seconds)
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(spec.output, index=False)
        _emit(f"saved {len(frame)} rows to {spec.output}")
    _cleanup_sdk_artifacts()


def download_bars(
    data_client: Any, spec: DownloadSpec, count: str, sleep_seconds: float
) -> pd.DataFrame:
    """Download all bars for one spec, paginating backward from end date."""
    start_dt = datetime.combine(spec.start, dt_time.min, tzinfo=HK_TZ)
    cursor = datetime.combine(spec.end, dt_time.max, tzinfo=HK_TZ)
    frames: list[pd.DataFrame] = []
    seen_oldest: set[pd.Timestamp] = set()
    while cursor >= start_dt:
        response = data_client.market_data.get_history_bar(
            spec.webull_symbol,
            spec.category,
            spec.timespan,
            count=count,
            end_time=_millis(cursor),
        )
        if response.status_code != 200:
            msg = f"{spec.canonical_symbol} returned HTTP {response.status_code}"
            raise RuntimeError(msg)
        payload = response.json()
        if not payload:
            break
        frame = _normalize_bars(payload, spec.canonical_symbol, spec.timespan)
        frames.append(frame)
        oldest = pd.to_datetime(frame["timestamp"]).min().tz_localize(HK_TZ)
        newest = pd.to_datetime(frame["timestamp"]).max().tz_localize(HK_TZ)
        _emit(f"{spec.canonical_symbol} {spec.timespan}: {len(frame)} bars {oldest} -> {newest}")
        if oldest in seen_oldest or oldest <= start_dt:
            break
        seen_oldest.add(oldest)
        cursor = oldest.to_pydatetime() - pd.Timedelta(milliseconds=1)
        time.sleep(sleep_seconds)
    if not frames:
        columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp")
    dates = pd.to_datetime(out["timestamp"]).dt.date
    out = out[(dates >= spec.start) & (dates <= spec.end)]
    return out.reset_index(drop=True)


def _normalize_bars(
    payload: list[dict[str, Any]], canonical_symbol: str, timespan: str
) -> pd.DataFrame:
    frame = pd.DataFrame(payload).rename(columns={"time": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert(HK_TZ)
    shift = _timespan_delta(timespan)
    if shift is not None:
        frame["timestamp"] = frame["timestamp"] + shift
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["symbol"] = canonical_symbol
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    return frame[columns].dropna().sort_values("timestamp")


def _build_client(endpoint: str) -> Any:
    try:
        from webull.core.client import ApiClient
        from webull.data.data_client import DataClient
    except ImportError as exc:
        msg = "Install the Webull SDK first: pip install webull-openapi-python-sdk"
        raise RuntimeError(msg) from exc
    app_key = os.environ.get("WEBULL_APP_KEY")
    app_secret = os.environ.get("WEBULL_APP_SECRET")
    if not app_key or not app_secret:
        msg = "WEBULL_APP_KEY and WEBULL_APP_SECRET must be set in .env or the environment."
        raise RuntimeError(msg)
    api_client = ApiClient(app_key, app_secret, "hk")
    api_client.add_endpoint("hk", _normalize_endpoint(endpoint))
    return DataClient(api_client)


def _build_specs(args: argparse.Namespace) -> list[DownloadSpec]:
    symbols = list(DEFAULT_UNIVERSE) if args.universe else args.symbol
    specs = []
    for canonical_symbol in symbols:
        default_webull, default_category = DEFAULT_UNIVERSE.get(
            canonical_symbol, (_to_webull_hk_symbol(canonical_symbol), args.category)
        )
        webull_symbol = args.webull_symbol or default_webull
        category = args.category or default_category
        output = args.output or _default_output(canonical_symbol, args.timespan)
        if len(symbols) > 1 and args.output:
            msg = "--output may only be used with one --symbol."
            raise ValueError(msg)
        specs.append(
            DownloadSpec(
                canonical_symbol=canonical_symbol,
                webull_symbol=webull_symbol,
                category=category,
                timespan=args.timespan,
                start=args.start,
                end=args.end,
                output=output,
            )
        )
    return specs


def _default_output(canonical_symbol: str, timespan: str) -> Path:
    safe_symbol = canonical_symbol.replace(".", "_").lower()
    return Path("data/raw") / f"webull_{safe_symbol}_{timespan.lower()}.csv"


def _to_webull_hk_symbol(symbol: str) -> str:
    root = symbol.upper().removesuffix(".HK").removeprefix("HK.")
    if root.isdigit():
        return root.zfill(5)
    return symbol


def _timespan_delta(timespan: str) -> pd.Timedelta | None:
    upper = timespan.upper()
    if upper.startswith("M") and upper[1:].isdigit():
        return pd.Timedelta(minutes=int(upper[1:]))
    if upper.startswith("S") and upper[1:].isdigit():
        return pd.Timedelta(seconds=int(upper[1:]))
    return None


def _millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _normalize_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint.strip().rstrip("/"))
    return parsed.netloc or endpoint


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Webull HK ETF/stock bars to data/raw.")
    parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Canonical symbol, e.g. 2800.HK",
    )
    parser.add_argument("--webull-symbol", help="Override Webull symbol, e.g. 02800")
    parser.add_argument("--category", default=None, help="Webull category, e.g. HK_ETF or HK_STOCK")
    parser.add_argument("--timespan", default="M30", help="M1, M5, M30, M60, D, etc.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("WEBULL_API_ENDPOINT", "api.sandbox.webull.hk"),
    )
    parser.add_argument("--count", default="1200", help="Rows per request; Webull max is 1200.")
    parser.add_argument("--sleep-seconds", type=float, default=1.05)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--universe",
        action="store_true",
        help="Download the configured ETF universe: 2800, 2828, 7299, 7568, 7226.",
    )
    args = parser.parse_args(argv)
    if not args.universe and not args.symbol:
        parser.error("Provide --symbol at least once, or use --universe.")
    if args.webull_symbol and len(args.symbol) != 1:
        parser.error("--webull-symbol may only be used with exactly one --symbol.")
    return args


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _cleanup_sdk_artifacts() -> None:
    for path in [Path("conf/token.txt"), Path("webull_data_sdk.log"), Path("webull_trade_sdk.log")]:
        path.unlink(missing_ok=True)


def _emit(message: str) -> None:
    os.write(1, f"{message}\n".encode())


if __name__ == "__main__":
    main()
