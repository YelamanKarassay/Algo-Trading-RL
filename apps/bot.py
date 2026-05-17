from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import quantphemes_rl  # noqa: F401
from quantphemes_rl.api_client._undocumented import UndocumentedQuantphemesClient
from quantphemes_rl.api_client.sync import get_position_state, patch_target
from quantphemes_rl.config import load_config
from quantphemes_rl.data.base import DayData, group_by_day
from quantphemes_rl.registry import build
from quantphemes_rl.state.base import MarketContext

HK_TZ = ZoneInfo("Asia/Hong_Kong")
@dataclass
class RuntimeState:
    today_date: str | None = None
    today_open: float | None = None
    yesterday_close: float | None = None
    last_action_at_decision: dict[str, int] = field(default_factory=dict)
def main(argv: list[str] | None = None) -> None:
    _load_env_file(Path(".env"))
    parser = argparse.ArgumentParser(description="Run the Quantphemes live trading bot.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy-id", default=os.environ.get("QUANTPHEMES_STRATEGY_ID"))
    parser.add_argument("--portfolio-id", default=os.environ.get("QUANTPHEMES_PORTFOLIO_ID"))
    parser.add_argument("--api-key")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--simulate-now")
    args = parser.parse_args(argv)
    if not args.strategy_id or not args.portfolio_id:
        parser.error("--strategy-id/--portfolio-id or matching .env values are required")
    run_bot(
        args.config,
        args.strategy_id,
        args.portfolio_id,
        args.api_key or os.environ.get("QUANTPHEMES_API_KEY"),
        args.dry_run,
        _parse_now(args.simulate_now),
    )
def run_bot(
    config_path: Path,
    strategy_id: str,
    portfolio_id: str,
    api_key: str | None,
    dry_run: bool,
    simulate_now: datetime | None = None,
) -> None:
    del portfolio_id
    cfg = load_config(config_path)
    encoder = build("state_encoder", cfg.state.encoder, **cfg.state.kwargs)
    agent = build("agent", cfg.agent.type, **cfg.agent.kwargs)
    q_state_path = Path(cfg.artifacts.q_state_path)
    if q_state_path.exists():
        agent.load(q_state_path)
    client = UndocumentedQuantphemesClient(api_key) if api_key else None
    log_dir = Path(cfg.artifacts.log_dir)
    runtime_path = Path("artifacts/runtime_state.json")
    runtime = load_runtime_state(runtime_path)
    now = simulate_now or datetime.now(HK_TZ)
    if simulate_now is None:
        _wait_for_open_window(now)
        now = datetime.now(HK_TZ)
    trading_date = now.date().isoformat()
    if runtime.today_date != trading_date:
        runtime = RuntimeState(
            today_date=trading_date,
            yesterday_close=_latest_historical_close(cfg, now.date()),
        )
    day = _local_day(cfg, now.date()) if client is None else None
    today_open, open_error = _safe_price_for("09:30", cfg, client, day)
    if today_open is None:
        _write_log(
            log_dir,
            now.date(),
            {
                "timestamp": _iso_now(),
                "status": "skip_no_open",
                "error": open_error,
            },
        )
        return
    runtime.today_date = trading_date
    runtime.today_open = today_open
    runtime.yesterday_close = runtime.yesterday_close or today_open
    save_runtime_state(runtime_path, runtime)
    for index, decision_time in enumerate(cfg.market.decision_times):
        if simulate_now is None:
            target_time = _same_day_time(datetime.now(HK_TZ), decision_time)
            if target_time < datetime.now(HK_TZ):
                continue
            _sleep_until(target_time - timedelta(minutes=1))
            _sleep_until(target_time)
        price, price_error = _safe_price_for(decision_time, cfg, client, day)
        if price is None:
            entry = _error_entry(
                "price_error",
                price_error or "price unavailable",
                decision_time=decision_time,
            )
            _write_log(log_dir, now.date(), entry)
            save_runtime_state(runtime_path, runtime)
            continue
        try:
            entry = handle_decision(
                client,
                strategy_id,
                cfg,
                agent,
                encoder,
                runtime,
                decision_time,
                index,
                price,
                dry_run,
            )
        except Exception as exc:
            entry = _error_entry("decision_error", _format_error(exc), decision_time=decision_time)
        _write_log(log_dir, now.date(), entry)
        save_runtime_state(runtime_path, runtime)
    if simulate_now is None:
        force_close_at = _same_day_time(datetime.now(HK_TZ), cfg.market.force_close_time)
        if force_close_at >= datetime.now(HK_TZ):
            _sleep_until(force_close_at)
    close, close_error = _safe_price_for(cfg.market.force_close_time, cfg, client, day)
    if close is None:
        close_entry = _error_entry("force_close_price_error", close_error or "price unavailable")
    else:
        try:
            close_entry = handle_force_close(
                client,
                strategy_id,
                cfg,
                close,
                dry_run,
            )
        except Exception as exc:
            close_entry = _error_entry("force_close_error", _format_error(exc))
    _write_log(log_dir, now.date(), close_entry)
    if close is not None:
        runtime.yesterday_close = close
        save_runtime_state(runtime_path, runtime)
def compute_target_quantity(
    action: int, cash: float, current_qty: int, price: float, lot_size: int, fee_rate: float
) -> int:
    if action == 0:
        return 0
    return current_qty + int(cash // (lot_size * price * (1.0 + fee_rate))) * lot_size
def handle_decision(
    client: Any,
    strategy_id: str,
    cfg: Any,
    agent: Any,
    encoder: Any,
    runtime: RuntimeState,
    decision_time: str,
    decision_index: int,
    price: float,
    dry_run: bool,
) -> dict[str, Any]:
    cash, current_qty = (
        (float(cfg.market.capital), 0)
        if client is None
        else get_position_state(client, strategy_id)
    )
    ctx = MarketContext(
        price,
        runtime.today_open or price,
        runtime.yesterday_close or runtime.today_open or price,
        price,
        decision_index,
        len(cfg.market.decision_times),
    )
    state = encoder.encode(ctx)
    act_at = getattr(agent, "act_at", None)
    action = int(
        act_at(decision_index, state, False) if callable(act_at) else agent.act(state, False)
    )
    target = compute_target_quantity(
        action,
        cash,
        current_qty,
        price,
        cfg.market.lot_size,
        cfg.market.fee_bps_one_side / 10_000,
    )
    status = "heartbeat" if target == current_qty else "target_changed"
    if not dry_run and client is not None:
        patch_target(client, strategy_id, _api_symbol(cfg.data.symbol), target)
    runtime.last_action_at_decision[decision_time] = action
    return {
        "timestamp": _iso_now(),
        "state_tuple": [ctx.current_price, ctx.today_open, ctx.yesterday_close, price],
        "state": state,
        "action": action,
        "price": price,
        "target_qty": target,
        "current_qty": current_qty,
        "dry_run": dry_run,
        "status": "dry_run" if dry_run and status != "heartbeat" else status,
    }
def handle_force_close(
    client: Any,
    strategy_id: str,
    cfg: Any,
    price: float,
    dry_run: bool,
    confirm_attempts: int = 3,
    confirm_sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    cash, current_qty = (
        (float(cfg.market.capital), 0)
        if client is None
        else get_position_state(client, strategy_id)
    )
    del cash
    post_qty = current_qty
    if not dry_run and client is not None:
        patch_target(client, strategy_id, _api_symbol(cfg.data.symbol), 0)
        post_qty = _confirm_target_quantity(
            client,
            strategy_id,
            0,
            attempts=confirm_attempts,
            sleep_seconds=confirm_sleep_seconds,
        )
    return {
        "timestamp": _iso_now(),
        "status": "force_close" if post_qty == 0 else "force_close_pending",
        "price": price,
        "target_qty": 0,
        "current_qty": current_qty,
        "post_qty": post_qty,
        "dry_run": dry_run,
    }
def _confirm_target_quantity(
    client: Any,
    strategy_id: str,
    target_qty: int,
    attempts: int = 3,
    sleep_seconds: float = 2.0,
) -> int:
    post_qty = target_qty
    for attempt in range(attempts):
        _, post_qty = get_position_state(client, strategy_id)
        if post_qty == target_qty:
            return post_qty
        if attempt < attempts - 1:
            time.sleep(sleep_seconds)
    return post_qty
def load_runtime_state(path: Path) -> RuntimeState:
    return RuntimeState(**json.loads(path.read_text("utf-8"))) if path.exists() else RuntimeState()
def save_runtime_state(path: Path, state: RuntimeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")
def _local_day(cfg: Any, target_date: date) -> DayData:
    source = build("data_source", cfg.data.source, path=cfg.data.path)
    days = group_by_day(
        source.load(cfg.data.symbol, cfg.data.start, cfg.data.end, "1h"), cfg.data.symbol
    )
    return next((day for day in days if day.date == target_date), days[0])
def _latest_historical_close(cfg: Any, before_date: date) -> float | None:
    source = build("data_source", cfg.data.source, path=cfg.data.path)
    days = group_by_day(
        source.load(cfg.data.symbol, cfg.data.start, cfg.data.end, "1h"), cfg.data.symbol
    )
    prior_days = [day for day in days if day.date < before_date]
    return prior_days[-1].close_price if prior_days else None
def _price_for(
    time_str: str,
    cfg: Any,
    client: UndocumentedQuantphemesClient | None,
    day: DayData | None,
) -> float | None:
    if day is not None:
        return day.prices_at_decision_times([time_str])[time_str]
    if client is None:
        return None
    payload = client.get_last_price(_price_symbol(cfg.data.symbol))
    if isinstance(payload, list):
        payload = payload[0]
    price = payload.get("price") or payload.get("last_price")
    return None if price in (None, 0, "0") else float(price)
def _safe_price_for(
    time_str: str,
    cfg: Any,
    client: UndocumentedQuantphemesClient | None,
    day: DayData | None,
) -> tuple[float | None, str | None]:
    try:
        return _price_for(time_str, cfg, client, day), None
    except Exception as exc:
        return None, _format_error(exc)
def _error_entry(
    status: str,
    error: str,
    decision_time: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "timestamp": _iso_now(),
        "status": status,
        "error": error,
    }
    if decision_time is not None:
        entry["decision_time"] = decision_time
    return entry
def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
def _write_log(log_dir: Path, trading_date: date, entry: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entry, sort_keys=True)
    with (log_dir / f"bot_{trading_date.isoformat()}.jsonl").open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    os.write(1, f"{text}\n".encode())
def _next_time(now: datetime, hhmm: str) -> datetime:
    hour, minute = [int(part) for part in hhmm.split(":")]
    target = now.astimezone(HK_TZ).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target >= now else target + timedelta(days=1)
def _same_day_time(now: datetime, hhmm: str) -> datetime:
    hour, minute = [int(part) for part in hhmm.split(":")]
    return now.astimezone(HK_TZ).replace(hour=hour, minute=minute, second=0, microsecond=0)
def _sleep_until(target: datetime) -> None:
    time.sleep(max(0.0, (target - datetime.now(HK_TZ)).total_seconds()))
def _wait_for_open_window(now: datetime) -> None:
    local_now = now.astimezone(HK_TZ)
    if not _is_trading_day(local_now.date()):
        _sleep_until(_next_trading_open(local_now))
        return
    open_buffer = local_now.replace(hour=9, minute=25, second=0, microsecond=0)
    open_time = local_now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = local_now.replace(hour=16, minute=5, second=0, microsecond=0)
    if local_now < open_buffer:
        _sleep_until(open_buffer)
        _sleep_until(open_time)
    elif local_now < open_time:
        _sleep_until(open_time)
    elif local_now > close_time:
        _sleep_until(_next_trading_open(local_now))
def _next_trading_open(now: datetime) -> datetime:
    target_date = now.astimezone(HK_TZ).date() + timedelta(days=1)
    while not _is_trading_day(target_date):
        target_date += timedelta(days=1)
    return datetime.combine(target_date, datetime.min.time(), tzinfo=HK_TZ).replace(
        hour=9,
        minute=30,
        second=0,
        microsecond=0,
    )
def _is_trading_day(value: date) -> bool:
    return value.weekday() < 5
def _parse_now(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value).astimezone(HK_TZ)
def _iso_now() -> str:
    return datetime.now(UTC).isoformat()
def _api_symbol(symbol: str) -> str:
    return symbol
def _price_symbol(symbol: str) -> str:
    return symbol.removesuffix(".HK")
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
if __name__ == "__main__":
    main()
