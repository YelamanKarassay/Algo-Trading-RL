from __future__ import annotations

from datetime import datetime
from typing import Any

from apps.bot import (
    HK_TZ,
    RuntimeState,
    _next_trading_open,
    _safe_price_for,
    compute_target_quantity,
    handle_decision,
    handle_force_close,
)
from quantphemes_rl.config import load_config


class FixedAgent:
    def __init__(self, action: int) -> None:
        self.action = action

    def act(self, state: int, training: bool) -> int:
        del state, training
        return self.action


class FixedEncoder:
    def encode(self, ctx: Any) -> int:
        del ctx
        return 3


class MockClient:
    def __init__(self, cash: float, qty: int) -> None:
        self.cash = cash
        self.qty = qty
        self.patches: list[dict[str, Any]] = []

    def get_strategy_quantities(self, strategy_id: str) -> dict[str, Any]:
        del strategy_id
        return {"cash": self.cash, "current_qty": self.qty}

    def get_cash_and_asset(self, strategy_id: str) -> dict[str, Any]:
        del strategy_id
        return {"data": {"cash": self.cash, "assets": [{"quantity": self.qty}]}}

    def update_holding(self, strategy_id: str, holdings: dict[str, Any]) -> dict[str, Any]:
        self.patches.append({"strategy_id": strategy_id, "holdings": holdings})
        self.qty = int(holdings["holdings"][0]["stocks"][0]["quantity"])
        return {"ok": True}


class UnfilledMockClient(MockClient):
    def update_holding(self, strategy_id: str, holdings: dict[str, Any]) -> dict[str, Any]:
        self.patches.append({"strategy_id": strategy_id, "holdings": holdings})
        return {"ok": True}


class TimeoutPriceClient:
    def get_last_price(self, symbol: str) -> dict[str, Any]:
        del symbol
        raise TimeoutError("api read timed out")


def test_compute_target_quantity_long() -> None:
    target = compute_target_quantity(
        action=1,
        cash=1_000_000.0,
        current_qty=0,
        price=20.0,
        lot_size=500,
        fee_rate=0.002,
    )

    assert target == 49_500


def test_compute_target_quantity_cash() -> None:
    assert compute_target_quantity(0, 1_000_000.0, 49_500, 20.0, 500, 0.002) == 0


def test_compute_target_quantity_ternary_half() -> None:
    target = compute_target_quantity(
        action=1,
        cash=1_000_000.0,
        current_qty=0,
        price=20.0,
        lot_size=500,
        fee_rate=0.002,
        num_actions=3,
    )

    assert target == 25_000


def test_compute_target_quantity_heartbeat_keeps_existing_full_position() -> None:
    target = compute_target_quantity(
        action=1,
        cash=0.0,
        current_qty=500,
        price=20.0,
        lot_size=500,
        fee_rate=0.002,
    )

    assert target == 500


def test_heartbeat_when_action_unchanged() -> None:
    client = MockClient(cash=0.0, qty=500)

    entry = handle_decision(
        client=client,
        strategy_id="strategy",
        cfg=load_config("experiments/production_2800.yaml"),
        agent=FixedAgent(1),
        encoder=FixedEncoder(),
        runtime=RuntimeState(today_open=20.0, yesterday_close=19.0),
        decision_time="10:30",
        decision_index=0,
        price=20.0,
        dry_run=False,
    )

    assert entry["status"] == "heartbeat"
    assert len(client.patches) == 1


def test_dry_run_does_not_patch() -> None:
    client = MockClient(cash=1_000_000.0, qty=0)

    entry = handle_decision(
        client=client,
        strategy_id="strategy",
        cfg=load_config("experiments/production_2800.yaml"),
        agent=FixedAgent(1),
        encoder=FixedEncoder(),
        runtime=RuntimeState(today_open=20.0, yesterday_close=19.0),
        decision_time="10:30",
        decision_index=0,
        price=20.0,
        dry_run=True,
    )

    assert entry["target_qty"] == 49_500
    assert client.patches == []


def test_force_close_patches_zero_when_live() -> None:
    client = MockClient(cash=0.0, qty=49_500)

    entry = handle_force_close(
        client=client,
        strategy_id="strategy",
        cfg=load_config("experiments/production_2800.yaml"),
        price=26.52,
        dry_run=False,
        confirm_attempts=1,
        confirm_sleep_seconds=0.0,
    )

    assert entry["status"] == "force_close"
    assert entry["target_qty"] == 0
    assert entry["current_qty"] == 49_500
    assert entry["post_qty"] == 0
    assert len(client.patches) == 1
    assert client.patches[0]["holdings"]["holdings"][0]["stocks"][0]["quantity"] == 0


def test_force_close_logs_pending_when_broker_does_not_flatten() -> None:
    client = UnfilledMockClient(cash=0.0, qty=49_500)

    entry = handle_force_close(
        client=client,
        strategy_id="strategy",
        cfg=load_config("experiments/production_2800.yaml"),
        price=26.52,
        dry_run=False,
        confirm_attempts=1,
        confirm_sleep_seconds=0.0,
    )

    assert entry["status"] == "force_close_pending"
    assert entry["target_qty"] == 0
    assert entry["current_qty"] == 49_500
    assert entry["post_qty"] == 49_500


def test_production_force_close_is_before_market_close() -> None:
    cfg = load_config("experiments/production_2800.yaml")

    assert cfg.market.force_close_time == "15:55"


def test_safe_price_for_converts_api_timeout_to_error() -> None:
    price, error = _safe_price_for(
        "10:00",
        load_config("experiments/production_2800.yaml"),
        TimeoutPriceClient(),
        None,
    )

    assert price is None
    assert error == "TimeoutError: api read timed out"


def test_next_trading_open_skips_weekend_from_sunday() -> None:
    now = datetime(2026, 5, 17, 12, 0, tzinfo=HK_TZ)

    assert _next_trading_open(now) == datetime(2026, 5, 18, 9, 30, tzinfo=HK_TZ)


def test_next_trading_open_skips_weekend_from_friday_after_close() -> None:
    now = datetime(2026, 5, 15, 16, 10, tzinfo=HK_TZ)

    assert _next_trading_open(now) == datetime(2026, 5, 18, 9, 30, tzinfo=HK_TZ)
