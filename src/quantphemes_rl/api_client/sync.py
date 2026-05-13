from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from quantphemes_rl.api_client.quantphemes import QuantphemesClient


def get_position_state(client: QuantphemesClient, strategy_id: str) -> tuple[float, int]:
    """Return broker cash and current quantity for a strategy."""
    cash_payload = client.get_cash_and_asset(strategy_id)
    data = cash_payload.get("data", cash_payload)
    cash = float(_first(data, ("cash", "available_cash", "availableCash"), 0.0))
    qty = 0
    for asset in data.get("assets", []):
        qty += int(_first(asset, ("quantity", "qty", "shares"), 0))
    return cash, qty


def patch_target(
    client: QuantphemesClient, strategy_id: str, symbol: str, target_qty: int
) -> dict[str, Any]:
    """Patch broker holdings to the requested target quantity."""
    payload = {
        "holdings": [
            {
                "effective_datetime": datetime.now(UTC).isoformat(),
                "stocks": [{"symbol": symbol, "quantity": target_qty}],
            }
        ]
    }
    return client.update_holding(strategy_id, payload)


def _first(payload: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default
