"""Clients for Quantphemes endpoints not listed in the public documentation.

These endpoints may change or disappear without notice. Keep usage isolated here instead of
mixing undocumented API calls into the documented Quantphemes client.
"""

from __future__ import annotations

from typing import Any

from quantphemes_rl.api_client.quantphemes import QuantphemesClient


class UndocumentedQuantphemesClient(QuantphemesClient):
    """Thin client for currently useful but undocumented Quantphemes endpoints."""

    def get_last_price(self, symbol: str) -> dict[str, Any]:
        """Return the latest price payload for a stock."""
        return self._request("GET", "/api/v1/stocks/last-price", params={"symbols": symbol})

    def get_prices(self, symbol: str, start: str, end: str) -> dict[str, Any]:
        """Return historical price payload for a stock."""
        return self._request(
            "GET",
            "/api/v1/stocks/prices",
            params={"symbols": symbol, "start": start, "end": end},
        )

    def get_cash_and_asset(self, strategy_id: str) -> dict[str, Any]:
        """Return cash and asset payload for a strategy."""
        return self._request("GET", f"/api/v1/strategy/{strategy_id}/cash-and-asset")
