from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


class QuantphemesAPIError(Exception):
    """Raised when the Quantphemes API returns a non-2xx response."""


class QuantphemesClient:
    """Thin client for documented Quantphemes API endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.quantphemes.com",
        timeout: float = 20.0,
        max_retries: int = 2,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def __repr__(self) -> str:
        return f"QuantphemesClient(base_url={self.base_url!r})"

    def list_portfolios(self) -> dict[str, Any]:
        """List available portfolios."""
        return self._request("GET", "/api/v1/portfolio")

    def get_portfolio_strategies(self, portfolio_id: str) -> dict[str, Any]:
        """Return strategies for a portfolio."""
        return self._request("GET", f"/api/v1/portfolio/{portfolio_id}/strategy")

    def get_holdings(self, strategy_id: str) -> dict[str, Any]:
        """Return holdings for a strategy."""
        return self._request("GET", f"/api/v1/strategy/{strategy_id}/holding")

    def get_orders(self, strategy_id: str) -> dict[str, Any]:
        """Return orders for a strategy."""
        return self._request("GET", f"/api/v1/strategy/{strategy_id}/order")

    def get_strategy_quantities(self, strategy_id: str) -> dict[str, Any]:
        """Return strategy quantities."""
        return self._request("GET", f"/api/v1/strategy/{strategy_id}/holding/strategy-quantity")

    def create_holding(self, strategy_id: str, holdings: dict[str, Any]) -> dict[str, Any]:
        """Create holdings for a strategy."""
        return self._request("POST", f"/api/v1/strategy/{strategy_id}/holding", json=holdings)

    def update_holding(self, strategy_id: str, holdings: dict[str, Any]) -> dict[str, Any]:
        """Update holdings for a strategy."""
        return self._request("PATCH", f"/api/v1/strategy/{strategy_id}/holding", json=holdings)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
                break
            except (requests.ConnectionError, requests.Timeout):
                if attempt >= self.max_retries:
                    raise
                log.warning(
                    "Quantphemes API request failed; retrying",
                    extra={"method": method, "path": path, "attempt": attempt + 1},
                )
                time.sleep(self.retry_sleep_seconds)
        if response is None:
            msg = f"Quantphemes API {method} {path} did not return a response."
            raise QuantphemesAPIError(msg)
        if not 200 <= response.status_code < 300:
            msg = (
                f"Quantphemes API {method} {path} failed with status "
                f"{response.status_code}: {response.text}"
            )
            raise QuantphemesAPIError(msg)
        return response.json()
