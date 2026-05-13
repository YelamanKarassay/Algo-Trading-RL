from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Cash, shares, and fee accounting for binary intraday positions."""

    cash: float
    shares: int
    position: Literal[0, 1]
    total_fees: float
    fee_rate_one_side: float = 0.002
    lot_size: int = 500

    def market_value(self, price: float) -> float:
        """Return cash plus marked-to-market shares."""
        return self.cash + self.shares * price

    def execute(self, target_position: int, price: float) -> float:
        """Move to the target binary position and return the fee paid."""
        if target_position not in {0, 1}:
            msg = f"target_position must be 0 or 1, got {target_position}."
            raise ValueError(msg)
        if price <= 0:
            msg = f"price must be positive, got {price}."
            raise ValueError(msg)
        if target_position == self.position:
            return 0.0
        if target_position == 1:
            return self._buy_max_affordable_lots(price)
        return self._sell_all(price)

    def start_new_day(self) -> None:
        """Reset intraday position flag while preserving cash."""
        if self.shares > 0:
            log.warning(
                "Starting new day while shares remain in portfolio",
                extra={"shares": self.shares, "cash": self.cash},
            )
        self.position = 0

    def _buy_max_affordable_lots(self, price: float) -> float:
        lot_notional = self.lot_size * price
        lot_cost_with_fee = lot_notional * (1.0 + self.fee_rate_one_side)
        lots = math.floor(self.cash / lot_cost_with_fee + 1e-12)
        shares_to_buy = lots * self.lot_size
        if shares_to_buy <= 0:
            return 0.0

        notional = shares_to_buy * price
        fee = notional * self.fee_rate_one_side
        total_cost = notional + fee
        if total_cost > self.cash:
            msg = "Calculated buy would make cash negative."
            raise RuntimeError(msg)

        self.cash -= total_cost
        self.shares += shares_to_buy
        self.position = 1
        self.total_fees += fee
        return fee

    def _sell_all(self, price: float) -> float:
        if self.shares <= 0:
            self.position = 0
            return 0.0

        notional = self.shares * price
        fee = notional * self.fee_rate_one_side
        self.cash += notional - fee
        self.shares = 0
        self.position = 0
        self.total_fees += fee
        return fee
