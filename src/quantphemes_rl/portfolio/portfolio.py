from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Portfolio:
    """Cash, shares, and fee accounting for discrete intraday positions."""

    cash: float
    shares: int
    position: int
    total_fees: float
    fee_rate_one_side: float = 0.002
    lot_size: int = 500
    max_position: int = 1

    def market_value(self, price: float) -> float:
        """Return cash plus marked-to-market shares."""
        return self.cash + self.shares * price

    def execute(self, target_position: int, price: float) -> float:
        """Move to the target discrete position and return the fee paid."""
        if target_position < 0 or target_position > self.max_position:
            msg = (
                f"target_position must be between 0 and {self.max_position}, "
                f"got {target_position}."
            )
            raise ValueError(msg)
        if price <= 0:
            msg = f"price must be positive, got {price}."
            raise ValueError(msg)
        if target_position == self.position:
            return 0.0
        if self.max_position == 1:
            if target_position == 1:
                return self._buy_max_affordable_lots(price)
            return self._sell_all(price)
        return self._rebalance_to_position(target_position, price)

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

    def _rebalance_to_position(self, target_position: int, price: float) -> float:
        if target_position == 0:
            return self._sell_all(price)
        target_shares = self._target_shares_for_fraction(target_position / self.max_position, price)
        if target_shares == self.shares:
            self.position = target_position
            return 0.0
        if target_shares > self.shares:
            return self._buy_shares(target_shares - self.shares, target_position, price)
        return self._sell_shares(self.shares - target_shares, target_position, price)

    def _target_shares_for_fraction(self, fraction: float, price: float) -> int:
        nav = self.market_value(price)
        target_notional = nav * fraction
        return math.floor(target_notional / (self.lot_size * price) + 1e-12) * self.lot_size

    def _buy_shares(self, shares_to_buy: int, target_position: int, price: float) -> float:
        shares_to_buy = max(0, shares_to_buy)
        while shares_to_buy > 0:
            notional = shares_to_buy * price
            fee = notional * self.fee_rate_one_side
            total_cost = notional + fee
            if total_cost <= self.cash + 1e-9:
                self.cash -= total_cost
                self.shares += shares_to_buy
                self.position = target_position
                self.total_fees += fee
                return fee
            shares_to_buy -= self.lot_size
        self.position = target_position if self.shares > 0 else 0
        return 0.0

    def _sell_shares(self, shares_to_sell: int, target_position: int, price: float) -> float:
        shares_to_sell = min(self.shares, max(0, shares_to_sell))
        if shares_to_sell <= 0:
            self.position = target_position
            return 0.0
        notional = shares_to_sell * price
        fee = notional * self.fee_rate_one_side
        self.cash += notional - fee
        self.shares -= shares_to_sell
        self.position = target_position if self.shares > 0 else 0
        self.total_fees += fee
        return fee
