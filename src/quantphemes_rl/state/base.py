from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketContext:
    """Market snapshot consumed by state encoders."""

    current_price: float
    today_open: float
    yesterday_close: float
    prev_decision_price: float
    decision_index: int
    decision_count: int
    recent_returns: list[float] | None = None


class StateEncoder(ABC):
    """Abstract state encoder interface."""

    @abstractmethod
    def encode(self, ctx: MarketContext) -> int:
        """Encode market context as a state index."""

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Return the number of possible encoded states."""
