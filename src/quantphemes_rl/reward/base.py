from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RewardFunction(ABC):
    """Abstract reward function interface."""

    @abstractmethod
    def compute(self, v_t: float, v_t_plus_1: float, info: dict[str, Any]) -> float:
        """Compute a scalar reward from consecutive portfolio values."""
