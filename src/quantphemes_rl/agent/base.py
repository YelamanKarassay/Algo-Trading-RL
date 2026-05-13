from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transition:
    """Single transition used by agents for learning."""

    t: int
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class Agent(ABC):
    """Abstract trading agent interface."""

    @abstractmethod
    def act(self, state: int, training: bool) -> int:
        """Choose an action."""

    @abstractmethod
    def update(self, transition: Transition) -> None:
        """Update agent state from a transition."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist agent state."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load agent state."""

    @abstractmethod
    def decay_epsilon(self) -> None:
        """Apply one episode of exploration decay."""
