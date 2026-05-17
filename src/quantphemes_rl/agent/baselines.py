from __future__ import annotations

import random
from pathlib import Path

from quantphemes_rl.agent.base import Agent, Transition
from quantphemes_rl.registry import register


class _NoOpAgent(Agent):
    def update(self, transition: Transition) -> None:
        """Ignore transitions."""
        del transition

    def save(self, path: Path) -> None:
        """No-op persistence for stateless agents."""
        del path

    def load(self, path: Path) -> None:
        """No-op load for stateless agents."""
        del path

    def decay_epsilon(self) -> None:
        """No-op exploration decay."""


@register("agent", "random_binary")
class RandomBinary(_NoOpAgent):
    """Uniform random binary-action baseline."""

    def act(self, state: int, training: bool) -> int:
        """Return a random binary action."""
        del state, training
        return random.choice([0, 1])


@register("agent", "always_long")
class AlwaysLong(_NoOpAgent):
    """Always fully invested baseline."""

    def __init__(self, action: int = 1) -> None:
        self.action = action

    def act(self, state: int, training: bool) -> int:
        """Return long action."""
        del state, training
        return self.action


@register("agent", "always_cash")
class AlwaysCash(_NoOpAgent):
    """Always cash baseline."""

    def act(self, state: int, training: bool) -> int:
        """Return cash action."""
        del state, training
        return 0


@register("agent", "buy_and_hold")
class BuyAndHold(_NoOpAgent):
    """Buy at the first decision and maintain previous action afterwards."""

    def __init__(self, action: int = 1) -> None:
        self.action = action
        self.has_acted_today = False
        self.previous_action = 0

    def act(self, state: int, training: bool) -> int:
        """Return long on first decision, then keep the previous action."""
        del state, training
        if not self.has_acted_today:
            self.previous_action = self.action
            self.has_acted_today = True
        return self.previous_action

    def decay_epsilon(self) -> None:
        """Reset day-local action memory."""
        self.has_acted_today = False
        self.previous_action = 0
