from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Literal

import numpy as np

from quantphemes_rl.agent.base import Agent, Transition
from quantphemes_rl.registry import register

SAVE_VERSION = 1


@register("agent", "qtable")
class QTableAgent(Agent):
    """Tabular Q-learning agent with one table per decision point."""

    def __init__(
        self,
        num_decision_points: int,
        num_states: int,
        num_actions: int = 2,
        epsilon_start: float = 0.4,
        epsilon_decay_per_episode: float = 1.0,
        epsilon_min: float = 0.01,
        alpha_mode: Literal["per_cell", "one_over_d"] = "per_cell",
        optimistic_init: float = 0.0,
        tie_break: Literal["random", "long", "cash"] = "random",
        long_margin: float = 0.0,
    ) -> None:
        if tie_break not in {"random", "long", "cash"}:
            msg = f"Unsupported tie_break '{tie_break}'."
            raise ValueError(msg)
        if alpha_mode not in {"per_cell", "one_over_d"}:
            msg = f"Unsupported alpha_mode '{alpha_mode}'."
            raise ValueError(msg)
        self.num_decision_points = num_decision_points
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon_decay_per_episode = epsilon_decay_per_episode
        self.epsilon_min = epsilon_min
        self.alpha_mode = alpha_mode
        self.optimistic_init = optimistic_init
        self.tie_break = tie_break
        self.long_margin = long_margin
        self.tables = np.full(
            (num_decision_points, num_states, num_actions),
            optimistic_init,
            dtype=float,
        )
        self.visits = np.zeros((num_decision_points, num_states, num_actions), dtype=int)
        self.day_counter = 0
        self.epsilon = epsilon_start

    def act(self, state: int, training: bool) -> int:
        """Choose epsilon-greedy action using the first decision table."""
        return self.act_at(0, state, training)

    def act_at(self, t: int, state: int, training: bool) -> int:
        """Choose epsilon-greedy action for a specific decision table."""
        if training and random.random() < self.epsilon:
            return random.choice(list(range(self.num_actions)))
        return self._argmax_action(t, state)

    def update(self, transition: Transition) -> None:
        """Apply one Q-learning update."""
        q_value = self.tables[transition.t, transition.state, transition.action]
        self.visits[transition.t, transition.state, transition.action] += 1
        alpha = self._alpha(transition)
        bootstrap = 0.0
        if not transition.done and transition.t + 1 < self.num_decision_points:
            bootstrap = float(np.max(self.tables[transition.t + 1, transition.next_state]))
        target = transition.reward + bootstrap
        self.tables[transition.t, transition.state, transition.action] = q_value + alpha * (
            target - q_value
        )

    def save(self, path: Path) -> None:
        """Persist Q-table state to pickle with a version byte."""
        payload = {
            "tables": self.tables,
            "visits": self.visits,
            "day_counter": self.day_counter,
            "epsilon": self.epsilon,
            "config": self._config(),
        }
        with path.open("wb") as handle:
            handle.write(bytes([SAVE_VERSION]))
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        """Load Q-table state from pickle."""
        with path.open("rb") as handle:
            version = handle.read(1)
            if version != bytes([SAVE_VERSION]):
                msg = f"Unsupported QTableAgent save version {version!r}."
                raise ValueError(msg)
            payload = pickle.load(handle)
        self.tables = payload["tables"]
        self.visits = payload["visits"]
        self.day_counter = payload["day_counter"]
        self.epsilon = payload["epsilon"]
        config = payload["config"]
        self.num_decision_points = config["num_decision_points"]
        self.num_states = config["num_states"]
        self.num_actions = config["num_actions"]
        self.epsilon_decay_per_episode = config["epsilon_decay_per_episode"]
        self.epsilon_min = config["epsilon_min"]
        self.alpha_mode = config["alpha_mode"]
        self.optimistic_init = config["optimistic_init"]
        self.tie_break = config["tie_break"]
        self.long_margin = config.get("long_margin", 0.0)

    def decay_epsilon(self) -> None:
        """Apply one episode of epsilon decay."""
        self.day_counter += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_per_episode)

    def state_visit_summary(self) -> dict[str, int | float]:
        """Return compact visit-count diagnostics."""
        state_visits = self.visits.sum(axis=(0, 2))
        return {
            "total_visits": int(self.visits.sum()),
            "visited_states": int(np.count_nonzero(state_visits)),
            "num_states": int(self.num_states),
            "coverage": float(np.count_nonzero(state_visits) / self.num_states),
        }

    def _argmax_action(self, t: int, state: int) -> int:
        q_values = self.tables[t, state]
        if (
            self.num_actions == 2
            and self.long_margin > 0.0
            and q_values[1] + self.long_margin >= q_values[0]
        ):
            return 1
        max_value = np.max(q_values)
        candidates = [action for action, value in enumerate(q_values) if value == max_value]
        if len(candidates) == 1:
            return candidates[0]
        if self.tie_break == "random":
            return random.choice(candidates)
        if self.tie_break == "long":
            return 1
        return 0

    def _alpha(self, transition: Transition) -> float:
        if self.alpha_mode == "per_cell":
            visits = self.visits[transition.t, transition.state, transition.action]
            return 1.0 / visits
        return 1.0 / max(1, self.day_counter + 1)

    def _config(self) -> dict[str, int | float | str]:
        return {
            "num_decision_points": self.num_decision_points,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "epsilon_decay_per_episode": self.epsilon_decay_per_episode,
            "epsilon_min": self.epsilon_min,
            "alpha_mode": self.alpha_mode,
            "optimistic_init": self.optimistic_init,
            "tie_break": self.tie_break,
            "long_margin": self.long_margin,
        }
