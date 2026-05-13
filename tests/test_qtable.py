from __future__ import annotations

import random

import numpy as np
import pytest

from quantphemes_rl.agent.base import Transition
from quantphemes_rl.agent.qtable import QTableAgent


def test_bellman_update_with_alpha_one_produces_reward_plus_max_next() -> None:
    agent = QTableAgent(num_decision_points=2, num_states=3, epsilon_start=0.0)
    agent.tables[1, 2, 0] = 0.5
    agent.tables[1, 2, 1] = 1.5

    agent.update(Transition(t=0, state=1, action=0, reward=2.0, next_state=2, done=False))

    assert agent.tables[0, 1, 0] == 3.5
    assert agent.visits[0, 1, 0] == 1


def test_tie_break_random_samples_about_half_each() -> None:
    random.seed(7)
    agent = QTableAgent(num_decision_points=1, num_states=1, epsilon_start=0.0)

    actions = [agent.act(0, training=False) for _ in range(10_000)]
    long_rate = sum(actions) / len(actions)

    assert 0.47 < long_rate < 0.53


def test_default_tie_break_does_not_always_return_cash() -> None:
    random.seed(11)
    agent = QTableAgent(num_decision_points=1, num_states=1, epsilon_start=0.0)

    actions = {agent.act(0, training=False) for _ in range(200)}

    assert actions == {0, 1}


def test_epsilon_zero_always_argmax() -> None:
    agent = QTableAgent(num_decision_points=1, num_states=1, epsilon_start=0.0)
    agent.tables[0, 0, 1] = 1.0

    assert {agent.act(0, training=True) for _ in range(100)} == {1}


def test_epsilon_one_always_uniform() -> None:
    random.seed(19)
    agent = QTableAgent(num_decision_points=1, num_states=1, epsilon_start=1.0)
    agent.tables[0, 0, 1] = 100.0

    actions = [agent.act(0, training=True) for _ in range(10_000)]
    long_rate = sum(actions) / len(actions)

    assert 0.47 < long_rate < 0.53


def test_save_load_round_trips_tables_and_visits(tmp_path) -> None:
    path = tmp_path / "qtable.pkl"
    agent = QTableAgent(num_decision_points=2, num_states=3, epsilon_start=0.2)
    agent.update(Transition(t=0, state=1, action=1, reward=2.0, next_state=2, done=True))
    agent.tables[1, 2, 0] = 4.0
    agent.decay_epsilon()
    agent.save(path)

    loaded = QTableAgent(num_decision_points=1, num_states=1)
    loaded.load(path)

    np.testing.assert_array_equal(loaded.tables, agent.tables)
    np.testing.assert_array_equal(loaded.visits, agent.visits)
    assert loaded.day_counter == agent.day_counter
    assert loaded.epsilon == pytest.approx(agent.epsilon)
