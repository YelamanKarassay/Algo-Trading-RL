from __future__ import annotations

import numpy as np
import pytest

from apps.federate import FederateError, merge_agents
from quantphemes_rl.agent.qtable import QTableAgent


def test_merge_visit_weighted_average() -> None:
    ours = _agent()
    theirs = _agent()
    ours.tables[0, 1, 1] = 2.0
    theirs.tables[0, 1, 1] = 10.0
    ours.visits[0, 1, 1] = 3
    theirs.visits[0, 1, 1] = 1

    merged, _ = merge_agents(ours, theirs)

    assert merged.tables[0, 1, 1] == 4.0
    assert merged.visits[0, 1, 1] == 4


def test_merge_zero_visits_both_sides() -> None:
    merged, _ = merge_agents(_agent(), _agent())

    assert merged.tables[0, 0, 0] == 0.0
    assert merged.visits[0, 0, 0] == 0


def test_merge_mismatched_shapes_raises() -> None:
    ours = _agent(num_states=3)
    theirs = _agent(num_states=4)

    with pytest.raises(FederateError, match="num_states"):
        merge_agents(ours, theirs)


def test_merge_one_sided() -> None:
    ours = _agent()
    theirs = _agent()
    ours.tables[0, 2, 1] = 7.0
    ours.visits[0, 2, 1] = 5

    merged, _ = merge_agents(ours, theirs)

    np.testing.assert_array_equal(merged.tables, ours.tables)
    np.testing.assert_array_equal(merged.visits, ours.visits)


def _agent(num_states: int = 3) -> QTableAgent:
    return QTableAgent(
        num_decision_points=2,
        num_states=num_states,
        num_actions=2,
        epsilon_start=0.2,
        epsilon_decay_per_episode=0.9,
        epsilon_min=0.01,
        alpha_mode="per_cell",
    )
