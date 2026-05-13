from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantphemes_rl.agent.base import Agent


def state_coverage(agent: Agent) -> pd.DataFrame:
    """Return per-decision, per-state training visit counts."""
    visits = _visits(agent)
    rows = [
        {"decision_point": t, "state": state, "visits": int(visits[t, state].sum())}
        for t in range(visits.shape[0])
        for state in range(visits.shape[1])
    ]
    return pd.DataFrame(rows)


def coverage_summary(agent: Agent) -> dict[str, object]:
    """Return aggregate training coverage diagnostics."""
    visits = _visits(agent)
    state_action_visits = visits.reshape(-1)
    return {
        "pct_visited": float(np.count_nonzero(state_action_visits) / state_action_visits.size),
        "pct_well_visited": float(
            np.count_nonzero(state_action_visits > 100) / state_action_visits.size
        ),
        "unvisited_cells": [
            (int(t), int(state), int(action))
            for t, state, action in np.argwhere(visits == 0)
        ],
    }


def plot_coverage_heatmap(agent: Agent, path: Path) -> None:
    """Plot decision-point by state heatmap with log visit color."""
    visits = _visits(agent).sum(axis=2)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    image = ax.imshow(np.log1p(visits), aspect="auto", cmap="viridis")
    ax.set_xlabel("State")
    ax.set_ylabel("Decision point")
    ax.set_title("State Coverage")
    fig.colorbar(image, ax=ax, label="log(1 + visits)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _visits(agent: Agent) -> np.ndarray:
    visits = getattr(agent, "visits", None)
    if visits is None:
        msg = "Agent does not expose visits."
        raise ValueError(msg)
    return np.asarray(visits)
