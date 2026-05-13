from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantphemes_rl.agent.qtable import QTableAgent


class FederateError(Exception):
    """Raised when two Q-table agents are incompatible."""


def main(argv: list[str] | None = None) -> None:
    """Merge two Q-table files using visit-weighted averaging."""
    parser = argparse.ArgumentParser(description="Federate two QTableAgent q_state.pkl files.")
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--theirs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    ours = load_agent(args.ours)
    theirs = load_agent(args.theirs)
    merged, report = merge_agents(ours, theirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(args.out)
    report_path = args.out.with_name(f"federate_report_{_timestamp()}.txt")
    report_path.write_text(report, encoding="utf-8")
    print(f"merged={args.out} report={report_path}")  # noqa: T201


def load_agent(path: Path) -> QTableAgent:
    """Load a QTableAgent from disk."""
    agent = QTableAgent(num_decision_points=1, num_states=1)
    agent.load(path)
    return agent


def merge_agents(ours: QTableAgent, theirs: QTableAgent) -> tuple[QTableAgent, str]:
    """Return visit-weighted merged agent plus report text."""
    _check_compatible(ours, theirs)
    merged = QTableAgent(
        num_decision_points=ours.num_decision_points,
        num_states=ours.num_states,
        num_actions=ours.num_actions,
        epsilon_start=ours.epsilon,
        epsilon_decay_per_episode=ours.epsilon_decay_per_episode,
        epsilon_min=ours.epsilon_min,
        alpha_mode=ours.alpha_mode,
        optimistic_init=0.0,
        tie_break=ours.tie_break,
    )
    n1 = ours.visits
    n2 = theirs.visits
    denominator = n1 + n2
    numerator = n1 * ours.tables + n2 * theirs.tables
    merged.tables = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(ours.tables, dtype=float),
        where=denominator != 0,
    )
    merged.visits = denominator
    merged.day_counter = max(ours.day_counter, theirs.day_counter)
    merged.epsilon = min(ours.epsilon, theirs.epsilon)
    return merged, merge_report(ours, theirs, merged)


def merge_report(ours: QTableAgent, theirs: QTableAgent, merged: QTableAgent) -> str:
    """Build a human-readable federated merge report."""
    newly_learned = np.argwhere((ours.visits == 0) & (theirs.visits > 0))
    base = np.maximum(np.abs(ours.tables), 1e-12)
    changed = np.argwhere(np.abs(merged.tables - ours.tables) / base > 0.10)
    max_abs_change = float(np.max(np.abs(merged.tables - ours.tables)))
    return "\n".join(
        [
            "Federated Q-table merge report",
            f"Generated UTC: {datetime.now(UTC).isoformat()}",
            f"Newly learned cells: {len(newly_learned)}",
            _format_cells("newly_learned", newly_learned),
            f"Cells changed >10% from ours: {len(changed)}",
            _format_cells("changed_gt_10pct", changed),
            f"Max absolute Q change: {max_abs_change:.12g}",
            "Pre-merge coordination checklist:",
            "- Reward function aligned (log_return on both sides)",
            "- Fee model identical",
            "- State encoding identical (three_feature with same thresholds)",
            "- Decision times identical",
        ]
    )


def _check_compatible(ours: QTableAgent, theirs: QTableAgent) -> None:
    checks = {
        "num_decision_points": (ours.num_decision_points, theirs.num_decision_points),
        "num_states": (ours.num_states, theirs.num_states),
        "num_actions": (ours.num_actions, theirs.num_actions),
        "alpha_mode": (ours.alpha_mode, theirs.alpha_mode),
        "epsilon_min": (ours.epsilon_min, theirs.epsilon_min),
        "epsilon_decay_per_episode": (
            ours.epsilon_decay_per_episode,
            theirs.epsilon_decay_per_episode,
        ),
    }
    mismatches = [
        f"{name}: ours={left!r} theirs={right!r}"
        for name, (left, right) in checks.items()
        if left != right
    ]
    if mismatches:
        raise FederateError("Incompatible Q-table agents:\n" + "\n".join(mismatches))


def _format_cells(label: str, cells: np.ndarray) -> str:
    preview = [tuple(int(value) for value in row) for row in cells[:20]]
    suffix = "" if len(cells) <= 20 else f" ... (+{len(cells) - 20} more)"
    return f"{label}: {preview}{suffix}"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
