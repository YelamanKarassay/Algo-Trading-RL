from __future__ import annotations

from pathlib import Path

from apps.bot import RuntimeState, load_runtime_state, save_runtime_state


def test_round_trip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "runtime_state.json"
    state = RuntimeState(
        today_date="2026-05-13",
        today_open=20.0,
        yesterday_close=19.5,
        last_action_at_decision={"10:30": 1},
    )

    save_runtime_state(path, state)
    loaded = load_runtime_state(path)

    assert loaded == state


def test_resume_after_crash(tmp_path: Path) -> None:
    path = tmp_path / "runtime_state.json"
    save_runtime_state(
        path,
        RuntimeState(
            today_date="2026-05-13",
            today_open=20.0,
            yesterday_close=19.5,
            last_action_at_decision={"10:30": 1, "11:30": 0},
        ),
    )

    loaded = load_runtime_state(path)

    assert loaded.yesterday_close == 19.5
    assert loaded.today_open == 20.0
    assert loaded.last_action_at_decision == {"10:30": 1, "11:30": 0}
