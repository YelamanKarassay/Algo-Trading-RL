'''
Train the rain the corrected 2800 30-min model: 
python -m apps.run_experiment experiments/exp_002_2800_30min.yaml
'''

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import quantphemes_rl  # noqa: F401
from quantphemes_rl.agent.base import Agent, Transition
from quantphemes_rl.config import ExperimentConfig, dump_config, load_config
from quantphemes_rl.data.base import DayData, group_by_day, validate_day
from quantphemes_rl.diagnostics.oracle import oracle_nav
from quantphemes_rl.diagnostics.report import generate_report
from quantphemes_rl.env.trading_env import TradingEnv
from quantphemes_rl.portfolio.portfolio import Portfolio
from quantphemes_rl.registry import build
from quantphemes_rl.reward.base import RewardFunction
from quantphemes_rl.state.base import StateEncoder

log = logging.getLogger(__name__)
RESULTS_ROOT = Path("results")


@dataclass(frozen=True)
class Window:
    """Walk-forward training/test day split."""

    training_days: list[DayData]
    test_days: list[DayData]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation summary for one agent."""

    name: str
    rows: list[dict[str, Any]]
    final_nav: float
    total_fees: float
    num_trades: int
    returns: list[float]
    step_rows: list[dict[str, Any]]


def main(argv: list[str] | None = None) -> None:
    """Run an experiment from a YAML config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    result_dir = run(args.path)
    print(f"completed {result_dir}")  # noqa: T201


def run(path: Path) -> Path:
    """Execute an experiment and return its result directory."""
    cfg = load_config(path)
    _set_seeds(cfg.seed)
    result_dir = _create_result_dir(cfg.name)
    dump_config(cfg, result_dir / "config.snapshot.yaml")
    (result_dir / "git_commit.txt").write_text(_git_commit_marker(), encoding="utf-8")

    data_source = build("data_source", cfg.data.source, path=cfg.data.path)
    encoder = build("state_encoder", cfg.state.encoder, **cfg.state.kwargs)
    reward_fn = build("reward_function", cfg.reward.function, **cfg.reward.kwargs)
    agent = build("agent", cfg.agent.type, **cfg.agent.kwargs)

    bars = data_source.load(cfg.data.symbol, cfg.data.start, cfg.data.end, "1h")
    required_times = [*cfg.market.decision_times, cfg.market.force_close_time]
    days = _filter_valid_days(group_by_day(bars, symbol=cfg.data.symbol), required_times)
    windows = _walk_forward(days, cfg)
    train_rows = _train(agent, encoder, reward_fn, windows, cfg)
    _write_csv(result_dir / "train_log.csv", train_rows)

    main_eval = _evaluate("main", agent, encoder, reward_fn, windows[-1].test_days, cfg)
    baseline_evals = {
        name: _evaluate(
            name,
            _build_baseline_agent(name, cfg),
            encoder,
            reward_fn,
            windows[-1].test_days,
            cfg,
            reset_between_days=True,
        )
        for name in cfg.evaluation.baselines
    }
    _write_csv(result_dir / "eval_log.csv", _eval_rows(main_eval, baseline_evals))
    _write_csv(result_dir / "step_log.csv", _step_rows(main_eval, baseline_evals))

    metrics = _metrics(main_eval, baseline_evals, agent, cfg)
    if cfg.evaluation.compute_oracle:
        oracle_final_nav, oracle_per_day_navs = oracle_nav(
            windows[-1].test_days,
            cfg.market.decision_times,
            cfg.market.force_close_time,
            cfg.market.fee_bps_one_side / 10_000,
            float(cfg.market.capital),
            cfg.market.lot_size,
        )
        metrics["oracle"] = {
            "final_nav": oracle_final_nav,
            "return": oracle_final_nav / cfg.market.capital - 1.0,
            "per_day_navs": oracle_per_day_navs,
        }
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    agent.save(result_dir / "q_state.pkl")
    generate_report(result_dir)
    _append_index(cfg, result_dir, metrics)
    return result_dir


def _build_baseline_agent(name: str, cfg: ExperimentConfig) -> Agent:
    max_action = int(cfg.agent.kwargs.get("num_actions", 2)) - 1
    if name in {"always_long", "buy_and_hold"} and max_action > 1:
        return build("agent", name, action=max_action)
    return build("agent", name)


def _filter_valid_days(days: list[DayData], required_times: list[str]) -> list[DayData]:
    valid_days = [day for day in days if validate_day(day, required_times, warn=False)]
    rejected = len(days) - len(valid_days)
    if rejected:
        log.warning(
            "Rejected days with missing required bars",
            extra={"rejected_days": rejected, "total_days": len(days)},
        )
    return valid_days


def _train(
    agent: Agent,
    encoder: StateEncoder,
    reward_fn: RewardFunction,
    windows: list[Window],
    cfg: ExperimentConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        for episode in range(cfg.training.episodes_per_window):
            training_days = list(window.training_days)
            random.shuffle(training_days)
            day_navs: list[float] = []
            num_trades = 0
            total_fees = 0.0
            for day in training_days:
                result = _run_day(agent, encoder, reward_fn, day, cfg, training=True)
                day_navs.append(result["final_nav"])
                num_trades += result["num_trades"]
                total_fees += result["total_fees"]
            agent.decay_epsilon()
            rows.append(
                {
                    "window": window_index,
                    "episode": episode,
                    "nav": _mean(day_navs),
                    "num_trades": num_trades,
                    "total_fees": total_fees,
                    "epsilon": getattr(agent, "epsilon", 0.0),
                    "pct_states_visited": _pct_states_visited(agent),
                }
            )
    return rows


def _evaluate(
    name: str,
    agent: Agent,
    encoder: StateEncoder,
    reward_fn: RewardFunction,
    days: list[DayData],
    cfg: ExperimentConfig,
    reset_between_days: bool = False,
) -> EvaluationResult:
    rows: list[dict[str, Any]] = []
    returns: list[float] = []
    total_fees = 0.0
    num_trades = 0
    final_nav = float(cfg.market.capital)
    step_rows: list[dict[str, Any]] = []
    for day_index, day in enumerate(days):
        result = _run_day(agent, encoder, reward_fn, day, cfg, training=False)
        final_nav = result["final_nav"]
        total_fees += result["total_fees"]
        num_trades += result["num_trades"]
        returns.append(final_nav / cfg.market.capital - 1.0)
        rows.append(
            {
                "agent": name,
                "day_index": day_index,
                "date": day.date.isoformat(),
                "nav": final_nav,
                "return": returns[-1],
                "num_trades": result["num_trades"],
                "total_fees": result["total_fees"],
            }
        )
        for step_row in result["step_rows"]:
            step_rows.append(
                {
                    "agent": name,
                    "day_index": day_index,
                    "date": day.date.isoformat(),
                    **step_row,
                }
            )
        if reset_between_days:
            agent.decay_epsilon()
    return EvaluationResult(name, rows, final_nav, total_fees, num_trades, returns, step_rows)


def _run_day(
    agent: Agent,
    encoder: StateEncoder,
    reward_fn: RewardFunction,
    day: DayData,
    cfg: ExperimentConfig,
    training: bool,
) -> dict[str, Any]:
    portfolio = Portfolio(
        cash=float(cfg.market.capital),
        shares=0,
        position=0,
        total_fees=0.0,
        fee_rate_one_side=cfg.market.fee_bps_one_side / 10_000,
        lot_size=cfg.market.lot_size,
        max_position=max(1, int(cfg.agent.kwargs.get("num_actions", 2)) - 1),
    )
    env = TradingEnv(
        day=day,
        encoder=encoder,
        reward_fn=reward_fn,
        portfolio=portfolio,
        decision_times=cfg.market.decision_times,
        force_close_time=cfg.market.force_close_time,
    )
    state = env.reset()
    done = False
    t = 0
    num_trades = 0
    step_rows: list[dict[str, Any]] = []
    last_info: dict[str, Any] = {"v_t_plus_1": portfolio.market_value(day.close_price)}
    while not done:
        action = _act(agent, t, state, training)
        next_state, reward, done, info = env.step(action)
        if info["fee"] > 0.0:
            num_trades += int(info["action_fee"] > 0.0) + int(info["force_close_fee"] > 0.0)
        if training:
            agent.update(
                Transition(
                    t=t,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )
        step_rows.append(
            {
                "decision_point": t,
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "fee": info["fee"],
                "action_fee": info["action_fee"],
                "force_close_fee": info["force_close_fee"],
                "nav": info["v_t_plus_1"],
                "position": info["position"],
            }
        )
        state = next_state
        last_info = info
        t += 1
    return {
        "final_nav": last_info["v_t_plus_1"],
        "num_trades": num_trades,
        "total_fees": portfolio.total_fees,
        "step_rows": step_rows,
    }


def _act(agent: Agent, t: int, state: int, training: bool) -> int:
    act_at = getattr(agent, "act_at", None)
    if callable(act_at):
        return int(act_at(t, state, training))
    return agent.act(state, training)


def _walk_forward(days: list[DayData], cfg: ExperimentConfig) -> list[Window]:
    initial_train_days = cfg.training.walk_forward["initial_train_days"]
    test_window_days = cfg.training.walk_forward["test_window_days"]
    if len(days) <= initial_train_days:
        initial_train_days = max(1, len(days) - 1)
    windows: list[Window] = []
    start = initial_train_days
    while start < len(days):
        end = min(len(days), start + test_window_days)
        windows.append(Window(training_days=days[:start], test_days=days[start:end]))
        start = end
    if not windows:
        msg = "Not enough days to create a train/test window."
        raise ValueError(msg)
    return windows


def _metrics(
    main_eval: EvaluationResult,
    baseline_evals: dict[str, EvaluationResult],
    agent: Agent,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    bnh = baseline_evals.get("buy_and_hold") or baseline_evals.get("always_long")
    bnh_return = 0.0 if bnh is None else bnh.final_nav / cfg.market.capital - 1.0
    main_return = main_eval.final_nav / cfg.market.capital - 1.0
    return {
        "capital": cfg.market.capital,
        "final_nav": main_eval.final_nav,
        "return": main_return,
        "sharpe": _sharpe(main_eval.returns),
        "sharpe_vs_bh": _sharpe(main_eval.returns) - (0.0 if bnh is None else _sharpe(bnh.returns)),
        "alpha_vs_bh": main_return - bnh_return,
        "num_trades": main_eval.num_trades,
        "total_fees": main_eval.total_fees,
        "pct_states_visited": _pct_states_visited(agent),
        "baselines": {
            name: {
                "final_nav": result.final_nav,
                "return": result.final_nav / cfg.market.capital - 1.0,
                "sharpe": _sharpe(result.returns),
                "alpha_vs_bh": result.final_nav / cfg.market.capital - 1.0 - bnh_return,
                "num_trades": result.num_trades,
                "total_fees": result.total_fees,
            }
            for name, result in baseline_evals.items()
        },
    }


def _eval_rows(
    main_eval: EvaluationResult, baseline_evals: dict[str, EvaluationResult]
) -> list[dict[str, Any]]:
    rows = list(main_eval.rows)
    for result in baseline_evals.values():
        rows.extend(result.rows)
    return rows


def _step_rows(
    main_eval: EvaluationResult, baseline_evals: dict[str, EvaluationResult]
) -> list[dict[str, Any]]:
    rows = list(main_eval.step_rows)
    for result in baseline_evals.values():
        rows.extend(result.step_rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_index(cfg: ExperimentConfig, result_dir: Path, metrics: dict[str, Any]) -> None:
    RESULTS_ROOT.mkdir(exist_ok=True)
    index_path = RESULTS_ROOT / "_index.csv"
    row = {
        "name": cfg.name,
        "result_dir": str(result_dir),
        "final_nav": metrics["final_nav"],
        "return": metrics["return"],
        "alpha_vs_bh": metrics["alpha_vs_bh"],
        "num_trades": metrics["num_trades"],
        "total_fees": metrics["total_fees"],
        "pct_states_visited": metrics["pct_states_visited"],
    }
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _create_result_dir(name: str) -> Path:
    RESULTS_ROOT.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    path = RESULTS_ROOT / f"{name}_{timestamp}"
    path.mkdir()
    return path


def _git_commit_marker() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            return f"uncommitted-{timestamp}"
        return head
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"uncommitted-{timestamp}"


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _pct_states_visited(agent: Agent) -> float:
    summary = getattr(agent, "state_visit_summary", None)
    if callable(summary):
        return float(summary()["coverage"])
    return 0.0


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


if __name__ == "__main__":
    main()
