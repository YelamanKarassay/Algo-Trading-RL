from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from apps.research_report import generate_research_report
from apps.run_experiment import run

ASSETS = {
    "2800.HK": 1.0,
    "2828.HK": 1.0,
    "7299.HK": 1.8,
    "7568.HK": 2.6,
    "7226.HK": 1.8,
}
INTERVALS = {
    "1h": ("M60", 1.0),
    "30m": ("M30", 1 / 1.4),
    "15m": ("M15", 1 / 1.8),
    "5m": ("M5", 1 / 2.5),
    "1m": ("M1", 1 / 4.0),
}
REWARDS: dict[str, dict[str, Any]] = {
    "log_return": {},
    "log_return_long_bias": {"beta": 0.00003},
    "fee_aware": {"fee_penalty": 1.0},
    "sparse_liquidation": {},
    "drawdown_penalty": {"drawdown_penalty": 1.0},
}
ENCODERS = ("three_feature", "with_rsi", "with_volatility", "with_volume", "zscore_bins")
ACTION_SPACES = {"binary": 2, "ternary": 3}
BASE_THRESHOLDS = {
    "delta_close": 0.015,
    "delta_open": 0.010,
    "delta_prev": 0.008,
}


@dataclass(frozen=True)
class ResearchSpec:
    asset: str
    interval: str
    reward: str
    encoder: str
    action_space: str


def main(argv: list[str] | None = None) -> None:
    """Download data, generate configs, run experiments, or report a research sweep."""
    parser = argparse.ArgumentParser(description="Manage the ETF research sweep.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--start", default="2018-01-01")
    download.add_argument("--end")
    download.add_argument("--sleep-seconds", default="1.05")
    download.add_argument("--interval", choices=INTERVALS, action="append")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--out-dir", type=Path, default=Path("experiments/research"))
    generate.add_argument("--baseline", type=Path, default=Path("experiments/_baseline.yaml"))
    generate.add_argument("--preset", choices=["core", "full"], default="core")

    runner = subparsers.add_parser("run")
    runner.add_argument("--config-dir", type=Path, default=Path("experiments/research"))
    runner.add_argument("--pattern", default="*.yaml")
    runner.add_argument("--limit", type=int)
    runner.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs whose experiment name already appears under results/.",
    )

    report = subparsers.add_parser("report")
    report.add_argument("--glob", default="results/research_*")
    report.add_argument("--out", type=Path, default=Path("results/research_report.html"))

    args = parser.parse_args(argv)
    if args.command == "download":
        _download(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "report":
        result_dirs = sorted(Path().glob(args.glob))
        generate_research_report(result_dirs, args.out)


def _download(args: argparse.Namespace) -> None:
    intervals = args.interval or list(INTERVALS)
    for interval in intervals:
        timespan = INTERVALS[interval][0]
        cmd = [
            sys.executable,
            "-m",
            "apps.download_webull_data",
            "--universe",
            "--timespan",
            timespan,
            "--start",
            args.start,
            "--sleep-seconds",
            args.sleep_seconds,
        ]
        if args.end:
            cmd.extend(["--end", args.end])
        subprocess.run(cmd, check=True)


def _generate(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = _specs(args.preset)
    inherits = os.path.relpath(args.baseline.resolve(), args.out_dir.resolve())
    manifest = []
    for spec in specs:
        payload = _config(spec, inherits=inherits)
        path = args.out_dir / f"{payload['name']}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        manifest.append({"name": payload["name"], "path": str(path)})
    (args.out_dir / "_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    print(f"generated {len(specs)} configs in {args.out_dir}")  # noqa: T201


def _run(args: argparse.Namespace) -> None:
    paths = sorted(
        path for path in args.config_dir.glob(args.pattern) if not path.name.startswith("_")
    )
    completed = _completed_experiment_names() if args.skip_existing else set()
    if completed:
        paths = [path for path in paths if path.stem not in completed]
    if args.limit is not None:
        paths = paths[: args.limit]
    for path in paths:
        print(f"running {path}")  # noqa: T201
        run(path)


def _completed_experiment_names(results_root: Path = Path("results")) -> set[str]:
    names: set[str] = set()
    if not results_root.exists():
        return names
    for metrics_path in results_root.glob("*/metrics.json"):
        config_path = metrics_path.parent / "config.snapshot.yaml"
        if not config_path.exists():
            continue
        try:
            names.add(str(yaml.safe_load(config_path.read_text(encoding="utf-8"))["name"]))
        except (OSError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError):
            continue
    return names


def _specs(preset: str) -> list[ResearchSpec]:
    intervals = list(INTERVALS)
    rewards = list(REWARDS)
    encoders = list(ENCODERS)
    action_spaces = list(ACTION_SPACES)
    if preset == "core":
        intervals = ["30m", "15m", "5m"]
        rewards = ["log_return", "log_return_long_bias", "fee_aware"]
        encoders = ["three_feature", "with_rsi", "with_volatility", "zscore_bins"]
        action_spaces = ["binary", "ternary"]
    return [
        ResearchSpec(asset, interval, reward, encoder, action_space)
        for asset in ASSETS
        for interval in intervals
        for reward in rewards
        for encoder in encoders
        for action_space in action_spaces
    ]


def _config(spec: ResearchSpec, inherits: str = "../_baseline.yaml") -> dict[str, Any]:
    asset_slug = spec.asset.removesuffix(".HK").lower()
    timespan, interval_multiplier = INTERVALS[spec.interval]
    num_actions = ACTION_SPACES[spec.action_space]
    thresholds = _thresholds(ASSETS[spec.asset] * interval_multiplier)
    decision_times = _decision_times(spec.interval)
    training = _training_params(spec.interval)
    num_states = 81 if spec.encoder in {"with_rsi", "with_volatility", "with_volume"} else 27
    name = (
        f"research_{asset_slug}_{spec.interval}_{spec.reward}_"
        f"{spec.encoder}_{spec.action_space}"
    )
    data_path = f"data/raw/webull_{spec.asset.replace('.', '_').lower()}_{timespan.lower()}.csv"
    return {
        "inherits": inherits,
        "name": name,
        "description": (
            f"Research sweep: {spec.asset} {spec.interval}, {spec.reward}, "
            f"{spec.encoder}, {spec.action_space} actions."
        ),
        "data": {
            "path": data_path,
            "symbol": spec.asset,
            "start": "2018-01-01",
            "end": "2026-05-17",
        },
        "market": {
            "decision_times": decision_times,
            "force_close_time": "16:00",
        },
        "state": {
            "encoder": spec.encoder,
            "kwargs": {"thresholds": thresholds},
        },
        "reward": {
            "function": spec.reward,
            "kwargs": REWARDS[spec.reward],
        },
        "agent": {
            "kwargs": {
                "num_decision_points": len(decision_times),
                "num_states": num_states,
                "num_actions": num_actions,
                "epsilon_start": 0.5,
                "epsilon_decay_per_episode": 0.98,
                "epsilon_min": 0.03,
                "alpha_mode": "per_cell",
                "optimistic_init": 0.0,
                "tie_break": "random",
            },
        },
        "training": {
            "walk_forward": {
                "initial_train_days": training["initial_train_days"],
                "test_window_days": training["test_window_days"],
            },
            "episodes_per_window": training["episodes_per_window"],
        },
        "evaluation": {
            "baselines": ["random_binary", "always_long", "buy_and_hold"],
            "compute_oracle": True,
            "feature_signal_check": True,
        },
    }


def _thresholds(multiplier: float) -> dict[str, list[float]]:
    return {
        key: [-round(value * multiplier, 10), round(value * multiplier, 10)]
        for key, value in BASE_THRESHOLDS.items()
    }


def _decision_times(interval: str) -> list[str]:
    if interval == "1h":
        return ["10:30", "11:30", "14:00", "15:00"]
    step = {"30m": 30, "15m": 15, "5m": 5, "1m": 1}[interval]
    return [
        *_time_grid("09:30", "11:30", step),
        *_time_grid("13:00", "15:30", step),
    ]


def _time_grid(start: str, end: str, step_minutes: int) -> list[str]:
    start_hour, start_minute = [int(part) for part in start.split(":")]
    end_hour, end_minute = [int(part) for part in end.split(":")]
    current = start_hour * 60 + start_minute + step_minutes
    end_value = end_hour * 60 + end_minute
    times = []
    while current <= end_value:
        times.append(f"{current // 60:02d}:{current % 60:02d}")
        current += step_minutes
    return times


def _training_params(interval: str) -> dict[str, int]:
    if interval == "1m":
        return {"initial_train_days": 60, "test_window_days": 20, "episodes_per_window": 2}
    if interval == "5m":
        return {"initial_train_days": 300, "test_window_days": 60, "episodes_per_window": 3}
    return {"initial_train_days": 500, "test_window_days": 125, "episodes_per_window": 5}


if __name__ == "__main__":
    main()
