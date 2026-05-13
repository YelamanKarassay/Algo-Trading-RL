from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def main(argv: list[str] | None = None) -> None:
    """Compare result folders from completed experiments."""
    parser = argparse.ArgumentParser(description="Compare Quantphemes RL experiment results.")
    parser.add_argument("result_dirs", nargs="+", type=Path, help="results/<experiment>/ folders")
    parser.add_argument("--csv", action="store_true", help="Write compare_<timestamp>.csv")
    parser.add_argument(
        "--plot", action="store_true", help="Render compare_<timestamp>.png NAV plot"
    )
    args = parser.parse_args(argv)

    frame = compare_results(args.result_dirs)
    print(frame.to_string(index=False))  # noqa: T201
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.csv:
        frame.to_csv(f"compare_{timestamp}.csv", index=False)
    if args.plot:
        plot_path = Path(f"compare_{timestamp}.png")
        plot_nav_curves(args.result_dirs, frame, plot_path)


def compare_results(result_dirs: list[Path]) -> pd.DataFrame:
    """Return a comparison table for result directories."""
    rows = [_row_for_result_dir(path) for path in result_dirs]
    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "asset",
            "interval",
            "agent",
            "final_nav",
            "return_pct",
            "sharpe",
            "alpha_vs_bh",
            "num_trades",
            "fees",
            "fees_pct_of_pnl",
        ],
    )


def plot_nav_curves(result_dirs: list[Path], frame: pd.DataFrame, path: Path) -> None:
    """Render side-by-side NAV curves or final NAV markers."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for result_dir in result_dirs:
        eval_path = result_dir / "eval_log.csv"
        if eval_path.exists() and eval_path.stat().st_size > 0:
            eval_log = pd.read_csv(eval_path)
            main = eval_log[eval_log["agent"] == "main"]
            if not main.empty:
                ax.plot(main["day_index"], main["nav"], marker="o", label=result_dir.name)
                plotted = True
    if not plotted:
        ax.bar(frame["name"], frame["final_nav"])
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Experiment NAV Comparison")
    ax.set_ylabel("NAV")
    ax.legend(loc="best") if plotted else None
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _row_for_result_dir(result_dir: Path) -> dict[str, Any]:
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((result_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    capital = float(metrics.get("capital", config["market"]["capital"]))
    pnl = float(metrics["final_nav"]) - capital
    fees = float(metrics.get("total_fees", 0.0))
    return {
        "name": config["name"],
        "asset": config["data"]["symbol"],
        "interval": _infer_interval(config),
        "agent": config["agent"]["type"],
        "final_nav": metrics["final_nav"],
        "return_pct": float(metrics["return"]) * 100,
        "sharpe": metrics.get("sharpe", 0.0),
        "alpha_vs_bh": metrics.get("alpha_vs_bh", 0.0),
        "num_trades": metrics.get("num_trades", 0),
        "fees": fees,
        "fees_pct_of_pnl": 0.0 if pnl == 0.0 else fees / abs(pnl) * 100,
    }


def _infer_interval(config: dict[str, Any]) -> str:
    path = str(config["data"].get("path", "")).lower()
    for candidate in ("30min", "5min", "eod", "1h"):
        if candidate in path:
            return candidate
    decision_count = len(config["market"].get("decision_times", []))
    if decision_count == 1:
        return "eod"
    if decision_count == 5:
        return "1h"
    if decision_count == 9:
        return "30min"
    return f"{decision_count}_decisions"


if __name__ == "__main__":
    main()
