from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

import quantphemes_rl  # noqa: F401
from quantphemes_rl.config import ExperimentConfig
from quantphemes_rl.data.base import group_by_day, validate_day
from quantphemes_rl.diagnostics.coverage import coverage_summary, state_coverage
from quantphemes_rl.diagnostics.signal import mutual_info_per_feature
from quantphemes_rl.registry import build


def generate_report(results_dir: Path) -> Path:
    """Render an HTML diagnostics report for a result directory."""
    cfg = ExperimentConfig.model_validate(
        yaml.safe_load((results_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    )
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    train_log = _read_csv(results_dir / "train_log.csv")
    eval_log = _read_csv(results_dir / "eval_log.csv")
    step_log = _read_csv(results_dir / "step_log.csv")
    agent = build("agent", cfg.agent.type, **cfg.agent.kwargs)
    q_state_path = results_dir / "q_state.pkl"
    if q_state_path.exists():
        agent.load(q_state_path)

    data_source = build("data_source", cfg.data.source, path=cfg.data.path)
    encoder = build("state_encoder", cfg.state.encoder, **cfg.state.kwargs)
    bars = data_source.load(cfg.data.symbol, cfg.data.start, cfg.data.end, "1h")
    required_times = [*cfg.market.decision_times, cfg.market.force_close_time]
    days = [
        day
        for day in group_by_day(bars, symbol=cfg.data.symbol)
        if validate_day(day, required_times)
    ]
    signal_table = mutual_info_per_feature(days, encoder, cfg.market.decision_times)
    cov_table = state_coverage(agent)
    cov_summary = coverage_summary(agent)

    context = {
        "config": cfg,
        "config_yaml": yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
        "metrics": metrics,
        "summary_rows": _summary_rows(metrics),
        "nav_curve_png": _plot_nav_curves(eval_log),
        "action_dist_png": _plot_action_distribution(step_log),
        "coverage_png": _plot_coverage(cov_table),
        "signal_rows": signal_table.to_dict(orient="records"),
        "coverage_summary": cov_summary,
        "fee_fraction": _fee_fraction(metrics),
        "reward_hist_png": _plot_reward_histogram(step_log),
        "oracle": metrics.get("oracle"),
        "train_log_rows": train_log.to_dict(orient="records"),
    }
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    html = env.get_template("report.html.j2").render(**context)
    output = results_dir / "report.html"
    output.write_text(html, encoding="utf-8")
    return output


def _summary_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "agent": "main",
            "final_nav": metrics["final_nav"],
            "return": metrics["return"],
            "sharpe": metrics.get("sharpe", 0.0),
            "num_trades": metrics["num_trades"],
            "total_fees": metrics["total_fees"],
            "alpha_vs_bh": metrics["alpha_vs_bh"],
        }
    ]
    for name, baseline in metrics.get("baselines", {}).items():
        rows.append(
            {
                "agent": name,
                "final_nav": baseline["final_nav"],
                "return": baseline["return"],
                "sharpe": baseline.get("sharpe", 0.0),
                "num_trades": baseline["num_trades"],
                "total_fees": baseline["total_fees"],
                "alpha_vs_bh": baseline.get("alpha_vs_bh", 0.0),
            }
        )
    if isinstance(metrics.get("oracle"), dict):
        oracle = metrics["oracle"]
        rows.append(
            {
                "agent": "oracle",
                "final_nav": oracle["final_nav"],
                "return": oracle["return"],
                "sharpe": 0.0,
                "num_trades": "",
                "total_fees": "",
                "alpha_vs_bh": "",
            }
        )
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _plot_nav_curves(eval_log: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not eval_log.empty:
        for agent, group in eval_log.groupby("agent"):
            ax.plot(group["day_index"], group["nav"], marker="o", label=agent)
        ax.legend()
    ax.set_title("NAV Curves")
    ax.set_xlabel("Evaluation day")
    ax.set_ylabel("NAV")
    return _fig_to_base64(fig)


def _plot_action_distribution(step_log: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not step_log.empty:
        main_steps = step_log[step_log["agent"] == "main"]
        counts = (
            main_steps.groupby(["decision_point", "action"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        counts.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Action Distribution by Decision Point")
    ax.set_xlabel("Decision point")
    ax.set_ylabel("Count")
    return _fig_to_base64(fig)


def _plot_coverage(cov_table: pd.DataFrame) -> str:
    pivot = cov_table.pivot(index="decision_point", columns="state", values="visits")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    image = ax.imshow(np_log1p(pivot.to_numpy()), aspect="auto", cmap="viridis")
    ax.set_title("State Coverage Heatmap")
    ax.set_xlabel("State")
    ax.set_ylabel("Decision point")
    fig.colorbar(image, ax=ax, label="log(1 + visits)")
    return _fig_to_base64(fig)


def _plot_reward_histogram(step_log: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not step_log.empty and "reward" in step_log:
        step_log["reward"].plot(kind="hist", bins=20, ax=ax)
    ax.set_title("Reward Histogram")
    ax.set_xlabel("Reward")
    return _fig_to_base64(fig)


def _fig_to_base64(fig: plt.Figure) -> str:
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _fee_fraction(metrics: dict[str, Any]) -> float:
    capital = float(metrics.get("capital", 1.0))
    gross = abs(metrics["final_nav"] - capital + metrics["total_fees"])
    if gross == 0.0:
        return 0.0
    return float(metrics["total_fees"] / gross)


def np_log1p(values: Any) -> Any:
    """Small indirection keeps numpy import local to the hot plot path."""
    import numpy as np

    return np.log1p(values)
