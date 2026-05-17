from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.io as pio
import yaml


def main(argv: list[str] | None = None) -> None:
    """Build an interactive HTML report for a batch of experiment results."""
    parser = argparse.ArgumentParser(description="Generate a research comparison report.")
    parser.add_argument("result_dirs", nargs="*", type=Path)
    parser.add_argument("--glob", default="results/research_*")
    parser.add_argument("--out", type=Path, default=Path("results/research_report.html"))
    args = parser.parse_args(argv)
    result_dirs = args.result_dirs or sorted(Path().glob(args.glob))
    generate_research_report(result_dirs, args.out)


def generate_research_report(result_dirs: list[Path], out: Path) -> Path:
    """Generate a Plotly HTML research report from result directories."""
    frame = _load_results(result_dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = _render_report(frame)
    out.write_text(html, encoding="utf-8")
    return out


def _load_results(result_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for result_dir in result_dirs:
        metrics_path = result_dir / "metrics.json"
        config_path = result_dir / "config.snapshot.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        row = _row(result_dir, config, metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def _row(result_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    capital = float(config["market"]["capital"])
    reward = config["reward"]["function"]
    encoder = config["state"]["encoder"]
    actions = int(config["agent"]["kwargs"].get("num_actions", 2))
    pnl = float(metrics["final_nav"]) - capital
    total_fees = float(metrics.get("total_fees", 0.0))
    return {
        "result_dir": str(result_dir),
        "name": config["name"],
        "asset": config["data"]["symbol"],
        "interval": _infer_interval(config),
        "reward": reward,
        "encoder": encoder,
        "actions": actions,
        "final_nav": float(metrics["final_nav"]),
        "return_pct": float(metrics["return"]) * 100.0,
        "alpha_vs_bh_pct": float(metrics.get("alpha_vs_bh", 0.0)) * 100.0,
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "num_trades": int(metrics.get("num_trades", 0)),
        "total_fees": total_fees,
        "fee_pct_capital": total_fees / capital * 100.0,
        "fees_pct_abs_pnl": 0.0 if pnl == 0.0 else total_fees / abs(pnl) * 100.0,
        "coverage_pct": float(metrics.get("pct_states_visited", 0.0)) * 100.0,
    }


def _render_report(frame: pd.DataFrame) -> str:
    if frame.empty:
        body = "<p>No completed research result folders were found.</p>"
        summary = ""
    else:
        best = frame.sort_values("alpha_vs_bh_pct", ascending=False).head(20)
        all_results = frame.sort_values("alpha_vs_bh_pct", ascending=False)
        nav_frame = _load_nav_curves(all_results.head(12)["result_dir"].tolist())
        best_row = all_results.iloc[0]
        best_alpha = f"{best_row['alpha_vs_bh_pct']:.3f}%"
        median_fee = f"{frame['fee_pct_capital'].median():.3f}%"
        summary = f"""
        <div class="kpi-grid">
          <div class="kpi"><span>Completed Runs</span><strong>{len(frame):,}</strong></div>
          <div class="kpi"><span>Best Candidate</span><strong>{best_row["name"]}</strong></div>
          <div class="kpi"><span>Best Alpha vs B&H</span><strong>{best_alpha}</strong></div>
          <div class="kpi"><span>Median Fee / Capital</span><strong>{median_fee}</strong></div>
        </div>
        """
        body = "\n".join(
            [
                "<h3>Decision Summary</h3>",
                summary,
                _fig_html(
                    px.bar(
                        best,
                        x="name",
                        y="alpha_vs_bh_pct",
                        color="asset",
                        title="Top 20 Experiments by Alpha vs Buy-and-Hold",
                    )
                ),
                _optional_fig(
                    nav_frame,
                    lambda data: px.line(
                        data,
                        x="day_index",
                        y="nav",
                        color="name",
                        title="NAV Curves for Top Candidates",
                        markers=True,
                    ),
                ),
                _fig_html(
                    px.scatter(
                        frame,
                        x="num_trades",
                        y="return_pct",
                        color="reward",
                        symbol="actions",
                        hover_data=["name", "asset", "interval", "encoder", "total_fees"],
                        title="Return vs Trade Count",
                    )
                ),
                _fig_html(
                    px.scatter(
                        frame,
                        x="coverage_pct",
                        y="alpha_vs_bh_pct",
                        color="interval",
                        size="num_trades",
                        hover_data=["name", "asset", "reward", "encoder"],
                        title="Alpha vs State Coverage",
                    )
                ),
                _fig_html(
                    px.box(
                        frame,
                        x="interval",
                        y="alpha_vs_bh_pct",
                        color="encoder",
                        title="Alpha Distribution by Interval and Encoder",
                    )
                ),
                _fig_html(
                    px.box(
                        frame,
                        x="actions",
                        y="alpha_vs_bh_pct",
                        color="reward",
                        title="Binary vs Ternary Action Space",
                    )
                ),
                _fig_html(
                    px.density_heatmap(
                        frame,
                        x="reward",
                        y="encoder",
                        z="alpha_vs_bh_pct",
                        histfunc="avg",
                        title="Average Alpha by Reward and Encoder",
                    )
                ),
                _fig_html(
                    px.density_heatmap(
                        frame,
                        x="asset",
                        y="interval",
                        z="alpha_vs_bh_pct",
                        histfunc="avg",
                        title="Average Alpha by Asset and Candle Interval",
                    )
                ),
                _fig_html(
                    px.scatter(
                        frame,
                        x="fees_pct_abs_pnl",
                        y="alpha_vs_bh_pct",
                        color="asset",
                        symbol="interval",
                        hover_data=["name", "reward", "encoder", "num_trades"],
                        title="Fee Burden vs Alpha",
                    )
                ),
                _fig_html(
                    px.parallel_categories(
                        best,
                        dimensions=["asset", "interval", "reward", "encoder", "actions"],
                        color="alpha_vs_bh_pct",
                        title="Top Candidate Design Patterns",
                    )
                ),
                "<h3>Full Results Table</h3>",
                all_results.to_html(
                    index=False,
                    classes="data-table",
                    float_format=lambda x: f"{x:,.4f}",
                ),
            ]
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quantphemes RL Research Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172033;
      background: #f5f7fb;
      line-height: 1.55;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 22px 60px; }}
    section {{
      background: #fff;
      border: 1px solid #d9e0ea;
      border-radius: 8px;
      padding: 22px;
      margin: 18px 0;
      box-shadow: 0 12px 30px rgba(31, 45, 64, 0.06);
    }}
    h1 {{ font-size: 34px; margin: 0 0 10px; }}
    h2 {{ margin-top: 0; }}
    h3 {{ margin: 24px 0 10px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 14px 0 22px;
    }}
    .kpi {{
      border: 1px solid #dde5ef;
      border-radius: 8px;
      padding: 14px;
      background: #f8fafc;
    }}
    .kpi span {{
      display: block;
      color: #607085;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .kpi strong {{
      display: block;
      margin-top: 6px;
      font-size: 18px;
      overflow-wrap: anywhere;
    }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #e1e7ef;
      padding: 8px;
      text-align: right;
    }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .muted {{ color: #607085; }}
  </style>
</head>
<body>
<main>
  <h1>Quantphemes RL ETF Research Report</h1>
  <p class="muted">
    Comparison of Q-table RL experiments across assets, intervals, reward functions,
    state encoders, and action spaces.
  </p>
  <section>
    <h2>Introduction</h2>
    <p>
      This research asks whether tabular reinforcement learning can identify intraday ETF
      policies that survive realistic transaction costs and outperform simple buy-and-hold
      or always-long baselines. The sweep covers broad-market ETFs, leveraged ETFs,
      multiple candle intervals, reward designs, state encoders, and binary or ternary
      position sizing.
    </p>
  </section>
  <section>
    <h2>Methodology</h2>
    <p>
      Each experiment is YAML-defined, uses the same fee model, trains walk-forward
      Q-tables, evaluates greedily, and writes reproducible metrics. The comparison
      emphasizes alpha versus buy-and-hold, trade count, fee burden, state coverage,
      and robustness across candle intervals.
    </p>
    <ul>
      <li><strong>Universe:</strong> 2800.HK, 2828.HK, 7299.HK, 7568.HK, 7226.HK.</li>
      <li><strong>Intervals:</strong> 1h, 30m, 15m, 5m, and 1m, subject to data availability.</li>
      <li><strong>Rewards:</strong> pure log return, long bias, fee aware, sparse liquidation,
        and drawdown penalty.</li>
      <li><strong>State encoders:</strong> baseline price bins plus RSI, volatility, volume,
        and z-score variants.</li>
      <li><strong>Actions:</strong> binary cash/full and ternary cash/half/full.</li>
    </ul>
  </section>
  <section>
    <h2>Main Results and Analysis</h2>
    {body}
  </section>
  <section>
    <h2>Limitations</h2>
    <p>
      The report depends on available historical data quality, complete HK session bars,
      and the assumption that paper-trading fills approximate target holdings. High-frequency
      experiments may overstate tradability because spread and slippage are not yet
      modelled separately from fees.
    </p>
  </section>
  <section>
    <h2>Conclusion</h2>
    <p>
      Prefer candidates that beat buy-and-hold after fees, use a reasonable number of
      trades, maintain good state coverage, and behave consistently across nearby intervals.
      Live deployment should remain limited to one or two candidates after dry-run validation.
    </p>
  </section>
</main>
</body>
</html>
"""


def _fig_html(fig: Any) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def _optional_fig(frame: pd.DataFrame, builder: Any) -> str:
    if frame.empty:
        return "<p class=\"muted\">NAV curves are unavailable for these result folders.</p>"
    return _fig_html(builder(frame))


def _load_nav_curves(result_dirs: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for result_dir_text in result_dirs:
        result_dir = Path(result_dir_text)
        eval_path = result_dir / "eval_log.csv"
        if not eval_path.exists():
            continue
        frame = pd.read_csv(eval_path)
        if frame.empty or "nav" not in frame.columns:
            continue
        frame = frame[frame["agent"] == "main"].copy()
        if frame.empty:
            continue
        frame["name"] = result_dir.name
        rows.append(frame[["name", "day_index", "date", "nav", "return"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _infer_interval(config: dict[str, Any]) -> str:
    path = str(config["data"].get("path", "")).lower()
    for candidate in ("m1", "m5", "m15", "m30", "m60", "1h", "30min", "5min"):
        if candidate in path:
            return {"m60": "1h", "m30": "30m", "m15": "15m", "m5": "5m", "m1": "1m"}.get(
                candidate, candidate
            )
    return f"{len(config['market'].get('decision_times', []))}_decisions"


if __name__ == "__main__":
    main()
