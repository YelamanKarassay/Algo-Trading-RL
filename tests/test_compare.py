from __future__ import annotations

import json
from pathlib import Path

from apps.compare import compare_results


def test_compare_outputs_expected_columns(tmp_path: Path) -> None:
    first = _result_dir(tmp_path, "exp_a", "2800.HK", 1_010_000.0)
    second = _result_dir(tmp_path, "exp_b", "7299.HK", 990_000.0)

    frame = compare_results([first, second])

    assert len(frame) == 2
    assert list(frame.columns) == [
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
    ]
    assert frame["name"].tolist() == ["exp_a", "exp_b"]


def _result_dir(tmp_path: Path, name: str, symbol: str, final_nav: float) -> Path:
    result_dir = tmp_path / name
    result_dir.mkdir()
    (result_dir / "metrics.json").write_text(
        json.dumps(
            {
                "capital": 1_000_000,
                "final_nav": final_nav,
                "return": final_nav / 1_000_000 - 1,
                "sharpe": 1.2,
                "alpha_vs_bh": 0.01,
                "num_trades": 4,
                "total_fees": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "config.snapshot.yaml").write_text(
        f"""
name: {name}
data:
  symbol: {symbol}
  path: data/raw/{symbol.replace(".", "_")}_1h.csv
market:
  capital: 1000000
  decision_times: ["10:30", "11:30", "13:30", "14:30", "15:30"]
agent:
  type: qtable
""",
        encoding="utf-8",
    )
    return result_dir
