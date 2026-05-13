from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from quantphemes_rl.data.base import DayData
from quantphemes_rl.state.base import MarketContext, StateEncoder

FEATURES = ["delta_close", "delta_open", "delta_prev"]


def mutual_info_per_feature(
    days: list[DayData], encoder: StateEncoder, decision_times: list[str]
) -> pd.DataFrame:
    """Compute bin-conditional returns and mutual information per state feature."""
    observations = _observations(days, encoder, decision_times)
    if not observations:
        return pd.DataFrame(columns=["feature", "bin", "mean_return", "std_return", "mi"])

    frame = pd.DataFrame(observations)
    feature_matrix = frame[FEATURES].to_numpy()
    returns = frame["next_return"].to_numpy()
    if len(set(returns)) <= 1:
        mi_values = [0.0 for _ in FEATURES]
    else:
        raw_mi_values = mutual_info_regression(feature_matrix, returns, random_state=0)
        mi_values = [
            0.0 if len(set(frame[feature])) <= 1 else float(value)
            for feature, value in zip(FEATURES, raw_mi_values, strict=True)
        ]
    rows: list[dict[str, Any]] = []
    for feature, mi_value in zip(FEATURES, mi_values, strict=True):
        for bin_value in sorted(frame[feature].unique()):
            subset = frame[frame[feature] == bin_value]["next_return"]
            rows.append(
                {
                    "feature": feature,
                    "bin": int(bin_value),
                    "mean_return": float(subset.mean()),
                    "std_return": float(subset.std(ddof=0)),
                    "mi": float(mi_value),
                }
            )
    return pd.DataFrame(rows)


def bin_conditional_returns(
    days: list[DayData], encoder: StateEncoder, decision_times: list[str]
) -> dict[str, dict[int, list[float]]]:
    """Return next-interval log returns grouped by feature bin."""
    histograms: dict[str, dict[int, list[float]]] = {
        feature: {0: [], 1: [], 2: []} for feature in FEATURES
    }
    for observation in _observations(days, encoder, decision_times):
        for feature in FEATURES:
            histograms[feature][int(observation[feature])].append(observation["next_return"])
    return histograms


def _observations(
    days: list[DayData], encoder: StateEncoder, decision_times: list[str]
) -> list[dict[str, float | int]]:
    observations: list[dict[str, float | int]] = []
    for day in days:
        times = [*decision_times]
        prices = day.prices_at_decision_times(times)
        for index, decision_time in enumerate(times[:-1]):
            current_price = prices[decision_time]
            next_price = prices[times[index + 1]]
            prev_price = prices[times[max(0, index - 1)]]
            ctx = MarketContext(
                current_price=current_price,
                today_open=day.open_price,
                yesterday_close=day.open_price,
                prev_decision_price=prev_price,
                decision_index=index,
                decision_count=len(decision_times),
            )
            encoded = encoder.encode(ctx)
            observations.append(
                {
                    "delta_close": encoded // 9,
                    "delta_open": (encoded % 9) // 3,
                    "delta_prev": encoded % 3,
                    "next_return": math.log(next_price / current_price),
                }
            )
    return observations
