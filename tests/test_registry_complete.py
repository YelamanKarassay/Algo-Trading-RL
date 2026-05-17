from __future__ import annotations

import quantphemes_rl
from quantphemes_rl.registry import list_registered


def test_all_baseline_plugins_registered_after_package_import() -> None:
    assert quantphemes_rl is not None
    assert list_registered("data_source") == ["bloomberg", "csv", "futu"]
    assert list_registered("state_encoder") == [
        "three_feature",
        "with_rsi",
        "with_volatility",
        "with_volume",
        "zscore_bins",
    ]
    assert list_registered("reward_function") == [
        "drawdown_penalty",
        "fee_aware",
        "log_return",
        "log_return_long_bias",
        "sparse_liquidation",
    ]
    assert list_registered("agent") == [
        "always_cash",
        "always_long",
        "buy_and_hold",
        "qtable",
        "random_binary",
    ]
