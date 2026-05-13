from __future__ import annotations

import random

from quantphemes_rl.agent.baselines import AlwaysCash, AlwaysLong, BuyAndHold, RandomBinary


def test_random_binary_returns_both_actions() -> None:
    random.seed(3)
    agent = RandomBinary()

    assert {agent.act(0, training=False) for _ in range(100)} == {0, 1}


def test_always_long_returns_one() -> None:
    assert AlwaysLong().act(0, training=False) == 1


def test_always_cash_returns_zero() -> None:
    assert AlwaysCash().act(0, training=False) == 0


def test_buy_and_hold_buys_first_then_maintains() -> None:
    agent = BuyAndHold()

    assert agent.act(0, training=False) == 1
    assert agent.act(1, training=False) == 1
    agent.decay_epsilon()
    assert agent.previous_action == 0
