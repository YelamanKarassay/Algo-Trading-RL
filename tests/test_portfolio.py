from __future__ import annotations

from quantphemes_rl.portfolio.portfolio import Portfolio


def test_hold_cash_no_fee() -> None:
    portfolio = Portfolio(cash=1_000_000.0, shares=0, position=0, total_fees=0.0)

    fee = portfolio.execute(0, 10.0)

    assert fee == 0.0
    assert portfolio.total_fees == 0.0
    assert portfolio.cash == 1_000_000.0


def test_hold_long_no_fee() -> None:
    portfolio = Portfolio(cash=500_000.0, shares=50_000, position=1, total_fees=100.0)

    fee = portfolio.execute(1, 10.0)

    assert fee == 0.0
    assert portfolio.total_fees == 100.0
    assert portfolio.shares == 50_000


def test_buy_charges_fee() -> None:
    portfolio = Portfolio(
        cash=1_002.0,
        shares=0,
        position=0,
        total_fees=0.0,
        fee_rate_one_side=0.002,
        lot_size=1,
    )

    fee = portfolio.execute(1, 100.0)

    assert fee == 2.0
    assert portfolio.shares == 10
    assert portfolio.cash == 0.0
    assert portfolio.total_fees == 2.0


def test_sell_charges_fee() -> None:
    portfolio = Portfolio(
        cash=0.0,
        shares=10,
        position=1,
        total_fees=0.0,
        fee_rate_one_side=0.002,
        lot_size=1,
    )

    fee = portfolio.execute(0, 110.0)

    assert fee == 2.2
    assert portfolio.cash == 1_097.8
    assert portfolio.shares == 0
    assert portfolio.position == 0


def test_lot_size_rounding() -> None:
    portfolio = Portfolio(cash=1_000_000.0, shares=0, position=0, total_fees=0.0)

    portfolio.execute(1, 10.0)

    assert portfolio.shares % 500 == 0


def test_cash_never_negative() -> None:
    portfolio = Portfolio(cash=1_000_000.0, shares=0, position=0, total_fees=0.0)

    portfolio.execute(1, 10.0)

    assert portfolio.cash >= 0.0


def test_three_action_half_position() -> None:
    portfolio = Portfolio(
        cash=1_000_000.0,
        shares=0,
        position=0,
        total_fees=0.0,
        fee_rate_one_side=0.002,
        lot_size=500,
        max_position=2,
    )

    portfolio.execute(1, 10.0)

    assert portfolio.position == 1
    assert portfolio.shares == 50_000
    assert portfolio.cash == 499_000.0


def test_three_action_full_position_from_half() -> None:
    portfolio = Portfolio(
        cash=1_000_000.0,
        shares=0,
        position=0,
        total_fees=0.0,
        fee_rate_one_side=0.002,
        lot_size=500,
        max_position=2,
    )

    portfolio.execute(1, 10.0)
    portfolio.execute(2, 10.0)

    assert portfolio.position == 2
    assert portfolio.cash >= 0.0
    assert portfolio.shares == 99_500
