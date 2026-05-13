from __future__ import annotations

from dataclasses import replace

from quantphemes_rl.data.base import DayData
from quantphemes_rl.portfolio.portfolio import Portfolio


def oracle_nav(
    days: list[DayData],
    decision_times: list[str],
    force_close_time: str,
    fee_rate_one_side: float,
    capital: float,
    lot_size: int,
) -> tuple[float, list[float]]:
    """Return ex-post optimal NAV from perfect-foresight binary actions."""
    per_day_navs = [
        _oracle_day_nav(day, decision_times, force_close_time, fee_rate_one_side, capital, lot_size)
        for day in days
    ]
    final_nav = per_day_navs[-1] if per_day_navs else capital
    return final_nav, per_day_navs


def _oracle_day_nav(
    day: DayData,
    decision_times: list[str],
    force_close_time: str,
    fee_rate_one_side: float,
    capital: float,
    lot_size: int,
) -> float:
    prices = day.prices_at_decision_times([*decision_times, force_close_time])
    states = [Portfolio(capital, 0, 0, 0.0, fee_rate_one_side, lot_size)]
    for decision_time in decision_times:
        price = prices[decision_time]
        candidates = []
        for portfolio in states:
            for target in (0, 1):
                candidate = replace(portfolio)
                candidate.execute(target, price)
                candidates.append(candidate)
        states = _best_by_position(candidates, prices[decision_time])

    close_price = prices[force_close_time]
    terminal_navs = []
    for portfolio in states:
        candidate = replace(portfolio)
        candidate.execute(0, close_price)
        terminal_navs.append(candidate.market_value(close_price))
    return max(terminal_navs)


def _best_by_position(portfolios: list[Portfolio], price: float) -> list[Portfolio]:
    best: dict[int, Portfolio] = {}
    for portfolio in portfolios:
        current = best.get(portfolio.position)
        if current is None or portfolio.market_value(price) > current.market_value(price):
            best[portfolio.position] = portfolio
    return list(best.values())
