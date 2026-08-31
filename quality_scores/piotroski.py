"""
piotroski.py — Piotroski F-Score (Piotroski, 2000, "Value Investing: The
Use of Historical Financial Statement Information to Separate Winners
from Losers", Journal of Accounting Research 38:1-41).

Nine binary criteria across three groups: profitability (4),
leverage/liquidity (3), operating efficiency (2). Each criterion either
passes (1) or fails (0).

Important design choice: a missing input maps a criterion to None, NOT
False. A score of 6/9 should mean "6 out of 9 could be evaluated and
passed", not "6 passed, the other 3 failed because the underlying metric
was missing". max_score reports how many criteria could actually be
evaluated for this ticker/period.

Usage:
    from db import get_engine
    from quality_scores.piotroski import piotroski_f_score, print_report

    engine = get_engine()
    result = piotroski_f_score("AAPL", engine)
    print_report(result)
"""

from dataclasses import dataclass
from datetime import date

from .line_items import get_annual_snapshots


@dataclass
class PiotroskiResult:
    ticker: str
    score: int
    max_score: int  # how many of the 9 criteria could be evaluated
    breakdown: dict[str, bool | None]
    fiscal_year: int
    filed_date: date


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def piotroski_f_score(ticker: str, engine, as_of=None) -> PiotroskiResult | None:
    snapshots = get_annual_snapshots(ticker, engine, as_of=as_of, n_years=2)

    if len(snapshots) < 2:
        # Can't run a YoY comparison with fewer than 2 years of annual
        # data (e.g. a recent IPO).
        return None

    curr, prev = snapshots[0], snapshots[1]

    roa_curr = _div(curr.net_income, curr.total_assets)
    roa_prev = _div(prev.net_income, prev.total_assets)

    lev_curr = _div(curr.lt_debt, curr.total_assets)
    lev_prev = _div(prev.lt_debt, prev.total_assets)

    cr_curr = _div(curr.current_assets, curr.current_liabilities)
    cr_prev = _div(prev.current_assets, prev.current_liabilities)

    margin_curr = _div(curr.gross_profit, curr.revenues)
    margin_prev = _div(prev.gross_profit, prev.revenues)

    turnover_curr = _div(curr.revenues, curr.total_assets)
    turnover_prev = _div(prev.revenues, prev.total_assets)

    def gt(a, b):
        return None if (a is None or b is None) else a > b

    def gt0(a):
        return None if a is None else a > 0

    breakdown = {
        "positive_roa": gt0(roa_curr),
        "positive_cfo": gt0(curr.operating_cf),
        "roa_improved": gt(roa_curr, roa_prev),
        "cfo_exceeds_net_income": (
            None
            if curr.operating_cf is None or curr.net_income is None
            else curr.operating_cf > curr.net_income
        ),
        "leverage_decreased": (
            None if lev_curr is None or lev_prev is None else lev_curr < lev_prev
        ),
        "current_ratio_improved": gt(cr_curr, cr_prev),
        "no_dilution": (
            None
            if curr.shares_outstanding is None or prev.shares_outstanding is None
            else curr.shares_outstanding <= prev.shares_outstanding
        ),
        "gross_margin_improved": gt(margin_curr, margin_prev),
        "asset_turnover_improved": gt(turnover_curr, turnover_prev),
    }

    valid = {k: v for k, v in breakdown.items() if v is not None}

    return PiotroskiResult(
        ticker=ticker,
        score=sum(valid.values()),
        max_score=len(valid),
        breakdown=breakdown,
        fiscal_year=curr.fiscal_year,
        filed_date=curr.filed_date,
    )


def print_report(result: PiotroskiResult | None) -> None:
    if result is None:
        print("Not enough annual data (need >= 2 years of 10-K filings).")
        return

    print(f"── Piotroski F-Score: {result.ticker} ──────────────────")
    print(f"   Fiscal year {result.fiscal_year} (filed {result.filed_date})")
    print(f"   Score: {result.score} / {result.max_score}")
    print()
    for k, v in result.breakdown.items():
        symbol = "PASS" if v is True else ("FAIL" if v is False else "N/A ")
        print(f"   [{symbol}]  {k}")
