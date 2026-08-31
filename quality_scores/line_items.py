"""
line_items.py — converts the long-format output of fundamentals.get_fundamentals()
into clean annual snapshots, ready to be consumed by quality-score modules
such as Piotroski F-Score.

Design principle: point-in-time correctness. Each AnnualSnapshot carries
its own filed_date, so the caller always knows WHEN this fiscal year was
actually known to the market — same philosophy as
fundamentals.get_fundamentals(as_of=...).

Scope of this first prototype: annual filings only (form_type='10-K',
fiscal_period='FY'). Piotroski's original paper works with annual data;
TTM (trailing-twelve-month) support would require stitching together
quarterly filings and adds complexity with no clear benefit for the
first working version.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from fundamentals import get_fundamentals

# Only the metrics needed by the Piotroski F-Score — keeps the query to
# Postgres narrow instead of pulling all 41 canonical metrics.
NEEDED_METRICS = [
    "Revenues",
    "GrossProfit",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",  # fallback if the above is missing
]


@dataclass
class AnnualSnapshot:
    fiscal_year: int  # derived from period_end.year -- display only, not a dedup key
    period_end: date
    filed_date: date
    revenues: float | None
    gross_profit: float | None
    net_income: float | None
    operating_cf: float | None
    total_assets: float | None
    current_assets: float | None
    current_liabilities: float | None
    lt_debt: float | None  # LongTermDebtNoncurrent + LongTermDebtCurrent
    shares_outstanding: float | None


def _earliest_per_period_end(df: pd.DataFrame) -> pd.DataFrame:
    """If the same (period_end, metric) has more than one row (e.g. a
    10-K/A restatement), keep the row with the EARLIEST filed_date — the
    same "first known to the market" rule used by fundamentals.py's
    upsert logic.

    IMPORTANT: we dedup/group on `period_end`, NOT on the `fiscal_year`
    column. `fiscal_year` is a label SEC/the filer attaches per data
    point, and for duration facts (Revenues, NetIncomeLoss, operating
    cash flow) shown as prior-year comparatives inside a LATER 10-K, that
    label can be wrong -- e.g. a genuine FY2024 figure has been observed
    mislabeled fiscal_year=2026 inside a FY2026 filing. Combined with
    "earliest filed_date wins", a wrong label with an old filed_date will
    silently beat the correct, freshly-filed figure.

    `period_end` is an actual calendar date carried on the XBRL context
    itself and does not drift this way, so it's the reliable key. Two
    different real fiscal years always have different period_end dates."""
    return (
        df.sort_values("filed_date")
        .drop_duplicates(subset=["period_end", "metric"], keep="first")
    )


def get_annual_snapshots(
    ticker: str,
    engine,
    as_of: str | date | None = None,
    n_years: int = 2,
) -> list[AnnualSnapshot]:
    """
    Returns the most recent `n_years` annual snapshots (newest first),
    filtered to form_type='10-K' and fiscal_period='FY'.

    If fewer than n_years annual filings are available (e.g. a recent
    IPO), returns whatever exists — the caller (e.g. piotroski_f_score)
    is responsible for handling that (a YoY comparison isn't possible
    with only 1 year of data).
    """
    df = get_fundamentals(ticker, engine, metrics=NEEDED_METRICS, as_of=as_of)

    if df.empty:
        return []

    annual = df[
        (df["form_type"] == "10-K") & (df["fiscal_period"] == "FY")
    ].copy()

    if annual.empty:
        return []

    annual = _earliest_per_period_end(annual)

    snapshots = []
    for period_end, group in annual.groupby("period_end"):
        vals = dict(zip(group["metric"], group["value"]))
        filed = group["filed_date"].max()  # filed_date of the annual filing

        lt_debt_nc = vals.get("LongTermDebtNoncurrent")
        lt_debt_c = vals.get("LongTermDebtCurrent")
        lt_debt = None
        if lt_debt_nc is not None or lt_debt_c is not None:
            lt_debt = (lt_debt_nc or 0) + (lt_debt_c or 0)

        shares = vals.get("CommonStockSharesOutstanding")
        if shares is None:
            shares = vals.get("WeightedAverageNumberOfSharesOutstandingBasic")

        snapshots.append(
            AnnualSnapshot(
                fiscal_year=period_end.year,
                period_end=period_end,
                filed_date=filed,
                revenues=vals.get("Revenues"),
                gross_profit=vals.get("GrossProfit"),
                net_income=vals.get("NetIncomeLoss"),
                operating_cf=vals.get("NetCashProvidedByUsedInOperatingActivities"),
                total_assets=vals.get("Assets"),
                current_assets=vals.get("AssetsCurrent"),
                current_liabilities=vals.get("LiabilitiesCurrent"),
                lt_debt=lt_debt,
                shares_outstanding=shares,
            )
        )

    snapshots.sort(key=lambda s: s.period_end, reverse=True)
    return snapshots[:n_years]
