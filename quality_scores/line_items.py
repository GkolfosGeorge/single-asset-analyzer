"""
line_items.py — converts the long-format output of fundamentals.get_fundamentals()
into clean annual snapshots, ready to be consumed by quality-score modules
such as Piotroski F-Score.

Design principle: point-in-time correctness. Each AnnualSnapshot carries
its own filed_date, so the caller always knows WHEN this fiscal year was
actually known to the market — same philosophy as
fundamentals.get_fundamentals(as_of=...).

Second design principle (added 2026-09): snapshots carry RAW components,
never pre-combined values. Earlier versions summed the two debt tags with
`(x or 0)` and silently fell back from period-end share count to weighted
average shares. Both let a year-over-year comparison mix two different
measures. Choosing a measure that exists in BOTH years being compared is
the scorer's job (see piotroski._consistent_pair), because only the
scorer knows which two years it is comparing.

Scope: annual filings only (form_type='10-K', fiscal_period='FY').
Piotroski's original paper works with annual data; TTM support would
require stitching together quarterly filings.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from fundamentals import get_fundamentals

NEEDED_METRICS = [
    "Revenues",
    "CostOfRevenue",  # fallback for gross profit: Revenues - CostOfRevenue
    "GrossProfit",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
]

# A fiscal year is 52 or 53 weeks (364/371 days) or a calendar year
# (365/366). Anything outside this window means a year is missing between
# two snapshots (or a fiscal-year-end change) and they must not be
# compared as "this year vs last year".
CONSECUTIVE_DAYS = (350, 380)


@dataclass
class AnnualSnapshot:
    fiscal_year: int  # derived from period_end.year -- display only, not a dedup key
    period_end: date
    filed_date: date
    revenues: float | None
    cost_of_revenue: float | None
    gross_profit_reported: float | None
    net_income: float | None
    operating_cf: float | None
    total_assets: float | None
    current_assets: float | None
    current_liabilities: float | None
    lt_debt_noncurrent: float | None
    lt_debt_current: float | None
    shares_period_end: float | None  # CommonStockSharesOutstanding
    shares_weighted_avg: float | None  # WeightedAverageNumberOfSharesOutstandingBasic

    @property
    def gross_profit_computed(self) -> float | None:
        if self.revenues is None or self.cost_of_revenue is None:
            return None
        return self.revenues - self.cost_of_revenue


def is_consecutive(later: AnnualSnapshot, earlier: AnnualSnapshot) -> bool:
    """True if `earlier` is the fiscal year immediately before `later`."""
    gap = (later.period_end - earlier.period_end).days
    return CONSECUTIVE_DAYS[0] <= gap <= CONSECUTIVE_DAYS[1]


def _earliest_per_period_end(df: pd.DataFrame) -> pd.DataFrame:
    """If the same (period_end, metric) has more than one row (e.g. a
    10-K/A restatement), keep the row with the EARLIEST filed_date — the
    same "first known to the market" rule used by fundamentals.py's
    upsert logic.

    IMPORTANT: we dedup/group on `period_end`, NOT on the `fiscal_year`
    column. `fiscal_year` is a label SEC/the filer attaches per data
    point, and for duration facts shown as prior-year comparatives inside
    a LATER 10-K, that label can be wrong. `period_end` is an actual
    calendar date carried on the XBRL context itself and does not drift
    this way, so it's the reliable key."""
    return (
        df.sort_values("filed_date")
        .drop_duplicates(subset=["period_end", "metric"], keep="first")
    )


def get_annual_snapshots(
    ticker: str,
    engine,
    as_of: str | date | None = None,
    n_years: int = 3,
) -> list[AnnualSnapshot]:
    """
    Returns the most recent `n_years` annual snapshots (newest first),
    filtered to form_type='10-K' and fiscal_period='FY'.

    Default is 3 years because Piotroski scales by BEGINNING-of-year
    assets: comparing year t with t-1 needs assets at the start of t-1,
    i.e. the end of t-2.

    Returns whatever exists if fewer years are available (e.g. a recent
    IPO). Consecutiveness is NOT enforced here; callers check it with
    is_consecutive().
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
        vals = {k: (None if pd.isna(v) else float(v)) for k, v in vals.items()}
        filed = group["filed_date"].max()

        snapshots.append(
            AnnualSnapshot(
                fiscal_year=period_end.year,
                period_end=period_end,
                filed_date=filed,
                revenues=vals.get("Revenues"),
                cost_of_revenue=vals.get("CostOfRevenue"),
                gross_profit_reported=vals.get("GrossProfit"),
                net_income=vals.get("NetIncomeLoss"),
                operating_cf=vals.get("NetCashProvidedByUsedInOperatingActivities"),
                total_assets=vals.get("Assets"),
                current_assets=vals.get("AssetsCurrent"),
                current_liabilities=vals.get("LiabilitiesCurrent"),
                lt_debt_noncurrent=vals.get("LongTermDebtNoncurrent"),
                lt_debt_current=vals.get("LongTermDebtCurrent"),
                shares_period_end=vals.get("CommonStockSharesOutstanding"),
                shares_weighted_avg=vals.get("WeightedAverageNumberOfSharesOutstandingBasic"),
            )
        )

    snapshots.sort(key=lambda s: s.period_end, reverse=True)
    return snapshots[:n_years]
