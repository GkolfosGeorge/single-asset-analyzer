"""
data_quality.py — finds companies whose annual (10-K) history has holes,
so research and universe-wide runs can exclude them until the upstream
ingest is fixed (see shared-data-layer: SEC fy/fp mislabels, missing
filings).

The list is computed LIVE from the database, never hardcoded: once the
ingest is fixed and the missing years are loaded, affected tickers drop
off the list automatically.

    report = data_quality_report(eng)          # one row per ticker
    summarize(report)                           # headline numbers
    by_year = gaps_by_year(report)              # % of companies missing each year
    excluded = excluded_tickers(report, window_years=3)

A "gap" = a fiscal year missing BETWEEN a company's first and last 10-K on
file. Data that simply starts late (IPO) or ends (acquisition) is not a
gap.
"""

import pandas as pd
from sqlalchemy import text

# Known problems the gap check cannot see. Add here with a reason.
MANUAL_EXCLUDE = {
    "CPWR": "ticker recycling: Compuware delisted 2014, yet 10-Ks on file for 2024-25",
    "STI": "ticker recycling: SunTrust merged into Truist 2019, yet 10-K on file for 2024",
}

_ANNUAL_PERIODS_SQL = text("""
    SELECT DISTINCT f.ticker, f.period_end
    FROM fundamentals f
    JOIN metric_lookup m ON m.id = f.metric_id
    JOIN form_type_lookup ft ON ft.id = f.form_type_id
    WHERE m.name = 'Assets' AND ft.name = '10-K' AND f.fiscal_period = 'FY'
""")


def load_annual_periods(engine) -> pd.DataFrame:
    """One row per (ticker, fiscal year-end on file). Assets is used as the
    marker because every 10-K reports it."""
    df = pd.read_sql(_ANNUAL_PERIODS_SQL, engine)
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


def analyze_periods(periods: pd.DataFrame) -> pd.DataFrame:
    """Pure function (no DB): one row per ticker with its missing years."""
    rows = []
    for ticker, g in periods.sort_values("period_end").groupby("ticker"):
        ends = list(g["period_end"])
        missing = []
        for earlier, later in zip(ends, ends[1:]):
            gap_days = (later - earlier).days
            if gap_days > 380:
                n_missing = round(gap_days / 365.25) - 1
                missing += [earlier.year + i for i in range(1, n_missing + 1)]
        rows.append({
            "ticker": ticker,
            "first_year": ends[0].year,
            "last_year": ends[-1].year,
            "years_on_file": len(ends),
            "missing_years": missing,
            "n_missing": len(missing),
            "last_missing_year": max(missing) if missing else None,
        })
    return pd.DataFrame(rows).set_index("ticker")


def data_quality_report(engine) -> pd.DataFrame:
    report = analyze_periods(load_annual_periods(engine))
    report["manual_reason"] = report.index.map(MANUAL_EXCLUDE).fillna("")
    return report


def has_gap_in_last(report: pd.DataFrame, window_years: int) -> pd.Series:
    """True if a year is missing within the company's own last
    `window_years` years (e.g. 3 = what the current Piotroski needs)."""
    cutoff = report["last_year"] - window_years + 1
    return report["last_missing_year"].notna() & (report["last_missing_year"] >= cutoff)


def excluded_tickers(report: pd.DataFrame, window_years: int | None = None) -> dict[str, str]:
    """
    {ticker: reason} to exclude.
      window_years=None -> any gap anywhere in history (strict: backtests)
      window_years=3    -> only gaps that affect the current Piotroski
    Manual exclusions always apply.
    """
    if window_years is None:
        flagged = report["n_missing"] > 0
    else:
        flagged = has_gap_in_last(report, window_years)
    out = {
        t: "missing years: " + ", ".join(map(str, report.at[t, "missing_years"]))
        for t in report.index[flagged]
    }
    out.update({t: r for t, r in MANUAL_EXCLUDE.items() if t in report.index})
    return dict(sorted(out.items()))


def clean_universe(report: pd.DataFrame, window_years: int | None = None) -> list[str]:
    excluded = excluded_tickers(report, window_years)
    return sorted(t for t in report.index if t not in excluded)


def gaps_by_year(report: pd.DataFrame, since: int = 2010) -> pd.DataFrame:
    """For each fiscal year: of the companies that existed in our data that
    year (first_year <= Y <= last_year), what % are missing it?"""
    last = int(report["last_year"].max())
    rows = []
    for y in range(since, last + 1):
        active = report[(report["first_year"] <= y) & (report["last_year"] >= y)]
        n_missing = sum(y in m for m in active["missing_years"])
        rows.append({"year": y, "companies": len(active), "missing": n_missing,
                     "pct_missing": 100 * n_missing / len(active) if len(active) else 0.0})
    return pd.DataFrame(rows).set_index("year")


def summarize(report: pd.DataFrame) -> None:
    n = len(report)

    def line(label, count):
        print(f"   {label:<48} {count:>4}  ({100 * count / n:4.1f}%)")

    print(f"── Ποιότητα ετήσιων δεδομένων: {n} εταιρείες ──")
    line("με κενό κάπου στο ιστορικό", int((report["n_missing"] > 0).sum()))
    for w in (3, 5, 10):
        line(f"με κενό στα τελευταία {w} χρόνια τους", int(has_gap_in_last(report, w).sum()))
    line("χειροκίνητες εξαιρέσεις (ticker recycling κ.λπ.)",
         int(report.index.isin(list(MANUAL_EXCLUDE)).sum()))
    print(f"   Χρονιές που λείπουν συνολικά: {int(report['n_missing'].sum())}")


def gaps_by_sector(report: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """% of companies with any gap, per sector. `sectors` = {ticker: sector},
    e.g. from sector_lookup.get_sectors(list(report.index))."""
    df = report.assign(sector=report.index.map(sectors).fillna("Unknown"),
                       has_gap=report["n_missing"] > 0)
    out = df.groupby("sector").agg(companies=("has_gap", "size"), with_gap=("has_gap", "sum"))
    out["pct_with_gap"] = 100 * out["with_gap"] / out["companies"]
    return out.sort_values("pct_with_gap", ascending=False)


def plot_gaps_by_year(by_year: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(by_year.index.astype(str), by_year["pct_missing"], color="#d1495b")
    for bar, (_, r) in zip(bars, by_year.iterrows()):
        ax.annotate(f"{int(r['missing'])}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylabel("% εταιρειών που λείπει η χρονιά")
    ax.set_title("Χρονιές που λείπουν ανά έτος (ο αριθμός = πόσες εταιρείες)",
                 fontsize=12, fontweight="bold", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
