"""
diagnostics.py — recurring checks for the quality scores and the data under
them. Read-only: nothing here writes to the database.

Three questions, asked after every change to line_items.py, piotroski.py or
the ingest:

1. Regression — did a code change alter any score it should not have?
   save_baseline() BEFORE the change, compare_to_baseline() AFTER it.
   Every row that comes back must have an explanation.

2. Coverage — for what share of the universe is each criterion N/A?
   coverage_report() + na_rates() / na_by_sector().

3. Integrity — is the fundamentals table internally consistent?
   integrity_report(). Run after every weekly ingest. Checks (b) and (c)
   must be 0; (a) is informational (late filers, amendments).
"""
from __future__ import annotations

import pandas as pd

from quality_scores.piotroski import piotroski_f_score

REGRESSION_TICKERS = ["MSFT", "AAPL", "PPL", "ULTA", "LHX"]
BASELINE_PATH = "data/piotroski_baseline.csv"
COVERAGE_PATH = "data/piotroski_coverage.csv"


# ── Regression ──────────────────────────────────────────────────────────────
def score_snapshot(tickers, engine, as_of=None) -> pd.DataFrame:
    """One row per ticker: score, max_score and each criterion (True/False/None)."""
    rows = []
    for t in tickers:
        r = piotroski_f_score(t, engine, as_of=as_of)
        if r is None:
            rows.append({"ticker": t})
            continue
        rows.append({"ticker": t, "score": r.score, "max_score": r.max_score, **r.breakdown})
    return pd.DataFrame(rows).set_index("ticker")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # CSV round-trips turn None into NaN and ints may become floats; compare as
    # text so True/False/None and 6 vs 6.0 line up regardless of dtype.
    out = df.astype(object).where(df.notna(), None)
    return out.map(lambda v: "None" if v is None
                   else str(int(v)) if isinstance(v, float) and v.is_integer()
                   else str(v))


def save_baseline(engine, tickers=REGRESSION_TICKERS, path=BASELINE_PATH) -> pd.DataFrame:
    snap = score_snapshot(tickers, engine)
    snap.to_csv(path)
    print(f"Baseline για {len(snap)} tickers → {path}")
    return snap


def compare_to_baseline(engine, path=BASELINE_PATH) -> pd.DataFrame:
    """Rows = (ticker, field) that changed since the baseline. Empty = no change."""
    before = pd.read_csv(path, index_col=0)
    after = score_snapshot(list(before.index), engine)
    cols = sorted(set(before.columns) | set(after.columns))
    b = _normalize(before.reindex(columns=cols))
    a = _normalize(after.reindex(index=before.index, columns=cols))
    diff = (b != a).stack()
    changed = diff[diff].index
    return pd.DataFrame(
        [{"ticker": t, "field": f, "before": b.at[t, f], "after": a.at[t, f]}
         for t, f in changed]
    )


# ── Coverage ────────────────────────────────────────────────────────────────
def coverage_report(engine, min_last_year=2025, save_to=COVERAGE_PATH) -> pd.DataFrame:
    """Piotroski breakdown for every clean, recently-reporting ticker."""
    from data_quality import data_quality_report, clean_universe

    report = data_quality_report(engine)
    universe = [t for t in clean_universe(report, window_years=3)
                if report.at[t, "last_year"] >= min_last_year]

    rows = []
    for i, t in enumerate(universe, 1):
        r = piotroski_f_score(t, engine)
        if r is not None:
            rows.append({"ticker": t, **r.breakdown})
        if i % 100 == 0:
            print(f"  ... {i}/{len(universe)}")

    cov = pd.DataFrame(rows).set_index("ticker")
    if save_to:
        cov.to_csv(save_to)
    return cov


def na_rates(cov: pd.DataFrame) -> pd.Series:
    """% of tickers where each criterion could not be evaluated."""
    return (cov.isna().mean() * 100).round(1).sort_values(ascending=False)


def na_by_sector(cov: pd.DataFrame,
                 cols=("leverage_decreased", "gross_margin_improved",
                       "current_ratio_improved")) -> pd.DataFrame:
    from sector_lookup import get_sectors

    sectors = cov.index.map(get_sectors(list(cov.index)))
    out = (cov[list(cols)].isna().groupby(sectors).mean() * 100).round(0)
    out.insert(0, "companies", cov.groupby(sectors).size())
    return out.sort_values(cols[0], ascending=False)


# ── Integrity ───────────────────────────────────────────────────────────────
# Oct 2026: comparative columns (same quarter last year, prior year-end
# balance sheet) were stored under the current filing's fy/fp label for
# ~165k rows, unnoticed for two months because annual data -- all that
# Piotroski reads -- was fine. These checks would have caught it on day one.
INTEGRITY_SKIP = ("PARA", "XOM")   # two CIKs on one ticker -- known, unrepaired


def integrity_report(engine, skip=INTEGRITY_SKIP) -> dict:
    """
    (a) rows filed > 300 days after their period ended, by form.
        Informational: late filers (e.g. SMCI 2019) and amendments land here
        legitimately. Baseline Oct 2026: 10-Q 288 rows / 3 tickers,
        10-K 85 / 2. A jump to thousands = something changed.
    (b) the same (ticker, metric, form, period_end) under 2+ fiscal years.
        Must be 0: one period, one row.
    (c) one value + one period_end under 3+ different (fiscal_year, fiscal_period).
        Must be 0: the "CTL pattern" -- a year-end balance copied into the
        next three quarters.
    Only metrics the parser owns (XBRL_TAG_MAP) are checked; legacy rows
    from the pre-built DB are out of scope.
    """
    from sqlalchemy import text
    from fundamentals import XBRL_TAG_MAP

    p = {"canon": list(XBRL_TAG_MAP), "skip": list(skip)}
    base = """FROM fundamentals f
              JOIN metric_lookup m     ON m.id  = f.metric_id
              JOIN form_type_lookup ft ON ft.id = f.form_type_id
              WHERE m.name = ANY(:canon) AND NOT (f.ticker = ANY(:skip))"""

    with engine.connect() as c:
        late = pd.read_sql(text(f"""
            SELECT ft.name AS form, COUNT(*) AS rows, COUNT(DISTINCT f.ticker) AS tickers
            {base} AND f.filed_date - f.period_end > 300
            GROUP BY 1 ORDER BY 2 DESC"""), c, params=p)
        dup = c.execute(text(f"""
            SELECT COUNT(*), COUNT(DISTINCT ticker) FROM (
                SELECT f.ticker, f.metric_id, f.form_type_id, f.period_end
                {base} AND f.period_start IS NOT NULL
                GROUP BY 1, 2, 3, 4 HAVING COUNT(DISTINCT f.fiscal_year) > 1) x"""), p).fetchone()
        copied = c.execute(text(f"""
            SELECT COUNT(*), COUNT(DISTINCT ticker) FROM (
                SELECT f.ticker, f.metric_id, f.period_end, f.value
                {base} AND ft.name IN ('10-Q', '10-K')
                GROUP BY 1, 2, 3, 4
                HAVING COUNT(DISTINCT (f.fiscal_year, f.fiscal_period)) >= 3) x"""), p).fetchone()
        size = c.execute(text(
            "SELECT pg_size_pretty(pg_database_size(current_database()))")).scalar()

    ok = dup[0] == 0 and copied[0] == 0
    print("── Integrity ─────────────────────────────────")
    print("(a) lag > 300 μέρες (πληροφοριακό):")
    print(late.to_string(index=False) if len(late) else "    0")
    print(f"(b) ίδια περίοδος, δύο fiscal_year : {dup[0]:,} σε {dup[1]} tickers")
    print(f"(c) ίδια τιμή σε 3+ περιόδους      : {copied[0]:,} σε {copied[1]} tickers")
    print(f"μέγεθος βάσης: {size}  (όριο Neon free tier: 512 MB)")
    print("✅ OK" if ok else "❌ ΠΡΟΣΟΧΗ — (b) ή (c) ≠ 0: σταμάτα και ψάξε πριν εμπιστευτείς τα P/E")
    return {"late": late, "dup": dup, "copied": copied, "size": size, "ok": ok}
