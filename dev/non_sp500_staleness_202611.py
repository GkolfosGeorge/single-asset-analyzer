"""
dev/non_sp500_staleness_202611.py — READ-ONLY (βήμα 4δ, 25/11/2026).

Hypothesis: the weekly ingest (daily-market-data, Mondays) writes ONLY today's
S&P 500, so the ~200 other companies of the peer table are FROZEN at their last
ingested filing. Example: FLR filed its Q3 2026 10-Q in early November, but our
last quarter is 06/2026. A frozen company keeps its yield until its last period
is older than 270 days, then drops out of the 🏷️ line.

What it measures, for every company of peer_data/peer_yields.csv:
  - in today's S&P 500? (same list as the weekly ingest)
  - last time ANY row was written for it (inserted_at), last period, last filing
  - "frozen": nothing written in the last 21 days although a new quarter is due
    (last period older than 120 days)
  - when it loses its yield: last period + 270 days
and the cost of writing them too: rows per filing and MB per row in Neon.

    python dev/non_sp500_staleness_202611.py --selftest   (no network, no database)
    python dev/non_sp500_staleness_202611.py              (~10 sec)
→ data/non_sp500_staleness_202611.csv
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

TABLE = "peer_data/peer_yields.csv"
OUT = "data/non_sp500_staleness_202611.csv"
STALE_DAYS = 270          # same limit as the card (MAX_TTM_AGE_DAYS)
DUE_DAYS = 120            # a new 10-Q is normally filed ≤ 45 (10-K ≤ 90) days after the quarter: 90 + 30
QUIET_DAYS = 21           # weekly ingest ran ≥ 3 times since
YIELDS = ["earnings_yield", "ebit_yield", "fcf_yield"]


def status(last_period, last_inserted, today) -> str:
    if pd.isna(last_period):
        return "χωρίς στοιχεία"
    age = (today - last_period).days
    quiet = pd.isna(last_inserted) or (today - last_inserted).days > QUIET_DAYS
    if age > STALE_DAYS:
        return "ήδη παλιά (> 270 μέρες)"
    if age > DUE_DAYS and quiet:
        return "παγωμένη"
    if age > DUE_DAYS:
        return "αργοπορεί (γράφεται, χωρίς νέο τρίμηνο)"
    return "φρέσκια"


def selftest() -> None:
    t = pd.Timestamp("2026-11-25")
    assert status(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-11-24"), t) == "φρέσκια"
    # FLR: last period 30/06, last write months ago
    assert status(pd.Timestamp("2026-06-30"), pd.Timestamp("2026-08-10"), t) == "παγωμένη"
    # same age but the ingest still writes it (late filer)
    assert status(pd.Timestamp("2026-06-30"), pd.Timestamp("2026-11-23"), t).startswith("αργοπορεί")
    assert status(pd.Timestamp("2025-12-31"), pd.Timestamp("2026-03-01"), t).startswith("ήδη")
    assert status(pd.NaT, pd.NaT, t) == "χωρίς στοιχεία"
    # recent period, never re-written: still fresh (nothing new is due yet)
    assert status(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-01"), t) == "φρέσκια"
    print("selftest ok")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    from sqlalchemy import text
    from db import get_engine
    from ticker_provider import get_tickers

    today = pd.Timestamp.today().normalize()
    table = pd.read_csv(TABLE)
    tickers = table["ticker"].tolist()
    sp = set(get_tickers("sp500", folder_path="cache"))
    eng = get_engine()
    with eng.connect() as conn:
        f = pd.DataFrame(conn.execute(text("""
            SELECT f.ticker, COUNT(*), MAX(f.period_end), MAX(f.filed_date), MAX(f.inserted_at)
            FROM fundamentals f JOIN form_type_lookup ft ON ft.id = f.form_type_id
            WHERE f.ticker = ANY(:t) AND ft.name IN ('10-Q', '10-K')
            GROUP BY f.ticker"""), {"t": tickers}).fetchall(),
            columns=["ticker", "rows", "last_period", "last_filed", "last_inserted"])
        # cost: rows written per filing (last 12 months, S&P), bytes per row
        per_filing = conn.execute(text("""
            SELECT COUNT(*)::float / NULLIF(COUNT(DISTINCT (ticker, filed_date)), 0)
            FROM fundamentals WHERE filed_date >= CURRENT_DATE - INTERVAL '365 days'""")).scalar()
        size_mb, n_rows = conn.execute(text("""
            SELECT pg_total_relation_size('fundamentals') / 1048576.0, (SELECT COUNT(*) FROM fundamentals)
        """)).fetchone()
        db_mb = conn.execute(text("SELECT pg_database_size(current_database()) / 1048576.0")).scalar()

    for c in ("last_period", "last_filed", "last_inserted"):
        f[c] = pd.to_datetime(f[c], utc=True).dt.tz_localize(None)
    d = table[["ticker"] + YIELDS].merge(f, on="ticker", how="left")
    d["in_sp500"] = d["ticker"].isin(sp)
    d["n_yields"] = d[YIELDS].notna().sum(axis=1)
    d["status"] = [status(p, i, today) for p, i in zip(d["last_period"], d["last_inserted"])]
    d["loses_yield_on"] = d["last_period"] + pd.Timedelta(days=STALE_DAYS)
    d["days_since_write"] = (today - d["last_inserted"]).dt.days
    d.drop(columns=YIELDS).to_csv(OUT, index=False)

    print(f"Πίνακας ομοίων: {len(d)} εταιρείες · S&P 500 σήμερα: {len(sp)} · "
          f"από τον πίνακα στο S&P: {int(d['in_sp500'].sum())} · εκτός: {int((~d['in_sp500']).sum())}")

    print("\n══ 1. Πότε γράφτηκε τελευταία φορά κάτι (μέρες πριν) ══")
    for flag, name in ((True, "S&P 500"), (False, "εκτός S&P")):
        x = d.loc[d["in_sp500"] == flag, "days_since_write"]
        print(f"  {name:<10} N={len(x):<4} διάμεσος {x.median():>5.0f} · "
              f"≤ 14 μέρες: {int((x <= 14).sum()):>3} · > 60 μέρες: {int((x > 60).sum()):>3}")

    print("\n══ 2. Κατάσταση ══")
    print(pd.crosstab(d["status"], d["in_sp500"].map({True: "S&P 500", False: "εκτός S&P"}),
                      margins=True, margins_name="σύνολο").to_string())

    frozen = d[(~d["in_sp500"]) & (d["n_yields"] > 0)].copy()
    print(f"\n══ 3. Εκτός S&P με απόδοση σήμερα: {len(frozen)} — πότε τη χάνουν (τελευταίο τρίμηνο + 270 μέρες) ══")
    sched = frozen["loses_yield_on"].dt.to_period("Q").value_counts().sort_index()
    gone_so_far = 0
    for q, n in sched.items():
        gone_so_far += n
        print(f"  ως {q}: {n:>3}  (σύνολο {gone_so_far:>3})")
    sp_line = int((d["n_yields"] > 0).sum())
    print(f"  Σήμερα με απόδοση: {sp_line}/{len(d)}. Αν δεν αλλάξει τίποτα, μέχρι το τέλος του 2027 "
          f"μένουν ~{sp_line - int((frozen['loses_yield_on'] <= '2027-12-31').sum())}.")

    print("\n══ 4. Παραδείγματα (εκτός S&P, πιο παλιά γραμμένα πρώτα) ══")
    cols = ["ticker", "last_period", "last_filed", "last_inserted", "status", "loses_yield_on"]
    ex = d[~d["in_sp500"]].sort_values("last_inserted").head(12)[cols]
    for c in cols[1:4] + ["loses_yield_on"]:
        ex[c] = ex[c].dt.strftime("%Y-%m-%d")
    print(ex.to_string(index=False))
    flr = d[d["ticker"] == "FLR"]
    if len(flr):
        r = flr.iloc[0]
        print(f"\n  FLR: στο S&P 500 = {r['in_sp500']} · τελευταίο τρίμηνο {r['last_period']:%Y-%m-%d} · "
              f"τελευταία εγγραφή {r['last_inserted']:%Y-%m-%d} · {r['status']}")

    print("\n══ 5. Κόστος αν το weekly ingest γράφει ΚΑΙ τις εκτός S&P ══")
    per_row_mb = size_mb / n_rows if n_rows else float("nan")
    n_out = int((~d["in_sp500"]).sum())
    per_year = n_out * 4 * (per_filing or 0) * per_row_mb
    print(f"  Βάση: {db_mb:,.0f} MB (πίνακας fundamentals {size_mb:,.0f} MB, {n_rows:,} γραμμές, "
          f"{per_row_mb * 1024:,.2f} KB/γραμμή)")
    print(f"  Γραμμές ανά κατάθεση (τελευταίοι 12 μήνες): {per_filing:,.0f}")
    print(f"  {n_out} εταιρείες × 4 καταθέσεις × {per_filing:,.0f} γραμμές ≈ "
          f"{n_out * 4 * (per_filing or 0):,.0f} γραμμές/έτος ≈ {per_year:,.1f} MB/έτος")
    print(f"  (το κενό μέχρι σήμερα: οι παγωμένες έχουν ήδη 1–2 καταθέσεις που λείπουν)")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
