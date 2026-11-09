"""
dev/tag_discovery_202611.py — READ-ONLY toward the database: what would step 1β
(cash and convertible debt) cost, and whom would it fix? Nothing is ingested.

    python dev/tag_discovery_202611.py            (~20–30 min, resumable, asks SEC)
    python dev/tag_discovery_202611.py --report   (reprints the summary)

→ data/tag_discovery_202611.csv   one row per ticker × tag found

Why: Neon is near its 0.5 GB limit, so we want NUMBERS before touching the ingest.
Only tags still alive count: a tag that stopped in 2018 (KLAC's
MarketableSecuritiesCurrent) doesn't help today's EV, though it still costs rows
and would help the history. So both figures are shown.
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd
from sqlalchemy import text

from db import get_engine
from fundamentals import fetch_companyfacts
from tag_discovery_202610 import summarize_tag, ticker_ciks

OUT = "data/tag_discovery_202611.csv"
SLEEP = 0.15
LIVE_AFTER = pd.Timestamp.today() - pd.Timedelta(days=270)
WATCH = ["GOOGL", "NVDA", "KLAC", "AAPL", "DXCM", "CMG", "PD", "ADI"]   # the outliers of the Yahoo check

CANDIDATES = {
    "cash_securities": [
        "ShortTermInvestments",                              # already ingested
        "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "AvailableForSaleSecuritiesCurrent",
        "DebtSecuritiesCurrent",
        "EquitySecuritiesFvNiCurrent",
        "OtherShortTermInvestments",
        "CashCashEquivalentsAndShortTermInvestments",        # combined: cash INCLUDED
        "AvailableForSaleSecuritiesDebtSecurities",          # total: may include noncurrent
        "MarketableSecurities",                              # total: may include noncurrent
    ],
    "convertible_debt": [
        "ConvertibleNotesPayable",
        "ConvertibleNotesPayableCurrent",
        "ConvertibleNotesPayableNoncurrent",
        "ConvertibleLongTermNotesPayable",
        "ConvertibleDebt",
        "ConvertibleDebtCurrent",
        "ConvertibleDebtNoncurrent",
        "ConvertibleSubordinatedDebt",
        "ConvertibleSubordinatedDebtCurrent",
        "ConvertibleSubordinatedDebtNoncurrent",
    ],
}
ALREADY_INGESTED = {"ShortTermInvestments"}


def survey(eng) -> None:
    tc = ticker_ciks(eng)
    done = set(pd.read_csv(OUT)["ticker"]) if os.path.exists(OUT) else set()
    todo = tc[~tc["ticker"].isin(done)]
    print(f"{len(tc)} tickers, {len(done)} έτοιμα, {len(todo)} μένουν\n")
    t0 = time.time()
    for i, (t, cik) in enumerate(todo.itertuples(index=False), 1):
        recs = []
        try:
            gaap = (fetch_companyfacts(int(cik)) or {}).get("facts", {}).get("us-gaap", {})
            for group, tags in CANDIDATES.items():
                for tag in tags:
                    concept = gaap.get(tag)
                    if concept:
                        s = summarize_tag([o for u in concept.get("units", {}).values() for o in u])
                        if s:
                            recs.append({"ticker": t, "group": group, "tag": tag, **s})
            if not recs:
                recs.append({"ticker": t, "group": "_none", "tag": "", "n_obs": 0, "n_periods": 0})
        except Exception as e:
            recs.append({"ticker": t, "group": "_error", "tag": str(e)[:120], "n_obs": 0, "n_periods": 0})
        pd.DataFrame(recs).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        time.sleep(SLEEP)
        if i % 50 == 0:
            print(f"  ... {i}/{len(todo)}  ~{(time.time() - t0) / i * (len(todo) - i) / 60:.0f} min")


def report(eng) -> None:
    raw = pd.read_csv(OUT)
    errors = raw[raw["group"] == "_error"]
    d = raw[~raw["group"].isin(["_none", "_error"])].copy()
    d["last_end"] = pd.to_datetime(d["last_end"])
    d["live"] = d["last_end"] >= LIVE_AFTER
    with eng.connect() as c:
        size, rows = c.execute(text(
            "SELECT pg_total_relation_size('fundamentals'), COUNT(*) FROM fundamentals")).fetchone()
        db_size = c.execute(text("SELECT pg_database_size(current_database())")).scalar()
    per_row = size / rows
    print(f"── Βάση: {db_size / 1e6:.0f} MB, fundamentals {size / 1e6:.0f} MB / {rows:,} γραμμές"
          f" = {per_row:.0f} bytes/γραμμή ──")
    print(f"Ερευνήθηκαν {raw['ticker'].nunique()} tickers ({len(errors)} σφάλματα)\n")

    new = d[~d["tag"].isin(ALREADY_INGESTED)]
    t = (new.groupby(["group", "tag"])
         .agg(εταιρείες=("ticker", "nunique"), ζωντανές=("live", "sum"), περίοδοι=("n_periods", "sum"))
         .reset_index())
    t["MB"] = (t["περίοδοι"] * per_row / 1e6).round(1)
    print("── Κόστος ανά tag (ζωντανές = έχουν τιμή στους τελευταίους ~9 μήνες) ──")
    print(t.sort_values(["group", "ζωντανές"], ascending=[True, False]).to_string(index=False))
    print("\nΣύνολο ανά ομάδα:")
    print(t.groupby("group")["MB"].sum().to_string())

    print("\n── Οι μετοχές του ελέγχου: ποια tags έχουν, και μέχρι πότε ──")
    w = d[d["ticker"].isin(WATCH)].sort_values(["ticker", "last_end"], ascending=[True, False])
    print(w[["ticker", "tag", "n_periods", "last_end", "live"]].to_string(index=False))

    print("\n── Πόσες εταιρείες έχουν ΠΑΡΑΠΑΝΩ από ένα ζωντανό tag χρεογράφων (κίνδυνος διπλομέτρησης) ──")
    cur = d[(d["group"] == "cash_securities") & d["live"]
            & ~d["tag"].isin(["CashCashEquivalentsAndShortTermInvestments"])]
    multi = cur.groupby("ticker")["tag"].apply(list)
    multi = multi[multi.map(len) > 1]
    print(f"  {len(multi)} εταιρείες, π.χ.:")
    for k, v in multi.head(8).items():
        print(f"  {k}: {', '.join(v)}")


if __name__ == "__main__":
    engine = get_engine()
    if "--report" not in sys.argv:
        survey(engine)
    report(engine)
