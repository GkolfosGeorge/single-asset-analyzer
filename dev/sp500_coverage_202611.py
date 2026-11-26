"""
dev/sp500_coverage_202611.py — READ-ONLY: is EVERY company of today's S&P 500
visible and up to date for the card? (17/11/2026: the mail scans tiers 4–5 and
may scan the whole index later.)

A company reaches the card through 4 layers, and is lost at the FIRST that fails:
    1. SEC     — the weekly ingest finds its CIK by ticker (today's SEC list)
    2. data    — `fundamentals` has rows under that ticker (weekly ingest, Mondays)
    3. visible — `companies` has the ticker: the analyzer's universe is
                 fundamentals JOIN companies (valuation/pe.load_universe)
    4. table   — it is in peer_data/peer_yields.csv with at least one yield
Example (suspected): VMRK has 1,928 rows since 24/08, but no `companies` row
→ lost at 3, invisible to the card and to the peer table.
It also flags data older than 270 days (the card then shows nothing).

Same S&P 500 list as the weekly ingest (ticker_provider.get_tickers("sp500")).

    python dev/sp500_coverage_202611.py --selftest   (no network, no database)
    python dev/sp500_coverage_202611.py              (~10 sec)
→ data/sp500_coverage_202611.csv
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

TABLE = "peer_data/peer_yields.csv"
MISSING = "data/missing_yields_202611.csv"
OUT = "data/sp500_coverage_202611.csv"
STALE_DAYS = 270
YIELDS = ["earnings_yield", "ebit_yield", "fcf_yield"]
LAYERS = ["1. χωρίς CIK στη SEC", "2. καθόλου στοιχεία στη βάση", "3. στοιχεία, αλλά όχι στον companies",
          "4α. ορατή, αλλά όχι στον πίνακα κλάδου", "4β. στον πίνακα χωρίς καμία απόδοση", "ok"]


def where_lost(r: dict) -> str:
    if not r["sec_cik"]:
        return LAYERS[0]
    if not r["rows"]:
        return LAYERS[1]
    if not r["in_companies"]:
        return LAYERS[2]
    if not r["in_table"]:
        return LAYERS[3]
    if not r["n_yields"]:
        return LAYERS[4]
    return LAYERS[5]


def selftest() -> None:
    base = {"sec_cik": 1, "rows": 10, "in_companies": True, "in_table": True, "n_yields": 2}
    assert where_lost(base) == "ok"
    assert where_lost({**base, "in_companies": False}) == LAYERS[2]
    assert where_lost({**base, "sec_cik": None, "rows": 0}) == LAYERS[0]      # the first failure wins
    assert where_lost({**base, "n_yields": 0}) == LAYERS[4]
    print("selftest: όλα τα τεστ πέρασαν ✓")


def main() -> None:
    from sqlalchemy import text
    from db import get_engine
    from ticker_provider import _load_sec_ticker_map, get_tickers

    sp = get_tickers("sp500", folder_path="cache")
    sec = _load_sec_ticker_map("cache")
    eng = get_engine()
    with eng.connect() as conn:
        f = pd.DataFrame(conn.execute(text("""
            SELECT ticker, COUNT(*), MAX(filed_date), MAX(period_end), MAX(inserted_at)
            FROM fundamentals WHERE ticker = ANY(:t) GROUP BY ticker"""), {"t": sp}).fetchall(),
            columns=["ticker", "rows", "last_filed", "last_period", "last_inserted"]).set_index("ticker")
        comp = set(r[0] for r in conn.execute(text("SELECT ticker FROM companies WHERE ticker = ANY(:t)"),
                                              {"t": sp}).fetchall())
    table = pd.read_csv(TABLE).set_index("ticker") if os.path.exists(TABLE) else pd.DataFrame(columns=YIELDS)
    as_of = pd.to_datetime(table["as_of"]).max() if "as_of" in table else pd.Timestamp.today()
    causes = pd.read_csv(MISSING).set_index("ticker") if os.path.exists(MISSING) else pd.DataFrame()

    out = []
    for t in sp:
        r = {"ticker": t, "sec_cik": sec.get(t.upper()),
             "rows": int(f.loc[t, "rows"]) if t in f.index else 0,
             "last_filed": f.loc[t, "last_filed"] if t in f.index else None,
             "last_period": f.loc[t, "last_period"] if t in f.index else None,
             "last_inserted": f.loc[t, "last_inserted"] if t in f.index else None,
             "in_companies": t in comp, "in_table": t in table.index,
             "n_yields": int(table.loc[t, YIELDS].notna().sum()) if t in table.index else 0}
        r["stale"] = bool(r["last_period"] is not None
                          and (as_of - pd.Timestamp(r["last_period"])).days > STALE_DAYS)
        r["lost_at"] = where_lost(r)
        if t in causes.index:
            r["P/E αιτία"] = causes.loc[t].get("P/E αιτία")
        out.append(r)
    df = pd.DataFrame(out)

    print(f"S&P 500 σήμερα: {len(df)} εταιρείες (ίδια λίστα με το weekly ingest)\n")
    for layer in LAYERS:
        sub = df[df["lost_at"] == layer]
        tail = "" if layer == "ok" else "  " + ", ".join(sub["ticker"])
        print(f"{len(sub):4d}  {layer}{tail}")
    in_tab = df[df["lost_at"] == "ok"]
    print(f"\nΣτον πίνακα με απόδοση: {len(in_tab)}/{len(df)} = {len(in_tab) / len(df):.1%}")
    stale = df[df["stale"]]
    if len(stale):
        print(f"Παλιά στοιχεία (τελευταία περίοδος > {STALE_DAYS} μέρες): {len(stale)} · "
              + ", ".join(f"{r.ticker} ({pd.Timestamp(r.last_period):%Y-%m})" for r in stale.itertuples()))
    lost3 = df[df["lost_at"] == LAYERS[2]]
    if len(lost3):
        print(f"\nΑόρατες (στοιχεία χωρίς companies): {len(lost3)} · πρώτη εγγραφή / τελευταία κατάθεση:")
        for r in lost3.itertuples():
            print(f"   {r.ticker:6s} CIK {r.sec_cik} · {r.rows:,} γραμμές · κατάθεση ως {r.last_filed}"
                  f" · γράφτηκε ως {pd.Timestamp(r.last_inserted):%Y-%m-%d}")
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
