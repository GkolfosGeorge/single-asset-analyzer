"""
dev/rename_dryrun_202611.py — READ-ONLY: step 2 of the ticker clean-up (17/11/2026).
8 live companies are in our database only under their OLD ticker, so the
ingest finds no CIK for them any more (the SEC lists the new one) and their
history froze:
    EQR→VMRK  FBHS→FBIN  GPS→GAP  IAC→PPLI  IGT→BRSL  NCR→VYX  PSKY→SKYD  RYI→RYZ
Nothing is written anywhere. For each pair it measures the two ways to fix it:

  A. RENAME: relabel the old ticker's rows to the new ticker (UPDATE, same CIK).
     No new space; then the normal ingest adds only the new filings.
  B. FRESH INGEST under the new ticker: the SEC's whole history comes again,
     next to the old rows → about the same number of rows TWICE.
     Example: FBHS has 9,000 rows → B adds ~9,000 more (~X MB), A adds 0.

It also checks what each option needs to be safe:
  - same CIK: our old ticker's CIK == the SEC's CIK for the new ticker
  - names alike (valuation/universe.similar): our title vs the SEC title
  - the new ticker is NOT already in `companies` / `fundamentals` (no clash)
  - Yahoo: does the new ticker have a price, and since when? If Yahoo moved the
    old history to the new ticker (FB→META did), the 10-year card keeps working.

    python dev/rename_dryrun_202611.py --selftest   (no network, no database)
    python dev/rename_dryrun_202611.py              (~1 min: 8 SEC downloads + Yahoo)
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd

PAIRS = [("EQR", "VMRK"), ("FBHS", "FBIN"), ("GPS", "GAP"), ("IAC", "PPLI"),
         ("IGT", "BRSL"), ("NCR", "VYX"), ("PSKY", "SKYD"), ("RYI", "RYZ")]
SLEEP = 0.3
NEON_LIMIT_MB = 500


def mb(rows: int, bytes_per_row: float) -> float:
    return rows * bytes_per_row / 1e6


def verdict(r: dict) -> str:
    """What blocks a rename, in plain words ('' = nothing)."""
    bad = []
    if not r["same_cik"]:
        bad.append("άλλο CIK")
    if not r["names_alike"]:
        bad.append("άλλο όνομα")
    if r["new_in_companies"] or r["new_rows"]:
        bad.append("το νέο σύμβολο υπάρχει ήδη στη βάση")
    if not r["new_has_price"]:
        bad.append("χωρίς τιμή Yahoo στο νέο σύμβολο")
    return ", ".join(bad)


def selftest() -> None:
    assert abs(mb(10_000, 250) - 2.5) < 1e-9
    ok = {"same_cik": True, "names_alike": True, "new_in_companies": False, "new_rows": 0,
          "new_has_price": True}
    assert verdict(ok) == ""
    assert verdict({**ok, "same_cik": False, "new_rows": 5}) == "άλλο CIK, το νέο σύμβολο υπάρχει ήδη στη βάση"
    print("selftest: όλα τα τεστ πέρασαν ✓")


def main() -> None:
    from sqlalchemy import text
    from db import get_engine
    from fundamentals import fetch_companyfacts, parse_companyfacts
    from valuation.ttm import load_prices
    from valuation.universe import load_sec, similar

    eng = get_engine()
    olds, news = [o for o, _ in PAIRS], [n for _, n in PAIRS]
    with eng.connect() as conn:
        total_mb = conn.execute(text("SELECT pg_database_size(current_database())")).scalar() / 1e6
        rel = conn.execute(text("SELECT pg_total_relation_size('fundamentals')")).scalar()
        n_all = conn.execute(text("SELECT reltuples::bigint FROM pg_class WHERE relname = 'fundamentals'")).scalar()
        comp = pd.DataFrame(conn.execute(text("SELECT ticker, cik, title FROM companies WHERE ticker = ANY(:t)"),
                                         {"t": olds + news}).fetchall(), columns=["ticker", "cik", "title"])
        rows = dict(conn.execute(text("""SELECT ticker, COUNT(*) FROM fundamentals WHERE ticker = ANY(:t)
                                         GROUP BY ticker"""), {"t": olds + news}).fetchall())
        last = dict(conn.execute(text("""SELECT ticker, MAX(filed_date) FROM fundamentals WHERE ticker = ANY(:t)
                                         GROUP BY ticker"""), {"t": olds}).fetchall())
    bpr = rel / max(n_all, 1)
    print(f"Βάση: {total_mb:.0f} MB από {NEON_LIMIT_MB} · fundamentals {rel / 1e6:.0f} MB για ~{n_all:,} γραμμές"
          f" → ~{bpr:.0f} bytes/γραμμή (μαζί με τα indexes)\n")

    sec = load_sec()
    sec_by_t = sec.drop_duplicates("ticker").set_index("ticker") if sec is not None else pd.DataFrame()
    end = pd.Timestamp.today().normalize()
    close, _ = load_prices(news, (end - pd.DateOffset(years=12)).date(), (end + pd.Timedelta(days=1)).date())

    out = []
    for old, new in PAIRS:
        c = comp.set_index("ticker")
        our_cik = int(c.loc[old, "cik"]) if old in c.index and pd.notna(c.loc[old, "cik"]) else None
        our_title = c.loc[old, "title"] if old in c.index else None
        sec_cik = int(sec_by_t.loc[new, "cik"]) if new in sec_by_t.index else None
        sec_title = sec_by_t.loc[new, "title"] if new in sec_by_t.index else None
        fresh = None
        if sec_cik:
            try:
                facts = fetch_companyfacts(sec_cik)
                fresh = len(parse_companyfacts(new, sec_cik, facts)) if facts else 0
            except Exception as e:
                print(f"  {new}: companyfacts {type(e).__name__}: {e}")
            time.sleep(SLEEP)
        px = close[new].dropna() if new in close.columns else pd.Series(dtype=float)
        r = {"old": old, "new": new, "our_cik": our_cik, "sec_cik": sec_cik,
             "same_cik": our_cik is not None and our_cik == sec_cik,
             "our_title": our_title, "sec_title": sec_title,
             "names_alike": similar(our_title, sec_title),
             "new_in_companies": new in c.index, "new_rows": int(rows.get(new, 0)),
             "old_rows": int(rows.get(old, 0)), "old_last_filed": last.get(old),
             "fresh_rows": fresh, "new_has_price": len(px) > 0,
             "price_since": px.index[0].date() if len(px) else None}
        r["blocks"] = verdict(r)
        out.append(r)
    df = pd.DataFrame(out)

    for _, r in df.iterrows():
        fresh = "—" if r["fresh_rows"] is None else f"{r['fresh_rows']:,}"
        print(f"══ {r['old']} → {r['new']} ══")
        print(f"  CIK: δικό μας {r['our_cik']} · SEC {r['sec_cik']} → {'ίδιο ✓' if r['same_cik'] else 'ΑΛΛΟ ✗'}")
        print(f"  όνομα: «{r['our_title']}» · SEC «{r['sec_title']}» → {'μοιάζει ✓' if r['names_alike'] else 'ΔΕΝ μοιάζει ✗'}")
        print(f"  γραμμές: παλιό σύμβολο {r['old_rows']:,} (τελευταία κατάθεση {r['old_last_filed']})"
              f" · νέο σύμβολο ήδη στη βάση {r['new_rows']:,} · πλήρες ingest του νέου {fresh}")
        print(f"  Yahoo {r['new']}: " + (f"τιμές από {r['price_since']}" if r["new_has_price"] else "ΚΑΜΙΑ τιμή"))
        print(f"  → μετονομασία: {'εμπόδια: ' + r['blocks'] if r['blocks'] else 'χωρίς εμπόδια'}\n")

    fresh_total = int(df["fresh_rows"].fillna(0).sum())
    old_total = int(df["old_rows"].sum())
    print("── Σύνολο ──")
    print(f"  A. Μετονομασία: {old_total:,} γραμμές αλλάζουν σύμβολο · νέος χώρος ~0 MB"
          f" (+ μόνο οι καταθέσεις μετά την αλλαγή συμβόλου)")
    print(f"  B. Πλήρες ingest: +{fresh_total:,} γραμμές · ~{mb(fresh_total, bpr):.1f} MB"
          f" → βάση ~{total_mb + mb(fresh_total, bpr):.0f} από {NEON_LIMIT_MB} MB")
    print(f"  Χωρίς εμπόδια για μετονομασία: {int((df['blocks'] == '').sum())}/{len(df)}")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/rename_dryrun_202611.csv", index=False)
    print("→ data/rename_dryrun_202611.csv")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
