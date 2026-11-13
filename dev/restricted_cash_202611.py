"""
dev/restricted_cash_202611.py — READ-ONLY: step 3β. Can the tag
    CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents   ("R")
stand in for our cash tag
    CashAndCashEquivalentsAtCarryingValue                           ("C")
when C is missing? R also holds RESTRICTED cash (money the company can't use:
escrow, client funds, collateral), so it is only safe where that part is small.
Nothing is written to the database; nothing is ingested.

How: on every balance-sheet date where a company reports BOTH tags, the extra
share = R / C − 1. Example:
    C 10.0, R 10.1  → +1%   (fine: R ≈ cash)
    C  2.0, R  9.0  → +350% (ADP-like: client funds; R must never be used)
Answers:
  1. how often the extra share is under 1% / 2% / 5% (per company: its WORST date
     in the last 8 dates with both)
  2. the companies whose C stopped but R goes on (PG, HSY, LULU...): what was
     their share back when they still reported both? That is the real test.
  3. what the ingest would cost (rows, MB), for R, its disposal-group variant and "Cash"

    python dev/restricted_cash_202611.py --selftest   (1 sec, no network, no database)
    python dev/restricted_cash_202611.py              (~20–30 min, resumable, asks SEC)
    python dev/restricted_cash_202611.py --report     (reprints the summary)

→ data/restricted_cash_202611.jsonl
"""

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from cash_values_202611 import tag_values

OUT = "data/restricted_cash_202611.jsonl"
SLEEP = 0.15
C = "CashAndCashEquivalentsAtCarryingValue"
R = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
R2 = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations"
CASH = "Cash"
TAGS = [C, R, R2, CASH]
LAST = 8
STALE_DAYS = 270
BYTES_PER_ROW = 294      # measured on 04/11/2026 (tag_discovery_202611)
BANDS = (0.01, 0.02, 0.05)


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def shares(c: dict, r: dict) -> list[tuple[str, float]]:
    """(date, R/C − 1) on the dates with both, oldest first. C ≤ 0 says nothing."""
    return [(d, r[d] / c[d] - 1) for d in sorted(set(c) & set(r)) if c[d] > 0]


def record(ticker: str, gaap: dict, today: pd.Timestamp) -> dict:
    vals = {t: tag_values(gaap, t) for t in TAGS}
    last = {t: max(v) if v else None for t, v in vals.items()}
    stale = lambda d: d is None or (today - pd.Timestamp(d)).days > STALE_DAYS
    both = shares(vals[C], vals[R])
    recent = both[-LAST:]
    return {"ticker": ticker,
            "last": last,
            "n": {t: len(v) for t, v in vals.items()},
            "worst_recent": max((s for _, s in recent), default=None),
            "worst_all": max((s for _, s in both), default=None),
            "n_both": len(both),
            "gap": stale(last[C]) and not stale(last[R]),   # C stopped, R goes on
            "last_both": both[-1][0] if both else None}


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def report() -> None:
    recs = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    d = pd.DataFrame(recs)
    print(f"── {len(d)} εταιρείες ──\n")

    print(f"── 1. Εταιρείες με ΚΑΙ τα δύο tags: πόσο μεγαλύτερο είναι το R (χειρότερη από τις {LAST} τελευταίες ημ/νίες) ──")
    w = d["worst_recent"].dropna()
    print(f"  {len(w)} εταιρείες · " + " · ".join(f"≤ {int(b * 100)}%: {int((w <= b).sum())}" for b in BANDS)
          + f" · > 5%: {int((w > .05).sum())} · διάμεσος {w.median() * 100:.2f}%")
    big = d[d["worst_recent"] > .05].sort_values("worst_recent", ascending=False)
    print("  Οι μεγαλύτερες (εδώ το R ΔΕΝ είναι ταμείο):")
    for r in big.head(20).itertuples():
        print(f"    {r.ticker:6s} +{r.worst_recent * 100:,.0f}%")

    print("\n── 2. Το C σταμάτησε, το R συνεχίζει (αυτές θα σώσει το ingest) ──")
    g = d[d["gap"]].sort_values("ticker")
    print(f"  {len(g)} εταιρείες")
    for r in g.itertuples():
        hist = (f"όταν είχαν και τα δύο (ως {r.last_both}, {r.n_both} ημ/νίες): χειρότερη +{r.worst_all * 100:.1f}%"
                if r.n_both else "ποτέ και τα δύο μαζί: δεν ελέγχεται")
        print(f"    {r.ticker:6s} C ως {r.last[C] or '—'} · R ως {r.last[R]} · {hist}")
    ok = g["worst_all"].dropna()
    print(f"  Από όσες ελέγχονται: ≤ 2% {int((ok <= .02).sum())}/{len(ok)} · > 5% {int((ok > .05).sum())}")

    print("\n── 3. Κόστος του ingest ──")
    names = {R: "R (ταμείο + δεσμευμένα)", R2: "R με disposal group", CASH: "Cash"}
    for t in (R, R2, CASH):
        n = d["n"].map(lambda x: x.get(t, 0))
        print(f"  {names[t]:26s} {int((n > 0).sum()):4d} εταιρείες · {int(n.sum()):6d} περίοδοι"
              f" ≈ {n.sum() * BYTES_PER_ROW / 1e6:.1f} MB")
    for t in (R2, CASH):
        alone = d[d["last"].map(lambda x: x.get(t) is not None)
                  & d["last"].map(lambda x: x.get(C) is None and x.get(R) is None)]
        print(f"  μόνο με {names[t]} (χωρίς C και R): {len(alone)} ({', '.join(sorted(alone['ticker'])[:20])})")


# ═════════════════════════════════════════════════════════════════════════
# 3. COLLECTION
# ═════════════════════════════════════════════════════════════════════════

def collect() -> None:
    from db import get_engine
    from fundamentals import fetch_companyfacts
    from tag_discovery_202610 import ticker_ciks

    cik = ticker_ciks(get_engine())
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["ticker"] for line in open(OUT, encoding="utf-8")}
    rest = [(t, c) for t, c in cik.itertuples(index=False) if t not in done]
    print(f"{len(cik)} εταιρείες, {len(done)} έτοιμες, {len(rest)} μένουν\n")
    today = pd.Timestamp.today().normalize()
    t0 = time.time()
    for i, (t, c) in enumerate(rest, 1):
        try:
            gaap = (fetch_companyfacts(int(c)) or {}).get("facts", {}).get("us-gaap", {})
            rec = record(t, gaap, today)
        except Exception as e:
            print(f"  {t}: σφάλμα {str(e)[:80]}")
            continue
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(SLEEP)
        if i % 50 == 0:
            print(f"  ... {i}/{len(rest)}  ~{(time.time() - t0) / i * (len(rest) - i) / 60:.0f} λεπτά ακόμα")


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST
# ═════════════════════════════════════════════════════════════════════════

def _gaap(series: dict) -> dict:
    """{tag: {date: value}} → companyfacts-like, one 10-Q per date."""
    return {t: {"units": {"USD": [{"end": d, "val": v, "form": "10-Q", "filed": d} for d, v in dv.items()]}}
            for t, dv in series.items()}


def selftest() -> None:
    today = pd.Timestamp("2026-11-11")
    assert shares({"a": 10.0, "b": 0.0}, {"a": 10.1, "b": 1.0, "c": 5.0})[0][0] == "a"
    assert len(shares({"a": 10.0, "b": 0.0}, {"a": 10.1, "b": 1.0})) == 1        # C = 0 says nothing

    # PG-like: both until 2019 (+1%), then only R → gap, historically safe
    g = _gaap({C: {"2019-03-31": 10.0, "2019-06-30": 10.0},
               R: {"2019-03-31": 10.1, "2019-06-30": 10.05, "2026-09-30": 15.4}})
    r = record("PG", g, today)
    assert r["gap"] and abs(r["worst_all"] - 0.01) < 1e-9 and r["n_both"] == 2 and r["last_both"] == "2019-06-30"

    # ADP-like: client funds → R is 4.5× cash; never a gap because C is alive
    g = _gaap({C: {"2026-09-30": 2.0}, R: {"2026-09-30": 9.0}})
    r = record("ADP", g, today)
    assert not r["gap"] and abs(r["worst_recent"] - 3.5) < 1e-9

    # only the last 8 dates count for "recent", all of them for "all"
    dates = [f"20{y:02d}-12-31" for y in range(10, 26)]
    g = _gaap({C: {d: 10.0 for d in dates}, R: {d: (20.0 if d == dates[0] else 10.0) for d in dates}})
    r = record("X", g, today)
    assert r["worst_recent"] == 0.0 and r["worst_all"] == 1.0

    # costs count periods per tag
    assert record("X", g, today)["n"] == {C: 16, R: 16, R2: 0, CASH: 0}
    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report()
    else:
        collect()
        report()
