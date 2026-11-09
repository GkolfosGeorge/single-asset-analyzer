"""
dev/cash_probe_202611.py — READ-ONLY: the last open questions on cash, before the ingest.

Uses data/cash_values_202611.jsonl (the previous step) and asks the SEC for three things:

  1. Does the newer tag found at DXCM fix the undercounted companies?
         DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent
     (the survey never looked for it: it replaced the older AFS tags from ~2024)
  2. Which tag holds exactly the missing money? DXCM-like:
         Yahoo 1.82 − cash 0.96 = 0.86 missing → list every tag worth ~0.86
  3. When we exceed Yahoo, is the amount inside cash? LYB-like: if the 2.09 of
     "securities" also appears as CashEquivalentsAtCarryingValue, adding it counts it twice.

It also retries Yahoo for recent companies that got no answer last time (NVDA, INTC, ...).
Nothing is written to the database.

    python dev/cash_probe_202611.py --selftest   (1 sec, no network, no database)
    python dev/cash_probe_202611.py              (~8–12 min, resumable)
    python dev/cash_probe_202611.py --report     (reprints the summary)

→ data/cash_probe_202611.jsonl
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

import cash_values_202611 as CV

SOURCE = "data/cash_values_202611.jsonl"
OUT = "data/cash_probe_202611.jsonl"
SLEEP = 0.15
NEW_CURRENT = "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent"
NEW_TOTAL = "DebtSecuritiesAvailableForSaleExcludingAccruedInterest"
RETRY_YAHOO_SINCE = "2026-01-01"
MATCH = 0.01                # "the same amount" = within 1%
MIN_AMOUNT = 1e6
MAX_MATCHES = 8
# hit-list: below or above Yahoo with today's best rule, or kept in an exception list
PROBLEM = ["DXCM", "EBAY", "VTRS", "KLAC", "CVNA", "DE", "KO", "AXON",
           "LOW", "LYB", "NWL", "RTX", "NXPI"]


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def best_current(v: dict, extra: tuple = ()) -> float:
    """
    The largest "current" securities tag, not the sum and not simply the first.
    These tags overlap as part and whole. PFE: 7.28 of debt securities are PART
    of the 10.72 of short-term investments, so 10.72 is the whole amount.
    """
    return max([v.get(t) or 0.0 for t in CV.CHAIN + list(extra)] + [0.0])


def cash_total(v: dict, extra: tuple = ()):
    _, cash = CV.first(v, CV.CASH)
    if cash is None:
        return v.get(CV.COMBINED)
    return cash + best_current(v, extra)


def matching_tags(gaap: dict, date: str, amount: float) -> list[tuple[str, float]]:
    """Every balance-sheet tag worth about `amount` on `date`."""
    if amount is None or abs(amount) < MIN_AMOUNT:
        return []
    out = []
    for tag in gaap:
        val = CV.tag_values(gaap, tag).get(date)
        if val is not None and abs(val / amount - 1) <= MATCH:
            out.append((tag, val))
    return sorted(out, key=lambda x: abs(x[1] / amount - 1))[:MAX_MATCHES]


def probe(rec: dict, gaap: dict) -> dict:
    """Adds the new tags and, for the hit-list, what holds the missing or extra money."""
    v = rec["values"]
    for tag in (NEW_CURRENT, NEW_TOTAL):
        val = CV.tag_values(gaap, tag).get(rec["date"])
        if val is not None:
            v[tag] = val
    out = {**rec, "gap_tags": [], "extra_tags": []}
    if rec["ticker"] in PROBLEM and rec.get("y_total") is not None:
        ours = cash_total(v, (NEW_CURRENT,))
        gap = rec["y_total"] - (ours or 0.0)
        out["gap"] = gap
        out["gap_tags"] = matching_tags(gaap, rec["date"], gap)
        if gap < 0:                     # above Yahoo: where else does the securities amount appear?
            amount = best_current(v, (NEW_CURRENT,))
            out["extra_tags"] = [m for m in matching_tags(gaap, rec["date"], amount)
                                 if m[0] not in CV.CHAIN + CV.TOTALS + [NEW_CURRENT, NEW_TOTAL]]
    return out


def rule_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    rules = {"σήμερα (Cash + STI)": lambda v: CV.cash_rules(v)["today"] or v.get(CV.COMBINED),
             "μεγαλύτερο Current": lambda v: cash_total(v),
             "μεγαλύτερο Current + νέο tag": lambda v: cash_total(v, (NEW_CURRENT,))}
    x = [r for r in records if r.get("y_total") and not r.get("financial")]
    for label, fn in rules.items():
        q = pd.Series([fn(r["values"]) / r["y_total"] - 1 for r in x if fn(r["values"]) is not None],
                      dtype=float)
        rows.append({"κανόνας": label, "n": len(q), "εντός_5%": int((q.abs() <= .05).sum()),
                     "πάνω_από_+5%": int((q > .05).sum()), "κάτω_από_-5%": int((q < -.05).sum())})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return "—" if x is None else f"{x / 1e9:,.3f}"


def report() -> None:
    records = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    known = sum(1 for r in records if r.get("y_total"))
    print(f"── {len(records)} εταιρείες · με Yahoo: {known} ──\n")

    print("── 1. Κανόνες έναντι Yahoo (χωρίς χρηματοοικονομικές) ──")
    print(rule_rows(records).to_string(index=False))

    print("\n── 2. Εταιρείες με το νέο tag ──")
    have = [r for r in records if r["values"].get(NEW_CURRENT) and not r.get("financial")]
    print(f"  {len(have)} εταιρείες, οι πρώτες 45:")
    for r in sorted(have, key=lambda r: r["ticker"])[:45]:
        v = r["values"]
        print(f"  {r['ticker']:6s} {r['date']}  ταμείο {bn(CV.first(v, CV.CASH)[1]):>8s}"
              f"  νέο {bn(v[NEW_CURRENT]):>8s}  παλιά {bn(best_current(v)):>8s}  Yahoo {bn(r.get('y_total')):>8s}"
              f"  με νέο {CV._dev(cash_total(v, (NEW_CURRENT,)), r.get('y_total')):>6s}")

    print("\n── 3. Λίστα ελέγχου: ποιο tag κρατά τη διαφορά ──")
    for r in sorted((r for r in records if r["ticker"] in PROBLEM), key=lambda r: r["ticker"]):
        v = r["values"]
        ours = cash_total(v, (NEW_CURRENT,))
        print(f"  {r['ticker']:6s} {r['date']}  δικό μας {bn(ours)}  Yahoo {bn(r.get('y_total'))}"
              f"  διαφορά {bn(r.get('gap'))}")
        for tag, val in r.get("gap_tags", []):
            print(f"         = διαφορά: {tag}  {bn(val)}")
        for tag, val in r.get("extra_tags", []):
            print(f"         = χρεόγραφα: {tag}  {bn(val)}")

    print("\n── 4. NVDA και όσες ξαναρωτήθηκαν στο Yahoo ──")
    retried = [r for r in records if r.get("retried")]
    print(f"  {len(retried)} ξαναρωτήθηκαν, απάντησαν {sum(1 for r in retried if r.get('y_total'))}")
    for r in sorted(retried, key=lambda r: r["ticker"]):
        if r.get("y_total") or r["ticker"] == "NVDA":
            ours = cash_total(r["values"], (NEW_CURRENT,))
            print(f"  {r['ticker']:6s} {r['date']}  δικό μας {bn(ours):>8s}  Yahoo {bn(r.get('y_total')):>8s}"
                  f"  {CV._dev(ours, r.get('y_total'))}")


# ═════════════════════════════════════════════════════════════════════════
# 3. COLLECTION
# ═════════════════════════════════════════════════════════════════════════

def collect() -> None:
    import yfinance as yf
    from db import get_engine
    from fundamentals import fetch_companyfacts
    from tag_discovery_202610 import ticker_ciks
    from yahoo_check_202611 import yahoo_symbol

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    source = [json.loads(line) for line in open(SOURCE, encoding="utf-8")]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["ticker"] for line in open(OUT, encoding="utf-8")}
    cik = dict(ticker_ciks(get_engine()).itertuples(index=False))
    rest = [r for r in source if r["ticker"] not in done and r["ticker"] in cik]
    print(f"{len(source)} εταιρείες, {len(done)} έτοιμες, {len(rest)} μένουν\n")

    t0 = time.time()
    for i, rec in enumerate(rest, 1):
        t = rec["ticker"]
        try:
            gaap = (fetch_companyfacts(int(cik[t])) or {}).get("facts", {}).get("us-gaap", {})
            retried = rec.get("y_total") is None and (rec.get("date") or "") >= RETRY_YAHOO_SINCE
            if retried:
                try:
                    time.sleep(0.5)
                    yb = CV.yahoo_balance(yf.Ticker(yahoo_symbol(t)).quarterly_balance_sheet)
                except Exception:
                    yb = CV.yahoo_balance(None)
                if yb["y_total"] is not None:
                    fresh = CV.record(t, gaap, yb)
                    fresh["financial"] = rec.get("financial")
                    rec = fresh
            out = probe(rec, gaap)
            out["retried"] = retried
        except Exception as e:
            print(f"  {t}: σφάλμα {str(e)[:80]}")
            continue
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        time.sleep(SLEEP)
        if i % 25 == 0:
            print(f"  ... {i}/{len(rest)}  ~{(time.time() - t0) / i * (len(rest) - i) / 60:.0f} λεπτά ακόμα")


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST: invented companies, no network, no database
# ═════════════════════════════════════════════════════════════════════════

def selftest() -> None:
    C, S, O, A = CV.CASH[0], CV.CHAIN[0], CV.CHAIN[5], CV.CHAIN[3]
    g = lambda **k: {t: {"units": {"USD": [{"end": "2026-09-30", "val": v, "form": "10-Q",
                                           "filed": "2026-10-30"}]}} for t, v in k.items()}

    # PFE-like: a part and its whole → the whole, once
    assert cash_total({C: 1.0, A: 7.0, O: 10.0}) == 11.0

    # the new tag counts only when asked for, and is not added to an equal old one
    v = {C: 1.0, A: 3.0, NEW_CURRENT: 3.0}
    assert cash_total(v) == 4.0 and cash_total(v, (NEW_CURRENT,)) == 4.0
    assert cash_total({C: 1.0, NEW_CURRENT: 3.0}, (NEW_CURRENT,)) == 4.0

    # no cash line: the combined line stands alone, and a footnote total is ignored
    assert cash_total({CV.COMBINED: 5.0, CV.TOTALS[0]: 3.0}) == 5.0
    assert cash_total({C: 1.0, CV.TOTALS[0]: 30.0}) == 1.0

    # DXCM-like: the missing money is found by amount, within 1%, closest first
    facts = g(**{NEW_CURRENT: 86.0e7, "SomethingClose": 85.5e7, "Unrelated": 40e7})
    assert [t for t, _ in matching_tags(facts, "2026-09-30", 86e7)] == [NEW_CURRENT, "SomethingClose"]
    assert matching_tags(facts, "2026-09-30", 100.0) == []      # a rounding gap means nothing
    assert matching_tags(facts, "2025-01-01", 86e7) == []

    # probe, below Yahoo: reports the gap
    rec = {"ticker": "DXCM", "date": "2026-09-30", "values": {C: 96e7}, "y_total": 182e7}
    p = probe(rec, g(**{NEW_CURRENT: 86e7}))
    assert p["values"][NEW_CURRENT] == 86e7 and p["gap"] == 0 and p["gap_tags"] == []
    rec = {"ticker": "DXCM", "date": "2026-09-30", "values": {C: 96e7}, "y_total": 182e7}
    p = probe(rec, g(**{"OtherAssetsCurrent": 86e7}))
    assert p["gap_tags"] == [("OtherAssetsCurrent", 86e7)] and p["extra_tags"] == []

    # probe, above Yahoo: the securities also appear as a cash-equivalent tag
    rec = {"ticker": "LYB", "date": "2026-09-30", "values": {C: 32e8, CV.CHAIN[1]: 21e8}, "y_total": 32e8}
    p = probe(rec, g(**{CV.CHAIN[1]: 21e8, "CashEquivalentsAtCarryingValue": 21e8}))
    assert p["extra_tags"] == [("CashEquivalentsAtCarryingValue", 21e8)]

    # a company outside the hit-list is not searched
    rec = {"ticker": "ZZZ", "date": "2026-09-30", "values": {C: 1e9}, "y_total": 2e9}
    assert probe(rec, g(Thing=1e9))["gap_tags"] == []

    # rule table: financials stay out
    recs = [{"values": {C: 1.0, O: 1.0}, "y_total": 2.0, "financial": False},
            {"values": {C: 9.0}, "y_total": 2.0, "financial": True}]
    t = rule_rows(recs)
    assert list(t["n"]) == [1, 1, 1] and list(t["εντός_5%"]) == [0, 1, 1]

    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report()
    else:
        collect()
        report()
