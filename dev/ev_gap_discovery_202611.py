"""
dev/ev_gap_discovery_202611.py — READ-ONLY: step 3 of the roadmap. For the 70
companies whose EV/EBIT is missing because DEBT (40) or CASH (30) is unknown
(dev/ebit_gap_202611.py, 11/11/2026): where is their debt or cash in the SEC data?
Nothing is written to the database; nothing is ingested.

For each company it asks the SEC (all us-gaap tags) and Yahoo (same balance-sheet date):
  1. our tags: when did each tag we read last have a value?
  2. Yahoo's amount (debt without leases, or cash + short-term investments)
  3. which SEC tags (or pairs of tags) are worth that amount, within 2%
  Example, ORCL-like: our tags stopped in 2024, Yahoo debt 105.0,
     SEC: LongTermNotesPayable 97.0 + NotesPayableCurrent 8.0 = 105.0  ← found
  4. "no debt": Yahoo 0 (or none) and no debt-like tag alive → paid off, not unknown
Yahoo is the COMPARATOR, not the truth: a match is a lead, read the filing to confirm.

    python dev/ev_gap_discovery_202611.py --selftest   (1 sec, no network, no database)
    python dev/ev_gap_discovery_202611.py              (~3–5 min, resumable)
    python dev/ev_gap_discovery_202611.py --report     (reprints the summary)

→ data/ev_gap_discovery_202611.jsonl
"""

import json
import logging
import os
import re
import sys
import time
from itertools import combinations

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from cash_values_202611 import nearest_date, tag_values
from valuation import ev as E

OUT = "data/ev_gap_discovery_202611.jsonl"
SLEEP = 0.5
MATCH = 0.02
ALIVE_DAYS = 270
TOP = 6

DEBT_GAP = ["AN", "ANET", "ANF", "CAL", "CAT", "CMG", "CPRT", "DLR", "DXCM", "ED", "EME", "EXR", "FFIV",
            "GM", "GME", "HST", "ILMN", "INCY", "IPGP", "KIM", "LKQ", "MNST", "NC", "NYT", "OMC", "ORCL",
            "P", "PD", "PLTR", "RHI", "SIG", "SLG", "SMCI", "TER", "TMUS", "TYL", "ULTA", "URBN", "VRSN", "WSM"]
CASH_GAP = ["AAL", "ALLE", "AM", "AR", "BWA", "CF", "CSX", "DHR", "DOV", "EL", "FDXF", "FLS", "GDDY", "GEV",
            "HAS", "HII", "HSY", "LULU", "MMI", "MMM", "PG", "PODD", "POOL", "REG", "SBUX", "SJM", "SUN",
            "TT", "WAB", "WMB"]

# what counts as a debt-like or cash-like tag (balance-sheet values only)
DEBT_RE = re.compile(r"(Debt|Notes?Payable|Borrowing|CommercialPaper|LineOfCredit|SeniorNotes|LoansPayable)")
DEBT_SKIP = re.compile(r"(FairValue|Interest|Rate|Percentage|Maturit|Issuance|Repayment|Proceeds|Payments|"
                       r"Amortization|Discount|Premium|Covenant|Securities|Investment|Receivable|Derivative|"
                       r"Extinguishment|Unamortized|Increase|Decrease|Capacity|Remaining|Guarantee)")
CASH_RE = re.compile(r"^Cash|CashEquivalents")
CASH_SKIP = re.compile(r"(Period|Increase|Decrease|Acquired|Paid|Received|Proceeds|Payments|Flow|Dividend|"
                       r"^Restricted|Collateral|Segregated|Settlement|FairValue|Disposed|Divested)")


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def candidates(gaap: dict, kind: str) -> list[str]:
    keep, skip = (DEBT_RE, DEBT_SKIP) if kind == "debt" else (CASH_RE, CASH_SKIP)
    return sorted(t for t in gaap if keep.search(t) and not skip.search(t))


def last_dates(gaap: dict, tags: list[str]) -> dict[str, str]:
    out = {}
    for t in tags:
        v = tag_values(gaap, t)
        if v:
            out[t] = max(v)
    return out


def alive(gaap: dict, tags: list[str], today: pd.Timestamp) -> list[str]:
    return [t for t, d in last_dates(gaap, tags).items() if (today - pd.Timestamp(d)).days <= ALIVE_DAYS]


def matches(values: dict[str, float], target: float) -> tuple[list, list]:
    """Single tags and pairs of tags worth `target` (within MATCH), closest first."""
    if not target:
        return [], []
    singles = sorted(((t, v) for t, v in values.items() if v and abs(v / target - 1) <= MATCH),
                     key=lambda x: abs(x[1] / target - 1))
    pairs = []
    pos = [(t, v) for t, v in values.items() if v and v > 0]
    for (a, va), (b, vb) in combinations(pos, 2):
        if abs((va + vb) / target - 1) <= MATCH:
            pairs.append((a, b, va + vb))
    pairs.sort(key=lambda x: abs(x[2] / target - 1))
    return singles[:TOP], pairs[:TOP]


def yahoo_amounts(qbs) -> dict:
    """Newest Yahoo date: debt without leases, cash + short-term investments."""
    out = {"y_date": None, "y_debt": None, "y_cash": None}
    if qbs is None or getattr(qbs, "empty", True):
        return out
    rows = {"cash": "Cash Cash Equivalents And Short Term Investments",
            "ltd": "Long Term Debt", "cur": "Current Debt"}
    have = [c for c in qbs.columns if any(n in qbs.index and pd.notna(qbs.loc[n, c]) for n in rows.values())]
    if not have:
        return out
    d = max(have)
    get = lambda n: float(qbs.loc[n, d]) if n in qbs.index and pd.notna(qbs.loc[n, d]) else None
    debt_parts = [x for x in (get(rows["ltd"]), get(rows["cur"])) if x is not None]
    total_debt = get("Total Debt")
    out["y_date"] = pd.Timestamp(d).date().isoformat()
    out["y_cash"] = get(rows["cash"])
    # Yahoo leaves both lines out for a company with no debt; "Total Debt" then holds only leases
    out["y_debt"] = sum(debt_parts) if debt_parts else (0.0 if total_debt is not None else None)
    return out


def examine(ticker: str, kind: str, gaap: dict, yb: dict, today: pd.Timestamp) -> dict:
    ours = E.DEBT_METRICS + E.CONV_METRICS if kind == "debt" else [E.CASH, E.COMBINED]
    cands = candidates(gaap, kind)
    target = yb["y_debt"] if kind == "debt" else yb["y_cash"]
    sec_dates = sorted({d for t in cands for d in tag_values(gaap, t)})
    date = nearest_date(sec_dates, yb["y_date"]) if yb["y_date"] else (sec_dates[-1] if sec_dates else None)
    values = {t: tag_values(gaap, t).get(date) for t in cands} if date else {}
    values = {t: v for t, v in values.items() if v is not None}
    singles, pairs = matches(values, target)
    alive_now = alive(gaap, cands, today)
    if kind == "debt" and target == 0 and not [t for t in alive_now if values.get(t)]:
        verdict = "χωρίς χρέος"
    elif singles or pairs:
        verdict = "βρέθηκε tag"
    elif target is None:
        verdict = "χωρίς Yahoo"
    else:
        verdict = "δεν βρέθηκε"
    return {"ticker": ticker, "kind": kind, "date": date, "target": target, "y_date": yb["y_date"],
            "ours_last": last_dates(gaap, ours), "singles": singles, "pairs": pairs,
            "alive": alive_now, "verdict": verdict}


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return "—" if x is None else f"{x / 1e9:,.2f}"


def report() -> None:
    recs = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    for kind, label in (("debt", "ΧΡΕΟΣ"), ("cash", "ΤΑΜΕΙΟ")):
        r = [x for x in recs if x["kind"] == kind]
        print(f"\n══ {label}: {len(r)} εταιρείες ══")
        print("  " + " · ".join(f"{k} {v}" for k, v in
                                pd.Series([x["verdict"] for x in r], dtype=object).value_counts().items()))
        for x in sorted(r, key=lambda x: (x["verdict"], x["ticker"])):
            ours = ", ".join(f"{t}→{d[:7]}" for t, d in sorted(x["ours_last"].items(), key=lambda i: i[1])[-3:])
            print(f"\n  {x['ticker']:5s} [{x['verdict']}]  Yahoo {bn(x['target'])} ({x['y_date']})"
                  f"  · δικά μας tags ως: {ours or 'κανένα'}")
            for t, v in x["singles"][:3]:
                print(f"        = {t}  {bn(v)}")
            for a, b, v in x["pairs"][:2]:
                print(f"        = {a} + {b}  {bn(v)}")
        # which tags would help the most companies
        c = pd.Series([t for x in r for t, _ in x["singles"]] +
                      [t for x in r for a, b, _ in x["pairs"] for t in (a, b)], dtype=object)
        print(f"\n  Τα tags που ταιριάζουν στις περισσότερες εταιρείες ({label.lower()}):")
        print(c.value_counts().head(15).to_string() if len(c) else "  κανένα")


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
    todo = [(t, "debt") for t in DEBT_GAP] + [(t, "cash") for t in CASH_GAP]
    done = set()
    if os.path.exists(OUT):
        done = {(json.loads(line)["ticker"], json.loads(line)["kind"]) for line in open(OUT, encoding="utf-8")}
    cik = dict(ticker_ciks(get_engine()).itertuples(index=False))
    rest = [x for x in todo if x not in done]
    print(f"{len(todo)} εταιρείες, {len(done)} έτοιμες, {len(rest)} μένουν\n")
    today = pd.Timestamp.today().normalize()
    for i, (t, kind) in enumerate(rest, 1):
        if t not in cik:
            print(f"  {t}: χωρίς CIK στη βάση")
            continue
        try:
            gaap = (fetch_companyfacts(int(cik[t])) or {}).get("facts", {}).get("us-gaap", {})
            try:
                yb = yahoo_amounts(yf.Ticker(yahoo_symbol(t)).quarterly_balance_sheet)
            except Exception:
                yb = yahoo_amounts(None)
            rec = examine(t, kind, gaap, yb, today)
        except Exception as e:
            print(f"  {t}: σφάλμα {str(e)[:80]}")
            continue
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(SLEEP)
        if i % 10 == 0:
            print(f"  ... {i}/{len(rest)}")


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST
# ═════════════════════════════════════════════════════════════════════════

def _gaap(values: dict, end: str = "2026-09-30") -> dict:
    return {t: {"units": {"USD": [{"end": end, "val": v, "form": "10-Q", "filed": "2026-10-30"}]}}
            for t, v in values.items()}


def selftest() -> None:
    today = pd.Timestamp("2026-11-11")
    # ORCL-like: our tag stopped, two new tags add up to Yahoo's debt
    g = {**_gaap({"LongTermNotesPayable": 97e9, "NotesPayableCurrent": 8e9, "DebtInstrumentFaceAmount": 5e9,
                  "LongTermDebtFairValue": 105e9}),
         **_gaap({"LongTermDebtNoncurrent": 80e9}, end="2023-05-31")}
    r = examine("X", "debt", g, {"y_date": "2026-09-30", "y_debt": 105e9, "y_cash": None}, today)
    assert r["verdict"] == "βρέθηκε tag"
    assert [p[:2] for p in r["pairs"]][0] == ("LongTermNotesPayable", "NotesPayableCurrent"), r["pairs"]
    assert "LongTermDebtFairValue" not in str(r)                 # fair value is not the carrying amount
    assert r["ours_last"] == {"LongTermDebtNoncurrent": "2023-05-31"}

    # PLTR-like: Yahoo shows no debt lines, nothing debt-like alive → paid off
    r = examine("X", "debt", _gaap({"LongTermDebtNoncurrent": 2e9}, end="2021-12-31"),
                {"y_date": "2026-09-30", "y_debt": 0.0, "y_cash": None}, today)
    assert r["verdict"] == "χωρίς χρέος"

    # PG-like: cash in the restricted-inclusive tag, not in ours
    r = examine("X", "cash", _gaap({"CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": 9.6e9,
                                    "CashAndCashEquivalentsPeriodIncreaseDecrease": 1e9}),
                {"y_date": "2026-09-30", "y_debt": None, "y_cash": 9.5e9}, today)
    assert r["singles"][0][0] == "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    assert "PeriodIncrease" not in str(r["singles"])

    # Yahoo amounts: newest date with data; no debt lines but a Total Debt (leases) → 0
    q = pd.DataFrame({pd.Timestamp("2026-09-30"): [5.0, None, None, 1.0],
                      pd.Timestamp("2026-06-30"): [4.0, 8.0, 1.0, 9.0]},
                     index=["Cash Cash Equivalents And Short Term Investments", "Long Term Debt",
                            "Current Debt", "Total Debt"])
    y = yahoo_amounts(q)
    assert y == {"y_date": "2026-09-30", "y_debt": 0.0, "y_cash": 5.0}, y
    assert yahoo_amounts(None)["y_date"] is None
    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report()
    else:
        collect()
        report()
