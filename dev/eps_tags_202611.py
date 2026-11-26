"""
dev/eps_tags_202611.py — READ-ONLY: what does each filing REALLY contain? (20/11/2026)
Last companies of the S&P 500 without a yield, several of them in the mail's
tiers 4–5 (V, BRK-B, XOM, C). dev/filing_gaps_202611.py found two things:
  - EPS diluted missing in EVERY filing (V, HSY, STZ, BRK-B, ERIE, KKR, ARES, BKR, KIM, REG)
  - recent filings with none of EPS / CFO / EBIT at all (C's 2026 10-Qs, TAP Q1 2026,
    KIM Q3, REG Q2–Q3, FRT Q2–Q3, XOM)
Nothing is written anywhere. For the latest 10-K and the last 3 10-Qs of each company:

  1. is the filing in companyfacts at all?  (facts with its accession number, any tag)
     0 → the SEC has not put it in companyfacts: no tag of ours can help
  2. which taxonomies it uses (us-gaap, dei, ifrs-full, srt, a company's own)
  3. for 5 families — EPS, net income, shares, CFO, EBIT — every tag the filing
     has for its OWN period, its value, and ✓ if the ingest already reads that tag
Example of what to look for:
     EPS   EarningsPerShareBasicAndDiluted  2.14   (not read)   → one new tag fixes it
     EPS   (nothing)  · shares (nothing) · net income ✓        → per class only (cause B)

    python dev/eps_tags_202611.py --selftest      (no network)
    python dev/eps_tags_202611.py                 (~1 min)
    python dev/eps_tags_202611.py V C             (only these)
"""

import os
import re
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd

TICKERS = ["V", "BRK-B", "HSY", "STZ", "ERIE", "KKR", "ARES", "BKR", "KIM", "REG",
           "C", "XOM", "TAP", "FRT", "FERG"]
FAMILIES = {
    "EPS": r"EarningsPerShare|IncomeLoss.*PerShare|PerShare.*(?:Basic|Diluted)",
    "κέρδη": r"^NetIncomeLoss|^ProfitLoss$|NetIncomeLossAvailableToCommon|IncomeLossFromContinuingOperations$",
    "μετοχές": r"WeightedAverage.*Shares|SharesOutstanding",
    "CFO": r"^NetCashProvidedByUsedInOperatingActivities",
    "EBIT": r"^OperatingIncomeLoss$",
}
TOL = 7


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def facts_of_filing(facts: dict, accn: str) -> pd.DataFrame:
    """Every fact (any taxonomy, any tag) reported in one filing."""
    rows = []
    for tax, tags in facts.get("facts", {}).items():
        for tag, body in tags.items():
            for unit, obs in body.get("units", {}).items():
                for o in obs:
                    if o.get("accn") == accn:
                        rows.append((tax, tag, unit, o.get("start"), o.get("end"), o.get("val"), o.get("filed")))
    df = pd.DataFrame(rows, columns=["tax", "tag", "unit", "start", "end", "val", "filed"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def own_period(df: pd.DataFrame, report) -> pd.DataFrame:
    """Facts for the filing's own period end (±TOL days)."""
    if df.empty:
        return df
    return df[(df["end"] - pd.Timestamp(report)).abs() <= pd.Timedelta(days=TOL)]


def family_tags(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """One line per tag of a family: the largest-duration value (so a 10-K shows the year)."""
    if df.empty:
        return df
    sub = df[df["tag"].str.contains(pattern, regex=True)].copy()
    if sub.empty:
        return sub
    sub["days"] = (sub["end"] - pd.to_datetime(sub["start"])).dt.days.fillna(0)
    return sub.sort_values("days", ascending=False).drop_duplicates(["tax", "tag", "unit"])


def selftest() -> None:
    f = {"facts": {
        "us-gaap": {
            "EarningsPerShareBasicAndDiluted": {"units": {"USD/shares": [
                {"accn": "A1", "start": "2025-01-01", "end": "2025-12-31", "val": 2.1, "filed": "2026-02-20"},
                {"accn": "A1", "start": "2024-01-01", "end": "2024-12-31", "val": 1.9, "filed": "2026-02-20"}]}},
            "NetIncomeLoss": {"units": {"USD": [
                {"accn": "A1", "start": "2025-10-01", "end": "2025-12-31", "val": 5, "filed": "2026-02-20"},
                {"accn": "A1", "start": "2025-01-01", "end": "2025-12-31", "val": 20, "filed": "2026-02-20"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"accn": "A1", "end": "2026-02-01", "val": 10, "filed": "2026-02-20"}]}}},
    }}
    df = facts_of_filing(f, "A1")
    assert len(df) == 5 and set(df["tax"]) == {"us-gaap", "dei"}
    own = own_period(df, "2025-12-31")
    assert len(own) == 3                                         # the 2024 comparative is out
    eps = family_tags(own, FAMILIES["EPS"])
    assert eps["tag"].tolist() == ["EarningsPerShareBasicAndDiluted"] and eps["val"].iloc[0] == 2.1
    ni = family_tags(own, FAMILIES["κέρδη"])
    assert ni["val"].tolist() == [20]                            # the full year, not the quarter
    assert facts_of_filing(f, "OTHER").empty
    print("selftest: όλα τα τεστ πέρασαν ✓")


# ═════════════════════════════════════════════════════════════════════════
# 2. RUN
# ═════════════════════════════════════════════════════════════════════════

def sec_get(url: str):
    import requests
    from ticker_provider import SEC_HEADERS
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def main(tickers: list[str]) -> None:
    from fundamentals import COMPANYFACTS_URL, SUBMISSIONS_URL, XBRL_TAG_MAP
    from ticker_provider import get_cik_for_ticker

    read = {t for tags in XBRL_TAG_MAP.values() for t in tags}
    for t in tickers:
        cik = get_cik_for_ticker(t)
        print(f"══ {t} · CIK {cik} ══")
        if cik is None:
            print("  χωρίς CIK\n")
            continue
        try:
            sub = sec_get(SUBMISSIONS_URL.format(cik=cik))
            time.sleep(0.2)
            facts = sec_get(COMPANYFACTS_URL.format(cik=cik))
            time.sleep(0.2)
        except Exception as e:
            print(f"  σφάλμα SEC: {type(e).__name__}: {e}\n")
            continue
        rec = sub.get("filings", {}).get("recent", {})
        filings = [(f, r, d, a) for f, r, d, a in zip(rec.get("form", []), rec.get("reportDate", []),
                                                      rec.get("filingDate", []), rec.get("accessionNumber", []))
                   if f in ("10-K", "10-Q") and r]
        latest_k = [x for x in filings if x[0] == "10-K"][:1]
        last_q = [x for x in filings if x[0] == "10-Q"][:3]
        for form, report, filed, accn in sorted(latest_k + last_q, key=lambda x: x[1]):
            df = facts_of_filing(facts, accn)
            taxes = ", ".join(f"{k} {v}" for k, v in df["tax"].value_counts().items()) if len(df) else "—"
            print(f"  {form} {report} (κατάθ. {filed}) · facts στο companyfacts: {len(df)} [{taxes}]")
            if df.empty:
                same_day = sum(1 for tags in facts.get("facts", {}).values() for b in tags.values()
                               for obs in b.get("units", {}).values() for o in obs if o.get("filed") == filed)
                print(f"     ✗ η κατάθεση ΔΕΝ υπάρχει στο companyfacts"
                      + (f" (αλλά {same_day} facts με ίδια ημερομηνία κατάθεσης)" if same_day else ""))
                continue
            own = own_period(df, report)
            for fam, pat in FAMILIES.items():
                tags = family_tags(own, pat)
                if tags.empty:
                    print(f"     {fam:8s} (τίποτα για την περίοδο {report})")
                    continue
                for r in tags.head(6).itertuples():
                    mark = "✓" if r.tag in read and r.tax == "us-gaap" else " "
                    v = f"{r.val:,.2f}" if abs(r.val) < 1000 else f"{r.val:,.0f}"
                    print(f"     {fam:8s} {mark} {r.tax}:{r.tag} [{r.unit}] {v}"
                          + (f" ({int(r.days)} μέρες)" if r.days else ""))
        print()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main([a.upper() for a in sys.argv[1:]] or TICKERS)
