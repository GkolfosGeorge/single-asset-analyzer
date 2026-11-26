"""
dev/companyfacts_gaps_202611.py — READ-ONLY (ομάδα 2, 25/11/2026).

Confirmed for DUK, NEE, PLD (Q2 + Q3 2026) and WMB (Q3 2026): the 10-Q exists
in EDGAR (submissions) but has 0 facts in companyfacts, the only source of our
ingest. No tag change can recover it. Before building a fallback that reads
the filing's own XBRL, this measures HOW BIG the gap is and WHERE it is:

  1. for every company of the peer table: every 10-Q / 10-K filed since
     2025-01-01 (submissions) → in companyfacts or not (facts with its accession)
  2. missing filings per sector and per month filed (did it start in 2026?)
  3. for the first 3 missing filings: does EDGAR serve the filing's own XBRL
     instance (the "_htm.xml" file in the filing folder)? That is what a
     fallback would read.

Resumable: results go to the CSV as it runs; a rerun skips tickers already done.

    python dev/companyfacts_gaps_202611.py --selftest
    python dev/companyfacts_gaps_202611.py DUK NEE        (only these, ~5 sec)
    python dev/companyfacts_gaps_202611.py --sp500        (~10–15 min)
    python dev/companyfacts_gaps_202611.py                (all ~690, ~20–30 min)
→ data/companyfacts_gaps_202611.csv
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd

TABLE = "peer_data/peer_yields.csv"
OUT = "data/companyfacts_gaps_202611.csv"
SINCE = "2025-01-01"
SLEEP = 0.15                 # SEC allows 10 requests/s
FORMS = ("10-K", "10-Q")
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def accessions_in_facts(facts: dict) -> set:
    return {o.get("accn") for tags in facts.get("facts", {}).values() for b in tags.values()
            for obs in b.get("units", {}).values() for o in obs}


def filings_since(sub: dict, since: str) -> list[dict]:
    rec = sub.get("filings", {}).get("recent", {})
    out = []
    for form, report, filed, accn in zip(rec.get("form", []), rec.get("reportDate", []),
                                         rec.get("filingDate", []), rec.get("accessionNumber", [])):
        if form in FORMS and report and filed >= since:
            out.append({"form": form, "report": report, "filed": filed, "accn": accn})
    return out


def instance_file(index: dict) -> str | None:
    """The XBRL instance EDGAR extracts from an inline-XBRL filing (…_htm.xml)."""
    names = [i.get("name", "") for i in index.get("directory", {}).get("item", [])]
    hits = [n for n in names if n.endswith("_htm.xml")]
    if hits:
        return hits[0]
    xml = [n for n in names if n.endswith(".xml") and not n.startswith(("FilingSummary", "R"))
           and "_cal" not in n and "_def" not in n and "_lab" not in n and "_pre" not in n]
    return xml[0] if xml else None


def selftest() -> None:
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        {"accn": "A1", "val": 1}, {"accn": "A2", "val": 2}]}}},
        "dei": {"X": {"units": {"shares": [{"accn": "A3", "val": 3}]}}}}}
    assert accessions_in_facts(facts) == {"A1", "A2", "A3"}
    sub = {"filings": {"recent": {
        "form": ["10-Q", "8-K", "10-K", "10-Q", "10-Q/A"],
        "reportDate": ["2026-09-30", "2026-10-01", "2025-12-31", "2024-09-30", "2026-06-30"],
        "filingDate": ["2026-11-05", "2026-10-02", "2026-02-20", "2024-11-01", "2026-08-30"],
        "accessionNumber": ["Q3", "8K", "K", "OLD", "AMEND"]}}}
    f = filings_since(sub, "2025-01-01")
    assert [x["accn"] for x in f] == ["Q3", "K"], f          # no 8-K, no amendment, nothing before 2025
    idx = {"directory": {"item": [{"name": "duk-20260930.htm"}, {"name": "duk-20260930_htm.xml"},
                                  {"name": "duk-20260930_cal.xml"}, {"name": "FilingSummary.xml"}]}}
    assert instance_file(idx) == "duk-20260930_htm.xml"
    old = {"directory": {"item": [{"name": "duk-20160930.xml"}, {"name": "duk-20160930_lab.xml"}]}}
    assert instance_file(old) == "duk-20160930.xml"
    assert instance_file({"directory": {"item": [{"name": "a.htm"}]}}) is None
    print("selftest ok")


# ═════════════════════════════════════════════════════════════════════════
# 2. RUN
# ═════════════════════════════════════════════════════════════════════════

def sec_get(url: str):
    import requests
    from ticker_provider import SEC_HEADERS
    r = requests.get(url, headers=SEC_HEADERS, timeout=90)
    r.raise_for_status()
    time.sleep(SLEEP)
    return r.json()


def scan(tickers: list[str], sectors: dict) -> None:
    from fundamentals import COMPANYFACTS_URL, SUBMISSIONS_URL
    from ticker_provider import get_cik_for_ticker

    done = set(pd.read_csv(OUT)["ticker"]) if os.path.exists(OUT) else set()
    todo = [t for t in tickers if t not in done]
    print(f"Εταιρείες: {len(tickers)} · ήδη στο {OUT}: {len(tickers) - len(todo)} · τώρα: {len(todo)}")
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        rows = []
        cik = get_cik_for_ticker(t)
        if cik is None:
            rows.append({"ticker": t, "status": "χωρίς CIK"})
        else:
            try:
                sub = sec_get(SUBMISSIONS_URL.format(cik=cik))
                have = accessions_in_facts(sec_get(COMPANYFACTS_URL.format(cik=cik)))
                fl = filings_since(sub, SINCE)
                if not fl:
                    rows.append({"ticker": t, "cik": cik, "status": "καμία 10-Q/10-K από 2025"})
                for f in fl:
                    rows.append({"ticker": t, "cik": cik, **f,
                                 "status": "ok" if f["accn"] in have else "λείπει"})
            except Exception as e:
                rows.append({"ticker": t, "cik": cik, "status": f"σφάλμα: {type(e).__name__}"})
        df = pd.DataFrame(rows)
        df["sector"] = sectors.get(t, "")
        df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        if i % 25 == 0 or i == len(todo):
            left = (time.time() - t0) / i * (len(todo) - i)
            print(f"  [{i}/{len(todo)}] {t} · απομένουν ~{left / 60:.0f} λεπτά", flush=True)


def report(tickers: list[str]) -> None:
    d = pd.read_csv(OUT, dtype=str)
    d = d[d["ticker"].isin(tickers)]
    f = d[d["status"].isin(["ok", "λείπει"])].copy()
    miss = f[f["status"] == "λείπει"].copy()
    print(f"\n══ 1. Σύνοψη ══")
    print(f"  Εταιρείες: {d['ticker'].nunique()} · καταθέσεις 10-Q/10-K από 2025: {len(f)} · "
          f"λείπουν από το companyfacts: {len(miss)} ({len(miss) / max(len(f), 1):.0%}) "
          f"σε {miss['ticker'].nunique()} εταιρείες")
    other = d[~d["status"].isin(["ok", "λείπει"])]
    if len(other):
        print("  Άλλο: " + ", ".join(f"{s} {n}" for s, n in other["status"].value_counts().items()))
    if miss.empty:
        return

    print("\n══ 2α. Ανά μήνα κατάθεσης (ξεκίνησε κάποια στιγμή;) ══")
    f["month"] = f["filed"].str[:7]
    m = f.groupby("month")["status"].agg(n="size", missing=lambda s: (s == "λείπει").sum())
    m["%"] = (m["missing"] / m["n"] * 100).round(0)
    print(m.to_string())

    print("\n══ 2β. Ανά τομέα (εταιρείες με ≥ 1 κατάθεση που λείπει) ══")
    by = d.groupby("sector")["ticker"].nunique().rename("εταιρείες").to_frame()
    by["με κενό"] = miss.groupby("sector")["ticker"].nunique()
    by = by.fillna(0).astype(int)
    by["%"] = (by["με κενό"] / by["εταιρείες"] * 100).round(0)
    print(by.sort_values("με κενό", ascending=False).to_string())

    print("\n══ 2γ. Εταιρείες με κενό (καταθέσεις που λείπουν) ══")
    g = miss.groupby(["sector", "ticker"])["report"].apply(lambda s: " ".join(sorted(x[:7] for x in s)))
    for (sec, t), rep in g.items():
        print(f"  {sec[:22]:<22} {t:<6} {rep}")

    print("\n══ 3. Διαβάζεται το XBRL της ίδιας της κατάθεσης; (3 πρώτες που λείπουν) ══")
    for r in miss.sort_values("filed", ascending=False).head(3).itertuples():
        acc = r.accn.replace("-", "")
        url = ARCHIVE.format(cik=int(float(r.cik)), acc=acc)
        try:
            name = instance_file(sec_get(url))
            print(f"  {r.ticker} {r.form} {r.report}: " + (f"✓ {name}" if name else "✗ κανένα αρχείο instance"))
        except Exception as e:
            print(f"  {r.ticker} {r.form} {r.report}: σφάλμα {type(e).__name__}: {e}")
    print(f"\n→ {OUT}")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    table = pd.read_csv(TABLE)
    sectors = dict(zip(table["ticker"], table["sector"].fillna("")))
    args = [a.upper() for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        tickers = args
    elif "--sp500" in sys.argv:
        from ticker_provider import get_tickers
        tickers = sorted(get_tickers("sp500", folder_path="cache"))
    else:
        tickers = table["ticker"].tolist()
    scan(tickers, sectors)
    report(tickers)


if __name__ == "__main__":
    main()
