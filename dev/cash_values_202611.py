"""
dev/cash_values_202611.py — READ-ONLY: which securities tag can be ADDED to cash?

tag_discovery_202611.py told us which tags EXIST. It could not tell us whether
two tags hold the same money, because it never looked at the amounts. This
script looks at the amounts, on the same balance-sheet date as Yahoo:

    KLAC-like:  cash 2.0 + securities tag 3.2 = 5.2   vs Yahoo cash+STI 5.2  ✓
    double:     cash 2.0 + tag A 3.2 + tag B 3.2 = 8.4 vs Yahoo 5.2           ✗

Nothing is written to the database; nothing is ingested.

    python dev/cash_values_202611.py --selftest   (1 sec, no network, no database)
    python dev/cash_values_202611.py              (asks SEC and Yahoo, resumable)
    python dev/cash_values_202611.py --report     (reprints the summary)

→ data/cash_values_202611.jsonl   one line per ticker
Needs data/tag_discovery_202611.csv from the previous step.

Yahoo is a COMPARATOR here, not the truth: when it disagrees, we read the filing.
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

SURVEY = "data/tag_discovery_202611.csv"
OUT = "data/cash_values_202611.jsonl"
SINCE = "2025-06-01"        # a tag that ended before this can't affect today's EV
SLEEP = 0.5                 # between tickers, to stay under Yahoo's throttling
BAND = 0.05                 # the roadmap's criterion: within 5%
WATCH = ["GOOGL", "NVDA", "KLAC", "AAPL", "DXCM", "CMG", "PD", "ADI", "AMZN", "ADP", "AMD", "AKAM"]

CASH = ["CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations"]
COMBINED = "CashCashEquivalentsAndShortTermInvestments"     # cash INCLUDED: never add it to cash
ASSETS_CURRENT = "AssetsCurrent"

# Balance-sheet style lines, "current" by name. Order = priority.
CHAIN = ["ShortTermInvestments",
         "MarketableSecuritiesCurrent",
         "DebtSecuritiesCurrent",
         "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
         "AvailableForSaleSecuritiesCurrent",
         "OtherShortTermInvestments"]
# Footnote totals: may contain long-term securities or cash equivalents.
TOTALS = ["AvailableForSaleSecuritiesDebtSecurities", "MarketableSecurities"]

DEBT_NC = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
DEBT_CUR = ["DebtCurrent", "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
DEBT_TOTAL = ["LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities", "LongTermDebt"]
CONV_NC = ["ConvertibleLongTermNotesPayable", "ConvertibleDebtNoncurrent",
           "ConvertibleNotesPayableNoncurrent", "ConvertibleSubordinatedDebtNoncurrent"]
CONV_CUR = ["ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent",
            "ConvertibleSubordinatedDebtCurrent"]
CONV_TOTAL = ["ConvertibleDebt", "ConvertibleNotesPayable", "ConvertibleSubordinatedDebt"]

ALL_TAGS = (CASH + [COMBINED, ASSETS_CURRENT] + CHAIN + TOTALS + DEBT_NC + DEBT_CUR
            + DEBT_TOTAL + CONV_NC + CONV_CUR + CONV_TOTAL)
# what the "annual only" count looks at
WATCHED_TAGS = CHAIN + TOTALS + [COMBINED] + CONV_NC + CONV_CUR + CONV_TOTAL


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def tag_values(gaap: dict, tag: str) -> dict:
    """{period_end: value} of a balance-sheet tag. Earliest filing wins, like the ingest."""
    best = {}
    units = gaap.get(tag, {}).get("units", {})
    for o in units.get("USD", []):
        form, end = o.get("form") or "", o.get("end")
        if o.get("start") or not end or not form.startswith(("10-Q", "10-K")) or o.get("val") is None:
            continue                            # "start" = a duration, not a balance-sheet value
        if end not in best or (o.get("filed") or "") < best[end][0]:
            best[end] = (o.get("filed") or "", float(o["val"]))
    return {e: v[1] for e, v in best.items()}


def annual_dates(gaap: dict) -> list[str]:
    """Balance-sheet dates that came from a 10-K's own cash line."""
    out = set()
    for tag in CASH:
        for o in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if (o.get("form") or "").startswith("10-K") and o.get("end") and not o.get("start"):
                out.add(o["end"])
    return sorted(out)


def nearest_date(dates, target: str, days: int = 3) -> str | None:
    """Yahoo and the SEC can label a 52/53-week quarter a day or two apart."""
    if not dates:
        return None
    t = pd.Timestamp(target)
    best = min(dates, key=lambda d: abs((pd.Timestamp(d) - t).days))
    return best if abs((pd.Timestamp(best) - t).days) <= days else None


def first(v: dict, names: list[str]):
    """First tag with a positive amount. A stated 0 means "none on this line"."""
    for n in names:
        if v.get(n) is not None and v[n] > 0:
            return n, v[n]
    return None, None


def snapshot(gaap: dict, date: str | None = None) -> tuple[str | None, dict]:
    """All tags on one date (default: the latest cash date) → (date, {tag: value})."""
    cash_dates = sorted({d for t in CASH for d in tag_values(gaap, t)})
    if not cash_dates:
        return None, {}
    d = nearest_date(cash_dates, date) if date else cash_dates[-1]
    if d is None:
        return None, {}
    out = {}
    for tag in ALL_TAGS:
        val = tag_values(gaap, tag).get(d)
        if val is not None:
            out[tag] = val
    return d, out


def annual_only(gaap: dict) -> list[str]:
    """
    Tags with a value on the last 10-K date but not on the newest date.
    A December company's 10-K date is 10 months old by November, so a 9-month
    "live" test would call these dead although the company still uses them.
    """
    fy = annual_dates(gaap)
    latest, _ = snapshot(gaap)
    if not fy or latest is None or fy[-1] >= latest:
        return []
    return [t for t in WATCHED_TAGS
            if fy[-1] in tag_values(gaap, t) and latest not in tag_values(gaap, t)]


def cash_rules(v: dict) -> dict:
    """The candidate definitions of "cash + securities", side by side."""
    _, cash = first(v, CASH)
    chain_tag, chain_val = first(v, CHAIN)
    total_tag, total_val = first(v, TOTALS)
    combined = v.get(COMBINED)
    current = v.get(ASSETS_CURRENT)
    out = {"today": None, "chain": None, "chain_total": None, "sum_all": None,
           "chain_tag": chain_tag, "total_tag": None, "guard_failed": False}
    if cash is None:
        out["chain"] = out["chain_total"] = combined      # the combined line replaces, never adds
        return out
    out["today"] = cash + (v.get("ShortTermInvestments") or 0.0)
    out["chain"] = cash + (chain_val or 0.0)
    out["sum_all"] = cash + sum(v.get(t) or 0.0 for t in CHAIN + TOTALS)
    out["chain_total"] = out["chain"]
    if chain_tag is None and total_tag is not None:
        # A total can hold long-term securities: cash + total above ALL current assets proves it
        if current is not None and cash + total_val <= current:
            out["chain_total"], out["total_tag"] = cash + total_val, total_tag
        else:
            out["guard_failed"] = True
    return out


def regular_debt(v: dict):
    """The same rule as valuation.ev.debt_series, for one date. None = no debt tag."""
    _, nc = first(v, DEBT_NC)
    _, cur = first(v, DEBT_CUR)
    _, tot = first(v, DEBT_TOTAL)
    if nc is not None:
        return nc + (cur or 0.0)
    return tot


def convertible_debt(v: dict):
    _, nc = first(v, CONV_NC)
    _, cur = first(v, CONV_CUR)
    _, tot = first(v, CONV_TOTAL)
    if nc is None and cur is None:
        return tot
    return (nc or 0.0) + (cur or 0.0)


def within(a, b, band: float = BAND) -> bool:
    return a is not None and b not in (None, 0) and abs(a / b - 1) <= band


def convertible_verdict(regular, conv, yahoo_debt) -> str:
    """Are the convertibles already inside the ordinary debt tags?"""
    if pd.isna(regular):                # in a DataFrame a missing None comes back as NaN
        return "ΜΟΝΟ μετατρέψιμα"
    if pd.isna(yahoo_debt):
        return "χωρίς Yahoo"
    if within(regular, yahoo_debt):
        return "μέσα στο χρέος"
    if within(regular + conv, yahoo_debt):
        return "ΕΞΩ από το χρέος"
    return "ασαφές"


def yahoo_balance(qbs) -> dict:
    """Newest Yahoo date that has cash + short-term investments, with its debt."""
    out = {"y_date": None, "y_total": None, "y_debt": None}
    name = "Cash Cash Equivalents And Short Term Investments"
    if qbs is None or getattr(qbs, "empty", True) or name not in qbs.index:
        return out
    s = qbs.loc[name].dropna()
    if s.empty:
        return out
    d = max(s.index)
    out["y_date"], out["y_total"] = pd.Timestamp(d).date().isoformat(), float(s[d])
    parts = [float(qbs.loc[n, d]) for n in ("Long Term Debt", "Current Debt")
             if n in qbs.index and pd.notna(qbs.loc[n, d])]
    out["y_debt"] = sum(parts) if parts else None
    return out


def record(ticker: str, gaap: dict, yb: dict) -> dict:
    date, v = snapshot(gaap, yb["y_date"])
    if date is None and yb["y_date"]:                   # no SEC date near Yahoo's: compare nothing
        date, v = snapshot(gaap)
        yb = {"y_date": None, "y_total": None, "y_debt": None}
    return {"ticker": ticker, "date": date, "values": v, **yb, "annual_only": annual_only(gaap)}


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def _bn(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x / 1e9:,.2f}"


def _dev(a, b) -> str:
    return "—" if a is None or b in (None, 0) or pd.isna(a) or pd.isna(b) else f"{(a / b - 1) * 100:+.0f}%"


def enrich(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rules = cash_rules(r["values"])
        rows.append({**{k: r.get(k) for k in ("ticker", "date", "y_total", "y_debt", "financial")},
                     **rules, "regular": regular_debt(r["values"]),
                     "conv": convertible_debt(r["values"]), "values": r["values"]})
    return pd.DataFrame(rows)


def rule_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    names = {"today": "σήμερα (Cash + STI)", "chain": "αλυσίδα", "chain_total": "αλυσίδα + total με φρένο",
             "sum_all": "άθροισμα όλων (το λάθος)"}
    for col, label in names.items():
        x = df[df[col].notna() & df["y_total"].notna()]
        ratio = (x[col] / x["y_total"] - 1).astype(float)
        out.append({"κανόνας": label, "n": len(x), "εντός_5%": int((ratio.abs() <= BAND).sum()),
                    "πάνω_από_+5%": int((ratio > BAND).sum()), "κάτω_από_-5%": int((ratio < -BAND).sum())})
    return pd.DataFrame(out)


def tag_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    x = df[df["y_total"].notna()]
    for tag in CHAIN + TOTALS:
        hit = over = n = 0
        for row in x.itertuples():
            v = row.values
            _, cash = first(v, CASH)
            if cash is None or not v.get(tag):
                continue
            n += 1
            ratio = (cash + v[tag]) / row.y_total - 1
            hit += abs(ratio) <= BAND
            over += ratio > BAND
        out.append({"tag": tag, "εταιρείες": n, "cash+tag εντός 5%": hit, "πάνω από Yahoo": over})
    return pd.DataFrame(out)


def report() -> None:
    records = [json.loads(line) for line in open(OUT, encoding="utf-8")]
    df = enrich(records)
    fin = df["financial"].fillna(False).astype(bool)
    print(f"── {len(df)} εταιρείες με αποτέλεσμα · με Yahoo: {df['y_total'].notna().sum()}"
          f" · χρηματοοικονομικές (εκτός πινάκων): {fin.sum()} ──\n")
    d = df[~fin].copy()

    print("── 1. Ποιος κανόνας συμφωνεί με το Yahoo; ──")
    print(rule_table(d).to_string(index=False))

    print("\n── 2. Κάθε tag μόνο του: ταμείο + tag έναντι Yahoo ──")
    print(tag_table(d).to_string(index=False))

    print("\n── 3. Έλεγχος μέσα στη SEC: ταμείο + αλυσίδα έναντι του έτοιμου αθροίσματος ──")
    both = [r for r in d.itertuples() if r.values.get(COMBINED) is not None
            and first(r.values, CASH)[1] is not None]
    ok = sum(within(r.chain, r.values[COMBINED], 0.01) for r in both)
    print(f"  {ok}/{len(both)} συμφωνούν εντός 1%")

    print("\n── 4. Οι μετοχές του ελέγχου ──")
    w = d[d["ticker"].isin(WATCH)].sort_values("ticker")
    print(pd.DataFrame({"ticker": w["ticker"], "ημ/νία": w["date"], "tag αλυσίδας": w["chain_tag"].fillna("—"),
                        "tag total": w["total_tag"].fillna("—"), "δικό_μας": w["chain_total"].map(_bn),
                        "Yahoo": w["y_total"].map(_bn),
                        "απόκλιση": [_dev(a, b) for a, b in zip(w["chain_total"], w["y_total"])]}
                       ).to_string(index=False))

    print("\n── 5. Εταιρείες που σώζει μόνο ένα tag total (η περίπτωση της KLAC) ──")
    t = d[d["total_tag"].notna() | d["guard_failed"]]
    print(f"  {len(t)} εταιρείες · το φρένο απέρριψε {int(t['guard_failed'].sum())}")
    for r in t.head(40).itertuples():
        print(f"  {r.ticker:6s} {r.date}  με total {_bn(r.chain_total if r.total_tag else None):>9s}"
              f"  χωρίς {_bn(r.chain):>9s}  Yahoo {_bn(r.y_total):>9s}"
              f"  {'ΦΡΕΝΟ' if r.guard_failed else _dev(r.chain_total, r.y_total)}")

    print("\n── 6. Μεγαλύτερες αποκλίσεις του «αλυσίδα + total» ──")
    b = d[d["chain_total"].notna() & d["y_total"].notna()].copy()
    b["ratio"] = (b["chain_total"] / b["y_total"] - 1).astype(float)
    b = b[b["ratio"].abs() > BAND].sort_values("ratio", key=abs, ascending=False)
    print(f"  {len(b)} εκτός 5%, οι πρώτες 25:")
    for r in b.head(25).itertuples():
        print(f"  {r.ticker:6s} {r.date}  δικό μας {_bn(r.chain_total):>9s}  Yahoo {_bn(r.y_total):>9s}"
              f"  {r.ratio * 100:+.0f}%  {r.chain_tag or r.total_tag or 'μόνο ταμείο'}")

    print("\n── 7. Μετατρέψιμα: είναι ήδη μέσα στο χρέος που διαβάζουμε; ──")
    c = df[df["conv"].notna()].sort_values("ticker")
    verdicts = [convertible_verdict(r.regular, r.conv, r.y_debt) for r in c.itertuples()]
    print(pd.Series(verdicts, dtype=object).value_counts().to_string())
    print()
    for r, verdict in zip(c.itertuples(), verdicts):
        print(f"  {r.ticker:6s} {r.date}  μετατρέψιμα {_bn(r.conv):>7s}  χρέος {_bn(r.regular):>8s}"
              f"  Yahoo {_bn(r.y_debt):>8s}  {verdict}")

    print("\n── 8. Tags που υπάρχουν στο 10-K αλλά όχι στο νεότερο τρίμηνο ──")
    print("  (το «ζωντανές» του προηγούμενου script τα μετρούσε ως νεκρά)")
    s = pd.Series([t for r in records for t in r.get("annual_only", [])], dtype=object).value_counts()
    print(s.to_string() if len(s) else "  κανένα")


# ═════════════════════════════════════════════════════════════════════════
# 3. COLLECTION
# ═════════════════════════════════════════════════════════════════════════

def tickers_to_check() -> list[str]:
    d = pd.read_csv(SURVEY)
    d = d[d["group"].isin(["cash_securities", "convertible_debt"]) & (d["last_end"].astype(str) >= SINCE)]
    return sorted(set(d["ticker"]) | set(WATCH))


def collect() -> None:
    import yfinance as yf
    from db import get_engine
    from fundamentals import fetch_companyfacts
    from tag_discovery_202610 import ticker_ciks
    from yahoo_check_202611 import load_sectors, yahoo_symbol

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    todo = tickers_to_check()
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["ticker"] for line in open(OUT, encoding="utf-8")}
    cik = dict(ticker_ciks(get_engine()).itertuples(index=False))
    sectors = load_sectors(todo)
    rest = [t for t in todo if t not in done and t in cik]
    print(f"{len(todo)} εταιρείες, {len(done)} έτοιμες, {len(rest)} μένουν\n")

    t0 = time.time()
    for i, t in enumerate(rest, 1):
        try:
            facts = fetch_companyfacts(int(cik[t]))
            gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
            try:
                yb = yahoo_balance(yf.Ticker(yahoo_symbol(t)).quarterly_balance_sheet)
            except Exception:
                yb = yahoo_balance(None)
            rec = record(t, gaap, yb)
        except Exception as e:
            print(f"  {t}: σφάλμα {str(e)[:80]}")
            continue
        rec["financial"] = sectors.get(t) == "Financial Services"
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(SLEEP)
        if i % 25 == 0:
            print(f"  ... {i}/{len(rest)}  ~{(time.time() - t0) / i * (len(rest) - i) / 60:.0f} λεπτά ακόμα")


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST: invented companies, no network, no database
# ═════════════════════════════════════════════════════════════════════════

def _gaap(values: dict, form: str = "10-Q", end: str = "2026-09-30") -> dict:
    return {t: {"units": {"USD": [{"end": end, "val": v, "form": form, "filed": "2026-10-30"}]}}
            for t, v in values.items()}


def selftest() -> None:
    C, M, A, T = CASH[0], "MarketableSecuritiesCurrent", CHAIN[3], TOTALS[0]

    # same money in two tags: priority takes it once, the sum takes it twice
    r = cash_rules({C: 2.0, M: 3.0, A: 3.0, T: 3.0, ASSETS_CURRENT: 9.0})
    assert r["chain"] == 5.0 and r["chain_tag"] == M and r["sum_all"] == 11.0

    # KLAC class: only a total. Used, because 2 + 3 fits inside current assets.
    r = cash_rules({C: 2.0, T: 3.0, ASSETS_CURRENT: 9.0})
    assert r["chain"] == 2.0 and r["chain_total"] == 5.0 and r["total_tag"] == T

    # total holding long-term securities: 2 + 30 > 9 of current assets, refused
    r = cash_rules({C: 2.0, T: 30.0, ASSETS_CURRENT: 9.0})
    assert r["chain_total"] == 2.0 and r["guard_failed"]

    # no way to apply the brake without current assets: refused too
    assert cash_rules({C: 2.0, T: 3.0})["guard_failed"]

    # the combined line is never added on top of cash
    assert cash_rules({C: 2.0, M: 3.0, COMBINED: 5.0})["chain"] == 5.0
    assert cash_rules({COMBINED: 5.0})["chain"] == 5.0

    # a stated 0 on the first line must not hide the next one
    assert cash_rules({C: 2.0, CHAIN[0]: 0.0, M: 3.0})["chain"] == 5.0

    # comparatives: the earliest filing wins, and durations are ignored
    g = {C: {"units": {"USD": [
        {"end": "2026-06-30", "val": 7, "form": "10-Q", "filed": "2026-10-30"},
        {"end": "2026-06-30", "val": 6, "form": "10-Q", "filed": "2026-07-30"},
        {"end": "2026-06-30", "val": 5, "form": "8-K", "filed": "2026-07-01"},
        {"end": "2026-06-30", "start": "2026-01-01", "val": 4, "form": "10-Q", "filed": "2026-07-01"}]}}}
    assert tag_values(g, C) == {"2026-06-30": 6.0}

    # dates: a 52/53-week label two days off still matches, a whole quarter does not
    assert nearest_date(["2026-09-28"], "2026-09-30") == "2026-09-28"
    assert nearest_date(["2026-06-30"], "2026-09-30") is None
    assert snapshot(_gaap({C: 2.0, M: 3.0}), "2026-09-29")[1] == {C: 2.0, M: 3.0}

    # debt: same rule as ev.py, no double count of the current part in a total
    assert regular_debt({"LongTermDebtNoncurrent": 40.0, "DebtCurrent": 3.0, "LongTermDebt": 43.0}) == 43.0
    assert regular_debt({"LongTermDebt": 43.0}) == 43.0
    assert regular_debt({}) is None
    assert convertible_debt({CONV_NC[0]: 2.0, CONV_CUR[0]: 1.0, "ConvertibleDebt": 3.0}) == 3.0
    assert convertible_debt({"ConvertibleDebt": 3.0}) == 3.0
    assert convertible_debt({}) is None

    # convertibles inside / outside / alone
    assert convertible_verdict(10.0, 2.0, 10.0) == "μέσα στο χρέος"
    assert convertible_verdict(10.0, 2.0, 12.0) == "ΕΞΩ από το χρέος"
    assert convertible_verdict(None, 2.0, 2.0) == "ΜΟΝΟ μετατρέψιμα"
    assert convertible_verdict(float("nan"), 2.0, 2.0) == "ΜΟΝΟ μετατρέψιμα"
    assert convertible_verdict(10.0, 2.0, float("nan")) == "χωρίς Yahoo"
    assert convertible_verdict(10.0, 2.0, 30.0) == "ασαφές"

    # annual-only: on the 10-K date but gone from the newest quarter
    g = {C: {"units": {"USD": [{"end": "2025-12-31", "val": 1, "form": "10-K", "filed": "2026-02-01"},
                               {"end": "2026-09-30", "val": 1, "form": "10-Q", "filed": "2026-10-30"}]}},
         **_gaap({A: 1.0}, "10-K", "2025-12-31")}
    assert annual_only(g) == [A]

    # Yahoo: newest date with a value, its debt without leases
    q = pd.DataFrame({pd.Timestamp("2026-09-30"): [None, None, None],
                      pd.Timestamp("2026-06-30"): [5.0, 8.0, 1.0]},
                     index=["Cash Cash Equivalents And Short Term Investments", "Long Term Debt", "Current Debt"])
    y = yahoo_balance(q)
    assert y["y_date"] == "2026-06-30" and y["y_total"] == 5.0 and y["y_debt"] == 9.0
    assert yahoo_balance(None)["y_date"] is None

    # when Yahoo's date doesn't exist at the SEC, nothing is compared
    r = record("X", _gaap({C: 2.0}), {"y_date": "2025-01-31", "y_total": 9.0, "y_debt": None})
    assert r["date"] == "2026-09-30" and r["y_total"] is None

    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report()
    else:
        collect()
        report()
