"""
dev/filing_gaps_202611.py — READ-ONLY: cause A of the roadmap (step 8α).
Some S&P 500 companies have recent rows in `fundamentals` but no yield, because
a whole FILING is missing: CRM's 10-K of 01/2026, TAP's Q1 2026, XOM's mid-2025
quarters, C stopped at 12/2025. Without it no TTM can be built.
Nothing is written anywhere.

For each company and each 10-K / 10-Q filed since 2024 (the SEC's own list,
`submissions`), and for 3 key metrics (EPS diluted, CFO, EBIT), it finds WHERE
the number got lost, with the same parser the ingest uses (parse_companyfacts):

    "στη βάση"                     ✓ fine
    "όχι στη βάση: κλειδί πιασμένο"  the parser produced the row, but the database
                                   already has a row with the same
                                   (fiscal_year, fiscal_period, form) for ANOTHER
                                   period → ON CONFLICT DO NOTHING kept the old one
    "όχι στη βάση: δεν γράφτηκε"    the parser produced it, nothing blocks it:
                                   the ingest simply hasn't run / failed (C?)
    "parser: ίδια ετικέτα fy/fp"    the SEC labels this filing with the SAME
                                   (fy, fp) as an EARLIER filing → the parser keeps
                                   the earliest (Stage 2) and this one is lost
    "parser: κόπηκε"                dropped by Stage 0/1 (own-period rules)
    "SEC: χωρίς fy/fp"              the facts have no fiscal year/period label
    "SEC: μόνο συγκριτικά"          the filing tags this metric only for OTHER periods
    "SEC: χωρίς το tag"             the filing has no fact for this metric at all
                                   (another tag, or only per share class: V, HSY?)

Example: CRM, 10-K, period 2026-01-31. If the SEC labels it fy=2025 FY — the
same label as the 10-K of 2025-01-31 — the earlier one wins and the FY2026
annual numbers never reach the database → "parser: ίδια ετικέτα fy/fp".

    python dev/filing_gaps_202611.py --selftest      (no network, no database)
    python dev/filing_gaps_202611.py                 (~2–3 min, 2 SEC downloads per company)
    python dev/filing_gaps_202611.py CRM TAP         (only these)
→ data/filing_gaps_202611.csv
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd

TICKERS = ["CRM", "XOM", "TAP", "C", "APTV", "KR", "CRWD", "WDAY", "FERG", "HONA", "MTD", "Q",
           "BKR", "ERIE", "KIM", "REG", "FRT", "V", "HSY", "STZ", "KKR", "ARES", "BRK-B"]
METRICS = {"EPS": "EarningsPerShareDiluted",
           "CFO": "NetCashProvidedByUsedInOperatingActivities",
           "EBIT": "OperatingIncomeLoss"}
SINCE = "2024-01-01"
TOL = 7                 # fundamentals.OWN_PERIOD_TOLERANCE_DAYS
OUT = "data/filing_gaps_202611.csv"
OK = "στη βάση"


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def near(a, b, tol: int = TOL) -> bool:
    return abs((pd.Timestamp(a) - pd.Timestamp(b)).days) <= tol


def classify(filing: dict, raw: pd.DataFrame, parsed: pd.DataFrame, db: pd.DataFrame) -> tuple[str, str]:
    """
    filing: accn, form, report (period end), filed.
    raw:    every SEC observation of ONE metric (accn, fy, fp, form, period_end, filed_date)
    parsed: what parse_companyfacts kept for that metric
    db:     our rows for that metric (fiscal_year, fiscal_period, form_type, period_end)
    → (status, detail)
    """
    rep, form = filing["report"], filing["form"]
    if not db.empty and ((db["form_type"] == form) & db["period_end"].map(lambda d: near(d, rep))).any():
        return OK, ""

    own = raw[(raw["accn"] == filing["accn"])]
    if own.empty:
        return "SEC: χωρίς το tag", ""
    own_p = own[own["period_end"].map(lambda d: near(d, rep))]
    if own_p.empty:
        return "SEC: μόνο συγκριτικά", f"περίοδοι {', '.join(sorted(own['period_end'].dt.strftime('%Y-%m').unique()))}"
    labeled = own_p.dropna(subset=["fy", "fp"])
    if labeled.empty:
        return "SEC: χωρίς fy/fp", ""
    fy, fp = int(labeled["fy"].iloc[0]), labeled["fp"].iloc[0]
    label = f"fy={fy} {fp}"

    p = parsed[(parsed["form_type"] == form) & parsed["period_end"].map(lambda d: near(d, rep))] \
        if not parsed.empty else parsed
    if not p.empty:
        row = p.iloc[0]
        clash = db[(db["fiscal_year"] == int(row["fiscal_year"])) & (db["fiscal_period"] == row["fiscal_period"])
                   & (db["form_type"] == form)] if not db.empty else db
        if not clash.empty:
            return "όχι στη βάση: κλειδί πιασμένο", (f"{label}: η βάση έχει ήδη την περίοδο "
                                                      f"{pd.Timestamp(clash['period_end'].iloc[0]):%Y-%m-%d}")
        return "όχι στη βάση: δεν γράφτηκε", label

    same = parsed[(parsed["fiscal_year"] == fy) & (parsed["fiscal_period"] == fp) & (parsed["form_type"] == form)] \
        if not parsed.empty else parsed
    if not same.empty:
        w = same.iloc[0]
        return "parser: ίδια ετικέτα fy/fp", (f"{label} και στην κατάθεση της {pd.Timestamp(w['filed_date']):%Y-%m-%d}"
                                              f" (περίοδος {pd.Timestamp(w['period_end']):%Y-%m-%d}), κρατήθηκε εκείνη")
    return "parser: κόπηκε", label


def selftest() -> None:
    T = pd.Timestamp
    f26 = {"accn": "K26", "form": "10-K", "report": T("2026-01-31"), "filed": T("2026-03-05")}
    raw = pd.DataFrame([
        ("K25", 2025, "FY", "10-K", T("2025-01-31"), T("2025-03-06")),
        ("K26", 2025, "FY", "10-K", T("2026-01-31"), T("2026-03-05")),     # mislabeled: fy 2025 again
        ("K26", 2025, "FY", "10-K", T("2025-01-31"), T("2026-03-05")),     # comparative
    ], columns=["accn", "fy", "fp", "form", "period_end", "filed_date"])
    parsed = pd.DataFrame([(2025, "FY", "10-K", T("2025-01-31"), T("2025-03-06"))],
                          columns=["fiscal_year", "fiscal_period", "form_type", "period_end", "filed_date"])
    db = pd.DataFrame([(2025, "FY", "10-K", T("2025-01-31"))],
                      columns=["fiscal_year", "fiscal_period", "form_type", "period_end"])
    assert classify(f26, raw, parsed, db)[0] == "parser: ίδια ετικέτα fy/fp"
    # in the database → fine
    db_ok = pd.concat([db, pd.DataFrame([(2026, "FY", "10-K", T("2026-01-30"))], columns=db.columns)])
    assert classify(f26, raw, parsed, db_ok)[0] == OK
    # parser keeps it, the key is free → not written (ingest didn't run)
    p2 = pd.concat([parsed, pd.DataFrame([(2026, "FY", "10-K", T("2026-01-31"), T("2026-03-05"))],
                                         columns=parsed.columns)])
    raw2 = raw.assign(fy=[2025, 2026, 2026])
    assert classify(f26, raw2, p2, db)[0] == "όχι στη βάση: δεν γράφτηκε"
    # parser keeps it, but the database has fy=2026 FY for another period → key taken
    db_clash = pd.concat([db, pd.DataFrame([(2026, "FY", "10-K", T("2025-01-31"))], columns=db.columns)])
    s, d = classify(f26, raw2, p2, db_clash)
    assert s == "όχι στη βάση: κλειδί πιασμένο" and "2025-01-31" in d
    # SEC side
    assert classify({**f26, "accn": "X"}, raw, parsed, db)[0] == "SEC: χωρίς το tag"
    only_comp = raw[raw["period_end"] != T("2026-01-31")]
    assert classify(f26, only_comp, parsed, db)[0] == "SEC: μόνο συγκριτικά"
    assert classify(f26, raw.assign(fy=None), parsed, db)[0] == "SEC: χωρίς fy/fp"
    # parser dropped it for another reason
    assert classify(f26, raw2, parsed, db)[0] == "parser: κόπηκε"
    print("selftest: όλα τα τεστ πέρασαν ✓")


# ═════════════════════════════════════════════════════════════════════════
# 2. DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════

def flatten(facts: dict, metric: str) -> pd.DataFrame:
    """Every us-gaap observation of one metric, like parse_companyfacts' first step."""
    from fundamentals import XBRL_TAG_MAP
    us = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in XBRL_TAG_MAP[metric]:
        for unit, obs in (us.get(tag) or {}).get("units", {}).items():
            for o in obs:
                rows.append((o.get("accn"), o.get("fy"), o.get("fp"), o.get("form"), o.get("end"), o.get("filed")))
    df = pd.DataFrame(rows, columns=["accn", "fy", "fp", "form", "period_end", "filed_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    return df


def sec_get(url: str, timeout: int):
    import requests
    from ticker_provider import SEC_HEADERS
    t0 = time.time()
    r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json(), time.time() - t0, len(r.content) / 1e6


def filings(sub: dict) -> list[dict]:
    rec = sub.get("filings", {}).get("recent", {})
    out = []
    for form, rep, filed, accn in zip(rec.get("form", []), rec.get("reportDate", []),
                                      rec.get("filingDate", []), rec.get("accessionNumber", [])):
        if form in ("10-K", "10-Q") and rep and rep >= SINCE:
            out.append({"accn": accn, "form": form, "report": pd.Timestamp(rep), "filed": pd.Timestamp(filed)})
    return sorted(out, key=lambda f: f["report"])


def db_rows(engine, ticker: str) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT m.name, f.fiscal_year, f.fiscal_period, ft.name, f.period_end, f.filed_date, f.cik
        FROM fundamentals f JOIN metric_lookup m ON m.id = f.metric_id
        JOIN form_type_lookup ft ON ft.id = f.form_type_id
        WHERE f.ticker = :t AND m.name = ANY(:m) AND f.period_end >= CAST(:s AS DATE) - 400
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"t": ticker, "m": list(METRICS.values()), "s": SINCE}).fetchall()
    df = pd.DataFrame(rows, columns=["metric", "fiscal_year", "fiscal_period", "form_type", "period_end",
                                     "filed_date", "cik"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    return df


# ═════════════════════════════════════════════════════════════════════════
# 3. RUN
# ═════════════════════════════════════════════════════════════════════════

def main(tickers: list[str]) -> None:
    from db import get_engine
    from fundamentals import COMPANYFACTS_URL, SUBMISSIONS_URL, parse_companyfacts
    from ticker_provider import get_cik_for_ticker

    eng = get_engine()
    out = []
    for t in tickers:
        cik = get_cik_for_ticker(t)
        db = db_rows(eng, t)
        db_ciks = sorted(int(c) for c in db["cik"].dropna().unique())
        print(f"══ {t} · CIK SEC {cik} · CIK στη βάση {db_ciks or '—'} ══")
        if cik is None:
            print("  χωρίς CIK\n")
            continue
        try:
            sub, _, _ = sec_get(SUBMISSIONS_URL.format(cik=cik), 30)
            time.sleep(0.2)
            try:
                facts, secs, mb = sec_get(COMPANYFACTS_URL.format(cik=cik), 30)
                note = f"{mb:.1f} MB σε {secs:.0f} s"
            except Exception as e:      # the ingest uses 30 s: does a bigger timeout help?
                facts, secs, mb = sec_get(COMPANYFACTS_URL.format(cik=cik), 180)
                note = f"⚠️ αποτυγχάνει με 30 s ({type(e).__name__}) · {mb:.1f} MB σε {secs:.0f} s με 180 s"
            time.sleep(0.2)
        except Exception as e:
            print(f"  σφάλμα SEC: {type(e).__name__}: {e}\n")
            out.append({"ticker": t, "status": f"σφάλμα SEC: {type(e).__name__}"})
            continue
        parsed_all = parse_companyfacts(t, cik, facts)
        print(f"  companyfacts: {note}")
        for f in filings(sub):
            cells = []
            for short, metric in METRICS.items():
                raw = flatten(facts, metric)
                parsed = parsed_all[parsed_all["metric"] == metric] if not parsed_all.empty else parsed_all
                mdb = db[db["metric"] == metric]
                s, d = classify(f, raw, parsed, mdb)
                cells.append(f"{short} {'✓' if s == OK else '✗ ' + s}")
                out.append({"ticker": t, "form": f["form"], "report": f["report"].date(),
                            "filed": f["filed"].date(), "metric": short, "status": s, "detail": d})
            if any("✗" in c for c in cells):
                print(f"  {f['form']:5s} {f['report']:%Y-%m-%d} (κατάθ. {f['filed']:%Y-%m-%d}) · " + " · ".join(cells))
                for r in out[-3:]:
                    if r.get("detail"):
                        print(f"        {r['metric']}: {r['detail']}")
        print()

    df = pd.DataFrame(out)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT, index=False)
    bad = df[df["status"].notna() & (df["status"] != OK)]
    print("── Σύνοψη: κενά ανά αιτία (κατάθεση × μέγεθος) ──")
    for s, g in bad.groupby("status"):
        print(f"  {len(g):4d}  {s:32s} {', '.join(sorted(g['ticker'].unique()))}")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main([a.upper() for a in sys.argv[1:]] or TICKERS)
