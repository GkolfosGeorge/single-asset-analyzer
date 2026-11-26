"""
dev/missing_yields_202611.py — READ-ONLY: Phase 2, step 8α of the roadmap.
WHY is a yield missing from the weekly peer table (peer_data/peer_yields.csv)?
One cause per missing yield and company, then grouped, so that we fix first
the causes that cover the most companies. Nothing is written to the database.

The table has three yields: earnings (P/E), FCF, EBIT/EV. Each one needs some
ingredients, and a yield is missing when the FIRST of them is missing:
    P/E      : company still filing → price → EPS TTM
    FCF      : (bank/insurer: expected) → filing → price → shares → CFO TTM
               → capex TTM → CFO and capex for the SAME quarter
    EBIT/EV  : (bank/insurer: expected) → filing → price → shares → EBIT TTM
               → cash → debt
Each ingredient is judged on the table's own day (as_of), with the same
production functions and the same 270-day rule as the card:
    "λείπει"      no row of that tag in the database
    "παλιό"       newest period older than 270 days (stopped being filed)
    "δεν βγαίνει" recent rows exist, but no TTM/value can be built from them
                  (e.g. a missing quarter)
Example: EPS rows up to 2026-06, but Q1 2026 is missing → the TTM stops at
2025-12 → "EPS: δεν βγαίνει".
If every ingredient is fine TODAY, the yield should have been there: then the
problem was in the table's run (e.g. Yahoo dropped the price that Saturday).

For a missing EPS/CFO/shares tag it also checks a few ALTERNATIVE tags (only those
that exist in our database) and says if one of them is fresh: a lead, not a fix.

    python dev/missing_yields_202611.py --selftest        (1 sec, no network, no database)
    python dev/missing_yields_202611.py                   (~3–6 min)
    python dev/missing_yields_202611.py --report          (reprints the summary from the CSV)
    python dev/missing_yields_202611.py --ticker XOM KO   (detail for a few companies)

→ data/missing_yields_202611.csv
"""

import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.getcwd())

import pandas as pd

from valuation import ev as E
from valuation import yields as Y
from valuation.ttm import (MAX_TTM_AGE_DAYS, _build_quarters, build_latest, build_ttm,
                           repair_share_scale)

TABLE = "peer_data/peer_yields.csv"
OUT = "data/missing_yields_202611.csv"
YEARS_BACK = 8              # enough for TTMs and for the cash/debt rules that look at the past
CHUNK = 60                  # tickers per database query
STALE_PRICE_DAYS = 10       # same as valuation/peer_table.py
EXAMPLES = 12

EPS = "EarningsPerShareDiluted"
EBIT = E.EBIT
ALT = {     # tried only if they exist in metric_lookup
    "EPS": ["EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic",
            "IncomeLossFromContinuingOperationsPerDilutedShare"],
    "CFO": ["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "μετοχές": ["WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                "WeightedAverageNumberOfSharesOutstandingBasic"],
}
BASE_METRICS = list(dict.fromkeys(
    [EPS, Y.CFO, Y.SHARES, Y.NET_INCOME, EBIT] + Y.CAPEX_TAGS + E.ALL_DEBT_METRICS + E.CASH_METRICS))
YIELDS = {"earnings_yield": "P/E", "fcf_yield": "FCF", "ebit_yield": "EV/EBIT"}
EXPECTED = "τράπεζα/ασφαλιστική (αναμενόμενο)"
RUN_PROBLEM = "όλα υπάρχουν σήμερα → πρόβλημα στο τρέξιμο του πίνακα"


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def _age(as_of: pd.Timestamp, d) -> int:
    return (as_of - pd.Timestamp(d)).days


def _newest(series: pd.DataFrame, as_of: pd.Timestamp):
    """Newest period_end usable on as_of (available ≤ as_of), or None."""
    if series is None or series.empty:
        return None
    known = series[pd.to_datetime(series["available"]) <= as_of]
    return None if known.empty else pd.Timestamp(known["period_end"].max())


def _last_gap(rows: pd.DataFrame) -> str:
    """Last ✗ line of the quarter builder: why a TTM could not be built."""
    try:
        _, _, _, log = _build_quarters(rows)
    except Exception:
        return ""
    bad = [x for x in log if "✗" in x]
    return bad[-1] if bad else ""


def judge(label: str, rows: pd.DataFrame, series: pd.DataFrame, as_of: pd.Timestamp,
          flow: bool = True) -> tuple[str, str]:
    """
    One ingredient → ("ok" | "λείπει" | "παλιό" | "δεν βγαίνει", detail).
    rows: raw DB rows (already filed ≤ as_of); series: what production built from them.
    """
    if rows is None or rows.empty:
        return "λείπει", f"κανένα {label} στη βάση"
    last_raw = pd.Timestamp(rows["period_end"].max())
    newest = _newest(series, as_of)
    if newest is not None and _age(as_of, newest) <= MAX_TTM_AGE_DAYS:
        return "ok", f"{label} ως {newest:%Y-%m}"
    if _age(as_of, last_raw) > MAX_TTM_AGE_DAYS:
        return "παλιό", f"τελευταία περίοδος {last_raw:%Y-%m-%d}"
    stop = f"{newest:%Y-%m}" if newest is not None else "πουθενά"
    gap = _last_gap(rows) if flow else ""
    return "δεν βγαίνει", (f"στοιχεία ως {last_raw:%Y-%m} αλλά η τιμή σταματά {stop}"
                           + (f" · {gap}" if gap else ""))


def _rows(by_metric: dict, *names) -> pd.DataFrame:
    parts = [by_metric[n] for n in names if n in by_metric and not by_metric[n].empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["value", "period_start", "period_end", "filed_date", "form_type", "metric"])


def ingredients(by_metric: dict, as_of: pd.Timestamp) -> dict[str, tuple[str, str]]:
    """Every ingredient of one company, judged with the production functions."""
    r = lambda *n: _rows(by_metric, *n)          # noqa: E731
    out = {}
    eps = r(EPS)
    out["EPS"] = judge("EPS", eps, build_ttm(eps, per_share=True, value_col="v"), as_of)

    shares_rows = repair_share_scale(r(Y.SHARES), r(Y.NET_INCOME), eps)
    out["μετοχές"] = judge("μετοχές", shares_rows,
                           build_latest(shares_rows, shares=True, value_col="v"), as_of, flow=False)

    cfo = r(Y.CFO)
    out["CFO"] = judge("CFO", cfo, build_ttm(cfo, value_col="v"), as_of)
    capex_rows = {m: r(m) for m in Y.CAPEX_TAGS}
    out["capex"] = judge("capex", r(*Y.CAPEX_TAGS), Y.capex_ttm(capex_rows), as_of)
    fcf = Y.fcf_ttm(cfo, capex_rows) if not cfo.empty else pd.DataFrame()
    out["FCF"] = judge("CFO − capex", r(Y.CFO, *Y.CAPEX_TAGS), fcf, as_of, flow=False)

    ebit = r(EBIT)
    out["EBIT"] = judge("EBIT", ebit, build_ttm(ebit, value_col="v"), as_of)
    cash = r(*E.CASH_METRICS)
    out["ταμείο"] = judge("ταμείο", cash, E.cash_series(cash) if not cash.empty else None,
                          as_of, flow=False)
    debt = r(*E.ALL_DEBT_METRICS)
    if debt.empty:      # production assumes 0 debt: not a reason for a missing yield
        out["χρέος"] = ("ok", "κανένα tag χρέους → υπόθεση 0")
    else:
        out["χρέος"] = judge("χρέος", debt, E.debt_series(debt), as_of, flow=False)

    for key, alts in ALT.items():
        if out[key][0] == "ok":
            continue
        good = []
        for a in alts:
            ar = r(a)
            if ar.empty:
                continue
            s = (build_latest(ar, shares=True, value_col="v") if key == "μετοχές"
                 else build_ttm(ar, per_share=(key == "EPS"), value_col="v"))
            if judge(a, ar, s, as_of, flow=key != "μετοχές")[0] == "ok":
                good.append(a)
        if good:
            st, det = out[key]
            out[key] = (st, f"{det} · φρέσκο εναλλακτικό: {', '.join(good)}")
    return out


CHAIN = {
    "earnings_yield": ["EPS"],
    "fcf_yield": ["μετοχές", "CFO", "capex", "FCF"],
    "ebit_yield": ["μετοχές", "EBIT", "ταμείο", "χρέος"],
}


def cause(yield_col: str, ing: dict, last_period, has_price: bool, financial: bool,
          as_of: pd.Timestamp) -> tuple[str, str]:
    """The FIRST missing ingredient of one yield → (cause, detail)."""
    if financial and yield_col != "earnings_yield":
        return EXPECTED, ""
    if last_period is None or _age(as_of, last_period) > MAX_TTM_AGE_DAYS:
        lp = "καμία" if last_period is None else f"{pd.Timestamp(last_period):%Y-%m-%d}"
        return "σταμάτησε να καταθέτει", f"τελευταία περίοδος {lp}" + ("" if has_price else " · ούτε τιμή")
    if not has_price:
        return "χωρίς πρόσφατη τιμή (Yahoo)", ""
    for key in CHAIN[yield_col]:
        st, det = ing[key]
        if st != "ok":
            if key == "FCF":
                return "CFO και capex σε διαφορετικά τρίμηνα", det
            return f"{key}: {st}", det
    return RUN_PROBLEM, ""


# ═════════════════════════════════════════════════════════════════════════
# 2. DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════

def existing_metrics(engine, names: list[str]) -> list[str]:
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT name FROM metric_lookup WHERE name = ANY(:n)"),
                            {"n": list(names)}).fetchall()
    return [r[0] for r in rows]


def load_rows(engine, tickers: list[str], metrics: list[str], as_of, since) -> pd.DataFrame:
    """Like ttm.load_metrics, plus a period window, in chunks of tickers."""
    from sqlalchemy import text
    sql = text("""
        SELECT f.ticker, f.value, f.period_start, f.period_end, f.filed_date,
               ft.name AS form_type, m.name AS metric
        FROM fundamentals f
        JOIN metric_lookup m     ON m.id  = f.metric_id
        JOIN form_type_lookup ft ON ft.id = f.form_type_id
        WHERE m.name = ANY(:metrics) AND f.ticker = ANY(:tickers)
          AND ft.name IN ('10-Q', '10-K')
          AND f.filed_date <= CAST(:as_of AS DATE) AND f.period_end >= CAST(:since AS DATE)
    """)
    parts = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        with engine.connect() as conn:
            rows = conn.execute(sql, {"metrics": metrics, "tickers": chunk,
                                      "as_of": str(as_of.date()), "since": str(since.date())}).fetchall()
        parts.append(pd.DataFrame(rows, columns=["ticker", "value", "period_start", "period_end",
                                                 "filed_date", "form_type", "metric"]))
        print(f"  ... βάση {min(i + CHUNK, len(tickers))}/{len(tickers)}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("period_start", "period_end", "filed_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def last_periods(engine, tickers: list[str], as_of) -> dict:
    """Newest period of ANY metric per company (filed ≤ as_of): is it still filing?"""
    from sqlalchemy import text
    sql = text("""
        SELECT ticker, MAX(period_end) FROM fundamentals
        WHERE ticker = ANY(:t) AND filed_date <= CAST(:as_of AS DATE)
        GROUP BY ticker
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"t": list(tickers), "as_of": str(as_of.date())}).fetchall()
    return {t: pd.Timestamp(d) for t, d in rows if d is not None}


def price_ok(tickers: list[str], as_of: pd.Timestamp) -> dict[str, bool]:
    """ONE batch download: did each ticker have a price in the 10 days before as_of?"""
    from valuation.ttm import load_prices
    close, _ = load_prices(tickers, (as_of - timedelta(days=30)).date(), (as_of + timedelta(days=1)).date())
    out = {}
    for t in tickers:
        s = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
        s = s[s.index <= as_of]
        out[t] = bool(len(s)) and _age(as_of, s.index[-1]) <= STALE_PRICE_DAYS
    return out


# ═════════════════════════════════════════════════════════════════════════
# 3. RUN + REPORT
# ═════════════════════════════════════════════════════════════════════════

def run(only: list[str] | None = None) -> pd.DataFrame:
    from db import get_engine
    table = pd.read_csv(TABLE)
    as_of = pd.Timestamp(table["as_of"].max())
    miss = table[table[list(YIELDS)].isna().any(axis=1)]
    if only:
        miss = table[table["ticker"].isin(only)]
    tickers = miss["ticker"].tolist()
    print(f"Πίνακας της {as_of:%d/%m/%Y}: {len(table)} εταιρείες, {len(tickers)} με τουλάχιστον μία "
          f"απόδοση που λείπει", flush=True)

    eng = get_engine()
    t0 = time.time()
    alts = existing_metrics(eng, [a for v in ALT.values() for a in v])
    absent = [a for v in ALT.values() for a in v if a not in alts]
    print(f"Εναλλακτικά tags στη βάση: {', '.join(alts) or 'κανένα'}"
          + (f" · ΟΧΙ στη βάση: {', '.join(absent)}" if absent else ""), flush=True)
    rows = load_rows(eng, tickers, BASE_METRICS + alts, as_of, as_of - pd.DateOffset(years=YEARS_BACK))
    last = last_periods(eng, tickers, as_of)
    print(f"Βάση: {len(rows):,} γραμμές σε {time.time() - t0:.0f} s · τιμές από Yahoo...", flush=True)
    prices = price_ok(tickers, as_of)

    out = []
    grouped = {t: g for t, g in rows.groupby("ticker")}
    for _, row in miss.iterrows():
        t = row["ticker"]
        g = grouped.get(t, rows.iloc[0:0])
        by_metric = {m: x.drop(columns=["ticker"]) for m, x in g.groupby("metric")}
        try:
            ing = ingredients(by_metric, as_of)
        except Exception as e:
            ing = None
            err = f"{type(e).__name__}: {e}"
        rec = {"ticker": t, "sector": row["sector"], "industry": row["industry"],
               "financial": bool(row["financial"]), "has_price": prices.get(t, False),
               "last_period": last.get(t)}
        for col, name in YIELDS.items():
            if pd.notna(row[col]) and not only:
                continue
            if pd.notna(row[col]):
                rec[f"{name} αιτία"], rec[f"{name} λεπτομέρεια"] = "υπάρχει", f"{row[col]:.4f}"
            elif ing is None:
                rec[f"{name} αιτία"], rec[f"{name} λεπτομέρεια"] = "σφάλμα στο script", err
            else:
                c, d = cause(col, ing, last.get(t), prices.get(t, False), bool(row["financial"]), as_of)
                rec[f"{name} αιτία"], rec[f"{name} λεπτομέρεια"] = c, d
        if only and ing is not None:
            rec["συστατικά"] = " | ".join(f"{k}: {s} ({d})" for k, (s, d) in ing.items())
        out.append(rec)
    print(f"Έτοιμο σε {(time.time() - t0) / 60:.1f} λεπτά\n")
    return pd.DataFrame(out)


def report(df: pd.DataFrame) -> None:
    for name in YIELDS.values():
        c = f"{name} αιτία"
        if c not in df.columns:
            continue
        sub = df[df[c].notna()]
        real = sub[sub[c] != EXPECTED]
        print(f"══ {name}: λείπει σε {len(sub)} εταιρείες"
              f" ({len(sub) - len(real)} τράπεζες/ασφαλιστικές, αναμενόμενο) ══")
        counts = real[c].value_counts()
        for cause_, n in counts.items():
            ex = real.loc[real[c] == cause_, "ticker"].tolist()
            more = f" … +{len(ex) - EXAMPLES}" if len(ex) > EXAMPLES else ""
            print(f"  {n:4d}  {cause_:45s} {', '.join(ex[:EXAMPLES])}{more}")
        print()

    alive = df[df["has_price"].astype(bool)]
    for name in YIELDS.values():
        c, d = f"{name} αιτία", f"{name} λεπτομέρεια"
        if c not in df.columns:
            continue
        alt = alive[alive[d].fillna("").str.contains("εναλλακτικό")]
        if len(alt):
            print(f"Φρέσκο εναλλακτικό tag ({name}): {len(alt)} εταιρείες · {', '.join(alt['ticker'][:20])}")

    cols = [f"{n} αιτία" for n in YIELDS.values() if f"{n} αιτία" in df.columns]
    run_pb = df[(df[cols] == RUN_PROBLEM).any(axis=1)]
    if len(run_pb):
        print(f"\nΠρόβλημα στο τρέξιμο (όλα υπάρχουν σήμερα): {len(run_pb)} εταιρείες · "
              f"{', '.join(run_pb['ticker'][:30])}")
    stopped = df[(df[cols] == "σταμάτησε να καταθέτει").any(axis=1)]
    still_traded = stopped[stopped["has_price"].astype(bool)]
    print(f"Σταμάτησαν να καταθέτουν: {len(stopped)}, από τις οποίες {len(still_traded)} έχουν ακόμα τιμή "
          f"(πιθανή αλλαγή CIK ή εξαγορά): {', '.join(still_traded['ticker'][:30])}")
    print(f"\n→ {OUT}")


def print_detail(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        lp = r["last_period"]
        print(f"══ {r['ticker']} ({r['industry']}) · τιμή: {'ναι' if r['has_price'] else 'ΟΧΙ'}"
              f" · τελευταία περίοδος: {lp:%Y-%m-%d}" if pd.notna(lp) else
              f"══ {r['ticker']} · τελευταία περίοδος: καμία")
        for name in YIELDS.values():
            if f"{name} αιτία" in r and pd.notna(r[f"{name} αιτία"]):
                print(f"  {name:8s} {r[f'{name} αιτία']} · {r[f'{name} λεπτομέρεια'] or ''}")
        for part in str(r.get("συστατικά", "")).split(" | "):
            print(f"     {part}")
        print()


# ═════════════════════════════════════════════════════════════════════════
# 4. SELFTEST (synthetic data, no database, no network)
# ═════════════════════════════════════════════════════════════════════════

def _q(metric, ends, values, form="10-Q", days=91, lag=40, instant=False):
    ends = pd.to_datetime(ends)
    return pd.DataFrame({
        "value": values,
        "period_start": pd.NaT if instant else ends - pd.Timedelta(days=days),
        "period_end": ends,
        "filed_date": ends + pd.Timedelta(days=lag),
        "form_type": form,
        "metric": metric,
    })


def selftest() -> None:
    as_of = pd.Timestamp("2026-11-14")
    ends = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
    fresh_eps = _q(EPS, ends, [1.0, 1.0, 1.0, 1.0])

    # 1. the four states of one ingredient
    s = build_ttm(fresh_eps, per_share=True, value_col="v")
    assert judge("EPS", fresh_eps.iloc[0:0], s, as_of)[0] == "λείπει"
    assert judge("EPS", fresh_eps, s, as_of)[0] == "ok"
    old = _q(EPS, ["2024-09-30", "2024-12-31", "2025-01-31"], [1, 1, 1])
    assert judge("EPS", old, build_ttm(old, per_share=True, value_col="v"), as_of)[0] == "παλιό"
    gap = fresh_eps.drop(index=2)          # Q1 2026 missing → no TTM ends in 2026
    st, det = judge("EPS", gap, build_ttm(gap, per_share=True, value_col="v"), as_of)
    assert st == "δεν βγαίνει", (st, det)

    # 2. ingredients + cause chain
    shares = _q(Y.SHARES, ends, [1e9] * 4)
    ni = _q(Y.NET_INCOME, ends, [1e9] * 4)
    by = {EPS: fresh_eps, Y.SHARES: shares, Y.NET_INCOME: ni}
    ing = ingredients(by, as_of)
    assert ing["EPS"][0] == "ok" and ing["μετοχές"][0] == "ok"
    assert ing["CFO"][0] == "λείπει" and ing["χρέος"][0] == "ok"      # no debt tags → assumed 0
    last = pd.Timestamp("2026-06-30")
    assert cause("earnings_yield", ing, last, True, False, as_of) == (RUN_PROBLEM, "")
    assert cause("fcf_yield", ing, last, True, False, as_of)[0] == "CFO: λείπει"
    assert cause("ebit_yield", ing, last, True, False, as_of)[0] == "EBIT: λείπει"
    assert cause("fcf_yield", ing, last, True, True, as_of)[0] == EXPECTED
    assert cause("earnings_yield", ing, last, False, False, as_of)[0] == "χωρίς πρόσφατη τιμή (Yahoo)"
    c, d = cause("earnings_yield", ing, pd.Timestamp("2025-01-31"), True, False, as_of)
    assert c == "σταμάτησε να καταθέτει" and "2025-01-31" in d

    # 3. a fresh alternative tag is reported, the cause stays the same
    alt = _q("EarningsPerShareBasicAndDiluted", ends, [2.0] * 4)
    ing2 = ingredients({EPS: old, "EarningsPerShareBasicAndDiluted": alt}, as_of)
    assert ing2["EPS"][0] == "παλιό" and "EarningsPerShareBasicAndDiluted" in ing2["EPS"][1]

    # 4. CFO and capex both fresh but for different quarters → their own cause
    cfo = _q(Y.CFO, ends, [5.0] * 4)
    capex = _q(Y.CAPEX, ["2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31"], [1.0] * 4)
    ing3 = ingredients({**by, Y.CFO: cfo, Y.CAPEX: capex}, as_of)
    assert ing3["CFO"][0] == "ok" and ing3["capex"][0] == "ok" and ing3["FCF"][0] != "ok"
    assert cause("fcf_yield", ing3, last, True, False, as_of)[0] == "CFO και capex σε διαφορετικά τρίμηνα"

    # 5. balance-sheet ingredients: stale cash, fresh debt
    cash = _q(E.CASH, ["2024-12-31"], [5e8], instant=True)
    debt = _q("LongTermDebt", ["2026-06-30"], [2e9], instant=True)
    ing4 = ingredients({**by, E.CASH: cash, "LongTermDebt": debt,
                        EBIT: _q(EBIT, ends, [3e8] * 4)}, as_of)
    assert ing4["EBIT"][0] == "ok" and ing4["ταμείο"][0] == "παλιό" and ing4["χρέος"][0] == "ok"
    assert cause("ebit_yield", ing4, last, True, False, as_of)[0] == "ταμείο: παλιό"
    print("selftest: όλα τα τεστ πέρασαν ✓")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report(pd.read_csv(OUT, parse_dates=["last_period"]))
    elif "--ticker" in sys.argv:
        print_detail(run(only=sys.argv[sys.argv.index("--ticker") + 1:]))
    else:
        df = run()
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUT, index=False)
        report(df)
