"""
dev/ebit_gap_202611.py — READ-ONLY: step 2 of the roadmap (EBIT fallback).
Nothing is written. Only the database: no SEC, no Yahoo.

Question 1. WHY does a company have no EV/EBIT today? The card says only "—".
    Possible reasons, checked in this order:
      no OperatingIncomeLoss at all · OperatingIncomeLoss stopped (> 270 days)
      · EBIT ≤ 0 (losses) · debt unknown · cash unknown · ok
    Only non-financial companies that still file (revenue or profit < 270 days).

Question 2. Which REPLACEMENT for EBIT can we trust? Four formulas, built only
    from metrics already in the database, each tested on the companies that DO
    have OperatingIncomeLoss (there we know the right answer):
      F1  Revenues − CostsAndExpenses             (all costs in one line)
      F2  GrossProfit − OperatingExpenses         (operating costs after COGS)
      F3  GrossProfit − SG&A − R&D                (R&D = 0 if never reported)
      F4  NetIncome + IncomeTax + InterestExpense (bottom-up)
    Example, a company with OperatingIncomeLoss 10.0:
      F1 = 100 − 90 = 10.0 ✓   F4 = 6 + 2 + 1.5 = 9.5 (−5%, other income inside)
    A formula is usable only if it agrees with the real EBIT on most companies.

Question 3. For the companies of question 1 without EBIT: which formulas
    could fill the gap today?

Run from the repo root:
    python dev/ebit_gap_202611.py --selftest   (1 sec, no database)
    python dev/ebit_gap_202611.py              (~2–4 min, reads the database)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from valuation import ev as E
from valuation.ttm import MAX_TTM_AGE_DAYS, build_ttm

# (metric in the database, source tags allowed or None = any)
PARTS = {
    "OIL": ("OperatingIncomeLoss", None),
    "REV": ("Revenues", None),
    "CAE": ("OperatingExpenses", {"CostsAndExpenses"}),
    "OPEX": ("OperatingExpenses", {"OperatingExpenses"}),
    "GP": ("GrossProfit", None),
    "SGA": ("SellingGeneralAndAdministrativeExpense", None),
    "RD": ("ResearchAndDevelopmentExpense", None),
    "NI": ("NetIncomeLoss", None),
    "TAX": ("IncomeTaxExpenseBenefit", None),
    "INT": ("InterestExpense", {"InterestExpense", "InterestExpenseDebt"}),   # NOT the net-interest tag
}
# (required parts, optional parts, formula). An optional part is 0 only for a
# company that NEVER reports it; if it reports it on other dates, a date without
# it is skipped (R&D: 0 would overstate EBIT).
FORMULAS = {
    "F1 Έσοδα − σύνολο εξόδων": (["REV", "CAE"], [], lambda p: p["REV"] - p["CAE"]),
    "F2 Μικτό − λειτουργικά έξοδα": (["GP", "OPEX"], [], lambda p: p["GP"] - p["OPEX"]),
    "F3 Μικτό − SG&A − R&D": (["GP", "SGA"], ["RD"], lambda p: p["GP"] - p["SGA"] - p.get("RD", 0.0)),
    "F4 Καθαρά + φόρος + τόκοι": (["NI", "TAX", "INT"], [], lambda p: p["NI"] + p["TAX"] + p["INT"]),
}
WATCH = ["CAT", "ORCL", "PLTR", "CVX", "XOM", "KLAC", "CRM", "INTC", "DXCM", "FFIV", "PRGO"]
CHUNK = 100
BANDS = (0.05, 0.10)
VALIDATE_LAST = 4       # compare on each company's last 4 TTM dates where both exist


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def part_ttm(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One ticker's rows (with metric + source_tag) → {part: TTM frame}."""
    out = {}
    for name, (metric, sources) in PARTS.items():
        r = rows[rows["metric"] == metric]
        if sources is not None:
            r = r[r["source_tag"].isin(sources)]
        t = build_ttm(r, value_col="v") if not r.empty else pd.DataFrame()
        if not t.empty:
            out[name] = t
    return out


def formula_series(parts: dict[str, pd.DataFrame], formula: str) -> pd.DataFrame:
    """TTM of one formula on the period_ends where all its required parts exist."""
    need, optional, fn = FORMULAS[formula]
    if any(n not in parts for n in need):
        return pd.DataFrame(columns=["period_end", "available", "v"])
    m = None
    for n in need:
        t = parts[n][["period_end", "available", "v"]].rename(columns={"available": f"a_{n}", "v": n})
        m = t if m is None else m.merge(t, on="period_end")
    if m.empty:
        return pd.DataFrame(columns=["period_end", "available", "v"])
    for n in (o for o in optional if o in parts):
        t = parts[n][["period_end", "v"]].rename(columns={"v": n})
        m = m.merge(t, on="period_end", how="left")
        m = m[m[n].notna()]
    m["v"] = m.apply(lambda r: fn({k: r[k] for k in m.columns if k in PARTS}), axis=1)
    m["available"] = m[[c for c in m.columns if c.startswith("a_")]].max(axis=1)
    return m[["period_end", "available", "v"]].sort_values("period_end").reset_index(drop=True)


def age_days(frame: pd.DataFrame, today: pd.Timestamp):
    if frame is None or frame.empty:
        return None
    return (today - pd.Timestamp(frame["period_end"].max())).days


def last_value(frame: pd.DataFrame):
    if frame is None or frame.empty:
        return None
    return float(frame.sort_values("period_end")["v" if "v" in frame else frame.columns[-1]].iloc[-1])


def reason(parts: dict, debt: pd.DataFrame, debt_rows_empty: bool, cash: pd.DataFrame,
           today: pd.Timestamp) -> str:
    oil = parts.get("OIL")
    if oil is None:
        return "1 χωρίς OperatingIncomeLoss"
    if age_days(oil, today) > MAX_TTM_AGE_DAYS:
        return "2 OperatingIncomeLoss σταμάτησε"
    if last_value(oil) <= 0:
        return "3 EBIT ≤ 0 (ζημιές)"
    if not debt_rows_empty and (debt.empty or age_days(debt, today) > MAX_TTM_AGE_DAYS):
        return "4 χρέος άγνωστο ή παλιό"
    if cash.empty or age_days(cash, today) > MAX_TTM_AGE_DAYS:
        return "5 ταμείο άγνωστο ή παλιό"
    return "6 ok"


def still_filing(parts: dict, today: pd.Timestamp) -> bool:
    ages = [age_days(parts.get(n), today) for n in ("REV", "NI")]
    return any(a is not None and a <= MAX_TTM_AGE_DAYS for a in ages)


def validate(parts: dict) -> dict[str, list[float]]:
    """Per formula: relative deviations from OperatingIncomeLoss on the last common dates."""
    out = {}
    if "OIL" not in parts:
        return out
    oil = parts["OIL"][["period_end", "v"]].rename(columns={"v": "oil"})
    for f in FORMULAS:
        s = formula_series(parts, f)
        if s.empty:
            continue
        m = s.merge(oil, on="period_end").sort_values("period_end").tail(VALIDATE_LAST)
        m = m[m["oil"].abs() > 0]
        if not m.empty:
            out[f] = list((m["v"] / m["oil"] - 1).abs())
    return out


def fillable(parts: dict, today: pd.Timestamp) -> dict[str, float]:
    """Formulas with a fresh value today → {formula: value}."""
    out = {}
    for f in FORMULAS:
        s = formula_series(parts, f)
        if not s.empty and age_days(s, today) <= MAX_TTM_AGE_DAYS:
            out[f] = last_value(s)
    return out


def check_ticker(rows: pd.DataFrame, today: pd.Timestamp) -> dict:
    parts = part_ttm(rows)
    debt_rows = rows[rows["metric"].isin(E.ALL_DEBT_METRICS)]
    debt = E.debt_series(debt_rows)
    cash = E.cash_series(rows[rows["metric"].isin(E.CASH_METRICS)])
    return {"filing": still_filing(parts, today),
            "reason": reason(parts, debt, debt_rows.empty, cash, today),
            "valid": validate(parts), "fill": fillable(parts, today),
            "oil": last_value(parts.get("OIL")), "oil_age": age_days(parts.get("OIL"), today)}


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return "—" if x is None else f"{x / 1e9:,.2f}"


def report(res: dict[str, dict], financial: set[str]) -> None:
    nf = {t: r for t, r in res.items() if t not in financial}
    live = {t: r for t, r in nf.items() if r["filing"]}
    print(f"── {len(res)} εταιρείες · χωρίς χρηματοοικονομικές {len(nf)} · που καταθέτουν ακόμα {len(live)} ──\n")

    print("── 1. Γιατί λείπει σήμερα το EV/EBIT ──")
    s = pd.Series({t: r["reason"] for t, r in live.items()})
    for k, v in s.value_counts().sort_index().items():
        names = sorted(s[s == k].index)
        shown = ", ".join(names[:45]) + (" ..." if len(names) > 45 else "")
        print(f"  {k[2:]:32s} {v:4d}" + ("" if k.startswith("6") else f"   {shown}"))

    print("\n── 2. Αξιοπιστία κάθε τύπου, στις εταιρείες που ΕΧΟΥΝ OperatingIncomeLoss ──")
    print(f"   (σύγκριση στις τελευταίες {VALIDATE_LAST} κοινές ημερομηνίες TTM κάθε εταιρείας)")
    rows = []
    for f in FORMULAS:
        per_company = [max(v[f]) for v in (r["valid"] for r in nf.values()) if f in v]
        x = pd.Series(per_company, dtype=float)
        rows.append({"τύπος": f, "εταιρείες": len(x),
                     **{f"πάντα εντός {int(b * 100)}%": int((x <= b).sum()) for b in BANDS},
                     "διάμεσος χειρότερης": f"{x.median() * 100:.1f}%" if len(x) else "—"})
    print(pd.DataFrame(rows).to_string(index=False))
    print("   «πάντα εντός» = και στις 4 ημερομηνίες· «χειρότερη» = η μεγαλύτερη από τις 4 αποκλίσεις")

    print("\n── 3. Χωρίς EBIT σήμερα (λόγοι 1–2): ποιοι τύποι θα έδιναν τιμή ──")
    gap = {t: r for t, r in live.items() if r["reason"][0] in "12"}
    if not gap:
        print("  καμία")
    for f in FORMULAS:
        have = sorted(t for t, r in gap.items() if f in r["fill"])
        print(f"  {f:30s} {len(have):3d}/{len(gap)}   {', '.join(have[:30])}")
    none = sorted(t for t, r in gap.items() if not r["fill"])
    print(f"  {'κανένας τύπος':30s} {len(none):3d}/{len(gap)}   {', '.join(none[:40])}")

    print("\n── 4. Οι μετοχές του ελέγχου (TTM, δισ.) ──")
    for t in WATCH:
        r = res.get(t)
        if r is None:
            print(f"  {t:6s} δεν βρέθηκε")
            continue
        fills = " · ".join(f"{f[:2]} {bn(v)}" for f, v in r["fill"].items()) or "κανένας τύπος"
        age = f"{r['oil_age']} μέρες" if r["oil_age"] is not None else "—"
        print(f"  {t:6s} {r['reason'][2:]:30s} OIL {bn(r['oil']):>7s} ({age})   τύποι: {fills}")


# ═════════════════════════════════════════════════════════════════════════
# 3. RUN
# ═════════════════════════════════════════════════════════════════════════

def load_rows(metrics: list[str], tickers: list[str], engine) -> pd.DataFrame:
    """Like valuation.ttm.load_metrics, plus the source tag of each row."""
    from sqlalchemy import text
    sql = text("""
        SELECT f.ticker, f.value, f.period_start, f.period_end, f.filed_date,
               ft.name AS form_type, m.name AS metric, st.name AS source_tag
        FROM fundamentals f
        JOIN metric_lookup m       ON m.id  = f.metric_id
        JOIN form_type_lookup ft   ON ft.id = f.form_type_id
        JOIN source_tag_lookup st  ON st.id = f.source_tag_id
        WHERE m.name = ANY(:metrics) AND f.ticker = ANY(:tickers) AND ft.name IN ('10-Q', '10-K')
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"metrics": list(metrics), "tickers": list(tickers)}).fetchall()
    df = pd.DataFrame(rows, columns=["ticker", "value", "period_start", "period_end", "filed_date",
                                     "form_type", "metric", "source_tag"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("period_start", "period_end", "filed_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def main() -> None:
    from db import get_engine
    from valuation.pe import load_universe
    from yahoo_check_202611 import load_sectors

    eng = get_engine()
    tickers = load_universe(eng)
    financial = {t for t, s in load_sectors(tickers).items() if s == "Financial Services"}
    metrics = sorted({m for m, _ in PARTS.values()} | set(E.ALL_DEBT_METRICS) | set(E.CASH_METRICS))
    today = pd.Timestamp.today().normalize()
    print(f"{len(tickers)} εταιρείες · ανάγνωση από τη βάση σε κομμάτια των {CHUNK}...")
    res = {}
    for i in range(0, len(tickers), CHUNK):
        rows = load_rows(metrics, tickers[i:i + CHUNK], eng)
        for t, g in rows.groupby("ticker"):
            try:
                res[t] = check_ticker(g, today)
            except Exception as e:
                print(f"  {t}: σφάλμα {str(e)[:100]}")
        print(f"  ... {min(i + CHUNK, len(tickers))}/{len(tickers)}")
    print()
    report(res, financial)


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST: invented companies, no database
# ═════════════════════════════════════════════════════════════════════════

def _rows(table: dict) -> pd.DataFrame:
    """{(metric, source): {fiscal year end: value}} → annual 10-K rows."""
    out = []
    for (metric, source), years in table.items():
        for end, v in years.items():
            e = pd.Timestamp(end)
            out.append({"ticker": "X", "value": float(v), "period_start": e - pd.Timedelta(days=364),
                        "period_end": e, "filed_date": e + pd.Timedelta(days=50),
                        "form_type": "10-K", "metric": metric, "source_tag": source or metric})
    return pd.DataFrame(out)


def _bal(table: dict) -> pd.DataFrame:
    """Balance-sheet rows (no period_start)."""
    out = []
    for m, years in table.items():
        for end, v in years.items():
            e = pd.Timestamp(end)
            out.append({"ticker": "X", "value": float(v), "period_start": pd.NaT, "period_end": e,
                        "filed_date": e + pd.Timedelta(days=50), "form_type": "10-K",
                        "metric": m, "source_tag": m})
    return pd.DataFrame(out)


def selftest() -> None:
    today = pd.Timestamp("2026-06-30")   # 6 months after the invented FY 2025
    y = {"2025-12-31": None}
    inc = {("OperatingIncomeLoss", None): {"2025-12-31": 10.0, "2024-12-31": 8.0},
           ("Revenues", None): {"2025-12-31": 100.0, "2024-12-31": 90.0},
           ("OperatingExpenses", "CostsAndExpenses"): {"2025-12-31": 90.0, "2024-12-31": 82.0},
           ("OperatingExpenses", "OperatingExpenses"): {"2025-12-31": 30.0},
           ("GrossProfit", None): {"2025-12-31": 40.0},
           ("SellingGeneralAndAdministrativeExpense", None): {"2025-12-31": 25.0},
           ("ResearchAndDevelopmentExpense", None): {"2025-12-31": 5.0},
           ("NetIncomeLoss", None): {"2025-12-31": 6.0},
           ("IncomeTaxExpenseBenefit", None): {"2025-12-31": 2.0},
           ("InterestExpense", "InterestExpense"): {"2025-12-31": 1.5}}
    bal = {E.CASH: {"2025-12-31": 5.0}, "LongTermDebt": {"2025-12-31": 20.0}}
    rows = pd.concat([_rows(inc), _bal(bal)])
    del y

    # the four formulas, worked by hand
    p = part_ttm(rows)
    got = {f[:2]: last_value(formula_series(p, f)) for f in FORMULAS}
    assert got == {"F1": 10.0, "F2": 10.0, "F3": 10.0, "F4": 9.5}, got

    # validation: F1 exact on both years, F4 5% off on its one year
    v = validate(p)
    assert v["F1 Έσοδα − σύνολο εξόδων"] == [0.0, 0.0]
    assert abs(v["F4 Καθαρά + φόρος + τόκοι"][0] - 0.05) < 1e-9

    # the net-interest tag is NOT interest expense: F4 must not be built from it
    r2 = rows[~((rows["metric"] == "InterestExpense"))]
    r2 = pd.concat([r2, _rows({("InterestExpense", "InterestIncomeExpenseNet"): {"2025-12-31": -1.5}})])
    assert "F4 Καθαρά + φόρος + τόκοι" not in fillable(part_ttm(r2), today)

    # the two OperatingExpenses tags stay apart: total costs ≠ operating costs
    assert p["CAE"]["v"].iloc[-1] == 90.0 and p["OPEX"]["v"].iloc[-1] == 30.0

    # R&D: reported on other dates but not this one → F3 not built for that date;
    # never reported → counted as 0
    r3 = rows[rows["metric"] != "ResearchAndDevelopmentExpense"]
    assert last_value(formula_series(part_ttm(r3), "F3 Μικτό − SG&A − R&D")) == 15.0
    r4 = pd.concat([r3, _rows({("ResearchAndDevelopmentExpense", None): {"2024-12-31": 4.0}})])
    assert formula_series(part_ttm(r4), "F3 Μικτό − SG&A − R&D").empty

    # reasons, in order
    debt = E.debt_series(rows[rows["metric"].isin(E.ALL_DEBT_METRICS)])
    cash = E.cash_series(rows[rows["metric"].isin(E.CASH_METRICS)])
    assert reason(p, debt, False, cash, today) == "6 ok"
    no_oil = {k: v for k, v in p.items() if k != "OIL"}
    assert reason(no_oil, debt, False, cash, today) == "1 χωρίς OperatingIncomeLoss"
    assert reason(p, debt, False, cash, pd.Timestamp("2027-06-30")).startswith("2 ")
    neg = dict(p, OIL=p["OIL"].assign(v=-1.0))
    assert reason(neg, debt, False, cash, today).startswith("3 ")
    empty = pd.DataFrame(columns=["period_end", "available", "debt"])
    assert reason(p, empty, False, cash, today).startswith("4 ")
    assert reason(p, empty, True, cash, today) == "6 ok"        # no debt tags at all = 0 debt, as ev.py
    assert reason(p, debt, False, pd.DataFrame(columns=["period_end"]), today).startswith("5 ")

    # still filing: revenue < 270 days old
    assert still_filing(p, today) and not still_filing(p, pd.Timestamp("2027-06-30"))

    # the whole flow on one ticker, and the report
    r = check_ticker(rows, today)
    assert r["reason"] == "6 ok" and r["filing"] and len(r["fill"]) == 4
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report({"X": r, "Y": check_ticker(rows[rows["metric"] != "OperatingIncomeLoss"], today)}, set())
    out = buf.getvalue()
    assert "χωρίς OperatingIncomeLoss" in out and "4/1" not in out
    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
