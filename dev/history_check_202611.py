"""
dev/history_check_202611.py — READ-ONLY: before changing ev.py, three questions
about the HISTORY of the tags ingested in trading-shared-data v0.1.26.
Nothing is written to the database. No SEC, no Yahoo: only the database.

  A. Convertibles: are they INSIDE the ordinary debt tags, or OUTSIDE (to be added)?
     No outside truth exists for old dates, so we watch what happens when the
     convertibles change. Example, a company converts 1.0 bn of notes into shares:
         inside:  ordinary debt 5.0 → 4.0  (it fell with them)
         outside: ordinary debt 5.0 → 5.0  (it didn't move)
  B. Cash dips: a securities tag present in the quarters before and after, but
     missing on one date (NVDA-like: tagged only in 10-Qs). Then cash looks
     like it fell on that date although nothing happened:
         Q2  cash 20 + securities 30 = 50
         FY  cash 22 + (missing)     = 22   ← fake dip
         Q1  cash 21 + securities 31 = 52
  C. Securities already inside cash (TGT 2019-like): the company's own total
     "cash + short-term investments" equals cash ALONE, so adding securities
     counts them twice. Today's ev.py adds ShortTermInvestments, so this also
     checks the code in production.

Run from the repo root:
    python dev/history_check_202611.py --selftest   (1 sec, no database)
    python dev/history_check_202611.py              (~1–3 min, reads the database)
"""

import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from valuation import ev as EV

CASH = EV.CASH
COMBINED = "CashCashEquivalentsAndShortTermInvestments"
CURRENT = ["ShortTermInvestments", "MarketableSecuritiesCurrent", "DebtSecuritiesCurrent",
           "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent",
           "OtherShortTermInvestments",
           "DebtSecuritiesAvailableForSaleExcludingAccruedInterestCurrent"]
CONV_NC = ["ConvertibleLongTermNotesPayable", "ConvertibleDebtNoncurrent",
           "ConvertibleNotesPayableNoncurrent", "ConvertibleSubordinatedDebtNoncurrent"]
CONV_CUR = ["ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent",
            "ConvertibleSubordinatedDebtCurrent"]
CONV_TOTAL = ["ConvertibleDebt", "ConvertibleNotesPayable", "ConvertibleSubordinatedDebt"]
METRICS = EV.DEBT_METRICS + [CASH, COMBINED] + CURRENT + CONV_NC + CONV_CUR + CONV_TOTAL

MIN_MOVE = 50e6         # a convertible change smaller than this says nothing
MIN_MOVE_SHARE = 0.10   # ...nor one under 10% of the convertibles
TOL = 0.25              # "debt moved by the same amount" = within 25% of the move
MAX_GAP_DAYS = 200      # neighbours further apart than this are not "the quarter before/after"
MIN_DIP_SHARE = 0.05    # a missing amount under 5% of cash+securities is noise
SAME = 0.01             # "equal" totals = within 1%
CHUNK = 100


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def _pos(row: pd.Series, name: str) -> float:
    v = row.get(name)
    return float(v) if v is not None and pd.notna(v) and v > 0 else 0.0


def convertibles(row: pd.Series):
    """Non-current + current; only a total when neither part exists. None = no tag at all."""
    nc = next((_pos(row, n) for n in CONV_NC if pd.notna(row.get(n))), None)
    cur = next((_pos(row, n) for n in CONV_CUR if pd.notna(row.get(n))), None)
    if nc is None and cur is None:
        return next((_pos(row, n) for n in CONV_TOTAL if pd.notna(row.get(n))), None)
    return (nc or 0.0) + (cur or 0.0)


def securities(row: pd.Series) -> float | None:
    """The largest 'current' securities tag (the cash rule chosen on 04/11). None = no tag."""
    vals = [_pos(row, n) for n in CURRENT if pd.notna(row.get(n))]
    return max(vals) if vals else None


def wide(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One ticker's rows → table per balance-sheet date (earliest filing wins) + form of each date."""
    val, _ = EV._wide(rows)
    r = rows[rows["period_start"].isna()].sort_values("filed_date")
    form = r.drop_duplicates("period_end").set_index("period_end")["form_type"]
    return val.sort_index(), form.reindex(val.index)


def convertible_moves(val: pd.DataFrame, debt: pd.DataFrame) -> list[dict]:
    """
    Consecutive dates where the convertibles moved materially, and what ordinary
    debt did meanwhile. `debt` = EV.debt_series output (the rule in production).
    """
    reg = debt.set_index("period_end")["debt"] if not debt.empty else pd.Series(dtype=float)
    conv = val.apply(convertibles, axis=1).dropna()
    both = pd.DataFrame({"conv": conv, "reg": reg}).dropna().sort_index()
    out = []
    for (d1, a), (d2, b) in zip(both.iloc[:-1].iterrows(), both.iloc[1:].iterrows()):
        if (d2 - d1).days > MAX_GAP_DAYS:
            continue
        dc, dr = b["conv"] - a["conv"], b["reg"] - a["reg"]
        if abs(dc) < max(MIN_MOVE, MIN_MOVE_SHARE * max(a["conv"], b["conv"])):
            continue
        if abs(dr - dc) <= TOL * abs(dc):
            verdict = "μέσα"
        elif abs(dr) <= TOL * abs(dc):
            verdict = "έξω"
        else:
            verdict = "ασαφές"
        out.append({"date": d2, "dconv": dc, "dreg": dr, "verdict": verdict})
    return out


def conv_profile(val: pd.DataFrame, debt: pd.DataFrame) -> dict:
    """Counts per company: dates with convertibles, with no ordinary debt, with debt < convertibles."""
    conv = val.apply(convertibles, axis=1).dropna()
    conv = conv[conv > 0]
    reg = debt.set_index("period_end")["debt"] if not debt.empty else pd.Series(dtype=float)
    r = reg.reindex(conv.index)
    return {"dates": len(conv), "only_conv": int(r.isna().sum()),
            "reg_below_conv": int((r < conv).sum())}


def cash_dips(val: pd.DataFrame, form: pd.Series) -> list[dict]:
    """Dates with cash but no securities tag, while the dates before AND after have one."""
    if CASH not in val.columns:
        return []
    v = val[val[CASH].notna()]
    # in a Series a missing None becomes NaN, and NaN counts as "true": 0 means none
    sec = v.apply(securities, axis=1).astype(float).fillna(0.0)
    out = []
    dates = list(v.index)
    for i in range(1, len(dates) - 1):
        d0, d, d2 = dates[i - 1], dates[i], dates[i + 1]
        if (d - d0).days > MAX_GAP_DAYS or (d2 - d).days > MAX_GAP_DAYS:
            continue
        if sec[d] > 0 or sec[d0] <= 0 or sec[d2] <= 0:
            continue
        missing = (sec[d0] + sec[d2]) / 2
        total = (v.at[d0, CASH] + v.at[d2, CASH]) / 2 + missing
        if missing < MIN_DIP_SHARE * total:
            continue
        out.append({"date": d, "form": form.get(d), "missing": missing, "share": missing / total})
    return out


def inside_cash(val: pd.DataFrame) -> list[dict]:
    """Dates where the company's own cash+investments total says where the securities are."""
    out = []
    for d, row in val.iterrows():
        cash, comb = row.get(CASH), row.get(COMBINED)
        if pd.isna(cash) or pd.isna(comb) or comb <= 0:
            continue
        sec = securities(row)
        sti = _pos(row, "ShortTermInvestments")
        if not sec:
            continue
        if abs(cash / comb - 1) <= SAME:
            verdict = "μέσα στο ταμείο"
        elif abs((cash + sec) / comb - 1) <= SAME:
            verdict = "ξεχωριστά"
        else:
            verdict = "κανένα"
        out.append({"date": d, "verdict": verdict, "sti_counted_today": sti > 0})
    return out


def check_ticker(rows: pd.DataFrame) -> dict:
    val, form = wide(rows)
    debt = EV.debt_series(rows[rows["metric"].isin(EV.DEBT_METRICS)])
    return {"moves": convertible_moves(val, debt), "profile": conv_profile(val, debt),
            "dips": cash_dips(val, form), "inside": inside_cash(val)}


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return f"{x / 1e9:,.2f}"


def report(results: dict[str, dict], financial: set[str]) -> None:
    res = {t: r for t, r in results.items() if t not in financial}
    print(f"── {len(results)} εταιρείες, {len(res)} χωρίς τις χρηματοοικονομικές ──\n")

    print("── A. Μετατρέψιμα: όταν άλλαξαν, τι έκανε το κανονικό χρέος; ──")
    rows = []
    for t, r in res.items():
        p = r["profile"]
        if not p["dates"]:
            continue
        v = pd.Series([m["verdict"] for m in r["moves"]], dtype=object).value_counts()
        rows.append({"ticker": t, "ημ/νίες": p["dates"], "χωρίς_κανονικό": p["only_conv"],
                     "κανονικό<μετατρ.": p["reg_below_conv"], "μέσα": int(v.get("μέσα", 0)),
                     "έξω": int(v.get("έξω", 0)), "ασαφές": int(v.get("ασαφές", 0))})
    a = pd.DataFrame(rows)
    if a.empty:
        print("  καμία εταιρεία με μετατρέψιμα")
    else:
        print(f"  {len(a)} εταιρείες με μετατρέψιμα σε κάποια ημερομηνία")
        print(f"  κινήσεις: μέσα {a['μέσα'].sum()} · έξω {a['έξω'].sum()} · ασαφές {a['ασαφές'].sum()}")
        print(f"  εταιρείες με ΜΟΝΟ «έξω»: {int(((a['έξω'] > 0) & (a['μέσα'] == 0)).sum())}"
              f" · ΜΟΝΟ «μέσα»: {int(((a['μέσα'] > 0) & (a['έξω'] == 0)).sum())}"
              f" · και τα δύο: {int(((a['μέσα'] > 0) & (a['έξω'] > 0)).sum())}")
        print("\n  Όσες έχουν κίνηση, ημερομηνίες χωρίς κανονικό χρέος, ή κανονικό < μετατρέψιμα:")
        show = a[(a[["μέσα", "έξω", "ασαφές", "χωρίς_κανονικό", "κανονικό<μετατρ."]].sum(axis=1) > 0)]
        print(show.sort_values("ticker").to_string(index=False) if len(show) else "  καμία")
        print("\n  Οι κινήσεις αναλυτικά (δισ.), μέχρι 40:")
        moves = [(t, m) for t, r in sorted(res.items()) for m in r["moves"]]
        for t, m in moves[:40]:
            print(f"  {t:6s} {m['date'].date()}  μετατρέψιμα {bn(m['dconv']):>7s}"
                  f"  κανονικό {bn(m['dreg']):>7s}  {m['verdict']}")

    print("\n── B. Βουτιές ταμείου: χρεόγραφα πριν και μετά, όχι στη μέση ──")
    dips = [{"ticker": t, **d} for t, r in res.items() for d in r["dips"]]
    if not dips:
        print("  καμία")
    else:
        b = pd.DataFrame(dips)
        print(f"  {len(b)} ημερομηνίες σε {b['ticker'].nunique()} εταιρείες"
              f" · από 10-K: {int((b['form'] == '10-K').sum())} · από 10-Q: {int((b['form'] == '10-Q').sum())}")
        print(f"  μερίδιο που «χάνεται» από ταμείο+χρεόγραφα: διάμεσος {b['share'].median() * 100:.0f}%,"
              f" πάνω από 30%: {int((b['share'] > .3).sum())}")
        recent = b[b["date"] >= pd.Timestamp.today() - pd.Timedelta(days=550)]
        print(f"  στους τελευταίους 18 μήνες: {len(recent)} ημερομηνίες"
              f" ({', '.join(sorted(recent['ticker'].unique())[:30])})")
        print("\n  Οι μεγαλύτερες (δισ.), μέχρι 30:")
        for r in b.sort_values("missing", ascending=False).head(30).itertuples():
            print(f"  {r.ticker:6s} {r.date.date()} {r.form or '?':5s}  λείπουν {bn(r.missing):>7s}"
                  f"  ({r.share * 100:.0f}% του συνόλου)")

    print("\n── C. Χρεόγραφα ήδη μέσα στο ταμείο (σύμφωνα με το άθροισμα της ίδιας της εταιρείας) ──")
    ins = [{"ticker": t, **d} for t, r in res.items() for d in r["inside"]]
    if not ins:
        print("  καμία ημερομηνία με άθροισμα και χρεόγραφα")
    else:
        c = pd.DataFrame(ins)
        print(c["verdict"].value_counts().to_string())
        bad = c[c["verdict"] == "μέσα στο ταμείο"]
        print(f"\n  «μέσα στο ταμείο»: {len(bad)} ημερομηνίες σε {bad['ticker'].nunique()} εταιρείες"
              f" · ο σημερινός κώδικας τις μετράει δύο φορές σε {int(bad['sti_counted_today'].sum())}")
        per = bad.groupby("ticker")["date"].agg(["min", "max", "count"])
        for t, r in per.iterrows():
            print(f"  {t:6s} {r['min'].date()} – {r['max'].date()}  ({r['count']} ημ/νίες)")


# ═════════════════════════════════════════════════════════════════════════
# 3. RUN
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    from db import get_engine
    from valuation.pe import load_universe
    from valuation.ttm import load_metrics
    from yahoo_check_202611 import load_sectors

    eng = get_engine()
    tickers = load_universe(eng)
    sectors = load_sectors(tickers)
    financial = {t for t, s in sectors.items() if s == "Financial Services"}
    print(f"{len(tickers)} εταιρείες · ανάγνωση από τη βάση σε κομμάτια των {CHUNK}...")
    results = {}
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        rows = load_metrics(METRICS, part, eng)
        for t, g in rows.groupby("ticker"):
            try:
                results[t] = check_ticker(g)
            except Exception as e:
                print(f"  {t}: σφάλμα {str(e)[:80]}")
        print(f"  ... {min(i + CHUNK, len(tickers))}/{len(tickers)}")
    print()
    report(results, financial)


# ═════════════════════════════════════════════════════════════════════════
# 4. SELF-TEST: invented companies, no database
# ═════════════════════════════════════════════════════════════════════════

def _rows(table: dict, forms: dict | None = None) -> pd.DataFrame:
    """{period_end: {metric: value}} → rows as load_metrics returns them."""
    out = []
    for end, vals in table.items():
        f = (forms or {}).get(end, "10-Q")
        for m, v in vals.items():
            out.append({"ticker": "X", "value": float(v), "period_start": pd.NaT,
                        "period_end": pd.Timestamp(end), "filed_date": pd.Timestamp(end) + pd.Timedelta(days=35),
                        "form_type": f, "metric": m})
    return pd.DataFrame(out)


def selftest() -> None:
    LTD, CN = "LongTermDebt", "ConvertibleLongTermNotesPayable"
    MSC, AFS = "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"

    # A. inside: debt fell with the convertibles
    r = check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 4e9, CN: 0}}))
    assert [m["verdict"] for m in r["moves"]] == ["μέσα"]
    # outside: debt didn't move
    r = check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 5e9, CN: 0}}))
    assert [m["verdict"] for m in r["moves"]] == ["έξω"]
    # unclear: debt moved, but by a different amount
    r = check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 7e9, CN: 2e9}}))
    assert [m["verdict"] for m in r["moves"]] == ["ασαφές"]
    # small moves and dates a year apart say nothing
    r = check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 5e9, CN: 1.02e9}}))
    assert r["moves"] == []
    r = check_ticker(_rows({"2024-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 5e9, CN: 0}}))
    assert r["moves"] == []
    # DXCM-like: convertibles with no ordinary debt tag at all
    r = check_ticker(_rows({"2025-03-31": {CN: 1e9, CASH: 1e9}}))
    assert r["profile"] == {"dates": 1, "only_conv": 1, "reg_below_conv": 0}
    # debt below convertibles: they cannot be fully inside
    r = check_ticker(_rows({"2025-03-31": {LTD: 0.5e9, CN: 1e9}}))
    assert r["profile"]["reg_below_conv"] == 1
    # parts beat the total; a total alone counts
    assert convertibles(pd.Series({CN: 2.0, "ConvertibleDebtCurrent": 1.0, "ConvertibleDebt": 3.0})) == 3.0
    assert convertibles(pd.Series({"ConvertibleDebt": 3.0})) == 3.0
    assert convertibles(pd.Series({LTD: 3.0})) is None

    # B. NVDA-like dip: securities in the 10-Qs, missing in the 10-K
    t = {"2025-07-31": {CASH: 20e9, MSC: 30e9}, "2026-01-31": {CASH: 22e9},
         "2025-10-31": {CASH: 21e9, MSC: 31e9}, "2026-04-30": {CASH: 21e9, MSC: 31e9}}
    r = check_ticker(_rows(t, {"2026-01-31": "10-K"}))
    assert [(d["date"].date().isoformat(), d["form"]) for d in r["dips"]] == [("2026-01-31", "10-K")]
    # a different tag on the middle date is NOT a dip (the largest-tag rule takes it)
    t["2026-01-31"] = {CASH: 22e9, AFS: 30e9}
    assert check_ticker(_rows(t))["dips"] == []
    # a company that sold its securities for good is not a dip
    t = {"2025-07-31": {CASH: 20e9, MSC: 30e9}, "2025-10-31": {CASH: 50e9}, "2026-01-31": {CASH: 50e9}}
    assert check_ticker(_rows(t))["dips"] == []
    # tiny securities: noise
    t = {"2025-07-31": {CASH: 20e9, MSC: 0.1e9}, "2025-10-31": {CASH: 20e9}, "2026-01-31": {CASH: 20e9, MSC: 0.1e9}}
    assert check_ticker(_rows(t))["dips"] == []

    # C. TGT-like: the total equals cash alone → securities inside cash, counted twice today
    r = check_ticker(_rows({"2019-02-02": {CASH: 1.556e9, "ShortTermInvestments": 0.769e9, COMBINED: 1.556e9}}))
    assert [(d["verdict"], d["sti_counted_today"]) for d in r["inside"]] == [("μέσα στο ταμείο", True)]
    r = check_ticker(_rows({"2025-03-31": {CASH: 2e9, MSC: 3e9, COMBINED: 5e9}}))
    assert [(d["verdict"], d["sti_counted_today"]) for d in r["inside"]] == [("ξεχωριστά", False)]
    r = check_ticker(_rows({"2025-03-31": {CASH: 2e9, MSC: 3e9, COMBINED: 9e9}}))
    assert r["inside"][0]["verdict"] == "κανένα"

    # the report runs on all of it, and financials are left out
    res = {"A": check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 5e9, CN: 0}})),
           "B": check_ticker(_rows({"2025-03-31": {LTD: 5e9, CN: 1e9}, "2025-06-30": {LTD: 4e9, CN: 0}}))}
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(res, financial={"B"})
    assert "μέσα 0 · έξω 1" in buf.getvalue()
    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
