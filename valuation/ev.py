"""
valuation/ev.py — Enterprise value and EV/EBIT, point-in-time, for one stock.

WHY EV and not market cap: P/E and FCF yield look only at the shares. Two
companies with the same market cap of 100 bn, one with 50 bn of debt and
one with 20 bn of net cash, are not equally cheap: buying the WHOLE first
company costs 150 bn, the second 80 bn.
    EV = market cap + debt − cash
EBIT (operating income) is the profit BEFORE interest, i.e. the profit
that pays both lenders and shareholders — so it matches EV, which also
includes both.
    EV/EBIT 15  ↔  EBIT/EV ("earnings yield", Greenblatt) 6.7%
EBIT/EV is kept even when EBIT < 0 (EV/EBIT is then meaningless).

DEBT — the XBRL tags overlap, so they must be combined without counting
anything twice. On one balance-sheet date:
    LongTermDebt                    = long-term debt, current part INCLUDED
    LongTermDebtNoncurrent          = only the part due after 12 months
    LongTermDebtCurrent             = only the part due within 12 months
    DebtCurrent                     = ALL debt due within 12 months
                                      (current LT debt + short-term borrowings)
    LongTermDebtAndCapitalLeaseObligations[Current] = same, incl. finance leases
  Rule: debt = non-current part + current part, each from the first tag that
  exists. Example HD-like: Noncurrent 40 bn + DebtCurrent 3 bn = 43 bn.
  Only a TOTAL tag (LongTermDebt 43) → use it; never LongTermDebt + LongTermDebtCurrent
  (that would count the 3 bn twice).

CASH = cash & equivalents + short-term investments (if tagged).

Not included (standard simplifications, said out loud): operating leases,
preferred stock, minority interests. For BANKS and INSURERS EV/EBIT has no
meaning (their "debt" is their raw material — deposits), so the summary warns.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ttm import _fmt, asof_daily, build_ttm, load_metric, load_metrics, newest_only
from . import yields as Y

EBIT = "OperatingIncomeLoss"
CASH = "CashAndCashEquivalentsAtCarryingValue"
STI = "ShortTermInvestments"
DEBT_NONCURRENT = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
DEBT_CURRENT = ["DebtCurrent", "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
DEBT_TOTAL = ["LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities", "LongTermDebt"]
DEBT_METRICS = DEBT_NONCURRENT + DEBT_CURRENT + DEBT_TOTAL
FINANCIAL_SECTORS = {"Financial Services"}


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC
# ═════════════════════════════════════════════════════════════════════════

def _wide(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Balance-sheet rows (many metrics) → two tables indexed by period_end:
    values and filed dates, one column per metric. Earliest filing wins
    (later filings repeat the same date as a comparative column).
    """
    r = rows[rows["period_start"].isna()].dropna(subset=["value", "period_end", "filed_date"])
    r = r.sort_values("filed_date").drop_duplicates(["metric", "period_end"])
    val = r.pivot(index="period_end", columns="metric", values="value")
    filed = r.pivot(index="period_end", columns="metric", values="filed_date")
    return val, filed


def _first(row: pd.Series, names: list[str]):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return n, float(row[n])
    return None, None


def debt_series(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Total debt per balance-sheet date → columns period_end, available, debt, method.
    `rows` = output of load_metric for DEBT_METRICS, with a `metric` column.
    Dates where no rule applies are dropped, so the previous known value
    stays in force (up to MAX_TTM_AGE_DAYS) instead of becoming 0.
    """
    cols = ["period_end", "available", "debt", "method"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    val, filed = _wide(rows)
    has_long = val.columns.isin(DEBT_NONCURRENT + DEBT_TOTAL).any()
    out = []
    for pe, row in val.iterrows():
        nc_name, nc = _first(row, DEBT_NONCURRENT)
        cur_name, cur = _first(row, DEBT_CURRENT)
        tot_name, tot = _first(row, DEBT_TOTAL)
        if nc is not None:
            debt, used = nc + (cur or 0.0), [nc_name] + ([cur_name] if cur_name else [])
            method = f"{nc_name} + {cur_name}" if cur_name else f"{nc_name} (χωρίς βραχυπρόθεσμο)"
        elif tot is not None:
            debt, used, method = tot, [tot_name], tot_name
            # short-term borrowings = DebtCurrent − current part of LT debt (already inside the total)
            if pd.notna(row.get("DebtCurrent")) and pd.notna(row.get("LongTermDebtCurrent")):
                extra = max(float(row["DebtCurrent"]) - float(row["LongTermDebtCurrent"]), 0.0)
                if extra > 0:
                    debt += extra
                    used += ["DebtCurrent", "LongTermDebtCurrent"]
                    method += " + βραχυπρόθεσμος δανεισμός"
        elif cur is not None and not has_long:
            # Only for companies that NEVER report long-term debt. Otherwise a
            # missing tag on one date would turn 43 bn of debt into 5 bn.
            debt, used, method = cur, [cur_name], f"{cur_name} (μόνο βραχυπρόθεσμο)"
        else:
            continue
        available = max(filed.loc[pe, u] for u in used) + pd.Timedelta(days=1)
        out.append({"period_end": pe, "available": available, "debt": debt, "method": method})
    if not out:
        return pd.DataFrame(columns=cols)
    return newest_only(pd.DataFrame(out, columns=cols))


def cash_series(rows: pd.DataFrame) -> pd.DataFrame:
    """Cash + short-term investments per date → period_end, available, cash."""
    cols = ["period_end", "available", "cash"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    val, filed = _wide(rows)
    if CASH not in val.columns:
        return pd.DataFrame(columns=cols)
    v = val[val[CASH].notna()]
    sti = v[STI].fillna(0.0) if STI in v.columns else 0.0
    out = pd.DataFrame({"period_end": v.index, "cash": (v[CASH] + sti).to_numpy(),
                        "available": filed.loc[v.index, CASH].to_numpy()})
    out["available"] = pd.to_datetime(out["available"]) + pd.Timedelta(days=1)
    return newest_only(out[cols])


# ═════════════════════════════════════════════════════════════════════════
# 2. ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class EVHistory:
    ticker: str
    daily: pd.DataFrame     # FCFHistory.daily + debt, cash, ev, ebit_ttm, ev_ebit, ebit_yield
    status: str
    debt_note: str          # how debt was built (or "no debt tags → 0")


def _load_many(metrics, ticker, engine, as_of):
    return load_metrics(metrics, [ticker], engine, as_of=as_of)     # one query, not one per tag


def ev_history(ticker: str, engine, years: int = 10, as_of=None,
               base: "Y.FCFHistory | None" = None) -> EVHistory:
    """
    `base`: an FCFHistory already built for the same ticker/window (reuses its
    prices, shares and market cap instead of downloading them again).
    """
    base = base or Y.fcf_history(ticker, engine, years=years, as_of=as_of)
    d = base.daily.copy()
    if d.empty:
        return EVHistory(ticker, d, base.status, "")

    debt_rows = _load_many(DEBT_METRICS, ticker, engine, as_of)
    cash_rows = _load_many([CASH, STI], ticker, engine, as_of)
    ebit = build_ttm(load_metric(EBIT, [ticker], engine, as_of=as_of), value_col="ebit_ttm")

    debt = debt_series(debt_rows)
    if debt_rows.empty:
        d["debt"], note = 0.0, "κανένα tag χρέους στη βάση → υπόθεση 0 (έλεγξε ότι πράγματι δεν έχει χρέος)"
    else:
        d["debt"] = asof_daily(d.index, debt, "debt").to_numpy()
        note = debt["method"].iloc[-1] if not debt.empty else "tags χρέους υπάρχουν αλλά κανένας κανόνας δεν ταίριαξε"
    d["cash"] = asof_daily(d.index, cash_series(cash_rows), "cash").to_numpy()
    d["ev"] = d["mcap"] + d["debt"] - d["cash"]
    d["ebit_ttm"] = asof_daily(d.index, ebit, "ebit_ttm").to_numpy()
    d["ebit_yield"] = d["ebit_ttm"] / d["ev"]
    d["ev_ebit"] = np.where(d["ebit_ttm"] > 0, d["ev"] / d["ebit_ttm"], np.nan)

    last = d.iloc[-1]
    if ebit.empty:
        status = f"χωρίς EBIT στη βάση ({EBIT}): άλλο tag ή τράπεζα/ασφαλιστική"
    elif pd.isna(last["cash"]):
        status = f"χωρίς ταμείο στη βάση ({CASH})"
    elif pd.isna(last["debt"]):
        status = "χρέος άγνωστο (παλιό ή ελλιπές)"
    elif pd.isna(last["ebit_yield"]):
        status = "παλιά δεδομένα EBIT ή market cap"
    elif last["ebit_ttm"] <= 0:
        status = "αρνητικό EBIT (ζημιές): EV/EBIT χωρίς νόημα, βλ. EBIT/EV"
    else:
        status = "ok"
    return EVHistory(ticker, d, status, note)


def _sector(ticker: str) -> str | None:
    try:
        from .pe import get_classification
        return get_classification([ticker])[ticker]["sector"]
    except Exception:
        return None


def print_ev_summary(h: EVHistory) -> None:
    d = h.daily
    if d.empty:
        print(f"{h.ticker}: {h.status}")
        return
    last = d.iloc[-1]
    bn = 1e9
    print(f"── {h.ticker} — EV/EBIT (TTM) ── {d.index[-1].date()} ──")
    if _sector(h.ticker) in FINANCIAL_SECTORS:
        print("  ⚠️  Χρηματοοικονομική εταιρεία: το EV/EBIT δεν έχει νόημα εδώ (το «χρέος» είναι καταθέσεις).")
    if pd.notna(last["ev"]):
        print(f"  EV = market cap {last['mcap'] / bn:,.1f} + χρέος {last['debt'] / bn:,.1f}"
              f" − ταμείο {last['cash'] / bn:,.1f} = {last['ev'] / bn:,.1f} bn$")
        print(f"     χρέος από: {h.debt_note}")
    if pd.notna(last["ebit_ttm"]):
        print(f"  EBIT TTM: {last['ebit_ttm'] / bn:,.2f} bn$   → EBIT/EV {last['ebit_yield'] * 100:.2f}%")
    if h.status != "ok":
        print(f"  EV/EBIT: —  ({h.status})")
        return
    hist = d["ev_ebit"].dropna()
    pct = (hist > last["ev_ebit"]).mean() * 100     # days when it was MORE expensive
    print(f"  EV/EBIT: {last['ev_ebit']:.1f}   (P/E-λογική, αλλά με το χρέος μέσα)")
    print(f"  Διάμεσος της ίδιας ({hist.index[0].year}–): {hist.median():.1f}"
          f"  → σήμερα είναι φθηνότερη απ' ό,τι στο {pct:.0f}% των ημερών")
    net = last["debt"] - last["cash"]
    if pd.notna(net) and last["mcap"] > 0:
        why = ("το χρέος την κάνει ακριβότερη απ' ό,τι δείχνει το P/E" if net > 0
               else "το ταμείο την κάνει φθηνότερη απ' ό,τι δείχνει το P/E")
        print(f"  Καθαρό χρέος / market cap: {net / last['mcap'] * 100:+.0f}%  ({why})")


def debug_ev(ticker: str, engine) -> None:
    """Every balance-sheet tag on the latest dates, the debt rule chosen, and a Yahoo cross-check."""
    rows = _load_many(DEBT_METRICS + [CASH, STI], ticker, engine, None)
    if rows.empty:
        print(f"{ticker}: κανένα tag χρέους/ταμείου.")
    else:
        val, _ = _wide(rows)
        t = val.tail(3).T.map(lambda v: "—" if pd.isna(v) else _fmt(v))
        print(f"── {ticker}: tags ισολογισμού, τελευταίες 3 ημερομηνίες ──")
        print(t.to_string())
        ds = debt_series(rows[rows["metric"].isin(DEBT_METRICS)])
        if not ds.empty:
            x = ds.tail(3).copy()
            x["debt"] = x["debt"].map(_fmt)
            print("\n── Χρέος που υπολογίστηκε ──")
            print(x.to_string(index=False))

    h = ev_history(ticker, engine, years=3)
    print()
    print_ev_summary(h)
    if h.daily.empty or h.daily["ev"].dropna().empty:
        print("\n  (έλεγχος με Yahoo: δεν υπάρχει δικό μας EV για σύγκριση)")
        return
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        ours = h.daily["ev"].dropna().iloc[-1]
        print(f"\n  Έλεγχος με Yahoo: EV δικό μας {ours / 1e9:,.1f} bn vs Yahoo "
              f"{info['enterpriseValue'] / 1e9:,.1f} bn → {(ours / info['enterpriseValue'] - 1) * 100:+.1f}%")
        print(f"                   χρέος δικό μας {h.daily['debt'].iloc[-1] / 1e9:,.1f} vs Yahoo "
              f"{info.get('totalDebt', np.nan) / 1e9:,.1f} bn (το Yahoo συχνά μετράει και leases)")
    except Exception as e:
        print(f"\n  (έλεγχος με Yahoo δεν έγινε: {e})")
