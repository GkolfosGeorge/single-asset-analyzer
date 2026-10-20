"""
valuation/payouts.py — P/S and shareholder yield, point-in-time, for one stock.

P/S = market cap / revenue TTM
  Works for loss-makers, where P/E and EV/EBIT are meaningless.
  Example: market cap 50 bn, revenue 10 bn → P/S 5.
  Caveat: it ignores margins. P/S 5 is expensive for a supermarket (3%
  margin) and cheap for a software company (30% margin). Only compare a
  stock with ITS OWN history or with very similar companies.

SHAREHOLDER YIELD = dividend yield + buyback yield
  What the company handed back to shareholders in cash in the last year.
  Example: dividends 2 bn + buybacks 3 bn, market cap 100 bn → 5%.

  DIVIDENDS come from Yahoo, not the SEC: dividend per share with EX-DATE
  in the last 365 days / price. A SEC cash-flow TTM counts payments by
  fiscal quarter, and a 52/53-week quarter can end just before or just
  after a payment date. KO pays around April 1: Q1-2025 ended March 28
  (payment fell in Q2), Q1-2026 ended April 3 (payment fell in Q1), so the
  TTM to April 2026 held FIVE payments (11.0 bn vs 8.8 bn for the year).
  By ex-date there are always four for a quarterly payer.
  BUYBACKS stay on the SEC TTM: they happen throughout the quarter, not
  on one day, so quarter boundaries don't distort them.

  Buybacks are GROSS cash spent. Many companies (especially tech) pay
  employees in stock and buy shares back just to cancel that dilution, so
  the share count does not fall. That's why we also show the 12-month
  change in diluted shares: buybacks 3% but shares −0.5% means most of the
  buyback only offset stock compensation.
    net reduction in shares = shares 1 year ago / shares today − 1

Missing buyback tag: a company that has NEVER tagged buybacks is taken as
buying back 0. One that tagged them before but not recently is left
unknown (NaN), because it may have switched to another tag.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ttm import MAX_TTM_AGE_DAYS, asof_daily, build_ttm, load_metric
from . import yields as Y

REVENUE = "Revenues"
BUYBACKS = "PaymentsForRepurchaseOfCommonStock"


@dataclass
class PayoutHistory:
    ticker: str
    daily: pd.DataFrame      # close, shares, mcap, revenue_ttm, ps, div_ttm, buyback_ttm,
                             # div_yield, buyback_yield, sh_yield, share_change_12m
    notes: list[str]


def _ttm_or_zero(metric: str, ticker, engine, as_of, idx, col, notes, label):
    rows = load_metric(metric, [ticker], engine, as_of=as_of)
    if rows.empty:
        notes.append(f"{label}: κανένα tag ({metric}) → 0")
        return pd.Series(0.0, index=idx)
    s = asof_daily(idx, build_ttm(rows, value_col=col), col)
    if pd.isna(s.iloc[-1]):
        notes.append(f"{label}: παλιά δεδομένα (> {MAX_TTM_AGE_DAYS} μέρες): σταμάτησε ή άλλαξε tag")
    return s


def payout_history(ticker: str, engine, years: int = 10, as_of=None,
                   base: "Y.FCFHistory | None" = None) -> PayoutHistory:
    """`base`: an FCFHistory for the same ticker/window, to reuse prices and market cap."""
    base = base or Y.fcf_history(ticker, engine, years=years, as_of=as_of)
    d = base.daily.copy()
    notes: list[str] = []
    if d.empty:
        return PayoutHistory(ticker, d, [base.status])
    idx = d.index

    rev = build_ttm(load_metric(REVENUE, [ticker], engine, as_of=as_of), value_col="revenue_ttm")
    d["revenue_ttm"] = asof_daily(idx, rev, "revenue_ttm").to_numpy()
    d["ps"] = np.where(d["revenue_ttm"] > 0, d["mcap"] / d["revenue_ttm"], np.nan)
    if rev.empty:
        notes.append(f"P/S: χωρίς έσοδα στη βάση ({REVENUE}): άλλο tag")

    d["div_yield"] = d["div_ps_365d"] / d["close"]           # Yahoo, by ex-date
    d["div_ttm"] = d["div_ps_365d"] * d["shares"]            # for display, in $
    d["buyback_ttm"] = _ttm_or_zero(BUYBACKS, ticker, engine, as_of, idx, "buyback_ttm", notes,
                                    "Buybacks").to_numpy()
    d["buyback_yield"] = d["buyback_ttm"] / d["mcap"]
    d["sh_yield"] = d["div_yield"] + d["buyback_yield"]
    # share_change_12m comes from yields.fcf_history (period vs period, no spikes)
    return PayoutHistory(ticker, d, notes)


def _own_percentile(series: pd.Series, value: float, higher_is_cheaper: bool) -> tuple[float, float]:
    h = series.dropna()
    cheaper_than = (h < value).mean() if higher_is_cheaper else (h > value).mean()
    return h.median(), cheaper_than * 100


def print_payout_summary(h: PayoutHistory) -> None:
    d = h.daily
    if d.empty:
        print(f"{h.ticker}: {'; '.join(h.notes)}")
        return
    last = d.iloc[-1]
    print(f"── {h.ticker} — P/S και επιστροφές στους μετόχους (TTM) ── {d.index[-1].date()} ──")

    if pd.notna(last["ps"]):
        med, pct = _own_percentile(d["ps"], last["ps"], higher_is_cheaper=False)
        print(f"  P/S: {last['mcap'] / 1e9:,.1f} / έσοδα {last['revenue_ttm'] / 1e9:,.1f} bn = {last['ps']:.2f}"
              f"   (διάμεσος της ίδιας {med:.2f} → φθηνότερη απ' ό,τι στο {pct:.0f}% των ημερών)")
    else:
        print("  P/S: —")

    if pd.notna(last["sh_yield"]):
        print(f"  Shareholder yield: {last['sh_yield'] * 100:.2f}%"
              f" = μερίσματα {last['div_yield'] * 100:.2f}% ({last['div_ps_365d']:.2f}$/μετοχή,"
              f" ≈ {last['div_ttm'] / 1e9:,.2f} bn)"
              f" + buybacks {last['buyback_yield'] * 100:.2f}% ({last['buyback_ttm'] / 1e9:,.2f} bn)")
    else:
        print("  Shareholder yield: —  (βλ. σημειώσεις)")

    ch = last["share_change_12m"]
    if pd.notna(ch):
        trend = "λιγότερες" if ch > 0 else "περισσότερες"
        print(f"  Μετοχές σε 12 μήνες: {abs(ch) * 100:.1f}% {trend}")
        if pd.notna(last["buyback_yield"]) and last["buyback_yield"] > 0.01 and ch < last["buyback_yield"] / 3:
            print("  ⚠️  Τα buybacks είναι πολύ μεγαλύτερα από τη μείωση των μετοχών: μεγάλο μέρος τους"
                  " απλώς αντισταθμίζει μετοχές που δίνονται σε εργαζόμενους.")
    for n in h.notes:
        print(f"  ℹ️  {n}")
