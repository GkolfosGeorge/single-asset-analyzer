"""
valuation/peer_table.py — TODAY's yields of every company, for the comparison
with peers (Phase 2, Nov 2026). Built once a week by jobs/build_peer_table.py
on GitHub Actions; the mail only READS the result.

One row per company: its classification and three yields, the same numbers
the card shows, turned upside down so that "higher = cheaper":
    earnings_yield = EPS TTM / price       (1 / P/E;    negative = losses)
    ebit_yield     = EBIT TTM / EV         (1 / EV/EBIT)
    fcf_yield      = FCF TTM / market cap
Example: P/E 20 → earnings_yield 5%. A loss-maker gets a negative yield and
simply ranks dearest, without special cases.

Why weekly and not in every mail: the card takes ~7 s per company, so all
~700 take ~90 min (dev/peer_timing_202611.py, 13/11/2026). The mail compares
the company's yield of TODAY with its peers' yields of up to a week ago.
"""

from datetime import date, timedelta

import pandas as pd

from . import ev as E
from . import pe as P
from . import yields as Y
from .ttm import load_prices

YEARS = 2                    # enough for TTM and splits; the card's 10-year history isn't needed here
STALE_PRICE_DAYS = 10        # no price for 10 days = no longer traded
COLUMNS = ["ticker", "as_of", "sector", "industry", "earnings_yield", "ebit_yield", "fcf_yield",
           "financial", "note"]


def _last(d: pd.DataFrame, col: str, end: pd.Timestamp):
    """Newest value of `col`, only if the price behind it is recent."""
    if d is None or d.empty or col not in d.columns:
        return None
    s = d[col].dropna()
    if s.empty or s.index[-1] < end - timedelta(days=STALE_PRICE_DAYS):
        return None
    return float(s.iloc[-1])


def company_yields(ticker: str, engine, end: pd.Timestamp | None = None) -> dict:
    """Today's three yields of one company (None where it can't be computed)."""
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=YEARS)
    out = {"earnings_yield": None, "ebit_yield": None, "fcf_yield": None, "financial": False, "note": ""}
    notes = []

    try:
        eps = P.load_eps([ticker], engine)
        if not eps.empty:
            dl_start = (start - timedelta(days=P.SPLIT_LOOKBACK_DAYS)).date()
            close, splits = load_prices([ticker], dl_start, (end + timedelta(days=1)).date())
            if ticker in close.columns:
                sp = splits[ticker] if ticker in splits.columns else None
                d = P.daily_pe(close[ticker], P.build_ttm(eps, sp))
                d = d[d.index <= end].astype({"close": float, "eps_ttm": float})
                d["earnings_yield"] = d["eps_ttm"] / d["close"]
                out["earnings_yield"] = _last(d, "earnings_yield", end)
    except Exception as e:
        notes.append(f"P/E: {type(e).__name__}")

    try:
        fcf = Y.fcf_history(ticker, engine, years=YEARS)
        out["fcf_yield"] = _last(fcf.daily, "fcf_yield", end)
        ev = E.ev_history(ticker, engine, years=YEARS, base=fcf)
        out["ebit_yield"] = _last(ev.daily, "ebit_yield", end)
        # same test as the card: banks/insurers have neither operating income nor capex
        out["financial"] = bool(ev.status.startswith("χωρίς EBIT") and fcf.status.startswith("χωρίς capex"))
    except Exception as e:
        notes.append(f"FCF/EV: {type(e).__name__}")

    out["note"] = "; ".join(notes)
    return out


def build(engine, tickers: list[str], classification: dict[str, dict],
          end: pd.Timestamp | None = None, progress_every: int = 50) -> pd.DataFrame:
    end = pd.Timestamp(end) if end is not None else pd.Timestamp(date.today())
    rows = []
    for i, t in enumerate(tickers, 1):
        c = classification.get(t, {})
        rows.append({"ticker": t, "as_of": end.date().isoformat(),
                     "sector": c.get("sector"), "industry": c.get("industry"),
                     **company_yields(t, engine, end)})
        if progress_every and i % progress_every == 0:
            print(f"  ... {i}/{len(tickers)}", flush=True)
    return pd.DataFrame(rows, columns=COLUMNS)


def coverage(table: pd.DataFrame) -> int:
    """Companies with at least one yield."""
    y = table[["earnings_yield", "ebit_yield", "fcf_yield"]]
    return int(y.notna().any(axis=1).sum())
