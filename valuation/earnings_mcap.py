"""
valuation/earnings_mcap.py — TODAY's earnings yield for companies with NO EPS
(saa v0.1.8, Nov 2026).

Eight S&P 500 companies never report a single diluted EPS figure to the SEC
(dev/eps_tags_202611.py, 20/11/2026): V, BRK-B, HSY, STZ, ERIE, KKR, ARES, BKR.
Most have several share classes (V: A/B/C, BRK: A/B, HSY: common/B) and report EPS
per class only, which the SEC's companyfacts leaves out; they have no weighted
share count either. Their NET INCOME is there. So, without any per-share number:

    earnings yield = net income TTM / market cap        (= 1 / P/E)
Example, V: 23.5 bn / 796 bn = 2.95%  →  P/E ≈ 34.

Net income = the part AVAILABLE TO COMMON shareholders when the company reports it
(NetIncomeLossAvailableToCommonStockholdersBasic), else total net income. P/E is the
common shareholder's ratio: ARES's 9-month net income is 378.6 m but only 302.6 m is
available to common (the rest goes to preferred and founders' units); with the total,
its P/E came out 36% below Yahoo's (dev/ep_mcap_check_202611.py, 24/11/2026).

The market cap comes from Yahoo, which adds up all share classes. Yahoo gives only
TODAY's market cap, so this is used only for today (the weekly peer table and the
card's current P/E), never for the 10-year history. It is a FALLBACK: a company
with an EPS-based yield never gets here.
"""

from datetime import date

import pandas as pd

from .ttm import MAX_TTM_AGE_DAYS, build_ttm, load_metrics

NET_INCOME = "NetIncomeLoss"
NI_COMMON = "NetIncomeLossAvailableToCommonStockholdersBasic"   # preferred when fresh
MAX_AS_OF_LAG_DAYS = 10        # Yahoo's market cap is today's: only for (about) today


def ni_ttm_latest(rows: pd.DataFrame, end: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    """Newest net income TTM known on `end`, if its period is ≤ MAX_TTM_AGE_DAYS old."""
    if rows is None or rows.empty:
        return None, None
    ttm = build_ttm(rows, value_col="v")
    if ttm.empty:
        return None, None
    known = ttm[pd.to_datetime(ttm["available"]) <= end]
    if known.empty:
        return None, None
    last = known.sort_values("period_end").iloc[-1]
    if (end - pd.Timestamp(last["period_end"])).days > MAX_TTM_AGE_DAYS:
        return None, None
    return float(last["v"]), pd.Timestamp(last["period_end"])


def pick_net_income(rows: pd.DataFrame, end: pd.Timestamp):
    """
    (ni_ttm, period_end, basis): net income available to common if its TTM is at
    least as recent as total net income's, else total net income.
    """
    by = {m: rows[rows["metric"] == m] for m in (NI_COMMON, NET_INCOME)} if not rows.empty else {}
    common, c_end = ni_ttm_latest(by.get(NI_COMMON), end)
    total, t_end = ni_ttm_latest(by.get(NET_INCOME), end)
    if common is not None and (t_end is None or c_end >= t_end - pd.Timedelta(days=7)):
        return common, c_end, "κοινοί μέτοχοι"
    if total is not None:
        return total, t_end, "καθαρά κέρδη"
    return None, None, None


def yahoo_market_cap(ticker: str) -> float | None:
    """Today's market cap from Yahoo (all share classes), or None."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        mc = t.fast_info["market_cap"]
    except Exception:
        mc = None
    if not mc:
        try:
            mc = t.info.get("marketCap")
        except Exception:
            mc = None
    return float(mc) if mc and mc > 0 else None


def earnings_yield_from_mcap(ticker: str, engine, end=None, market_cap: float | None = None) -> dict | None:
    """
    {"earnings_yield", "ni_ttm", "basis", "market_cap", "period_end"} or None.
    None when: `end` is not about today, no fresh net income TTM, or no market cap.
    """
    today = pd.Timestamp(date.today())
    end = pd.Timestamp(end) if end is not None else today
    if (today - end).days > MAX_AS_OF_LAG_DAYS:
        return None
    rows = load_metrics([NI_COMMON, NET_INCOME], [ticker], engine)
    ni, period_end, basis = pick_net_income(rows, end)
    if ni is None:
        return None
    mc = market_cap if market_cap is not None else yahoo_market_cap(ticker)
    if not mc:
        return None
    return {"earnings_yield": ni / mc, "ni_ttm": ni, "basis": basis, "market_cap": mc,
            "period_end": period_end}
