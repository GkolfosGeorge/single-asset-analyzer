"""
valuation/yields.py — Market cap and FCF yield, point-in-time, for one stock.

MARKET CAP = split-adjusted close × diluted shares in TODAY's share units.
  Shares: WeightedAverageNumberOfDilutedSharesOutstanding of the latest
  quarter (or the full year right after a 10-K). It is an AVERAGE over the
  period, so for heavy buyers it lags a little: a company retiring 1% of
  its shares per quarter shows ~0.5% more shares than it really has.
  We use it because it covers 764 companies vs 603 for the period-end count.

FCF (free cash flow) TTM = operating cash flow TTM − capex TTM
  Example: CFO 136 bn, capex 65 bn → FCF 71 bn.
  Both come from the cash-flow statement, i.e. from CUMULATIVE 10-Q figures
  that ttm.py turns into quarters (H1 − Q1 = Q2, ...).

FCF YIELD = FCF TTM / market cap  (the inverse of P/FCF)
  71 bn / 3,000 bn = 2.4%  ↔  P/FCF = 42.
  Unlike P/E, a negative yield is still meaningful (the company burns cash),
  so it is kept, not hidden.

No peers here: comparison with the sector comes in Βήμα 3.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .ttm import (MAX_TTM_AGE_DAYS, _build_quarters, _fmt, asof_daily, build_latest,
                  build_ttm, load_metric, load_prices, newest_only, yoy_change)

CFO = "NetCashProvidedByUsedInOperatingActivities"
CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment"
SHARES = "WeightedAverageNumberOfDilutedSharesOutstanding"
SPLIT_LOOKBACK_DAYS = 500


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC
# ═════════════════════════════════════════════════════════════════════════

def fcf_ttm(cfo_rows: pd.DataFrame, capex_rows: pd.DataFrame) -> pd.DataFrame:
    """
    TTM FCF per period, columns: period_end, available, cfo_ttm, capex_ttm, fcf_ttm.
    A period counts only when BOTH CFO and capex TTM exist for the same
    period_end; it becomes usable when the later of the two is filed
    (normally the same filing).
    """
    cols = ["period_end", "available", "cfo_ttm", "capex_ttm", "fcf_ttm"]
    cfo = build_ttm(cfo_rows, value_col="cfo_ttm")
    capex = build_ttm(capex_rows, value_col="capex_ttm")
    if cfo.empty or capex.empty:
        return pd.DataFrame(columns=cols)
    m = cfo.merge(capex, on="period_end", suffixes=("_cfo", "_capex"))
    if m.empty:
        return pd.DataFrame(columns=cols)
    m["available"] = m[["available_cfo", "available_capex"]].max(axis=1)
    m["fcf_ttm"] = m["cfo_ttm"] - m["capex_ttm"]
    return newest_only(m[cols])


@dataclass
class FCFHistory:
    ticker: str
    daily: pd.DataFrame      # close, shares, mcap, cfo_ttm, capex_ttm, fcf_ttm, fcf_yield
    status: str              # "ok" or why FCF yield is missing


def build_daily(close: pd.Series, shares: pd.DataFrame, fcf: pd.DataFrame) -> pd.DataFrame:
    """Daily close + values known that day → market cap and FCF yield."""
    close = close.dropna().astype(float)
    d = pd.DataFrame({"close": close})
    d["shares"] = asof_daily(d.index, shares, "shares").to_numpy()
    d["mcap"] = d["close"] * d["shares"]
    for c in ("cfo_ttm", "capex_ttm", "fcf_ttm"):
        d[c] = asof_daily(d.index, fcf, c).to_numpy() if not fcf.empty else np.nan
    d["fcf_yield"] = d["fcf_ttm"] / d["mcap"]
    return d


# ═════════════════════════════════════════════════════════════════════════
# 2. ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════

def fcf_history(ticker: str, engine, years: int = 10, as_of=None) -> FCFHistory:
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=years)
    dl_start = (start - timedelta(days=SPLIT_LOOKBACK_DAYS)).date()

    close, splits = load_prices([ticker], dl_start, (end + timedelta(days=1)).date())
    if ticker not in close.columns or close[ticker].dropna().empty:
        return FCFHistory(ticker, pd.DataFrame(), "χωρίς τιμές (yfinance)")
    sp = splits[ticker] if ticker in splits.columns else None
    rows = {m: load_metric(m, [ticker], engine, as_of=as_of) for m in (CFO, CAPEX, SHARES)}

    shares = build_latest(rows[SHARES], sp, shares=True, value_col="shares")
    fcf = fcf_ttm(rows[CFO], rows[CAPEX])
    d = build_daily(close[ticker], shares, fcf)
    # positive = fewer shares than one year earlier (period vs period)
    d["share_change_12m"] = asof_daily(d.index, yoy_change(shares, "shares"), "chg").to_numpy()
    d = d[(d.index >= start) & (d.index <= end)]

    if rows[CAPEX].empty:
        status = f"χωρίς capex στη βάση ({CAPEX}): η εταιρεία χρησιμοποιεί άλλο tag"
    elif rows[CFO].empty:
        status = "χωρίς CFO στη βάση"
    elif shares.empty:
        status = "χωρίς πλήθος μετοχών στη βάση"
    elif d.empty or pd.isna(d["fcf_yield"].iloc[-1]):
        status = f"παλιά ή ελλιπή δεδομένα (> {MAX_TTM_AGE_DAYS} μέρες ή CFO/capex σε διαφορετικά τρίμηνα)"
    else:
        status = "ok"
    return FCFHistory(ticker, d, status)


def print_fcf_summary(h: FCFHistory) -> None:
    d = h.daily.dropna(subset=["close"]) if not h.daily.empty else h.daily
    if d.empty:
        print(f"{h.ticker}: {h.status}")
        return
    last = d.iloc[-1]
    print(f"── {h.ticker} — FCF yield (TTM) ── {d.index[-1].date()} ──")
    if pd.notna(last["mcap"]):
        print(f"  Market cap: {last['close']:.2f}$ × {last['shares'] / 1e6:,.0f}M μετοχές"
              f" = {last['mcap'] / 1e9:,.1f} bn$")
    if h.status != "ok":
        print(f"  FCF yield: —  ({h.status})")
        return
    print(f"  FCF TTM: CFO {last['cfo_ttm'] / 1e9:,.2f} bn − capex {last['capex_ttm'] / 1e9:,.2f} bn"
          f" = {last['fcf_ttm'] / 1e9:,.2f} bn$")
    y = last["fcf_yield"]
    pfcf = f"P/FCF {1 / y:.1f}" if y > 0 else "αρνητικό FCF: η εταιρεία καίει μετρητά"
    print(f"  FCF yield: {y * 100:.2f}%   ({pfcf})")
    hist = d["fcf_yield"].dropna()
    if len(hist) > 250:
        pct = (hist < y).mean() * 100
        # higher yield = cheaper, so "yield above X% of days" = "cheaper than on X% of days"
        print(f"  Διάμεσος της ίδιας ({hist.index[0].year}–): {hist.median() * 100:.2f}%"
              f"  → σήμερα είναι φθηνότερη απ' ό,τι στο {pct:.0f}% των ημερών")
        if pct <= 5:
            print("  ⚠️  Κοντά στο ακριβότερο σημείο του διαστήματος. Δες το διάγραμμα: αν το CFO"
                  " ανεβαίνει και πέφτει το FCF, αιτία είναι οι επενδύσεις (capex), όχι μόνο η τιμή.")


def debug_fcf(ticker: str, engine, n: int = 5) -> None:
    """
    How the latest FCF and market cap are built, step by step — plus an
    independent check of the market cap against Yahoo's own number.
    """
    h = fcf_history(ticker, engine, years=2)
    _, splits = load_prices([ticker], (date.today() - timedelta(days=3 * 365)))
    sp = splits[ticker] if ticker in splits.columns else None

    for name, metric in (("CFO", CFO), ("Capex", CAPEX)):
        rows = load_metric(metric, [ticker], engine)
        print(f"\n── {ticker} {name}: τελευταία τρίμηνα (source = από πού βγήκε) ──")
        if rows.empty:
            print(f"  καμία γραμμή {metric}")
            continue
        _, _, q, log = _build_quarters(rows)
        t = q.tail(n).copy()
        t["value"] = t["value"].map(_fmt)
        print(t[["period_start", "period_end", "filed_date", "value", "source"]].to_string(index=False))
        bad = [l for l in log[-6:] if "✗" in l or "⚠️" in l]
        for l in bad:
            print("  " + l)

    shares = build_latest(load_metric(SHARES, [ticker], engine), sp, shares=True, value_col="shares")
    print(f"\n── {ticker} μετοχές (diluted, μέσος όρος περιόδου, σε σημερινές μετοχές) ──")
    bad = pd.DataFrame(shares.attrs.get("dropped", []))
    if len(bad):
        print(f"  ⚠️ απορρίφθηκαν {len(bad)} γραμμές ως λάθος κλίμακας (π.χ. χιλιάδες αντί για μονάδες):")
        print(bad[["period_end", "filed_date", "value"]].to_string(index=False))
    if not shares.empty:
        s = shares.tail(3).copy()
        s["shares"] = (s["shares"] / 1e6).map("{:,.0f}M".format)
        print(s.to_string(index=False))

    print()
    print_fcf_summary(h)
    try:
        import yfinance as yf
        yahoo = yf.Ticker(ticker).fast_info["market_cap"]
        ours = h.daily["mcap"].dropna().iloc[-1]
        print(f"\n  Έλεγχος market cap: δικό μας {ours / 1e9:,.1f} bn vs Yahoo {yahoo / 1e9:,.1f} bn"
              f"  → διαφορά {(ours / yahoo - 1) * 100:+.1f}%")
    except Exception as e:
        print(f"\n  (έλεγχος με Yahoo δεν έγινε: {e})")
