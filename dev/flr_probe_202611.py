"""
Why is FLR's EV +15% vs Yahoo? (Φάση 1, βήμα 4β, 25/11/2026). Read-only.

Hypothesis: buybacks. We use the diluted WEIGHTED AVERAGE of the last quarter
(e.g. 162M); Yahoo uses the shares that exist TODAY (e.g. 155M). For a company
with little debt, EV ≈ market cap, so a +5% share gap lands almost whole in EV.

  Part 1  the EV gap split into its pieces (from data/yahoo_check_202611*.csv,
          no network): market cap, debt, cash. Both rounds (03/11, 25/11).
  Part 2  shares in the database: diluted average per quarter and the
          period-end count (CommonStockSharesOutstanding), next to Yahoo's
          "Ordinary Shares Number" per quarter and today's count.
  Part 3  what-if: our EV with period-end shares instead of the average.
          If that lands within 5% of Yahoo → explained by the share basis.

Usage (from the single_asset_analyzer folder):
    python dev/flr_probe_202611.py              # FLR
    python dev/flr_probe_202611.py FLR TPR WRB  # also the other buyback names
    python dev/flr_probe_202611.py --selftest
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

ROUNDS = [("03/11", "data/yahoo_check_202611_r1.csv"), ("25/11", "data/yahoo_check_202611.csv")]
DILUTED = "WeightedAverageNumberOfDilutedSharesOutstanding"
CSO = "CommonStockSharesOutstanding"
OK = 0.05
QUARTER_DAYS = (80, 100)


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (tested in --selftest)
# ═════════════════════════════════════════════════════════════════════════

def dev(ours, theirs) -> float:
    if ours is None or theirs is None or pd.isna(ours) or pd.isna(theirs) or theirs == 0:
        return np.nan
    return ours / theirs - 1


def decompose(r: dict) -> dict:
    """EV gap = mcap gap + debt gap − cash gap, each also as % of Yahoo's EV (ex leases)."""
    parts = {
        "market cap": r["mcap"] - r["y_mcap"],
        "χρέος": r["debt"] - r["y_debt_xl"],
        "ταμείο": -(r["cash"] - r["y_cash"]),
    }
    base = r["y_ev_xl"]
    return {k: (v, v / base if base else np.nan) for k, v in parts.items()}


def what_if_ev(r: dict, shares: float) -> float:
    """Our EV if the market cap used `shares` instead of the period average."""
    if shares is None or pd.isna(shares):
        return np.nan
    return shares * r["close"] + r["debt"] - r["cash"]


def verdict(dev_now: float, dev_whatif: float, biggest: str) -> str:
    if pd.isna(dev_now):
        return "δεν συγκρίνεται (λείπει EV)"
    if abs(dev_now) <= OK:
        return "εντός 5%, τίποτα να εξηγηθεί"
    if pd.notna(dev_whatif) and abs(dev_whatif) <= OK:
        return (f"ΕΞΗΓΗΣΗ: βάση μετοχών (επαναγορές). Με μετοχές τέλους περιόδου "
                f"{dev_now * 100:+.0f}% → {dev_whatif * 100:+.0f}%")
    if pd.notna(dev_whatif) and abs(dev_whatif) < abs(dev_now):
        return (f"ΜΕΡΙΚΗ εξήγηση από τις μετοχές ({dev_now * 100:+.0f}% → {dev_whatif * 100:+.0f}%)· "
                f"το υπόλοιπο από: {biggest}")
    return f"ΟΧΙ οι μετοχές· το μεγαλύτερο κομμάτι είναι: {biggest}"


def quarterly_diluted(rows: pd.DataFrame) -> pd.DataFrame:
    """3-month diluted averages, latest filing per period_end."""
    if rows.empty:
        return rows
    d = rows.copy()
    d["days"] = (d["period_end"] - d["period_start"]).dt.days
    d = d[d["days"].between(*QUARTER_DAYS)]
    return d.sort_values("filed_date").groupby("period_end").tail(1).sort_values("period_end")


def period_end_counts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    return rows.sort_values("filed_date").groupby("period_end").tail(1).sort_values("period_end")


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x / 1e9:,.2f} bn"


def mn(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x / 1e6:,.1f}M"


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.1f}%"


def part1(ticker: str) -> dict | None:
    print(f"\n══ 1. {ticker}: από τι αποτελείται η απόκλιση του EV (χωρίς leases) ══")
    last = None
    for label, path in ROUNDS:
        if not os.path.exists(path):
            print(f"  {label}: λείπει το {path}")
            continue
        df = pd.read_csv(path)
        hit = df[df["ticker"] == ticker]
        if hit.empty:
            print(f"  {label}: η {ticker} δεν είναι στο αρχείο")
            continue
        r = hit.iloc[0].to_dict()
        need = ["close", "shares", "mcap", "debt", "cash", "ev", "y_shares", "y_mcap",
                "y_debt_xl", "y_cash", "y_ev_xl"]
        if any(pd.isna(r.get(k)) for k in need):
            print(f"  {label}: λείπουν στοιχεία ({', '.join(k for k in need if pd.isna(r.get(k)))})")
            continue
        print(f"\n  ── γύρος {label} · δικό μας τρίμηνο {str(r.get('period_end', ''))[:10]} · τιμή {r['close']:.2f}")
        print(f"     {'':<12}{'δικό μας':>14}{'Yahoo':>14}{'απόκλιση':>10}")
        print(f"     {'μετοχές':<12}{mn(r['shares']):>14}{mn(r['y_shares']):>14}{pct(dev(r['shares'], r['y_shares'])):>10}")
        for name, a, b in (("market cap", "mcap", "y_mcap"), ("χρέος", "debt", "y_debt_xl"),
                           ("ταμείο", "cash", "y_cash"), ("EV", "ev", "y_ev_xl")):
            print(f"     {name:<12}{bn(r[a]):>14}{bn(r[b]):>14}{pct(dev(r[a], r[b])):>10}")
        print("     Πόσο από την απόκλιση του EV φέρνει κάθε κομμάτι:")
        for k, (v, p) in decompose(r).items():
            print(f"       {k:<11} {v / 1e9:+,.2f} bn  = {pct(p)} του EV του Yahoo")
        last = r
    return last


def part2(ticker: str, engine) -> dict:
    from valuation.ttm import load_metrics
    print(f"\n══ 2. {ticker}: μετοχές ανά τρίμηνο ══")
    rows = load_metrics([DILUTED, CSO], [ticker], engine)
    dil = quarterly_diluted(rows[rows["metric"] == DILUTED]).tail(6)
    cso = period_end_counts(rows[rows["metric"] == CSO]).tail(6)

    yq, y_now = pd.Series(dtype=float), np.nan
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.replace(".", "-"))
        bs = t.quarterly_balance_sheet
        if bs is not None and "Ordinary Shares Number" in bs.index:
            yq = bs.loc["Ordinary Shares Number"].dropna()
            yq.index = pd.to_datetime(yq.index)
        info = t.info or {}
        y_now = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding") or np.nan
    except Exception as e:
        print(f"  (Yahoo δεν απάντησε: {e})")

    ends = sorted(set(dil["period_end"]) | set(cso["period_end"]) | set(yq.index))[-6:]
    print(f"  {'τέλος τριμήνου':<16}{'diluted μ.ό.':>14}{'τέλους (SEC)':>14}{'Yahoo τέλους':>14}")
    for e in ends:
        a = dil.loc[dil["period_end"] == e, "value"]
        b = cso.loc[cso["period_end"] == e, "value"]
        c = yq[(abs((yq.index - e).days) <= 10)] if len(yq) else yq
        print(f"  {str(e)[:10]:<16}{mn(a.iloc[0] if len(a) else np.nan):>14}"
              f"{mn(b.iloc[0] if len(b) else np.nan):>14}{mn(c.iloc[0] if len(c) else np.nan):>14}")
    print(f"  Yahoo σήμερα (όλες οι κατηγορίες): {mn(y_now)}")
    if cso.empty:
        print("  ⚠️ Δεν υπάρχει CommonStockSharesOutstanding στη βάση για αυτή την εταιρεία.")
    return {
        "cso_last": cso["value"].iloc[-1] if len(cso) else np.nan,
        "cso_end": cso["period_end"].iloc[-1] if len(cso) else pd.NaT,
        "yahoo_q_last": yq.sort_index().iloc[-1] if len(yq) else np.nan,
        "y_now": y_now,
    }


def part3(ticker: str, r: dict, s: dict) -> None:
    print(f"\n══ 3. {ticker}: τι θα γινόταν με μετοχές τέλους περιόδου ══")
    dev_now = dev(r["ev"], r["y_ev_xl"])
    print(f"  Σήμερα (μέσος όρος diluted):            EV {bn(r['ev'])}  → {pct(dev_now)}")
    best = np.nan
    for label, sh in (("SEC, τέλος τελευταίου τριμήνου", s["cso_last"]),
                      ("Yahoo, τέλος τελευταίου τριμήνου", s["yahoo_q_last"])):
        ev = what_if_ev(r, sh)
        d = dev(ev, r["y_ev_xl"])
        print(f"  Με μετοχές {label:<32} ({mn(sh)}): EV {bn(ev)}  → {pct(d)}")
        if pd.isna(best) and pd.notna(d):
            best = d
    parts = decompose(r)
    biggest = max(parts, key=lambda k: abs(parts[k][0]))
    print(f"\n  ➜ {verdict(dev_now, best, biggest)}")


# ═════════════════════════════════════════════════════════════════════════
# 3. SELFTEST
# ═════════════════════════════════════════════════════════════════════════

def selftest() -> None:
    # FLR-like: 162M average vs 155M today, price 50, debt 1.1 bn, cash 2.0 bn
    r = dict(close=50.0, shares=162e6, mcap=8.1e9, debt=1.1e9, cash=2.0e9, ev=7.2e9,
             y_shares=155e6, y_mcap=7.75e9, y_debt_xl=1.1e9, y_cash=2.0e9, y_ev_xl=6.85e9)
    p = decompose(r)
    assert abs(p["market cap"][0] - 0.35e9) < 1, p
    assert p["χρέος"][0] == 0 and p["ταμείο"][0] == 0
    assert abs(sum(v for v, _ in p.values()) - (r["ev"] - r["y_ev_xl"])) < 1
    ev = what_if_ev(r, 155e6)
    assert abs(ev - 6.85e9) < 1
    v = verdict(dev(r["ev"], r["y_ev_xl"]), dev(ev, r["y_ev_xl"]), "market cap")
    assert v.startswith("ΕΞΗΓΗΣΗ"), v
    # cash is the cause, shares don't help
    r2 = dict(r, shares=155e6, mcap=7.75e9, cash=1.0e9, ev=7.85e9)
    p2 = decompose(r2)
    big = max(p2, key=lambda k: abs(p2[k][0]))
    assert big == "ταμείο", big
    v2 = verdict(dev(r2["ev"], r2["y_ev_xl"]), dev(what_if_ev(r2, 155e6), r2["y_ev_xl"]), big)
    assert v2.startswith("ΟΧΙ"), v2
    # partial
    v3 = verdict(0.15, 0.08, "ταμείο")
    assert v3.startswith("ΜΕΡΙΚΗ"), v3
    assert verdict(0.03, np.nan, "x").startswith("εντός")
    assert verdict(np.nan, np.nan, "x").startswith("δεν")
    # quarterly filter: keeps 3-month, drops YTD, keeps the latest filing
    t = pd.Timestamp
    rows = pd.DataFrame({
        "period_start": [t("2026-07-01"), t("2026-01-01"), t("2026-07-01")],
        "period_end": [t("2026-09-30")] * 3,
        "filed_date": [t("2026-11-01"), t("2026-11-01"), t("2026-11-10")],
        "value": [160e6, 165e6, 161e6]})
    q = quarterly_diluted(rows)
    assert len(q) == 1 and q["value"].iloc[0] == 161e6, q
    print("selftest ok")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--selftest" in args:
        selftest()
        return
    tickers = [a.upper() for a in args] or ["FLR"]
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from db import get_engine
    engine = get_engine()
    for t in tickers:
        r = part1(t)
        s = part2(t, engine)
        if r is not None:
            part3(t, r, s)
        print()


if __name__ == "__main__":
    main()
