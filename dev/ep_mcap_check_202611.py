"""
dev/ep_mcap_check_202611.py — READ-ONLY check of saa v0.1.8 (20/11/2026):
P/E from net income TTM / Yahoo market cap, next to Yahoo's own trailing P/E.

    python dev/ep_mcap_check_202611.py            # the 8 companies with no EPS + a few controls
    python dev/ep_mcap_check_202611.py --table    # every company of peer_data/peer_yields.csv
                                                  # with no earnings yield today (~2–3 min)

Columns: our P/E (= market cap / net income TTM), Yahoo's trailingPE, the gap.
Within ~10% is fine (Yahoo's TTM may include one quarter more or less, or use
EPS from continuing operations). Large gaps are listed at the end.
Controls (AAPL, MSFT, JPM) have EPS, so the card never uses this path for them;
they are here only to show the formula agrees with Yahoo on easy cases.
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd

from valuation import earnings_mcap as EM

TARGETS = ["V", "BRK-B", "HSY", "STZ", "ERIE", "KKR", "ARES", "BKR"]
CONTROLS = ["AAPL", "MSFT", "JPM"]


def main(tickers: list[str]) -> None:
    import yfinance as yf
    from db import get_engine
    eng = get_engine()
    rows = []
    for t in tickers:
        r = EM.earnings_yield_from_mcap(t, eng)
        try:
            ytpe = yf.Ticker(t).info.get("trailingPE")
        except Exception:
            ytpe = None
        ours = (1 / r["earnings_yield"]) if r and r["earnings_yield"] > 0 else None
        gap = (ours / ytpe - 1) if ours and ytpe else None
        rows.append({"ticker": t, "basis": r["basis"] if r else None,
                     "ni_ttm_bn": r["ni_ttm"] / 1e9 if r else None,
                     "mcap_bn": r["market_cap"] / 1e9 if r else None,
                     "period_end": r["period_end"].date() if r else None,
                     "pe_ours": ours, "pe_yahoo": ytpe, "gap": gap})
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    f = lambda v, fmt, suf="": "—" if v is None or pd.isna(v) else format(v, fmt) + suf     # noqa: E731
    print(f"{'ticker':7s} {'βάση':14s} {'κέρδη TTM':>10s} {'κεφ/ση':>8s} {'περίοδος':>11s} {'P/E μας':>8s} {'Yahoo':>7s} {'διαφορά':>8s}")
    for r in df.itertuples():
        print(f"{r.ticker:7s} {str(r.basis or '—'):14s} {f(r.ni_ttm_bn, ',.1f', 'δ'):>10s} {f(r.mcap_bn, ',.0f', 'δ'):>8s} {str(r.period_end or '—'):>11s}"
              f" {f(r.pe_ours, '.1f'):>8s} {f(r.pe_yahoo, '.1f'):>7s} {f(r.gap, '+.0%'):>8s}")
    big = df[df["gap"].notna() & (df["gap"].abs() > 0.10)]
    none = df[df["pe_ours"].isna()]
    print(f"\nΕντός 10% από Yahoo: {int((df['gap'].abs() <= 0.10).sum())}/{int(df['gap'].notna().sum())}")
    if len(big):
        print(f"Διαφορά > 10%: {', '.join(big['ticker'])}")
    if len(none):
        print(f"Χωρίς P/E (ζημιές, παλιά κέρδη ή χωρίς κεφαλαιοποίηση): {', '.join(none['ticker'])}")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/ep_mcap_check_202611.csv", index=False)
    print("→ data/ep_mcap_check_202611.csv")


if __name__ == "__main__":
    if "--table" in sys.argv:
        t = pd.read_csv("peer_data/peer_yields.csv")
        main(t.loc[t["earnings_yield"].isna(), "ticker"].tolist())
    else:
        main([a.upper() for a in sys.argv[1:]] or TARGETS + CONTROLS)
