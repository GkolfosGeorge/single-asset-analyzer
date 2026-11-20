"""
dev/shares_fallback_impact_202611.py — READ-ONLY: which companies does the v0.1.7
basic-shares fallback change? (20/11/2026). Database only, ~1 min.

valuation.yields.with_basic_fallback adds BASIC weighted shares for periods that have
NO diluted row. Companies that file both are untouched. This lists, for the universe:
  - "μόνο basic"      no diluted row at all → today no market cap, after: one
  - "νεότερο basic"   the newest basic period is newer than the newest diluted one
                      → today's share count gets fresher (XOM: diluted stops 12/2025)
  - "κενά στο ιστορικό" basic fills older periods only (history of 💰 / FCF yield)
and, where both exist for the same period, how far apart they are (a sanity check:
basic ≤ diluted, normally within 1–2%).

    python dev/shares_fallback_impact_202611.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from valuation.pe import load_universe
from valuation.ttm import load_metrics
from valuation.yields import SHARES, SHARES_BASIC, with_basic_fallback


def main() -> None:
    from db import get_engine
    eng = get_engine()
    uni = load_universe(eng)
    rows = load_metrics([SHARES, SHARES_BASIC], uni, eng)
    out = []
    for t, g in rows.groupby("ticker"):
        dil, bas = g[g["metric"] == SHARES], g[g["metric"] == SHARES_BASIC]
        filled = len(with_basic_fallback(dil, bas)) - len(dil)
        if not filled:
            continue
        last_d = dil["period_end"].max() if len(dil) else None
        last_b = bas["period_end"].max()
        if last_d is None:
            kind = "μόνο basic"
        elif last_b > last_d + pd.Timedelta(days=45):
            kind = "νεότερο basic"
        else:
            kind = "κενά στο ιστορικό"
        both = dil.merge(bas, on=["period_start", "period_end"], suffixes=("_d", "_b"))
        ratio = (both["value_b"] / both["value_d"]).median() if len(both) else None
        out.append({"ticker": t, "kind": kind, "filled_periods": filled,
                    "last_diluted": last_d, "last_basic": last_b, "ratio": ratio})
    df = pd.DataFrame(out)
    print(f"Universe {len(uni)} · εταιρείες που αλλάζουν: {len(df)}\n")
    for kind in ["μόνο basic", "νεότερο basic", "κενά στο ιστορικό"]:
        sub = df[df["kind"] == kind]
        print(f"── {kind}: {len(sub)} ──")
        for r in sub.sort_values("ticker").itertuples():
            ld = "—" if r.last_diluted is None or pd.isna(r.last_diluted) else f"{r.last_diluted:%Y-%m}"
            ra = "" if r.ratio is None or pd.isna(r.ratio) else f" · basic/diluted {r.ratio:.3f}"
            print(f"   {r.ticker:6s} +{r.filled_periods:3d} περίοδοι · diluted ως {ld} · basic ως {r.last_basic:%Y-%m}{ra}")
        print()
    odd = df[df["ratio"].notna() & ((df["ratio"] > 1.001) | (df["ratio"] < 0.95))]
    if len(odd):
        print(f"⚠️ basic/diluted εκτός 0,95–1,00 (να ελεγχθούν): {', '.join(odd['ticker'])}")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/shares_fallback_impact_202611.csv", index=False)
    print("→ data/shares_fallback_impact_202611.csv")


if __name__ == "__main__":
    main()
