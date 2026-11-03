"""
dev/shares_scale_check_202611.py — READ-ONLY: how many companies have ASH's problem?

ASH's diluted shares are in the DB as 46 instead of 46,000,000: the company
tagged them in millions. The ×50 guard in ttm.build_latest compares a row
with its NEIGHBOURS, so it can't see the error when the neighbours are wrong too.

This looks at the whole universe, without Yahoo, using two independent facts:
  1. no listed company in the universe has under 1 million diluted shares
  2. diluted shares should be close to net income / diluted EPS
     (ASH-like: 46 vs 150,000,000 / 3.26 = 46,000,000)

    python dev/shares_scale_check_202611.py        (~1 min, nothing is written)
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from db import get_engine
from valuation.pe import EPS_METRIC, load_universe
from valuation.ttm import load_metrics
from valuation.yields import SHARES

MIN_SHARES = 1e6
NI = "NetIncomeLoss"
RECENT = pd.Timestamp.today() - pd.Timedelta(days=500)

engine = get_engine()
universe = load_universe(engine)
rows = load_metrics([SHARES, EPS_METRIC, NI], universe, engine)
print(f"Universe {len(universe)} · γραμμές {len(rows):,}")

s = rows[rows["metric"] == SHARES].dropna(subset=["value"])
low = s[s["value"] < MIN_SHARES]
print(f"\n══ Γραμμές με < 1 εκατ. μετοχές: {len(low)} σε {low['ticker'].nunique()} εταιρείες ══")
if len(low):
    g = low.groupby("ticker").agg(rows=("value", "size"), first=("period_end", "min"),
                                  last=("period_end", "max"), latest_value=("value", "last"))
    g["all_rows"] = s.groupby("ticker").size().reindex(g.index)
    g["recent"] = g["last"] >= RECENT
    print(g.sort_values("rows", ascending=False).to_string())

# same ticker, same period (10-K/10-Q, same dates): shares vs NI / EPS
key = ["ticker", "period_start", "period_end"]
w = (rows.dropna(subset=["value", "period_start"]).sort_values("filed_date")
     .drop_duplicates(key + ["metric"]).pivot(index=key, columns="metric", values="value"))
w = w.dropna(subset=[SHARES, EPS_METRIC, NI])
w = w[(w[EPS_METRIC].abs() >= 0.10) & (w[NI].abs() >= 1e6)].reset_index()
w = w[w["period_end"] >= RECENT]
w["implied"] = (w[NI] / w[EPS_METRIC]).abs()
w["ratio"] = w[SHARES] / w["implied"]
w = w.sort_values("period_end").groupby("ticker").tail(1)
bad = w[(w["ratio"] > 3) | (w["ratio"] < 1 / 3)]
print(f"\n══ Τελευταία περίοδος: μετοχές ≠ NI / EPS κατά > ×3: {len(bad)} από {len(w)} εταιρείες ══")
print(bad[["ticker", "period_end", SHARES, "implied", "ratio"]]
      .rename(columns={SHARES: "shares"}).sort_values("ratio").to_string(index=False, float_format="{:,.4g}".format))
