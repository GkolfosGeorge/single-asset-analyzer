"""
dev/shares_repair_regression_202611.py — READ-ONLY: what does repair_share_scale
change across the WHOLE universe? Nothing is written anywhere.

    python dev/shares_repair_regression_202611.py        (~1 min)

Every changed row must make sense: factors are only 1000 / 1,000,000 (or their
inverses), and after the repair no company may stay off by a power of 1000.

Third, independent yardstick: CommonStockSharesOutstanding (nearest date, ±100
days). If the ORIGINAL row agreed with it (within ×3), the original was probably
right and the repair wrong: such rows are listed as conflicts.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

from db import get_engine
from valuation.pe import load_universe
from valuation.ttm import load_metrics, repair_share_scale
from valuation.yields import EPS, NET_INCOME, SHARES

engine = get_engine()
universe = load_universe(engine)
CSO = "CommonStockSharesOutstanding"
rows = load_metrics([SHARES, NET_INCOME, EPS, CSO], universe, engine)
print(f"Universe {len(universe)} · γραμμές {len(rows):,}")

changed, after = [], []
for t, g in rows.groupby("ticker"):
    parts = [g[g["metric"] == m] for m in (SHARES, NET_INCOME, EPS)]
    r = repair_share_scale(*parts)
    after.append(r.assign(ticker=t))
    changed += [dict(x, ticker=t) for x in r.attrs["repaired"]]

ch = pd.DataFrame(changed)
print(f"\n══ Διορθώθηκαν {len(ch)} γραμμές σε {ch['ticker'].nunique() if len(ch) else 0} εταιρείες ══")
if len(ch):
    print("Κατανομή συντελεστών:")
    print(ch["factor"].value_counts().to_string())
    g = ch.groupby("ticker").agg(rows=("value", "size"), first=("period_end", "min"),
                                 last=("period_end", "max"), factor=("factor", "last"))
    print(g.sort_values("last", ascending=False).head(12).to_string())

    r = rows[rows["metric"] == CSO].dropna(subset=["value", "period_end"])
    r = r.sort_values("filed_date").drop_duplicates(["ticker", "period_end"])[["ticker", "period_end", "value"]]
    # merge_asof refuses to mix date resolutions (DB [s] vs Timestamps [ns])
    ch["period_end"] = pd.to_datetime(ch["period_end"]).dt.as_unit("ns")
    r["period_end"] = pd.to_datetime(r["period_end"]).dt.as_unit("ns")
    c = pd.merge_asof(ch.sort_values("period_end"), r.rename(columns={"value": "cso"}).sort_values("period_end"),
                      on="period_end", by="ticker", direction="nearest", tolerance=pd.Timedelta(days=100))
    c["repaired"] = c["value"] * c["factor"]
    c["orig_vs_cso"] = (c["value"] / c["cso"]).abs()
    c["new_vs_cso"] = (c["repaired"] / c["cso"]).abs()
    known = c[c["cso"].notna()]
    agree = known[np.abs(np.log(known["new_vs_cso"])) <= np.log(3)]
    conflict = known[np.abs(np.log(known["orig_vs_cso"])) <= np.log(3)]
    print(f"\n══ Τρίτο μέτρο (CommonStockSharesOutstanding): υπάρχει σε {len(known)}/{len(c)} γραμμές ══")
    print(f"  η διόρθωση συμφωνεί: {len(agree)}")
    print(f"  ΣΥΓΚΡΟΥΣΗ (το αρχικό ταίριαζε, η διόρθωση όχι): {len(conflict)}")
    if len(conflict):
        print(conflict[["ticker", "period_end", "value", "factor", "repaired", "cso"]]
              .to_string(index=False, float_format="{:,.6g}".format))
    rest = known.drop(agree.index).drop(conflict.index)
    print(f"  κανένα από τα δύο δεν ταιριάζει: {len(rest)}")
    if len(rest):
        print(rest[["ticker", "period_end", "value", "factor", "repaired", "cso"]].head(25)
              .to_string(index=False, float_format="{:,.6g}".format))
    inv = ch[ch["factor"] < 1]
    print(f"\n══ Διορθώσεις προς τα κάτω ({len(inv)} γραμμές) ══")
    print(inv.groupby("ticker").agg(rows=("value", "size"), last=("period_end", "max"),
                                    value=("value", "last"), factor=("factor", "last")).to_string())

a = pd.concat(after)
low = a[(a["value"] < 1e6) & (a["period_end"] >= pd.Timestamp.today() - pd.Timedelta(days=500))]
last = a.sort_values("period_end").groupby("ticker").tail(1)
print(f"\n══ Μένουν < 1 εκατ. μετοχές στα τελευταία ~1,5 χρόνια: {low['ticker'].nunique()} εταιρείες ══")
print(low.sort_values("period_end").groupby("ticker").tail(1)[["ticker", "period_end", "value"]]
      .to_string(index=False))
huge = last[last["value"] > 3e10]
print(f"\n══ Τελευταία γραμμή > 30 δισ. μετοχές: {len(huge)} ══")
print(huge[["ticker", "period_end", "value"]].to_string(index=False))
