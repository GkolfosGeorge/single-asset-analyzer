"""
dev/yahoo_check_explain_202611.py — WHY did the stocks deviate? Reads only
data/yahoo_check_202611.csv (no database, no Yahoo, a second to run).

    python dev/yahoo_check_explain_202611.py

Prints each outlier side by side, in bn $, e.g.
    KLAC  ours cash 1.53   Yahoo 4.73   −68%
Limit: |deviation| > 5%.
"""

import numpy as np
import pandas as pd

LIMIT = 0.05
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200,
              "display.max_colwidth", 48)

df = pd.read_csv("data/yahoo_check_202611.csv", keep_default_na=True)
df["financial"] = df.get("financial", False)


def table(title, dev, cols, money=()):
    if dev not in df.columns:
        return
    t = df[df[dev].abs() > LIMIT].copy()
    print(f"\n══ {title} ({len(t)}) ══")
    if t.empty:
        print("  καμία")
        return
    t["gap"] = (t[dev] * 100).map("{:+.0f}%".format)
    for c in money:
        t[c] = (t[c] / 1e9).map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
    show = [c for c in ["ticker", "group"] + list(cols) + ["gap"] if c in t.columns]
    print(t.sort_values(dev, key=abs, ascending=False)[show].to_string(index=False))


print("══ Γιατί n/a (τυχαίο δείγμα, όσες δεν έχουν P/E ή EV) ══")
rnd = df[df["group"] == "random"]
ev_col = rnd["dev_ev"] if "dev_ev" in rnd else pd.Series(np.nan, index=rnd.index)
t = rnd[rnd["dev_pe"].isna() | ev_col.isna()].copy()
t["P/E"] = np.where(t["dev_pe"].isna(), "n/a", "ok")
t["EV"] = np.where(t["dev_ev"].isna(), "n/a", "ok")
for c in ("eps", "y_debt", "y_cash", "ev"):
    t[c] = t[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}" if abs(v) < 1e6 else f"{v / 1e9:,.2f}bn")
t["period_end"] = t["period_end"].astype(str).str[:10]
t["y_latest_q"] = t["y_latest_q"].astype(str).str[:10]
t["fcf_status"] = t["fcf_status"].astype(str).str[:28]
print(t[["ticker", "sector", "financial", "P/E", "EV", "eps", "ev", "y_debt", "y_cash",
         "period_end", "y_latest_q", "fcf_status"]].to_string(index=False))

table("Μετοχές", "dev_shares", ["shares", "y_shares", "recent_splits", "known_gap"])
table("FCF", "dev_fcf", ["cfo_ttm", "y_cfo", "capex_ttm", "y_capex", "capex_tag"],
      ["cfo_ttm", "y_cfo", "capex_ttm", "y_capex"])
table("Ταμείο", "dev_cash", ["cash", "y_cash"], ["cash", "y_cash"])
table("Χρέος χωρίς leases", "dev_debt_xl", ["debt", "y_debt_xl", "y_leases", "debt_method"],
      ["debt", "y_debt_xl", "y_leases"])
table("EBIT", "dev_ebit", ["ebit", "y_ebit", "y_ebit_src"], ["ebit", "y_ebit"])
table("EV χωρίς leases", "dev_ev_xl", ["mcap", "y_mcap", "debt", "y_debt_xl", "cash", "y_cash"],
      ["mcap", "y_mcap", "debt", "y_debt_xl", "cash", "y_cash"])
table("P/E (ίδιο τρίμηνο)", "dev_pe", ["pe", "y_pe", "eps", "y_eps", "y_eps_src", "known_gap"])
