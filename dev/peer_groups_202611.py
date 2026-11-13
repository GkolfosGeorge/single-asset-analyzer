"""
dev/peer_groups_202611.py — READ-ONLY: Phase 2, step 5. How big are the peer
groups if we compare each company with its INDUSTRY (yfinance classification)?
Nothing is written; only reads data/classification.json (no SEC, no prices).

Why it matters: a comparison "cheaper than 70% of its industry" needs enough
companies in the industry. Example:
    "Semiconductors" with 25 companies → a rank means something
    "Tobacco" with 3 companies         → 1 of 3 is noise, compare with the SECTOR
The rule we will choose: industry if it has at least N companies, else sector.
This script shows, for N = 5, 8, 10, 15, how many companies land where.

    python dev/peer_groups_202611.py --selftest   (no database)
    python dev/peer_groups_202611.py              (~5 sec; reads the database for the universe)
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

CLASSIFICATION = "data/classification.json"
THRESHOLDS = [5, 8, 10, 15]
FINANCIAL = "Financial Services"
SHOW = ["Semiconductors", "Software - Infrastructure", "Banks - Regional", "Oil & Gas Integrated",
        "Drug Manufacturers - General", "REIT - Specialty", "Restaurants", "Tobacco"]


def table(cls: dict[str, dict]) -> pd.DataFrame:
    rows = [{"ticker": t, "sector": v.get("sector"), "industry": v.get("industry")} for t, v in cls.items()]
    d = pd.DataFrame(rows, columns=["ticker", "sector", "industry"])
    return d[d["sector"].notna() & (d["sector"] != "Unknown") & d["industry"].notna()]


def assign(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Peer group = industry if it has ≥ n companies, else the sector (labelled as such)."""
    size = d.groupby("industry")["ticker"].transform("size")
    out = d.copy()
    out["group"] = d["industry"].where(size >= n, "τομέας: " + d["sector"])
    out["group_size"] = out.groupby("group")["ticker"].transform("size")
    out["level"] = (size >= n).map({True: "κλάδος", False: "τομέας"})
    return out


def report(d: pd.DataFrame, universe_n: int) -> None:
    nf = d[d["sector"] != FINANCIAL]
    print(f"── {universe_n} εταιρείες στο universe · με κλάδο στο cache {len(d)}"
          f" · χωρίς χρηματοοικονομικές {len(nf)} ──\n")
    print(f"  τομείς: {nf['sector'].nunique()} · κλάδοι: {nf['industry'].nunique()}")
    sizes = nf.groupby("industry").size()
    print(f"  μέγεθος κλάδου: διάμεσος {sizes.median():.0f} · 1 εταιρεία: {int((sizes == 1).sum())} κλάδοι"
          f" · ≥ 10: {int((sizes >= 10).sum())} κλάδοι\n")

    print("── 1. Κανόνας «κλάδος αν έχει ≥ N εταιρείες, αλλιώς τομέας» (χωρίς χρηματοοικονομικές) ──")
    rows = []
    for n in THRESHOLDS:
        a = assign(nf, n)
        rows.append({"N": n, "με κλάδο": int((a["level"] == "κλάδος").sum()),
                     "με τομέα": int((a["level"] == "τομέας").sum()),
                     "κλάδοι που μένουν": a.loc[a["level"] == "κλάδος", "industry"].nunique(),
                     "μικρότερη ομάδα": int(a["group_size"].min())})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n── 2. Οι 25 μεγαλύτεροι κλάδοι ──")
    print(sizes.sort_values(ascending=False).head(25).to_string())

    print("\n── 3. Τομείς (αν όλα πήγαιναν στον τομέα) ──")
    print(nf.groupby("sector").size().sort_values(ascending=False).to_string())

    print("\n── 4. Δείγματα κλάδων, για έλεγχο με το μάτι ──")
    for ind in SHOW:
        m = sorted(d.loc[d["industry"] == ind, "ticker"])
        print(f"  {ind} ({len(m)}): {', '.join(m[:30])}")

    print("\n── 5. Εταιρείες του universe χωρίς κλάδο στο cache ──")


def main() -> None:
    from db import get_engine
    from valuation.pe import get_classification, load_universe

    tickers = load_universe(get_engine())
    cls = {}
    if os.path.exists(CLASSIFICATION):
        cls = json.load(open(CLASSIFICATION, encoding="utf-8")).get("data", {})
    missing = [t for t in tickers if t not in cls]
    if len(missing) > 0.2 * len(tickers):
        print(f"Το cache δεν έχει {len(missing)} εταιρείες: ρωτάω το Yahoo (~{len(missing) // 3 // 60 + 1} λεπτά)")
        cls.update(get_classification(tickers, CLASSIFICATION))
        missing = [t for t in tickers if t not in cls or cls[t].get("sector") in (None, "Unknown")]
    d = table({t: cls[t] for t in tickers if t in cls})
    report(d, len(tickers))
    print(f"  {len(missing)}: {', '.join(sorted(missing)[:40])}")


def selftest() -> None:
    cls = {**{f"S{i}": {"sector": "Technology", "industry": "Semiconductors"} for i in range(9)},
           "T1": {"sector": "Consumer Defensive", "industry": "Tobacco"},
           "T2": {"sector": "Consumer Defensive", "industry": "Tobacco"},
           "F1": {"sector": "Consumer Defensive", "industry": "Packaged Foods"},
           "U": {"sector": "Unknown", "industry": "Unknown"}}
    d = table(cls)
    assert "U" not in set(d["ticker"])
    a = assign(d, 8).set_index("ticker")
    assert a.loc["S0", "group"] == "Semiconductors" and a.loc["S0", "group_size"] == 9
    assert a.loc["T1", "group"] == "τομέας: Consumer Defensive" and a.loc["T1", "group_size"] == 3
    a = assign(d, 10).set_index("ticker")
    assert a.loc["S0", "level"] == "τομέας"
    print("✅ selftest: όλα τα συνθετικά σενάρια πέρασαν")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
