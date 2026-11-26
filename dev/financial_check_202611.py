"""
dev/financial_check_202611.py — READ-ONLY check of v0.1.9 (26/11/2026).

"Financial" now comes from Yahoo's industry (valuation/peers.is_financial), not
from missing tags. Two parts:

  1. the weekly table (peer_data/peer_yields.csv, no network): which companies
     change flag, old (tags) → new (industry)
  2. real cards (database + Yahoo, ~7 s each) for companies on both sides of
     the line: a bank must show P/E only and no cash-flow 🚩; SPGI / V / ADP
     must show EV/EBIT and FCF when they have them.

    python dev/financial_check_202611.py            (~2 min)
    python dev/financial_check_202611.py C JPM      (only these cards)
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

TABLE = "peer_data/peer_yields.csv"
CARDS = ["C", "JPM", "GS", "SCHW", "PGR", "MET", "BRK-B", "COF",       # → financial
         "V", "MA", "SPGI", "AON", "BLK",                               # Financial Services, not financial
         "STT", "PFG",                                                  # hand-made list
         "ADP", "PSX", "CAR"]                                           # old test said financial


def part1() -> None:
    from valuation.peers import is_financial
    d = pd.read_csv(TABLE)
    d["new"] = [is_financial(t, i, f) for t, i, f in zip(d["ticker"], d["industry"], d["financial"])]
    on, off = d[~d["financial"] & d["new"]], d[d["financial"] & ~d["new"]]
    print(f"══ 1. Πίνακας ομοίων ({d['as_of'].max()}): χρηματοοικονομικές {int(d['financial'].sum())} → "
          f"{int(d['new'].sum())} ══")
    print(f"  Γίνονται χρηματοοικονομικές ({len(on)}): " + " ".join(on["ticker"]))
    print(f"  Παύουν ({len(off)}): " + " ".join(off["ticker"]))
    have = off[off[["ebit_yield", "fcf_yield"]].notna().any(axis=1)]
    print(f"     από αυτές με EV/EBIT ή FCF που θα εμφανιστεί πλέον: {len(have)} "
          + (" ".join(have["ticker"]) if len(have) else ""))


def part2(tickers: list[str]) -> None:
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from db import get_engine
    from valuation.quick import format_card, quick_snapshot
    eng = get_engine()
    print("\n══ 2. Κάρτες ══")
    for t in tickers:
        s = quick_snapshot(t, eng, with_earnings_date=False)
        print(f"\n── {t} · χρηματοοικονομική = {s.financial}"
              + (f" · σφάλματα: {'; '.join(s.errors)}" if s.errors else ""))
        print("\n".join(format_card(s)))


def main() -> None:
    args = [a.upper() for a in sys.argv[1:]]
    if not args:
        part1()
    part2(args or CARDS)


if __name__ == "__main__":
    main()
