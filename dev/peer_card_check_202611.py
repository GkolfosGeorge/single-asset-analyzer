"""
dev/peer_card_check_202611.py — READ-ONLY: the 🏷️ peer line on real cards, for a
check by eye before release v0.1.6. Nothing is written.

For each ticker: the card as the mail will show it, then the numbers behind the
peer line: today's own yield, the peers' median, and "cheaper than k of n".
Example: "EV/EBIT  own 6.5% · median peers 4.0% · cheaper than 19/23 (83%) → φθηνή"

Needs the new valuation/quick.py and the table of the workflow:
    git pull                                  (brings peer_data/peer_yields.csv)
    python dev/peer_card_check_202611.py      (~2–3 min)
    python dev/peer_card_check_202611.py NVDA KO     (only these)
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

TICKERS = ["NVDA", "AMD", "AAPL", "MSFT", "KO", "XOM", "JPM", "HD", "PG", "CAT", "F", "PFE", "UNH", "CPWR"]
COLS = {"P/E": "earnings_yield", "EV/EBIT": "ebit_yield", "FCF": "fcf_yield"}


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def main() -> None:
    from db import get_engine
    from valuation import peers as PEERS
    from valuation import quick as Q

    eng = get_engine()
    table = Q.load_peer_table()
    print(f"Πίνακας: {len(table)} εταιρείες · ημ/νία {table['as_of'].max()}\n")
    cls = {r.ticker: {"sector": r.sector, "industry": r.industry} for r in table.itertuples()
           if isinstance(r.sector, str) and isinstance(r.industry, str)}
    groups = PEERS.peer_groups(cls)
    t = table.drop_duplicates("ticker").set_index("ticker")

    for tk in sys.argv[1:] or TICKERS:
        print(f"══ {tk} ══")
        try:
            s = Q.quick_snapshot(tk, eng, with_earnings_date=False)
        except Exception as e:
            print(f"  σφάλμα: {e}\n")
            continue
        print("\n".join(Q.format_card(s)))
        if not s.peer:
            print("  (καμία γραμμή κλάδου)" + (f" · σφάλματα: {s.errors}" if s.errors else ""))
            print()
            continue
        peers = [m for m in PEERS.members(groups, tk) if m != tk]
        print(f"  ── πίσω από τη γραμμή: {s.peer['level']} «{s.peer['group']}», όμοιες: {', '.join(sorted(peers))}")
        for name, col in COLS.items():
            if s.financial and col != "earnings_yield":
                continue
            own, vals = s.yields.get(col), t.loc[peers, col].dropna()
            z = s.peer["zones"].get(name)
            tail = (f"φθηνότερη από {z[1] * z[2]:.1f}/{z[2]} ({z[1] * 100:.0f}%) → {z[0]}" if z
                    else f"χωρίς ζώνη ({len(vals)} όμοιες με τιμή)")
            print(f"     {name:8s} δική της {pct(own):>7s} · διάμεσος ομοίων {pct(vals.median()) if len(vals) else '—':>7s}"
                  f" · {tail}")
        print()


if __name__ == "__main__":
    main()
