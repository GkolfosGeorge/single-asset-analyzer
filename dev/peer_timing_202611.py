"""
dev/peer_timing_202611.py — READ-ONLY: Phase 2, step 6. How long would it take to
compute TODAY's valuation of EVERY company, so each card can rank against its peers?
Nothing is written.

Two ways, timed on a small random sample and scaled up:
  A. the card as it is: quick_snapshot() once per company
     (each call: several database queries + 2 price downloads of 10 years)
  B. one batch price download for the whole universe (what a peer-table job would do)
Example: if A takes 6 s per company → 700 × 6 s = 70 min → too slow for every mail,
fine for a weekly job; B tells how much a batch download saves.

    python dev/peer_timing_202611.py           (~3–5 min)
"""

import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

SAMPLE = 15
SEED = 20261113


def main() -> None:
    from db import get_engine
    from valuation.pe import load_universe
    from valuation.quick import quick_snapshot
    from valuation.ttm import load_prices

    eng = get_engine()
    tickers = load_universe(eng)
    sample = random.Random(SEED).sample(tickers, SAMPLE)
    print(f"{len(tickers)} εταιρείες · δείγμα {SAMPLE}: {', '.join(sample)}\n")

    print("── A. quick_snapshot ανά εταιρεία ──")
    times, ok = [], 0
    for t in sample:
        t0 = time.time()
        try:
            s = quick_snapshot(t, eng, with_earnings_date=False)
            ok += s.pe is not None or s.ev_ebit is not None
        except Exception as e:
            print(f"  {t}: σφάλμα {str(e)[:60]}")
        times.append(time.time() - t0)
        print(f"  {t:6s} {times[-1]:5.1f} s")
    per = sum(times) / len(times)
    print(f"  μέσος χρόνος {per:.1f} s · με τιμή {ok}/{SAMPLE}"
          f" → για {len(tickers)} εταιρείες ≈ {per * len(tickers) / 60:.0f} λεπτά")

    print("\n── B. Τιμές όλου του universe σε ένα κατέβασμα (2 χρόνια) ──")
    end = pd.Timestamp.today().normalize()
    t0 = time.time()
    close, _ = load_prices(tickers, (end - pd.DateOffset(years=2)).date())
    dt = time.time() - t0
    have = int(close.notna().any().sum()) if not close.empty else 0
    print(f"  {dt:.0f} s · τιμές για {have}/{len(tickers)} εταιρείες")


if __name__ == "__main__":
    main()
