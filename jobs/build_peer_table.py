"""
jobs/build_peer_table.py — weekly: TODAY's yields of every company in the
universe → peer_data/peer_yields.csv (+ peer_data/classification.json).
Read by the mail's card for the comparison with peers. Reads the database,
writes ONLY these two files (no database writes).

Runs on GitHub Actions (.github/workflows/peer_table.yml), ~60–90 min.
Locally, for a quick look:
    python jobs/build_peer_table.py --limit 30      (writes peer_data/peer_yields_test.csv)
    python jobs/build_peer_table.py                 (the full table)

Safety: the full table replaces the old one ONLY if at least MIN_COVERAGE
companies got a yield; a broken run (Yahoo down, database down) leaves last
week's table in place and fails loudly.
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

OUT_DIR = "peer_data"
OUT = os.path.join(OUT_DIR, "peer_yields.csv")
TEST_OUT = os.path.join(OUT_DIR, "peer_yields_test.csv")
CLASSIFICATION = os.path.join(OUT_DIR, "classification.json")
MIN_COVERAGE = 500


def main() -> None:
    from db import get_engine
    from valuation.pe import get_classification, load_universe
    from valuation.peer_table import build, coverage

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    os.makedirs(OUT_DIR, exist_ok=True)

    eng = get_engine()
    tickers = load_universe(eng)
    if limit:
        tickers = tickers[:limit]
    t0 = time.time()
    cls = get_classification(tickers, CLASSIFICATION)
    print(f"{len(tickers)} εταιρείες · κλάδοι σε {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    table = build(eng, tickers, cls)
    n = coverage(table)
    print(f"\nΑποδόσεις για {n}/{len(tickers)} εταιρείες σε {(time.time() - t0) / 60:.0f} λεπτά")
    print(table[["earnings_yield", "ebit_yield", "fcf_yield"]].notna().sum().to_string())

    if limit:
        table.to_csv(TEST_OUT, index=False)
        print(f"→ {TEST_OUT} (δοκιμή, ο κανονικός πίνακας δεν άλλαξε)")
        return
    if n < MIN_COVERAGE:
        sys.exit(f"❌ Μόνο {n} εταιρείες με απόδοση (< {MIN_COVERAGE}): ο παλιός πίνακας μένει.")
    table.to_csv(OUT, index=False)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
