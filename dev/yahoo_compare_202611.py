"""
dev/yahoo_compare_202611.py — READ-ONLY: the Yahoo check of 03/11 vs today's rerun,
stock by stock (25/11/2026). Files only, no database, no network.

    python dev/yahoo_compare_202611.py
      round 1: data/yahoo_check_202611_r1.csv   (copy of the 03/11 results)
      round 2: data/yahoo_check_202611.csv      (today's run of dev/yahoo_check_202611.py)

For P/E, EV (with and without leases) and FCF, each stock gets one word:
    "διορθώθηκε"    was outside 5%, now within
    "χάλασε"        was within 5%, now outside            ← each needs an explanation
    "ok"            within 5% both times
    "ακόμα εκτός"   outside both times (the number shows by how much)
    "νέο"           n/a in round 1, a value now (e.g. a missing EV now computed)
    "χάθηκε"        a value in round 1, n/a now
Then the Phase 1 criterion (≥ 90% of the RANDOM sample within 5%) for both rounds.
"""

import os
import sys

import pandas as pd

R1, R2 = "data/yahoo_check_202611_r1.csv", "data/yahoo_check_202611.csv"
METRICS = [("dev_pe", "P/E"), ("dev_ev_xl", "EV χωρίς leases"), ("dev_ev", "EV με leases"),
           ("dev_fcf", "FCF")]
OK = 0.05


def word(a, b) -> str:
    ia, ib = pd.notna(a), pd.notna(b)
    if not ia and not ib:
        return ""
    if not ia:
        return "νέο"
    if not ib:
        return "χάθηκε"
    wa, wb = abs(a) <= OK, abs(b) <= OK
    if wa and wb:
        return "ok"
    if not wa and wb:
        return "διορθώθηκε"
    if wa and not wb:
        return "χάλασε"
    return "ακόμα εκτός"


def criterion(df: pd.DataFrame, col: str) -> str:
    rnd = df[df["group"] == "random"]
    v = rnd[col].dropna() if col in rnd else pd.Series(dtype=float)
    if v.empty:
        return "—"
    k = int((v.abs() <= OK).sum())
    return f"{k}/{len(v)} = {k / len(v):.0%}"


def compare(r1: pd.DataFrame, r2: pd.DataFrame) -> pd.DataFrame:
    m = r1.merge(r2, on="ticker", how="outer", suffixes=("_1", "_2"))
    m["group"] = m["group_2"].fillna(m["group_1"])
    for col, _ in METRICS:
        a = m.get(f"{col}_1", pd.Series(float("nan"), index=m.index))
        b = m.get(f"{col}_2", pd.Series(float("nan"), index=m.index))
        m[f"{col}_word"] = [word(x, y) for x, y in zip(a, b)]
    return m


def selftest() -> None:
    assert word(0.20, 0.01) == "διορθώθηκε" and word(0.01, 0.20) == "χάλασε"
    assert word(0.01, -0.02) == "ok" and word(float("nan"), 0.01) == "νέο"
    assert word(0.30, 0.25) == "ακόμα εκτός" and word(0.01, float("nan")) == "χάθηκε"
    r1 = pd.DataFrame({"ticker": ["A", "B"], "group": "random", "dev_pe": [0.2, 0.01]})
    r2 = pd.DataFrame({"ticker": ["A", "B"], "group": "random", "dev_pe": [0.01, 0.02]})
    m = compare(r1, r2)
    assert m["dev_pe_word"].tolist() == ["διορθώθηκε", "ok"]
    assert criterion(r1, "dev_pe") == "1/2 = 50%" and criterion(r2, "dev_pe") == "2/2 = 100%"
    print("selftest: όλα τα τεστ πέρασαν ✓")


def main() -> None:
    for f in (R1, R2):
        if not os.path.exists(f):
            sys.exit(f"Λείπει το {f}")
    r1, r2 = pd.read_csv(R1), pd.read_csv(R2)
    m = compare(r1, r2)

    print("══ Κριτήριο Φάσης 1: ≥ 90% του ΤΥΧΑΙΟΥ δείγματος εντός 5% ══")
    print(f"  {'':18s} {'03/11':>14s} {'σήμερα':>14s}")
    for col, name in METRICS:
        print(f"  {name:18s} {criterion(r1, col):>14s} {criterion(r2, col):>14s}")

    for col, name in METRICS:
        print(f"\n══ {name} ══")
        for w in ["διορθώθηκε", "νέο", "χάλασε", "χάθηκε", "ακόμα εκτός"]:
            sub = m[m[f"{col}_word"] == w]
            if sub.empty:
                continue
            items = []
            for r in sub.itertuples():
                a, b = getattr(r, f"{col}_1", float("nan")), getattr(r, f"{col}_2", float("nan"))
                fa = "n/a" if pd.isna(a) else f"{a:+.0%}"
                fb = "n/a" if pd.isna(b) else f"{b:+.0%}"
                items.append(f"{r.ticker} ({fa}→{fb})")
            print(f"  {w:12s} {len(sub):3d}: " + ", ".join(items))
        n_ok = int((m[f"{col}_word"] == "ok").sum())
        print(f"  {'ok':12s} {n_ok:3d}")
    m.to_csv("data/yahoo_compare_202611.csv", index=False)
    print("\n→ data/yahoo_compare_202611.csv")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
