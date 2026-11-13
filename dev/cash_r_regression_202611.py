"""
dev/cash_r_regression_202611.py — READ-ONLY regression for the "cash including
restricted cash" rule in valuation/ev.py (trading-shared-data v0.1.27 tags).
No SEC, no Yahoo calls: only the database and the saved Yahoo files. Nothing is written.

Old ev.py (v0.1.4) versus new, cash only (debt is not touched):
  1. dates the OLD rule had: did any value change? (should be ~0: the new tag is
     used only where the old rule had nothing)
  2. dates the new rule adds, and companies whose NEWEST cash becomes fresh
  3. against Yahoo, same balance-sheet date (data/ev_gap_discovery_202611.jsonl,
     data/cash_values_202611.jsonl)
  4. the companies to watch (PG, MMM, ... and WAB, CVX, ADP which must NOT change)

Setup (Git Bash, repo root), BEFORE replacing valuation/ev.py:
    git show HEAD:valuation/ev.py > valuation/_ev_v014.py
Then put the new ev.py in place and run:
    python dev/cash_r_regression_202611.py          (~1–2 min)
Afterwards: rm valuation/_ev_v014.py   (not for the repo)
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from valuation import ev as NEW

CHUNK = 100
FRESH_DAYS = 270
BAND = 0.05
YAHOO_FILES = ["data/ev_gap_discovery_202611.jsonl", "data/cash_values_202611.jsonl"]
WATCH = ["PG", "MMM", "HSY", "LULU", "DHR", "CSX", "SBUX", "DOV", "GDDY", "EL", "WMB", "GEV", "AAL",
         "WAB", "CVX", "ADP", "ICE", "TGT", "AAPL"]


def load_old():
    try:
        from valuation import _ev_v014 as OLD
    except ImportError:
        sys.exit("Λείπει το valuation/_ev_v014.py. Τρέξε πρώτα:\n"
                 "    git show HEAD:valuation/ev.py > valuation/_ev_v014.py")
    assert not hasattr(OLD, "RESTRICTED"), "Το _ev_v014.py είναι ήδη το νέο ev.py (ξαναπάρ' το από το HEAD)"
    assert hasattr(NEW, "RESTRICTED"), "Το valuation/ev.py δεν είναι το νέο"
    return OLD


def compare_one(rows: pd.DataFrame, OLD, today: pd.Timestamp) -> dict:
    old = OLD.cash_series(rows[rows["metric"].isin(OLD.CASH_METRICS)])
    new = NEW.cash_series(rows[rows["metric"].isin(NEW.CASH_METRICS)])
    o = old.set_index("period_end")["cash"] if not old.empty else pd.Series(dtype=float)
    n = new.set_index("period_end")["cash"] if not new.empty else pd.Series(dtype=float)
    common = o.index.intersection(n.index)
    changed = [(d, o[d], n[d]) for d in common if abs(n[d] - o[d]) > 1e-6 * max(abs(o[d]), 1)]
    lost = [d for d in o.index if d not in n.index]
    added = [d for d in n.index if d not in o.index]
    newest = lambda s: (s.index.max(), float(s[s.index.max()])) if len(s) else (None, None)
    od, ov = newest(o)
    nd, nv = newest(n)
    fresh = lambda d: d is not None and (today - d).days <= FRESH_DAYS
    method = new.sort_values("period_end")["method"].iloc[-1] if not new.empty else None
    # what the guard sees in the DATABASE: dates with both cash and cash+restricted
    val, _ = NEW._wide(rows[rows["metric"].isin(NEW.CASH_METRICS)]) if not rows.empty else (pd.DataFrame(), None)
    overlap = []
    for tag in NEW.RESTRICTED:
        if NEW.CASH in val.columns and tag in val.columns:
            b = val[[NEW.CASH, tag]].dropna()
            b = b[b[NEW.CASH] > 0]
            if len(b):
                overlap.append(f"{'R' if tag == NEW.RESTRICTED[0] else 'R2'}: {len(b)} κοινές,"
                               f" μέγ. {(b[tag] / b[NEW.CASH] - 1).abs().max() * 100:.1f}%")
    return {"changed": changed, "lost": lost, "added": len(added),
            "added_r": int(new["method"].fillna("").str.startswith("ταμείο με δεσμευμένα").sum()) if not new.empty else 0,
            "old_date": od, "old": ov, "new_date": nd, "new": nv, "method": method,
            "became_fresh": fresh(nd) and not fresh(od), "by_date": (o, n),
            "overlap": " · ".join(overlap) or "καμία κοινή ημ/νία C και R"}


def bn(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x / 1e9:,.2f}"


def load_yahoo() -> dict[str, tuple[str, float]]:
    out = {}
    for path in YAHOO_FILES:
        if not os.path.exists(path):
            print(f"⚠️  Δεν βρέθηκε το {path}")
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "kind" in r and r["kind"] != "cash":
                continue
            date, val = r.get("date"), r.get("target", r.get("y_total"))
            if date and val:
                out.setdefault(r["ticker"], (date, val))
    return out


def report(res: dict[str, dict], financial: set[str], yahoo: dict) -> None:
    nf = {t: r for t, r in res.items() if t not in financial}
    print(f"── {len(res)} εταιρείες, {len(nf)} χωρίς τις χρηματοοικονομικές ──\n")

    print("── 1. Ημερομηνίες που είχε ο ΠΑΛΙΟΣ κανόνας (πρέπει να μείνουν ίδιες) ──")
    ch = [(t, c) for t, r in res.items() for c in r["changed"]]
    lost = [(t, d) for t, r in res.items() for d in r["lost"]]
    print(f"  άλλαξαν: {len(ch)} ημ/νίες σε {len({t for t, _ in ch})} εταιρείες"
          f" · χάθηκαν: {len(lost)} σε {len({t for t, _ in lost})} εταιρείες")
    for t, (d, o, n) in ch[:20]:
        print(f"    {t:6s} {d.date()}  {bn(o)} → {bn(n)}")
    for t, d in lost[:20]:
        print(f"    {t:6s} {d.date()}  χάθηκε")

    print("\n── 2. Τι προσθέτει ο νέος κανόνας (χωρίς χρηματοοικονομικές) ──")
    print(f"  νέες ημ/νίες: {sum(r['added'] for r in nf.values())}"
          f" σε {sum(1 for r in nf.values() if r['added'])} εταιρείες"
          f" · από αυτές με «ταμείο με δεσμευμένα»: {sum(r['added_r'] for r in nf.values())}")
    fresh = sorted(t for t, r in nf.items() if r["became_fresh"])
    print(f"  εταιρείες που αποκτούν ΦΡΕΣΚΟ ταμείο: {len(fresh)}")
    for t in fresh:
        r = nf[t]
        print(f"    {t:6s} παλιό {bn(r['old']):>7s} ({str(r['old_date'])[:10]}) → νέο {bn(r['new']):>7s}"
              f" ({str(r['new_date'])[:10]})  {r['method']}")

    print("\n── 3. Έναντι Yahoo, ίδια ημερομηνία ──")
    rows = []
    for t, (date, y) in yahoo.items():
        if t not in nf:
            continue
        o, n = nf[t]["by_date"]
        d = pd.Timestamp(date)
        near = lambda s: s if s.empty else s[(s.index >= d - pd.Timedelta(days=3))
                                             & (s.index <= d + pd.Timedelta(days=3))]
        ov, nv = near(o), near(n)
        rows.append({"t": t, "old": float(ov.iloc[-1]) / y - 1 if len(ov) else None,
                     "new": float(nv.iloc[-1]) / y - 1 if len(nv) else None})
    c = pd.DataFrame(rows, columns=["t", "old", "new"])
    c[["old", "new"]] = c[["old", "new"]].astype(float)      # None → NaN, so .abs() works
    if not c.empty:
        for col, label in (("old", "παλιός"), ("new", "νέος")):
            x = c[col].dropna()
            print(f"  {label:7s}: με τιμή {len(x)}/{len(c)} · εντός 5% {int((x.abs() <= BAND).sum())}"
                  f" · πάνω {int((x > BAND).sum())} · κάτω {int((x < -BAND).sum())}")
        worse = c[c["old"].notna() & (c["old"].abs() <= BAND) & ~(c["new"].abs() <= BAND)]
        print(f"  ΧΑΛΑΣΑΝ: {len(worse)}" + (f" → {', '.join(worse['t'])}" if len(worse) else ""))
        gained = c[c["old"].isna() & c["new"].notna()]
        print(f"  ΝΕΕΣ με Yahoo: {len(gained)} · εντός 5% {int((gained['new'].abs() <= BAND).sum())}")
        for r in gained[gained["new"].abs() > BAND].itertuples():
            print(f"    εκτός 5%: {r.t} {r.new * 100:+.0f}%")

    print("\n── 4. Οι μετοχές του ελέγχου (νεότερη τιμή, δισ.) ──")
    for t in WATCH:
        r = res.get(t)
        if r is None:
            print(f"  {t:6s} δεν βρέθηκε")
            continue
        print(f"  {t:6s} {bn(r['old']):>7s} ({str(r['old_date'])[:10]}) → {bn(r['new']):>7s}"
              f" ({str(r['new_date'])[:10]})  {r['method'] or '—'}")
        print(f"         φρένο: {r['overlap']}")


def main() -> None:
    from db import get_engine
    from valuation.pe import load_universe
    from valuation.ttm import load_metrics
    from yahoo_check_202611 import load_sectors

    OLD = load_old()
    eng = get_engine()
    tickers = load_universe(eng)
    financial = {t for t, s in load_sectors(tickers).items() if s == "Financial Services"}
    today = pd.Timestamp.today().normalize()
    print(f"{len(tickers)} εταιρείες · ανάγνωση από τη βάση σε κομμάτια των {CHUNK}...")
    res = {}
    for i in range(0, len(tickers), CHUNK):
        rows = load_metrics(NEW.CASH_METRICS, tickers[i:i + CHUNK], eng)
        for t, g in rows.groupby("ticker"):
            try:
                res[t] = compare_one(g, OLD, today)
            except Exception as e:
                print(f"  {t}: σφάλμα {str(e)[:100]}")
        print(f"  ... {min(i + CHUNK, len(tickers))}/{len(tickers)}")
    print()
    report(res, financial, load_yahoo())


if __name__ == "__main__":
    main()
