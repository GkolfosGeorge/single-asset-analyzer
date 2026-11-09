"""
dev/ev_regression_202611.py — READ-ONLY regression for the new cash/debt rules in
valuation/ev.py (Nov 2026). Compares the OLD and NEW ev.py on every company:
  1. newest cash and debt: what changed, and by how much
  2. history: how many dates each new rule touched
  3. against Yahoo (data/cash_values_202611.jsonl, same balance-sheet date):
     cash within 5%, old vs new; debt of the companies with convertibles
  4. the companies to watch, one line each (BXP, BKNG, AMGN, SMCI, PD, ...)
No SEC, no Yahoo calls: only the database and the saved file. Nothing is written.

Setup (Git Bash, repo root), BEFORE replacing valuation/ev.py:
    git show HEAD:valuation/ev.py > valuation/_ev_v013.py
Then put the new ev.py in place and run:
    python dev/ev_regression_202611.py          (~1–3 min)
Afterwards: rm valuation/_ev_v013.py   (not for the repo)
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "dev"))

import pandas as pd

from valuation import ev as NEW

YAHOO = "data/cash_values_202611.jsonl"
CHUNK = 100
BAND = 0.05
CHANGE = 0.01           # newest value moved by more than 1% = "changed"
WATCH = ["BXP", "BKNG", "AMGN", "SMCI", "PD", "DXCM", "DDOG", "AKAM", "NOW", "MIR", "VIAV", "PANW", "TYL", "EXR", "CHRS",
         "NVDA", "GOOGL", "KLAC", "PFE", "AAPL", "TGT", "SNPS", "APH", "WAT", "DISH", "LYB"]


def load_old():
    try:
        from valuation import _ev_v013 as OLD
    except ImportError:
        sys.exit("Λείπει το valuation/_ev_v013.py. Τρέξε πρώτα:\n"
                 "    git show HEAD:valuation/ev.py > valuation/_ev_v013.py")
    assert not hasattr(OLD, "CASH_METRICS"), "Το _ev_v013.py είναι ήδη το νέο ev.py (ξαναπάρ' το από το HEAD)"
    assert hasattr(NEW, "CASH_METRICS"), "Το valuation/ev.py δεν είναι το νέο"
    return OLD


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC
# ═════════════════════════════════════════════════════════════════════════

def by_date(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.set_index("period_end")[col] if not frame.empty else pd.Series(dtype=float)


def newest(frame: pd.DataFrame, col: str):
    """(period_end, value, method) of the newest known row, or (None, None, None)."""
    if frame.empty:
        return None, None, None
    r = frame.sort_values(["available", "period_end"]).iloc[-1]
    return r["period_end"], float(r[col]), r.get("method")


def compare_one(rows: pd.DataFrame, OLD) -> dict:
    old_cash = OLD.cash_series(rows[rows["metric"].isin([OLD.CASH, OLD.STI])])
    new_cash = NEW.cash_series(rows[rows["metric"].isin(NEW.CASH_METRICS)])
    old_debt = OLD.debt_series(rows[rows["metric"].isin(OLD.DEBT_METRICS)])
    new_debt = NEW.debt_series(rows[rows["metric"].isin(NEW.ALL_DEBT_METRICS)])
    out = {}
    for name, o, n, col in (("cash", old_cash, new_cash, "cash"), ("debt", old_debt, new_debt, "debt")):
        od, ov, _ = newest(o, col)
        nd, nv, nm = newest(n, col)
        out[name] = {"old_date": od, "old": ov, "new_date": nd, "new": nv, "method": nm}
        out[name + "_by_date"] = (by_date(o, col), by_date(n, col))
    methods = new_cash["method"].fillna("") if not new_cash.empty else pd.Series(dtype=object)
    out["hist"] = {
        "cash_inside": int(methods.str.startswith("ταμείο (τα χρεόγραφα").sum()),
        "cash_new_tag": int((methods.str.startswith("ταμείο + ")
                             & (methods != f"ταμείο + {NEW.STI}")).sum()),
        "cash_skipped": int(len(set(by_date(old_cash, "cash").index) - set(by_date(new_cash, "cash").index))),
        "debt_conv_only": int(new_debt["method"].fillna("").str.startswith("μόνο μετατρέψιμα").sum())
        if not new_debt.empty else 0,
    }
    return out


def kind(old, new) -> str:
    if old is None and new is None:
        return "κανένα"
    if old is None:
        return "ΝΕΟ"
    if new is None:
        return "ΧΑΘΗΚΕ"
    if old == 0 and new == 0:
        return "ίδιο"
    if old == 0 or abs(new / old - 1) > CHANGE:
        return "πάνω" if new > old else "κάτω"
    return "ίδιο"


def dev(a, b):
    if a is None or b in (None, 0) or pd.isna(a) or pd.isna(b):
        return None
    return a / b - 1


# ═════════════════════════════════════════════════════════════════════════
# 2. REPORT
# ═════════════════════════════════════════════════════════════════════════

def bn(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x / 1e9:,.2f}"


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:+.0f}%"


def report(res: dict[str, dict], financial: set[str], yahoo: dict[str, dict]) -> None:
    nf = {t: r for t, r in res.items() if t not in financial}
    print(f"── {len(res)} εταιρείες, {len(nf)} χωρίς τις χρηματοοικονομικές ──\n")

    print("── 1. Νεότερη τιμή: παλιός έναντι νέου κανόνα (χωρίς χρηματοοικονομικές) ──")
    for name in ("cash", "debt"):
        k = pd.Series([kind(r[name]["old"], r[name]["new"]) for r in nf.values()], dtype=object)
        print(f"  {'ταμείο' if name == 'cash' else 'χρέος':6s}: " +
              " · ".join(f"{lab} {int((k == lab).sum())}" for lab in ["ίδιο", "πάνω", "κάτω", "ΝΕΟ", "ΧΑΘΗΚΕ"]))
    print("\n  Ταμείο: οι 25 μεγαλύτερες αλλαγές (δισ.)")
    ch = [(t, r["cash"]) for t, r in nf.items() if kind(r["cash"]["old"], r["cash"]["new"]) not in ("ίδιο", "κανένα")]
    ch.sort(key=lambda x: -abs((x[1]["new"] or 0) - (x[1]["old"] or 0)))
    for t, c in ch[:25]:
        print(f"  {t:6s} {str(c['new_date'])[:10]}  {bn(c['old']):>8s} → {bn(c['new']):>8s}  {c['method']}")
    print("\n  Χρέος: όλες οι αλλαγές")
    ch = [(t, r["debt"]) for t, r in nf.items() if kind(r["debt"]["old"], r["debt"]["new"]) not in ("ίδιο", "κανένα")]
    for t, c in sorted(ch):
        print(f"  {t:6s} παλιό {bn(c['old']):>7s} ({str(c['old_date'])[:10]})"
              f" → νέο {bn(c['new']):>7s} ({str(c['new_date'])[:10]})  {c['method']}")
    if not ch:
        print("  καμία")

    print("\n── 2. Ιστορικό: πόσες ημερομηνίες άγγιξε κάθε κανόνας ──")
    h = pd.DataFrame([r["hist"] for r in nf.values()], index=list(nf))
    for col, label in (("cash_new_tag", "ταμείο + νέο tag χρεογράφων"),
                       ("cash_inside", "χρεόγραφα μέσα στο ταμείο (δεν προστίθενται)"),
                       ("cash_skipped", "παραλείφθηκαν (κενό χρεογράφων)"),
                       ("debt_conv_only", "χρέος μόνο από μετατρέψιμα")):
        print(f"  {label:46s} {int(h[col].sum()):6d} ημ/νίες σε {int((h[col] > 0).sum())} εταιρείες")

    print("\n── 3. Έναντι Yahoo, ίδια ημερομηνία ισολογισμού ──")
    cash_rows, debt_rows = [], []
    for t, y in yahoo.items():
        if t not in nf or not y.get("date"):
            continue
        d = pd.Timestamp(y["date"])
        o, n = nf[t]["cash_by_date"]
        if y.get("y_total"):
            cash_rows.append({"ticker": t, "old": dev(o.get(d), y["y_total"]), "new": dev(n.get(d), y["y_total"])})
        if y.get("y_debt") and y.get("has_conv"):
            o, n = nf[t]["debt_by_date"]
            debt_rows.append({"ticker": t, "date": y["date"], "old": o.get(d), "new": n.get(d), "yahoo": y["y_debt"]})
    c = pd.DataFrame(cash_rows)
    if not c.empty:
        for col, label in (("old", "παλιός"), ("new", "νέος")):
            x = c[col].dropna()
            print(f"  ταμείο, {label:7s}: {int((x.abs() <= BAND).sum())}/{len(x)} εντός 5%"
                  f" · πάνω {int((x > BAND).sum())} · κάτω {int((x < -BAND).sum())}")
        worse = c[(c["old"].abs() <= BAND) & ~(c["new"].abs() <= BAND)]
        print(f"  ΧΑΛΑΣΑΝ με τον νέο (ήταν εντός 5%): {len(worse)}"
              + (" → " + ", ".join(f"{r.ticker} {pct(r.new)}" for r in worse.itertuples()) if len(worse) else ""))
    print("\n  Χρέος των εταιρειών με μετατρέψιμα (Yahoo χωρίς leases):")
    for r in sorted(debt_rows, key=lambda r: r["ticker"]):
        print(f"  {r['ticker']:6s} {r['date']}  παλιό {bn(r['old']):>7s}  νέο {bn(r['new']):>7s}"
              f"  Yahoo {bn(r['yahoo']):>7s}  νέο έναντι Yahoo {pct(dev(r['new'], r['yahoo']))}")

    print("\n── 4. Οι μετοχές του ελέγχου (νεότερη τιμή, δισ.) ──")
    for t in WATCH:
        r = res.get(t)
        if r is None:
            print(f"  {t:6s} δεν βρέθηκε")
            continue
        c, d = r["cash"], r["debt"]
        print(f"  {t:6s} ταμείο {bn(c['old']):>7s} → {bn(c['new']):>7s}   χρέος {bn(d['old']):>7s} → {bn(d['new']):>7s}"
              f"   ({d['method'] or 'χωρίς χρέος'})")


# ═════════════════════════════════════════════════════════════════════════
# 3. RUN
# ═════════════════════════════════════════════════════════════════════════

def load_yahoo() -> dict[str, dict]:
    if not os.path.exists(YAHOO):
        print(f"⚠️  Δεν βρέθηκε το {YAHOO}: η ενότητα 3 θα είναι κενή")
        return {}
    out = {}
    for line in open(YAHOO, encoding="utf-8"):
        r = json.loads(line)
        r["has_conv"] = any(k in r.get("values", {}) for k in NEW.CONV_METRICS)
        out[r["ticker"]] = r
    return out


def main() -> None:
    from db import get_engine
    from valuation.pe import load_universe
    from valuation.ttm import load_metrics
    from yahoo_check_202611 import load_sectors

    OLD = load_old()
    eng = get_engine()
    tickers = load_universe(eng)
    financial = {t for t, s in load_sectors(tickers).items() if s == "Financial Services"}
    metrics = sorted(set(NEW.ALL_DEBT_METRICS + NEW.CASH_METRICS + OLD.DEBT_METRICS + [OLD.CASH, OLD.STI]))
    print(f"{len(tickers)} εταιρείες · ανάγνωση από τη βάση σε κομμάτια των {CHUNK}...")
    res = {}
    for i in range(0, len(tickers), CHUNK):
        rows = load_metrics(metrics, tickers[i:i + CHUNK], eng)
        for t, g in rows.groupby("ticker"):
            try:
                res[t] = compare_one(g, OLD)
            except Exception as e:
                print(f"  {t}: σφάλμα {str(e)[:100]}")
        print(f"  ... {min(i + CHUNK, len(tickers))}/{len(tickers)}")
    print()
    report(res, financial, load_yahoo())


if __name__ == "__main__":
    main()
