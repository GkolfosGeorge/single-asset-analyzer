"""
dev/yahoo_check_202611.py — READ-ONLY: how far are the numbers of the card from
Yahoo's? Nothing is written to the database; results go to data/ (gitignored).

Run from the repo root, in Git Bash:

  1. Pre-flight on a few tickers (~30 sec):
         python dev/yahoo_check_202611.py --tickers MSFT,KLAC,JPM
  2. The full sample (~5 min):
         python dev/yahoo_check_202611.py
     → data/yahoo_sample_202611.csv   the sample (kept, so reruns use the SAME stocks)
     → data/yahoo_check_202611.csv    one row per ticker, every number
     → printed summary: deviation bands per metric, and every deviation > 15%

Our side uses the SAME functions as the mail card (fcf_history, ev_history,
build_ttm, daily_pe), so this measures the card, not a copy of it.

FOUR TRAPS a plain "our P/E vs Yahoo's P/E" would fall into:

1. DIFFERENT QUARTER. In earnings season Yahoo may already have the quarter
   that SEC filings (and so our DB) don't yet. MSFT: ours ends June, Yahoo's
   ends September → a "deviation" that is only freshness. So Yahoo's figures
   are rebuilt from its quarterly statements for the quarter WE ended on.
   Yahoo's published number (trailingEps, enterpriseValue) is kept next to it.

2. DIFFERENT PRICE. Yahoo's ratios use its live price, ours the last close.
   Both sides here use OUR close, so a difference is never about the price.

3. DIFFERENT DEFINITION.
   - info["freeCashflow"] is LEVERED FCF (after interest, with other
     adjustments), not CFO − capex. So we use its statement rows instead.
   - "Total Debt" on Yahoo includes lease obligations; ours deliberately
     doesn't. So there are two checks: with leases and without.
   - Yahoo has no EV/EBIT, so it's Yahoo EV / Yahoo TTM operating income.

4. RATIOS HIDE THE CAUSE. EV off by 10% tells you nothing; mcap, debt and
   cash are compared one by one, and `ev_driver` names the biggest gap.

Example of the deviation: ours 21, Yahoo 20 → +5.0%.
Bands: ≤5% ok, 5–15% look, >15% must have an explanation (roadmap, Φάση 1).
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

SAMPLE_OUT = "data/yahoo_sample_202611.csv"
OUT = "data/yahoo_check_202611.csv"
CLASSIFICATION = "data/classification.json"
SEED = 20261103
N_RANDOM = 35
SLEEP = 0.5                 # between tickers, to stay under Yahoo's throttling
PERIOD_TOL_DAYS = 10        # Yahoo's quarter end vs ours (52/53-week calendars)
QUARTER_GAP_DAYS = (75, 125)
BAND_OK, BAND_LOOK = 0.05, 0.15
FINANCIAL = "Financial Services"

# Stocks where something is EXPECTED to be hard. Reported apart, because hand-
# picked hard cases would make the rate of the random part meaningless.
HARD = {
    "split": ["KLAC", "APH", "NVDA", "NFLX", "ORLY", "CMG"],
    "bank/insurer": ["JPM", "BAC", "PGR"],
    "reit": ["O", "AMT"],
    "share classes": ["GOOGL", "V"],
    "no OperatingIncomeLoss": ["CVX", "XOM"],
    "leases": ["DAL", "HD"],
}
KNOWN_GAPS = {"PPL", "SPGI", "CRM", "KR", "APTV", "GHC", "LHX", "TAP", "PARA", "XOM",
              "V", "GOOG", "GOOGL"}      # roadmap: "Γνωστά κενά"

# label, needs a non-financial company?, printed in the summary?
METRICS = [
    ("shares", False, True), ("mcap", False, True),
    ("eps", False, True), ("pe", False, True), ("pe_pub", False, True),
    ("fcf", True, True), ("fcf_yield", True, True),
    ("debt", True, True), ("debt_xl", True, True), ("cash", True, True),
    ("ev", True, True), ("ev_xl", True, True), ("ev_pub", True, True),
    ("ebit", True, True), ("ev_ebit", True, True),
]
NOTES = {
    "pe_pub": "με το δημοσιευμένο trailingEps",
    "debt": "Yahoo με leases",
    "debt_xl": "Yahoo χωρίς leases",
    "ev": "Yahoo με leases",
    "ev_xl": "Yahoo χωρίς leases",
    "ev_pub": "δημοσιευμένο, δικό του τρίμηνο και τιμή",
}


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (no DB, no network — see tests/test_yahoo_check.py)
# ═════════════════════════════════════════════════════════════════════════

def yahoo_symbol(ticker: str) -> str:
    """BF.B → BF-B: the dot is the cause of the 404s of the card (roadmap, bug 11)."""
    return ticker.replace(".", "-")


def deviation(ours, yahoo) -> float:
    """
    (ours − Yahoo) / |Yahoo|. ours 21, Yahoo 20 → +0.05.
    |Yahoo| so that a negative FCF keeps its sign: ours −90, Yahoo −100 → +0.10.
    NaN if either is missing or Yahoo is 0.
    """
    if ours is None or yahoo is None or pd.isna(ours) or pd.isna(yahoo) or yahoo == 0:
        return np.nan
    return float((ours - yahoo) / abs(yahoo))


def band(dev: float) -> str:
    if pd.isna(dev):
        return "n/a"
    return "ok" if abs(dev) <= BAND_OK else ("look" if abs(dev) <= BAND_LOOK else "far")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval of a proportion: with n = 35, 90% really means roughly 75–97%."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, mid - half), min(1.0, mid + half))


def stratified_sample(sectors: dict[str, str], n: int, seed: int, exclude=()) -> list[str]:
    """
    Random sample, each sector in proportion to its size (largest remainders),
    at least 1 per sector. 700 companies, 70 in Energy, n = 35 → ~3–4 from Energy.
    Sorted inputs + fixed seed = the same stocks on any machine.
    """
    groups: dict[str, list[str]] = {}
    for t in sorted(sectors):
        if t not in exclude and sectors[t] not in (None, "Unknown"):
            groups.setdefault(sectors[t], []).append(t)
    total = sum(len(v) for v in groups.values())
    if total == 0:
        return []
    exact = {s: len(v) * n / total for s, v in groups.items()}
    take = {s: max(1, int(x)) for s, x in exact.items()}
    by_rest = sorted(groups, key=lambda s: exact[s] - int(exact[s]), reverse=True)
    i = 0
    while sum(take.values()) < n and i < 10 * len(by_rest):
        s = by_rest[i % len(by_rest)]
        if take[s] < len(groups[s]):
            take[s] += 1
        i += 1
    rng = random.Random(seed)
    out = []
    for s in sorted(groups):
        out += rng.sample(groups[s], min(take[s], len(groups[s])))
    return sorted(out)


def _row(table: pd.DataFrame | None, names: list[str]) -> pd.Series | None:
    """First of `names` that exists in a Yahoo statement. Yahoo renames rows at times."""
    if table is None or table.empty:
        return None
    for n in names:
        if n in table.index:
            s = pd.to_numeric(table.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return None


def _position(s: pd.Series, period_end: pd.Timestamp) -> int | None:
    """Position of the column closest to period_end, if within PERIOD_TOL_DAYS."""
    if s is None or s.empty or period_end is None or pd.isna(period_end):
        return None
    gaps = np.abs((s.index - period_end).days)
    i = int(np.argmin(gaps))
    return i if gaps[i] <= PERIOD_TOL_DAYS else None


def ttm_at(s: pd.Series | None, period_end: pd.Timestamp) -> float:
    """
    Sum of the 4 quarters ending at period_end. NaN unless all four exist and
    are consecutive: 3 quarters out of 4 would look like a −25% "deviation".
    """
    i = _position(s, period_end)
    if i is None or i < 3:
        return np.nan
    w = s.iloc[i - 3:i + 1]
    gaps = pd.Series(w.index).diff().dt.days.iloc[1:]
    if w.isna().any() or not gaps.between(*QUARTER_GAP_DAYS).all():
        return np.nan
    return float(w.sum())


def value_at(s: pd.Series | None, period_end: pd.Timestamp) -> float:
    i = _position(s, period_end)
    return float(s.iloc[i]) if i is not None else np.nan


def yahoo_numbers(info: dict, qis, qcf, qbs, period_end, close: float) -> dict:
    """Yahoo's figures for OUR quarter and at OUR close → flat dict, prefix y_."""
    info = info or {}
    y = {}

    # ── shares: all classes (GOOGL: A + B + C) ──
    shares = info.get("impliedSharesOutstanding") or info.get("sharesOutstanding")
    if not shares and info.get("marketCap") and info.get("regularMarketPrice"):
        shares = info["marketCap"] / info["regularMarketPrice"]
    y["y_shares"] = float(shares) if shares else np.nan
    y["y_mcap"] = y["y_shares"] * close

    # ── which quarters does Yahoo have? ──
    eps_row = _row(qis, ["Diluted EPS"])
    any_row = _row(qis, ["Net Income", "Total Revenue", "Diluted EPS"])
    latest = any_row.dropna().index.max() if any_row is not None and any_row.notna().any() else pd.NaT
    y["y_latest_q"] = latest
    y["yahoo_newer"] = bool(pd.notna(latest) and pd.notna(period_end)
                            and (latest - period_end).days > PERIOD_TOL_DAYS)

    # ── EPS / P/E ──
    trailing = info.get("trailingEps")
    y["y_eps_pub"] = float(trailing) if trailing is not None else np.nan
    y["y_eps"] = ttm_at(eps_row, period_end)
    y["y_eps_src"] = "quarters"
    if pd.isna(y["y_eps"]) and not y["yahoo_newer"]:     # Yahoo often lacks old quarterly EPS
        y["y_eps"], y["y_eps_src"] = y["y_eps_pub"], "trailingEps"
    y["y_pe"] = close / y["y_eps"] if y["y_eps"] and y["y_eps"] > 0 else np.nan
    y["y_pe_pub"] = close / y["y_eps_pub"] if y["y_eps_pub"] and y["y_eps_pub"] > 0 else np.nan

    # ── FCF = CFO − capex (NOT info["freeCashflow"], see trap 3) ──
    cfo = ttm_at(_row(qcf, ["Operating Cash Flow"]), period_end)
    capex = ttm_at(_row(qcf, ["Capital Expenditure"]), period_end)
    y["y_cfo"], y["y_capex"] = cfo, abs(capex) if pd.notna(capex) else np.nan
    y["y_fcf"] = cfo - y["y_capex"]
    y["y_fcf_yield"] = y["y_fcf"] / y["y_mcap"]

    # ── balance sheet ──
    debt = value_at(_row(qbs, ["Total Debt"]), period_end)
    leases = value_at(_row(qbs, ["Capital Lease Obligations"]), period_end)
    if pd.notna(debt) and pd.isna(leases):
        leases = 0.0
    cash = value_at(_row(qbs, ["Cash Cash Equivalents And Short Term Investments",
                               "Cash And Cash Equivalents"]), period_end)
    y["y_debt"], y["y_leases"], y["y_cash"] = debt, leases, cash
    y["y_debt_xl"] = debt - leases
    y["y_ev"] = y["y_mcap"] + debt - cash
    y["y_ev_xl"] = y["y_mcap"] + y["y_debt_xl"] - cash
    ev_pub = info.get("enterpriseValue")
    y["y_ev_pub"] = float(ev_pub) if ev_pub else np.nan

    # ── EBIT ── "as reported" is closest to OperatingIncomeLoss; Yahoo's
    # "Operating Income" is normalized (strips e.g. restructuring).
    y["y_ebit"] = ttm_at(_row(qis, ["Total Operating Income As Reported"]), period_end)
    y["y_ebit_src"] = "as reported"
    if pd.isna(y["y_ebit"]):
        y["y_ebit"] = ttm_at(_row(qis, ["Operating Income"]), period_end)
        y["y_ebit_src"] = "operating income"
    # Yahoo's "EBIT" (pretax + interest): reference for step 2 (CVX, XOM, KLAC)
    y["y_ebit_alt"] = ttm_at(_row(qis, ["EBIT"]), period_end)
    y["y_ev_ebit"] = y["y_ev_xl"] / y["y_ebit"] if y["y_ebit"] and y["y_ebit"] > 0 else np.nan
    return y


def ev_driver(r: dict) -> str:
    """Which piece explains the EV gap. EV gap = mcap gap + debt gap − cash gap."""
    parts = {
        "market cap": r.get("mcap", np.nan) - r.get("y_mcap", np.nan),
        "leases": -r.get("y_leases", np.nan),
        "χρέος": r.get("debt", np.nan) - r.get("y_debt_xl", np.nan),
        "ταμείο": -(r.get("cash", np.nan) - r.get("y_cash", np.nan)),
    }
    parts = {k: v for k, v in parts.items() if pd.notna(v)}
    if not parts:
        return ""
    k = max(parts, key=lambda p: abs(parts[p]))
    return f"{k} {parts[k] / 1e9:+,.1f} bn"


def split_hint(dev_shares: float, recent_splits: list[float]) -> str:
    """Shares off by about a split's ratio → the split was applied wrongly (or twice)."""
    if pd.isna(dev_shares) or not recent_splits:
        return ""
    x = 1 + dev_shares
    for r in recent_splits:
        if r > 1 and (abs(x / r - 1) < 0.05 or abs(x * r - 1) < 0.05):
            return f"μετοχές ×{x:.2f}: ταιριάζει με split {r:g}:1"
    return ""


def add_deviations(r: dict, financial: bool) -> dict:
    pairs = {"shares": "shares", "mcap": "mcap", "eps": "eps", "pe": "pe",
             "fcf": "fcf", "fcf_yield": "fcf_yield", "debt": "debt", "cash": "cash",
             "ev": "ev", "ebit": "ebit", "ev_ebit": "ev_ebit"}
    for name, ours in pairs.items():
        r[f"dev_{name}"] = deviation(r.get(ours), r.get(f"y_{name}"))
    r["dev_pe_pub"] = deviation(r.get("pe"), r.get("y_pe_pub"))
    r["dev_debt_xl"] = deviation(r.get("debt"), r.get("y_debt_xl"))
    r["dev_ev_xl"] = deviation(r.get("ev"), r.get("y_ev_xl"))
    r["dev_ev_pub"] = deviation(r.get("ev"), r.get("y_ev_pub"))
    if financial:      # deposits are their raw material: EV, EBIT, FCF don't apply
        for name, needs_nonfin, _ in METRICS:
            if needs_nonfin:
                r[f"dev_{name}"] = np.nan
    r["ev_driver"] = "" if financial else ev_driver(r)
    r["split_hint"] = split_hint(r.get("dev_shares", np.nan), r.get("recent_splits") or [])
    return r


def band_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, _, shown in METRICS:
        col = f"dev_{name}"
        if not shown or col not in df.columns:
            continue
        b = df[col].map(band)
        n = int((b != "n/a").sum())
        rows.append({"δείκτης": name, "N": n, "≤5%": int((b == "ok").sum()),
                     "5–15%": int((b == "look").sum()), ">15%": int((b == "far").sum()),
                     "n/a": int((b == "n/a").sum()),
                     "διάμεση απόκλιση": f"{df[col].median() * 100:+.1f}%" if n else "—",
                     "τι μετράει": NOTES.get(name, "")})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 2. DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════

def load_sectors(tickers: list[str]) -> dict[str, str]:
    """
    Reads the classification cache directly, even if older than 30 days: a
    stale sector is harmless for drawing a sample, and get_classification
    would throw the cache away and spend ~10 min refetching 778 tickers.
    """
    from valuation.pe import get_classification
    if os.path.exists(CLASSIFICATION):
        try:
            data = json.load(open(CLASSIFICATION, encoding="utf-8")).get("data", {})
            known = {t: data[t].get("sector") for t in tickers if t in data}
            if len(known) >= 0.8 * len(tickers):
                return known
        except Exception as e:
            print(f"⚠️  Το {CLASSIFICATION} δεν διαβάστηκε ({e})")
    return {t: v["sector"] for t, v in get_classification(tickers, CLASSIFICATION).items()}


def build_sample(engine) -> pd.DataFrame:
    """The sample is drawn ONCE, then read from the file."""
    if os.path.exists(SAMPLE_OUT):
        s = pd.read_csv(SAMPLE_OUT, keep_default_na=False)
        print(f"Δείγμα από {SAMPLE_OUT}: {len(s)} μετοχές")
        return s
    from valuation.pe import load_universe
    universe = load_universe(engine)
    sectors = load_sectors(universe)
    in_db = set(universe)
    hard = {t: g for g, ts in HARD.items() for t in ts}
    missing = sorted(set(hard) - in_db)
    if missing:
        print(f"⚠️  Εκτός universe, δεν μπαίνουν: {', '.join(missing)}")
    picked = stratified_sample(sectors, N_RANDOM, SEED, exclude=set(hard))
    rows = [{"ticker": t, "group": "random", "hard_reason": "", "sector": sectors.get(t, "")}
            for t in picked]
    rows += [{"ticker": t, "group": "hard", "hard_reason": g, "sector": sectors.get(t, "")}
             for t, g in hard.items() if t in in_db]
    s = pd.DataFrame(rows)
    s.to_csv(SAMPLE_OUT, index=False)
    print(f"Δείγμα: {len(picked)} τυχαίες + {len(s) - len(picked)} δύσκολες → {SAMPLE_OUT}")
    print(s[s["group"] == "random"]["sector"].value_counts().to_string())
    return s


def our_numbers(ticker: str, engine) -> dict:
    """The card's numbers, from the card's functions."""
    from datetime import date, timedelta
    from valuation import ev as E, pe as P, yields as Y
    from valuation.ttm import load_prices

    r = {"ticker": ticker, "as_of": None}
    today = pd.Timestamp(date.today())
    fcf = Y.fcf_history(ticker, engine, years=2)
    r["fcf_status"] = fcf.status
    if fcf.daily.empty:
        return r
    h = E.ev_history(ticker, engine, base=fcf)
    r["ev_status"], r["debt_method"] = h.status, h.debt_note
    last = h.daily.iloc[-1]
    r["as_of"] = h.daily.index[-1].date()
    r["close"] = float(last["close"])
    for c in ("shares", "mcap", "cfo_ttm", "capex_ttm", "debt", "cash", "ev", "ebit_ttm", "ev_ebit"):
        r[c] = float(last[c]) if pd.notna(last[c]) else np.nan
    r["fcf"], r["fcf_yield"] = float(last["fcf_ttm"]), float(last["fcf_yield"])
    r["ebit"] = r.pop("ebit_ttm")
    r["capex_tag"] = last.get("capex_tag")

    # P/E exactly as quick_snapshot builds it
    close, splits = load_prices([ticker], (today - timedelta(days=1200)).date(),
                                (today + timedelta(days=1)).date())
    sp = splits[ticker] if ticker in splits.columns else None
    r["recent_splits"] = [float(x) for x in sp[sp > 0].tolist()] if sp is not None else []
    eps = P.load_eps([ticker], engine)
    r["period_end"] = pd.NaT
    if not eps.empty and ticker in close.columns:
        ttm = P.build_ttm(eps, sp)
        d = P.daily_pe(close[ticker], ttm)
        if len(d):
            r["eps"] = float(d["eps_ttm"].iloc[-1]) if pd.notna(d["eps_ttm"].iloc[-1]) else np.nan
            r["pe"] = float(d["pe"].iloc[-1]) if pd.notna(d["pe"].iloc[-1]) else np.nan
        if not ttm.empty:
            r["period_end"] = pd.Timestamp(ttm["period_end"].iloc[-1])
    return r


def yahoo_raw(symbol: str):
    import yfinance as yf
    t = yf.Ticker(symbol)
    return t.info, t.quarterly_income_stmt, t.quarterly_cashflow, t.quarterly_balance_sheet


# ═════════════════════════════════════════════════════════════════════════
# 3. ORCHESTRATION + REPORT
# ═════════════════════════════════════════════════════════════════════════

def check_one(ticker: str, engine, financial: bool) -> dict:
    r = our_numbers(ticker, engine)
    if r.get("as_of") is None:
        return r
    info, qis, qcf, qbs = yahoo_raw(yahoo_symbol(ticker))
    r.update(yahoo_numbers(info, qis, qcf, qbs, r.get("period_end"), r["close"]))
    # same rule as the card (quick.py), plus the sector for those it misses
    card_rule = (str(r.get("ev_status", "")).startswith("χωρίς EBIT")
                 and str(r.get("fcf_status", "")).startswith("χωρίς capex"))
    r["financial"] = bool(financial or card_rule)
    r = add_deviations(r, r["financial"])
    r["recent_splits"] = ",".join(f"{x:g}" for x in r.get("recent_splits") or [])
    return r


def _crit(df: pd.DataFrame, col: str) -> str:
    v = df[col].dropna()
    if v.empty:
        return "κανένα δεδομένο"
    k = int((v.abs() <= BAND_OK).sum())
    lo, hi = wilson(k, len(v))
    return f"{k}/{len(v)} = {k / len(v):.0%} (95% διάστημα {lo:.0%}–{hi:.0%})"


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)
    newer = df[df["yahoo_newer"] == True] if "yahoo_newer" in df.columns else df.iloc[0:0]  # noqa: E712
    print(f"\n══ Τρίμηνα ══\n  Το Yahoo έχει νεότερο τρίμηνο από τη βάση σε {len(newer)}/{len(df)} μετοχές."
          "\n  Οι στήλες χωρίς _pub συγκρίνουν το ΙΔΙΟ τρίμηνο. Αν το Yahoo δεν κρατά αρκετά"
          "\n  παλιά τρίμηνα, βγαίνει n/a, όχι λάθος νούμερο.")

    for name, part in (("ΤΥΧΑΙΟ δείγμα", df[df["group"] == "random"]),
                       ("ΔΥΣΚΟΛΕΣ", df[df["group"] == "hard"])):
        if len(part):
            print(f"\n══ {name} ({len(part)} μετοχές) ══")
            print(band_table(part).to_string(index=False))

    rnd = df[df["group"] == "random"]
    if len(rnd):
        print("\n══ Κριτήριο Φάσης 1: ≥ 90% εντός 5% (μόνο το τυχαίο δείγμα) ══")
        for label, col in (("P/E", "dev_pe"), ("EV με leases", "dev_ev"),
                           ("EV χωρίς leases", "dev_ev_xl"), ("EV δημοσιευμένο", "dev_ev_pub")):
            if col in rnd.columns:
                print(f"  {label:<16} {_crit(rnd, col)}")

    print("\n══ Αποκλίσεις > 15% (καθεμία χρειάζεται εξήγηση) ══")
    n = 0
    for _, row in df.iterrows():
        bad = []
        for name, _, _ in METRICS:
            v = row.get(f"dev_{name}", np.nan)
            if pd.notna(v) and abs(v) > BAND_LOOK:
                bad.append(f"{name} {v * 100:+.0f}%")
        if bad:
            n += 1
            extra = [str(row[c]) for c in ("split_hint", "ev_driver") if row.get(c)]
            print(f"  {row['ticker']:<6} {row['group']:<6} " + " · ".join(bad)
                  + (f"   ← {'; '.join(extra)}" if extra else ""))
    if not n:
        print("  καμία")

    fail = df[df.get("as_of").isna()] if "as_of" in df.columns else df.iloc[0:0]
    errs = df[df["error"].astype(str) != ""] if "error" in df.columns else df.iloc[0:0]
    if len(fail) or len(errs):
        print("\n══ Δεν ελέγχθηκαν ══")
        for _, row in errs.iterrows():
            print(f"  {row['ticker']}: {row['error']}")
        for _, row in fail.iterrows():
            if not row.get("error"):
                print(f"  {row['ticker']}: {row.get('fcf_status', '')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", help="pre-flight: comma-separated, nothing is saved")
    args = ap.parse_args()

    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    from db import get_engine
    engine = get_engine()
    if args.tickers:
        sample = pd.DataFrame({"ticker": args.tickers.split(","), "group": "random",
                               "hard_reason": "", "sector": ""})
    else:
        sample = build_sample(engine)

    rows = []
    for i, s in enumerate(sample.itertuples(index=False), 1):
        print(f"[{i}/{len(sample)}] {s.ticker}", flush=True)
        financial = s.sector == FINANCIAL or s.hard_reason == "bank/insurer"
        try:
            r = check_one(s.ticker, engine, financial)
            r["error"] = ""
        except Exception as e:                      # one bad ticker never stops the run
            r = {"ticker": s.ticker, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        r.update(group=s.group, hard_reason=s.hard_reason, sector=s.sector,
                 known_gap=s.ticker in KNOWN_GAPS)
        rows.append(r)
        time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    if args.tickers:
        for r in rows:
            print(f"\n── {r['ticker']} ──")
            for k, v in r.items():
                if k.startswith(("dev_",)) or k in ("period_end", "y_latest_q", "y_eps_src",
                                                    "y_ebit_src", "debt_method", "ev_driver",
                                                    "split_hint", "error"):
                    print(f"  {k:<14} {f'{v * 100:+.1f}%' if k.startswith('dev_') and pd.notna(v) else v}")
    else:
        df.to_csv(OUT, index=False)
        print(f"\nΑποθηκεύτηκε: {OUT}")
    report(df)


if __name__ == "__main__":
    main()
