"""
valuation/pe.py — Point-in-time P/E (TTM) for one stock and for its peers.

Three traps this module exists to avoid:

1. LOOKAHEAD. A quarter's EPS only enters the P/E on the day AFTER its
   filing (`filed_date + 1`), never on `period_end`. Same rule as the rest
   of the project (join on filed_date).

2. SPLITS. yfinance's `Close` is split-adjusted to TODAY's share units,
   but SEC EPS is in the share units of the day it was FILED (earliest
   filing wins in our DB). Example, AAPL 4:1 split in Aug 2020: FY2019
   diluted EPS was filed as $11.89, while the 2019 price is shown as ~$50
   (the real price was ~$200). 50 / 11.89 = 4.2 — a fake P/E. Dividing
   the EPS by every split that happened AFTER its filing date gives
   11.89 / 4 = 2.97 and 50 / 2.97 ≈ 17 — the true P/E.
   We use `Close` with auto_adjust=False (splits only, NOT dividends):
   a dividend-adjusted price would make the old P/E look artificially low.

3. MISSING Q4. Companies file 10-Qs for Q1–Q3 and a 10-K for the whole
   year, so there is no discrete Q4 row. Q4 = FY − (Q1+Q2+Q3).
   Example: FY EPS 10.00, quarters 2.10 + 2.40 + 2.30 → Q4 = 3.20.
   (Small approximation: EPS is per average share of each period, so
   quarterly EPS don't add up exactly — standard industry practice.)

TTM ("trailing twelve months") EPS = sum of the last 4 discrete quarters.
The generic TTM logic (any metric, YTD → quarters) lives in ttm.py.

Peers: every ticker in the fundamentals universe with the same yfinance
`sector` (or `industry`). Peer P/E per day = MEDIAN of peers with
positive TTM EPS. Known limitations (to be fixed in Βήμα 3): peers are
TODAY's universe and TODAY's classification (survivorship), and
loss-makers are excluded rather than handled.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from . import ttm as _ttm
from .ttm import QUARTER_DAYS, YEAR_DAYS, load_metric, load_prices, splits_after  # noqa: F401

EPS_METRIC = "EarningsPerShareDiluted"
MAX_EPS_AGE_DAYS = 270        # newest TTM older than this → company stopped filing → no P/E
SPLIT_LOOKBACK_DAYS = 500     # download prices/splits this far before the chart start
MIN_PEERS = 5                 # below this, a "sector median" is not meaningful
CLASSIFICATION_MAX_AGE_DAYS = 30


# ═════════════════════════════════════════════════════════════════════════
# 1. DATA ACCESS (DB, yfinance)
# ═════════════════════════════════════════════════════════════════════════

def load_universe(engine) -> list[str]:
    """Active universe = companies JOIN fundamentals (not companies alone)."""
    sql = text("""
        SELECT DISTINCT f.ticker
        FROM fundamentals f
        JOIN companies c ON c.ticker = f.ticker
        ORDER BY f.ticker
    """)
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(sql).fetchall()]


def load_eps(tickers: list[str], engine, as_of=None) -> pd.DataFrame:
    """Diluted EPS rows (original 10-Q/10-K only) for many tickers, ONE query."""
    return load_metric(EPS_METRIC, tickers, engine, as_of=as_of)


def _fetch_one_classification(yf, t: str, retries: int = 1) -> dict | None:
    """
    One ticker's sector/industry, or None. Yahoo sometimes answers a fast
    request with a stripped-down `info` (no sector) instead of an error —
    so an empty answer gets a pause and a retry instead of being trusted.
    """
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        sector, industry = info.get("sector"), info.get("industry")
        if sector:
            return {"sector": sector, "industry": industry or "Unknown"}
        if attempt < retries:
            time.sleep(2.0)
    return None


def get_classification(tickers: list[str], cache_path: str = "data/classification.json",
                       max_age_days: int = CLASSIFICATION_MAX_AGE_DAYS,
                       retry_failed_after_days: int = 7) -> dict[str, dict]:
    """
    {ticker: {"sector": ..., "industry": ...}} from yfinance, cached locally.

    Only SUCCESSFUL lookups are cached as data. Failures (delisted tickers →
    404, or Yahoo throttling) go to a separate "_failed" list with a
    timestamp and are retried after `retry_failed_after_days` — so one bad
    run can't poison the cache with "Unknown" for a month.
    """
    import yfinance as yf

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except Exception:
            cache = {}

    now = time.time()
    stale = (now - cache.get("_cached_at", 0)) / 86400 > max_age_days
    # Drop "Unknown" entries written by the first version of this function.
    data = {} if stale else {t: v for t, v in cache.get("data", {}).items()
                             if v.get("sector") not in (None, "Unknown")}
    failed = {} if stale else {t: ts for t, ts in cache.get("_failed", {}).items()
                               if (now - ts) / 86400 < retry_failed_after_days}
    to_fetch = [t for t in tickers if t not in data and t not in failed]

    if to_fetch:
        print(f"🔄 Sector/industry για {len(to_fetch)} tickers (μετά cache)...")
        for i, t in enumerate(to_fetch, 1):
            res = _fetch_one_classification(yf, t)
            if res:
                data[t] = res
                failed.pop(t, None)
            else:
                failed[t] = now
            if i % 50 == 0:
                print(f"  ✅ {i}/{len(to_fetch)}")
            time.sleep(0.3)
        cached_at = now if stale or "_cached_at" not in cache else cache["_cached_at"]
        path.write_text(json.dumps({"data": data, "_failed": failed,
                                    "_cached_at": cached_at}, indent=1))
        if failed:
            print(f"  ⚠️  Χωρίς sector: {len(failed)} tickers "
                  f"(π.χ. {', '.join(sorted(failed)[:8])}) — ξαναδοκιμάζονται σε "
                  f"{retry_failed_after_days} μέρες")

    unknown = {"sector": "Unknown", "industry": "Unknown"}
    return {t: data.get(t, unknown) for t in tickers}


# ═════════════════════════════════════════════════════════════════════════
# 2. PURE LOGIC (no DB, no network — testable with synthetic data)
# ═════════════════════════════════════════════════════════════════════════

# The TTM machinery lives in ttm.py (shared by every metric). These thin
# wrappers keep the EPS-specific names that the notebook and pe_plots use.

def _build_quarters(eps: pd.DataFrame, splits: pd.Series | None = None):
    return _ttm._build_quarters(eps, splits, per_share=True)


def build_ttm(eps: pd.DataFrame, splits: pd.Series | None = None) -> pd.DataFrame:
    """One ticker's EPS rows → columns period_end, available, eps_ttm (today's share units)."""
    return _ttm.build_ttm(eps, splits, per_share=True, value_col="eps_ttm")


def debug_ttm(ticker: str, engine, since: str = "2024-01-01") -> None:
    """Step-by-step TTM EPS for one ticker, WITH split adjustment (as the charts use it)."""
    _ttm.debug_ttm(ticker, engine, EPS_METRIC, per_share=True, since=since,
                   max_age_days=MAX_EPS_AGE_DAYS)


def daily_pe(close: pd.Series, ttm: pd.DataFrame) -> pd.DataFrame:
    """Daily close + the TTM EPS known that day → P/E (NaN when EPS ≤ 0 or stale)."""
    close = close.dropna()
    res = pd.DataFrame({"close": close})
    res["eps_ttm"], res["pe"], res["loss"] = np.nan, np.nan, False
    if close.empty or ttm.empty:
        return res

    # pandas ≥2 keeps the resolution it was given: DB dates arrive as
    # datetime64[s], yfinance as [ns]/[us]. merge_asof refuses to mix them,
    # so bring both sides to the same unit first.
    left = pd.DataFrame({"date": pd.DatetimeIndex(close.index).as_unit("ns")})
    right = ttm.sort_values("available")[["available", "period_end", "eps_ttm"]].copy()
    for c in ("available", "period_end"):
        right[c] = pd.to_datetime(right[c]).dt.as_unit("ns")
    m = pd.merge_asof(
        left, right,
        left_on="date", right_on="available", direction="backward",
    )
    stale = (m["date"] - m["period_end"]).dt.days > MAX_EPS_AGE_DAYS
    eps = m["eps_ttm"].where(~stale).to_numpy()
    res["eps_ttm"] = eps
    res["loss"] = eps <= 0
    res["pe"] = np.where(eps > 0, close.to_numpy() / eps, np.nan)
    return res


def peer_band(peer_pe: pd.DataFrame, min_peers: int = MIN_PEERS) -> pd.DataFrame:
    """Row-wise median and 25%–75% band of peer P/Es; NaN on days with too few peers."""
    peer_pe = peer_pe.apply(pd.to_numeric, errors="coerce").astype(float)
    n = peer_pe.notna().sum(axis=1)
    out = pd.DataFrame({
        "median": peer_pe.median(axis=1),
        "q25": peer_pe.quantile(0.25, axis=1),
        "q75": peer_pe.quantile(0.75, axis=1),
        "n_peers": n,
    })
    out.loc[n < min_peers, ["median", "q25", "q75"]] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════
# 3. ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PEHistory:
    ticker: str
    stock: pd.DataFrame                       # close, eps_ttm, pe, loss
    peers: pd.DataFrame                       # median, q25, q75, n_peers
    peer_group: str                           # "sector" | "industry"
    group_name: str                           # e.g. "Consumer Cyclical"
    peer_tickers: list[str] = field(default_factory=list)
    peer_status: dict[str, str] = field(default_factory=dict)   # why each peer is in/out today


def pe_history(ticker: str, engine, years: int = 10, as_of=None,
               peer_group: str = "sector") -> PEHistory:
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=years)
    dl_start = (start - timedelta(days=SPLIT_LOOKBACK_DAYS)).date()

    universe = sorted(set(load_universe(engine)) | {ticker})
    cls = get_classification(universe)
    group_name = cls[ticker][peer_group]
    if group_name == "Unknown":
        print(f"⚠️  Δεν βρέθηκε {peer_group} για {ticker} στο yfinance → χωρίς σύγκριση με κλάδο.")
    peers = [] if group_name == "Unknown" else [
        t for t in universe if t != ticker and cls[t][peer_group] == group_name]

    tickers = [ticker] + peers
    print(f"⬇️  Τιμές + EPS για {ticker} και {len(peers)} peers ({group_name})...")
    close, splits = load_prices(tickers, dl_start, (end + timedelta(days=1)).date())
    eps = load_eps(tickers, engine, as_of=as_of)

    eps_tickers = set(eps["ticker"].unique())

    def _one(t):
        """(daily frame or None, status of this ticker on the last day)"""
        if t not in close.columns or close[t].dropna().empty:
            return None, "χωρίς τιμές (yfinance)"
        if t not in eps_tickers:
            return None, "χωρίς EPS στη βάση"
        sp = splits[t] if t in splits.columns else None
        ttm = build_ttm(eps[eps["ticker"] == t], sp)
        if ttm.empty:
            return None, "EPS αλλά χωρίς συνεχόμενα τρίμηνα (TTM)"
        d = daily_pe(close[t], ttm)
        d = d[(d.index >= start) & (d.index <= end)]
        d = d.astype({"close": float, "eps_ttm": float, "pe": float, "loss": bool})
        if d.empty:
            return None, "χωρίς τιμές στο διάστημα"
        last = d.iloc[-1]
        if d.index[-1] < end - timedelta(days=10):
            status = "σταμάτησε να διαπραγματεύεται"
        elif pd.notna(last["pe"]):
            status = "ok"
        elif last["loss"]:
            status = "ζημιές (EPS ≤ 0)"
        else:
            status = f"παλιό EPS (> {MAX_EPS_AGE_DAYS} μέρες)"
        return d, status

    stock, _ = _one(ticker)
    if stock is None:
        stock = pd.DataFrame(columns=["close", "eps_ttm", "pe", "loss"], dtype=float)
    peer_pe, status = {}, {}
    for t in peers:
        d, status[t] = _one(t)
        if d is not None and d["pe"].notna().any():
            peer_pe[t] = d["pe"]
    if peer_pe:
        band = peer_band(pd.DataFrame(peer_pe).reindex(stock.index).astype(float))
    else:
        band = pd.DataFrame(np.nan, index=stock.index, columns=["median", "q25", "q75", "n_peers"])

    return PEHistory(ticker, stock, band.astype(float), peer_group, group_name,
                     sorted(peer_pe), status)


def print_peer_coverage(h: PEHistory, examples: int = 6) -> None:
    """Why only N of the peers count today — the sector median is only as good as this."""
    if not h.peer_status:
        return
    st = pd.Series(h.peer_status)
    print(f"── Κάλυψη peers: {h.peer_group} «{h.group_name}» ({len(st)} εταιρείες) ──")
    for reason, grp in st.groupby(st).groups.items():
        names = sorted(grp)
        more = f", +{len(names) - examples}" if len(names) > examples else ""
        print(f"  {len(names):>3}  {reason:<42} {', '.join(names[:examples])}{more}")


def print_pe_summary(h: PEHistory) -> None:
    s = h.stock.dropna(subset=["close"])
    if s.empty:
        print("Δεν υπάρχουν τιμές.")
        return
    last = s.iloc[-1]
    print(f"── {h.ticker} — P/E (TTM) ── {s.index[-1].date()} ──")
    if pd.isna(last["pe"]):
        why = "ζημιές (EPS ≤ 0)" if last["loss"] else "δεν υπάρχει πρόσφατο EPS"
        print(f"  P/E: —  ({why})")
        return
    pe_hist = s["pe"].dropna()
    pct = (pe_hist < last["pe"]).mean() * 100
    print(f"  Τιμή {last['close']:.2f} / EPS TTM {last['eps_ttm']:.2f} = P/E {last['pe']:.1f}")
    print(f"  Διάμεσος P/E της ίδιας ({pe_hist.index[0].year}–): {pe_hist.median():.1f}"
          f"  → το σημερινό είναι υψηλότερο από το {pct:.0f}% των ημερών")
    med = h.peers["median"].reindex(s.index).iloc[-1] if not h.peers.empty else np.nan
    if pd.notna(med):
        n = int(h.peers["n_peers"].reindex(s.index).iloc[-1])
        prem = (last["pe"] / med - 1) * 100
        print(f"  Διάμεσος P/E {h.peer_group} «{h.group_name}» ({n} εταιρείες): {med:.1f}"
              f"  → {'ακριβότερη' if prem > 0 else 'φθηνότερη'} κατά {abs(prem):.0f}%")
    else:
        print(f"  Σύγκριση με {h.peer_group}: μη διαθέσιμη (λιγότερες από {MIN_PEERS} εταιρείες)")


# ═════════════════════════════════════════════════════════════════════════
# 4. P/E BANDS & RETURN DECOMPOSITION (built on PEHistory.stock)
# ═════════════════════════════════════════════════════════════════════════

BAND_PERCENTILES = (10, 25, 50, 75, 90)
UNSTABLE_X = 3.0     # EPS ×3 or ÷3 within one year → "base effect", not interpretable


def pe_band_levels(stock: pd.DataFrame, percentiles=BAND_PERCENTILES) -> dict[int, float]:
    """
    The stock's OWN P/E percentiles over the window, e.g. {25: 16.2, 50: 19.8, ...}.

    Why own percentiles and not fixed 15/20/25: a "normal" P/E differs a lot
    by business (utility ~17, software ~35). Fixed lines would call every
    software stock expensive forever.

    NOTE: percentiles use the WHOLE window, i.e. the 2018 band already
    "knows" the P/Es of 2024. Fine for a descriptive chart; NOT usable as a
    backtest signal (that would need an expanding window).
    """
    pe = stock["pe"].dropna()
    if pe.empty:
        return {}
    return {p: float(np.percentile(pe, p)) for p in percentiles}


def pe_band_prices(stock: pd.DataFrame, levels: dict[int, float]) -> pd.DataFrame:
    """
    "Where the price would be at P/E = level": price_at_level = EPS_TTM × level.
    Example: EPS 5$, levels 15/20/25 → 75$ / 100$ / 125$.
    NaN on loss days (no meaningful band).
    """
    eps = stock["eps_ttm"].where(stock["eps_ttm"] > 0)
    return pd.DataFrame({p: eps * lvl for p, lvl in levels.items()}, index=stock.index)


def _decompose(p0, p1, e0, e1):
    """
    price ratio = EPS ratio × P/E ratio  (exact: P = EPS × P/E).
    Split the % price change into two parts that ADD UP exactly, using logs
    (logs turn × into +):
        price 50→60 (+20%), EPS ×1.5, P/E ×0.8
        ln1.2=0.182 = ln1.5 (0.405) + ln0.8 (−0.223)
        EPS part = 20% × 0.405/0.182 = +44.6%
        P/E part = 20% × −0.223/0.182 = −24.6%     → 44.6 − 24.6 = 20 ✓
    """
    price_r, eps_r = p1 / p0, e1 / e0
    pe_r = price_r / eps_r
    lp, le, lm = np.log(price_r), np.log(eps_r), np.log(pe_r)
    total = price_r - 1
    if abs(lp) < 1e-3:                       # ~flat price: shares undefined → log %
        eps_part, pe_part = le, lm
    else:
        eps_part, pe_part = total * le / lp, total * lm / lp
    return price_r, eps_r, pe_r, total, eps_part, pe_part


def decompose_by_year(stock: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar-year price change split into "earnings grew" vs "P/E changed".
    A year is skipped when EPS ≤ 0 at either end (P/E undefined).
    The first row runs from the first valid day of the window.
    Price return only — dividends are not included.
    """
    s = stock.dropna(subset=["close"])
    valid = s[s["pe"].notna()]
    if valid.empty:
        return pd.DataFrame()

    points = [valid.index[0]] + [g.index[-1] for _, g in s.groupby(s.index.year)]
    points = sorted(set(points))
    rows = []
    for a, b in zip(points[:-1], points[1:]):
        r0, r1 = s.loc[a], s.loc[b]
        if pd.isna(r0["pe"]) or pd.isna(r1["pe"]) or (b - a).days < 30:
            continue
        price_r, eps_r, pe_r, total, ep, mp = _decompose(r0["close"], r1["close"],
                                                         r0["eps_ttm"], r1["eps_ttm"])
        if a == points[0] and a.year == b.year and (b - a).days < 340:
            label = f"{b.year} (από {a:%d/%m})"          # window starts mid-year
        elif b == points[-1] and b.year != a.year and (b - a).days < 340:
            label = f"{b.year} YTD"                         # current, unfinished year
        else:
            label = str(b.year)
        # EPS near zero at one end (e.g. 0.05 → 1.20 = ×24): the split is
        # mathematically right but meaningless (±300% bars cancelling out).
        unstable = not (1 / UNSTABLE_X <= eps_r <= UNSTABLE_X)
        rows.append({"period": label, "start": a, "end": b, "unstable": unstable,
                     "price_x": price_r, "eps_x": eps_r, "pe_x": pe_r,
                     "price_pct": total * 100, "from_eps_pct": ep * 100, "from_pe_pct": mp * 100})
    return pd.DataFrame(rows)


def decompose_total(stock: pd.DataFrame) -> dict | None:
    """Whole window: first valid day → last valid day."""
    valid = stock.dropna(subset=["close", "pe"])
    if len(valid) < 2:
        return None
    a, b = valid.iloc[0], valid.iloc[-1]
    price_r, eps_r, pe_r, total, ep, mp = _decompose(a["close"], b["close"], a["eps_ttm"], b["eps_ttm"])
    return {"start": valid.index[0], "end": valid.index[-1], "price_x": price_r,
            "eps_x": eps_r, "pe_x": pe_r, "price_pct": total * 100,
            "from_eps_pct": ep * 100, "from_pe_pct": mp * 100,
            "pe_start": a["pe"], "pe_end": b["pe"]}


def print_band_summary(h: PEHistory) -> None:
    levels = pe_band_levels(h.stock)
    last = h.stock.dropna(subset=["close"]).iloc[-1] if not h.stock.empty else None
    if not levels or last is None or pd.isna(last["pe"]):
        print("P/E bands: μη διαθέσιμα (ζημιές ή έλλειψη EPS).")
        return
    print(f"── {h.ticker} — P/E bands (ιστορικό της ίδιας) ──")
    for p, lvl in levels.items():
        print(f"  {p:>2}% των ημερών είχε P/E κάτω από {lvl:5.1f} → τιμή σήμερα θα ήταν "
              f"{last['eps_ttm'] * lvl:8.2f}$")
    print(f"  Πραγματική τιμή: {last['close']:.2f}$ (P/E {last['pe']:.1f})")


def print_decomposition(h: PEHistory) -> None:
    t = decompose_total(h.stock)
    if t is None:
        print("Ανάλυση απόδοσης: μη διαθέσιμη.")
        return
    print(f"── {h.ticker} — από πού ήρθε η απόδοση "
          f"({t['start']:%m/%Y} → {t['end']:%m/%Y}, χωρίς μερίσματα) ──")
    print(f"  Τιμή ×{t['price_x']:.2f} = κέρδη ×{t['eps_x']:.2f}  ×  P/E ×{t['pe_x']:.2f}"
          f"   (P/E {t['pe_start']:.1f} → {t['pe_end']:.1f})")
    print(f"  {t['price_pct']:+.0f}% συνολικά = {t['from_eps_pct']:+.0f}% από κέρδη "
          f"{t['from_pe_pct']:+.0f}% από αλλαγή P/E")
    if t["pe_start"] > 80 or t["pe_end"] > 80:
        print("  ⚠️  P/E > 80 στην αρχή ή στο τέλος (EPS κοντά στο 0): το σύνολο παραμορφώνεται.")
    y = decompose_by_year(h.stock)
    if not y.empty:
        y = y.copy()
        y.loc[y["unstable"], ["from_eps_pct", "from_pe_pct"]] = np.nan
        print(y[["period", "price_pct", "from_eps_pct", "from_pe_pct"]]
              .rename(columns={"period": "Έτος", "price_pct": "Τιμή %",
                               "from_eps_pct": "από κέρδη %", "from_pe_pct": "από P/E %"})
              .round(1).to_string(index=False, na_rep="—"))
        if y["unstable"].any():
            print("  — = EPS κοντά στο 0 στην αρχή ή στο τέλος του έτους (μεταβολή > ×3):"
                  " ο διαχωρισμός δεν έχει νόημα.")
