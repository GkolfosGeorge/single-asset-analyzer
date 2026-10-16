"""
valuation/ttm.py — Point-in-time TTM ("trailing twelve months") for ANY
flow metric: EPS, revenue, EBIT, operating cash flow, capex, dividends...

A "flow" metric is measured OVER a period (revenue of Q2 = sales from
April to June). The opposite is an "instant" metric, measured ON a day
(debt, cash, shares outstanding) — those are NOT summed and don't belong here.

Same three traps as pe.py, now in one place:

1. LOOKAHEAD. A value enters the TTM on `filed_date + 1`, never on period_end.

2. SPLITS — only for PER-SHARE metrics (EPS, dividends per share).
   A split changes the number of shares, not the dollars:
     AAPL 4:1 split (Aug 2020): FY2019 EPS 11.89 → 2.97 in today's shares,
     but FY2019 revenue 260 bn$ stays 260 bn$.
   So `per_share=True` divides by later splits; dollar metrics are untouched.

3. MISSING QUARTERS. SEC filings don't always contain the discrete quarter:
   - Income statement (EPS, revenue): 10-Q has the discrete quarter, but
     there is no Q4 10-Q → Q4 = FY − (Q1+Q2+Q3).
   - Cash-flow statement (CFO, capex): 10-Qs are usually CUMULATIVE
     ("year to date"): Q1 = 3 months, then H1 = 6 months, 9M = 9 months.
     Example, CFO: Q1 = 100, H1 = 230, 9M = 390, FY = 520
       → Q2 = 230 − 100 = 130, Q3 = 390 − 230 = 160, Q4 = 520 − 390 = 130.
   Both are the SAME rule: a cumulative figure minus the quarters already
   known inside it gives the one quarter that is missing at its end.
   We apply it to H1, 9M and FY rows in date order, so a quarter derived
   from H1 helps derive the next one from 9M, and so on.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

QUARTER_DAYS = (80, 100)      # a fiscal quarter is 13 weeks (~91 days)
H1_DAYS = (170, 200)          # 6-month year-to-date
NINE_M_DAYS = (260, 290)      # 9-month year-to-date
YEAR_DAYS = (350, 380)        # 52/53-week or calendar year
DAYS_PER_QUARTER = 91.3
MAX_TTM_AGE_DAYS = 270        # newest TTM older than this → company stopped filing

COLS = ["period_start", "period_end", "filed_date", "value"]


# ═════════════════════════════════════════════════════════════════════════
# 1. DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════

def load_metric(metric: str, tickers: list[str], engine, as_of=None) -> pd.DataFrame:
    """One metric's rows (original 10-Q/10-K only) for many tickers, ONE query."""
    where = ["m.name = :metric", "f.ticker = ANY(:tickers)", "ft.name IN ('10-Q', '10-K')"]
    params = {"metric": metric, "tickers": list(tickers)}
    if as_of:
        where.append("f.filed_date <= CAST(:as_of AS DATE)")
        params["as_of"] = str(as_of)

    sql = text(f"""
        SELECT f.ticker, f.value, f.period_start, f.period_end, f.filed_date,
               ft.name AS form_type
        FROM fundamentals f
        JOIN metric_lookup m     ON m.id  = f.metric_id
        JOIN form_type_lookup ft ON ft.id = f.form_type_id
        WHERE {' AND '.join(where)}
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    df = pd.DataFrame(rows, columns=["ticker", "value", "period_start",
                                     "period_end", "filed_date", "form_type"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("period_start", "period_end", "filed_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def load_prices(tickers: list[str], start, end=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (close, splits), both DataFrames indexed by date, one column per
    ticker. close = split-adjusted, NOT dividend-adjusted. splits = ratio on
    the split day (4.0 for 4:1, 0.1 for a 1:10 reverse split), 0 otherwise.
    """
    import yfinance as yf

    raw = yf.download(list(tickers), start=str(start), end=None if end is None else str(end),
                      auto_adjust=False, actions=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _field(name):
        if isinstance(raw.columns, pd.MultiIndex):
            if name not in raw.columns.get_level_values(0):
                return pd.DataFrame(0.0, index=raw.index, columns=list(tickers))
            out = raw[name]
        else:  # very old yfinance: single ticker → flat columns
            out = raw[[name]].rename(columns={name: tickers[0]})
        return out

    close, splits = _field("Close"), _field("Stock Splits").fillna(0.0)
    for df in (close, splits):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return close, splits


# ═════════════════════════════════════════════════════════════════════════
# 2. PURE LOGIC (no DB, no network — testable with synthetic data)
# ═════════════════════════════════════════════════════════════════════════

def splits_after(splits: pd.Series, dates: pd.Series) -> np.ndarray:
    """For each date d: product of split ratios on days STRICTLY after d."""
    s = splits[splits > 0]
    if s.empty:
        return np.ones(len(dates))
    return np.array([s[s.index > d].prod() for d in dates], dtype=float)


def _fmt(v: float) -> str:
    """2.37 for per-share numbers, 1,234M for dollar amounts."""
    return f"{v / 1e6:,.0f}M" if abs(v) >= 1e6 else f"{v:.2f}"


def _build_quarters(rows: pd.DataFrame, splits: pd.Series | None = None,
                    per_share: bool = False):
    """
    Returns (q, fy, quarters, log):
      q        — discrete 10-Q quarters as reported
      fy       — full-year 10-K rows
      quarters — q + derived quarters (column `source`: "10-Q", "Q2←H1",
                 "Q3←9M", "Q4←FY"), sorted by period_end
      log      — one line per cumulative row (H1/9M/FY): derived or why not
    """
    df = rows.dropna(subset=COLS).copy()
    empty = pd.DataFrame(columns=COLS)
    if df.empty:
        return empty, empty, empty.assign(source=pd.Series(dtype=str)), []

    # (a) per-share only: adjust EVERY row with the splits after ITS OWN
    #     filing date, before any adding/subtracting.
    if per_share and splits is not None and len(splits):
        df["value"] = df["value"] / splits_after(splits, df["filed_date"])

    dur = (df["period_end"] - df["period_start"]).dt.days
    is_q, is_k = df["form_type"] == "10-Q", df["form_type"] == "10-K"
    q = df[is_q & dur.between(*QUARTER_DAYS)]
    ytd = df[is_q & (dur.between(*H1_DAYS) | dur.between(*NINE_M_DAYS))]
    fy = df[is_k & dur.between(*YEAR_DAYS)]
    # Same period reported under two fiscal labels → keep the first filing.
    q = q.sort_values("filed_date").drop_duplicates("period_end")
    ytd = ytd.sort_values("filed_date").drop_duplicates("period_end")
    fy = fy.sort_values("filed_date").drop_duplicates("period_end")

    # (b) cumulative − known quarters inside it = the missing last quarter.
    #     Processed in period_end order so H1 → Q2 is known before 9M → Q3.
    known = q[COLS].assign(source="10-Q")
    cum = pd.concat([ytd.assign(kind="YTD"), fy.assign(kind="FY")], ignore_index=True)
    cum = cum.sort_values(["period_end", "filed_date"], kind="stable")
    log = []
    for _, c in cum.iterrows():
        n_before = int(round((c["period_end"] - c["period_start"]).days / DAYS_PER_QUARTER)) - 1
        label = "FY" if c["kind"] == "FY" else ("H1" if n_before == 1 else "9M")
        q_name = f"Q{n_before + 1}"
        tag = f"{label} {c['period_start']:%Y-%m-%d}→{c['period_end']:%Y-%m-%d}"
        clash = known["period_end"] == c["period_end"]
        if clash.any() and c["kind"] == "YTD":
            continue                         # discrete 10-Q exists: nothing to derive
        # A 3-month 10-Q ending ON the fiscal year end should not exist (there is
        # no Q4 10-Q). When it does, it is an anomaly — e.g. KDP 2018: after the
        # Keurig merger a 10-Q carried the accounting acquirer's quarter, with
        # EPS on a completely different share base. FY − (Q1+Q2+Q3) comes from
        # the company's own chain of filings, so it wins (see below).
        inside = known[(known["period_start"] >= c["period_start"] - timedelta(days=10)) &
                       (known["period_end"] <= c["period_end"] - timedelta(days=60))]
        if len(inside) != n_before:
            log.append(f"{tag}: ✗ βρέθηκαν {len(inside)} τρίμηνα μέσα (θέλουμε {n_before})")
            continue
        start = inside["period_end"].max() + timedelta(days=1)
        days = (c["period_end"] - start).days
        if not QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
            log.append(f"{tag}: ✗ {q_name} θα είχε {days} μέρες")
            continue
        val = c["value"] - inside["value"].sum()
        log.append(f"{tag}: ✓ {q_name} = {_fmt(c['value'])} − {_fmt(inside['value'].sum())}"
                   f" = {_fmt(val)}")
        if clash.any():
            old_val = known.loc[clash, "value"].iloc[0]
            log.append(f"{tag}: ⚠️ υπήρχε και 10-Q {q_name} = {_fmt(old_val)} στο τέλος της χρήσης"
                       f" → κρατάμε το {_fmt(val)} από το FY")
            known = known[~clash]
        new = pd.DataFrame([{"period_start": start, "period_end": c["period_end"],
                             "filed_date": c["filed_date"], "value": val,
                             "source": f"{q_name}←{label}"}])
        known = pd.concat([known, new], ignore_index=True)

    quarters = (known.sort_values("period_end", kind="stable")
                .drop_duplicates("period_end").reset_index(drop=True))
    return q, fy, quarters, log


def build_ttm(rows: pd.DataFrame, splits: pd.Series | None = None,
              per_share: bool = False, value_col: str = "ttm") -> pd.DataFrame:
    """
    One ticker's rows of ONE metric → TTM series, columns:
        period_end  — end of the latest quarter in the TTM window
        available   — first day the market could use it (last filing + 1 day)
        <value_col> — sum of the last 4 quarters (split-adjusted if per_share)
    """
    cols = ["period_end", "available", value_col]
    q, fy, quarters, _ = _build_quarters(rows, splits, per_share)
    if fy.empty and quarters.empty:
        return pd.DataFrame(columns=cols)

    # (c) TTM = last 4 quarters, only if they are truly consecutive
    ttm = []
    for i in range(3, len(quarters)):
        w = quarters.iloc[i - 3:i + 1]
        span = (w["period_end"].iloc[-1] - w["period_start"].iloc[0]).days
        gaps = w["period_end"].diff().dt.days.iloc[1:]
        if YEAR_DAYS[0] <= span <= YEAR_DAYS[1] and gaps.between(*QUARTER_DAYS).all():
            ttm.append({"period_end": w["period_end"].iloc[-1],
                        "available": w["filed_date"].max(),
                        value_col: w["value"].sum()})
    # A full-year 10-K IS a TTM by itself (covers companies with no 10-Q data).
    fy_ttm = fy.rename(columns={"filed_date": "available", "value": value_col})[cols]
    out = pd.concat([fy_ttm, pd.DataFrame(ttm, columns=cols)], ignore_index=True)
    out = out.drop_duplicates("period_end", keep="first")      # FY wins on ties: exact, no math
    out["available"] = out["available"] + timedelta(days=1)    # filed after the close → usable next day

    # (d) on any day, use the NEWEST period known by then (a late filing for
    #     an older period must not replace a newer one)
    out = out.sort_values(["available", "period_end"]).reset_index(drop=True)
    newest_before = out["period_end"].cummax().shift()
    out = out[newest_before.isna() | (out["period_end"] > newest_before)]
    return out.reset_index(drop=True)


def newest_only(out: pd.DataFrame) -> pd.DataFrame:
    """
    Rows with `available` and `period_end`: keep a row only if its period is
    newer than everything known before it (a late filing for an OLD period
    must not replace a newer one on the timeline).
    """
    out = out.sort_values(["available", "period_end"]).reset_index(drop=True)
    newest_before = out["period_end"].cummax().shift()
    return out[newest_before.isna() | (out["period_end"] > newest_before)].reset_index(drop=True)


def build_latest(rows: pd.DataFrame, splits: pd.Series | None = None,
                 shares: bool = False, value_col: str = "value") -> pd.DataFrame:
    """
    "Latest known value" series for metrics that must NOT be summed:
    balance-sheet items (debt, cash — `instant`, no period_start) and
    average share counts (one quarter or one year, never 6M/9M YTD).

    shares=True: share counts go the OPPOSITE way of EPS on a split —
    they are MULTIPLIED by later splits (AAPL FY2019: 4.65 bn × 4 = 18.6 bn
    in today's shares), so that  split-adjusted price × shares = market cap.

    Columns: period_end, available (filed + 1 day), <value_col>.
    """
    cols = ["period_end", "available", value_col]
    df = rows.dropna(subset=["value", "period_end", "filed_date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    if shares and splits is not None and len(splits):
        df["value"] = df["value"] * splits_after(splits, df["filed_date"])
    dur = (df["period_end"] - df["period_start"]).dt.days
    keep = (df["period_start"].isna()
            | ((df["form_type"] == "10-Q") & dur.between(*QUARTER_DAYS))
            | ((df["form_type"] == "10-K") & dur.between(*YEAR_DAYS)))
    df = df[keep].sort_values("filed_date").drop_duplicates("period_end")
    out = df.rename(columns={"filed_date": "available", "value": value_col})[cols].copy()
    out["available"] = out["available"] + timedelta(days=1)
    return newest_only(out)


def asof_daily(dates, frame: pd.DataFrame, col: str,
               max_age_days: int = MAX_TTM_AGE_DAYS) -> pd.Series:
    """
    For each day: the value of `col` known that day (available ≤ day), or
    NaN if its period ended more than `max_age_days` ago (company stopped
    filing). Same rule as daily_pe, for any series.
    """
    idx = pd.DatetimeIndex(dates)
    if frame.empty:
        return pd.Series(np.nan, index=idx)
    left = pd.DataFrame({"date": idx.as_unit("ns")})
    right = frame.sort_values("available")[["available", "period_end", col]].copy()
    for c in ("available", "period_end"):
        right[c] = pd.to_datetime(right[c]).dt.as_unit("ns")
    m = pd.merge_asof(left, right, left_on="date", right_on="available", direction="backward")
    stale = (m["date"] - m["period_end"]).dt.days > max_age_days
    return pd.Series(m[col].where(~stale).to_numpy(dtype=float), index=idx)


def debug_ttm(ticker: str, engine, metric: str, per_share: bool = False,
              since: str = "2024-01-01", max_age_days: int = MAX_TTM_AGE_DAYS) -> None:
    """
    Show, step by step, how one ticker's TTM of `metric` is built: raw DB
    rows, derived quarters, final TTM. Per-share metrics are shown WITH
    split adjustment (today's share units), like the charts use them.
    """
    rows = load_metric(metric, [ticker], engine)
    if rows.empty:
        print(f"{ticker}: καμία γραμμή {metric} στη βάση.")
        return
    rows = rows.sort_values("period_end")
    rows["days"] = (rows["period_end"] - rows["period_start"]).dt.days
    recent = rows[rows["period_end"] >= since]
    note = " (όπως κατατέθηκαν, χωρίς split)" if per_share else ""
    print(f"── {ticker}: γραμμές {metric} στη βάση από {since}{note} ──")
    print(recent[["form_type", "period_start", "period_end", "days", "filed_date", "value"]]
          .to_string(index=False) if not recent.empty else "  (καμία)")
    print(f"  (σύνολο στη βάση: {len(rows)} γραμμές, τελευταίο filed {rows['filed_date'].max():%Y-%m-%d})")

    sp = None
    if per_share:
        try:
            _, splits = load_prices([ticker], rows["filed_date"].min().date())
            if ticker in splits.columns:
                sp = splits[ticker]
                done = sp[sp > 0]
                if len(done):
                    print("  splits: " + ", ".join(f"{d:%Y-%m-%d} ×{r:g}" for d, r in done.items())
                          + "  → οι τιμές παρακάτω είναι σε ΣΗΜΕΡΙΝΕΣ μετοχές")
        except Exception as e:
            print(f"  ⚠️ splits δεν κατέβηκαν ({e}) — οι τιμές παρακάτω ΔΕΝ είναι προσαρμοσμένες")

    _, _, quarters, log = _build_quarters(rows, sp, per_share)
    print("\n── Παραγωγή τριμήνων που λείπουν ──")
    for line in log[-6:]:
        print("  " + line)
    ttm = build_ttm(rows, sp, per_share)
    print("\n── Τελευταία TTM ──")
    print(ttm.tail(4).to_string(index=False) if not ttm.empty else "  (κανένα)")
    if not ttm.empty:
        age = (pd.Timestamp(date.today()) - ttm["period_end"].iloc[-1]).days
        print(f"  Ηλικία τελευταίου TTM: {age} μέρες (όριο {max_age_days})")
