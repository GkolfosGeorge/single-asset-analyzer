"""
valuation/snapshot.py — All valuation metrics of one stock in ONE table.

Each metric is compared with the stock's OWN history over the same window:
    "φθηνότερη από X% των ημερών" = on X% of past days it was more expensive.
    Example: P/E today 18, past P/Es mostly 20–30 → cheaper than 90% of days.

Direction matters: a LOW P/E is cheap, but a HIGH FCF yield is cheap.
Shareholder yield is shown for information only (more cash returned is not
"cheap" or "expensive" by itself).

Labels (descriptive, not a buy/sell signal):
    ≥ 75%  → φθηνή για τα δικά της δεδομένα
    ≤ 25%  → ακριβή για τα δικά της δεδομένα
    else   → κοντά στα συνηθισμένα της
"""

import numpy as np
import pandas as pd

# metric label, (history object attr, column), lower_is_cheaper, format
METRICS = [
    ("P/E (TTM)",         ("pe", "pe"),         True,  "{:.1f}"),
    ("EV/EBIT (TTM)",     ("ev", "ev_ebit"),    True,  "{:.1f}"),
    ("FCF yield",         ("fcf", "fcf_yield"), False, "{:.2%}"),
    ("P/S (TTM)",         ("po", "ps"),         True,  "{:.2f}"),
    ("Shareholder yield", ("po", "sh_yield"),   None,  "{:.2%}"),
]


def _series(obj, col) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    df = obj.stock if hasattr(obj, "stock") else obj.daily
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _label(pct: float) -> str:
    if pct >= 75:
        return "φθηνή για τα δικά της δεδομένα"
    if pct <= 25:
        return "ακριβή για τα δικά της δεδομένα"
    return "κοντά στα συνηθισμένα της"


def valuation_table(pe=None, fcf=None, ev=None, po=None) -> pd.DataFrame:
    """
    pe: PEHistory, fcf: FCFHistory, ev: EVHistory, po: PayoutHistory (any may be None).
    Returns one row per metric: today, own median, % of days it was cheaper/more
    expensive, label. A metric with no value today shows "—" and the reason
    is in the module's own summary (e.g. losses, missing tag, stale data).
    """
    objs = {"pe": pe, "fcf": fcf, "ev": ev, "po": po}
    rows = []
    for name, (key, col), lower_cheap, fmt in METRICS:
        s = _series(objs[key], col)
        hist = s.dropna()
        today = s.iloc[-1] if len(s) else np.nan
        if pd.isna(today) or hist.empty:
            rows.append({"Δείκτης": name, "Σήμερα": "—", "Διάμεσος της ίδιας": "—",
                         "Φθηνότερη από % ημερών": "—", "Ανάγνωση": "μη διαθέσιμο"})
            continue
        med = hist.median()
        if lower_cheap is None:
            pct_txt, reading = "", "πληροφοριακό"
        else:
            # share of past days when the stock was MORE expensive than today
            pct = ((hist > today) if lower_cheap else (hist < today)).mean() * 100
            pct_txt, reading = f"{pct:.0f}%", _label(pct)
        rows.append({"Δείκτης": name, "Σήμερα": fmt.format(today),
                     "Διάμεσος της ίδιας": fmt.format(med),
                     "Φθηνότερη από % ημερών": pct_txt, "Ανάγνωση": reading})
    return pd.DataFrame(rows)


def print_snapshot(ticker: str, pe=None, fcf=None, ev=None, po=None) -> pd.DataFrame:
    t = valuation_table(pe, fcf, ev, po)
    start = None
    for o in (pe, fcf):
        df = None if o is None else (o.stock if hasattr(o, "stock") else o.daily)
        if df is not None and not df.empty:
            start = df.index[0]
            break
    since = f" (ιστορικό από {start:%m/%Y})" if start is not None else ""
    print(f"══ {ticker} — αποτίμηση με μια ματιά{since} ══")
    print(t.to_string(index=False))
    print("\nΣύγκριση μόνο με το δικό της παρελθόν, όχι με τον κλάδο. Περιγραφή, όχι σήμα αγοράς/πώλησης.")
    return t
