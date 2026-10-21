"""
valuation/quick.py — a FAST, stock-only valuation card for automated
messages (the mean reversion daily mail).

Differences from the notebook:
  - NO peers: the sector line needs prices for ~100 companies (slow) and is
    not trusted before Βήμα 3 anyway. Every comparison is with the stock's
    OWN 10-year history.
  - EVERY part fails on its own. A missing tag, a bank without EBIT, an ETF
    that isn't in the SEC database, Yahoo hiccups: that part is left out of
    the card, the rest still shows. quick_snapshot() never raises.

    snap = quick_snapshot("HD", engine)
    print("\\n".join(format_card(snap)))     # 2–3 lines under the technical block
    print("\\n".join(format_line(snap)))     # 1 line, for the watchlist
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import ev as E
from . import pe as P
from . import yields as Y
from .ttm import load_prices

CHEAP_PCT, EXPENSIVE_PCT = 75, 25          # same thresholds as snapshot.py
EARNINGS_SOON_DAYS = 14                    # warn if results are this close
RECENT_REPORT_DAYS = 7                     # "results came out N days ago"
CAPEX_JUMP_X = 1.5                         # capex TTM ×1.5 in a year → mention it

# Piotroski criteria in the same order/groups as quality_scores/piotroski.py.
# (kept here as plain names so this module imports without quality_scores)
F_GROUPS = [
    ("Κερδοφορία", ["positive_roa", "positive_cfo", "roa_improved", "cfo_exceeds_net_income"]),
    ("Χρέος/ρευστότητα", ["leverage_decreased", "current_ratio_improved", "no_dilution"]),
    ("Αποδοτικότητα", ["gross_margin_improved", "asset_turnover_improved"]),
]
F_PASS = {   # what a ✓ means (legend)
    "positive_roa": "κερδοφόρα", "positive_cfo": "θετικές ταμειακές ροές",
    "roa_improved": "ROA ↑", "cfo_exceeds_net_income": "ταμείο > λογιστικά κέρδη",
    "leverage_decreased": "λιγότερο χρέος", "current_ratio_improved": "ρευστότητα ↑",
    "no_dilution": "όχι νέες μετοχές",
    "gross_margin_improved": "μεικτό περιθώριο ↑", "asset_turnover_improved": "αποδοτικότητα assets ↑",
}
F_FAIL = {   # what a ✗ means, in the card
    "positive_roa": "ζημιές", "positive_cfo": "αρνητικές ταμειακές ροές",
    "roa_improved": "ROA ↓", "cfo_exceeds_net_income": "κέρδη > ταμείο",
    "leverage_decreased": "χρέος ↑", "current_ratio_improved": "ρευστότητα ↓",
    "no_dilution": "νέες μετοχές",
    "gross_margin_improved": "μεικτό περιθώριο ↓", "asset_turnover_improved": "αποδοτικότητα assets ↓",
}


@dataclass
class Metric:
    value: float
    pct: float          # % of past days that were MORE expensive than today

    @property
    def label(self) -> str:
        if self.pct >= CHEAP_PCT:
            return "cheap"
        if self.pct <= EXPENSIVE_PCT:
            return "expensive"
        return "normal"


@dataclass
class QuickSnapshot:
    ticker: str
    pe: Metric | None = None
    ev_ebit: Metric | None = None
    fcf_yield: Metric | None = None
    eps_growth_1y: float | None = None       # +0.06 = +6%
    eps_negative: bool = False
    fscore: tuple[int, int] | None = None    # (score, max_score)
    fscore_detail: dict[str, bool | None] = field(default_factory=dict)   # criterion → ✓/✗/None
    div_yield: float | None = None
    capex_x_1y: float | None = None          # capex TTM now / a year ago
    days_since_report: int | None = None
    days_to_earnings: int | None = None
    financial: bool = False                   # bank/insurer: EV/EBIT, FCF don't apply
    errors: list[str] = field(default_factory=list)

    @property
    def has_fundamentals(self) -> bool:
        return any(x is not None for x in (self.pe, self.ev_ebit, self.fcf_yield, self.fscore))


# ═════════════════════════════════════════════════════════════════════════
# 1. BUILD
# ═════════════════════════════════════════════════════════════════════════

def _own_metric(s: pd.Series, lower_is_cheaper: bool) -> Metric | None:
    hist = pd.to_numeric(s, errors="coerce").dropna()
    if hist.empty or pd.isna(s.iloc[-1]):
        return None
    today = float(s.iloc[-1])
    pct = ((hist > today) if lower_is_cheaper else (hist < today)).mean() * 100
    return Metric(today, float(pct))


def _value_year_ago(s: pd.Series, end: pd.Timestamp) -> float | None:
    s = s.dropna()
    s = s[s.index <= end - pd.Timedelta(days=365)]
    return float(s.iloc[-1]) if len(s) else None


def _next_earnings_days(ticker: str, today: pd.Timestamp) -> int | None:
    """Next earnings date from Yahoo, if it gives one in the future."""
    import yfinance as yf
    cal = yf.Ticker(ticker).calendar
    dates = []
    if isinstance(cal, dict):
        v = cal.get("Earnings Date")
        dates = v if isinstance(v, (list, tuple)) else [v]
    elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
        dates = list(cal.loc["Earnings Date"].values)
    future = [pd.Timestamp(d) for d in dates if d is not None and pd.Timestamp(d) >= today.normalize()]
    return (min(future) - today.normalize()).days if future else None


def quick_snapshot(ticker: str, engine, years: int = 10, as_of=None,
                   with_earnings_date: bool = True) -> QuickSnapshot:
    snap = QuickSnapshot(ticker)
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=years)

    def part(name, fn):
        try:
            fn()
        except Exception as e:                      # one part never breaks the card
            snap.errors.append(f"{name}: {type(e).__name__}: {str(e)[:80]}")

    # ── P/E and EPS growth (stock only, split-adjusted, point-in-time) ──
    def _pe():
        eps = P.load_eps([ticker], engine, as_of=as_of)
        if eps.empty:
            return
        snap.days_since_report = int((end - eps["filed_date"].max()).days)
        dl_start = (start - timedelta(days=P.SPLIT_LOOKBACK_DAYS)).date()
        close, splits = load_prices([ticker], dl_start, (end + timedelta(days=1)).date())
        if ticker not in close.columns:
            return
        sp = splits[ticker] if ticker in splits.columns else None
        d = P.daily_pe(close[ticker], P.build_ttm(eps, sp))
        d = d[(d.index >= start) & (d.index <= end)]
        if d.empty:
            return
        snap.pe = _own_metric(d["pe"], lower_is_cheaper=True)
        now, ago = d["eps_ttm"].dropna(), _value_year_ago(d["eps_ttm"], end)
        if len(now):
            snap.eps_negative = bool(now.iloc[-1] <= 0)
            if ago is not None and ago > 0 and now.iloc[-1] > 0:
                snap.eps_growth_1y = float(now.iloc[-1] / ago - 1)
    part("P/E", _pe)

    # ── FCF yield, EV/EBIT, dividend, capex trend (one price download) ──
    def _fcf_ev():
        fcf = Y.fcf_history(ticker, engine, years=years, as_of=as_of)
        if fcf.daily.empty:
            return
        d = fcf.daily
        snap.fcf_yield = _own_metric(d["fcf_yield"], lower_is_cheaper=False)
        dps = d["div_ps_365d"].iloc[-1]
        if pd.notna(dps) and d["close"].iloc[-1] > 0:
            snap.div_yield = float(dps / d["close"].iloc[-1])
        cap_now, cap_ago = d["capex_ttm"].dropna(), _value_year_ago(d["capex_ttm"], end)
        if len(cap_now) and cap_ago and cap_ago > 0:
            snap.capex_x_1y = float(cap_now.iloc[-1] / cap_ago)
        ev = E.ev_history(ticker, engine, as_of=as_of, base=fcf)
        if not ev.daily.empty:
            snap.ev_ebit = _own_metric(ev.daily["ev_ebit"], lower_is_cheaper=True)
        # Banks/insurers have NEITHER operating income NOR capex. "No EBIT" alone
        # is not enough: oil majors (CVX, XOM) don't report an operating-income
        # line either, but they do have capex.
        snap.financial = (ev.status.startswith("χωρίς EBIT")
                          and fcf.status.startswith("χωρίς capex"))
    part("FCF/EV", _fcf_ev)

    # ── Piotroski ──
    def _fscore():
        from quality_scores.piotroski import piotroski_f_score
        r = piotroski_f_score(ticker, engine, as_of=as_of)
        if r is not None and r.max_score > 0:
            snap.fscore = (int(r.score), int(r.max_score))
            snap.fscore_detail = dict(r.breakdown)
    part("F-Score", _fscore)

    if with_earnings_date and as_of is None:
        part("earnings date", lambda: setattr(snap, "days_to_earnings", _next_earnings_days(ticker, end)))
    return snap


# ═════════════════════════════════════════════════════════════════════════
# 2. FORMAT (plain text, for e-mail)
# ═════════════════════════════════════════════════════════════════════════

_WORD = {"cheap": "φθηνή", "expensive": "ακριβή", "normal": "κανονική"}


def verdict(snap: QuickSnapshot) -> tuple[str, str]:
    """(emoji, phrase) from P/E, EV/EBIT, FCF yield — whichever exist."""
    ms = [m for m in (snap.pe, snap.ev_ebit, snap.fcf_yield) if m is not None]
    if not ms:
        return "⚪", "Αποτίμηση μη διαθέσιμη"
    # Disagreement is the most informative case (e.g. MSFT 2026: P/E and
    # EV/EBIT normal, FCF yield the most expensive in 10 years) — so
    # "normal" only when ALL agree, and a clear call only when the majority
    # agrees and nobody points the other way. Everything else is "mixed".
    labels = [m.label for m in ms]
    n_c, n_e, half = labels.count("cheap"), labels.count("expensive"), math.ceil(len(ms) / 2)
    if n_c >= half and n_e == 0:
        return "🟢", "Φθηνή για τα δικά της δεδομένα"
    if n_e >= half and n_c == 0:
        return "🔴", "Ακριβή για τα δικά της δεδομένα"
    if n_c == 0 and n_e == 0:
        return "🟡", "Κοντά στα συνηθισμένα της"
    return "🟡", "Μικτή εικόνα"


def _metric_txt(name: str, m: Metric | None, fmt: str) -> str | None:
    if m is None:
        return None
    txt = f"{name} {format(m.value, fmt)}"
    if m.pct <= 1:
        return txt + " (ακριβότερη 10ετίας)"
    if m.pct >= 99:
        return txt + " (φθηνότερη 10ετίας)"
    return txt if m.label == "normal" else f"{txt} ({_WORD[m.label]})"


def _when(days: int) -> str:
    return {0: "σήμερα", 1: "αύριο"}.get(days, f"σε {days} μέρες")


def _ago(days: int) -> str:
    return {0: "σήμερα", 1: "χθες"}.get(days, f"πριν {days} μέρες")


def format_card(snap: QuickSnapshot, indent: str = "  ") -> list[str]:
    """2–4 lines to put under a ticker's technical block. [] never: always says something."""
    if not snap.has_fundamentals:
        if snap.errors:      # something broke (DB, Yahoo) — don't blame the ticker
            return [f"{indent}📊 Θεμελιώδη: η ανάλυση απέτυχε σήμερα ({snap.errors[0][:60]})"]
        return [f"{indent}📊 Θεμελιώδη: μη διαθέσιμα (ETF, ξένη εταιρεία ή εκτός βάσης)"]

    lines = []
    emoji, phrase = verdict(snap)
    parts = [_metric_txt("P/E", snap.pe, ".0f"),
             _metric_txt("EV/EBIT", snap.ev_ebit, ".0f"),
             _metric_txt("FCF", snap.fcf_yield, ".1%")]
    lines.append(f"{indent}💰 {emoji} {phrase}   " + " · ".join(p for p in parts if p))

    q = []
    if snap.eps_negative:
        q.append("Ζημιές (EPS ≤ 0)")
    elif snap.eps_growth_1y is not None:
        q.append(f"Κέρδη {snap.eps_growth_1y:+.0%} σε 1 χρόνο")
    if snap.div_yield:
        q.append(f"μέρισμα {snap.div_yield:.1%}")
    if q:
        lines.append(f"{indent}📈 " + " · ".join(q))
    lines += _health_lines(snap, indent)

    if snap.days_to_earnings is not None and snap.days_to_earnings <= EARNINGS_SOON_DAYS:
        lines.append(f"{indent}⚠️ Αποτελέσματα {_when(snap.days_to_earnings)}: "
                     f"ένα gap μπορεί να περάσει το stop")
    if snap.days_since_report is not None and snap.days_since_report <= RECENT_REPORT_DAYS:
        lines.append(f"{indent}⚠️ Αποτελέσματα {_ago(snap.days_since_report)}: "
                     f"η πτώση μπορεί να έχει θεμελιώδη αιτία")
    if (snap.capex_x_1y and snap.capex_x_1y >= CAPEX_JUMP_X and snap.fcf_yield
            and snap.fcf_yield.label == "expensive"):
        lines.append(f"{indent}ℹ️ Τα μετρητά πάνε σε επενδύσεις (capex ×{snap.capex_x_1y:.1f} σε 1 χρόνο)")
    if snap.financial:
        lines.append(f"{indent}ℹ️ Χρηματοοικονομική εταιρεία: EV/EBIT και FCF δεν έχουν νόημα")
    return lines


def _health_lines(snap: QuickSnapshot, indent: str) -> list[str]:
    """
    🩺 Υγεία 6/9   Κερδοφορία ✓✓✓✓ · Χρέος/ρευστότητα ✓✗✓ · Αποδοτικότητα ✗✗
       ✗ ρευστότητα ↓ · μεικτό περιθώριο ↓ · αποδοτικότητα assets ↓
    – = not evaluated (missing data), so 6/8 means 8 criteria could be judged.
    """
    if not snap.fscore:
        return []
    sym = {True: "✓", False: "✗", None: "–"}
    det = snap.fscore_detail
    groups = " · ".join(f"{name} " + "".join(sym[det.get(k)] for k in keys) for name, keys in F_GROUPS)
    out = [f"{indent}🩺 Υγεία {snap.fscore[0]}/{snap.fscore[1]}   {groups}"]
    fails = [F_FAIL[k] for _, keys in F_GROUPS for k in keys if det.get(k) is False]
    if fails:
        out.append(f"{indent}   ✗ " + " · ".join(fails))
    return out


def format_line(snap: QuickSnapshot, indent: str = "  ") -> list[str]:
    """One line for the watchlist; [] if there are no fundamentals (ETF, foreign filer)."""
    if not snap.has_fundamentals:
        return []
    emoji, phrase = verdict(snap)
    bits = [f"{emoji} {phrase.split(' για')[0].lower()}"]
    if snap.fscore:
        bits.append(f"Υγεία {snap.fscore[0]}/{snap.fscore[1]}")
    if snap.days_to_earnings is not None and snap.days_to_earnings <= EARNINGS_SOON_DAYS:
        bits.append(f"⚠️ αποτελέσματα {_when(snap.days_to_earnings)}")
    return [f"{indent}📊 " + " · ".join(bits)]


def legend() -> list[str]:
    out = [
        "Θεμελιώδη: 🟢 φθηνή / 🟡 κανονική ή μικτή / 🔴 ακριβή = σε σχέση με το ΔΙΚΟ της",
        "10ετές ιστορικό (P/E, EV/EBIT, FCF yield), όχι με τον κλάδο.",
        "🩺 Υγεία = Piotroski F-Score, φέτος vs πέρσι (7–9 υγιής, 4–6 μέτρια, 0–3 αδύναμη).",
        "   ✓ πέρασε · ✗ κόπηκε · – δεν αξιολογήθηκε (λείπουν δεδομένα). Σειρά ανά ομάδα:",
    ]
    for name, keys in F_GROUPS:
        out.append(f"   {name}: " + " · ".join(f"{i}) {F_PASS[k]}" for i, k in enumerate(keys, 1)))
    out.append("Πληροφορία, όχι σήμα.")
    return out
