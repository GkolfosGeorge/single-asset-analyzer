"""
piotroski.py — Piotroski F-Score (Piotroski, 2000, "Value Investing: The
Use of Historical Financial Statement Information to Separate Winners
from Losers", Journal of Accounting Research 38:1-41).

Nine binary criteria across three groups: profitability (4),
leverage/liquidity (3), operating efficiency (2).

Design rules:
  1. Missing input -> criterion is None, NOT False. `max_score` counts
     only criteria that could be evaluated.
  2. Scaling follows the paper: ROA and asset turnover use BEGINNING-of-
     year total assets; leverage uses AVERAGE total assets. This needs
     three consecutive annual snapshots (t, t-1, t-2). If t-2 is missing,
     the criteria that need it return None rather than a wrong answer.
  3. A year-over-year comparison always uses the SAME measure in both
     years (see _consistent_pair). Never mix e.g. period-end shares in one
     year with weighted-average shares in the other.
  4. Share-count jumps that look like a stock split (x2, x4, 1/10, ...)
     make `no_dilution` None: our point-in-time data keeps the prior year
     as ORIGINALLY reported (pre-split), so a split would otherwise read
     as massive dilution.

Every criterion also carries a CriterionDetail with the underlying
numbers, so reports/charts can show HOW MUCH, not just pass/fail.

Usage:
    from db import get_engine
    from quality_scores.piotroski import piotroski_f_score, print_report

    engine = get_engine()
    result = piotroski_f_score("AAPL", engine)
    print_report(result)
"""

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .line_items import AnnualSnapshot, get_annual_snapshots, is_consecutive

GROUPS = {
    "profitability": ["positive_roa", "positive_cfo", "roa_improved", "cfo_exceeds_net_income"],
    "leverage_liquidity": ["leverage_decreased", "current_ratio_improved", "no_dilution"],
    "efficiency": ["gross_margin_improved", "asset_turnover_improved"],
}
CRITERIA = [c for group in GROUPS.values() for c in group]

LABELS = {
    "el": {
        "positive_roa": "Κερδοφόρα (ROA > 0)",
        "positive_cfo": "Θετικές ταμειακές ροές",
        "roa_improved": "ROA βελτιώθηκε",
        "cfo_exceeds_net_income": "Ταμείο > λογιστικά κέρδη",
        "leverage_decreased": "Λιγότερο χρέος",
        "current_ratio_improved": "Καλύτερη ρευστότητα",
        "no_dilution": "Όχι νέες μετοχές",
        "gross_margin_improved": "Μεικτό περιθώριο ↑",
        "asset_turnover_improved": "Αποδοτικότητα assets ↑",
        "profitability": "Κερδοφορία",
        "leverage_liquidity": "Χρέος & ρευστότητα",
        "efficiency": "Αποδοτικότητα",
    },
    "en": {
        "positive_roa": "Profitable (ROA > 0)",
        "positive_cfo": "Positive operating cash flow",
        "roa_improved": "ROA improved",
        "cfo_exceeds_net_income": "Cash flow > net income",
        "leverage_decreased": "Leverage decreased",
        "current_ratio_improved": "Current ratio improved",
        "no_dilution": "No new shares issued",
        "gross_margin_improved": "Gross margin improved",
        "asset_turnover_improved": "Asset turnover improved",
        "profitability": "Profitability",
        "leverage_liquidity": "Leverage & liquidity",
        "efficiency": "Efficiency",
    },
}

NOTES = {
    "el": {
        "no_t2": "λείπει το έτος t-2 (χρειάζεται για τα assets αρχής χρονιάς)",
        "roa_sign_only": "χωρίς assets αρχής χρονιάς — κρίθηκε από το πρόσημο των κερδών",
        "no_debt_pair": "δεν υπάρχει ίδιο μέτρο χρέους και στις δύο χρονιές",
        "no_shares_pair": "δεν υπάρχει ίδιο μέτρο μετοχών και στις δύο χρονιές",
        "no_gp_pair": "δεν υπάρχει μεικτό κέρδος και στις δύο χρονιές",
        "split": "πιθανό split {ratio} — η σύγκριση δεν έχει νόημα",
        "debt_total": "χρέος: μακροπρόθεσμο + τρέχον μέρος",
        "debt_noncurrent": "χρέος: μόνο μακροπρόθεσμο (ίδιο μέτρο και στα δύο έτη)",
        "debt_ltd": "χρέος: LongTermDebt (μαζί με το τρέχον μέρος)",
        "debt_lease_total": "χρέος: μαζί με finance leases και τρέχον μέρος",
        "debt_lease_sum": "χρέος: μακροπρόθεσμο + τρέχον, μαζί με finance leases",
        "debt_lease_noncurrent": "χρέος: μόνο μακροπρόθεσμο, μαζί με finance leases",
        "shares_period_end": "μετοχές στο τέλος της χρονιάς",
        "shares_weighted": "μέσος όρος μετοχών στη χρονιά",
        "gp_reported": "δηλωμένο μεικτό κέρδος",
        "gp_computed": "μεικτό κέρδος = έσοδα − κόστος πωληθέντων",
    },
    "en": {
        "no_t2": "year t-2 missing (needed for beginning-of-year assets)",
        "roa_sign_only": "no beginning-of-year assets — judged by the sign of net income",
        "no_debt_pair": "no debt measure available in both years",
        "no_shares_pair": "no share-count measure available in both years",
        "no_gp_pair": "no gross profit available in both years",
        "split": "likely stock split {ratio} — comparison not meaningful",
        "debt_total": "debt: noncurrent + current portion",
        "debt_noncurrent": "debt: noncurrent only (same measure both years)",
        "debt_ltd": "debt: LongTermDebt (current part included)",
        "debt_lease_total": "debt: incl. finance leases and current part",
        "debt_lease_sum": "debt: noncurrent + current, incl. finance leases",
        "debt_lease_noncurrent": "debt: noncurrent only, incl. finance leases",
        "shares_period_end": "period-end shares",
        "shares_weighted": "weighted-average shares",
        "gp_reported": "reported gross profit",
        "gp_computed": "gross profit = revenues - cost of revenue",
    },
}

# Common split ratios (forward); reverse splits are 1/n.
_SPLIT_RATIOS = [1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20]
_SPLIT_TOLERANCE = 0.08  # +-8%: leaves room for a year of buybacks/issuance


@dataclass
class CriterionDetail:
    key: str
    passed: bool | None
    # Bars to draw, in order; the LAST one is the value being judged.
    # Each entry: (bar_key, value). bar_key is translated by the plotter.
    bars: list[tuple[str, float | None]]
    kind: str  # "pct" | "x" | "usd" | "shares" -> how to format the values
    note: str = ""  # key into NOTES: why a criterion is None, or which measure was used
    note_args: dict = field(default_factory=dict)


@dataclass
class PiotroskiResult:
    ticker: str
    score: int
    max_score: int  # how many of the 9 criteria could be evaluated
    breakdown: dict[str, bool | None]
    fiscal_year: int
    period_end: date
    filed_date: date
    prev_fiscal_year: int | None = None
    details: dict[str, CriterionDetail] = field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────────

def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _avg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2


def _gt(a, b):
    return None if (a is None or b is None) else a > b


def _consistent_pair(curr: AnnualSnapshot, prev: AnnualSnapshot, options):
    """
    options: list of (label, getter). Returns (curr_value, prev_value, label)
    for the FIRST option available in BOTH years, else (None, None, "").

    Example (debt): if year t reports noncurrent + current debt but year
    t-1 reports only noncurrent, the first option (the sum) fails for
    t-1, so we fall through to "noncurrent only" and compare like with
    like, instead of 100 vs 70 where 30 was simply not reported.
    """
    for label, getter in options:
        c, p = getter(curr), getter(prev)
        if c is not None and p is not None:
            return c, p, label
    return None, None, ""


def _sum(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else a + b


# Order matters: the first option available in BOTH years wins.
# The first two are the pre-v0.1.2 options, kept first ON PURPOSE: a company
# that was already scored keeps exactly the same measure (no regression).
# The rest only fill gaps — companies that report debt only under the
# tags added in Sept 2026 (HD, KLAC, GEV...).
# Each option is ONE definition used in both years; we never add a total
# tag (LongTermDebt) to a current-part tag, which would count it twice.
_DEBT_OPTIONS = [
    ("debt_total", lambda s: _sum(s.lt_debt_noncurrent, s.lt_debt_current)),
    ("debt_noncurrent", lambda s: s.lt_debt_noncurrent),
    ("debt_ltd", lambda s: s.lt_debt_total),
    ("debt_lease_total", lambda s: s.lt_debt_lease_total),
    ("debt_lease_sum", lambda s: _sum(s.lt_debt_lease_noncurrent, s.lt_debt_lease_current)),
    ("debt_lease_noncurrent", lambda s: s.lt_debt_lease_noncurrent),
]
_SHARES_OPTIONS = [
    ("shares_period_end", lambda s: s.shares_period_end),
    ("shares_weighted", lambda s: s.shares_weighted_avg),
]
_GROSS_PROFIT_OPTIONS = [
    ("gp_reported", lambda s: s.gross_profit_reported),
    ("gp_computed", lambda s: s.gross_profit_computed),
]


def _looks_like_split(curr_shares: float, prev_shares: float) -> float | None:
    """Returns the split ratio (e.g. 4 or 0.1) if the change looks like a
    stock split, else None."""
    if prev_shares <= 0:
        return None
    r = curr_shares / prev_shares
    for n in _SPLIT_RATIOS:
        if abs(r / n - 1) <= _SPLIT_TOLERANCE:
            return n
        if abs(r * n - 1) <= _SPLIT_TOLERANCE:
            return 1 / n
    return None


# ── main scorer ───────────────────────────────────────────────────────────

def piotroski_f_score(ticker: str, engine, as_of=None) -> PiotroskiResult | None:
    snapshots = get_annual_snapshots(ticker, engine, as_of=as_of, n_years=3)
    return score_from_snapshots(ticker, snapshots)


def score_from_snapshots(ticker: str, snapshots: list[AnnualSnapshot]) -> PiotroskiResult | None:
    """Pure function (no DB) — split out so it can be unit-tested."""
    if len(snapshots) < 2 or not is_consecutive(snapshots[0], snapshots[1]):
        # No valid "this year vs last year" pair (recent IPO, or a
        # missing year between the two most recent 10-Ks).
        return None

    curr, prev = snapshots[0], snapshots[1]
    prev2 = None
    if len(snapshots) >= 3 and is_consecutive(prev, snapshots[2]):
        prev2 = snapshots[2]
    no_t2 = "no_t2"

    d: dict[str, CriterionDetail] = {}

    # Beginning-of-year assets: end of the previous fiscal year.
    begin_assets_curr = prev.total_assets
    begin_assets_prev = prev2.total_assets if prev2 else None

    # 1. positive_roa — the sign of ROA equals the sign of net income
    #    (assets are positive), so it needs no scaling to pass/fail.
    roa_curr = _div(curr.net_income, begin_assets_curr)
    d["positive_roa"] = CriterionDetail(
        "positive_roa",
        None if curr.net_income is None else curr.net_income > 0,
        [("curr", roa_curr)],
        "pct",
        "" if roa_curr is not None or curr.net_income is None
        else "roa_sign_only",
    )

    # 2. positive_cfo
    d["positive_cfo"] = CriterionDetail(
        "positive_cfo",
        None if curr.operating_cf is None else curr.operating_cf > 0,
        [("curr", curr.operating_cf)],
        "usd",
    )

    # 3. roa_improved
    roa_prev = _div(prev.net_income, begin_assets_prev)
    d["roa_improved"] = CriterionDetail(
        "roa_improved", _gt(roa_curr, roa_prev),
        [("prev", roa_prev), ("curr", roa_curr)], "pct",
        "" if prev2 else no_t2,
    )

    # 4. cfo_exceeds_net_income (accruals) — both sides share the same
    #    denominator, so scaling cancels out.
    d["cfo_exceeds_net_income"] = CriterionDetail(
        "cfo_exceeds_net_income", _gt(curr.operating_cf, curr.net_income),
        [("net_income", curr.net_income), ("cfo", curr.operating_cf)], "usd",
    )

    # 5. leverage_decreased — long-term debt / AVERAGE total assets.
    debt_c, debt_p, debt_label = _consistent_pair(curr, prev, _DEBT_OPTIONS)
    lev_curr = _div(debt_c, _avg(curr.total_assets, prev.total_assets))
    lev_prev = _div(debt_p, _avg(prev.total_assets, begin_assets_prev))
    lev_note = debt_label
    if not debt_label:
        lev_note = "no_debt_pair"
    elif not prev2:
        lev_note = no_t2
    d["leverage_decreased"] = CriterionDetail(
        "leverage_decreased",
        None if lev_curr is None or lev_prev is None else lev_curr < lev_prev,
        [("prev", lev_prev), ("curr", lev_curr)], "pct", lev_note,
    )

    # 6. current_ratio_improved — point-in-time ratio, no scaling.
    cr_curr = _div(curr.current_assets, curr.current_liabilities)
    cr_prev = _div(prev.current_assets, prev.current_liabilities)
    d["current_ratio_improved"] = CriterionDetail(
        "current_ratio_improved", _gt(cr_curr, cr_prev),
        [("prev", cr_prev), ("curr", cr_curr)], "x",
    )

    # 7. no_dilution
    sh_c, sh_p, sh_label = _consistent_pair(curr, prev, _SHARES_OPTIONS)
    passed, note, note_args = None, sh_label or "no_shares_pair", {}
    if sh_label:
        split = _looks_like_split(sh_c, sh_p)
        if split is not None:
            ratio = f"{split:g}:1" if split >= 1 else f"1:{1 / split:g}"
            note, note_args = "split", {"ratio": ratio}
        else:
            passed = sh_c <= sh_p
    d["no_dilution"] = CriterionDetail(
        "no_dilution", passed, [("prev", sh_p), ("curr", sh_c)], "shares", note, note_args,
    )

    # 8. gross_margin_improved
    gp_c, gp_p, gp_label = _consistent_pair(curr, prev, _GROSS_PROFIT_OPTIONS)
    gm_curr = _div(gp_c, curr.revenues)
    gm_prev = _div(gp_p, prev.revenues)
    d["gross_margin_improved"] = CriterionDetail(
        "gross_margin_improved", _gt(gm_curr, gm_prev),
        [("prev", gm_prev), ("curr", gm_curr)], "pct",
        gp_label or "no_gp_pair",
    )

    # 9. asset_turnover_improved — revenues / BEGINNING-of-year assets.
    to_curr = _div(curr.revenues, begin_assets_curr)
    to_prev = _div(prev.revenues, begin_assets_prev)
    d["asset_turnover_improved"] = CriterionDetail(
        "asset_turnover_improved", _gt(to_curr, to_prev),
        [("prev", to_prev), ("curr", to_curr)], "x",
        "" if prev2 else no_t2,
    )

    breakdown = {k: d[k].passed for k in CRITERIA}
    valid = [v for v in breakdown.values() if v is not None]

    return PiotroskiResult(
        ticker=ticker,
        score=sum(valid),
        max_score=len(valid),
        breakdown=breakdown,
        fiscal_year=curr.fiscal_year,
        period_end=curr.period_end,
        filed_date=curr.filed_date,
        prev_fiscal_year=prev.fiscal_year,
        details={k: d[k] for k in CRITERIA},
    )


# ── red flags ("βέτο") ────────────────────────────────────────────────────
# The 9 criteria are NOT equally serious, but the F-Score adds them up as
# if they were: 6/9 from failing the three cash/profit checks and 6/9 from
# failing liquidity, turnover and a 0.3% share increase look the same.
# Red flags pull the serious failures out so the card shows them
# regardless of the total. They do NOT change score/max_score.
#
# Thresholds are judgement calls, NOT backtested (that needs the
# investor-screens walk-forward). Small failures stay ordinary ✗:
#   - no_dilution: +0.5% shares from stock comp is not a red flag; +2% is.
#   - accruals: CFO a bit below net income is routine; CFO < 80% of it
#     means a fifth of the reported profit never arrived as cash.
DILUTION_FLAG = 0.02
ACCRUALS_FLAG = 0.80

FLAG_TEXT = {
    "el": {
        "losses": "ζημιές στο 10-K",
        "negative_cfo": "αρνητικές ταμειακές ροές",
        "low_cash_conversion": "ταμείο {gap:.0%} κάτω από τα κέρδη",
        "dilution": "νέες μετοχές +{pct:.1%}",
    },
    "en": {
        "losses": "net loss in the 10-K",
        "negative_cfo": "negative operating cash flow",
        "low_cash_conversion": "cash flow {gap:.0%} below net income",
        "dilution": "share count +{pct:.1%}",
    },
}


@dataclass
class RedFlag:
    key: str  # "losses" | "negative_cfo" | "low_cash_conversion" | "dilution"
    criterion: str  # the Piotroski criterion it comes from
    args: dict = field(default_factory=dict)

    def text(self, lang: str = "el") -> str:
        return FLAG_TEXT[lang][self.key].format(**self.args)


def red_flags(result: PiotroskiResult | None) -> list[RedFlag]:
    """
    Example: net income 100, CFO 60 -> cash conversion 60% < 80%
             -> RedFlag("low_cash_conversion", gap=40%).
             Shares 1,000M -> 1,031M (+3.1% > 2%) -> RedFlag("dilution").
    A criterion that is None (missing data) never raises a flag.
    """
    if result is None:
        return []
    det, flags = result.details, []

    if det["positive_roa"].passed is False:
        flags.append(RedFlag("losses", "positive_roa"))

    cfo_det = det["positive_cfo"]
    if cfo_det.passed is False:
        flags.append(RedFlag("negative_cfo", "positive_cfo"))

    # Accruals: only meaningful when there IS a profit to convert into cash,
    # and redundant when CFO is already negative (flagged above).
    ni, cfo = dict(det["cfo_exceeds_net_income"].bars).get("net_income"), cfo_det.bars[0][1]
    if (det["cfo_exceeds_net_income"].passed is False and cfo_det.passed is True
            and ni and ni > 0 and cfo / ni < ACCRUALS_FLAG):
        flags.append(RedFlag("low_cash_conversion", "cfo_exceeds_net_income",
                             {"gap": 1 - cfo / ni}))

    dil = det["no_dilution"]
    if dil.passed is False:
        prev_sh, curr_sh = dict(dil.bars)["prev"], dict(dil.bars)["curr"]
        growth = curr_sh / prev_sh - 1
        if growth >= DILUTION_FLAG:
            flags.append(RedFlag("dilution", "no_dilution", {"pct": growth}))

    return flags


# ── history ───────────────────────────────────────────────────────────────

def piotroski_history(ticker: str, engine, n_years: int = 10) -> pd.DataFrame:
    """
    F-Score for each of the last `n_years` fiscal years, each computed
    as_of that year's own 10-K filing date — i.e. exactly what an
    investor could have computed on the day the report came out.
    One row per fiscal year, oldest first.
    """
    snaps = get_annual_snapshots(ticker, engine, n_years=n_years)
    rows = []
    for s in sorted(snaps, key=lambda s: s.period_end):
        r = piotroski_f_score(ticker, engine, as_of=s.filed_date)
        if r is None or r.period_end != s.period_end:
            continue
        rows.append({
            "fiscal_year": r.fiscal_year, "filed_date": r.filed_date,
            "score": r.score, "max_score": r.max_score,
            "failed": r.max_score - r.score, "n_a": len(CRITERIA) - r.max_score,
            **r.breakdown,
        })
    return pd.DataFrame(rows)


# ── text report ───────────────────────────────────────────────────────────

def fmt_value(v: float | None, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "x":
        return f"{v:.2f}x"
    scale = abs(v)
    unit = "$" if kind == "usd" else ""
    if scale >= 1e9:
        return f"{unit}{v / 1e9:.2f}B"
    if scale >= 1e6:
        return f"{unit}{v / 1e6:.1f}M"
    return f"{unit}{v:,.0f}"


_BAR_TEXT = {
    "el": {"prev": "πέρσι", "curr": "φέτος", "cfo": "ταμείο", "net_income": "κέρδη"},
    "en": {"prev": "prior", "curr": "current", "cfo": "CFO", "net_income": "NI"},
}


def note_text(det: CriterionDetail, lang: str = "el") -> str:
    if not det.note:
        return ""
    return NOTES[lang][det.note].format(**det.note_args)


def print_report(result: PiotroskiResult | None, lang: str = "el") -> None:
    if result is None:
        print("Δεν υπάρχουν 2 συνεχόμενα έτη 10-K — δεν γίνεται σύγκριση."
              if lang == "el" else
              "Not enough data (need 2 consecutive years of 10-K filings).")
        return

    lab, bt = LABELS[lang], _BAR_TEXT[lang]
    print(f"── Piotroski F-Score: {result.ticker} ──────────────────")
    print(f"   FY{result.fiscal_year} vs FY{result.prev_fiscal_year}  (filed {result.filed_date})")
    print(f"   Score: {result.score} / {result.max_score}")
    for group, keys in GROUPS.items():
        print(f"\n   {lab[group]}")
        for k in keys:
            det = result.details[k]
            symbol = "PASS" if det.passed is True else ("FAIL" if det.passed is False else "N/A ")
            nums = "  ".join(f"{bt[b]} {fmt_value(v, det.kind)}" for b, v in det.bars)
            print(f"   [{symbol}]  {lab[k]:<28} {nums}")
            if det.note:
                print(f"            ↳ {note_text(det, lang)}")
    flags = red_flags(result)
    if flags:
        print("\n   🚩 " + " · ".join(f.text(lang) for f in flags))
