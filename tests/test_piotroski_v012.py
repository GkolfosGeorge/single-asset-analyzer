"""
Tests for v0.1.2: debt tags in Piotroski + red flags in the card.
No database: snapshots are built by hand. Run from the repo root:
    python -m pytest tests -q
"""
import sys
import types
from datetime import date

# line_items imports `fundamentals` (trading-shared-data); stub it so the
# tests run anywhere. The scorer itself never touches the DB.
sys.modules.setdefault("fundamentals", types.SimpleNamespace(get_fundamentals=None))

from quality_scores.line_items import AnnualSnapshot  # noqa: E402
from quality_scores.piotroski import red_flags, score_from_snapshots  # noqa: E402
from valuation.quick import QuickSnapshot, _health_lines  # noqa: E402


def snap(year, **kw):
    base = dict(
        fiscal_year=year, period_end=date(year, 12, 31), filed_date=date(year + 1, 2, 15),
        revenues=1000.0, cost_of_revenue=600.0, gross_profit_reported=None,
        net_income=100.0, operating_cf=150.0, total_assets=2000.0,
        current_assets=500.0, current_liabilities=400.0,
        lt_debt_noncurrent=None, lt_debt_current=None,
        shares_period_end=100.0, shares_weighted_avg=100.0,
    )
    base.update(kw)
    return AnnualSnapshot(**base)


# ── debt ────────────────────────────────────────────────────────────────────

def test_only_longtermdebt_is_now_scored():
    # HD-like: no Noncurrent/Current tags, only LongTermDebt. Before v0.1.2 → None.
    r = score_from_snapshots("X", [snap(2025, lt_debt_total=400.0),
                                   snap(2024, lt_debt_total=500.0),
                                   snap(2023, lt_debt_total=500.0)])
    d = r.details["leverage_decreased"]
    assert d.passed is True
    assert d.note == "debt_ltd"


def test_old_measure_still_wins_no_regression():
    # Company reports BOTH the old pair and LongTermDebt: the old pair is used,
    # exactly as in v0.1.1.
    kw = dict(lt_debt_noncurrent=300.0, lt_debt_current=50.0, lt_debt_total=999.0)
    r = score_from_snapshots("X", [snap(2025, **kw), snap(2024, **kw), snap(2023, **kw)])
    assert r.details["leverage_decreased"].note == "debt_total"


def test_measures_are_never_mixed_across_years():
    # 2025 has only LongTermDebt, 2024 only the lease total → no common measure.
    r = score_from_snapshots("X", [snap(2025, lt_debt_total=400.0),
                                   snap(2024, lt_debt_lease_total=500.0),
                                   snap(2023, lt_debt_lease_total=500.0)])
    assert r.details["leverage_decreased"].passed is None
    assert r.details["leverage_decreased"].note == "no_debt_pair"


def test_lease_pair_is_summed_not_double_counted():
    kw = dict(lt_debt_lease_noncurrent=300.0, lt_debt_lease_current=50.0)
    r = score_from_snapshots("X", [snap(2025, **kw), snap(2024, **kw), snap(2023, **kw)])
    d = r.details["leverage_decreased"]
    assert d.note == "debt_lease_sum"
    assert abs(d.bars[1][1] - 350.0 / 2000.0) < 1e-12


# ── red flags ───────────────────────────────────────────────────────────────

def _score(curr_kw):
    return score_from_snapshots("X", [snap(2025, **curr_kw), snap(2024), snap(2023)])


def test_healthy_company_has_no_flags():
    assert red_flags(_score({})) == []


def test_small_dilution_is_not_a_flag():
    r = _score({"shares_period_end": 100.5})          # +0.5%
    assert r.breakdown["no_dilution"] is False
    assert red_flags(r) == []


def test_large_dilution_is_a_flag():
    flags = red_flags(_score({"shares_period_end": 104.1}))
    assert [f.key for f in flags] == ["dilution"]
    assert flags[0].text("el") == "νέες μετοχές +4.1%"


def test_low_cash_conversion():
    # NI 100, CFO 60 → 40% of profit never arrived as cash
    flags = red_flags(_score({"operating_cf": 60.0}))
    assert [f.key for f in flags] == ["low_cash_conversion"]
    assert flags[0].text("el") == "ταμείο 40% κάτω από τα κέρδη"


def test_cfo_slightly_below_profit_is_not_a_flag():
    assert red_flags(_score({"operating_cf": 90.0})) == []


def test_negative_cfo_does_not_double_flag_accruals():
    flags = red_flags(_score({"operating_cf": -20.0}))
    assert [f.key for f in flags] == ["negative_cfo"]


def test_split_is_not_dilution():
    r = _score({"shares_period_end": 400.0})          # 4:1 split
    assert r.breakdown["no_dilution"] is None
    assert red_flags(r) == []


# ── card ────────────────────────────────────────────────────────────────────

def test_card_moves_flagged_items_out_of_the_x_line():
    r = _score({"operating_cf": 60.0, "shares_period_end": 104.1})
    q = QuickSnapshot("X", fscore=(r.score, r.max_score), fscore_detail=dict(r.breakdown),
                      fscore_flags=[(f.criterion, f.text("el")) for f in red_flags(r)])
    lines = _health_lines(q, "")
    assert lines[1] == "   🚩 ταμείο 40% κάτω από τα κέρδη · νέες μετοχές +4.1%"
    assert all("νέες μετοχές" not in l for l in lines[2:])
