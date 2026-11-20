"""v0.1.7: basic weighted shares fill periods with no diluted row (XOM since its new CIK)."""

import pandas as pd

from valuation.yields import with_basic_fallback

T = pd.Timestamp


def rows(*items):
    return pd.DataFrame([{"period_start": T(s), "period_end": T(e), "value": v,
                          "filed_date": T(e) + pd.Timedelta(days=40), "form_type": "10-Q"}
                         for s, e, v in items])


def test_diluted_wins_where_both_exist():
    dil = rows(("2026-01-01", "2026-03-31", 4200.0))
    bas = rows(("2026-01-01", "2026-03-31", 4188.0))
    out = with_basic_fallback(dil, bas)
    assert out["value"].tolist() == [4200.0]


def test_basic_fills_only_the_missing_periods():
    dil = rows(("2025-01-01", "2025-03-31", 4300.0))
    bas = rows(("2025-01-01", "2025-03-31", 4290.0), ("2026-01-01", "2026-06-30", 4188.0))
    out = with_basic_fallback(dil, bas).sort_values("period_end")
    assert out["value"].tolist() == [4300.0, 4188.0]


def test_no_diluted_at_all_uses_basic():
    bas = rows(("2026-01-01", "2026-09-30", 4172.0))
    out = with_basic_fallback(rows(), bas)
    assert out["value"].tolist() == [4172.0]


def test_no_basic_leaves_diluted_unchanged():
    dil = rows(("2026-01-01", "2026-03-31", 4200.0))
    assert with_basic_fallback(dil, rows()).equals(dil)
    assert with_basic_fallback(dil, None).equals(dil)
