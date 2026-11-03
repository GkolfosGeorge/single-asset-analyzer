"""A metric with only annual rows must not break fcf_ttm (PRGO, WU, NVDA, Nov 2026)."""

import pandas as pd

from valuation import ttm as T, yields as Y


def _rows(ends, days, form):
    ends = pd.to_datetime(ends)
    d = pd.DataFrame({"value": 10.0, "period_start": ends - pd.Timedelta(days=days),
                      "period_end": ends, "filed_date": ends + pd.Timedelta(days=30),
                      "form_type": form})
    for c in ("period_start", "period_end", "filed_date"):
        d[c] = d[c].astype("datetime64[s]")
    return d


QUARTERS = ["2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30",
            "2025-09-30", "2025-12-31", "2026-03-31"]
YEARS = ["2023-12-31", "2024-12-31", "2025-12-31"]


def test_fy_only_keeps_datetime_dates():
    t = T.build_ttm(_rows(YEARS, 364, "10-K"), value_col="x")
    assert len(t) == 3
    assert pd.api.types.is_datetime64_any_dtype(t["period_end"])
    assert pd.api.types.is_datetime64_any_dtype(t["available"])


def test_fcf_with_annual_capex_and_quarterly_cfo():
    r = Y.fcf_ttm(_rows(QUARTERS, 90, "10-Q"), {Y.CAPEX: _rows(YEARS, 364, "10-K")})
    # 2025-12-31 is the only date on both sides; CFO TTM 40, capex 10
    assert list(r["fcf_ttm"]) == [30.0]


def test_fcf_with_annual_cfo_and_quarterly_capex():
    r = Y.fcf_ttm(_rows(YEARS, 364, "10-K"), {Y.CAPEX: _rows(QUARTERS, 90, "10-Q")})
    assert list(r["fcf_ttm"]) == [-30.0]


def test_quarterly_only_is_unchanged():
    t = T.build_ttm(_rows(QUARTERS, 90, "10-Q"), value_col="x")
    assert len(t) == 5 and (t["x"] == 40.0).all()
