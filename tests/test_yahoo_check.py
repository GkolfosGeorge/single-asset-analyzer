"""Synthetic tests for dev/yahoo_check_202611.py (no database, no network)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "dev" / "yahoo_check_202611.py"
spec = importlib.util.spec_from_file_location("yahoo_check", path)
yc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yc)

Q = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]


def statement(rows: dict, cols=Q) -> pd.DataFrame:
    """Like yfinance: rows = items, columns = dates, NEWEST FIRST."""
    df = pd.DataFrame(rows, index=pd.to_datetime(cols)).T
    return df[df.columns[::-1]]


def test_deviation_examples():
    assert yc.deviation(21, 20) == pytest.approx(0.05)
    assert yc.deviation(-90, -100) == pytest.approx(0.10)
    assert yc.deviation(90, -100) == pytest.approx(1.90)      # opposite sign = far
    assert np.isnan(yc.deviation(None, 20)) and np.isnan(yc.deviation(5, 0))


def test_bands_edges():
    assert [yc.band(x) for x in (0.05, 0.051, -0.15, 0.151, np.nan)] == \
           ["ok", "look", "look", "far", "n/a"]


def test_symbol():
    assert yc.yahoo_symbol("BF.B") == "BF-B" and yc.yahoo_symbol("MSFT") == "MSFT"


def test_wilson_known_value():
    lo, hi = yc.wilson(9, 10)
    assert lo == pytest.approx(0.596, abs=0.01) and hi == pytest.approx(0.982, abs=0.01)


def test_ttm_uses_our_quarter_not_yahoos_newest():
    s = yc._row(statement({"X": [1, 2, 3, 4, 100]}), ["X"])
    assert yc.ttm_at(s, pd.Timestamp("2026-06-30")) == 10      # 1+2+3+4, not the newest 100
    assert yc.ttm_at(s, pd.Timestamp("2026-07-03")) == 10      # 52-week calendar: tolerance
    assert np.isnan(yc.ttm_at(s, pd.Timestamp("2025-12-31")))  # fewer than 4 quarters back


def test_ttm_refuses_holes():
    s = yc._row(statement({"X": [1, 2, np.nan, 4, 5]}), ["X"])
    assert np.isnan(yc.ttm_at(s, pd.Timestamp("2026-06-30")))
    gap = yc._row(statement({"X": [1, 2, 3, 4]}, ["2025-06-30", "2025-09-30", "2026-03-31", "2026-06-30"]), ["X"])
    assert np.isnan(yc.ttm_at(gap, pd.Timestamp("2026-06-30")))  # one quarter skipped


def yahoo_case():
    info = {"impliedSharesOutstanding": 1000, "trailingEps": 5.5, "enterpriseValue": 123_000,
            "freeCashflow": 999_999}
    qis = statement({"Diluted EPS": [1, 1, 1.5, 1.5, 2], "Net Income": [1, 1, 1, 1, 1],
                     "Total Operating Income As Reported": [100, 100, 100, 100, 900],
                     "Operating Income": [200, 200, 200, 200, 900], "EBIT": [110, 110, 110, 110, 900]})
    qcf = statement({"Operating Cash Flow": [250, 250, 250, 250, 900],
                     "Capital Expenditure": [-50, -50, -50, -50, -900]})
    qbs = statement({"Total Debt": [1, 1, 1, 3000, 9], "Capital Lease Obligations": [0, 0, 0, 1000, 9],
                     "Cash Cash Equivalents And Short Term Investments": [1, 1, 1, 500, 9]})
    return info, qis, qcf, qbs


def test_yahoo_numbers_worked_example():
    y = yc.yahoo_numbers(*yahoo_case(), pd.Timestamp("2026-06-30"), close=100.0)
    assert y["y_mcap"] == 100_000
    assert y["yahoo_newer"] is True
    assert y["y_eps"] == 5.0 and y["y_pe"] == 20.0 and y["y_eps_src"] == "quarters"
    assert y["y_pe_pub"] == pytest.approx(100 / 5.5)
    assert y["y_fcf"] == 800                     # 1000 − 200: NOT info["freeCashflow"]
    assert y["y_debt"] == 3000 and y["y_debt_xl"] == 2000
    assert y["y_ev"] == 102_500 and y["y_ev_xl"] == 101_500
    assert y["y_ebit"] == 400 and y["y_ebit_src"] == "as reported"   # not normalized 800
    assert y["y_ebit_alt"] == 440


def test_missing_quarterly_eps_falls_back_only_if_same_quarter():
    info, qis, qcf, qbs = yahoo_case()
    qis = qis.drop(index="Diluted EPS")
    same = yc.yahoo_numbers(info, qis, qcf, qbs, pd.Timestamp("2026-09-30"), 100.0)
    assert same["y_eps"] == 5.5 and same["y_eps_src"] == "trailingEps"
    older = yc.yahoo_numbers(info, qis, qcf, qbs, pd.Timestamp("2026-06-30"), 100.0)
    assert np.isnan(older["y_eps"])              # trailingEps is another quarter → n/a


def test_no_leases_row_means_zero_not_nan():
    info, qis, qcf, qbs = yahoo_case()
    y = yc.yahoo_numbers(info, qis, qcf, qbs.drop(index="Capital Lease Obligations"),
                         pd.Timestamp("2026-06-30"), 100.0)
    assert y["y_debt_xl"] == 3000


def test_empty_yahoo_gives_nans_not_a_crash():
    y = yc.yahoo_numbers({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                         pd.Timestamp("2026-06-30"), 100.0)
    assert np.isnan(y["y_pe"]) and np.isnan(y["y_ev"]) and y["yahoo_newer"] is False


def test_ev_driver_finds_cash():
    r = {"mcap": 100e9, "y_mcap": 100.5e9, "y_leases": 1e9, "debt": 20e9, "y_debt_xl": 20e9,
         "cash": 5e9, "y_cash": 30e9}
    assert yc.ev_driver(r) == "ταμείο +25.0 bn"


def test_split_hint():
    assert "10:1" in yc.split_hint(-0.90, [10.0])      # ours is 1/10 of Yahoo's
    assert "10:1" in yc.split_hint(9.05, [10.0])       # ours is ×10
    assert yc.split_hint(0.02, [10.0]) == ""
    assert yc.split_hint(-0.90, []) == ""


def test_financials_get_no_ev():
    r = yc.add_deviations({"pe": 10, "y_pe": 10, "ev": 5, "y_ev": 9}, financial=True)
    assert r["dev_pe"] == 0 and np.isnan(r["dev_ev"]) and r["ev_driver"] == ""


def test_stratified_sample():
    sectors = {f"A{i}": "Tech" for i in range(100)} | {f"B{i}": "Energy" for i in range(10)} \
        | {"X": "Unknown", "Y": None, "Z": "Tech"}
    a = yc.stratified_sample(sectors, 11, seed=1, exclude={"Z"})
    assert a == yc.stratified_sample(dict(reversed(list(sectors.items()))), 11, seed=1, exclude={"Z"})
    assert len(a) == 11 and sum(t.startswith("B") for t in a) == 1
    assert not {"X", "Y", "Z"} & set(a)
    assert len(yc.stratified_sample({"A": "x", "B": "y"}, 5, seed=1)) == 2     # never more than exist
    assert yc.stratified_sample({}, 5, seed=1) == []


def test_band_table_counts():
    t = yc.band_table(pd.DataFrame({"dev_pe": [0.01, -0.02, 0.1, 0.5, np.nan]}))
    row = t[t["δείκτης"] == "pe"].iloc[0]
    assert (row["N"], row["≤5%"], row["5–15%"], row[">15%"], row["n/a"]) == (4, 2, 1, 1, 1)
