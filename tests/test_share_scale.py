"""Shares tagged in the wrong unit (MCD, ASH, OM, WAT, Nov 2026). No DB, no network."""

import numpy as np
import pandas as pd

from valuation import ttm as T, yields as Y

ENDS = pd.to_datetime(["2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30",
                       "2025-12-31", "2026-03-31", "2026-06-30"])


def rows(values, ends=ENDS, form="10-Q", days=90):
    ends = pd.to_datetime(ends)
    return pd.DataFrame({"value": values, "period_start": ends - pd.Timedelta(days=days),
                         "period_end": ends, "filed_date": ends + pd.Timedelta(days=30),
                         "form_type": form})


NI = rows(2.1e9)
EPS = rows(2.95)
TRUE = 2.1e9 / 2.95


def fixed(shares, ni=NI, eps=EPS):
    return T.repair_share_scale(rows(shares), ni, eps)


def test_mcd_millions_are_multiplied_back():
    r = fixed([TRUE] * 4 + [711.1] * 4)
    assert r["value"].iloc[:4].tolist() == [TRUE] * 4                  # good rows untouched
    assert (r["value"].iloc[4:] == 711.1e6).all()
    assert [x["factor"] for x in r.attrs["repaired"]] == [1e6] * 4


def test_thousands():
    assert (fixed(711_100.0)["value"] == 711.1e6).all()


def test_wat_too_big_is_divided():
    assert (fixed(711.1e9)["value"] == 711.1e6).all()


def test_correct_shares_are_untouched():
    r = fixed(700e6)
    assert (r["value"] == 700e6).all() and r.attrs["repaired"] == []


def test_real_differences_are_not_repairs():
    for shares in (TRUE * 0.23, TRUE * 3.1, TRUE * 50, TRUE / 50, TRUE * 10):   # IBKR, VNO, splits
        r = fixed(shares)
        assert r["value"].iloc[0] == shares and r.attrs["repaired"] == []


def test_power_with_big_residual_is_not_applied():
    # 1000 × 10 away: ×1000 would still miss NI / EPS by ×10 → leave it
    r = fixed(TRUE / 10_000)
    assert r.attrs["repaired"] == []


def test_wrong_yardstick_cannot_break_a_good_row():
    # BE: shares right, NI / EPS 1000 times too small → ÷1000 would give 234,931
    r = T.repair_share_scale(rows(234_931_000.0), rows(1_174_655.0), rows(5.0))
    assert (r["value"] == 234_931_000.0).all() and r.attrs["repaired"] == []
    # the other direction: ×1000 would give 20 trillion shares
    r = T.repair_share_scale(rows(20e9), rows(4e12), rows(0.2))
    assert (r["value"] == 20e9).all() and r.attrs["repaired"] == []


def test_no_yardstick_no_change():
    assert fixed(711.1, eps=rows(0.05)).attrs["repaired"] == []             # EPS near zero
    assert fixed(711.1, ni=rows(100.0)).attrs["repaired"] == []             # NI near zero
    assert fixed(711.1, ni=NI.iloc[0:0]).attrs["repaired"] == []
    assert T.repair_share_scale(rows(1.0).iloc[0:0], NI, EPS).empty


def test_loss_makers_work_too():
    r = fixed(711.1, ni=rows(-2.1e9), eps=rows(-2.95))
    assert (r["value"] == 711.1e6).all()


def test_only_the_same_period_counts():
    s = rows(711.1)
    ni = rows(2.1e9, ends=ENDS + pd.Timedelta(days=1))
    assert T.repair_share_scale(s, ni, EPS).attrs["repaired"] == []


def test_input_is_not_changed():
    s = rows(711.1)
    T.repair_share_scale(s, NI, EPS)
    assert (s["value"] == 711.1).all()


def test_earliest_filing_is_the_yardstick():
    late = rows(0.295).assign(filed_date=lambda d: d["filed_date"] + pd.Timedelta(days=400))
    r = T.repair_share_scale(rows(711.1), NI, pd.concat([late, EPS]))
    assert (r["value"] == 711.1e6).all()


def test_with_build_latest_all_recent_rows_wrong():
    """Why the ×50 guard alone was not enough: wrong rows confirm each other."""
    raw = rows([TRUE] * 4 + [711.1] * 4)
    before = T.build_latest(raw, shares=True, value_col="s")
    assert before["s"].iloc[-1] == 711.1
    after = T.build_latest(T.repair_share_scale(raw, NI, EPS), shares=True, value_col="s")
    assert after["s"].iloc[-1] == 711.1e6


def test_row_without_yardstick_is_dropped_by_the_guard():
    eps = EPS.copy()
    eps.loc[7, "value"] = 0.01                                              # last period: no yardstick
    r = T.build_latest(T.repair_share_scale(rows([711.1] * 8), NI, eps), shares=True, value_col="s")
    assert r["s"].iloc[-1] == 711.1e6 and r["period_end"].iloc[-1] == ENDS[6]


def test_fcf_history_end_to_end(monkeypatch):
    def fake_metrics(metrics, tickers, engine, as_of=None):
        parts = {Y.SHARES: rows(711.1), Y.NET_INCOME: NI, Y.EPS: EPS,
                 Y.CFO: rows(5e9), Y.CAPEX: rows(1e9)}
        return pd.concat([p.assign(metric=m) for m, p in parts.items() if m in metrics])

    def fake_prices(tickers, start, end=None, retries=1, with_dividends=False):
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
        close = pd.DataFrame({tickers[0]: 300.0}, index=idx)
        zero = pd.DataFrame({tickers[0]: 0.0}, index=idx)
        return close, zero, zero

    monkeypatch.setattr(Y, "load_metrics", fake_metrics)
    monkeypatch.setattr(Y, "load_prices", fake_prices)
    last = Y.fcf_history("MCD", None, years=1).daily.iloc[-1]
    assert np.isclose(last["mcap"], 300 * 711.1e6)                          # 213 bn, not 213,000 $
    assert np.isclose(last["fcf_yield"], 16e9 / (300 * 711.1e6))
