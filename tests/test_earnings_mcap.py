"""v0.1.8: earnings yield = net income TTM / market cap, for companies with no EPS (V, BRK-B, HSY…)."""

from datetime import date

import pandas as pd

from valuation import earnings_mcap as EM
from valuation import peer_table as PT
from valuation import quick as Q

T = pd.Timestamp
TODAY = T(date.today())


def quarters(last_end: pd.Timestamp, n=4, value=6e9, lag=30):
    """n consecutive 3-month 10-Q rows of net income ending at last_end."""
    ends = [last_end - pd.DateOffset(months=3 * k) for k in reversed(range(n))]
    return pd.DataFrame({
        "value": [value] * n,
        "period_start": [e - pd.DateOffset(months=3) + pd.Timedelta(days=1) for e in ends],
        "period_end": ends,
        "filed_date": [e + pd.Timedelta(days=lag) for e in ends],
        "form_type": "10-Q",
        "metric": EM.NET_INCOME,
    })


def test_ni_ttm_fresh_sum_of_four_quarters():
    end = TODAY.normalize()
    last = (end - pd.Timedelta(days=60)).to_period("Q").end_time.normalize()
    ni, pe_end = EM.ni_ttm_latest(quarters(last), end)
    assert ni == 24e9 and abs((pe_end - last).days) <= 1


def test_ni_ttm_too_old_is_none():
    last = (TODAY - pd.Timedelta(days=400)).to_period("Q").end_time.normalize()
    assert EM.ni_ttm_latest(quarters(last), TODAY) == (None, None)


def test_yield_from_mcap(monkeypatch):
    last = (TODAY - pd.Timedelta(days=60)).to_period("Q").end_time.normalize()
    monkeypatch.setattr(EM, "load_metrics", lambda m, t, e: quarters(last, value=5.875e9))   # total only
    r = EM.earnings_yield_from_mcap("V", engine=None, market_cap=796e9)
    assert abs(r["earnings_yield"] - 23.5e9 / 796e9) < 1e-12        # 2.95% → P/E ≈ 34
    # no market cap → None; a past date → None (Yahoo's market cap is today's)
    monkeypatch.setattr(EM, "yahoo_market_cap", lambda t: None)
    assert EM.earnings_yield_from_mcap("V", engine=None) is None
    assert EM.earnings_yield_from_mcap("V", engine=None, end=TODAY - pd.Timedelta(days=60),
                                       market_cap=796e9) is None


def test_prefers_net_income_available_to_common():
    last = (TODAY - pd.Timedelta(days=60)).to_period("Q").end_time.normalize()
    total = quarters(last, value=126e6)
    common = quarters(last, value=101e6).assign(metric=EM.NI_COMMON)
    ni, _, basis = EM.pick_net_income(pd.concat([total, common]), TODAY)
    assert ni == 404e6 and basis == "κοινοί μέτοχοι"
    # common is older than total → total
    old = quarters(last - pd.DateOffset(months=6), value=101e6).assign(metric=EM.NI_COMMON)
    ni, _, basis = EM.pick_net_income(pd.concat([total, old]), TODAY)
    assert ni == 504e6 and basis == "καθαρά κέρδη"
    # only total (V, BRK-B)
    assert EM.pick_net_income(total, TODAY)[2] == "καθαρά κέρδη"
    assert EM.pick_net_income(total.iloc[0:0], TODAY) == (None, None, None)


def test_card_shows_pe_without_history():
    s = Q.QuickSnapshot("V", pe_now=33.9, fscore=(7, 9))
    line = Q.format_card(s)[0]
    assert "P/E ≈ 34 (χωρίς ιστορικό)" in line
    assert Q.QuickSnapshot("V", pe_now=33.9).has_fundamentals


def test_peer_table_uses_fallback_only_without_eps(monkeypatch):
    class Empty:
        empty = True
    monkeypatch.setattr(PT.P, "load_eps", lambda t, e: pd.DataFrame())          # no EPS at all
    monkeypatch.setattr(PT.Y, "fcf_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip")))
    monkeypatch.setattr(PT.EM, "earnings_yield_from_mcap",
                        lambda t, e, end=None: {"earnings_yield": 0.0295, "ni_ttm": 1, "market_cap": 1,
                                                "period_end": TODAY})
    out = PT.company_yields("V", engine=None)
    assert out["earnings_yield"] == 0.0295 and "κέρδη ÷ κεφαλαιοποίηση" in out["note"]
