"""valuation/peer_table.py — the weekly yields table, with invented histories."""

from types import SimpleNamespace

import pandas as pd

from valuation import peer_table as T

END = pd.Timestamp("2026-11-13")


def daily(col, values, last_day=END):
    idx = pd.date_range(end=last_day, periods=len(values), freq="D")
    return pd.DataFrame({col: values}, index=idx)


def test_last_value_only_if_the_price_is_recent():
    assert T._last(daily("x", [0.01, 0.02]), "x", END) == 0.02
    assert T._last(daily("x", [0.01, 0.02], END - pd.Timedelta(days=30)), "x", END) is None
    assert T._last(daily("x", [0.01, None]), "x", END) == 0.01      # NaN at the end → the day before
    assert T._last(pd.DataFrame(), "x", END) is None


def test_company_yields_from_the_existing_histories(monkeypatch):
    eps = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(T.P, "load_eps", lambda tickers, engine: eps)
    close = daily("AAA", [100.0, 100.0])
    monkeypatch.setattr(T, "load_prices", lambda t, s, e: (close, pd.DataFrame()))
    monkeypatch.setattr(T.P, "build_ttm", lambda e, sp: "ttm")
    monkeypatch.setattr(T.P, "daily_pe", lambda c, ttm: pd.DataFrame(
        {"close": c, "eps_ttm": 5.0, "pe": 20.0, "loss": False}))
    fcf = SimpleNamespace(daily=daily("fcf_yield", [0.03, 0.04]), status="ok")
    monkeypatch.setattr(T.Y, "fcf_history", lambda t, e, years: fcf)
    monkeypatch.setattr(T.E, "ev_history", lambda t, e, years, base: SimpleNamespace(
        daily=daily("ebit_yield", [0.06]), status="ok"))
    y = T.company_yields("AAA", None, END)
    assert y["earnings_yield"] == 0.05 and y["fcf_yield"] == 0.04 and y["ebit_yield"] == 0.06
    assert y["financial"] is False and y["note"] == ""


def test_a_bank_is_marked_and_one_failing_part_doesnt_stop_the_rest(monkeypatch):
    monkeypatch.setattr(T.P, "load_eps", lambda tickers, engine: (_ for _ in ()).throw(RuntimeError("db")))
    fcf = SimpleNamespace(daily=pd.DataFrame(), status="χωρίς capex στη βάση")
    monkeypatch.setattr(T.Y, "fcf_history", lambda t, e, years: fcf)
    monkeypatch.setattr(T.E, "ev_history", lambda t, e, years, base: SimpleNamespace(
        daily=pd.DataFrame(), status="χωρίς EBIT στη βάση"))
    y = T.company_yields("JPM", None, END)
    assert y["financial"] is True and y["earnings_yield"] is None and y["note"].startswith("P/E")


def test_build_and_coverage(monkeypatch):
    vals = {"A": 0.05, "B": None}
    monkeypatch.setattr(T, "company_yields", lambda t, e, end: {
        "earnings_yield": vals[t], "ebit_yield": None, "fcf_yield": None, "financial": False, "note": ""})
    t = T.build(None, ["A", "B"], {"A": {"sector": "Technology", "industry": "Semiconductors"}}, END,
                progress_every=0)
    assert list(t.columns) == T.COLUMNS and len(t) == 2
    assert t.set_index("ticker").loc["A", "industry"] == "Semiconductors"
    assert T.coverage(t) == 1
