"""v0.1.9: "financial" from Yahoo's industry, not from missing tags. Real tickers, no network."""

import pandas as pd
import pytest

from valuation import peers as PEERS
from valuation import quick as Q


@pytest.mark.parametrize("ticker, industry, fallback, expected", [
    # banks, broker-dealers, insurers: financial even though they report capex (C) — the old test missed them
    ("C", "Banks - Diversified", False, True),
    ("FITB", "Banks - Regional", False, True),
    ("GS", "Capital Markets", False, True),
    ("PGR", "Insurance - Property & Casualty", False, True),
    ("MET", "Insurance - Life", False, True),
    ("BRK-B", "Insurance - Diversified", False, True),
    ("FNMA", "Mortgage Finance", True, True),
    ("COF", "Credit Services", False, True),
    # Financial Services, but ordinary businesses: FCF and EV/EBIT mean something
    ("SPGI", "Financial Data & Stock Exchanges", False, False),
    ("AON", "Insurance Brokers", False, False),
    ("BLK", "Asset Management", False, False),
    # the two hand-made lists
    ("V", "Credit Services", False, False),
    ("MA", "Credit Services", True, False),
    ("STT", "Asset Management", False, True),
    ("PFG", "Asset Management", True, True),
    ("APO", "Asset Management", True, True),
    # old test wrongly said financial: the industry wins
    ("ADP", "Software - Application", True, False),
    ("PSX", "Oil & Gas Refining & Marketing", True, False),
    # no industry: the old test decides
    ("XYZ", None, True, True),
    ("XYZ", float("nan"), False, False),
    ("XYZ", "", True, True),
])
def test_is_financial(ticker, industry, fallback, expected):
    assert PEERS.is_financial(ticker, industry, fallback) is expected


def test_quick_uses_the_table_industry(monkeypatch):
    """The card's flag comes from the weekly table's industry; without the table, the old test stays."""
    t = pd.DataFrame([{"ticker": "C", "industry": "Banks - Diversified"},
                      {"ticker": "ADP", "industry": "Software - Application"}])
    monkeypatch.setattr(Q, "load_peer_table", lambda: t)
    snap = Q.QuickSnapshot("C")
    row = t[t["ticker"] == "C"]
    assert PEERS.is_financial("C", row["industry"].iloc[0], snap.financial) is True


def test_verdict_with_only_todays_pe():
    s = Q.QuickSnapshot("V", pe_now=34.0)
    assert Q.verdict(s) == ("⚪", "Χωρίς ιστορικό για σύγκριση")
    assert Q.verdict(Q.QuickSnapshot("X")) == ("⚪", "Αποτίμηση μη διαθέσιμη")
    card = Q.format_card(s)
    assert card[0].strip() == "💰 ⚪ Χωρίς ιστορικό για σύγκριση   P/E ≈ 34 (χωρίς ιστορικό)"
    assert Q.format_line(s) == ["  📊 ⚪ χωρίς ιστορικό"]


def test_financial_card_hides_ev_and_fcf():
    m = Q.Metric(value=12.0, pct=50)
    s = Q.QuickSnapshot("C", pe=m, ev_ebit=Q.Metric(8.0, 90), fcf_yield=Q.Metric(0.1, 90), financial=True)
    line = Q.format_card(s)[0]
    assert "EV/EBIT" not in line and "FCF" not in line and "P/E 12" in line
    assert any("Χρηματοοικονομική εταιρεία" in l for l in Q.format_card(s))


def test_bank_gets_no_cash_flow_warnings():
    det = {"positive_roa": True, "positive_cfo": False, "roa_improved": True, "cfo_exceeds_net_income": False,
           "leverage_decreased": False, "current_ratio_improved": None, "no_dilution": True,
           "gross_margin_improved": None, "asset_turnover_improved": True}
    flags = [("positive_cfo", "αρνητικές ταμειακές ροές"), ("leverage_decreased", "χρέος ↑ 20%")]
    bank = Q.QuickSnapshot("C", fscore=(5, 7), fscore_detail=det, fscore_flags=flags, financial=True)
    out = "\n".join(Q._health_lines(bank, ""))
    assert "ταμειακές" not in out and "κέρδη > ταμείο" not in out
    assert "χρέος ↑ 20%" in out                            # the other flags stay
    # cash-flow criteria shown "–" and not counted: 5/7 had positive_cfo ✗ and cfo>NI ✗ → 5/5
    assert out.startswith("🩺 Υγεία 5/5   Κερδοφορία ✓–✓–")
    other = Q.QuickSnapshot("X", fscore=(5, 7), fscore_detail=det, fscore_flags=flags, financial=False)
    o = "\n".join(Q._health_lines(other, ""))
    assert "αρνητικές ταμειακές ροές" in o and "κέρδη > ταμείο" in o


def test_jpm_like_score_and_watchlist_line():
    det = {"positive_roa": True, "positive_cfo": False, "roa_improved": False, "cfo_exceeds_net_income": False,
           "leverage_decreased": False, "current_ratio_improved": None, "no_dilution": True,
           "gross_margin_improved": None, "asset_turnover_improved": False}
    jpm = Q.QuickSnapshot("JPM", pe=Q.Metric(13.0, 50), fscore=(2, 7), fscore_detail=det, financial=True)
    assert Q._health_lines(jpm, "")[0].startswith("🩺 Υγεία 2/5   Κερδοφορία ✓–✗–")
    assert "Υγεία 2/5" in Q.format_line(jpm)[0]
    # a passed cash-flow criterion also stops counting: ✓✓ → – –, 4/6 → 3/4
    det2 = dict(det, positive_cfo=True, cfo_exceeds_net_income=True, roa_improved=True)
    assert Q._health(Q.QuickSnapshot("X", fscore=(4, 6), fscore_detail=det2, financial=True))[:2] == (2, 4)
    assert Q._health(Q.QuickSnapshot("X", fscore=(4, 6), fscore_detail=det2, financial=False))[:2] == (4, 6)
