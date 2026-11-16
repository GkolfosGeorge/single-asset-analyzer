"""The 🏷️ peer line of the card (valuation/quick.py + the weekly table). Invented companies."""

import pandas as pd

from valuation import quick as Q

TODAY = pd.Timestamp("2026-11-13")


def table(as_of="2026-11-07", n=9, industry="Semiconductors", sector="Technology", extra=()):
    """n companies S0..S{n-1}: E/P 1%..n%, EBIT/EV 2%..2n%, FCF 0.5%..; plus extra rows."""
    rows = [{"ticker": f"S{i}", "as_of": as_of, "sector": sector, "industry": industry,
             "earnings_yield": (i + 1) / 100, "ebit_yield": 2 * (i + 1) / 100,
             "fcf_yield": (i + 1) / 200, "financial": False, "note": ""} for i in range(n)]
    return pd.DataFrame(rows + list(extra))


def test_zones_against_the_peers_not_against_itself():
    # S0..S8; X is the 10th: E/P 8.5% → cheaper than 8 of 9 → φθηνή
    #                       EBIT/EV 1% → cheaper than none → ακριβή
    #                       FCF 2.4% → between 2.0% and 2.5%: cheaper than 4 of 9 → στο μέσο
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors"}
    own = {"earnings_yield": 0.085, "ebit_yield": 0.01, "fcf_yield": 0.024}
    r = Q.peer_comparison("X", own, False, table(extra=[x]), TODAY)
    assert r["group"] == "Semiconductors" and r["level"] == "κλάδος" and r["size"] == 10
    assert {k: v[0] for k, v in r["zones"].items()} == \
        {"P/E": "φθηνή", "EV/EBIT": "ακριβή", "FCF": "στο μέσο"}
    assert r["zones"]["P/E"][2] == 9                   # 9 peers, the company itself excluded


def test_today_values_are_used_not_the_table_row_of_the_company():
    # the table still has last week's E/P 0.5% for X; today's own value 9.5% decides
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors",
         "earnings_yield": 0.005}
    r = Q.peer_comparison("X", {"earnings_yield": 0.095}, False, table(extra=[x]), TODAY)
    assert r["zones"]["P/E"][0] == "φθηνή"


def test_financials_get_pe_only():
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors"}
    own = {"earnings_yield": 0.05, "ebit_yield": 0.5, "fcf_yield": 0.5}
    r = Q.peer_comparison("X", own, True, table(extra=[x]), TODAY)
    assert list(r["zones"]) == ["P/E"]


def test_missing_own_value_drops_only_that_metric():
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors"}
    r = Q.peer_comparison("X", {"earnings_yield": 0.05, "ebit_yield": None}, False, table(extra=[x]), TODAY)
    assert list(r["zones"]) == ["P/E"]


def test_old_table_is_not_shown():
    x = {"ticker": "X", "as_of": "2026-10-20", "sector": "Technology", "industry": "Semiconductors"}
    assert Q.peer_comparison("X", {"earnings_yield": 0.05}, False,
                             table(as_of="2026-10-20", extra=[x]), TODAY) is None


def test_company_not_in_the_table():
    assert Q.peer_comparison("ZZZ", {"earnings_yield": 0.05}, False, table(), TODAY) is None


def test_too_small_group_falls_back_to_sector():
    # 9 in Semiconductors, X alone in "Solar" (same sector) → compared with the sector (10)
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Solar"}
    r = Q.peer_comparison("X", {"earnings_yield": 0.05}, False, table(extra=[x]), TODAY)
    assert r["level"] in ("τομέας", "ομάδα") and r["size"] >= 10


def test_no_group_at_all():
    # only 4 companies in the whole sector → no comparison
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors"}
    assert Q.peer_comparison("X", {"earnings_yield": 0.05}, False, table(n=3, extra=[x]), TODAY) is None


def test_card_line():
    s = Q.QuickSnapshot("X")
    s.pe = Q.Metric(20.0, 0.5, 2016)
    s.peer = {"group": "Ημιαγωγοί", "level": "ομάδα", "size": 24, "as_of": "2026-11-07",
              "zones": {"P/E": ("στο μέσο", 0.5, 23), "EV/EBIT": ("φθηνή", 0.8, 23)}}
    lines = Q.format_card(s)
    assert any(l.strip() == "🏷️ Κλάδος: Ημιαγωγοί (24): P/E στο μέσο · EV/EBIT φθηνή" for l in lines), lines
    s.peer = {**s.peer, "level": "τομέας", "group": "Technology"}
    assert any("🏷️ Κλάδος: τομέας Technology (24)" in l for l in Q.format_card(s))
    s.peer = None
    assert not any("🏷️" in l for l in Q.format_card(s))


def test_a_loss_is_shown_as_loss_not_as_expensive():
    # Ford, Nov 2026: E/P −16.5% among car makers → "P/E ζημιές", FCF still ranked
    x = {"ticker": "X", "as_of": "2026-11-07", "sector": "Technology", "industry": "Semiconductors"}
    r = Q.peer_comparison("X", {"earnings_yield": -0.165, "fcf_yield": 0.024}, False,
                          table(extra=[x]), TODAY)
    assert r["zones"]["P/E"][0] == "ζημιές" and r["zones"]["FCF"][0] == "στο μέσο"
    s = Q.QuickSnapshot("X")
    s.pe, s.peer = Q.Metric(20.0, 50.0, 2016), r
    assert any("P/E ζημιές · FCF στο μέσο" in l for l in Q.format_card(s))
