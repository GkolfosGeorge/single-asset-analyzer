"""valuation/peers.py — peer groups and zones (Nov 2026). Invented classifications."""

from valuation import peers as P


def cls(**industries):
    """industry=(sector, count) → {ticker: {...}} with tickers like Semiconductors_0."""
    out = {}
    for ind, (sector, n) in industries.items():
        for i in range(n):
            out[f"{ind}_{i}"] = {"sector": sector, "industry": ind.replace("_", " ")}
    return out


def test_industry_when_big_enough():
    g = P.peer_groups(cls(Semiconductors=("Technology", 9))).set_index("ticker")
    assert g.loc["Semiconductors_0", "level"] == "κλάδος" and g.loc["Semiconductors_0", "size"] == 9


def test_group_when_the_industry_is_small():
    # MO-like: Tobacco 2 + Beverages 7 → group of 9
    c = {**cls(Tobacco=("Consumer Defensive", 2)),
         **{f"B{i}": {"sector": "Consumer Defensive", "industry": "Beverages - Non-Alcoholic"} for i in range(7)}}
    g = P.peer_groups(c).set_index("ticker")
    assert g.loc["Tobacco_0", "level"] == "ομάδα" and g.loc["Tobacco_0", "group"] == "Ποτά & καπνός"
    assert g.loc["Tobacco_0", "size"] == 9
    assert sorted(P.members(P.peer_groups(c), "Tobacco_0"))[:2] == ["B0", "B1"]


def test_sector_when_the_group_is_small_too():
    c = {**cls(Tobacco=("Consumer Defensive", 2)),
         **{f"H{i}": {"sector": "Consumer Defensive", "industry": "Household & Personal Products"} for i in range(7)}}
    g = P.peer_groups(c).set_index("ticker")
    assert g.loc["Tobacco_0", "level"] == "τομέας" and g.loc["Tobacco_0", "size"] == 9


def test_nothing_when_even_the_sector_is_small():
    g = P.peer_groups(cls(Tobacco=("Consumer Defensive", 2))).set_index("ticker")
    assert g.loc["Tobacco_0", "level"] == "—" and P.members(P.peer_groups(cls(Tobacco=("Consumer Defensive", 2))), "Tobacco_0") == []


def test_unknown_classification_is_left_out():
    g = P.peer_groups({"X": {"sector": "Unknown", "industry": "Unknown"}})
    assert g.empty


def test_zones():
    peers = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08]
    assert P.zone(0.10, peers)[0] == "φθηνή"            # cheaper than all 7
    assert P.zone(0.005, peers)[0] == "ακριβή"          # dearer than all 7
    assert P.zone(0.05, peers)[0] == "στο μέσο"         # 4 of 7 = 0.57
    z, share, n = P.zone(0.065, peers)                  # 5 of 7 = 0.71
    assert z == "φθηνή" and abs(share - 5 / 7) < 1e-9 and n == 7


def test_zone_needs_enough_peers_and_a_value():
    assert P.zone(0.05, [0.01, 0.02, 0.03])[0] is None
    assert P.zone(None, [0.01] * 10)[0] is None
    assert P.zone(0.05, [None] * 10)[2] == 0


def test_a_loss_maker_ranks_dearest():
    # negative earnings yield: below every profitable peer
    assert P.zone(-0.02, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])[0] == "ακριβή"


def test_every_mapped_industry_has_a_known_group_name():
    assert all(isinstance(v, str) and v for v in P.INDUSTRY_GROUPS.values())
    assert len(set(P.INDUSTRY_GROUPS.values())) >= 25
