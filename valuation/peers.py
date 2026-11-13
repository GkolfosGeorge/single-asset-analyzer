"""
valuation/peers.py — who a company is compared with, and where it ranks.

PEER GROUP (Nov 2026). Yahoo's industries are too small on their own: in our
universe the median industry has 4 companies, and a rank among 4 is noise.
So three levels, the first with at least MIN_PEERS companies wins:
    1. the industry            ("Semiconductors", 15)
    2. the group of related industries below (hand-made, one-off judgment)
                               ("Τρόφιμα" = Packaged Foods + Confectioners + ...)
    3. the sector              ("Consumer Defensive")
Example: MO (Tobacco, 2 companies) → group "Ποτά & καπνός" (9) → compared with 9.
Limit: Yahoo gives TODAY's industry; fine for the card, to be noted in backtests.

RANK. Yields, not multiples (E/P, EBIT/EV, FCF/P, S/P): a loss-maker then simply
has the lowest yield, and one P/E of 785 doesn't distort anything. The card shows
three zones rather than a precise percentile, because in a group of 8 the rank
moves in steps of 12.5%:
    cheaper than ≥ 2/3 of the group → "φθηνή για τον κλάδο"
    dearer than  ≥ 2/3 of the group → "ακριβή για τον κλάδο"
    otherwise                       → "στο μέσο"
"""

import pandas as pd

MIN_PEERS = 8
CHEAP_SHARE = 2 / 3

# industry (Yahoo) → group. Industries not listed fall straight to the sector.
INDUSTRY_GROUPS = {
    # Basic Materials
    "Specialty Chemicals": "Χημικά", "Chemicals": "Χημικά", "Agricultural Inputs": "Χημικά",
    "Copper": "Μέταλλα & δομικά υλικά", "Gold": "Μέταλλα & δομικά υλικά",
    "Other Industrial Metals & Mining": "Μέταλλα & δομικά υλικά", "Steel": "Μέταλλα & δομικά υλικά",
    "Building Materials": "Μέταλλα & δομικά υλικά",
    # Communication Services
    "Entertainment": "Μέσα ενημέρωσης", "Broadcasting": "Μέσα ενημέρωσης", "Publishing": "Μέσα ενημέρωσης",
    "Advertising Agencies": "Μέσα ενημέρωσης",
    "Internet Content & Information": "Internet & gaming", "Electronic Gaming & Multimedia": "Internet & gaming",
    # Consumer Cyclical
    "Apparel Retail": "Λιανική", "Department Stores": "Λιανική", "Home Improvement Retail": "Λιανική",
    "Specialty Retail": "Λιανική", "Internet Retail": "Λιανική", "Auto & Truck Dealerships": "Λιανική",
    "Discount Stores": "Λιανική", "Grocery Stores": "Λιανική",
    "Apparel Manufacturing": "Ένδυση & πολυτελή", "Footwear & Accessories": "Ένδυση & πολυτελή",
    "Luxury Goods": "Ένδυση & πολυτελή",
    "Auto Manufacturers": "Αυτοκίνητα", "Auto Parts": "Αυτοκίνητα", "Recreational Vehicles": "Αυτοκίνητα",
    "Lodging": "Ταξίδια & αναψυχή", "Resorts & Casinos": "Ταξίδια & αναψυχή",
    "Travel Services": "Ταξίδια & αναψυχή", "Leisure": "Ταξίδια & αναψυχή",
    "Residential Construction": "Κατοικία", "Furnishings, Fixtures & Appliances": "Κατοικία",
    # Consumer Defensive
    "Beverages - Brewers": "Ποτά & καπνός", "Beverages - Non-Alcoholic": "Ποτά & καπνός",
    "Beverages - Wineries & Distilleries": "Ποτά & καπνός", "Tobacco": "Ποτά & καπνός",
    "Packaged Foods": "Τρόφιμα", "Confectioners": "Τρόφιμα", "Farm Products": "Τρόφιμα",
    "Food Distribution": "Τρόφιμα",
    # Energy
    "Oil & Gas E&P": "Πετρέλαιο: παραγωγή", "Oil & Gas Integrated": "Πετρέλαιο: παραγωγή",
    "Oil & Gas Drilling": "Πετρέλαιο: παραγωγή", "Thermal Coal": "Πετρέλαιο: παραγωγή",
    "Oil & Gas Equipment & Services": "Πετρέλαιο: υποδομές & διύλιση",
    "Oil & Gas Refining & Marketing": "Πετρέλαιο: υποδομές & διύλιση",
    "Oil & Gas Midstream": "Πετρέλαιο: υποδομές & διύλιση",
    # Financial Services
    "Banks - Diversified": "Τράπεζες", "Banks - Regional": "Τράπεζες", "Mortgage Finance": "Τράπεζες",
    "Insurance - Life": "Ασφαλιστικές", "Insurance - Property & Casualty": "Ασφαλιστικές",
    "Insurance - Diversified": "Ασφαλιστικές", "Insurance - Reinsurance": "Ασφαλιστικές",
    "Insurance - Specialty": "Ασφαλιστικές",
    "Asset Management": "Διαχείριση κεφαλαίων & αγορές", "Capital Markets": "Διαχείριση κεφαλαίων & αγορές",
    "Financial Data & Stock Exchanges": "Διαχείριση κεφαλαίων & αγορές",
    # Healthcare
    "Drug Manufacturers - General": "Φάρμακα", "Drug Manufacturers - Specialty & Generic": "Φάρμακα",
    "Medical Devices": "Ιατρικά μηχανήματα", "Medical Instruments & Supplies": "Ιατρικά μηχανήματα",
    "Healthcare Plans": "Υπηρεσίες υγείας", "Medical Care Facilities": "Υπηρεσίες υγείας",
    "Medical Distribution": "Υπηρεσίες υγείας", "Health Information Services": "Υπηρεσίες υγείας",
    "Pharmaceutical Retailers": "Υπηρεσίες υγείας",
    # Industrials
    "Specialty Industrial Machinery": "Μηχανήματα", "Farm & Heavy Construction Machinery": "Μηχανήματα",
    "Tools & Accessories": "Μηχανήματα", "Metal Fabrication": "Μηχανήματα", "Conglomerates": "Μηχανήματα",
    "Electrical Equipment & Parts": "Ηλεκτρολογικός εξοπλισμός",
    "Pollution & Treatment Controls": "Ηλεκτρολογικός εξοπλισμός",
    "Airlines": "Μεταφορές", "Railroads": "Μεταφορές", "Trucking": "Μεταφορές",
    "Integrated Freight & Logistics": "Μεταφορές", "Marine Shipping": "Μεταφορές",
    "Rental & Leasing Services": "Μεταφορές",
    "Building Products & Equipment": "Κατασκευές", "Engineering & Construction": "Κατασκευές",
    "Specialty Business Services": "Υπηρεσίες επιχειρήσεων", "Consulting Services": "Υπηρεσίες επιχειρήσεων",
    "Staffing & Employment Services": "Υπηρεσίες επιχειρήσεων",
    "Security & Protection Services": "Υπηρεσίες επιχειρήσεων", "Waste Management": "Υπηρεσίες επιχειρήσεων",
    "Personal Services": "Υπηρεσίες επιχειρήσεων", "Industrial Distribution": "Υπηρεσίες επιχειρήσεων",
    "Business Equipment & Supplies": "Υπηρεσίες επιχειρήσεων", "Real Estate Services": "Υπηρεσίες επιχειρήσεων",
    # Real Estate
    "REIT - Retail": "REIT εμπορικά & ειδικά", "REIT - Office": "REIT εμπορικά & ειδικά",
    "REIT - Diversified": "REIT εμπορικά & ειδικά", "REIT - Industrial": "REIT εμπορικά & ειδικά",
    "REIT - Hotel & Motel": "REIT εμπορικά & ειδικά", "REIT - Healthcare Facilities": "REIT εμπορικά & ειδικά",
    "REIT - Specialty": "REIT εμπορικά & ειδικά",
    # Technology
    "Software - Application": "Λογισμικό", "Software - Infrastructure": "Λογισμικό",
    "Semiconductors": "Ημιαγωγοί", "Semiconductor Equipment & Materials": "Ημιαγωγοί", "Solar": "Ημιαγωγοί",
    "Computer Hardware": "Hardware & εξοπλισμός", "Communication Equipment": "Hardware & εξοπλισμός",
    "Electronic Components": "Hardware & εξοπλισμός", "Consumer Electronics": "Hardware & εξοπλισμός",
    "Scientific & Technical Instruments": "Hardware & εξοπλισμός",
    # Utilities
    "Utilities - Regulated Electric": "Κοινής ωφέλειας", "Utilities - Diversified": "Κοινής ωφέλειας",
    "Utilities - Independent Power Producers": "Κοινής ωφέλειας", "Utilities - Regulated Gas": "Κοινής ωφέλειας",
    "Utilities - Regulated Water": "Κοινής ωφέλειας", "Utilities - Renewable": "Κοινής ωφέλειας",
}


def peer_groups(classification: dict[str, dict], min_peers: int = MIN_PEERS) -> pd.DataFrame:
    """
    {ticker: {"sector", "industry"}} → one row per ticker: level, group, size.
    level ∈ {"κλάδος", "ομάδα", "τομέας", "—"}; "—" = even the sector is too small.
    """
    rows = [{"ticker": t, "sector": v.get("sector"), "industry": v.get("industry")}
            for t, v in classification.items()
            if v.get("sector") not in (None, "Unknown") and v.get("industry") not in (None, "Unknown")]
    d = pd.DataFrame(rows, columns=["ticker", "sector", "industry"])
    if d.empty:
        return d.assign(level=[], group=[], size=[])
    d["grp"] = d["industry"].map(INDUSTRY_GROUPS)
    n_ind = d.groupby("industry")["ticker"].transform("size")
    n_grp = d.groupby("grp")["ticker"].transform("size")          # NaN group → NaN size
    n_sec = d.groupby("sector")["ticker"].transform("size")
    level, group, size = [], [], []
    for i, r in d.iterrows():
        if n_ind[i] >= min_peers:
            level.append("κλάδος"), group.append(r["industry"]), size.append(int(n_ind[i]))
        elif pd.notna(r["grp"]) and n_grp[i] >= min_peers:
            level.append("ομάδα"), group.append(r["grp"]), size.append(int(n_grp[i]))
        elif n_sec[i] >= min_peers:
            level.append("τομέας"), group.append(r["sector"]), size.append(int(n_sec[i]))
        else:
            level.append("—"), group.append(None), size.append(0)
    return d.drop(columns="grp").assign(level=level, group=group, size=size)


def members(groups: pd.DataFrame, ticker: str) -> list[str]:
    """Every ticker in the same peer group, the company itself included."""
    row = groups[groups["ticker"] == ticker]
    if row.empty or row["level"].iloc[0] == "—":
        return []
    lvl, grp = row["level"].iloc[0], row["group"].iloc[0]
    if lvl == "κλάδος":
        return sorted(groups.loc[groups["industry"] == grp, "ticker"])
    if lvl == "ομάδα":
        return sorted(groups.loc[groups["industry"].map(INDUSTRY_GROUPS) == grp, "ticker"])
    return sorted(groups.loc[groups["sector"] == grp, "ticker"])


def zone(own_yield, peer_yields) -> tuple[str | None, float | None, int]:
    """
    Where a yield (higher = cheaper) sits among its peers (the company excluded).
    → (zone, share of peers it is cheaper than, number of peers with a value).
    Example: own 5%, peers 1,2,3,4,6,7% → cheaper than 4 of 6 = 0.67 → "φθηνή".
    """
    peers = pd.Series(peer_yields, dtype=float).dropna()
    if own_yield is None or pd.isna(own_yield) or len(peers) < MIN_PEERS - 1:
        return None, None, len(peers)
    below = float((peers < own_yield).sum()) + 0.5 * float((peers == own_yield).sum())
    share = below / len(peers)
    if share >= CHEAP_SHARE:
        return "φθηνή", share, len(peers)
    if share <= 1 - CHEAP_SHARE:
        return "ακριβή", share, len(peers)
    return "στο μέσο", share, len(peers)
