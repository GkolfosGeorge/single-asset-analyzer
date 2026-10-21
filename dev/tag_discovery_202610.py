"""
dev/tag_discovery_202610.py — READ-ONLY: which XBRL tags does each company
actually use? Nothing is written to the database.

Two modes (run from the repo root, in Git Bash):

  1. Survey the whole universe for candidate tags (~20–30 min, resumable):
         python dev/tag_discovery_202610.py
     → data/tag_discovery_202610.csv   (one row per ticker × tag found)
     → printed summary: companies gained per group, rows to add, MB estimate

  2. Find every tag of ONE company whose name matches a word:
         python dev/tag_discovery_202610.py --find KO Dividend
     → years covered per tag, so you see which tag fills a gap

Why this exists: before adding tags to the ingest we want NUMBERS — how
many companies gain capex, how many rows (= MB on Neon's 512 MB free tier)
each tag costs — instead of guessing tag names.
"""

import os
import sys
import time

sys.path.insert(0, os.getcwd())

import pandas as pd
from sqlalchemy import text

from db import get_engine
from fundamentals import fetch_companyfacts

OUT = "data/tag_discovery_202610.csv"
SLEEP = 0.15
FORMS = {"10-Q", "10-K"}

CANDIDATES = {
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",        # already ingested
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
        "PaymentsToAcquireOilAndGasPropertyAndEquipment",
        "PaymentsToAcquireOilAndGasProperty",
        "PaymentsToAcquireRealEstate",
        "PaymentsToDevelopRealEstateAssets",
        "PaymentsToAcquireAndDevelopRealEstate",
        "PaymentsForConstructionInProcess",
    ],
    "finance_lease": [
        "FinanceLeaseLiability",
        "FinanceLeaseLiabilityCurrent",
        "FinanceLeaseLiabilityNoncurrent",
        "CapitalLeaseObligations",
        "CapitalLeaseObligationsCurrent",
        "CapitalLeaseObligationsNoncurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "CommercialPaper",
        "OtherShortTermBorrowings",
    ],
    "operating_lease": [            # information only: not planned for EV yet
        "OperatingLeaseLiability",
        "OperatingLeaseLiabilityCurrent",
        "OperatingLeaseLiabilityNoncurrent",
    ],
    "dividends": [
        "PaymentsOfDividends",                               # already ingested
        "PaymentsOfDividendsCommonStock",                    # already ingested
        "PaymentsOfOrdinaryDividends",
        "DividendsCommonStockCash",
        "DividendsCash",
    ],
}
ALREADY_INGESTED = {"PaymentsToAcquirePropertyPlantAndEquipment",
                    "PaymentsOfDividends", "PaymentsOfDividendsCommonStock"}


def ticker_ciks(engine) -> pd.DataFrame:
    """Every ticker in the DB with its most recent CIK (works for delisted ones too)."""
    sql = text("""
        SELECT DISTINCT ON (ticker) ticker, cik
        FROM fundamentals
        ORDER BY ticker, filed_date DESC
    """)
    with engine.connect() as c:
        return pd.DataFrame(c.execute(sql).fetchall(), columns=["ticker", "cik"])


def summarize_tag(observations: list) -> dict | None:
    rows = [o for o in observations if o.get("form") in FORMS and o.get("end")]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    has_start = df["start"].notna() if "start" in df else pd.Series(False, index=df.index)
    ends = pd.to_datetime(df["end"])
    # the same period repeats as a comparative in later filings: count periods, not observations
    n_periods = len({(o.get("start") or "", o["end"]) for o in rows})
    return {"n_obs": len(df), "n_periods": n_periods,
            "kind": "duration" if has_start.mean() > 0.5 else "instant",
            "first_end": ends.min().date(), "last_end": ends.max().date()}


def survey():
    eng = get_engine()
    tc = ticker_ciks(eng)
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["ticker"])
    todo = tc[~tc["ticker"].isin(done)]
    print(f"{len(tc)} tickers, {len(done)} already surveyed, {len(todo)} to go\n")

    t0 = time.time()
    for i, (t, cik) in enumerate(todo.itertuples(index=False), 1):
        recs = []
        try:
            facts = fetch_companyfacts(int(cik))
            gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
            for group, tags in CANDIDATES.items():
                for tag in tags:
                    concept = gaap.get(tag)
                    if not concept:
                        continue
                    obs = [o for unit in concept.get("units", {}).values() for o in unit]
                    s = summarize_tag(obs)
                    if s:
                        recs.append({"ticker": t, "group": group, "tag": tag, **s})
            if not recs:
                recs.append({"ticker": t, "group": "_none", "tag": "", "n_obs": 0, "n_periods": 0})
        except Exception as e:
            recs.append({"ticker": t, "group": "_error", "tag": str(e)[:120], "n_obs": 0, "n_periods": 0})
        pd.DataFrame(recs).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        time.sleep(SLEEP)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  ... {i}/{len(todo)}  ~{el / i * (len(todo) - i) / 60:.0f} min left")
    report(eng)


def report(eng):
    d = pd.read_csv(OUT)
    errors = d[d["group"] == "_error"]
    d = d[~d["group"].isin(["_none", "_error"])]
    n_tickers = pd.read_csv(OUT)["ticker"].nunique()

    with eng.connect() as c:
        size, rows = c.execute(text(
            "SELECT pg_total_relation_size('fundamentals'), COUNT(*) FROM fundamentals")).fetchone()
        db_size = c.execute(text("SELECT pg_database_size(current_database())")).scalar()
        have = pd.DataFrame(c.execute(text("""
            SELECT DISTINCT f.ticker, m.name FROM fundamentals f
            JOIN metric_lookup m ON m.id = f.metric_id
            WHERE m.name IN ('PaymentsToAcquirePropertyPlantAndEquipment',
                             'NetCashProvidedByUsedInOperatingActivities')
        """)).fetchall(), columns=["ticker", "metric"])
    bytes_per_row = size / rows
    print(f"── Βάση: {db_size / 1e6:.0f} MB συνολικά, fundamentals {size / 1e6:.0f} MB / {rows:,} γραμμές"
          f" = {bytes_per_row:.0f} bytes/γραμμή (με τα indexes) ──\n")
    print(f"Ερευνήθηκαν {n_tickers} tickers ({len(errors)} σφάλματα λήψης)\n")

    # 1. capex: who lacks it today, and which alternative tag would fill it
    with_cfo = set(have.loc[have["metric"] == "NetCashProvidedByUsedInOperatingActivities", "ticker"])
    with_capex = set(have.loc[have["metric"] == "PaymentsToAcquirePropertyPlantAndEquipment", "ticker"])
    lacking = with_cfo - with_capex
    alt = d[(d["group"] == "capex") & ~d["tag"].isin(ALREADY_INGESTED) & d["ticker"].isin(lacking)]
    gained = alt.groupby("ticker")["tag"].apply(list)
    print(f"── CAPEX: {len(lacking)} εταιρείες έχουν CFO αλλά όχι capex στη βάση ──")
    print(f"   με κάποιο εναλλακτικό tag: {len(gained)}   χωρίς κανένα: {len(lacking) - len(gained)}")
    print(alt.groupby("tag").agg(εταιρείες=("ticker", "nunique"), περίοδοι=("n_periods", "sum"))
          .sort_values("εταιρείες", ascending=False).to_string())
    multi = gained[gained.map(len) > 1]
    if len(multi):
        print(f"   ⚠️ {len(multi)} εταιρείες έχουν ΠΕΡΙΣΣΟΤΕΡΑ από ένα εναλλακτικά tags, π.χ.:")
        print("   " + "; ".join(f"{t}: {', '.join(v)}" for t, v in multi.head(5).items()))
    nothing = sorted(lacking - set(gained.index))
    print(f"   χωρίς κανένα tag (π.χ. τράπεζες/ασφαλιστικές): {', '.join(nothing[:25])}"
          f"{' ...' if len(nothing) > 25 else ''}\n")

    # 2. every candidate tag not yet ingested: coverage and cost
    new = d[~d["tag"].isin(ALREADY_INGESTED)]
    cost = (new.groupby(["group", "tag"])
            .agg(εταιρείες=("ticker", "nunique"), περίοδοι=("n_periods", "sum"))
            .reset_index())
    # each period becomes ~1 row per form (10-Q or 10-K) after dedup; periods is the right order of magnitude
    cost["MB_περίπου"] = (cost["περίοδοι"] * bytes_per_row / 1e6).round(1)
    print("── Κόστος ανά tag που ΔΕΝ έχουμε ακόμα (περίοδοι ≈ γραμμές που θα μπουν) ──")
    print(cost.sort_values(["group", "εταιρείες"], ascending=[True, False]).to_string(index=False))
    for g, sub in cost.groupby("group"):
        print(f"   {g:16s} σύνολο ≈ {sub['MB_περίπου'].sum():5.1f} MB")


def find(ticker: str, word: str):
    eng = get_engine()
    tc = ticker_ciks(eng)
    row = tc[tc["ticker"] == ticker]
    if row.empty:
        print(f"{ticker}: δεν υπάρχει στη βάση")
        return
    facts = fetch_companyfacts(int(row["cik"].iloc[0])) or {}
    out = []
    for tax, concepts in facts.get("facts", {}).items():
        for tag, concept in concepts.items():
            if word.lower() not in tag.lower():
                continue
            obs = [o for unit in concept.get("units", {}).values() for o in unit]
            s = summarize_tag(obs)
            if s:
                years = sorted({pd.Timestamp(o["end"]).year for o in obs if o.get("form") in FORMS})
                out.append({"taxonomy": tax, "tag": tag, "kind": s["kind"], "periods": s["n_periods"],
                            "years": f"{years[0]}–{years[-1]}" if years else "",
                            "missing_years": ",".join(str(y) for y in range(years[0], years[-1] + 1)
                                                      if y not in years) if years else ""})
    if not out:
        print(f"{ticker}: κανένα tag με «{word}»")
        return
    print(f"── {ticker}: tags που περιέχουν «{word}» (μόνο 10-Q/10-K) ──")
    print(pd.DataFrame(out).sort_values(["taxonomy", "tag"]).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--find":
        find(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--report":
        report(get_engine())
    else:
        survey()
