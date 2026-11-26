"""
dev/ticker_map_audit_202611.py — READ-ONLY: do our tickers still point to the
right company? Compares the `companies` table (ticker → CIK) with TODAY's SEC
list (company_tickers.json: one row per ticker that trades today).
Nothing is written to the database.

Why (17/11/2026): dev/missing_yields_202611.py found 32 companies with no price and
15 that "stopped filing" while still having a price. Checked by hand:
    BK   → same company, new ticker BNY (21/05/2026)
    CTRA → absorbed by DVN (07/05/2026)
    AVB + EQR → merged into VMRK (17/08/2026)
The fundamentals are stored by TICKER, so a company that changes ticker or CIK
silently drops out of the card.

Each company of our universe gets one status:
    "ok"                         the SEC lists our ticker with our CIK
    "νέο σύμβολο (ίδιο CIK)"     our CIK trades today under another ticker
                                 (BK → BNY). Fix: rename / map the ticker.
    "αλλαγή CIK (ίδιο όνομα)"    the SEC lists our ticker, but with another CIK
                                 and a similar name (a re-registration, e.g. a
                                 move to another state). Fix: read the new CIK too.
    "σύμβολο σε άλλη εταιρεία"   the SEC lists our ticker with another CIK and a
                                 different name: the ticker was recycled. The price
                                 we download belongs to someone else → exclude.
    "εκτός λίστας SEC"           neither our ticker nor our CIK trade today
                                 (acquired, merged, delisted) → exclude.
Example: ours BLL, CIK 9389. The SEC has CIK 9389 under BALL → "νέο σύμβολο";
if BALL is ALSO in our universe, BLL is just the old history of the same company.

Names are compared loosely (upper case, without INC, CORP, /DE/, punctuation...):
"EXXON MOBIL CORP" ~ "Exxon Mobil Corporation". A lead, not a proof.

    python dev/ticker_map_audit_202611.py --selftest   (1 sec, no network, no database)
    python dev/ticker_map_audit_202611.py              (~10 sec)
    python dev/ticker_map_audit_202611.py --report     (reprints the summary from the CSV)

→ data/ticker_map_audit_202611.csv
"""

import os
import re
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

OUT = "data/ticker_map_audit_202611.csv"
MISSING = "data/missing_yields_202611.csv"      # from dev/missing_yields_202611.py, optional
TABLE = "peer_data/peer_yields.csv"
OK, NEW_TICKER, NEW_CIK, RECYCLED, GONE = (
    "ok", "νέο σύμβολο (ίδιο CIK)", "αλλαγή CIK (ίδιο όνομα)",
    "σύμβολο σε άλλη εταιρεία", "εκτός λίστας SEC")
ORDER = [NEW_TICKER, NEW_CIK, RECYCLED, GONE, OK]
NOISE = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "THE", "HOLDINGS",
         "HOLDING", "GROUP", "NV", "SA", "AG", "LLC", "LP", "TRUST", "INCORPORATED", "DE", "NEW", "CL", "A"}
EXAMPLES = 25


# ═════════════════════════════════════════════════════════════════════════
# 1. PURE LOGIC (covered by --selftest)
# ═════════════════════════════════════════════════════════════════════════

def norm_ticker(t: str) -> str:
    return str(t).upper().replace(".", "-").strip()


def name_words(title) -> list[str]:
    """'Exxon Mobil Corp /NJ/' → ['EXXON', 'MOBIL']"""
    words = re.sub(r"/[A-Z]{2}/", " ", str(title or "").upper())
    words = re.sub(r"[^A-Z0-9 ]", " ", words).split()
    return [w for w in words if w not in NOISE]


def similar(a, b) -> bool:
    """Loose name match: same first word (or one glued: EXXONMOBIL), or half the words in common."""
    wa, wb = name_words(a), name_words(b)
    if not wa or not wb:
        return False
    ja, jb = "".join(wa), "".join(wb)
    if wa[0] == wb[0] or ja.startswith(wb[0]) or jb.startswith(wa[0]):
        return True
    common = set(wa) & set(wb)
    return len(common) >= max(1, min(len(wa), len(wb)) / 2)


def classify(ours: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """
    ours: ticker, cik, title (our universe) · sec: ticker, cik, title (SEC today).
    Returns ours + status, sec_tickers (all tickers of our CIK at the SEC),
    sec_cik / sec_title (who owns our ticker at the SEC), also_ours.
    """
    sec = sec.assign(ticker=sec["ticker"].map(norm_ticker))
    by_ticker = sec.drop_duplicates("ticker").set_index("ticker")
    by_cik = sec.groupby("cik")["ticker"].apply(list).to_dict()
    title_of_cik = sec.drop_duplicates("cik").set_index("cik")["title"].to_dict()
    our_tickers = set(ours["ticker"].map(norm_ticker))

    out = []
    for _, r in ours.iterrows():
        t, cik = norm_ticker(r["ticker"]), (int(r["cik"]) if pd.notna(r["cik"]) else None)
        sec_tickers = by_cik.get(cik, [])
        owner = by_ticker.loc[t] if t in by_ticker.index else None
        rec = {"ticker": r["ticker"], "cik": cik, "title": r["title"],
               "sec_tickers": " ".join(sec_tickers), "sec_cik": None, "sec_title": None,
               "also_ours": " ".join(x for x in sec_tickers if x != t and x in our_tickers)}
        if owner is not None:
            rec["sec_cik"], rec["sec_title"] = int(owner["cik"]), owner["title"]
        if owner is not None and int(owner["cik"]) == cik:
            rec["status"] = OK
        elif sec_tickers:
            rec["status"] = NEW_TICKER
            rec["sec_title"] = rec["sec_title"] or title_of_cik.get(cik)
        elif owner is not None:
            rec["status"] = NEW_CIK if similar(r["title"], owner["title"]) else RECYCLED
        else:
            rec["status"] = GONE
        out.append(rec)
    return pd.DataFrame(out)


# ═════════════════════════════════════════════════════════════════════════
# 2. DATA ACCESS
# ═════════════════════════════════════════════════════════════════════════

def load_ours(engine) -> pd.DataFrame:
    from sqlalchemy import text
    sql = text("""
        SELECT c.ticker, c.cik, c.title FROM companies c
        WHERE EXISTS (SELECT 1 FROM fundamentals f WHERE f.ticker = c.ticker)
        ORDER BY c.ticker
    """)
    with engine.connect() as conn:
        return pd.DataFrame(conn.execute(sql).fetchall(), columns=["ticker", "cik", "title"])


def load_sec() -> pd.DataFrame:
    from ticker_provider import SEC_HEADERS, SEC_TICKERS_URL, _download_with_retry
    raw = _download_with_retry(SEC_TICKERS_URL, headers=SEC_HEADERS).json()
    return pd.DataFrame([{"ticker": v["ticker"], "cik": int(v["cik_str"]), "title": v["title"]}
                         for v in raw.values()])


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Joins what the earlier scripts know: missing yields, still has a price."""
    if os.path.exists(TABLE):
        t = pd.read_csv(TABLE)
        y = t.set_index("ticker")[["earnings_yield", "ebit_yield", "fcf_yield"]]
        df["yields_in_table"] = df["ticker"].map(y.notna().sum(axis=1)).fillna(0).astype(int)
    if os.path.exists(MISSING):
        m = pd.read_csv(MISSING).set_index("ticker")
        df["has_price"] = df["ticker"].map(m["has_price"])
        df["last_period"] = df["ticker"].map(m["last_period"])
    return df


# ═════════════════════════════════════════════════════════════════════════
# 3. REPORT
# ═════════════════════════════════════════════════════════════════════════

def report(df: pd.DataFrame) -> None:
    print(f"Universe: {len(df)} εταιρείες\n")
    for st in ORDER:
        sub = df[df["status"] == st]
        print(f"══ {st}: {len(sub)} ══")
        if st == OK or sub.empty:
            print()
            continue
        for _, r in sub.head(EXAMPLES if st == GONE else len(sub)).iterrows():
            y = f" · αποδόσεις {int(r['yields_in_table'])}/3" if "yields_in_table" in r and pd.notna(
                r.get("yields_in_table")) else ""
            if st == NEW_TICKER:
                extra = f"→ SEC: {r['sec_tickers']}" + (f" · ΗΔΗ στο universe: {r['also_ours']}"
                                                        if isinstance(r["also_ours"], str) and r["also_ours"] else "")
            elif st in (NEW_CIK, RECYCLED):
                extra = (f"→ SEC: CIK {int(r['sec_cik'])} «{r['sec_title']}»"
                         f" (δικό μας CIK {int(r['cik']) if pd.notna(r['cik']) else '—'})")
            else:
                extra = ""
            print(f"  {r['ticker']:7s} «{str(r['title'])[:38]}» {extra}{y}")
        if st == GONE and len(sub) > EXAMPLES:
            print(f"  … +{len(sub) - EXAMPLES}: {', '.join(sub['ticker'][EXAMPLES:])}")
        print()

    if "yields_in_table" in df.columns:
        bad = df[(df["status"] != OK) & (df["yields_in_table"] > 0)]
        if len(bad):
            print(f"⚠️ Μη-ok αλλά ΜΕ αποδόσεις στον πίνακα κλάδου (πιθανώς τιμή άλλης εταιρείας ή"
                  f" παλιά δεδομένα): {len(bad)} · {', '.join(bad['ticker'])}")
    print(f"\n→ {OUT}")


# ═════════════════════════════════════════════════════════════════════════
# 4. SELFTEST
# ═════════════════════════════════════════════════════════════════════════

def selftest() -> None:
    assert name_words("Exxon Mobil Corp /NJ/") == ["EXXON", "MOBIL"]
    assert similar("EXXON MOBIL CORP", "Exxon Mobil Corporation")
    assert similar("EXXON MOBIL CORP", "ExxonMobil Corp")
    assert not similar("COMPUWARE CORP", "Ocean Power Inc")
    assert not similar("", "Apple Inc")
    assert norm_ticker("brk.b") == "BRK-B"

    ours = pd.DataFrame([
        ("AAPL", 320193, "Apple Inc."),           # ok
        ("BK", 1390777, "Bank of New York Mellon"),  # ticker changed, same CIK
        ("BLL", 9389, "BALL Corp"),               # renamed AND the new ticker is already ours
        ("BALL", 9389, "BALL Corp"),
        ("XOM", 34088, "EXXON MOBIL CORP"),       # same ticker, new CIK, same name
        ("CPWR", 859014, "COMPUWARE CORP"),       # ticker recycled
        ("CTRA", 858470, "Coterra Energy Inc."),  # absorbed
        ("BRK.B", 1067983, "Berkshire Hathaway"),  # dot vs dash
    ], columns=["ticker", "cik", "title"])
    sec = pd.DataFrame([
        ("AAPL", 320193, "Apple Inc."), ("BNY", 1390777, "Bank of New York Mellon Corp"),
        ("BALL", 9389, "BALL Corp"), ("XOM", 2100001, "Exxon Mobil Corporation"),
        ("CPWR", 2000002, "Ocean Power Inc"), ("BRK-B", 1067983, "BERKSHIRE HATHAWAY INC"),
    ], columns=["ticker", "cik", "title"])
    r = classify(ours, sec).set_index("ticker")["status"].to_dict()
    assert r == {"AAPL": OK, "BK": NEW_TICKER, "BLL": NEW_TICKER, "BALL": OK, "XOM": NEW_CIK,
                 "CPWR": RECYCLED, "CTRA": GONE, "BRK.B": OK}, r
    c = classify(ours, sec).set_index("ticker")
    assert c.loc["BK", "sec_tickers"] == "BNY" and c.loc["BLL", "also_ours"] == "BALL"
    assert c.loc["XOM", "sec_cik"] == 2100001
    print("selftest: όλα τα τεστ πέρασαν ✓")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--report" in sys.argv:
        report(pd.read_csv(OUT))
    else:
        from db import get_engine
        ours = load_ours(get_engine())
        sec = load_sec()
        print(f"SEC σήμερα: {len(sec):,} σύμβολα · δική μας βάση: {len(ours)} εταιρείες")
        df = add_context(classify(ours, sec))
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUT, index=False)
        report(df)
