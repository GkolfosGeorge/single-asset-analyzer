"""
valuation/universe.py — which tickers of our database are still a LIVE company
under that ticker (Nov 2026). Used by the weekly peer table (jobs/build_peer_table.py).

Our fundamentals are stored by TICKER. Companies merge, get acquired or change
ticker, and the old ticker stays in our database forever. Measured on 17/11/2026
(dev/ticker_map_audit_202611.py): 72 of 777 tickers are dead:
    53 not on the SEC list at all       (CTRA → absorbed by DVN, AVB, EA, HES...)
    18 old tickers of a company we ALSO have under its new ticker
                                        (BK → BNY, BLL → BALL, FB → META...)
     1 ticker now used by another company (PARA → Parabolic Technologies)
They can never get a yield, and they made peer groups look bigger than they are
(an industry of 8 with 3 dead ones is really an industry of 5).

TODAY's SEC list (company_tickers.json: one row per traded ticker) decides:
    "ok"                         our ticker, our CIK                     → kept
    "νέο σύμβολο (ίδιο CIK)"     our CIK trades under another ticker:
                                   new ticker also in our universe        → dropped (old history)
                                   new ticker NOT in our universe         → kept (until it is ingested)
    "αλλαγή CIK (ίδιο όνομα)"    same ticker, new CIK, similar name (XOM) → kept
    "σύμβολο σε άλλη εταιρεία"   same ticker, new CIK, other name (PARA)  → dropped
    "εκτός λίστας SEC"           neither ticker nor CIK trade today        → dropped
Automatic on purpose: three 2026 mergers (CTRA, AVB/EQR, BK) would each have
needed a hand-made list.

Safety: if the SEC list can't be downloaded, or more than MAX_DROP_SHARE of the
universe would be dropped (a format change, a bad download), NOTHING is dropped
and the table is built as before.
"""

import re

import pandas as pd

OK, NEW_TICKER, NEW_CIK, RECYCLED, GONE = (
    "ok", "νέο σύμβολο (ίδιο CIK)", "αλλαγή CIK (ίδιο όνομα)",
    "σύμβολο σε άλλη εταιρεία", "εκτός λίστας SEC")
MAX_DROP_SHARE = 0.20
NOISE = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "THE", "HOLDINGS",
         "HOLDING", "GROUP", "NV", "SA", "AG", "LLC", "LP", "TRUST", "INCORPORATED", "DE", "NEW", "CL", "A"}


def norm_ticker(t: str) -> str:
    return str(t).upper().replace(".", "-").strip()


def name_words(title) -> list[str]:
    """'Exxon Mobil Corp /NJ/' → ['EXXON', 'MOBIL']"""
    words = re.sub(r"/[A-Z]{2}/", " ", str(title or "").upper())
    words = re.sub(r"[^A-Z0-9 ]", " ", words).split()
    return [w for w in words if w not in NOISE]


def similar(a, b) -> bool:
    """Loose name match: same first word (or glued: EXXONMOBIL), or half the words in common."""
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
    → ours + status, sec_tickers (tickers of our CIK at the SEC), sec_cik / sec_title
      (who owns our ticker at the SEC), also_ours (new tickers we already have).
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
    return pd.DataFrame(out, columns=["ticker", "cik", "title", "sec_tickers", "sec_cik", "sec_title",
                                      "also_ours", "status"])


def dead(c: pd.DataFrame) -> pd.Series:
    """Rows of classify() that are dropped (see the module docstring)."""
    old_history = (c["status"] == NEW_TICKER) & c["also_ours"].fillna("").ne("")
    return c["status"].isin([GONE, RECYCLED]) | old_history


def live_tickers(tickers: list[str], ours: pd.DataFrame, sec: pd.DataFrame | None,
                 max_drop_share: float = MAX_DROP_SHARE) -> tuple[list[str], pd.DataFrame, str]:
    """
    → (tickers to keep, dropped rows of classify(), message).
    Keeps everything when the SEC list is missing or the drop looks implausible.
    """
    if sec is None or sec.empty:
        return list(tickers), classify(ours.iloc[0:0], ours.iloc[0:0]), \
            "⚠️ χωρίς λίστα SEC: κρατήθηκαν όλες οι εταιρείες"
    c = classify(ours[ours["ticker"].isin(tickers)], sec)
    drop = c[dead(c)]
    if len(drop) > max_drop_share * len(tickers):
        return list(tickers), drop.iloc[0:0], (f"⚠️ θα έφευγαν {len(drop)}/{len(tickers)} (> "
                                                f"{max_drop_share:.0%}): κρατήθηκαν όλες, έλεγξε τη λίστα SEC")
    gone = set(drop["ticker"])
    counts = drop["status"].where(drop["status"] != NEW_TICKER, "παλιό σύμβολο (το νέο υπάρχει ήδη)")
    msg = f"Εκτός: {len(drop)} νεκρά σύμβολα · " + " · ".join(f"{k} {v}" for k, v in counts.value_counts().items())
    return [t for t in tickers if t not in gone], drop, msg


# ── data access (database, SEC) ───────────────────────────────────────────

def load_companies(engine) -> pd.DataFrame:
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker, cik, title FROM companies")).fetchall()
    return pd.DataFrame(rows, columns=["ticker", "cik", "title"])


def load_sec() -> pd.DataFrame | None:
    """Today's SEC ticker list, or None if it can't be downloaded."""
    try:
        from ticker_provider import SEC_HEADERS, SEC_TICKERS_URL, _download_with_retry
        raw = _download_with_retry(SEC_TICKERS_URL, headers=SEC_HEADERS).json()
        return pd.DataFrame([{"ticker": v["ticker"], "cik": int(v["cik_str"]), "title": v["title"]}
                             for v in raw.values()])
    except Exception as e:
        print(f"⚠️ λίστα SEC: {type(e).__name__}: {e}")
        return None
