"""valuation/universe.py — dead tickers out of the weekly peer table (invented SEC list)."""

import pandas as pd

from valuation import universe as U

OURS = pd.DataFrame([
    ("AAPL", 320193, "Apple Inc."),               # ok
    ("BK", 1390777, "Bank of New York Mellon"),   # new ticker BNY, and BNY is ours too → out
    ("BNY", 1390777, "Bank of New York Mellon"),
    ("FBHS", 1519751, "Fortune Brands"),          # new ticker FBIN, NOT ours → kept
    ("XOM", 34088, "EXXON MOBIL CORP"),           # same ticker, new CIK, same name → kept
    ("PARA", 813828, "Paramount Global"),         # ticker recycled → out
    ("CTRA", 858470, "Coterra Energy Inc."),      # absorbed → out
    ("BRK.B", 1067983, "Berkshire Hathaway"),     # dot vs dash → ok
], columns=["ticker", "cik", "title"])
SEC = pd.DataFrame([
    ("AAPL", 320193, "Apple Inc."), ("BNY", 1390777, "Bank of New York Mellon Corp"),
    ("FBIN", 1519751, "Fortune Brands Innovations"), ("XOM", 2115436, "ExxonMobil Holdings Corp"),
    ("PARA", 1826011, "Parabolic Technologies, Inc."), ("BRK-B", 1067983, "BERKSHIRE HATHAWAY INC"),
], columns=["ticker", "cik", "title"])
ALL = OURS["ticker"].tolist()


def test_names():
    assert U.name_words("Exxon Mobil Corp /NJ/") == ["EXXON", "MOBIL"]
    assert U.similar("EXXON MOBIL CORP", "ExxonMobil Holdings Corp")
    assert not U.similar("Paramount Global", "Parabolic Technologies, Inc.")
    assert not U.similar("", "Apple Inc")


def test_classify():
    c = U.classify(OURS, SEC).set_index("ticker")
    assert c["status"].to_dict() == {
        "AAPL": U.OK, "BK": U.NEW_TICKER, "BNY": U.OK, "FBHS": U.NEW_TICKER, "XOM": U.NEW_CIK,
        "PARA": U.RECYCLED, "CTRA": U.GONE, "BRK.B": U.OK}
    assert c.loc["BK", "also_ours"] == "BNY" and c.loc["FBHS", "also_ours"] == ""
    assert c.loc["XOM", "sec_cik"] == 2115436


def test_live_tickers_drops_only_the_dead():
    keep, drop, msg = U.live_tickers(ALL, OURS, SEC, max_drop_share=0.5)
    assert keep == ["AAPL", "BNY", "FBHS", "XOM", "BRK.B"]          # order kept
    assert sorted(drop["ticker"]) == ["BK", "CTRA", "PARA"]
    assert msg.startswith("Εκτός: 3")


def test_no_sec_list_keeps_everything():
    keep, drop, msg = U.live_tickers(ALL, OURS, None)
    assert keep == ALL and drop.empty and "χωρίς λίστα SEC" in msg
    assert list(drop.columns)[:2] == ["ticker", "cik"]               # the job can still write excluded.csv


def test_implausible_drop_keeps_everything():
    keep, drop, msg = U.live_tickers(ALL, OURS, SEC.iloc[:1], max_drop_share=0.2)  # SEC list cut short
    assert keep == ALL and drop.empty and "κρατήθηκαν όλες" in msg


def test_only_the_given_tickers_are_judged():
    keep, drop, _ = U.live_tickers(["AAPL", "CTRA"], OURS, SEC, max_drop_share=0.5)
    assert keep == ["AAPL"] and drop["ticker"].tolist() == ["CTRA"]
