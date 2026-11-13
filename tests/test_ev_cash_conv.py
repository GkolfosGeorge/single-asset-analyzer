"""Cash securities and convertible debt in ev.py (Nov 2026, trading-shared-data v0.1.26).
Invented companies; the numbers mirror the real cases named in each test."""

import pandas as pd

from valuation import ev as E

C, STI, COMB = E.CASH, E.STI, E.COMBINED
MSC, AFS = "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"
OSTI = "OtherShortTermInvestments"
LTD, NC, DC = "LongTermDebt", "LongTermDebtNoncurrent", "DebtCurrent"
CLT, CNC = "ConvertibleLongTermNotesPayable", "ConvertibleNotesPayableCurrent"


def rows(table: dict, filed_lag: dict | None = None) -> pd.DataFrame:
    """{period_end: {metric: value}} → rows as load_metrics returns them (balance sheet: no start)."""
    out = []
    for end, vals in table.items():
        for m, v in vals.items():
            lag = (filed_lag or {}).get(m, 35)
            out.append({"ticker": "X", "value": float(v), "period_start": pd.NaT,
                        "period_end": pd.Timestamp(end),
                        "filed_date": pd.Timestamp(end) + pd.Timedelta(days=lag),
                        "form_type": "10-Q", "metric": m})
    return pd.DataFrame(out)


def cash(table, **kw):
    return E.cash_series(rows(table, **kw)).set_index("period_end")["cash"].to_dict()


def debt(table):
    d = E.debt_series(rows(table))
    return {k.date().isoformat(): (v, m) for k, v, m in zip(d["period_end"], d["debt"], d["method"])}


T = pd.Timestamp


# ── cash ────────────────────────────────────────────────────────────────

def test_old_tags_give_the_old_result():
    # a company with only cash + ShortTermInvestments: same as before Nov 2026...
    assert cash({"2025-03-31": {C: 2.0, STI: 1.0}, "2025-06-30": {C: 3.0, STI: 0.5}}) == \
        {T("2025-03-31"): 3.0, T("2025-06-30"): 3.5}
    # ...except the gap rule, which applies to ShortTermInvestments too:
    # before, 2025-06-30 was cash alone (3.0); now the 3.0 of March stays one quarter
    assert cash({"2025-03-31": {C: 2.0, STI: 1.0}, "2025-06-30": {C: 3.0}}) == {T("2025-03-31"): 3.0}


def test_largest_current_tag_not_the_sum():
    # PFE 2026-06: debt securities 7.28 are PART of short-term investments 10.72
    assert cash({"2026-06-28": {C: 0.98, AFS: 7.28, OSTI: 10.72}}) == {T("2026-06-28"): 0.98 + 10.72}


def test_securities_inside_cash_are_not_added():
    # TGT 2019: the company's total equals cash alone
    assert cash({"2019-02-02": {C: 1.556, STI: 0.769, COMB: 1.556}}) == {T("2019-02-02"): 1.556}
    # ...while a total equal to cash + securities changes nothing
    assert cash({"2025-03-31": {C: 2.0, MSC: 3.0, COMB: 5.0}}) == {T("2025-03-31"): 5.0}


def test_no_cash_line_uses_the_company_total_alone():
    assert cash({"2025-03-31": {COMB: 5.0, MSC: 3.0}}) == {T("2025-03-31"): 5.0}


def test_missing_securities_for_one_date_keep_the_previous_value():
    # NVDA: securities in the 10-Qs, not tagged in the 10-K of 2026-01
    t = {"2025-10-26": {C: 11.0, MSC: 49.0}, "2026-01-25": {C: 10.0}, "2026-04-26": {C: 20.0, MSC: 30.0}}
    assert cash(t) == {T("2025-10-26"): 60.0, T("2026-04-26"): 50.0}


def test_securities_sold_for_good_cost_one_quarter_at_most():
    t = {"2025-03-31": {C: 2.0, MSC: 3.0}, "2025-06-30": {C: 5.0}, "2025-09-30": {C: 5.0},
         "2025-12-31": {C: 5.0}}
    assert cash(t) == {T("2025-03-31"): 5.0, T("2025-09-30"): 5.0, T("2025-12-31"): 5.0}


def test_a_stated_zero_is_not_a_gap():
    t = {"2025-03-31": {C: 2.0, MSC: 3.0}, "2025-06-30": {C: 5.0, MSC: 0.0}}
    assert cash(t) == {T("2025-03-31"): 5.0, T("2025-06-30"): 5.0}


def test_quarters_far_apart_are_not_a_gap():
    t = {"2024-03-31": {C: 2.0, MSC: 3.0}, "2025-03-31": {C: 5.0}}
    assert cash(t) == {T("2024-03-31"): 5.0, T("2025-03-31"): 5.0}


def test_available_waits_for_the_later_filing():
    # cash filed after 35 days, the securities tag only after 60: known after 60
    d = E.cash_series(rows({"2025-03-31": {C: 2.0, MSC: 3.0}}, filed_lag={MSC: 60}))
    assert d["available"].iloc[0] == T("2025-03-31") + pd.Timedelta(days=61)


# ── debt ────────────────────────────────────────────────────────────────

def test_ordinary_debt_rule_is_unchanged():
    # HD-like, from the module docstring
    assert debt({"2025-03-31": {NC: 40.0, DC: 3.0, LTD: 43.0}}) == \
        {"2025-03-31": (43.0, "LongTermDebtNoncurrent + DebtCurrent")}


def test_convertibles_alone_when_there_is_no_ordinary_debt():
    # DXCM / DDOG: the convertibles are all their debt
    r = debt({"2025-06-30": {CLT: 1.24, CNC: 0.5}, "2025-09-30": {CLT: 1.24, CNC: 0.0}})
    assert [v for v, _ in r.values()] == [1.74, 1.24]
    assert "μόνο μετατρέψιμα" in r["2025-09-30"][1]


def test_convertibles_are_not_added_to_ordinary_debt():
    # VIAV / CRM: inside as often as outside → ordinary debt alone, as before
    assert debt({"2025-06-30": {LTD: 5.0, CLT: 1.0}}) == {"2025-06-30": (5.0, "LongTermDebt")}


def test_old_ordinary_debt_without_convertibles_blocks_them():
    # BXP: ordinary debt 3.30 years ago and no convertibles then; today only 0.98
    # of convertibles while the real debt (~15.6) sits in tags we don't read
    r = debt({"2019-06-30": {LTD: 3.30}, "2026-03-31": {"ConvertibleDebt": 0.98}})
    assert list(r) == ["2019-06-30"]


def test_old_ordinary_debt_that_was_the_convertibles_allows_them():
    # PANW: its LongTermDebt used to be the convertibles themselves
    r = debt({"2023-07-31": {LTD: 1.99, "ConvertibleDebtNoncurrent": 1.99},
              "2026-07-31": {"ConvertibleDebtNoncurrent": 1.77}})
    assert r["2026-07-31"][0] == 1.77


def test_ordinary_debt_well_above_the_convertibles_blocks_them():
    r = debt({"2023-06-30": {LTD: 5.0, CLT: 1.0}, "2025-06-30": {CLT: 1.0}})
    assert list(r) == ["2023-06-30"]


def test_a_stated_zero_of_convertibles_is_not_zero_debt():
    # EXR 2020: ConvertibleDebt = 0 must not turn a REIT's debt into 0
    assert debt({"2020-12-31": {"ConvertibleDebt": 0.0}}) == {}


def test_only_the_past_counts_for_the_guard():
    # ordinary debt appearing LATER must not block an earlier date (no look-ahead)
    r = debt({"2024-06-30": {CLT: 1.0}, "2025-06-30": {LTD: 5.0}})
    assert r["2024-06-30"][0] == 1.0 and r["2025-06-30"][0] == 5.0


def test_convertible_parts_beat_the_total():
    assert E.convertibles(pd.Series({CLT: 2.0, CNC: 1.0, "ConvertibleDebt": 3.0}))[0] == 3.0
    assert E.convertibles(pd.Series({"ConvertibleDebt": 3.0}))[0] == 3.0
    assert E.convertibles(pd.Series({LTD: 3.0}))[0] is None


# ── cash including restricted cash (v0.1.27 tags) ─────────────────────────

RC = E.RESTRICTED[0]
RC2 = E.RESTRICTED[1]


def test_restricted_total_used_when_cash_stops_and_history_agrees():
    # PG: both tags equal until 2019, then only cash+restricted
    t = {"2019-03-31": {C: 10.0, RC: 10.0}, "2019-06-30": {C: 10.2, RC: 10.25},
         "2026-09-30": {RC: 15.4}}
    assert cash(t)[T("2026-09-30")] == 15.4
    d = E.cash_series(rows(t))
    assert d["method"].iloc[-1].startswith("ταμείο με δεσμευμένα")


def test_restricted_total_never_used_when_history_differs():
    # WAB / ADP: restricted cash is a big part → no cash at all on the later date
    t = {"2019-03-31": {C: 1.0, RC: 5.0}, "2019-06-30": {C: 1.0, RC: 5.2}, "2026-09-30": {RC: 6.0}}
    assert T("2026-09-30") not in cash(t)


def test_restricted_total_needs_two_past_dates():
    t = {"2019-03-31": {C: 10.0, RC: 10.0}, "2026-09-30": {RC: 15.4}}
    assert T("2026-09-30") not in cash(t)


def test_restricted_total_never_replaces_a_cash_line_that_exists():
    t = {"2019-03-31": {C: 10.0, RC: 10.0}, "2019-06-30": {C: 10.0, RC: 10.0},
         "2026-09-30": {C: 12.0, RC: 30.0}}
    assert cash(t)[T("2026-09-30")] == 12.0


def test_company_total_beats_the_restricted_total():
    # TGT: no cash line, but its own cash + short-term investments total
    t = {"2019-03-31": {C: 10.0, RC: 10.0}, "2019-06-30": {C: 10.0, RC: 10.0},
         "2026-09-30": {COMB: 5.4, RC: 5.6}}
    assert cash(t)[T("2026-09-30")] == 5.4


def test_securities_are_added_to_the_restricted_total_too():
    t = {"2019-03-31": {C: 10.0, RC: 10.0}, "2019-06-30": {C: 10.0, RC: 10.0},
         "2026-09-30": {RC: 4.8, MSC: 0.3}}
    assert cash(t)[T("2026-09-30")] == 4.8 + 0.3


def test_only_the_past_counts_for_the_restricted_guard():
    # agreement that appears only LATER must not unlock an earlier date
    t = {"2018-06-30": {RC: 9.0}, "2019-03-31": {C: 10.0, RC: 10.0}, "2019-06-30": {C: 10.0, RC: 10.0}}
    assert T("2018-06-30") not in cash(t)


def test_disposal_group_variant_when_the_plain_total_is_missing():
    t = {"2019-03-31": {C: 10.0, RC2: 10.0}, "2019-06-30": {C: 10.0, RC2: 10.1}, "2026-09-30": {RC2: 4.8}}
    assert cash(t)[T("2026-09-30")] == 4.8
