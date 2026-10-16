"""
valuation/yields_plots.py — Charts for FCFHistory (see yields.py).

Panel 1: where FCF comes from — CFO TTM, capex TTM and their difference.
         A falling FCF line with a rising CFO line means capex is the cause
         (MSFT 2025–26: AI data centers), not a weaker business.
Panel 2: FCF yield vs the stock's OWN history (median, 25%–75% band).
         Above the band = cheap for this stock, below = expensive.
         Same caveat as the P/E bands: percentiles use the whole window,
         so they describe, they don't backtest.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_fcf(h, figsize=(11, 7)):
    d = h.daily.dropna(subset=["close"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    if d.empty or d["fcf_ttm"].dropna().empty:
        ax1.text(0.5, 0.5, f"{h.ticker}: {h.status}", ha="center", va="center",
                 transform=ax1.transAxes)
        return fig

    bn = 1e9
    ax1.plot(d.index, d["cfo_ttm"] / bn, color="#2a6fdb", lw=1.6, label="CFO (TTM)")
    ax1.plot(d.index, d["capex_ttm"] / bn, color="#d9822b", lw=1.6, label="Capex (TTM)")
    ax1.plot(d.index, d["fcf_ttm"] / bn, color="#1f9d55", lw=2.4, label="FCF = CFO − capex")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("δισ. $")
    ax1.set_title(f"{h.ticker} — από πού έρχεται το free cash flow", loc="left")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(alpha=0.25)

    y = d["fcf_yield"] * 100
    yv = y.dropna()
    q25, q50, q75 = np.percentile(yv, [25, 50, 75])
    ax2.axhspan(q25, q75, color="#1f9d55", alpha=0.10,
                label=f"25%–75% της ίδιας ({q25:.1f}%–{q75:.1f}%)")
    ax2.axhline(q50, color="#1f9d55", lw=1, ls="--", label=f"διάμεσος {q50:.1f}%")
    ax2.plot(y.index, y, color="black", lw=1.4, label="FCF yield")
    ax2.fill_between(y.index, y, 0, where=y < 0, color="#c0392b", alpha=0.25,
                     label="αρνητικό FCF")
    last = yv.iloc[-1]
    ax2.annotate(f"{last:.2f}%", (yv.index[-1], last), xytext=(6, 0),
                 textcoords="offset points", va="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax2.set_title("FCF yield (FCF TTM / market cap) — ψηλά = φθηνή για τα δικά της δεδομένα",
                  loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig
