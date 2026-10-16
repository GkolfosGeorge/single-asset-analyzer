"""
valuation/payouts_plots.py — Charts for PayoutHistory (see payouts.py).

Panel 1: P/S vs the stock's OWN history (median, 25%–75% band).
Panel 2: shareholder yield, stacked: dividends + buybacks (% of market cap),
         with the real 12-month change in share count on top. If the
         buyback area is thick but the line stays near 0, the buybacks
         mostly offset stock given to employees.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def plot_payouts(h, figsize=(11, 7)):
    d = h.daily
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if d.empty:
        ax1.text(0.5, 0.5, h.ticker, ha="center", va="center", transform=ax1.transAxes)
        return fig

    ps = d["ps"].dropna()
    if ps.empty:
        ax1.text(0.5, 0.5, "P/S: —", ha="center", va="center", transform=ax1.transAxes)
    else:
        q25, q50, q75 = np.percentile(ps, [25, 50, 75])
        ax1.axhspan(q25, q75, color="#8e44ad", alpha=0.10,
                    label=f"25%–75% της ίδιας ({q25:.2f}–{q75:.2f})")
        ax1.axhline(q50, color="#8e44ad", lw=1, ls="--", label=f"διάμεσος {q50:.2f}")
        ax1.plot(d.index, d["ps"], color="black", lw=1.4, label="P/S")
        ax1.annotate(f"{ps.iloc[-1]:.2f}", (ps.index[-1], ps.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, fontweight="bold")
        ax1.legend(loc="best", frameon=False, fontsize=8)
    ax1.set_title(f"{h.ticker} — P/S (market cap / έσοδα TTM): χαμηλά = φθηνή για τα δικά της δεδομένα",
                  loc="left")
    ax1.grid(alpha=0.25)

    div = (d["div_yield"] * 100).fillna(0).clip(lower=0)
    bb = (d["buyback_yield"] * 100).fillna(0).clip(lower=0)
    ax2.stackplot(d.index, div, bb, colors=["#1f9d55", "#2a6fdb"], alpha=0.35,
                  labels=["μερίσματα", "buybacks"])
    ax2.plot(d.index, d["share_change_12m"] * 100, color="black", lw=1.4,
             label="πραγματική μείωση μετοχών (12 μήνες)")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax2.set_title("Shareholder yield (% του market cap) και τι έμεινε πραγματικά στους μετόχους",
                  loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig
