"""
valuation/ev_plots.py — Charts for EVHistory (see ev.py).

Panel 1: what EV is made of — market cap, + debt, − cash. When the gap
         between the EV line and the market-cap line grows, the company
         is borrowing (or spending its cash); P/E alone can't see that.
Panel 2: EV/EBIT vs the stock's OWN history (median, 25%–75% band).
         Below the band = cheap for this stock. Same caveat as the other
         bands: percentiles use the whole window (descriptive, not a backtest).
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_ev(h, figsize=(11, 7)):
    d = h.daily.dropna(subset=["ev"]) if not h.daily.empty else h.daily
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if d.empty:
        ax1.text(0.5, 0.5, f"{h.ticker}: {h.status}", ha="center", va="center",
                 transform=ax1.transAxes)
        return fig

    bn = 1e9
    ax1.plot(d.index, d["mcap"] / bn, color="#2a6fdb", lw=1.4, label="Market cap")
    ax1.plot(d.index, d["ev"] / bn, color="black", lw=2, label="EV = mcap + χρέος − ταμείο")
    ax1.fill_between(d.index, d["mcap"] / bn, d["ev"] / bn, where=d["ev"] >= d["mcap"],
                     color="#c0392b", alpha=0.20, label="καθαρό χρέος")
    ax1.fill_between(d.index, d["mcap"] / bn, d["ev"] / bn, where=d["ev"] < d["mcap"],
                     color="#1f9d55", alpha=0.20, label="καθαρό ταμείο")
    ax1.set_ylabel("δισ. $")
    ax1.set_title(f"{h.ticker} — από τι αποτελείται η αξία της επιχείρησης (EV)", loc="left")
    ax1.legend(loc="upper left", frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    r = d["ev_ebit"].dropna()
    if r.empty:
        ax2.text(0.5, 0.5, f"EV/EBIT: {h.status}", ha="center", va="center",
                 transform=ax2.transAxes)
    else:
        q25, q50, q75 = np.percentile(r, [25, 50, 75])
        ax2.axhspan(q25, q75, color="#2a6fdb", alpha=0.10,
                    label=f"25%–75% της ίδιας ({q25:.1f}–{q75:.1f})")
        ax2.axhline(q50, color="#2a6fdb", lw=1, ls="--", label=f"διάμεσος {q50:.1f}")
        ax2.plot(d.index, d["ev_ebit"], color="black", lw=1.4, label="EV/EBIT")
        ax2.annotate(f"{r.iloc[-1]:.1f}", (r.index[-1], r.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, fontweight="bold")
        ax2.set_title("EV/EBIT — χαμηλά = φθηνή για τα δικά της δεδομένα (κενά = αρνητικό EBIT)",
                      loc="left")
        ax2.legend(loc="upper left", frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig
