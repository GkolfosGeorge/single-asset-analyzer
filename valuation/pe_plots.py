"""
valuation/pe_plots.py — price on top, P/E (stock vs peers) underneath.
Kept separate from pe.py so batch runs never import matplotlib.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .pe import PEHistory

BLUE, ORANGE, BAND, LOSS = "#1f5fa8", "#e07b39", "#f3d2bc", "#e6e6e6"


def _shade_losses(ax, stock):
    """Grey background on days with TTM EPS ≤ 0 (no P/E exists there)."""
    loss = stock["loss"].astype(bool).to_numpy()
    if not loss.any():
        return
    idx = stock.index
    edges = np.flatnonzero(np.diff(np.r_[0, loss.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(idx[a], idx[b - 1], color=LOSS, zorder=0)


def plot_pe(h: PEHistory):
    s = h.stock.dropna(subset=["close"]).astype({"close": float, "eps_ttm": float, "pe": float})
    if s.empty:
        print("Δεν υπάρχουν δεδομένα για διάγραμμα.")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})

    # ── top: price ─────────────────────────────────────────────────────
    _shade_losses(ax1, s)
    ax1.plot(s.index, s["close"], color=BLUE, linewidth=1.3)
    ax1.set_ylabel("Τιμή ($, split-adjusted)")
    ax1.set_title(f"{h.ticker} — τιμή μετοχής", loc="left", fontsize=13, fontweight="bold")

    # ── bottom: P/E vs peers ───────────────────────────────────────────
    _shade_losses(ax2, s)
    p = h.peers.reindex(s.index).astype(float)
    if p["median"].notna().any():
        ax2.fill_between(s.index, p["q25"], p["q75"], color=BAND, alpha=0.8,
                         label=f"{h.peer_group}: 25%–75% των εταιρειών", zorder=1)
        ax2.plot(s.index, p["median"], color=ORANGE, linewidth=1.4,
                 label=f"{h.peer_group} «{h.group_name}» — διάμεσος", zorder=2)
    ax2.plot(s.index, s["pe"], color=BLUE, linewidth=1.5, label=f"{h.ticker} P/E (TTM)", zorder=3)

    pe = s["pe"].dropna()
    if not pe.empty:
        own_med = pe.median()
        ax2.axhline(own_med, color=BLUE, linestyle="--", linewidth=1,
                    label=f"{h.ticker} διάμεσος περιόδου ({own_med:.1f})")
        # When EPS is near zero, P/E explodes (EPS 0.05 → P/E 600) and one
        # spike would flatten everything else. Cap the axis at 3.5× the
        # stock's own median, but never below the peers' band.
        q75max = p["q75"].max() if p["q75"].notna().any() else 0
        top = max(min(np.nanpercentile(pe, 98), 3.5 * own_med), q75max * 1.1) * 1.1
        ax2.set_ylim(0, top)
        if (pe > top).any():
            ax2.text(0.99, 0.97, f"τιμές P/E > {top:.0f} κόβονται (EPS κοντά στο 0)",
                     transform=ax2.transAxes, ha="right", va="top", fontsize=8, color="#666")
        ax2.annotate(f"{pe.iloc[-1]:.1f}", xy=(pe.index[-1], pe.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", color=BLUE, fontweight="bold")

    ax2.set_ylabel("P/E (TTM)")
    ax2.set_title("P/E: μετοχή vs κλάδος (γκρι φόντο = ζημιές, δεν ορίζεται P/E)",
                  loc="left", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9, frameon=False)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# P/E BANDS
# ═════════════════════════════════════════════════════════════════════════
from .pe import pe_band_levels, pe_band_prices, decompose_by_year, decompose_total

GREEN, RED, GREY = "#3c9a5f", "#c0392b", "#888888"


def plot_pe_bands(h: PEHistory):
    """
    Price vs "price at the stock's usual P/E".
    Green zone = P/E between its 25th and 75th percentile (normal valuation).
    Price below the zone → cheap vs its own history; above → expensive.
    """
    s = h.stock.dropna(subset=["close"])
    levels = pe_band_levels(s)
    if s.empty or not levels:
        print("P/E bands: δεν υπάρχουν αρκετά δεδομένα.")
        return None
    bands = pe_band_prices(s, levels)

    fig, ax = plt.subplots(figsize=(13, 6))
    _shade_losses(ax, s)
    ax.fill_between(s.index, bands[25], bands[75], color=GREEN, alpha=0.15,
                    label=f"«κανονική» ζώνη: P/E {levels[25]:.1f}–{levels[75]:.1f}")
    ax.plot(s.index, bands[50], color=GREEN, linewidth=1.2,
            label=f"τιμή στον διάμεσο P/E ({levels[50]:.1f})")
    ax.plot(s.index, bands[10], color=GREEN, linewidth=0.9, linestyle=":",
            label=f"P/E {levels[10]:.1f} (φθηνό 10%)")
    ax.plot(s.index, bands[90], color=RED, linewidth=0.9, linestyle=":",
            label=f"P/E {levels[90]:.1f} (ακριβό 10%)")
    ax.plot(s.index, s["close"], color=BLUE, linewidth=1.6, label=f"{h.ticker} τιμή", zorder=5)

    last = s.iloc[-1]
    if pd.notna(last["pe"]):
        ax.annotate(f"{last['close']:.2f}$\nP/E {last['pe']:.1f}", xy=(s.index[-1], last["close"]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=BLUE, fontweight="bold", fontsize=9)

    ax.set_ylabel("$ (split-adjusted)")
    ax.set_title(f"{h.ticker} — τιμή vs «τιμή στο συνηθισμένο P/E» "
                 f"(γκρι = ζημιές)", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════
# RETURN DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════

def plot_decomposition(h: PEHistory):
    """
    Left: cumulative ×-factors on a log axis (on a log axis, price = EPS + P/E
    distances, so you SEE the multiplication as stacking).
    Right: per-year % price change split into "from earnings" + "from P/E".
    """
    s = h.stock.dropna(subset=["close"])
    tot = decompose_total(s)
    years = decompose_by_year(s)
    if tot is None:
        print("Ανάλυση απόδοσης: δεν υπάρχουν αρκετά δεδομένα.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # ── left: cumulative indices from the first valid day ─────────────────
    v = s[s.index >= tot["start"]]
    base = v.iloc[0]
    eps_idx = (v["eps_ttm"] / base["eps_ttm"]).where(v["eps_ttm"] > 0)
    pe_idx = v["pe"] / base["pe"]
    ax1.plot(v.index, v["close"] / base["close"], color=BLUE, linewidth=1.8,
             label=f"τιμή ×{tot['price_x']:.2f}")
    ax1.plot(v.index, eps_idx, color=GREEN, linewidth=1.4, label=f"κέρδη (EPS) ×{tot['eps_x']:.2f}")
    ax1.plot(v.index, pe_idx, color=ORANGE, linewidth=1.4, label=f"P/E ×{tot['pe_x']:.2f}")
    ax1.axhline(1, color=GREY, linewidth=0.8)
    ax1.set_yscale("log")
    # Near-zero EPS sends the EPS/P/E lines to ×0.01 or ×50; scale the axis on
    # the PRICE line (the thing we explain) and let the spikes leave the frame.
    pr = (v["close"] / base["close"]).to_numpy()
    lo, hi = min(pr.min(), 1) / 2.5, max(pr.max(), 1) * 2.5
    ax1.set_ylim(lo, hi)
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50) if lo <= t <= hi]
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"×{t:g}" for t in ticks])
    ax1.minorticks_off()
    ax1.set_title("Σωρευτικά (λογαριθμικός άξονας)", loc="left", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)

    # ── right: per-year stacked contributions ─────────────────────────────
    if not years.empty:
        x = np.arange(len(years))
        ok = ~years["unstable"].to_numpy()
        e = np.where(ok, years["from_eps_pct"], 0.0)
        m = np.where(ok, years["from_pe_pct"], 0.0)
        # stack positives upward and negatives downward so bars never overlap
        e_pos, e_neg = np.clip(e, 0, None), np.clip(e, None, 0)
        m_pos, m_neg = np.clip(m, 0, None), np.clip(m, None, 0)
        ax2.bar(x, e_pos, color=GREEN, label="από κέρδη")
        ax2.bar(x, e_neg, color=GREEN)
        ax2.bar(x, m_pos, bottom=e_pos, color=ORANGE, label="από αλλαγή P/E")
        ax2.bar(x, m_neg, bottom=e_neg, color=ORANGE)
        ax2.scatter(x, years["price_pct"], color=BLUE, zorder=5, s=28, label="τιμή (σύνολο)")
        for xi in x[~ok]:
            ax2.annotate("EPS≈0", (xi, 0), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=7, color=GREY, rotation=90)
        ax2.axhline(0, color=GREY, linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.split(" (")[0] for p in years["period"]], rotation=45, ha="right")
        ax2.set_ylabel("% μεταβολή τιμής")
        ax2.legend(loc="upper left", fontsize=9, frameon=False)
    ax2.set_title("Ανά έτος", loc="left", fontsize=12, fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{h.ticker} — από πού ήρθε η απόδοση: κέρδη ή P/E;  (χωρίς μερίσματα)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
