"""
piotroski_plots.py — visual reports for the Piotroski F-Score.

Kept separate from piotroski.py so that batch/universe runs never import
matplotlib; only the notebook does.

    plot_report(result)            one company, one year: 9 panels with the
                                   actual numbers behind each pass/fail
    plot_history(history_df)       F-Score per fiscal year (from
                                   piotroski.piotroski_history)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .piotroski import (
    CRITERIA, GROUPS, LABELS, PiotroskiResult, _BAR_TEXT, fmt_value, note_text,
)

GREEN, RED, GREY, LIGHT = "#2e9d5b", "#d1495b", "#9aa0a6", "#c9ced6"


def _status(passed):
    if passed is True:
        return "✓", GREEN
    if passed is False:
        return "✗", RED
    return "–", GREY


def plot_report(result: PiotroskiResult | None, lang: str = "el"):
    """
    3 rows = the 3 Piotroski groups. In every panel the LAST bar is the
    value being judged (green = pass, red = fail, grey = could not be
    evaluated); earlier bars are the comparison point, in light grey.
    """
    if result is None:
        print("Δεν υπάρχουν αρκετά δεδομένα για διάγραμμα." if lang == "el"
              else "Not enough data to plot.")
        return None

    lab, bt = LABELS[lang], _BAR_TEXT[lang]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5))

    for row, (group, keys) in enumerate(GROUPS.items()):
        axes[row, 0].annotate(
            lab[group], xy=(-0.42, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=12, fontweight="bold",
        )
        for col in range(4):
            ax = axes[row, col]
            if col >= len(keys):
                ax.axis("off")
                continue
            det = result.details[keys[col]]
            symbol, color = _status(det.passed)

            names = [bt[b] for b, _ in det.bars]
            values = [v if v is not None else 0 for _, v in det.bars]
            colors = [LIGHT] * (len(values) - 1) + [color]
            bars = ax.bar(names, values, color=colors, width=0.55)
            ax.axhline(0, color="black", linewidth=0.8)
            if len(values) == 1:
                ax.set_xlim(-1.2, 1.2)  # keep a lone bar from filling the panel

            for bar, (_, v) in zip(bars, det.bars):
                ax.annotate(
                    fmt_value(v, det.kind),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3 if bar.get_height() >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=9,
                )

            ax.set_title(f"{symbol}  {lab[det.key]}", color=color,
                         fontsize=11, fontweight="bold", loc="left")
            ax.set_yticks([])
            ax.margins(y=0.25)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if det.note:
                ax.text(0.5, -0.22, note_text(det, lang), transform=ax.transAxes,
                        ha="center", va="top", fontsize=8, color="#555", wrap=True)

    # Summary panel (row 3, last two cells).
    for c in (2, 3):
        axes[2, c].remove()
    ax_sum = fig.add_subplot(3, 4, (11, 12))
    passed = result.score
    failed = result.max_score - result.score
    n_a = len(CRITERIA) - result.max_score
    ax_sum.barh([0], [passed], color=GREEN)
    ax_sum.barh([0], [failed], left=[passed], color=RED)
    ax_sum.barh([0], [n_a], left=[passed + failed], color=GREY)
    ax_sum.set_xlim(0, 9)
    ax_sum.set_xticks(range(10))
    ax_sum.set_yticks([])
    for side in ("top", "right", "left"):
        ax_sum.spines[side].set_visible(False)
    ax_sum.set_title(
        f"{result.score} / {result.max_score}" + (
            f"   (δεν αξιολογήθηκαν: {n_a})" if lang == "el" else f"   (not evaluated: {n_a})"
        ) * bool(n_a),
        fontsize=16, fontweight="bold", loc="left",
    )
    legend = ["πέρασε", "απέτυχε", "δεν αξιολογήθηκε"] if lang == "el" else ["pass", "fail", "n/a"]
    ax_sum.legend(handles=[Patch(color=c, label=l) for c, l in zip((GREEN, RED, GREY), legend)],
                  loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

    header = (f"{result.ticker} — Piotroski F-Score   FY{result.fiscal_year} vs "
              f"FY{result.prev_fiscal_year}   ({'κατάθεση' if lang == 'el' else 'filed'} "
              f"{result.filed_date})")
    fig.suptitle(header, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.96), h_pad=3.5)
    return fig


def plot_history(history, ticker: str = "", lang: str = "el"):
    """Stacked bars per fiscal year: passed / failed / not evaluated (= 9)."""
    if history is None or history.empty:
        print("Δεν υπάρχει ιστορικό." if lang == "el" else "No history.")
        return None

    x = history["fiscal_year"].astype(str)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(history) + 2), 4.5))
    ax.bar(x, history["score"], color=GREEN)
    ax.bar(x, history["failed"], bottom=history["score"], color=RED)
    ax.bar(x, history["n_a"], bottom=history["score"] + history["failed"], color=GREY)
    for xi, s, m in zip(x, history["score"], history["max_score"]):
        ax.annotate(f"{s}/{m}", xy=(xi, 9), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0, 10)
    ax.set_yticks(range(10))
    ax.axhline(7, color=GREEN, linestyle=":", linewidth=1)
    ax.axhline(3, color=RED, linestyle=":", linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend = ["πέρασε", "απέτυχε", "δεν αξιολογήθηκε"] if lang == "el" else ["pass", "fail", "n/a"]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip((GREEN, RED, GREY), legend)],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    title = "Piotroski F-Score ανά χρονιά" if lang == "el" else "Piotroski F-Score by fiscal year"
    ax.set_title(f"{ticker} — {title}".strip(" —"), fontsize=13, fontweight="bold", loc="left")
    fig.tight_layout()
    return fig
