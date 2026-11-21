#!/usr/bin/env python3
"""Concept sheets for lesson 00 (introduction to probability).

One page per notebook that summarizes its central idea. Every number is
computed here from the same parameters the notebook uses.
Images are written to coursera/00_introduction_to_probability/img/.

Run from anywhere:  python scripts/make_lesson00_concept_images.py
"""
from functools import partial
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from viz_style import (REPO, SURFACE, CARD, INK, INK2, MUTED, GRID, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, frac_label, bars, grouped_bars, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "00_introduction_to_probability"
save = partial(_save, folder=LESSON / "img")


# ================================================================ shared bits
def mosaic(ax, split, frac_left, frac_right, column_names, yes_name, no_name, title):
    """Unit-square area diagram: columns split by the first event, rows by the second."""
    for x0, width, frac, name in [(0, split, frac_left, column_names[0]),
                                  (split, 1 - split, frac_right, column_names[1])]:
        ax.add_patch(Rectangle((x0, 0), width, frac, facecolor=BLUE, edgecolor=SURFACE, linewidth=3))
        ax.add_patch(Rectangle((x0, frac), width, 1 - frac, facecolor=BLUE_LIGHT, edgecolor=SURFACE,
                               linewidth=3))
        ax.text(x0 + width / 2, frac / 2, f"{yes_name}\n{frac:.0%}\narea {width * frac:.2f}",
                ha="center", va="center", color="white", fontsize=10.5, weight="bold")
        ax.text(x0 + width / 2, frac + (1 - frac) / 2, f"{no_name}\narea {width * (1 - frac):.2f}",
                ha="center", va="center", color=INK, fontsize=10.5)
        ax.text(x0 + width / 2, -0.06, f"{name} ({width:.0%})", ha="center", va="top", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=8)


def tree(ax, root_label, first, second, highlight=()):
    """Two-level probability tree. first: [(name, p)], second: {name: [(name, p)]}."""
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    root = (1.2, 3.0)
    level1 = {name: (4.0, y) for (name, _), y in zip(first, [4.6, 1.4])}
    ax.text(root[0] - 0.15, root[1], root_label, ha="right", va="center", fontsize=11)
    leaves = []
    for (name, p) in first:
        node = level1[name]
        ax.plot([root[0], node[0]], [root[1], node[1]], color=MUTED, linewidth=1.5, zorder=1)
        ax.text((root[0] + node[0]) / 2, (root[1] + node[1]) / 2 + 0.22, f"{p:g}", ha="center",
                fontsize=11, color=INK2)
        ax.add_patch(FancyBboxPatch((node[0] - 0.7, node[1] - 0.25), 1.4, 0.5, boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=MUTED, zorder=2))
        ax.text(*node, name, ha="center", va="center", fontsize=10.5, zorder=3)
        for (leaf, q), dy in zip(second[name], [0.8, -0.8]):
            pos = (6.6, node[1] + dy)
            ax.plot([node[0] + 0.7, pos[0]], [node[1], pos[1]], color=MUTED, linewidth=1.5, zorder=1)
            ax.text((node[0] + 0.7 + pos[0]) / 2, (node[1] + pos[1]) / 2 + 0.2, f"{q:g}", ha="center",
                    fontsize=11, color=INK2)
            key = (name, leaf)
            hot = key in highlight
            ax.text(pos[0] + 0.15, pos[1], f"{name}, {leaf}:  {p:g} × {q:g} = {p * q:.2f}", va="center",
                    fontsize=11, color=ORANGE if hot else INK, weight="bold" if hot else "normal")
            leaves.append(p * q)
    return leaves


def bayes(prior, p_evidence_given_a, p_evidence_given_not_a):
    numerator = prior * p_evidence_given_a
    return numerator / (numerator + (1 - prior) * p_evidence_given_not_a)


# ================================================================ shared bits for notebooks 01-08, 13
TWO_DICE = list(product(range(1, 7), repeat=2))


def tiles(ax, labels, colors, columns, title, faded=(), fontsize=12, gap=0.1):
    """Outcome tiles in a grid, one colored square per outcome; faded indices drawn pale."""
    rows = (len(labels) + columns - 1) // columns
    for k, (label, color) in enumerate(zip(labels, colors)):
        r, c = divmod(k, columns)
        alpha = 0.22 if k in faded else 1.0
        ax.add_patch(Rectangle((c, -r), 1 - gap, 1 - gap, facecolor=color, alpha=alpha, linewidth=0))
        dark = color in (BLUE, ORANGE, AQUA) and k not in faded
        ax.text(c + (1 - gap) / 2, -r + (1 - gap) / 2, label, ha="center", va="center",
                fontsize=fontsize, color="white" if dark else INK, alpha=0.45 if k in faded else 1.0,
                weight="bold" if dark else "normal")
    ax.set_xlim(-0.1, columns)
    ax.set_ylim(-rows + 1 - 0.1, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=8)


def dice_grid(ax, color_of, title, faded=lambda pair: False, labels=False):
    """6 x 6 table of two-dice outcomes: rows = first die (1 on top), columns = second die."""
    for first, second in TWO_DICE:
        pair = (first, second)
        fade = faded(pair)
        ax.add_patch(Rectangle((second - 1, 6 - first), 1, 1, facecolor=color_of(pair),
                               alpha=0.25 if fade else 1.0, edgecolor=SURFACE, linewidth=2))
        if labels:
            ax.text(second - 0.5, 6 - first + 0.5, f"{first},{second}", ha="center", va="center",
                    fontsize=9, color=INK, alpha=0.4 if fade else 1.0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(6) + 0.5, range(1, 7))
    ax.set_yticks(np.arange(6) + 0.5, range(6, 0, -1))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("second die")
    ax.set_ylabel("first die")
    ax.set_title(title, pad=8)


def legend_below(ax, entries, y=-0.13):
    handles = [Rectangle((0, 0), 1, 1, color=color) for _, color in entries]
    ax.legend(handles, [name for name, _ in entries], loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=len(entries), fontsize=9.5, handlelength=1, columnspacing=0.8)


def sample_space_box(ax, title):
    ax.add_patch(Rectangle((0, 0), 10, 6.4, facecolor=CARD, edgecolor=MUTED))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=8)


# ================================================================ notebook 01
def sheet_01_what_is_probability():
    total_kids, soccer = 10, 3
    rng = np.random.default_rng(seed=42)
    first_run = rng.integers(0, 2, size=(100_000, 2))
    estimate = np.all(first_run == 1, axis=1).mean()
    assert round(estimate, 4) == 0.2509
    flips = rng.integers(0, 2, size=(10_000, 2))
    running = np.cumsum(np.all(flips == 1, axis=1)) / np.arange(1, 10_001)

    fig = new_figure("What is probability? Count the favorable outcomes, divide by all outcomes",
                     "P(E) = |E| / |S| when every outcome is equally likely. "
                     "The event E is a subset of the sample space S.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    sample_space_box(ax, f"P(soccer) = {soccer}/{total_kids} = {soccer / total_kids:.1f}")
    ax.add_patch(Circle((2.6, 3.0), 2.0, facecolor=BLUE, alpha=0.15, edgecolor=BLUE, linewidth=2))
    for x, y in [(1.9, 3.5), (3.3, 3.5), (2.6, 2.3)]:
        ax.add_patch(Circle((x, y), 0.4, color=ORANGE))
    for x, y in [(5.7, 4.2), (7.1, 4.2), (8.5, 4.2), (5.7, 2.8), (7.1, 2.8), (8.5, 2.8), (7.1, 1.4)]:
        ax.add_patch(Circle((x, y), 0.4, color=BLUE_LIGHT))
    ax.text(0.25, 6.0, f"sample space S: all {total_kids} kids", fontsize=11, va="top", color=INK2)
    ax.text(2.6, 0.55, f"event E: {soccer} play soccer", ha="center", fontsize=11, color=ORANGE,
            weight="bold")
    ax.text(7.1, 0.55, f"{total_kids - soccer} don't", ha="center", fontsize=11, color=INK2)

    ax = fig.add_subplot(gs[0, 1])
    for i, n in enumerate([1, 2, 3]):
        outcomes = ["".join(o) for o in product("HT", repeat=n)]
        for k, o in enumerate(outcomes):
            hot = o == "H" * n
            ax.add_patch(Rectangle((k, -i * 1.5), 0.9, 0.9, facecolor=ORANGE if hot else BLUE_LIGHT,
                                   linewidth=0))
            ax.text(k + 0.45, -i * 1.5 + 0.45, o, ha="center", va="center", fontsize=10,
                    color="white" if hot else INK, weight="bold" if hot else "normal")
    ax.set_xlim(-0.1, 8)
    ax.set_ylim(-3.2, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([0.45, -1.05, -2.55], ["1 coin\nP = 1/2", "2 coins\nP = 1/4", "3 coins\nP = 1/8"])
    ax.tick_params(length=0, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Each coin doubles |S|; all heads stays 1 outcome", pad=8)

    panel(fig.add_subplot(gs[0, 2]), "The vocabulary", [
        "Experiment: a process with an uncertain",
        "    result (flip a coin, pick a kid)",
        "Outcome: one possible result, e.g. H",
        "Sample space S: all outcomes, {H, T}",
        "Event E: the outcomes you care about",
        "",
        "P(E) = favorable / total = |E| / |S|",
        "Always from 0 (impossible) to 1 (certain);",
        "3/10 = 0.3 = 30% are the same number.",
    ], spacing=0.09)

    n = np.arange(1, 7)
    bars(fig.add_subplot(gs[1, 0]), [f"{k}\n|S|={2 ** k}" for k in n], 0.5 ** n,
         "P(all heads) with n coins = 1 / 2ⁿ", "", ylim=0.6,
         label_fmt=lambda v: f"1/{round(1 / v)}", highlight=1)

    ax = fig.add_subplot(gs[1, 1])
    trials = np.arange(1, 10_001)
    ax.semilogx(trials, running, color=BLUE, linewidth=1.8)
    ax.axhline(0.25, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.text(1.3, 0.28, "theory: 1/4", color=ORANGE, fontsize=10.5, weight="bold")
    ax.annotate(f"after 10,000: {running[-1]:.4f}", (10_000, running[-1]), textcoords="offset points",
                xytext=(-6, 22), ha="right", fontsize=10.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of trials (log scale)")
    ax.set_ylabel("frequency of HH")
    ax.grid(True)
    ax.set_title("Simulated frequency of HH settles at 1/4")

    panel(fig.add_subplot(gs[1, 2]), "Key values", [
        "`event          favorable/total   P`",
        "`soccer kid          3/10       0.3`",
        "`H, 1 coin           1/2        0.5`",
        "`HH, 2 coins         1/4        0.25`",
        "`HHH, 3 coins        1/8        0.125`",
        "`5 heads, 5 coins    1/32       0.031`",
        "",
        f"Simulation, 100,000 pairs: P(HH) ≈ {estimate:.4f}",
        "Frequencies converge to probabilities.",
    ], spacing=0.09)
    save(fig, name="01_what_is_probability.png")


# ================================================================ notebook 02
def sheet_02_dice_example():
    rng = np.random.default_rng(seed=42)
    n_rolls = 10_000
    rolls = rng.integers(1, 7, size=n_rolls)
    faces, counts = np.unique(rolls, return_counts=True)
    frequencies = counts / n_rolls
    assert [round(f, 4) for f in frequencies] == [0.1681, 0.1718, 0.1662, 0.1671, 0.1643, 0.1625]
    running_six = np.cumsum(rolls == 6) / np.arange(1, n_rolls + 1)
    doubles = sum(1 for a, b in TWO_DICE if a == b)
    sum7 = sum(1 for a, b in TWO_DICE if a + b == 7)
    assert (len(TWO_DICE), doubles, sum7) == (36, 6, 6)

    fig = new_figure("Rolling dice: equally likely outcomes, counted on a grid",
                     "One die has 6 outcomes, two dice have 6 × 6 = 36. "
                     "Over many rolls, frequencies settle at the probabilities.")
    gs = grid(fig, top=0.84, wspace=0.3)

    tiles(fig.add_subplot(gs[0, 0]), [str(k) for k in range(1, 7)], [BLUE_LIGHT] * 5 + [ORANGE], 3,
          "One die: |S| = 6, P(6) = 1/6", fontsize=20)

    dice_grid(fig.add_subplot(gs[0, 1]), lambda p: ORANGE if p == (6, 6) else BLUE_LIGHT,
              "Two dice: |S| = 36, P(6,6) = 1/36", labels=True)

    panel(fig.add_subplot(gs[0, 2]), "Counting on the grid", [
        "P(E) = |E| / |S|, with every cell equally likely.",
        "",
        "`one die    P(6)        = 1/6  ≈ 0.167`",
        "`two dice   P(6,6)      = 1/36 ≈ 0.028`",
        f"`two dice   P(double)   = {doubles}/36 ≈ 0.167`",
        f"`two dice   P(sum = 7)  = {sum7}/36 ≈ 0.167`",
        "`3 dice     P(6,6,6)    = 1/216`",
        "",
        "Repeating an experiment multiplies |S|:",
        "6 → 36 → 216, just like coins 2 → 4 → 8.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    bars(ax, [str(f) for f in faces], frequencies, f"Frequencies after {n_rolls:,} rolls", "die face",
         ylabel="frequency", ylim=0.24, label_fmt=lambda v: f"{v:.3f}")
    ax.axhline(1 / 6, color=ORANGE, linestyle="--", linewidth=1.6, zorder=3)
    ax.text(5.45, 1 / 6 + 0.035, "1/6 ≈ 0.167", ha="right", color=ORANGE, fontsize=10.5, weight="bold")

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(np.arange(1, n_rolls + 1), running_six, color=BLUE, linewidth=1.8)
    ax.axhline(1 / 6, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.text(1.3, 0.21, "theory: 1/6", color=ORANGE, fontsize=10.5, weight="bold")
    ax.annotate(f"after {n_rolls:,}: {running_six[-1]:.4f}", (n_rolls, running_six[-1]),
                textcoords="offset points", xytext=(-6, 22), ha="right", fontsize=10.5)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("number of rolls (log scale)")
    ax.set_ylabel("frequency of a six")
    ax.grid(True)
    ax.set_title("The frequency of a six approaches 1/6")

    panel(fig.add_subplot(gs[1, 2]), "Takeaways", [
        "Fair die: six equally likely outcomes.",
        "Two dice: ordered pairs, so (1,2) and (2,1)",
        "are different cells of the 36.",
        "",
        "Few rolls: frequencies jump around.",
        "Many rolls: every face lands near 1/6",
        f"(here between {frequencies.min():.3f} and {frequencies.max():.3f}).",
    ], spacing=0.09)
    save(fig, name="02_dice_example.png")


# ================================================================ notebook 03
def sheet_03_complement_rule():
    rng = np.random.default_rng(seed=42)
    rolls = rng.integers(1, 7, size=100_000)
    p_not_six_sim = (rolls != 6).mean()
    assert round(p_not_six_sim, 4) == 0.8341
    coins = ["".join(o) for o in product("HT", repeat=3)]
    assert sum(o != "HHH" for o in coins) == 7
    p_four = 1 - (5 / 6) ** 4
    assert round(p_four, 4) == 0.5177

    fig = new_figure("The complement rule: P(not A) = 1 − P(A)",
                     "An event and its complement split the sample space, so their probabilities add to 1.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), 10, 6.4, facecolor=ORANGE, alpha=0.22, edgecolor=MUTED))
    ax.add_patch(Circle((3.0, 3.2), 2.1, facecolor=BLUE, edgecolor=SURFACE, linewidth=2))
    ax.text(3.0, 3.2, "A: soccer\nP(A) = 0.3", ha="center", va="center", fontsize=12, color="white",
            weight="bold")
    ax.text(7.5, 3.2, "A': not soccer\nP(A') = 1 − 0.3\n= 0.7", ha="center", va="center", fontsize=12,
            weight="bold")
    ax.text(0.25, 6.05, "sample space: P = 1", fontsize=11, va="top", color=INK2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("A' is everything outside A", pad=8)

    ax = fig.add_subplot(gs[0, 1])
    names = ["soccer\n(10 kids)", "three heads\n(3 coins)", "rolling a 6\n(1 die)"]
    p_event = np.array([0.3, 1 / 8, 1 / 6])
    labels_event, labels_comp = ["0.3", "1/8", "1/6"], ["0.7", "7/8", "5/6"]
    positions = np.arange(3)
    ax.bar(positions, p_event, width=0.6, color=BLUE, zorder=2)
    ax.bar(positions, 1 - p_event, bottom=p_event, width=0.6, color=ORANGE, alpha=0.55, zorder=2)
    for x, p, le, lc in zip(positions, p_event, labels_event, labels_comp):
        ax.text(x, p / 2, f"P(A) = {le}", ha="center", va="center", color="white", fontsize=10,
                weight="bold")
        ax.text(x, p + (1 - p) / 2, f"P(A') = {lc}", ha="center", va="center", fontsize=10, weight="bold")
    ax.axhline(1, color=INK, linewidth=1)
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("probability")
    ax.grid(axis="y", zorder=0)
    ax.set_title("P(A) + P(A') = 1 every time")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        "A' (\"A prime\") = A does not happen.",
        "Every outcome is in exactly one of A, A'.",
        "",
        "P(A) + P(A') = 1",
        "P(A') = 1 − P(A)",
        "",
        "Use it when the event is awkward to count",
        "but its complement is easy.",
    ], spacing=0.09)

    tiles(fig.add_subplot(gs[1, 0]), coins, [BLUE] + [ORANGE] * 7, 4,
          "Not three heads: 7 of 8 outcomes = 1 − 1/8", fontsize=13)

    n = np.arange(1, 11)
    bars(fig.add_subplot(gs[1, 1]), [str(k) for k in n], 1 - (5 / 6) ** n,
         "P(at least one six) = 1 − (5/6)ⁿ", "number of rolls n", ylim=1.0,
         label_fmt=lambda v: f"{v:.2f}", highlight=3)

    panel(fig.add_subplot(gs[1, 2]), "Rule vs. direct count", [
        "`not soccer   7/10 = 1 − 3/10 = 0.7`",
        "`not HHH      7/8  = 1 − 1/8  = 0.875`",
        "`not a 6      5/6  = 1 − 1/6  = 0.8333`",
        f"Simulated P(not 6), 100,000 rolls: {p_not_six_sim:.4f}",
        "",
        "\"At least one six in 4 rolls\" is hard to count;",
        "its complement \"no six at all\" is (5/6)⁴, so",
        f"P = 1 − (5/6)⁴ = {p_four:.4f} (orange bar).",
    ], spacing=0.09)
    save(fig, name="03_complement_rule.png")


# ================================================================ notebook 04
def sheet_04_sum_rule_disjoint():
    sum7 = {d for d in TWO_DICE if sum(d) == 7}
    sum10 = {d for d in TWO_DICE if sum(d) == 10}
    diff1 = {d for d in TWO_DICE if abs(d[0] - d[1]) == 1}
    diff2 = {d for d in TWO_DICE if abs(d[0] - d[1]) == 2}
    assert not (sum7 & sum10) and not (diff1 & diff2)
    assert (len(sum7), len(sum10), len(diff1), len(diff2)) == (6, 3, 10, 8)
    rng = np.random.default_rng(seed=42)
    sums = rng.integers(1, 7, size=(200_000, 2)).sum(axis=1)
    simulated = ((sums == 7) | (sums == 10)).mean()
    assert round(simulated, 4) == 0.2519

    fig = new_figure("The sum rule for disjoint events: P(A ∪ B) = P(A) + P(B)",
                     "If two events share no outcome, adding their probabilities counts every outcome once.")
    gs = grid(fig, top=0.84, bottom=0.1, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    sample_space_box(ax, "Each kid plays only one sport")
    ax.add_patch(Circle((2.8, 3.1), 1.75, facecolor=BLUE, alpha=0.3, edgecolor=BLUE, linewidth=2))
    ax.add_patch(Circle((7.1, 3.1), 2.0, facecolor=ORANGE, alpha=0.3, edgecolor=ORANGE, linewidth=2))
    ax.text(2.8, 3.1, "soccer\nP = 0.3", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(7.1, 3.1, "basketball\nP = 0.4", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(5.0, 0.35, "no overlap:  P(S ∪ B) = 0.3 + 0.4 = 0.7", ha="center", fontsize=11, color=INK2)

    ax = fig.add_subplot(gs[0, 1])
    tiles(ax, [str(k) for k in range(1, 7)],
          [BLUE_LIGHT, BLUE, BLUE_LIGHT, BLUE, ORANGE, BLUE], 3,
          "One die: even or five = 3/6 + 1/6 = 4/6", fontsize=20)
    legend_below(ax, [("even {2, 4, 6}", BLUE), ("five {5}", ORANGE)], y=0.0)

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        "Disjoint (mutually exclusive): A and B",
        "cannot happen in the same trial.",
        "",
        "Union A ∪ B = \"A or B\":",
        "    P(A ∪ B) = P(A) + P(B)",
        "",
        "It works because no outcome is in both,",
        "so none is counted twice.",
    ], spacing=0.09)

    colors = lambda pair, a, b: BLUE if pair in a else ORANGE if pair in b else "#eef3fa"
    ax = fig.add_subplot(gs[1, 0])
    dice_grid(ax, lambda p: colors(p, sum7, sum10), "Sum 7 or sum 10: 6/36 + 3/36 = 9/36")
    legend_below(ax, [("sum 7 (6)", BLUE), ("sum 10 (3)", ORANGE)])
    ax = fig.add_subplot(gs[1, 1])
    dice_grid(ax, lambda p: colors(p, diff1, diff2), "Difference 1 or 2: 10/36 + 8/36 = 18/36")
    legend_below(ax, [("difference 1 (10)", BLUE), ("difference 2 (8)", ORANGE)])

    panel(fig.add_subplot(gs[1, 2]), "Check, and a warning", [
        "`P(sum 7 or 10)   = 9/36  = 0.25`",
        f"`simulated, 200,000 rolls: {simulated:.4f}`",
        "`P(diff 1 or 2)   = 18/36 = 0.5`",
        "",
        "Odd {1, 3, 5} or five {5} are NOT disjoint:",
        "3/6 + 1/6 = 4/6 counts the 5 twice;",
        "the true answer is 3/6.",
        "Overlapping events need a correction",
        "(inclusion-exclusion, notebook 05).",
    ], spacing=0.09)
    save(fig, name="04_sum_rule_disjoint.png")


# ================================================================ notebook 06
def sheet_06_independence_product_rule():
    rng = np.random.default_rng(seed=42)
    kids = np.array([1] * 40 + [0] * 60)
    fans_in_room1 = np.array([rng.permutation(kids)[:30].sum() for _ in range(10_000)])
    assert round(fans_in_room1.mean(), 2) == 11.99
    dice = rng.integers(1, 7, size=(500_000, 2))
    simulated_66 = np.all(dice == 6, axis=1).mean()
    assert round(simulated_66, 6) == 0.027930

    fig = new_figure("Independence: when one event says nothing about the other, multiply",
                     "Product rule for independent events: P(A ∩ B) = P(A) · P(B). "
                     "\"And\" means intersection.")
    gs = grid(fig, top=0.84, bottom=0.1, wspace=0.3)

    mosaic(fig.add_subplot(gs[0, 0]), 0.4, 0.3, 0.3, ["soccer", "no soccer"], "room 1", "room 2",
           "40% like soccer, 30% sit in room 1")

    ax = fig.add_subplot(gs[0, 1])
    values, counts = np.unique(fans_in_room1, return_counts=True)
    ax.bar(values, counts, width=0.8, color=BLUE, zorder=2)
    ax.axvline(12, color=ORANGE, linestyle="--", linewidth=1.8, zorder=3)
    ax.text(16.5, counts.max() * 0.85, f"expected 0.4 × 30 = 12\naverage {fans_in_room1.mean():.2f}",
            fontsize=10.5, color=ORANGE, weight="bold", va="top")
    ax.set_xlabel("soccer fans in room 1 (30 kids)")
    ax.set_ylabel("number of splits")
    ax.set_ylim(0, counts.max() * 1.12)
    ax.grid(axis="y", zorder=0)
    ax.set_title("10,000 random room splits")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        "Independent: knowing A happened does not",
        "change the probability of B.",
        "    coin tosses: independent",
        "    chess moves: not independent",
        "",
        "P(A ∩ B) = P(A) · P(B)",
        "P(soccer ∩ room 1) = 0.4 × 0.3 = 0.12",
        "",
        "Many events: P(A₁ ∩ … ∩ Aₙ) = P(A₁) ⋯ P(Aₙ)",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    dice_grid(ax, lambda p: ORANGE if p == (6, 6) else BLUE if 6 in p else "#eef3fa",
              "P(6,6) = 1/6 × 1/6 = 1/36")
    legend_below(ax, [("1st is 6 or 2nd is 6", BLUE), ("both: 1 cell of 36", ORANGE)])

    ax = fig.add_subplot(gs[1, 1])
    n = np.arange(1, 11)
    ax.semilogy(n, 0.5 ** n, "o-", color=BLUE, linewidth=2, markersize=6, label="all heads, (1/2)ⁿ")
    ax.semilogy(n, (1 / 6) ** n, "o-", color=ORANGE, linewidth=2, markersize=6, label="all sixes, (1/6)ⁿ")
    ax.annotate("5 heads: 1/32", (5, 0.5 ** 5), textcoords="offset points", xytext=(8, 8), fontsize=10.5)
    ax.annotate(f"10 sixes: {(1 / 6) ** 10:.2e}", (10, (1 / 6) ** 10), textcoords="offset points",
                xytext=(-8, 10), ha="right", fontsize=10.5)
    ax.set_xticks(n)
    ax.set_xlabel("number of independent flips / rolls")
    ax.set_ylabel("probability (log scale)")
    ax.grid(True, which="major")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Streaks shrink exponentially")

    panel(fig.add_subplot(gs[1, 2]), "Three ways to P(6,6)", [
        "`product rule   (1/6)²   0.027778`",
        "`enumeration    1/36     0.027778`",
        f"`simulated      500,000  {simulated_66:.6f}`",
        "",
        "No need to list 6¹⁰ outcomes:",
        f"P(ten sixes) = (1/6)¹⁰ ≈ {(1 / 6) ** 10:.2e}",
        "",
        "Only for independent events. Dependent",
        "events need P(B | A) (notebook 08).",
    ], spacing=0.09)
    save(fig, name="06_independence_product_rule.png")


# ================================================================ notebook 08
def sheet_08_conditional_probability():
    rng = np.random.default_rng(seed=42)
    dice = rng.integers(1, 7, size=(500_000, 2))
    sums = dice.sum(axis=1)
    first_is_6 = dice[:, 0] == 6
    sim_given = (sums[first_is_6] == 10).mean()
    sim_prior = (sums == 10).mean()
    assert first_is_6.sum() == 83_194
    assert (round(sim_given, 4), round(sim_prior, 4)) == (0.1677, 0.0833)
    given = [sum(1 for a, b in TWO_DICE if a == k and a + b == 10) / 6 for k in range(1, 7)]
    assert given[0] == 0 and given[5] == 1 / 6

    fig = new_figure("Conditional probability: new information shrinks the sample space",
                     "P(B | A) = (outcomes in both A and B) / (outcomes in A).  "
                     "General product rule: P(A ∩ B) = P(A) · P(B | A).")
    gs = grid(fig, top=0.84, bottom=0.1, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    cells = {"HH": (0, 1), "HT": (1, 1), "TH": (0, 0), "TT": (1, 0)}
    for x0, keep_h in [(0, False), (3.2, True)]:
        for o, (x, y) in cells.items():
            gone = keep_h and o[0] == "T"
            color = ORANGE if o == "HH" else BLUE_LIGHT
            ax.add_patch(Rectangle((x0 + x, y), 0.92, 0.92, facecolor=color, alpha=0.18 if gone else 1.0,
                                   linewidth=0))
            ax.text(x0 + x + 0.46, y + 0.46, o, ha="center", va="center", fontsize=14,
                    color="white" if o == "HH" else INK, alpha=0.35 if gone else 1.0, weight="bold")
    ax.annotate("", xy=(3.05, 1.0), xytext=(2.05, 1.0), arrowprops=dict(arrowstyle="->", lw=2, color=INK2))
    ax.text(2.55, 1.25, "1st is H", ha="center", fontsize=10, color=INK2)
    ax.text(0.96, -0.3, "P(HH) = 1/4", ha="center", va="top", fontsize=12)
    ax.text(4.16, -0.3, "P(HH | 1st H) = 1/2", ha="center", va="top", fontsize=12, color=ORANGE,
            weight="bold")
    ax.set_xlim(-0.1, 5.3)
    ax.set_ylim(-0.9, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Two coins: half the outcomes are ruled out", pad=8)

    ax = fig.add_subplot(gs[0, 1])
    dice_grid(ax, lambda p: ORANGE if sum(p) == 10 else BLUE_LIGHT,
              "Given 1st = 6: only the bottom row remains", faded=lambda p: p[0] != 6, labels=True)
    ax.add_patch(Rectangle((0, 0), 6, 1, fill=False, edgecolor=INK, linewidth=2.5, clip_on=False))

    panel(fig.add_subplot(gs[0, 2]), "The definition", [
        "P(B | A): probability of B given A happened.",
        "Discard outcomes outside A, recount inside:",
        "",
        "P(sum 10) = 3/36 = 1/12",
        "P(sum 10 | 1st 6) = 1/6   (went up)",
        "P(sum 10 | 1st 1) = 0     (impossible)",
        "",
        "Independent events: P(B | A) = P(B),",
        "the condition changes nothing.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    bars(ax, [f"1st = {k}" for k in range(1, 7)], given, "P(sum 10 | first die)", "condition",
         ylim=0.25, label_fmt=lambda v: frac_label(v, 6), highlight=5)
    ax.axhline(1 / 12, color=INK2, linestyle=":", linewidth=1.5, zorder=3)
    ax.text(-0.35, 1 / 12 + 0.006, "no information: 1/12", fontsize=10, color=INK2)

    grouped_bars(fig.add_subplot(gs[1, 1]), ["P(sum 10)", "P(sum 10 | 1st 6)"],
                 [("theory", [1 / 12, 1 / 6], BLUE_LIGHT), ("simulated", [sim_prior, sim_given], BLUE)],
                 "Simulation: condition = filter the trials", "", ylim=0.25,
                 label_fmt=lambda v: f"{v:.4f}")
    plt.gca().text(0.98, 0.97, f"500,000 rolls, {first_is_6.sum():,} kept", transform=plt.gca().transAxes,
                   ha="right", va="top", fontsize=10, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "General product rule", [
        "P(A ∩ B) = P(A) · P(B | A)   (always)",
        "",
        "P(1st 6 ∩ sum 10) = 6/36 × 1/6 = 1/36,",
        "matching the single cell (6, 4).",
        "",
        "Two aces without replacement:",
        f"4/52 × 3/51 = {4 / 52 * 3 / 51:.5f}",
        "",
        "Independence is the special case",
        "P(B | A) = P(B).",
    ], spacing=0.085)
    save(fig, name="08_conditional_probability.png")


# ================================================================ notebook 13
def sheet_13_prior_event_posterior():
    examples = [
        ("spam email", "contains \"lottery\"", 0.2, 14 / 24, "0.2", "14/24 ≈ 0.583"),
        ("disease test", "tested positive", 0.0001, 99 / 10_098, "0.0001", "≈ 0.0098"),
        ("dice sum = 10", "1st die is 6", 3 / 36, 1 / 6, "3/36", "1/6"),
        ("two heads", "1st coin is H", 1 / 4, 1 / 2, "1/4", "1/2"),
    ]
    assert abs(bayes(0.2, 0.7, 0.125) - 14 / 24) < 1e-12
    assert abs(bayes(0.0001, 0.99, 0.01) - 99 / 10_098) < 1e-12

    fig = new_figure("Prior → event → posterior: one pattern behind every update",
                     "The posterior P(A | E) is the prior P(A) recomputed in the smaller world the event E leaves.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])
    for row, (name, event, prior, posterior, prior_text, post_text) in enumerate(examples):
        y = len(examples) - 1 - row
        ax.annotate("", xy=(posterior, y), xytext=(prior, y),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color=MUTED, shrinkA=7, shrinkB=7))
        ax.plot(prior, y, "o", color=BLUE_LIGHT, markersize=13, zorder=3)
        ax.plot(posterior, y, "o", color=ORANGE, markersize=13, zorder=3)
        ax.text(prior, y - 0.3, prior_text, ha="right" if prior > 0.05 else "center", va="top",
                fontsize=10.5, color=INK2)
        ax.text(posterior, y - 0.3, post_text, ha="left" if prior > 0.05 else "center", va="top",
                fontsize=10.5, color=ORANGE, weight="bold")
        ax.text(np.sqrt(prior * posterior), y + 0.16, f"{event}  (× {posterior / prior:.0f})",
                ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_yticks(range(len(examples)), [e[0] for e in examples][::-1])
    ax.set_xscale("log")
    ax.set_xlim(3e-5, 1.5)
    ax.set_ylim(-0.75, len(examples) - 0.4)
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    ax.set_xlabel("probability (log scale)")
    ax.grid(axis="x", which="major")
    ax.set_title("Four examples: prior (light blue) → posterior (orange)")

    panel(fig.add_subplot(gs[0, 2]), "The three parts", [
        "Prior P(A): the probability before",
        "    knowing anything else.",
        "Event E: information arrives.",
        "Posterior P(A | E): recomputed",
        "    using that information.",
        "",
        "P(A)   →  event E  →   P(A | E)",
        "",
        "Same ratio, smaller world.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    dice_grid(ax, lambda p: ORANGE if sum(p) == 10 else BLUE_LIGHT, "Same event, two different conditions",
              faded=lambda p: p[0] not in (3, 6))
    for row in (3, 6):
        ax.add_patch(Rectangle((0, 6 - row), 6, 1, fill=False, edgecolor=INK, linewidth=2.2, clip_on=False))
    ax.text(6.15, 0.5, "1st = 6: 1/6", va="center", fontsize=10.5, color=ORANGE, weight="bold",
            clip_on=False)
    ax.text(6.15, 3.5, "1st = 3: 0", va="center", fontsize=10.5, color=INK2, weight="bold", clip_on=False)

    bars(fig.add_subplot(gs[1, 1]), ["prior\nP(sum 10)", "given\n1st = 6", "given\n1st = 3"],
         [3 / 36, 1 / 6, 0.0], "The posterior can go up or down", "", ylim=0.22,
         label_fmt=lambda v: frac_label(v, 36), highlight=1)
    plt.gca().text(2, 0.008, "0", ha="center", fontsize=10, color=INK)

    panel(fig.add_subplot(gs[1, 2]), "Reading the update", [
        "Spam: 20/100 of all emails, then",
        "14/24 of the emails with \"lottery\".",
        "",
        "Up: spam, the test, 1st die 6, 1st coin H.",
        "Down: 1st die 3 makes sum 10 impossible;",
        "a reply to your own email lowers P(spam).",
        "",
        "Either way, the posterior is the better-",
        "informed estimate.",
    ], spacing=0.09)
    save(fig, name="13_prior_event_posterior.png")


# ================================================================ notebook 05
def sheet_05_inclusion_exclusion():
    kids, soccer, basketball, both = 10, 6, 5, 3
    union = soccer + basketball - both
    dice = list(product(range(1, 7), repeat=2))
    sum7 = {d for d in dice if sum(d) == 7}
    diff1 = {d for d in dice if abs(d[0] - d[1]) == 1}
    assert (len(sum7), len(diff1), len(sum7 & diff1), len(sum7 | diff1)) == (6, 10, 2, 14)

    fig = new_figure("Inclusion–exclusion: adding overlapping events counts the overlap twice",
                     "P(A ∪ B) = P(A) + P(B) − P(A ∩ B).  Subtract the overlap once and the count is right.")
    gs = grid(fig, top=0.84, bottom=0.1, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), 10, 6.4, facecolor=CARD, edgecolor=MUTED))
    ax.add_patch(Circle((3.9, 3.2), 2.2, facecolor=BLUE, alpha=0.35, edgecolor=BLUE, linewidth=2))
    ax.add_patch(Circle((6.1, 3.2), 2.2, facecolor=ORANGE, alpha=0.35, edgecolor=ORANGE, linewidth=2))
    for x, text in [(2.8, f"{soccer - both}\nsoccer\nonly"), (5.0, f"{both}\nboth"),
                    (7.2, f"{basketball - both}\nbasketball\nonly")]:
        ax.text(x, 3.2, text, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(9.6, 0.4, f"{kids - union} play neither", ha="right", fontsize=11, color=INK2)
    ax.text(2.3, 5.75, f"soccer: {soccer}", ha="center", color=BLUE, fontsize=12, weight="bold")
    ax.text(7.7, 5.75, f"basketball: {basketball}", ha="center", color=ORANGE, fontsize=12, weight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{kids} kids: who plays what", pad=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axisbelow(True)
    ax.barh(1, soccer, color=BLUE, height=0.5)
    ax.barh(1, basketball - both, left=soccer, color=ORANGE, height=0.5)
    ax.barh(1, both, left=soccer + basketball - both, color=ORANGE, height=0.5, hatch="///",
            edgecolor="white", linewidth=0)
    ax.barh(0, union, color=AQUA, height=0.5)
    ax.axvline(kids, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(kids + 0.15, 1.45, "only 10 kids", fontsize=10, color=INK2)
    ax.text(soccer + basketball + 0.2, 1, f"= {soccer + basketball}\n(3 counted twice)", va="center", fontsize=10.5)
    ax.text(union + 0.2, 0, f"= {union}", va="center", fontsize=10.5)
    ax.set_yticks([1, 0], [f"{soccer} + {basketball}", f"{soccer} + {basketball} − {both}"])
    ax.set_xlim(0, 14.5)
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlabel("kids")
    ax.grid(axis="x")
    ax.set_title("Naive sum vs. inclusion–exclusion")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        "Counts:  |S ∪ B| = |S| + |B| − |S ∩ B|",
        f"            = {soccer} + {basketball} − {both} = {union}",
        "",
        "Probabilities:",
        "P(S ∪ B) = 0.6 + 0.5 − 0.3 = 0.8",
        "",
        "Disjoint events have P(A ∩ B) = 0,",
        "so the rule shrinks to P(A) + P(B).",
        "Without the overlap you cannot solve it.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    colors = {"both": ORANGE, "sum": BLUE, "diff": AQUA, "none": "#eef3fa"}
    for first, second in dice:
        key = ("both" if (first, second) in sum7 & diff1 else "sum" if (first, second) in sum7
               else "diff" if (first, second) in diff1 else "none")
        ax.add_patch(Rectangle((second - 1, 6 - first), 1, 1, facecolor=colors[key], edgecolor=SURFACE,
                               linewidth=2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(6) + 0.5, range(1, 7))
    ax.set_yticks(np.arange(6) + 0.5, range(6, 0, -1))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("second die")
    ax.set_ylabel("first die")
    ax.set_title("Two dice: sum 7 or difference 1", pad=8)
    handles = [Rectangle((0, 0), 1, 1, color=colors[k]) for k in ("sum", "diff", "both")]
    ax.legend(handles, ["sum 7 only (4)", "diff. 1 only (8)", "both (2)"], loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9.5, handlelength=1, columnspacing=0.8)

    ax = fig.add_subplot(gs[1, 1])
    bars(ax, ["naive\n6 + 10", "inclusion–exclusion\n6 + 10 − 2", "direct count\nof the union"],
         [16 / 36, 14 / 36, len(sum7 | diff1) / 36], "P(sum 7 or difference 1)", "", ylim=0.6,
         label_fmt=lambda v: f"{round(v * 36)}/36", highlight=0)

    panel(fig.add_subplot(gs[1, 2]), "Takeaways", [
        "Disjoint (mutually exclusive):",
        "    P(A ∪ B) = P(A) + P(B)",
        "Joint (can happen together):",
        "    P(A ∪ B) = P(A) + P(B) − P(A ∩ B)",
        "",
        "The naive sum (orange bar) overshoots;",
        "the corrected sum matches a direct count.",
    ], spacing=0.09)
    save(fig, name="05_inclusion_exclusion.png")


# ================================================================ notebook 07
def p_no_match(n, days=365):
    return float(np.prod((days - np.arange(n)) / days))


def sheet_07_birthday_problem():
    ns = np.arange(1, 101)
    no_match = np.array([p_no_match(n) for n in ns])
    tipping = int(ns[np.argmax(no_match < 0.5)])
    assert tipping == 23
    p23, p30 = 1 - p_no_match(23), 1 - p_no_match(30)
    you_match = 1 - (364 / 365) ** 22

    fig = new_figure("The birthday problem: 23 people are enough for a shared birthday to be likely",
                     "Compute the complement, P(all birthdays different), as a product, and subtract it from 1.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(ns, 1 - no_match, color=BLUE, linewidth=2.4)
    ax.plot(ns, no_match, color=ORANGE, linewidth=2.4)
    ax.text(62, 0.93, "P(at least one shared birthday)", color=BLUE, fontsize=11.5, weight="bold")
    ax.text(62, 0.1, "P(all birthdays different)", color=ORANGE, fontsize=11.5, weight="bold")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    for n, p in [(23, p23), (30, p30)]:
        ax.plot(n, p, "o", color=INK, markersize=7, zorder=3)
        ax.annotate(f"{n} people → {p:.1%}", (n, p), textcoords="offset points", xytext=(10, -14),
                    fontsize=11)
    ax.axvline(23, color=INK2, linestyle="--", linewidth=1.2)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of people")
    ax.set_ylabel("probability")
    ax.grid(True)
    ax.set_title("A match becomes likely surprisingly fast (365-day year)")

    panel(fig.add_subplot(gs[0, 2]), "One person at a time", [
        "Person 1: any day         365/365",
        "Person 2: avoid 1 day     364/365",
        "Person 3: avoid 2 days    363/365",
        "Person k: avoid k − 1     (366 − k)/365",
        "",
        "Independent, so multiply:",
        r"$P(\mathrm{no\ match}) = \dfrac{365}{365} \cdot \dfrac{364}{365} \cdots \dfrac{365 - n + 1}{365}$",
        "",
        "",
        "P(shared) = 1 − P(no match)",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    people = np.arange(1, 61)
    ax.plot(people, people, color=BLUE, linewidth=2.2)
    ax.plot(people, people * (people - 1) / 2, color=ORANGE, linewidth=2.2)
    ax.text(54, 110, "people", color=BLUE, fontsize=11, weight="bold")
    ax.text(30, 650, "pairs of people", color=ORANGE, fontsize=11, weight="bold")
    ax.plot(23, 253, "o", color=INK, markersize=7)
    ax.annotate("23 people = 253 pairs", (23, 253), textcoords="offset points", xytext=(-6, 14),
                fontsize=10.5, ha="center")
    ax.set_xlabel("number of people")
    ax.set_ylabel("count")
    ax.grid(True)
    ax.set_title("Why intuition fails: any pair can match")

    bars(fig.add_subplot(gs[1, 1]), ["some pair\nshares a birthday", "someone shares\nYOUR birthday"],
         [p23, you_match], "In a group of 23 people", "", ylim=0.7, label_fmt=lambda v: f"{v:.1%}",
         highlight=0)

    lines = ["`people   P(no match)   P(shared)`"]
    for n in [2, 9, 20, 23, 30, 50, 100]:
        lines.append(f"`{n:>6}   {p_no_match(n):>11.4f}   {1 - p_no_match(n):>9.4f}`")
    lines += ["", "With 366 people a match is certain."]
    panel(fig.add_subplot(gs[1, 2]), "Key values", lines, spacing=0.085)
    save(fig, name="07_birthday_problem.png")


# ================================================================ notebook 09
def sheet_09_probability_tree():
    p_s, p_r_s, p_r_ns = 0.4, 0.8, 0.5
    fig = new_figure("Probability trees: multiply along each path for dependent events",
                     "100 kids, 40 play soccer. Running shoes: 80% of soccer players but only 50% of the others.")
    gs = grid(fig, top=0.84, wspace=0.3)

    leaves = tree(fig.add_subplot(gs[0, :2]), "100 kids",
                  [("soccer", p_s), ("no soccer", 1 - p_s)],
                  {"soccer": [("shoes", p_r_s), ("no shoes", 1 - p_r_s)],
                   "no soccer": [("shoes", p_r_ns), ("no shoes", 1 - p_r_ns)]},
                  highlight=[("soccer", "shoes"), ("no soccer", "shoes")])
    assert abs(sum(leaves) - 1) < 1e-12
    p_r = p_s * p_r_s + (1 - p_s) * p_r_ns
    p_s_given_r = p_s * p_r_s / p_r

    panel(fig.add_subplot(gs[0, 2]), "How to read it", [
        "First split: the first event, with P(S).",
        "Second split: CONDITIONAL probabilities,",
        "e.g. P(shoes | soccer) = 0.8.",
        "",
        "General product rule along a path:",
        "P(S ∩ R) = P(S) · P(R | S) = 0.4 × 0.8",
        "",
        "Leaves cover every scenario once:",
        "0.32 + 0.08 + 0.30 + 0.30 = 1",
    ], spacing=0.09)

    mosaic(fig.add_subplot(gs[1, 0]), p_s, 0.5, 0.5, ["soccer", "no soccer"], "room 1", "room 2",
           "Random rooms: independent (lines align)")
    mosaic(fig.add_subplot(gs[1, 1]), p_s, p_r_s, p_r_ns, ["soccer", "no soccer"], "shoes", "no shoes",
           "Running shoes: dependent (lines don't align)")

    panel(fig.add_subplot(gs[1, 2]), "Reading the tree backwards", [
        "Add the orange leaves for P(shoes):",
        f"P(R) = 0.32 + 0.30 = {p_r:.2f}",
        "",
        "Among the kids wearing running shoes:",
        f"P(S | R) = 0.32 / {p_r:.2f} = {p_s_given_r:.3f}",
        "",
        f"Higher than P(S) = {p_s}: seeing the shoes",
        "makes soccer more likely. That reversal",
        "is Bayes' theorem (notebook 10).",
    ], spacing=0.09)
    save(fig, name="09_probability_tree.png")


# ================================================================ notebook 11
def sheet_11_bayes_formula():
    from fractions import Fraction
    prior, sens, false_pos = Fraction(1, 10_000), Fraction(99, 100), Fraction(1, 100)
    numerator = prior * sens
    evidence = numerator + (1 - prior) * false_pos
    posterior = numerator / evidence
    assert posterior == Fraction(99, 10_098)
    spam_post = bayes(0.20, 0.60, 0.04)

    fig = new_figure("Deriving Bayes' theorem in four steps",
                     "A = sick, B = diagnosed sick. Every step uses a rule from earlier notebooks.")
    top = fig.add_gridspec(1, 4, top=0.86, bottom=0.64, left=0.04, right=0.975, wspace=0.18)
    steps = [
        ("1. Conditional probability", [r"$P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$", "", "",
                                         "Need the top and", "the bottom."]),
        ("2. Numerator: product rule", [r"$P(A \cap B) = P(A)\,P(B \mid A)$", "",
                                        "sick, times correctly", "diagnosed given sick"]),
        ("3. Denominator: total prob.", [r"$P(B) = P(A)\,P(B \mid A)$", r"$\quad + P(A')\,P(B \mid A')$", "",
                                         "two disjoint groups:", "sick or healthy"]),
        ("4. Bayes' theorem", [r"$P(A \mid B) = \dfrac{P(A)\,P(B \mid A)}{P(B)}$", "", "",
                               "denominator from step 3"]),
    ]
    cards = []
    for k, (heading, lines) in enumerate(steps):
        cards.append(fig.add_subplot(top[0, k]))
        panel(cards[-1], heading, lines, fontsize=12.5, spacing=0.12)
    for left, right in zip(cards, cards[1:]):
        a, b = left.get_position(), right.get_position()
        fig.text((a.x1 + b.x0) / 2, (a.y0 + a.y1) / 2, "→", fontsize=24, color=ORANGE,
                 ha="center", va="center", weight="bold")

    bottom = fig.add_gridspec(1, 3, top=0.51, bottom=0.07, left=0.06, right=0.975, wspace=0.3)
    ax = fig.add_subplot(bottom[0, 0])
    per_million = 1_000_000
    sick_pos, healthy_pos = float(numerator) * per_million, float(evidence - numerator) * per_million
    ax.barh(0, sick_pos, color=ORANGE, height=0.5)
    ax.barh(0, healthy_pos, left=sick_pos, color=BLUE_LIGHT, height=0.5)
    ax.set_xlim(0, (sick_pos + healthy_pos) * 1.02)
    ax.set_ylim(-0.6, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("people diagnosed sick, out of 1,000,000")
    ax.text(0, 0.45, f"P(A ∩ B) = 0.000099: {sick_pos:.0f} sick", fontsize=10.5, color=ORANGE,
            weight="bold", va="bottom")
    ax.text((sick_pos + healthy_pos) * 0.98, -0.42, f"P(A' ∩ B) = 0.009999: {healthy_pos:,.0f} healthy",
            fontsize=10.5, color=INK2, ha="right")
    ax.set_title("P(B) = 0.010098 is made of two pieces")

    panel(fig.add_subplot(bottom[0, 1]), "Plug in the numbers", [
        r"$P(A \mid B) = \dfrac{0.0001 \times 0.99}{0.0001 \times 0.99 + 0.9999 \times 0.01}$",
        "",
        "",
        r"$= \dfrac{0.000099}{0.010098} \approx 0.0098$",
        "",
        "",
        "With exact fractions: 99/10,098 (= 1/102),",
        "the same counts as the million-people table.",
    ], spacing=0.09)
    grouped_bars(fig.add_subplot(bottom[0, 2]), ["disease test", "spam: word 'free'"],
                 [("prior", [0.0001, 0.20], BLUE_LIGHT), ("posterior", [float(posterior), spam_post], BLUE)],
                 "One formula, any domain", "", ylim=1.0, label_fmt=lambda v: f"{v:.4f}")
    save(fig, name="11_bayes_formula.png")


# ================================================================ notebook 12
def email_grid(ax, kinds, columns, title):
    """kinds: list of (is_spam, has_word). Spam orange, ham blue; emails without the word faded."""
    for k, (spam, word) in enumerate(kinds):
        r, c = divmod(k, columns)
        ax.add_patch(Rectangle((c, -r), 0.82, 0.82, facecolor=ORANGE if spam else BLUE,
                               alpha=1.0 if word else 0.18, linewidth=0))
    rows = (len(kinds) + columns - 1) // columns
    ax.set_xlim(-0.2, columns)
    ax.set_ylim(-rows + 0.6, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=8)


def sheet_12_bayes_spam():
    spam_word, spam_no, ham_word, ham_no = 14, 6, 10, 70
    prior = (spam_word + spam_no) / 100
    p_w_spam, p_w_ham = spam_word / 20, ham_word / 80
    posterior = bayes(prior, p_w_spam, p_w_ham)
    posterior_without = bayes(prior, 1 - p_w_spam, 1 - p_w_ham)
    assert abs(posterior - 14 / 24) < 1e-12

    fig = new_figure("Bayes' theorem as deletion: the spam example",
                     "100 emails, 20 of them spam. Keep only the emails containing \"lottery\", then count.")
    gs = grid(fig, top=0.84, wspace=0.3)
    kinds = ([(True, True)] * spam_word + [(True, False)] * spam_no +
             [(False, True)] * ham_word + [(False, False)] * ham_no)
    ax = fig.add_subplot(gs[0, 0])
    email_grid(ax, kinds, 10, "All 100 emails")
    handles = [Rectangle((0, 0), 1, 1, color=ORANGE), Rectangle((0, 0), 1, 1, color=BLUE),
               Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25)]
    ax.legend(handles, ["spam, \"lottery\"", "ham, \"lottery\"", "faded: no \"lottery\""],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5, handlelength=1,
              columnspacing=0.8)

    ax = fig.add_subplot(gs[0, 1])
    email_grid(ax, [(True, True)] * spam_word + [(False, True)] * ham_word, 6,
               f"After deleting the rest: {spam_word + ham_word} emails")
    ax.text(3, -4.6, f"{spam_word} spam out of {spam_word + ham_word}  →  "
            f"{spam_word}/{spam_word + ham_word} = {posterior:.3f}", ha="center", fontsize=12, weight="bold")

    bars(fig.add_subplot(gs[0, 2]), ["prior\nP(spam)", "posterior\nP(spam | lottery)"], [prior, posterior],
         "Seeing \"lottery\" updates the belief", "", ylim=1.0, label_fmt=lambda v: f"{v:.3f}", highlight=1)

    panel(fig.add_subplot(gs[1, :2]), "The same answer from the formula", [
        f"`prior       P(spam)           = 20/100 = {prior}`",
        f"`likelihood  P(lottery | spam) = 14/20  = {p_w_spam}`",
        f"`likelihood  P(lottery | ham)  = 10/80  = {p_w_ham}`",
        "",
        r"$P(\mathrm{spam} \mid \mathrm{lottery}) = \dfrac{0.2 \times 0.7}{0.2 \times 0.7 + 0.8 \times 0.125}"
        r" = \dfrac{0.14}{0.24} \approx 0.583$",
        "",
        "",
        "The formula is the deletion, done with proportions instead of counts.",
    ], spacing=0.095)
    panel(fig.add_subplot(gs[1, 2]), "A one-word classifier", [
        "Classify as spam if P(spam | email) > 0.5.",
        "",
        f"Contains \"lottery\":  {posterior:.3f} → spam",
        f"No \"lottery\":         {posterior_without:.3f} → ham",
        "",
        "Without the word, use the complements:",
        "P(no lottery | spam) = 0.3,",
        "P(no lottery | ham) = 0.875.",
    ], spacing=0.09)
    save(fig, name="12_bayes_spam_deletion.png")


# ================================================================ notebook 14
def sheet_14_naive_bayes():
    p_spam = 0.2
    words = {"lottery": (14 / 20, 10 / 80), "winning": (15 / 20, 8 / 80)}

    def naive_bayes(present):
        spam_term, ham_term = p_spam, 1 - p_spam
        for w in present:
            spam_term *= words[w][0]
            ham_term *= words[w][1]
        return spam_term / (spam_term + ham_term), spam_term, ham_term

    p_both, spam_both, ham_both = naive_bayes(["lottery", "winning"])
    assert abs(p_both - 0.105 / 0.115) < 1e-9

    fig = new_figure("Naive Bayes: multiply the evidence from each word",
                     "Assume words are independent within each class, then every factor is a simple one-word count.")
    gs = grid(fig, top=0.84, wspace=0.32)

    bars(fig.add_subplot(gs[0, 0]), ["prior", "lottery", "winning", "both"],
         [p_spam, naive_bayes(["lottery"])[0], naive_bayes(["winning"])[0], p_both],
         "P(spam | words seen)", "words in the email", ylim=1.1, label_fmt=lambda v: f"{v:.3f}",
         highlight=3)

    steps = ["prior", "× lottery", "× winning"]
    spam_terms = [p_spam, p_spam * words["lottery"][0], spam_both]
    ham_terms = [1 - p_spam, (1 - p_spam) * words["lottery"][1], ham_both]
    grouped_bars(fig.add_subplot(gs[0, 1]), steps,
                 [("spam side", spam_terms, ORANGE), ("ham side", ham_terms, BLUE)],
                 "Each word multiplies both sides", "", ylim=1.0, label_fmt=lambda v: f"{v:.3g}")
    plt.gca().legend(loc="upper right", fontsize=10)

    panel(fig.add_subplot(gs[0, 2]), "The naive assumption", [
        "Given the class, words are independent:",
        "P(lottery ∩ winning | spam)",
        "    ≈ P(lottery | spam) · P(winning | spam)",
        "",
        "Both words:",
        f"spam: 0.2 × 0.7 × 0.75 = {spam_both:.3f}",
        f"ham:  0.8 × 0.125 × 0.1 = {ham_both:.3f}",
        f"P(spam | both) = {spam_both:.3f} / {spam_both + ham_both:.3f} = {p_both:.3f}",
    ], spacing=0.09)

    k = np.arange(1, 11)
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(k, 20 * 0.7 ** k, "o-", color=ORANGE, linewidth=2, markersize=6, label="spam emails")
    ax.semilogy(k, 80 * 0.125 ** k, "o-", color=BLUE, linewidth=2, markersize=6, label="ham emails")
    ax.axhline(1, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(10, 1.4, "one email", ha="right", fontsize=10, color=INK2)
    ax.set_xlabel("number of words required together")
    ax.set_ylabel("expected emails (log scale)")
    ax.set_xticks(k)
    ax.grid(True, which="major")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Direct counting runs out of emails")

    ax = fig.add_subplot(gs[1, 1])
    naive = 0.2 * 0.7 ** k / (0.2 * 0.7 ** k + 0.8 * 0.125 ** k)
    ax.plot(k, naive, "o-", color=BLUE, linewidth=2, markersize=6)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k)
    ax.set_xlabel("number of spammy words present")
    ax.set_ylabel("P(spam | words)")
    ax.grid(True)
    ax.set_title("Naive Bayes still gives an answer")

    panel(fig.add_subplot(gs[1, 2]), "Why it is used anyway", [
        "With 100 emails, almost none contain",
        "5 given words at once, so P(all words | spam)",
        "cannot be counted directly.",
        "",
        "One-word counts are always available.",
        "The independence assumption is false",
        "(\"good\" and \"morning\" travel together),",
        "but the classifier works surprisingly well.",
    ], spacing=0.09)
    save(fig, name="14_naive_bayes.png")


# ================================================================ notebook 10
def bayes_posterior(p_sick, p_pos_given_sick, p_pos_given_healthy):
    numerator = p_sick * p_pos_given_sick
    return numerator / (numerator + (1 - p_sick) * p_pos_given_healthy)


def sheet_10_bayes_disease_test():
    population = 1_000_000
    prevalence = 1 / 10_000
    effectiveness = 0.99
    sick = round(population * prevalence)
    healthy = population - sick
    true_pos = round(sick * effectiveness)
    false_neg = sick - true_pos
    false_pos = round(healthy * (1 - effectiveness))
    true_neg = healthy - false_pos
    positives = true_pos + false_pos
    posterior = true_pos / positives
    assert (true_pos, false_pos, positives) == (99, 9_999, 10_098)
    assert np.isclose(posterior, bayes_posterior(prevalence, 0.99, 0.01))
    update_factor = effectiveness / (positives / population)
    after_two = bayes_posterior(posterior, 0.99, 0.01)

    fig = new_figure("Bayes' theorem: why a positive test for a rare disease is usually a false alarm",
                     "1,000,000 people, a disease affecting 1 in 10,000, and a test that is 99% effective "
                     "for sick and healthy people alike.")
    gs = fig.add_gridspec(2, 3, top=0.85, bottom=0.06, left=0.04, right=0.975,
                          hspace=0.42, wspace=0.25, width_ratios=[1.6, 0.9, 1])

    # --- icon array: every positive test is one cell
    ax = fig.add_subplot(gs[0, 0])
    columns, rows = 102, 99
    assert columns * rows == positives
    people = np.zeros(positives)
    people[:true_pos] = 1  # the bottom row holds the 99 sick people
    ax.imshow(people.reshape(rows, columns), origin="lower", cmap=ListedColormap([BLUE_LIGHT, ORANGE]),
              interpolation="nearest", extent=(-0.5, columns - 0.5, -0.5, rows - 0.5))
    ax.axis("off")
    ax.set_title(f"All {positives:,} positive tests, one cell per person", pad=8)
    ax.text(columns + 4, rows * 0.82, f"{false_pos:,} healthy\n(false positives)", va="center",
            fontsize=11.5, color=INK2)
    ax.text(columns + 4, rows * 0.62, f"{true_pos} sick: the thin\norange row at\nthe bottom", va="center",
            fontsize=11.5, color=ORANGE, weight="bold")

    zoom_cols, zoom_rows = 22, 9
    inset = ax.inset_axes([1.08, -0.02, 0.62, 0.62 * zoom_rows / zoom_cols * 1.0])
    for r in range(zoom_rows):
        for c in range(zoom_cols):
            inset.add_patch(Rectangle((c - 0.36, r - 0.36), 0.72, 0.72, linewidth=0,
                                      facecolor=ORANGE if r == 0 else BLUE_LIGHT))
    inset.set_xlim(-0.6, zoom_cols - 0.4)
    inset.set_ylim(-0.6, zoom_rows - 0.4)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(INK2)
    inset.set_title("zoom: 1 square = 1 person", fontsize=10.5, weight="normal", color=INK2, pad=4)
    ax.indicate_inset_zoom(inset, edgecolor=INK2, alpha=0.9)

    # --- four-group table
    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    cells = [["Sick", f"{true_pos:,}  (true positive)", f"{false_neg:,}  (missed)", f"{sick:,}"],
             ["Healthy", f"{false_pos:,}  (false positive)", f"{true_neg:,}  (correct)", f"{healthy:,}"],
             ["Total", f"{positives:,}", f"{false_neg + true_neg:,}", f"{population:,}"]]
    table = ax.table(cellText=cells, colLabels=["", "Diagnosed sick", "Diagnosed healthy", "Total"],
                     colWidths=[0.16, 0.32, 0.32, 0.2], cellLoc="center", bbox=[0, 0.25, 1, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0 or c == 0 or r == 3:
            cell.set_text_props(weight="bold", color=INK)
        if c == 1 and r > 0:
            cell.set_facecolor("#fdf0ea")
    ax.set_title("The four groups (counts out of 1,000,000)", pad=4)
    ax.text(0.5, 0.08, f"You tested positive, so you are somewhere in the shaded column: "
            f"{true_pos} sick out of {positives:,}.\n"
            f"P(sick | positive) = {true_pos} / {positives:,} = {posterior:.4f}, less than 1%.",
            transform=ax.transAxes, ha="center", fontsize=12.5, color=INK)

    # --- probability tree
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("The probability tree: multiply along each path", pad=8)
    root, nodes = (0.9, 3.0), {"sick": (4.0, 4.6), "healthy": (4.0, 1.4)}
    leaves = {("sick", True): (7.2, 5.4), ("sick", False): (7.2, 3.8),
              ("healthy", True): (7.2, 2.2), ("healthy", False): (7.2, 0.6)}
    leaf_counts = {("sick", True): true_pos, ("sick", False): false_neg,
                   ("healthy", True): false_pos, ("healthy", False): true_neg}

    def edge(a, b, label):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, linewidth=1.5, zorder=1)
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.22, label, ha="center", fontsize=10.5, color=INK2)

    edge(root, nodes["sick"], "1/10,000")
    edge(root, nodes["healthy"], "9,999/10,000")
    for (group, positive), pos in leaves.items():
        edge(nodes[group], pos, "0.99" if positive == (group == "sick") else "0.01")
    ax.text(*root, "1,000,000\npeople", ha="right", va="center", fontsize=10.5, color=INK)
    for group, pos in nodes.items():
        ax.add_patch(FancyBboxPatch((pos[0] - 0.75, pos[1] - 0.28), 1.5, 0.56, boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor=MUTED, zorder=2))
        ax.text(*pos, f"{group}\n{sick if group == 'sick' else healthy:,}", ha="center", va="center",
                fontsize=9.5, zorder=3)
    for key, pos in leaves.items():
        positive = key[1]
        label = f"{'+' if positive else '−'} test: {leaf_counts[key]:,}"
        ax.text(pos[0] + 0.2, pos[1], label, va="center", fontsize=11,
                color=ORANGE if key == ("sick", True) else (INK if positive else MUTED),
                weight="bold" if positive else "normal")

    # --- Bayes formula
    panel(fig.add_subplot(gs[1, 1]), "Same answer, from the formula", [
        r"$P(S \mid +) = \dfrac{P(S)\,P(+ \mid S)}{P(S)\,P(+ \mid S) + P(S')\,P(+ \mid S')}$",
        "",
        "",
        r"$= \dfrac{0.0001 \times 0.99}{0.0001 \times 0.99 + 0.9999 \times 0.01} \approx 0.0098$",
        "",
        "",
        "posterior = prior × likelihood / evidence",
        "",
        f"A positive test multiplies the prior",
        f"by about {update_factor:.0f}, but {update_factor:.0f} × 0.0001 is still",
        "only about 0.0098. A strong test",
        "cannot rescue a tiny prior.",
    ], spacing=0.08)

    # --- posterior vs prevalence
    ax = fig.add_subplot(gs[1, 2])
    prevalences = np.logspace(-5, -0.3, 300)
    ax.semilogx(prevalences, bayes_posterior(prevalences, 0.99, 0.01), color=BLUE, linewidth=2.2)
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    for p, text, offset in [(prevalence, f"1 in 10,000 → {posterior:.2%}", (8, 10)),
                            (0.01, f"1 in 100 → {bayes_posterior(0.01, 0.99, 0.01):.0%}", (-12, 10))]:
        value = bayes_posterior(p, 0.99, 0.01)
        ax.plot(p, value, "o", color=ORANGE, markersize=8, zorder=3)
        ax.annotate(text, (p, value), textcoords="offset points", xytext=offset, fontsize=10.5,
                    ha="left" if offset[0] > 0 else "right", color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_title("P(sick | positive) vs. prevalence")
    ax.set_xlabel("prevalence P(sick), log scale")
    ax.set_ylabel("P(sick | positive)")
    ax.grid(True, which="major")
    ax.text(0.02, 0.97, f"A second positive test raises it to {after_two:.2f}", transform=ax.transAxes,
            fontsize=10.5, color=INK2, va="top")
    save(fig, name="10_bayes_disease_test.png")


if __name__ == "__main__":
    sheet_01_what_is_probability()
    sheet_02_dice_example()
    sheet_03_complement_rule()
    sheet_04_sum_rule_disjoint()
    sheet_05_inclusion_exclusion()
    sheet_06_independence_product_rule()
    sheet_07_birthday_problem()
    sheet_08_conditional_probability()
    sheet_09_probability_tree()
    sheet_10_bayes_disease_test()
    sheet_11_bayes_formula()
    sheet_12_bayes_spam()
    sheet_13_prior_event_posterior()
    sheet_14_naive_bayes()
