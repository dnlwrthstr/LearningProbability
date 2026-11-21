#!/usr/bin/env python3
"""Concept sheets for lesson 01 (probability distributions).

One page per notebook that summarizes its central idea. Every number is
computed here from the same parameters and random seeds the notebook uses
(each notebook starts with np.random.default_rng(seed=42); the draws below
follow the notebook's own draw order so simulated numbers match its output).
Images are written to coursera/01_probability_distribution/img/.

Run from anywhere:  python scripts/make_lesson01_concept_images.py
"""
from functools import partial
from itertools import product
from math import comb, erf, factorial, gamma, perm, sqrt

import numpy as np
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, frac_label, heat, bars, grouped_bars, panel, trapezoid)
from viz_style import save as _save

LESSON = REPO / "coursera" / "01_probability_distribution"
save = partial(_save, folder=LESSON / "img")


# ================================================================ helpers
def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))


def binomial_pmf(n, p):
    k = np.arange(n + 1)
    return k, np.array([comb(n, int(i)) * p ** i * (1 - p) ** (n - i) for i in k])


def chi2_pdf(x, k):
    x = np.asarray(x, dtype=float)
    safe = np.where(x > 0, x, 1.0)
    return np.where(x > 0, safe ** (k / 2 - 1) * np.exp(-safe / 2) / (2 ** (k / 2) * gamma(k / 2)), 0.0)


def call_center_pdf(n_points):
    """The notebooks' gamma-shaped call-duration density on [0, 5] (shape 4, scale 0.55)."""
    xs = np.linspace(0, 5, n_points)
    pdf = xs ** 3 * np.exp(-xs / 0.55)
    return xs, pdf / trapezoid(pdf, xs)


def area(xs, ys, lo, hi):
    mask = (xs >= lo) & (xs <= hi)
    return trapezoid(ys[mask], xs[mask])


def bracket(ax, x, y0, y1, text, color=ORANGE, side="right"):
    """A vertical double arrow between two heights with a label."""
    ax.annotate("", (x, y1), (x, y0), arrowprops=dict(arrowstyle="<->", color=color, linewidth=2))
    dx = 0.12 if side == "right" else -0.12
    ax.text(x + dx, (y0 + y1) / 2, text, ha="left" if side == "right" else "right", va="center",
            fontsize=10.5, color=color, weight="bold")


# ================================================================ notebook 02
def sheet_02_pmf():
    def pmf_heads(n_coins):
        outcomes = list(product("HT", repeat=n_coins))
        counts = np.bincount([o.count("H") for o in outcomes], minlength=n_coins + 1)
        return counts, counts / len(outcomes)

    rng = np.random.default_rng(seed=42)
    sim = rng.integers(0, 2, size=(100_000, 5)).sum(axis=1)
    sim_freq = np.array([(sim == k).mean() for k in range(6)])
    counts5, pmf5 = pmf_heads(5)
    assert counts5.tolist() == [1, 5, 10, 10, 5, 1]
    assert f"{sim_freq[2]:.4f}" == "0.3155" and f"{sim_freq[0]:.4f}" == "0.0311"

    fig = new_figure("The PMF: group equally likely outcomes by the value of X",
                     "X = number of heads. Middle values are likelier only because more outcomes "
                     "land on them: p(x) = (number of outcomes with X = x) / (all outcomes).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    outcomes = ["".join(o) for o in product("HT", repeat=3)]
    for k in range(4):
        stack = [o for o in outcomes if o.count("H") == k]
        for i, o in enumerate(stack):
            ax.add_patch(Rectangle((k - 0.4, i), 0.8, 0.88, facecolor=BLUE if k in (1, 2) else BLUE_LIGHT,
                                   edgecolor=SURFACE, linewidth=2))
            ax.text(k, i + 0.44, o, ha="center", va="center", fontsize=12,
                    color="white" if k in (1, 2) else INK)
        ax.text(k, len(stack) + 0.15, f"{len(stack)}/8", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 3.8)
    ax.set_xticks(range(4))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("X = number of heads in 3 tosses")
    ax.set_title("The 8 outcomes of 3 tosses, stacked by X")

    counts3, pmf3 = pmf_heads(3)
    bars(fig.add_subplot(gs[0, 1]), range(4), pmf3, "PMF of 3 coins: counts 1, 3, 3, 1 over 8",
         "x (heads)", ylabel="p(x) = P(X = x)", ylim=0.5)

    panel(fig.add_subplot(gs[0, 2]), "The probability mass function", [
        "For a discrete random variable X:",
        r"$p(x) = P(X = x)$",
        "",
        "It holds all the information about how",
        "probability spreads over X's values.",
        "",
        "Two requirements:",
        r"1. non-negative: $p(x) \geq 0$ for every x",
        r"2. normalized: $\sum_x\, p(x) = 1$",
        "",
        "",
        "Any function meeting both is a valid PMF.",
    ], spacing=0.08)

    sizes = gs[1, 0].subgridspec(1, 2, wspace=0.25)
    for i, n in enumerate([4, 5]):
        counts, pmf = pmf_heads(n)
        ax = fig.add_subplot(sizes[i])
        ax.bar(range(n + 1), pmf, width=0.65, color=BLUE, zorder=2)
        for k, c in enumerate(counts):
            ax.text(k, pmf[k] + 0.012, str(c), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(n + 1))
        ax.set_ylim(0, 0.45)
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"{n} coins: counts over {2 ** n}", fontsize=11)
        ax.set_xlabel("x (heads)")
        if i == 0:
            ax.set_ylabel("p(x)")
        else:
            ax.set_yticklabels([])

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(6), pmf5, width=0.62, color=BLUE_LIGHT, zorder=2, label="exact PMF (count / 32)")
    ax.plot(range(6), sim_freq, "o", color=ORANGE, markersize=9, zorder=3, label="100,000 simulated runs")
    ax.set_xticks(range(6))
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x (heads in 5 tosses)")
    ax.set_ylabel("probability")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("The PMF is what a histogram converges to")

    lines = ["`x   simulated   exact`"]
    lines += [f"`{k}     {sim_freq[k]:.4f}     {pmf5[k]:.4f}`" for k in range(6)]
    lines += ["", "3, 4 and 5 coins share one shape:",
              "the binomial distribution (next notebook).",
              "The 10 = ways to place 2 heads in 5 tosses."]
    panel(fig.add_subplot(gs[1, 2]), "Simulation vs. PMF, 5 coins", lines, spacing=0.08)
    save(fig, name="02_probability_mass_function.png")


# ================================================================ notebook 03
def sheet_03_binomial():
    rng = np.random.default_rng(seed=42)
    heads = (rng.random((200_000, 5)) < 0.5).sum(axis=1)
    ones = (rng.integers(1, 7, size=(200_000, 10)) == 1).sum(axis=1)
    sim_ones3 = (ones == 3).mean()
    _, pmf10 = binomial_pmf(10, 1 / 6)
    _, pmf_fair = binomial_pmf(5, 0.5)
    p_three_ones_5 = comb(5, 3) * (1 / 6) ** 3 * (5 / 6) ** 2
    assert factorial(5) // (factorial(2) * factorial(3)) == comb(5, 2) == 10
    assert np.isclose(pmf_fair[2], 10 / 32) and f"{p_three_ones_5:.4f}" == "0.0322"
    assert f"{pmf10[3]:.4f}" == "0.1550" and f"{sim_ones3:.4f}" == "0.1538"
    assert f"{(heads == 2).mean():.4f}" == "0.3130"

    fig = new_figure("The binomial distribution: successes in n independent trials",
                     "X ~ Binomial(n, p): number of orderings with x successes × the probability "
                     "of any one of them.")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "Counting orderings", [
        "How many 5-flip sequences have 2 heads?",
        "5! orders the coins, but heads are alike",
        "(÷ 2!) and tails are alike (÷ 3!):",
        r"$\binom{5}{2} = \dfrac{5!}{2!\,3!} = 10$",
        "",
        "",
        r"Symmetry: $\binom{n}{k} = \binom{n}{n-k}$,  e.g.",
        r"$\binom{10}{3} = \binom{10}{7} = 120$",
        "",
        "Choosing k heads = choosing n − k tails.",
    ], spacing=0.085)

    bars(fig.add_subplot(gs[0, 1]), range(6), pmf_fair, "Binomial(5, 0.5): P(X = 2) = 10/32 = 0.3125",
         "x (heads in 5 fair flips)", ylabel="p(x)", ylim=0.42, label_fmt=lambda v: f"{round(v * 32)}/32",
         highlight=2)

    ax = fig.add_subplot(gs[0, 2])
    bars(ax, range(11), pmf10, "Binomial(10, 1/6): ones in ten dice throws", "x (number of 1s)",
         ylabel="p(x)", color=BLUE, ylim=0.42, label_fmt=lambda v: f"{v:.3f}" if v >= 0.005 else "",
         highlight=3)
    ax.text(3.5, 0.3, f"P(X = 3) = {pmf10[3]:.4f}\nsimulated: {sim_ones3:.4f}", ha="left", fontsize=10.5,
            color=ORANGE, weight="bold")

    shapes = gs[1, 0].subgridspec(1, 3, wspace=0.18)
    for i, (p, name) in enumerate([(0.5, "fair: p = 0.5"), (0.3, "biased: p = 0.3"), (1 / 6, "die: p = 1/6")]):
        k, pmf = binomial_pmf(5, p)
        ax = fig.add_subplot(shapes[i])
        ax.bar(k, pmf, width=0.7, color=[BLUE, AQUA, ORANGE][i], zorder=2)
        ax.set_xticks(k)
        ax.set_ylim(0, 0.45)
        ax.grid(axis="y", zorder=0)
        ax.set_title(name, fontsize=11.5)
        ax.set_xlabel("x", fontsize=10)
        if i == 0:
            ax.set_ylabel("p(x), n = 5")
        else:
            ax.set_yticklabels([])
    fig.text(0.05 + (0.975 - 0.05) / 6 - 0.012, 0.425, "Changing p shifts and skews the PMF",
             ha="center", fontsize=13, weight="bold")

    tri = np.zeros((6, 7))
    labels = [["" for _ in range(7)] for _ in range(6)]
    for n in range(1, 7):
        for k in range(n + 1):
            tri[n - 1, k] = comb(n, k)
            labels[n - 1][k] = str(comb(n, k))
    heat(fig.add_subplot(gs[1, 1]), tri, [f"n = {n}" for n in range(1, 7)], range(7),
         "Fair-coin numerators: Pascal's triangle", "", "k", labels=labels, vmax=24,
         highlight_cells=[(4, 2)])

    panel(fig.add_subplot(gs[1, 2]), "The PMF and its parameters", [
        r"$p(x) = \binom{n}{x}\, p^x (1-p)^{n-x}$,   x = 0, …, n",
        "",
        "n = number of trials, p = success probability.",
        "",
        "A die is a biased coin: \"1\" vs \"not 1\",",
        "p = 1/6. Three 1s in 5 throws:",
        r"$10 \cdot (1/6)^3 (5/6)^2$" + f" = {p_three_ones_5:.4f}",
        "",
        "Simulation (200,000 runs) matches the",
        f"formula: P(X = 2) for 5 fair flips {(heads == 2).mean():.4f}.",
    ], spacing=0.08)
    save(fig, name="03_binomial_distribution.png")


# ================================================================ notebook 04
def sheet_04_binomial_coefficient():
    n, k = 5, 2
    assert perm(n, k) == 20 and factorial(k) == 2 and comb(n, k) == 10 and comb(5, 0) == 1
    ks = np.arange(6)
    counts = np.array([comb(5, int(i)) for i in ks])
    seq_prob = 0.3 ** ks * 0.7 ** (5 - ks)
    pmf = counts * seq_prob
    assert np.isclose(pmf.sum(), 1) and f"{seq_prob[2]:.5f}" == "0.03087" and f"{seq_prob[5]:.5f}" == "0.00243"
    rng = np.random.default_rng(seed=42)
    heads = (rng.random((200_000, 5)) < 0.3).sum(axis=1)
    sim = np.array([(heads == i).mean() for i in ks])
    assert f"{sim[0]:.4f}" == "0.1692" and f"{sim[1]:.4f}" == "0.3584"

    fig = new_figure("The binomial coefficient: ordered picks ÷ k! orderings",
                     r"$\binom{n}{k}$ counts the unordered ways to pick k of n. "
                     "Times one sequence's probability, it rebuilds the binomial PMF for any coin.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    pairs = sorted({tuple(sorted(pr)) for pr in product(range(1, 6), repeat=2) if pr[0] != pr[1]})
    for i, (a, b) in enumerate(pairs):
        for j, (x, y) in enumerate([(a, b), (b, a)]):
            ax.add_patch(Rectangle((i - 0.44, j), 0.88, 0.88, facecolor=BLUE if j == 0 else BLUE_LIGHT,
                                   edgecolor=SURFACE, linewidth=1.5))
            ax.text(i, j + 0.44, f"{x},{y}", ha="center", va="center", fontsize=9.5,
                    color="white" if j == 0 else INK)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 2.3)
    ax.set_xticks(range(10), ["{%d,%d}" % pr for pr in pairs], fontsize=8.5, rotation=45)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("the 10 unordered pairs")
    ax.set_title("20 ordered picks of 2 from {1..5} ÷ 2! = 10")

    panel(fig.add_subplot(gs[0, 1]), "Three steps, n = 5, k = 2", [
        "1. Ordered picks: n(n − 1)⋯(n − k + 1)",
        f"    5 · 4 = {perm(5, 2)}",
        f"2. Orderings of each set: k! = 2! = {factorial(2)}",
        f"    (4! = {factorial(4)} orderings of {{1, 2, 3, 4}})",
        "3. Divide:",
        r"$\binom{n}{k} = \dfrac{n(n-1)\cdots(n-k+1)}{k!} = \dfrac{n!}{k!\,(n-k)!}$",
        "",
        "",
        r"Choosing nothing: $\binom{5}{0} = \dfrac{5!}{5!\,0!} = 1$  (0! = 1)",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[0, 2])
    ordered = [perm(5, int(i)) for i in ks]
    grouped_bars(ax, ks, [("ordered: 5!/(5 − k)!", ordered, BLUE_LIGHT), ("unordered: C(5, k)", counts, BLUE)],
                 "Picking k of 5: ordered vs. unordered", "k", ylim=150, label_fmt=lambda v: str(int(v)))
    ax.set_ylabel("number of selections")
    for i in ks[2:]:
        ax.text(i, ordered[i] + 17, f"÷{factorial(int(i))}", ha="center", fontsize=9.5, color=ORANGE,
                weight="bold")

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(ks, seq_prob, width=0.62, color=MUTED, zorder=2)
    for i in ks:
        ax.text(i, seq_prob[i] + 0.004, f"{seq_prob[i]:.5f}", ha="center", fontsize=9.5)
    ax.axhline(1 / 32, color=BLUE, linestyle="--", linewidth=1.8, label="fair coin: 1/32 for every sequence")
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("k = heads in the sequence")
    ax.set_ylabel(r"$0.3^k \cdot 0.7^{5-k}$")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Biased coin (p = 0.3): one sequence's probability")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(ks, pmf, width=0.62, color=ORANGE, zorder=2, label="C(5, k) × sequence probability")
    ax.plot(ks, sim, "o", color=INK, markersize=6, zorder=3, label="200,000 simulated runs")
    for i in ks:
        ax.text(i, pmf[i] + 0.012, f"{pmf[i]:.4f}", ha="center", fontsize=9.5)
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.48)
    ax.set_xlabel("k (heads in 5 flips)")
    ax.set_ylabel("P(X = k)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Binomial(5, 0.3) = count × sequence probability")

    lines = ["`k   per-seq.   ×  #  =  P(X=k)`"]
    lines += [f"`{i}   {seq_prob[i]:.5f}   × {counts[i]:>2} =  {pmf[i]:.5f}`" for i in ks]
    lines += [f"`              sum =  {pmf.sum():.4f}`", "",
              "Counts alone no longer give probabilities,",
              "but all sequences with k heads still share one."]
    panel(fig.add_subplot(gs[1, 2]), "The PMF, assembled", lines, spacing=0.075)
    save(fig, name="04_binomial_coefficient.png")


# ================================================================ notebook 07
def sheet_07_pdf():
    def uniform_interval_prob(lo, hi, a=0, b=5):
        lo, hi = max(lo, a), min(hi, b)
        return max(hi - lo, 0) / (b - a)

    assert uniform_interval_prob(2, 3) == 0.2 and uniform_interval_prob(2, 2.5) == 0.1
    rng = np.random.default_rng(seed=42)
    calls = rng.uniform(0, 5, size=500_000)
    sim_23 = ((calls >= 2) & (calls <= 3)).mean()
    sim_225 = ((calls >= 2) & (calls <= 2.5)).mean()
    assert f"{sim_23:.4f}" == "0.2001" and f"{sim_225:.4f}" == "0.1001"
    xs, pdf = call_center_pdf(500)
    p12, p45 = area(xs, pdf, 1, 2), area(xs, pdf, 4, 5)
    assert f"{p12:.2f}" == "0.39" and f"{p45:.2f}" == "0.05"

    fig = new_figure("The PDF: probabilities are areas, not heights",
                     "For a continuous X, P(a ≤ X ≤ b) = area under f between a and b. "
                     "A single point is a line segment: area, and probability, 0.")
    gs = grid(fig, top=0.84, wspace=0.3)

    for col, (width, (lo, hi), title) in enumerate([(1.0, (2, 3), "Five 1-min windows: P(2 ≤ X ≤ 3) = 0.2"),
                                                    (0.5, (2, 2.5), "Ten 30-s windows: P(2 ≤ X ≤ 2.5) = 0.1")]):
        ax = fig.add_subplot(gs[0, col])
        for e in np.arange(0, 5, width):
            selected = lo <= e < hi
            ax.add_patch(Rectangle((e, 0), width, 0.2, facecolor=ORANGE if selected else BLUE_LIGHT,
                                   edgecolor=SURFACE, linewidth=2))
        ax.text((lo + hi) / 2, 0.215, f"area = {width:g} × 0.2 = {width * 0.2:.1f}", ha="center", va="bottom",
                fontsize=10.5, color=ORANGE, weight="bold")
        ax.axhline(0.2, color=INK2, linestyle=":", linewidth=1.2)
        ax.text(5.25, 0.2, "height 1/5\nunchanged", va="center", fontsize=9.5, color=INK2)
        ax.set_xlim(-0.3, 6.3)
        ax.set_ylim(0, 0.3)
        ax.set_xticks(range(6))
        ax.set_xlabel("call duration (min)")
        ax.set_ylabel("f(x), density")
        ax.set_title(title)

    panel(fig.add_subplot(gs[0, 2]), "Three requirements for a PDF f", [
        "1. defined for every real number",
        "    (it may be 0 over large regions)",
        r"2. $f(x) \geq 0$ everywhere",
        "3. total area under f = 1",
        "",
        "Uniform call center on [0, 5]:",
        "f = 1/5, forced by 5 × 1/5 = 1.",
        "",
        "Halve the window, halve the probability:",
        "the height never changed, the width did.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xs, pdf, color=BLUE, linewidth=2.2, label="PDF f(x)")
    for lo, hi, value, color in [(1, 2, p12, ORANGE), (4, 5, p45, AQUA)]:
        mask = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[mask], pdf[mask], color=color, alpha=0.75, label=f"P({lo} ≤ X ≤ {hi}) = {value:.2f}")
    ax.plot([2, 2], [0, np.interp(2, xs, pdf)], color=INK, linewidth=2.5, label="P(X = 2): a line, area 0")
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("call duration (min)")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Equal widths, unequal areas")

    ax = fig.add_subplot(gs[1, 1])
    grid_x = np.linspace(-2, 7, 2_000)
    uniform = np.where((grid_x >= 0) & (grid_x <= 5), 0.2, 0.0)
    candidates = [("uniform on [0, 5]", uniform, BLUE, "-"),
                  ("minus 0.05", uniform - 0.05, ORANGE, "--"),
                  ("times 2", 2 * uniform, AQUA, ":")]
    for name, fx, color, style in candidates:
        ok = (fx >= 0).all() and np.isclose(trapezoid(fx, grid_x), 1, atol=0.01)
        ax.plot(grid_x, fx, color=color, linestyle=style, linewidth=2.2,
                label=f"{name}: area {trapezoid(fx, grid_x):.2f}, {'valid' if ok else 'NOT a PDF'}")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylim(-0.1, 0.62)
    ax.set_xlabel("x")
    ax.set_ylabel("candidate f(x)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Checking the requirements numerically")

    panel(fig.add_subplot(gs[1, 2]), "PMF vs. PDF", [
        "`               PMF p(x)    PDF f(x)`",
        "`P(X = x)       can be > 0  always 0`",
        "`probability    heights     areas`",
        "`total          Σ p(x) = 1  area = 1`",
        "",
        "Simulated 500,000 uniform call times:",
        f"P(2 ≤ X ≤ 3) ≈ {sim_23:.4f}",
        f"P(2 ≤ X ≤ 2.5) ≈ {sim_225:.4f},  P(X = 2) ≈ {(calls == 2.0).mean():.4f}",
        "",
        "f is the rate at which probability",
        "accumulates around each point.",
    ], spacing=0.08)
    save(fig, name="07_probability_density_function.png")


# ================================================================ notebook 08
def sheet_08_cdf():
    p = np.array([0.15, 0.35, 0.30, 0.15, 0.05])
    F = np.cumsum(p)
    assert np.isclose(F[3] - F[1], 0.45)
    xs, pdf = call_center_pdf(1_000)
    dx = xs[1] - xs[0]
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * dx)])
    F1, F2, F3 = (np.interp(t, xs, cdf) for t in (1, 2, 3))
    assert f"{F2:.3f}" == "0.502" and f"{F3 - F2:.3f}" == "0.307" and f"{F1:.3f}" == "0.114"

    fig = new_figure(r"The CDF $F_X(x) = P(X \leq x)$: probabilities become heights",
                     "Accumulate probability from the left. Discrete: a staircase that jumps by p(x). "
                     "Continuous: a smooth climb whose height is the area so far.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(5) + 0.5, p, width=0.96, color=BLUE, zorder=2)
    for i, v in enumerate(p):
        ax.text(i + 0.5, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(6))
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("call duration (min), 1-min windows")
    ax.set_ylabel("p(x)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("PMF: probability per window")

    ax = fig.add_subplot(gs[0, 1])
    ax.step(np.arange(6), np.concatenate([[0], F]), where="post", color=BLUE, linewidth=2.2)
    ax.plot(np.arange(1, 6), F, "o", color=BLUE, markersize=7)
    for i, v in enumerate(F):
        ax.text(i + 1 - 0.1, v + 0.03, f"{v:.2f}", ha="right", fontsize=10)
    ax.hlines([F[1], F[3]], 2, 4.6, color=MUTED, linestyle=":", linewidth=1.2)
    bracket(ax, 4.6, F[1], F[3], f"F(4) − F(2)\n= {F[3] - F[1]:.2f}")
    ax.set_xlim(-0.2, 6.2)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(6))
    ax.set_xlabel("call duration (min)")
    ax.set_ylabel("F(x)")
    ax.grid(True)
    ax.set_title("CDF: jumps = PMF values, flat in between")

    panel(fig.add_subplot(gs[0, 2]), "Cumulative probability", [
        r"$F_X(x) = P(X \leq x)$   for every real x",
        "",
        "`window   p      F`",
        *[f"`{i}-{i + 1}      {p[i]:.2f}   {F[i]:.2f}`" for i in range(5)],
        "",
        "Interval probabilities by subtraction:",
        f"P(2 < X ≤ 4) = {F[3]:.2f} − {F[1]:.2f} = {F[3] - F[1]:.2f}",
    ], spacing=0.077)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xs, pdf, color=BLUE, linewidth=2.2)
    mask = xs <= 2
    ax.fill_between(xs[mask], pdf[mask], color=BLUE_LIGHT, label=f"area up to 2 = {F2:.2f}")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("call duration (min)")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("PDF: the area so far ...")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xs, cdf, color=BLUE, linewidth=2.2)
    for t, Ft in [(1, F1), (2, F2), (3, F3)]:
        ax.plot([t, t], [0, Ft], color=MUTED, linestyle=":", linewidth=1.2)
        ax.plot(t, Ft, "o", color=INK, markersize=6)
        ax.text(t - 0.1, Ft + 0.03, f"F({t}) = {Ft:.3f}", ha="right", fontsize=10)
    ax.hlines([F2, F3], 2, 3.6, color=MUTED, linestyle=":", linewidth=1.2)
    bracket(ax, 3.6, F2, F3, f"P(2 ≤ X ≤ 3)\n= {F3 - F2:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("call duration (min)")
    ax.set_ylabel("F(x)")
    ax.grid(True)
    ax.set_title("... is the CDF's height")

    panel(fig.add_subplot(gs[1, 2]), "Four properties of every CDF", [
        r"1. $0 \leq F_X(x) \leq 1$:  it is a probability",
        "2. left end (or limit): 0",
        "3. right end (or limit): 1",
        "4. never decreasing",
        "",
        f"Call-center curve: F(0) = {cdf[0]:.3f}, F(5) = {cdf[-1]:.3f},",
        f"never decreasing: {bool((np.diff(cdf) >= -1e-12).all())}.",
        "",
        "Capital F for the CDF, lowercase f for",
        "the density. An area becomes a height lookup.",
    ], spacing=0.085)
    save(fig, name="08_cumulative_distribution_function.png")


# ================================================================ notebook 10
def sheet_10_normal():
    rng = np.random.default_rng(seed=42)
    mu, sigma = 2, 3
    data = rng.normal(mu, sigma, size=200_000)
    x = rng.normal(2, 2.5, size=100_000)
    z = (x - 2) / 2.5
    assert f"{x.mean():.3f}" == "2.004" and f"{x.std():.3f}" == "2.491"
    assert f"{z.mean():.3f}" == "0.002" and f"{z.std():.3f}" == "0.997"
    within = [normal_cdf(k) - normal_cdf(-k) for k in (1, 2, 3)]
    assert [f"{w:.4f}" for w in within] == ["0.6827", "0.9545", "0.9973"]

    fig = new_figure(r"The normal distribution $N(\mu, \sigma^2)$",
                     "The bell that binomials turn into. μ sets the center, σ the spread; "
                     "standardizing with z = (x − μ)/σ puts every normal on one scale.")
    gs = grid(fig, top=0.84, wspace=0.3)

    seq = gs[0, 0].subgridspec(1, 3, wspace=0.2)
    for i, n in enumerate([2, 10, 100]):
        k, pmf = binomial_pmf(n, 0.5)
        m, s = n / 2, sqrt(n) / 2
        grid_x = np.linspace(m - 4 * s, m + 4 * s, 300)
        ax = fig.add_subplot(seq[i])
        ax.bar(k, pmf, width=0.9, color=BLUE, zorder=2)
        ax.set_yticks([])
        if n == 2:
            ax.set_xticks(k)
        ax.plot(grid_x, normal_pdf(grid_x, m, s), color=ORANGE, linewidth=2, zorder=3)
        if n == 100:
            ax.set_xlim(30, 70)
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"n = {n}", fontsize=11.5)
        ax.set_xlabel("heads", fontsize=10)
        if i == 0:
            ax.set_ylabel("Binomial(n, 0.5) PMF")
    fig.text(0.05 + (0.975 - 0.05) / 6 - 0.012, 0.88, "As n grows, the binomial traces a bell",
             ha="center", fontsize=13, weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    grid_x = np.linspace(-10, 14, 600)
    ax.hist(data, bins=80, density=True, color=BLUE_LIGHT, label="data: center 2, spread 3")
    ax.plot(grid_x, np.exp(-grid_x ** 2 / 2), color=MUTED, linewidth=2, label=r"1. raw $e^{-x^2/2}$")
    ax.plot(grid_x, np.exp(-(grid_x - mu) ** 2 / 2), color=AQUA, linewidth=2, linestyle="--",
            label="2. shift x → x − 2")
    ax.plot(grid_x, normal_pdf(grid_x, mu, sigma), color=ORANGE, linewidth=2.5,
            label="3. widen by σ = 3, divide by σ√(2π)")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Building the formula: three fixes")

    panel(fig.add_subplot(gs[0, 2]), "The normal PDF", [
        r"$f(x) = \dfrac{1}{\sigma\sqrt{2\pi}}\; e^{-(x-\mu)^2 / (2\sigma^2)}$",
        "",
        "",
        "",
        "μ = center (mean), σ = spread (std).",
        "Symmetric around μ, positive on all reals.",
        "",
        r"$X \sim N(\mu, \sigma^2)$: the second parameter",
        "is the variance σ², by convention.",
        "",
        r"Area of $e^{-(x-\mu)^2/2\sigma^2}$ is $\sigma\sqrt{2\pi}$: fix 3.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    grid_x = np.linspace(-6, 8, 600)
    for (m, s), color, style in [((0, 1), BLUE, "-"), ((0, 2), BLUE_LIGHT, "-"), ((3, 1), ORANGE, "-"),
                                 ((3, 0.5), AQUA, "--")]:
        ax.plot(grid_x, normal_pdf(grid_x, m, s), color=color, linestyle=style, linewidth=2.2,
                label=f"N({m}, {s:g}²)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("μ slides the bell; σ widens or sharpens it")

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-8, 12, 100)
    ax.hist(x, bins=bins, density=True, color=BLUE_LIGHT, label="X ~ N(2, 2.5²)")
    ax.hist(z, bins=bins, density=True, histtype="step", color=BLUE, linewidth=1.8, label="Z = (X − 2)/2.5")
    ax.annotate("", (2, 0.45), (7, 0.45), arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=2))
    ax.plot([7, 2], [0.45, 0.45], "o", color=ORANGE, markersize=6)
    ax.text(4.5, 0.47, "x = 7  →  z = (7 − 2)/2.5 = 2", ha="center", fontsize=10.5, color=ORANGE,
            weight="bold")
    ax.set_ylim(0, 0.64)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("Standardizing: subtract μ, divide by σ")

    panel(fig.add_subplot(gs[1, 2]), "The CDF has no closed form", [
        "Tables then, erf in software now:",
        f"`P(Z ≤ −1) = {normal_cdf(-1):.4f}   P(Z ≤ 0) = {normal_cdf(0):.4f}`",
        f"`P(Z ≤  1) = {normal_cdf(1):.4f}   P(Z ≤ 2) = {normal_cdf(2):.4f}`",
        "",
        f"Within 1σ, 2σ, 3σ: {within[0]:.4f}, {within[1]:.4f}, {within[2]:.4f}",
        "",
        "Simulated X ~ N(2, 2.5²), 100,000 draws:",
        f"X: center {x.mean():.3f}, spread {x.std():.3f}",
        f"Z: center {z.mean():.3f}, spread {z.std():.3f}  → N(0, 1)",
    ], spacing=0.085)
    save(fig, name="10_normal_distribution.png")


# ================================================================ notebook 11
def sheet_11_chi_squared():
    def chi2_cdf_1dof(w):
        return 2 * normal_cdf(sqrt(w)) - 1

    rng = np.random.default_rng(seed=42)
    w = rng.normal(0, 1, size=1_000_000) ** 2
    moments = {}
    for k in [1, 2, 5, 30]:
        wk = (rng.normal(0, 1, size=(300_000, k)) ** 2).sum(axis=1)
        moments[k] = (wk.mean(), wk.var())
    assert f"{(w <= 0.1).mean():.3f}" == "0.248" and f"{(w <= 1).mean():.4f}" == "0.6829"
    assert f"{moments[30][0]:.3f}" == "29.984" and f"{moments[5][1]:.3f}" == "10.002"
    assert f"{chi2_cdf_1dof(1):.4f}" == "0.6827" and f"{chi2_cdf_1dof(4):.4f}" == "0.9545"

    fig = new_figure(r"The chi-squared distribution: the power of Gaussian noise",
                     "Noise Z ~ N(0, 1), power W = Z² ~ χ²₁. Over k transmissions, "
                     r"$Z_1^2 + \cdots + Z_k^2 \sim \chi^2_k$: mean k, variance 2k.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    zs = np.linspace(-4, 4, 500)
    ax.plot(zs, normal_pdf(zs), color=BLUE, linewidth=2.2)
    for w0, color in [(4, BLUE_LIGHT), (1, ORANGE)]:
        mask = np.abs(zs) <= sqrt(w0)
        ax.fill_between(zs[mask], normal_pdf(zs[mask]), color=color,
                        label=f"W ≤ {w0} ⇔ |Z| ≤ {sqrt(w0):g}: {chi2_cdf_1dof(w0):.3f}")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("z (noise)")
    ax.set_ylabel("Gaussian PDF")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(r"$F_W(w)$ = Gaussian area between $\pm\sqrt{w}$")

    ax = fig.add_subplot(gs[0, 1])
    ws = np.linspace(0.001, 8, 400)
    ax.plot(ws, [chi2_cdf_1dof(v) for v in ws], color=BLUE, linewidth=2.2)
    for w0 in [1, 4]:
        F0 = chi2_cdf_1dof(w0)
        ax.plot([w0, w0], [0, F0], color=MUTED, linestyle=":", linewidth=1.2)
        ax.plot(w0, F0, "o", color=ORANGE, markersize=8)
        ax.text(w0 + 0.2, F0 - 0.08, f"F({w0}) = {F0:.3f}", fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("w (noise power)")
    ax.set_ylabel(r"$F_W(w)$")
    ax.grid(True)
    ax.set_title("χ²₁ CDF: steep near 0, then flattens")

    panel(fig.add_subplot(gs[0, 2]), "Deriving the χ²₁ density", [
        r"$F_W(w) = \Phi(\sqrt{w}) - \Phi(-\sqrt{w}) = 2\Phi(\sqrt{w}) - 1$",
        "",
        "The PDF is the slope of the CDF:",
        r"$f_W(w) = \dfrac{\varphi(\sqrt{w})}{\sqrt{w}} = \dfrac{e^{-w/2}}{\sqrt{2\pi w}}$,   w > 0",
        "",
        "",
        "",
        r"$e^{-w/2}$: exponential tail",
        r"$1/\sqrt{w}$: infinite at 0, yet area 1",
        "",
        "F(1) is the 1σ rule, F(4) the 2σ rule.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(w[w <= 8], bins=120, density=True, color=BLUE_LIGHT, label="1,000,000 simulated Z²")
    ax.plot(ws, chi2_pdf(ws, 1), color=ORANGE, linewidth=2.2, label=r"$e^{-w/2} / \sqrt{2\pi w}$")
    ax.text(0.6, 1.2, f"{(w <= 0.1).mean():.1%} of W\nlies below 0.1", fontsize=10.5, color=INK2)
    ax.set_ylim(0, 2)
    ax.set_xlabel("w (noise power)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Density piles up near 0")

    ax = fig.add_subplot(gs[1, 1])
    gk = np.linspace(0.01, 25, 600)
    for k, color, style in [(1, ORANGE, "-"), (2, AQUA, "-"), (5, BLUE, "-"), (10, INK2, "--")]:
        ax.plot(gk, chi2_pdf(gk, k), color=color, linestyle=style, linewidth=2.2, label=f"k = {k}")
    for k in [5, 10]:
        ax.plot(k - 2, chi2_pdf(k - 2, k), "o", color=INK, markersize=5)
    ax.text(8.3, 0.105, "peaks at the mode k − 2", fontsize=10, color=INK2)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("total noise power")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("More degrees of freedom: wider, more symmetric")

    lines = [r"$f(x) = \dfrac{x^{k/2-1}\, e^{-x/2}}{2^{k/2}\,\Gamma(k/2)}$,   x > 0", "", "", "",
             "`k     mean (theory)   variance (theory)`"]
    lines += [f"`{k:>2}   {moments[k][0]:6.3f}  ({k:>2})     {moments[k][1]:7.3f}  ({2 * k:>2})`"
              for k in [1, 2, 5, 30]]
    lines += ["(300,000 simulated sums each)", "",
              "k = 2 is exponential; k ≥ 3 starts at 0.",
              "Large k: approaches N(k, 2k) by the CLT."]
    panel(fig.add_subplot(gs[1, 2]), r"$\chi^2_k = Z_1^2 + \cdots + Z_k^2$", lines, spacing=0.075)
    save(fig, name="11_chi_squared_distribution.png")


# ================================================================ notebook 12
def sheet_12_sampling():
    outcomes = ["green", "blue", "orange"]
    probs = [0.3, 0.5, 0.2]
    colors = [AQUA, BLUE, ORANGE]
    boundaries = np.cumsum(probs)
    rng = np.random.default_rng(seed=42)
    u = rng.random(100_000)
    picks = np.searchsorted(boundaries, u, side="right")
    freq = [(picks == i).mean() for i in range(3)]
    assert [f"{f:.4f}" for f in freq] == ["0.2994", "0.5015", "0.1992"]

    grid_x = np.linspace(-5, 5, 4_001)
    cdf_grid = np.array([normal_cdf(g) for g in grid_x])
    x = np.interp(rng.random(200_000), cdf_grid, grid_x)
    us = rng.random(12)
    xs_hit = np.interp(us, cdf_grid, grid_x)
    x_062 = np.interp(0.62, cdf_grid, grid_x)
    assert f"{x.mean():.4f}" == "0.0001" and f"{x.std():.4f}" == "0.9987" and f"{(x <= 1).mean():.4f}" == "0.8413"
    assert f"{x_062:.2f}" == "0.31"

    fig = new_figure("Sampling from any distribution: uniform in, inverse CDF out",
                     "Computers draw u uniform on [0, 1]. Stack the probabilities (discrete) or read "
                     "u off the CDF (any distribution): x = F⁻¹(u).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    left = 0
    for name, p, color in zip(outcomes, probs, colors):
        ax.barh(0, p, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(left + p / 2, 0, f"{name}\n{p}", ha="center", va="center", color="white", fontsize=12,
                weight="bold")
        left += p
    ax.annotate("u = 0.62 → blue", (0.62, 0.27), xytext=(0.62, 0.75), ha="center", fontsize=11,
                weight="bold", arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.0)
    ax.set_xticks([0, 0.3, 0.8, 1.0])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("uniform draw u")
    ax.set_title("The strip: each outcome owns an interval")

    grouped_bars(fig.add_subplot(gs[0, 1]), outcomes, [("target", probs, BLUE_LIGHT),
                                                       ("sampled (100,000 draws)", freq, BLUE)],
                 "The strip method hits the targets", "outcome", ylim=0.7, label_fmt=lambda v: f"{v:.4f}")

    panel(fig.add_subplot(gs[0, 2]), "Two recipes", [
        "Strip method (discrete):",
        "1. draw u uniform on [0, 1]",
        "2. find the interval it landed in",
        "3. output that interval's outcome",
        "An interval of length p is hit with prob. p.",
        "",
        "CDF method (discrete and continuous):",
        "1. draw u on the vertical axis",
        "2. move across until you hit the CDF",
        r"3. read off $x = F^{-1}(u)$",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(grid_x, cdf_grid, color=BLUE, linewidth=2.2)
    for u0, x0 in zip(us, xs_hit):
        ax.plot([-4, x0], [u0, u0], color=MUTED, linewidth=0.8)
        ax.plot([x0, x0], [0, u0], color=MUTED, linewidth=0.8, linestyle=":")
        ax.plot(x0, u0, "o", color=BLUE, markersize=5)
    ax.plot(xs_hit, np.zeros_like(xs_hit), "v", color=ORANGE, markersize=8, clip_on=False, zorder=4)
    ax.plot(np.full_like(us, -3.85), us, "<", color=AQUA, markersize=7)
    ax.plot([-4, x_062, x_062], [0.62, 0.62, 0], color=ORANGE, linewidth=2)
    ax.text(x_062 + 0.15, 0.1, f"u = 0.62 → x ≈ {x_062:.2f}", fontsize=10.5, color=ORANGE, weight="bold",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("x")
    ax.set_ylabel("Φ(x): uniform picks here")
    ax.grid(True)
    ax.set_title("12 uniform picks read off the normal CDF")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(x, bins=100, density=True, color=BLUE_LIGHT, label="200,000 inverse-CDF samples")
    ax.plot(grid_x, normal_pdf(grid_x), color=ORANGE, linewidth=2.2, label="standard normal PDF")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("Uniform in, Gaussian out")

    panel(fig.add_subplot(gs[1, 2]), "Why it works", [
        "Where the CDF is steep (high density), a",
        "wide band of u maps to a narrow band of x,",
        "so many samples land there. Flat stretches",
        "catch almost nothing.",
        "",
        "The 200,000 samples:",
        f"mean {x.mean():.4f} (target 0)",
        f"spread {x.std():.4f} (target 1)",
        f"P(x ≤ 1) = {(x <= 1).mean():.4f} (target {normal_cdf(1):.4f})",
        "",
        "Inverse transform sampling: synthetic data",
        "from any distribution you can write down.",
    ], spacing=0.075)
    save(fig, name="12_sampling_from_a_distribution.png")


if __name__ == "__main__":
    sheet_02_pmf()
    sheet_03_binomial()
    sheet_04_binomial_coefficient()
    sheet_07_pdf()
    sheet_08_cdf()
    sheet_10_normal()
    sheet_11_chi_squared()
    sheet_12_sampling()
