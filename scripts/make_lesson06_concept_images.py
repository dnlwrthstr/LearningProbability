#!/usr/bin/env python3
"""Concept sheets for lesson 06 (confidence intervals).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data, parameters and random seeds the notebook uses.
Images are written to coursera/06_confidence_intervals/img/.

Run from anywhere:  python scripts/make_lesson06_concept_images.py
"""
from functools import partial
from math import ceil, erf, gamma, pi, sqrt

import numpy as np
import sympy as sp

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, bars, grouped_bars, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "06_confidence_intervals"
save = partial(_save, folder=LESSON / "img")


# ================================================================ helpers
def z_score(p):
    """Critical value z_p: the point with probability p to its left."""
    return float(sp.sqrt(2) * sp.erfinv(2 * sp.Float(p) - 1))


def z_for_confidence(level):
    return z_score(1 - (1 - level) / 2)


def within(z):
    return erf(z / sqrt(2))


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def t_pdf(t, df):
    coeff = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    return coeff * (1 + np.asarray(t, float) ** 2 / df) ** (-(df + 1) / 2)


def t_score(p, df):
    """t critical value by bisection on the numerically integrated density."""
    integrate = getattr(np, "trapezoid", None) or np.trapz
    target, lo, hi = p - 0.5, 0.0, 200.0
    for _ in range(100):
        mid = (lo + hi) / 2
        grid_t = np.linspace(0, mid, 4001)
        if integrate(t_pdf(grid_t, df), grid_t) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def interval_ladder(ax, centers, margin, mu, title, xlabel, max_rows=None, colors=(BLUE, ORANGE)):
    """Horizontal confidence intervals, one per row; misses in orange."""
    hits = np.abs(centers - mu) <= margin
    for i, (c, hit) in enumerate(zip(centers, hits)):
        color = colors[0] if hit else colors[1]
        ax.plot([c - margin[i] if np.ndim(margin) else c - margin, c + (margin[i] if np.ndim(margin) else margin)],
                [i, i], color=color, linewidth=1.6 if hit else 2.4)
        ax.plot(c, i, "o", color=color, markersize=2.5)
    ax.axvline(mu, color=INK, linewidth=1.8)
    ax.set_yticks([])
    ax.set_ylim(-1, len(centers) * 1.07 + 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return hits


def shaded_normal(ax, mu, sigma, z, color=BLUE, tail_color=ORANGE, label_center=None):
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
    y = normal_pdf(x, mu, sigma)
    inside = np.abs(x - mu) <= z * sigma
    ax.fill_between(x[inside], y[inside], color=BLUE_LIGHT, zorder=1)
    ax.fill_between(x[x < mu - z * sigma], y[x < mu - z * sigma], color=tail_color, alpha=0.7, zorder=1)
    ax.fill_between(x[x > mu + z * sigma], y[x > mu + z * sigma], color=tail_color, alpha=0.7, zorder=1)
    ax.plot(x, y, color=color, linewidth=2, zorder=2)
    return x, y


# ================================================================ notebook 01
def sheet_01_intro():
    mu, sigma = 170.0, 10.0
    margin = z_for_confidence(0.95) * sigma
    rng = np.random.default_rng(5)
    rng.normal(mu, sigma)
    sample_means = rng.normal(mu, sigma, size=10_000)
    coverage = (np.abs(sample_means - mu) <= margin).mean()

    fig = new_figure("Confidence intervals: a recipe that captures the unknown mean 95% of the time",
                     "Statistopia heights: μ = 170 cm (\"the key\", fixed but unknown), σ = 10. "
                     "Interval = sample mean ± margin of error.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    shaded_normal(ax, mu, sigma, 1.96)
    ax.annotate("", xy=(mu - margin, 0.043), xytext=(mu + margin, 0.043),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
    ax.text(mu, 0.045, f"95%: μ ± {margin:.1f}", ha="center", fontsize=11, weight="bold")
    ax.text(mu - 3.2 * sigma, 0.006, "2.5%", ha="center", color=ORANGE, fontsize=10.5, weight="bold")
    ax.text(mu + 3.2 * sigma, 0.006, "2.5%", ha="center", color=ORANGE, fontsize=10.5, weight="bold")
    ax.set_ylim(0, 0.05)
    ax.set_xlabel("sample mean x̄ (here: one person's height, n = 1)")
    ax.set_ylabel("density")
    ax.grid(axis="y")
    ax.set_title("Where the sample mean lands")

    rng100 = np.random.default_rng(1)
    centers = rng100.normal(mu, sigma, size=100)
    ax = fig.add_subplot(gs[0, 1:])
    hits = interval_ladder(ax, centers, margin, mu, "", "height (cm)")
    ax.set_title(f"100 samples, 100 intervals: {hits.sum()} contain μ (blue), {100 - hits.sum()} miss (orange)")
    ax.text(mu + 0.6, 104, "μ = 170", fontsize=10.5, weight="bold")

    panel(fig.add_subplot(gs[1, 0]), "The dictionary", [
        "the key (fixed)      →  population mean μ",
        "where you park       →  sample mean x̄",
        "search distance       →  margin of error",
        "searched stretch      →  confidence interval",
        "how sure you need be →  confidence 1 − α",
        "chance of missing     →  significance α",
    ], spacing=0.1)

    ax = fig.add_subplot(gs[1, 1])
    levels = [0.80, 0.95, 0.99]
    margins = [z_for_confidence(l) * sigma for l in levels]
    bars(ax, [f"{l:.0%}" for l in levels], margins, "Search distance by confidence level", "confidence level",
         ylabel="margin of error (cm)", ylim=30, label_fmt=lambda v: f"{v:.1f} cm", highlight=1)

    panel(fig.add_subplot(gs[1, 2]), "What 95% means", [
        "The margin is built so that x̄ lands within",
        "1.96σ of μ with probability 0.95.",
        "\"x̄ within m of μ\" = \"μ within m of x̄\".",
        "",
        f"10,000 repeats: {coverage:.1%} of intervals contain μ.",
        "",
        "After sampling, one interval either",
        "contains μ or not. The 95% belongs to",
        "the recipe, not to one interval.",
    ], spacing=0.09)
    save(fig, name="01_intro_confidence_intervals.png")


# ================================================================ notebook 02
def sheet_02_two_dials():
    mu, sigma = 170.0, 10.0
    levels = [0.70, 0.90, 0.95, 0.99]
    ns = [1, 2, 10, 100]

    def moe(level, n):
        return z_for_confidence(level) * sigma / sqrt(n)

    fig = new_figure("Two dials control the interval width: sample size and confidence level",
                     "Margin of error = z · σ / √n.  More data narrows it at no cost in confidence; "
                     "lower confidence narrows it by accepting more misses.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(130, 210, 600)
    for n, color in [(1, AQUA), (2, BLUE), (10, ORANGE)]:
        ax.plot(x, normal_pdf(x, mu, sigma / sqrt(n)), color=color, linewidth=2.2,
                label=f"n = {n}: margin {moe(0.95, n):.1f} cm")
    ax.set_xlabel("sample mean x̄ (cm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Dial 1: bigger n, taller and skinnier")

    ax = fig.add_subplot(gs[0, 1])
    z = np.linspace(-4, 4, 400)
    ax.plot(z, normal_pdf(z), color=INK2, linewidth=2)
    for level, color, fill in [(0.95, BLUE, BLUE_LIGHT), (0.70, ORANGE, "#f8c9b3")]:
        zc = z_for_confidence(level)
        ax.fill_between(z[np.abs(z) <= zc], normal_pdf(z[np.abs(z) <= zc]), color=fill)
        ax.axvline(zc, color=color, linestyle="--", linewidth=1.6)
        ax.axvline(-zc, color=color, linestyle="--", linewidth=1.6, label=f"{level:.0%}: z = {zc:.2f}")
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Dial 2: lower confidence, smaller z")

    lines = ["`        " + "".join(f"{f'{l:.0%}':>7s}" for l in levels) + "`"]
    for n in ns:
        lines.append(f"`n={n:<5d}" + "".join(f"{moe(l, n):>7.1f}" for l in levels) + "`")
    lines += ["", "Margins in cm. Down a column: data helps.", "Along a row to the left: risk helps."]
    panel(fig.add_subplot(gs[0, 2]), "Both dials at once", lines, spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    n_grid = np.linspace(1, 100, 300)
    for level, color in [(0.99, ORANGE), (0.95, BLUE), (0.70, AQUA)]:
        ax.plot(n_grid, z_for_confidence(level) * sigma / np.sqrt(n_grid), color=color, linewidth=2.2,
                label=f"{level:.0%}")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n (log scale)")
    ax.set_ylabel("margin of error (cm)")
    ax.legend(loc="upper right", fontsize=10, title="confidence", title_fontsize=10)
    ax.grid(True)
    ax.set_title("Margin vs n: a 1/√n curve for each level")

    rng = np.random.default_rng(12)
    cases = [(1, 0.95), (2, 0.95), (10, 0.95), (1, 0.70)]
    coverage = []
    for n, level in cases:
        m = moe(level, n)
        xbars = rng.normal(mu, sigma / sqrt(n), size=10_000)
        coverage.append((np.abs(xbars - mu) <= m).mean())
    bars(fig.add_subplot(gs[1, 1]), [f"n = {n}\n{l:.0%}" for n, l in cases], coverage,
         "Coverage tracks the level, not n", "", ylabel="share of intervals containing μ", ylim=1.15,
         label_fmt=lambda v: f"{v:.1%}", highlight=3)

    panel(fig.add_subplot(gs[1, 2]), "Two ways to halve the interval", [
        "Start: n = 1, 95%, margin 19.6 cm.",
        "",
        "With data: margin ∝ 1/√n, so n = 4.",
        "Still 95% confident.",
        "",
        "With risk: z from 1.96 to 0.98, which is",
        f"only {within(0.98):.0%} confidence.",
        "",
        "n buys narrowness; it never changes",
        "the success rate.",
    ], spacing=0.085)
    save(fig, name="02_two_dials.png")


# ================================================================ notebook 03
def sheet_03_margin_of_error():
    sigma, n, x_bar = 10.0, 49, 172.0
    se = sigma / sqrt(n)
    m = z_score(0.975) * se

    fig = new_figure("The margin of error: critical value × standard error",
                     "Standardize with z = (x̄ − μ) / (σ/√n), find the critical value, then flip the "
                     "inequality from bounding x̄ to bounding μ.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(-4, 4, 600)
    shaded_normal(ax, 0, 1, 1.96)
    for zc, label in [(-1.96, "z₀.₀₂₅ = −1.96"), (1.96, "z₀.₉₇₅ = +1.96")]:
        ax.axvline(zc, color=INK2, linestyle="--", linewidth=1.3)
        ax.text(zc + (-0.1 if zc < 0 else 0.1), 0.43, label, ha="right" if zc < 0 else "left", fontsize=10,
                weight="bold")
    ax.text(0, 0.15, "95%", ha="center", fontsize=14, weight="bold", color=BLUE)
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.grid(axis="y")
    ax.set_title("The z-distribution and its critical values")

    lines = ["`confidence   α      z(1 − α/2)`"]
    for alpha in (0.10, 0.05, 0.01):
        lines.append(f"`{1 - alpha:>9.0%}  {alpha:>5.2f}   {z_score(1 - alpha / 2):>10.3f}`")
    lines += ["", "z_p has probability p to its left.", "By symmetry z_(α/2) = −z_(1 − α/2),",
              "so one number suffices."]
    panel(fig.add_subplot(gs[0, 1]), "The lookup table", lines, spacing=0.09)

    panel(fig.add_subplot(gs[0, 2]), "From critical value to margin", [
        r"standard error $= \sigma / \sqrt{n}$",
        r"margin $= z_{1-\alpha/2} \cdot \sigma / \sqrt{n}$",
        "",
        f"n = 49, σ = 10:  SE = 10/7 = {se:.2f} cm",
        f"margin = 1.96 × {se:.2f} = {m:.1f} cm",
        f"interval = 172.0 ± {m:.1f} = [{x_bar - m:.1f}, {x_bar + m:.1f}]",
    ], spacing=0.1)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(130, 210, 600)
    for (mu_c, s_c), color in [((170, 10), BLUE), ((160, 5), ORANGE), ((185, 15), AQUA)]:
        ax.plot(x, normal_pdf(x, mu_c, s_c), color=color, linewidth=2, label=f"N({mu_c}, {s_c}²)")
        lo, hi = mu_c - 1.96 * s_c, mu_c + 1.96 * s_c
        ax.plot([lo, hi], [normal_pdf(lo, mu_c, s_c)] * 2, color=color, linewidth=1.2, linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Every normal: 95% within ±1.96σ of its mean")

    rng = np.random.default_rng(9)
    rng.normal(170, sigma, size=n)
    xbars = rng.normal(170, se, size=100_000)
    a = np.abs(xbars - 170) <= m
    b = (xbars - m <= 170) & (170 <= xbars + m)
    bars(fig.add_subplot(gs[1, 1]), ["P(x̄ within m of μ)", "P(interval contains μ)"], [a.mean(), b.mean()],
         "The flip: the same event, two phrasings", "", ylim=1.15, label_fmt=lambda v: f"{v:.3f}")

    panel(fig.add_subplot(gs[1, 2]), "The flip, and non-normal data", [
        "μ − m ≤ x̄ ≤ μ + m",
        "⇔  x̄ − m ≤ μ ≤ x̄ + m",
        f"Identical in all 100,000 samples: {bool((a == b).all())}.",
        "",
        "Population not normal? For n ≥ 30 the",
        "central limit theorem makes x̄ roughly",
        "normal anyway, so the recipe still works.",
    ], spacing=0.09)
    save(fig, name="03_margin_of_error.png")


# ================================================================ notebook 04
def sheet_04_recipe():
    x_bar, sigma, n = 170.0, 25.0, 49
    se = sigma / sqrt(n)
    m = z_score(0.975) * se
    assert abs(m - 7) < 0.01

    fig = new_figure("Calculating a confidence interval in six steps",
                     "Statistopia: 6,000 adults, a random sample of n = 49, x̄ = 170 cm, σ = 25 cm known. Target: 95%.")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "The six steps", [
        "1. sample mean:      x̄ = 170",
        "2. confidence level:  1 − α = 0.95",
        "3. critical value:    z₀.₉₇₅ = 1.96",
        f"4. standard error:   25/√49 = 25/7 ≈ {se:.2f}",
        "5. margin of error:   1.96 × 25/7 = 7 cm",
        "6. interval:              170 ± 7 = [163, 177]",
        "",
        "1.96 × 25 = 49 exactly, so the",
        "margin comes out to a round 7 cm.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(x_bar - 4 * se, x_bar + 4 * se, 500)
    ax.plot(x, normal_pdf(x, x_bar, se), color=BLUE, linewidth=2.2)
    inside = np.abs(x - x_bar) <= m
    ax.fill_between(x[inside], normal_pdf(x[inside], x_bar, se), color=BLUE_LIGHT)
    ax.annotate("", xy=(x_bar - m, 0.125), xytext=(x_bar + m, 0.125),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=2))
    ax.text(x_bar, 0.13, "[163, 177]", ha="center", fontsize=12, weight="bold", color=ORANGE)
    ax.axvline(x_bar, color=INK2, linewidth=1.2)
    ax.set_ylim(0, 0.145)
    ax.set_xlabel("height (cm)")
    ax.set_ylabel("density of x̄")
    ax.grid(True)
    ax.set_title("The worked example: 170 ± 7")

    ax = fig.add_subplot(gs[0, 2])
    for i, (level, color) in enumerate([(0.90, AQUA), (0.95, BLUE), (0.99, ORANGE)]):
        mm = z_for_confidence(level) * se
        ax.plot([x_bar - mm, x_bar + mm], [i, i], color=color, linewidth=6, solid_capstyle="butt")
        ax.text(x_bar + mm + 0.4, i, f"{level:.0%}: [{x_bar - mm:.1f}, {x_bar + mm:.1f}]", va="center",
                fontsize=10.5)
    ax.axvline(x_bar, color=INK2, linewidth=1)
    ax.set_xlim(158, 192)
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks([])
    ax.set_xlabel("height (cm)")
    ax.grid(axis="x")
    ax.set_title("One sample, three promises")

    rng = np.random.default_rng(23)
    population = rng.normal(170, 25, size=6000)
    true_mu = population.mean()
    ax = fig.add_subplot(gs[1, :2])
    centers = np.array([rng.choice(population, size=49, replace=False).mean() for _ in range(60)])
    hits = interval_ladder(ax, centers, m, true_mu, "", "height (cm)")
    ax.set_title(f"60 surveys of 49 islanders from a simulated 6,000: {hits.sum()} intervals contain the true mean")
    ax.text(true_mu + 0.3, 62.5, f"true mean {true_mu:.1f}", fontsize=10, weight="bold")

    panel(fig.add_subplot(gs[1, 2]), "The fine print", [
        "1. The sample is random.",
        "2. The population is normal, or n ≥ 30",
        "    so the CLT makes x̄ roughly normal.",
        "",
        "Both hold here: 49 picked at random,",
        "and 49 > 30.",
        "",
        "A survey on a basketball forum with",
        "1,000 answers fails assumption 1, and",
        "no sample size repairs that.",
    ], spacing=0.085)
    save(fig, name="04_six_step_recipe.png")


# ================================================================ notebook 05
def sheet_05_sample_size():
    sigma = 25.0

    def required(moe, level=0.95):
        return ceil((z_for_confidence(level) * sigma / moe) ** 2)

    def moe_at(n, level=0.95):
        return z_for_confidence(level) * sigma / sqrt(n)

    assert required(3) == 267 and required(3, 0.99) == 461

    fig = new_figure("Choosing the sample size: the margin formula, backwards",
                     "Want a margin of error of at most MOE?  n ≥ (z · σ / MOE)², rounded up.  Statistopia: σ = 25 cm.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    n_grid = np.linspace(20, 600, 400)
    ax.plot(n_grid, z_for_confidence(0.95) * sigma / np.sqrt(n_grid), color=BLUE, linewidth=2.2)
    ax.axhline(3, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.axvline(267, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.plot(267, moe_at(267), "o", color=ORANGE, markersize=9)
    ax.annotate("target 3 cm → n = 267", (267, 3), textcoords="offset points", xytext=(12, 14), fontsize=11,
                weight="bold", color=ORANGE)
    ax.set_xlabel("sample size n")
    ax.set_ylabel("margin of error at 95% (cm)")
    ax.grid(True)
    ax.set_title("Read the margin curve backwards")

    panel(fig.add_subplot(gs[0, 1]), "The calculation", [
        r"$\mathrm{MOE} = z \cdot \sigma / \sqrt{n} \;\Rightarrow\; n \geq \left(z\,\sigma / \mathrm{MOE}\right)^2$",
        "",
        "",
        f"(1.96 × 25 / 3)² = {(z_for_confidence(0.95) * 25 / 3) ** 2:.2f}  →  n = 267",
        "",
        f"n = 267: margin {moe_at(267):.3f} cm  ✓",
        f"n = 266: margin {moe_at(266):.3f} cm  ✗",
        "",
        "Always round up: rounding to the nearest",
        "integer can break the promise.",
    ], spacing=0.085)

    lines = ["`target MOE    95%     99%`"]
    for moe in (7, 3, 1):
        lines.append(f"`{moe:>6d} cm  {required(moe):>6d}  {required(moe, 0.99):>6d}`")
    lines += ["", "Cut the margin 7× (7 → 1 cm): the sample", "grows 7² = 49× (49 → 2,401).",
              "99% instead of 95% at 3 cm: 461 vs 267."]
    panel(fig.add_subplot(gs[0, 2]), "The price list", lines, spacing=0.09)

    ax = fig.add_subplot(gs[1, :2])
    moes = np.linspace(0.8, 8, 300)
    for level, color in [(0.90, AQUA), (0.95, BLUE), (0.99, ORANGE)]:
        ax.plot(moes, (z_for_confidence(level) * sigma / moes) ** 2, color=color, linewidth=2.2, label=f"{level:.0%}")
    for moe in (7, 3, 1):
        ax.plot(moe, required(moe), "o", color=BLUE, markersize=8)
        ax.annotate(f"{required(moe):,}", (moe, required(moe)), textcoords="offset points", xytext=(8, 6), fontsize=10.5)
    ax.set_yscale("log")
    ax.set_xlabel("desired margin of error (cm)")
    ax.set_ylabel("required n (log scale)")
    ax.legend(loc="upper right", fontsize=10, title="confidence", title_fontsize=10)
    ax.grid(True, which="major")
    ax.set_title("The inverse-square price of precision")

    rng = np.random.default_rng(31)
    m = moe_at(267)
    x_bars = rng.normal(170, sigma / sqrt(267), size=10_000)
    coverage = (np.abs(x_bars - 170) <= m).mean()
    panel(fig.add_subplot(gs[1, 2]), "Does 267 keep both promises?", [
        "Simulate 10,000 surveys of 267 people:",
        "",
        f"margin = {m:.3f} cm  (target ≤ 3)",
        f"coverage = {coverage:.1%}  (target 95%)",
        "",
        "Both precision and confidence enter",
        "squared: halving the margin or moving",
        "from 95% to 99% gets expensive fast.",
    ], spacing=0.09)
    save(fig, name="05_choosing_sample_size.png")


# ================================================================ notebook 06
def sheet_06_confidence_vs_probability():
    mu, sigma, n = 170.0, 25.0, 49
    margin = z_for_confidence(0.95) * sigma / sqrt(n)
    rng = np.random.default_rng(13)
    x_bar = rng.normal(mu, sigma / sqrt(n))
    trials = 2000
    x_bars = rng.normal(mu, sigma / sqrt(n), size=trials)
    contained = (np.abs(x_bars - mu) <= margin).astype(int)
    running = np.cumsum(contained) / np.arange(1, trials + 1)

    fig = new_figure("Confidence is not probability: μ is fixed, the intervals are random",
                     "\"95% of intervals built this way contain μ\" is right.  "
                     "\"There is a 95% chance μ is in [a, b]\" is wrong: μ is either in or out.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot([x_bar - margin, x_bar + margin], [0, 0], color=BLUE, linewidth=7, solid_capstyle="butt")
    ax.axvline(mu, color=INK, linewidth=1.8)
    ax.text(mu + 0.4, 0.55, "μ = 170 (fixed)", fontsize=11, weight="bold")
    ax.text(171, -0.35, f"your interval [{x_bar - margin:.1f}, {x_bar + margin:.1f}]", ha="left",
            fontsize=11, color=BLUE, weight="bold")
    ax.text(171, -0.65, "P(μ inside) = 1. Ask again: still 1.", ha="left", fontsize=10.5, color=INK2)
    ax.set_xlim(150, 190)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("height (cm)")
    ax.set_title("After sampling: one interval, nothing random")

    ax = fig.add_subplot(gs[0, 1])
    centers = x_bars[:40]
    hits = interval_ladder(ax, centers, margin, mu, "", "height (cm)")
    ax.set_title(f"Repeating the sampling: {hits.sum()} of 40 intervals contain μ")

    panel(fig.add_subplot(gs[0, 2]), "A language audit", [
        "✓  95% of intervals from this recipe",
        "     contain μ.",
        "✓  Before sampling, my interval will",
        "     capture μ with probability 0.95.",
        "✓  I'm 95% confident μ is in [a, b]",
        "     (confident = recipe success rate).",
        "✗  There's a 95% chance μ is in [a, b].",
        "✗  μ falls in my interval 95% of the time.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(np.arange(1, trials + 1), running, color=BLUE, linewidth=1.6)
    ax.axhline(0.95, color=ORANGE, linestyle="--", linewidth=1.6)
    ax.text(trials, 0.955, "confidence level 0.95", ha="right", fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel("number of repeated sampling experiments")
    ax.set_ylabel("share of intervals containing μ")
    ax.grid(True)
    ax.set_title(f"Each outcome is 0 or 1; their long-run rate is the 95% ({running[-1]:.1%} after {trials:,})")

    panel(fig.add_subplot(gs[1, 2]), "Why it matters", [
        "Random before sampling: x̄, so the",
        "interval. Never random: μ.",
        "",
        "If \"95% chance\" were true, you could bet",
        "19:1 on any single interval. An opponent",
        "who knows μ would accept only the bets",
        "on the 5% that miss, and win every one.",
        "",
        "Only the long-run rate protects you.",
    ], spacing=0.085)
    save(fig, name="06_confidence_vs_probability.png")


# ================================================================ notebook 07
def sheet_07_t_distribution():
    x_bar, s, n = 170.0, 25.0, 10
    se = s / sqrt(n)
    t9 = t_score(0.975, 9)
    z = z_score(0.975)
    assert abs(t9 - 2.262) < 0.001

    fig = new_figure("Unknown σ: use the t-distribution with n − 1 degrees of freedom",
                     "Replacing σ with the sample's s adds uncertainty. The t-distribution has fatter tails, "
                     "so its critical value is larger than 1.96.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    tt = np.linspace(-5, 5, 600)
    for df, color in [(1, ORANGE), (4, AQUA), (29, BLUE)]:
        ax.plot(tt, t_pdf(tt, df), color=color, linewidth=2, label=f"t, df = {df}")
    ax.plot(tt, normal_pdf(tt), color=INK, linewidth=2, linestyle="--", label="normal")
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("t closes in on the normal as df grows")

    dfs = [1, 4, 9, 29, 99]
    crit = [t_score(0.975, df) for df in dfs]
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(len(dfs)), crit, width=0.6, color=BLUE, zorder=2)
    ax.axhline(z, color=ORANGE, linestyle="--", linewidth=1.6, label="normal: z = 1.960")
    ax.legend(loc="upper right", fontsize=10)
    for i, v in enumerate(crit):
        ax.text(i, v * 1.08, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(1, 25)
    ax.set_xticks(range(len(dfs)), [f"df = {d}" for d in dfs])
    ax.set_ylabel("t₀.₉₇₅ (log scale)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("The critical value falls toward 1.96")

    panel(fig.add_subplot(gs[0, 2]), "Which score, which scale?", [
        "σ known:           z, with σ",
        "σ unknown:        t (df = n − 1), with s",
        "",
        r"t-score:  $t = \dfrac{\bar{x} - \mu}{s / \sqrt{n}}$",
        "",
        "",
        "n = 200: t₀.₉₇₅ = 1.972 ≈ 1.96, so the",
        "difference only matters for small n.",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    for i, (name, c, color) in enumerate([("wrong: z with s", z, ORANGE), ("right: t with s", t9, BLUE)]):
        m = c * se
        ax.plot([x_bar - m, x_bar + m], [i, i], color=color, linewidth=7, solid_capstyle="butt")
        ax.text(x_bar, i + 0.3, f"{name}: 170 ± {m:.1f} = [{x_bar - m:.1f}, {x_bar + m:.1f}]", ha="center",
                fontsize=10.5, color=color, weight="bold")
    ax.axvline(x_bar, color=INK2, linewidth=1)
    ax.set_xlim(145, 195)
    ax.set_ylim(-0.6, 1.8)
    ax.set_yticks([])
    ax.set_xlabel("height (cm)")
    ax.set_title(f"n = 10, s = 25: the honest interval is {t9 / z - 1:.0%} wider")

    rng = np.random.default_rng(17)
    rng.normal(170, 25, size=(1_000_000, n))
    surveys = rng.normal(170, 25, size=(10_000, n))
    xb = surveys.mean(axis=1)
    se_vals = surveys.std(axis=1, ddof=1) / np.sqrt(n)
    cov_z = (np.abs(xb - 170) <= z * se_vals).mean()
    cov_t = (np.abs(xb - 170) <= t9 * se_vals).mean()
    bars(fig.add_subplot(gs[1, 1]), ["z = 1.960\n(wrong)", "t = 2.262\n(right)"], [cov_z, cov_t],
         "Coverage with s, n = 10 (10,000 surveys)", "", ylabel="share containing μ", ylim=1.1,
         label_fmt=lambda v: f"{v:.1%}", highlight=0)

    panel(fig.add_subplot(gs[1, 2]), "Why the tails are fat", [
        "s varies from sample to sample. When it",
        "comes out small, the t-score gets large:",
        "extra spread that the normal ignores.",
        "",
        f"The z shortcut covers only {cov_z:.1%}, below",
        "the promised 95%: it looks more precise",
        "but quietly breaks the promise.",
        "",
        "Fewer data points = fatter tails = wider.",
    ], spacing=0.085)
    save(fig, name="07_t_distribution.png")


# ================================================================ notebook 08
def sheet_08_proportions():
    x, n = 24, 30
    p_hat = x / n
    se = sqrt(p_hat * (1 - p_hat) / n)
    m = z_score(0.975) * se
    assert abs(p_hat - m - 0.657) < 0.001

    fig = new_figure("Confidence intervals for proportions: p̂ ± z · √(p̂(1 − p̂)/n)",
                     "Same recipe as for a mean: a proportion is the mean of 0/1 answers. "
                     "Example: 24 of 30 people say yes, p̂ = 0.8.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot([p_hat - m, p_hat + m], [0, 0], color=BLUE, linewidth=8, solid_capstyle="butt")
    ax.plot(p_hat, 0, "o", color=ORANGE, markersize=11, zorder=3)
    ax.text(p_hat, 0.35, f"p̂ = 0.8,  95% CI [{p_hat - m:.3f}, {p_hat + m:.3f}]", ha="center", fontsize=11.5,
            weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("proportion (the only meaningful axis is 0 to 1)")
    ax.grid(axis="x")
    ax.set_title("The worked example on the 0–1 axis")

    panel(fig.add_subplot(gs[0, 1]), "Digit by digit", [
        "p̂(1 − p̂) = 0.8 × 0.2 = 0.16",
        "÷ n:            0.16 / 30 = 0.00533",
        f"√:                SE = {se:.4f}",
        f"× 1.96:        margin = {m:.3f}",
        f"interval:      [{p_hat - m:.3f}, {p_hat + m:.3f}] ≈ [0.66, 0.94]",
        "",
        "Same shape as for means: the",
        "standard error σ/√n becomes",
        "√(p(1 − p)/n), the Bernoulli std / √n.",
    ], spacing=0.09)

    rng = np.random.default_rng(41)          # the notebook's seed and draw order
    answers = rng.random((100_000, n)) < 0.8
    p_hats = answers.mean(axis=1)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(p_hats, bins=np.arange(0.5, 1.02, 1 / n) - 1 / (2 * n), density=True, color=BLUE_LIGHT,
            edgecolor=SURFACE, label="simulated p̂ (n = 30)")
    xs = np.linspace(0.5, 1, 300)
    sd = sqrt(0.8 * 0.2 / n)
    ax.plot(xs, normal_pdf(xs, 0.8, sd), color=ORANGE, linewidth=2.2, label="CLT: N(p, p(1 − p)/n)")
    ax.set_xlabel("p̂")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title(f"p̂ is a mean: std {p_hats.std():.4f} vs theory {sd:.4f}")

    ax = fig.add_subplot(gs[1, 0])
    ps = np.linspace(0, 1, 300)
    ax.plot(ps, np.sqrt(ps * (1 - ps) / n), color=BLUE, linewidth=2.2)
    for p, label in [(0.5, "0.091"), (0.8, "0.073"), (0.95, "0.040")]:
        ax.plot(p, sqrt(p * (1 - p) / n), "o", color=ORANGE, markersize=8)
        ax.annotate(f"p̂ = {p}: {label}", (p, sqrt(p * (1 - p) / n)), textcoords="offset points",
                    xytext=(-10, 10), ha="right" if p > 0.6 else "center", fontsize=10)
    ax.set_ylim(0, 0.105)
    ax.set_xlabel("p̂")
    ax.set_ylabel("standard error (n = 30)")
    ax.grid(True)
    ax.set_title("The noisiest question is the 50/50 one")

    coverage = []
    ns = [30, 100, 1000]
    for n_test in ns:
        counts = (rng.random((50_000, n_test)) < 0.8).sum(axis=1)
        ph = counts / n_test
        mm = z_score(0.975) * np.sqrt(ph * (1 - ph) / n_test)
        coverage.append((np.abs(ph - 0.8) <= mm).mean())
    bars(fig.add_subplot(gs[1, 1]), [f"n = {k}" for k in ns], coverage,
         "Does it deliver 95%? Almost (p = 0.8)", "", ylabel="coverage", ylim=1.1,
         label_fmt=lambda v: f"{v:.1%}")

    panel(fig.add_subplot(gs[1, 2]), "Planning a poll", [
        "Before polling p is unknown, so use the",
        "worst case p = 0.5:",
        r"$n \geq z^2 \cdot 0.25 / \mathrm{MOE}^2$",
        "",
        f"±5 points at 95%: n ≥ {1.96 ** 2 * 0.25 / 0.05 ** 2:.1f} → {ceil(z_score(0.975) ** 2 * 0.25 / 0.05 ** 2)}",
        f"±3 points at 95%: n ≥ {ceil(z_score(0.975) ** 2 * 0.25 / 0.03 ** 2)}",
        "",
        "That is the classic \"poll of about 400,",
        "margin ±5%\".",
    ], spacing=0.09)
    save(fig, name="08_proportions.png")


if __name__ == "__main__":
    sheet_01_intro()
    sheet_02_two_dials()
    sheet_03_margin_of_error()
    sheet_04_recipe()
    sheet_05_sample_size()
    sheet_06_confidence_vs_probability()
    sheet_07_t_distribution()
    sheet_08_proportions()
