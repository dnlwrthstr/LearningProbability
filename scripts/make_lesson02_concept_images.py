#!/usr/bin/env python3
"""Concept sheets for lesson 02 (describing distributions).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data and parameters the notebook uses.
Images are written to coursera/02_discribing_distribution/img/.

Run from anywhere:  python scripts/make_lesson02_concept_images.py
"""
from functools import partial
from itertools import permutations
from math import comb, erf, sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, frac_label, bars, grouped_bars, heat, panel, trapezoid)
from viz_style import save as _save

LESSON = REPO / "coursera" / "02_discribing_distribution"
save = partial(_save, folder=LESSON / "img")

DIE = np.arange(1, 7)
DIE_PMF = np.full(6, 1 / 6)


# ================================================================ helpers
def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def normal_within(k):
    """P(|Z| <= k) for a standard normal."""
    return erf(k / sqrt(2))


def moments(values, pmf):
    values, pmf = np.asarray(values, float), np.asarray(pmf, float)
    mu = np.sum(pmf * values)
    var = np.sum(pmf * (values - mu) ** 2)
    sigma = np.sqrt(var)
    skew = np.sum(pmf * ((values - mu) / sigma) ** 3)
    kurt = np.sum(pmf * ((values - mu) / sigma) ** 4)
    return mu, var, skew, kurt


def stems(ax, values, pmf, color=BLUE, width=0.6, labels=None, label_fmt=frac_label):
    ax.bar(values, pmf, width=width, color=color, zorder=2)
    if labels is not False:
        top = ax.get_ylim()[1]
        for v, p in zip(values, pmf):
            ax.text(v, p + top * 0.02, label_fmt(p), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.grid(axis="y", zorder=0)


def fulcrum(ax, x, label=None, color=ORANGE):
    """A balance-point triangle under the x-axis at x."""
    ax.plot(x, 0, marker="^", markersize=17, color=color, clip_on=False, zorder=5)
    if label:
        ax.annotate(label, (x, 0), textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=11, weight="bold", color=color, zorder=6,
                    bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.25"))


def vline(ax, x, label, color, style="-"):
    ax.axvline(x, color=color, linestyle=style, linewidth=2, label=label, zorder=3)


# ================================================================ notebook 01
def sheet_01_mean():
    ages, counts = np.array([0, 1, 2, 3]), np.array([3, 2, 4, 1])
    p_ages = counts / counts.sum()
    mean_ages = np.sum(ages * p_ages)
    heads = np.arange(4)
    p_heads = np.array([comb(3, k) for k in heads]) / 8
    mean_heads = np.sum(heads * p_heads)
    assert np.isclose(mean_ages, 1.3) and np.isclose(mean_heads, 1.5)

    fig = new_figure("The mean is the balance point of a distribution",
                     "E(X) = Σ x · p(x): a weighted average, with each value weighted by its probability.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(0, 5)
    stems(ax, ages, counts, label_fmt=lambda v: f"{v:.0f} kid" + ("s" if v != 1 else ""))
    fulcrum(ax, mean_ages, f"balances at {mean_ages:.1f}")
    ax.set_xticks(ages)
    ax.set_xlabel("age")
    ax.set_ylabel("kids")
    ax.set_title("The kids' ages: 3, 2, 4, 1 kids")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, 0.5)
    stems(ax, heads, p_heads)
    fulcrum(ax, mean_heads, f"balances at {mean_heads:.1f}")
    ax.set_xticks(heads)
    ax.set_xlabel("heads in 3 fair coins")
    ax.set_ylabel("probability")
    ax.set_title("Three coins, one dollar per head")

    panel(fig.add_subplot(gs[0, 2]), "A weighted average", [
        "Kids' ages as probabilities:",
        r"$E(X) = 0.3 \cdot 0 + 0.2 \cdot 1 + 0.4 \cdot 2 + 0.1 \cdot 3 = 1.3$",
        "",
        "Three coins:",
        r"$E(X) = \frac{1}{8} \cdot 0 + \frac{3}{8} \cdot 1 + \frac{3}{8} \cdot 2 + \frac{1}{8} \cdot 3 = 1.5$",
        "",
        "Fair price of a game = its expected payoff.",
        r"Heads pays 10 dollars: $E = 0.5 \cdot 10 + 0.5 \cdot 0 = 5$,",
        "so paying 4 is a good deal and 6 loses money.",
    ], spacing=0.09)

    rng = np.random.default_rng(42)
    plays = rng.choice([0, 10], size=2000)
    running = np.cumsum(plays) / np.arange(1, len(plays) + 1)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.arange(1, len(plays) + 1), running, color=BLUE, linewidth=1.8)
    ax.axhline(5, color=ORANGE, linestyle="--", linewidth=1.8)
    ax.text(2000, 5.25, "E(X) = 5", ha="right", color=ORANGE, fontsize=11, weight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, 10)
    ax.set_xlabel("number of plays (log scale)")
    ax.set_ylabel("average payoff (dollars)")
    ax.grid(True)
    ax.set_title("Coin game: the running average settles at 5")

    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between([0, 60], 0, 1 / 60, color=BLUE_LIGHT, step="pre", zorder=1)
    ax.plot([0, 0, 60, 60], [0, 1 / 60, 1 / 60, 0], color=BLUE, linewidth=2)
    ax.set_ylim(0, 0.025)
    fulcrum(ax, 30, "balances at 30")
    ax.set_xlim(-3, 63)
    ax.set_xlabel("waiting time for the bus (minutes)")
    ax.set_ylabel("density")
    ax.grid(axis="y")
    ax.set_title("Continuous: uniform on [0, 60]")

    ax = fig.add_subplot(gs[1, 2])
    x = np.linspace(0, 5, 400)
    ax.plot(x, np.exp(-x), color=BLUE, linewidth=2)
    median = np.log(2)
    ax.fill_between(x[x <= median], np.exp(-x[x <= median]), color=BLUE_LIGHT)
    vline(ax, 1.0, "mean = 1 (balance point)", ORANGE)
    vline(ax, median, f"median = ln 2 ≈ {median:.2f} (half the area)", INK2, "--")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The mean is not the halfway split")
    save(fig, name="01_mean_balance_point.png")


# ================================================================ notebook 02
def binomial_pmf(n, p):
    k = np.arange(n + 1)
    return k, np.array([comb(n, i) * p ** i * (1 - p) ** (n - i) for i in k])


def median_of_pmf(values, pmf):
    cdf = np.cumsum(pmf)
    lower = values[np.argmax(cdf >= 0.5 - 1e-12)]
    if np.isclose(cdf[values == lower][0], 0.5):
        return (lower + values[list(values).index(lower) + 1]) / 2
    return lower


def sheet_02_median_mode():
    salaries = np.array([22_000] * 9 + [2_300_000])
    mean_sal, median_sal = salaries.mean(), np.median(salaries)
    without = salaries[:9]

    fig = new_figure("Mean, median and mode: three answers to \"where is the center?\"",
                     "The median is the middle value, the mode the most frequent one. Skew and outliers pull them apart.")
    gs = grid(fig, top=0.84, wspace=0.3)
    style = {"mean": (ORANGE, "-"), "median": (BLUE, "--"), "mode": (AQUA, ":")}

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(np.full(9, 22_000), np.arange(9) * 0.25 + 0.3, s=60, color=BLUE_LIGHT, edgecolor=BLUE, zorder=3)
    ax.scatter([2_300_000], [0.3], s=60, color=BLUE_LIGHT, edgecolor=BLUE, zorder=3)
    ax.text(2_300_000, 0.65, "Michael\nJordan", ha="center", fontsize=10)
    ax.set_xscale("log")
    for name, value in [("mean", mean_sal), ("median", median_sal)]:
        color, ls = style[name]
        vline(ax, value, f"{name}: {value:,.0f}", color, ls)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xlabel("salary (dollars, log scale)")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.set_title("9 graduates and one superstar")

    for col, (p, title) in enumerate([(0.5, "Binomial(5, 0.5): symmetric"), (0.3, "Binomial(5, 0.3): skewed")]):
        k, pmf = binomial_pmf(5, p)
        mean, median = 5 * p, median_of_pmf(k, pmf)
        modes = k[np.isclose(pmf, pmf.max())]
        ax = fig.add_subplot(gs[0, col + 1])
        ax.set_ylim(0, 0.48)
        stems(ax, k, pmf, color=BLUE_LIGHT, label_fmt=lambda v: f"{v:.3f}")
        for m in modes:
            ax.bar(m, pmf[m], width=0.6, color=AQUA, zorder=2)
        vline(ax, mean, f"mean = {mean:g}", *style["mean"])
        vline(ax, median, f"median = {median:g}", *style["median"])
        ax.bar([], [], color=AQUA, label="mode" + ("s = " if len(modes) > 1 else " = ") +
               " and ".join(str(m) for m in modes))
        ax.set_xticks(k)
        ax.set_xlabel("heads in 5 coins")
        ax.set_ylabel("probability")
        ax.legend(loc="upper right", fontsize=9.5)
        ax.set_title(title)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(-3.5, 3.5, 400)
    ax.plot(x, normal_pdf(x), color=BLUE, linewidth=2)
    ax.axvline(0, color=INK2, linewidth=2)
    ax.text(0, 0.44, "mean = median = mode", fontsize=11, weight="bold", ha="center")
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.set_title("Normal: all three coincide")

    grouped_bars(fig.add_subplot(gs[1, 1]), ["9 graduates", "+ Michael Jordan"],
                 [("mean", [without.mean(), mean_sal], ORANGE), ("median", [np.median(without), median_sal], BLUE)],
                 "One outlier drags the mean, not the median", "", ylim=300_000,
                 label_fmt=lambda v: f"{v:,.0f}")
    ax = plt.gca()
    ax.set_ylabel("salary (dollars)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

    panel(fig.add_subplot(gs[1, 2]), "Which one to use", [
        "Mean: the balance point; uses every value,",
        "so one outlier can drag it far away.",
        "",
        "Median: the middle value (half below);",
        "robust to outliers. Use it for salaries.",
        "",
        "Mode: the most frequent value; can be",
        "two of them (5 fair coins: 2 and 3).",
        "",
        "Report all three: their disagreement",
        "is itself a sign of skew.",
    ], spacing=0.08)
    save(fig, name="02_median_mode.png")


# ================================================================ notebook 05
def sheet_05_variance():
    mu, var = moments(DIE, DIE_PMF)[:2]
    e_x2 = np.sum(DIE_PMF * DIE ** 2)
    assert np.isclose(var, 35 / 12) and np.isclose(e_x2 - mu ** 2, var)
    deviations = DIE - mu

    fig = new_figure("Variance: the average squared distance from the mean",
                     "Var(X) = E[(X − μ)²] = E[X²] − (E[X])².  Scaling: Var(aX + b) = a² Var(X).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(DIE, deviations, width=0.6, color=[ORANGE if d < 0 else BLUE for d in deviations], zorder=2)
    ax.axhline(0, color=INK2, linewidth=1)
    for x, d in zip(DIE, deviations):
        ax.text(x, d + (0.15 if d > 0 else -0.35), f"{d:+.1f}", ha="center", fontsize=10)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("die roll")
    ax.set_ylabel("x − 3.5")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Deviations cancel: their average is 0")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(DIE, deviations ** 2, width=0.6, color=BLUE, zorder=2)
    ax.axhline(var, color=ORANGE, linestyle="--", linewidth=1.8)
    ax.text(3.5, var + 0.25, f"average = 35/12 ≈ {var:.2f}", ha="center", color=ORANGE, fontsize=11, weight="bold")
    for x, d in zip(DIE, deviations):
        ax.text(x, d ** 2 + 0.12, f"{d ** 2:.2f}", ha="center", fontsize=9.5)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("die roll")
    ax.set_ylabel("(x − 3.5)²")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Squared deviations: their average is the variance")

    panel(fig.add_subplot(gs[0, 2]), "Two ways to compute it", [
        "Definition: average the squared deviations",
        r"$\mathrm{Var}(X) = \frac{1}{6}\left(2.5^2 + 1.5^2 + 0.5^2 + \ldots\right) = \frac{35}{12}$",
        "",
        "Shortcut:  E[X²] − (E[X])²",
        r"$= \frac{91}{6} - 3.5^2 = 15.17 - 12.25 = \frac{35}{12} \approx 2.92$",
        "",
        "Squaring stops positive and negative",
        "deviations from cancelling, and weights",
        "big deviations more.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    for values, color, offset, name in [((-2, 2), BLUE, 0.06, "win 2 / lose 2"), ((-1, 3), ORANGE, -0.06, "win 3 / lose 1")]:
        center = np.mean(values)
        ax.bar(np.array(values) + offset, [0.5, 0.5], width=0.12, color=color, label=name, zorder=2)
        ax.plot(center + offset, 0, marker="^", markersize=14, color=color, clip_on=False, zorder=5)
        ax.annotate("", xy=(values[1] + offset, 0.56 if color == BLUE else 0.62),
                    xytext=(center + offset, 0.56 if color == BLUE else 0.62),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.5))
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("payoff (dollars)")
    ax.set_ylabel("probability")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Same variance (4), different centers")

    ax = fig.add_subplot(gs[1, 1])
    y_values = 2 * DIE - 5
    ax.bar(DIE - 0.12, DIE_PMF, width=0.24, color=BLUE, label=f"X: die, Var = {var:.2f}", zorder=2)
    ax.bar(y_values + 0.12, DIE_PMF, width=0.24, color=ORANGE, label=f"Y = 2X − 5, Var = {4 * var:.2f}", zorder=2)
    ax.set_ylim(0, 0.3)
    ax.set_xticks(range(-3, 8))
    ax.set_xlabel("value")
    ax.set_ylabel("probability")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Y = 2X − 5: deviations double, variance ×4")

    panel(fig.add_subplot(gs[1, 2]), "Takeaways", [
        "±1 game: Var = 1.  ±100 game: Var = 10,000.",
        "Same mean, very different spread.",
        "",
        "Shifting (+b) moves the center but not",
        "the spread: the −1/+3 game has Var 4 too.",
        "",
        "Scaling by a multiplies deviations by a,",
        "so the variance by a²: 2.92 → 11.67.",
        "",
        "Units are squared (dollars²): fixed by the",
        "standard deviation, next notebook.",
    ], spacing=0.08)
    save(fig, name="05_variance.png")


# ================================================================ notebook 06
def sheet_06_standard_deviation():
    mu, sigma = 1.70, 0.10
    shares = [normal_within(k) for k in (1, 2, 3)]
    assert abs(shares[0] - 0.6827) < 1e-4 and abs(shares[2] - 0.9973) < 1e-4
    die_sd = sqrt(35 / 12)

    fig = new_figure("The standard deviation and the 68–95–99.7 rule",
                     "std(X) = √Var(X): spread in the same units as the data. For a normal: 68% within 1σ, 95% within 2σ, 99.7% within 3σ.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
    y = normal_pdf(x, mu, sigma)
    for k, color in [(3, "#cde2fb"), (2, "#86b6ef"), (1, "#3987e5")]:
        band = np.abs(x - mu) <= k * sigma
        ax.fill_between(x[band], y[band], color=color, zorder=1)
    ax.plot(x, y, color=INK, linewidth=1.5, zorder=2)
    for k, share, height in [(1, shares[0], 2.0), (2, shares[1], 0.75), (3, shares[2], 0.22)]:
        ax.annotate("", xy=(mu - k * sigma, height), xytext=(mu + k * sigma, height),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2))
        ax.text(mu, height + 0.1, f"{share:.1%} within ±{k}σ", ha="center", fontsize=11,
                weight="bold", color="white" if k == 1 else INK, zorder=4)
    ax.set_xticks([mu + k * sigma for k in range(-3, 4)],
                  [f"{mu + k * sigma:.2f}\n({k:+d}σ)" if k else f"{mu:.2f}\n(μ)" for k in range(-3, 4)])
    ax.set_xlabel("height (m), μ = 1.70, σ = 0.10")
    ax.set_ylabel("density")
    ax.set_title("Heights: about 68% between 1.60 and 1.80 m, 95% between 1.50 and 1.90 m")

    panel(fig.add_subplot(gs[0, 2]), "Why the square root", [
        "Variance has squared units:",
        "heights in m → variance in m².",
        "",
        "std = √Var is back in the data's units:",
        "±1 game: std 1.   ±100 game: std 100.",
        f"Fair die: √(35/12) ≈ {die_sd:.2f} pips from 3.5.",
        "",
        "A 2.00 m person is at +3σ: taller than",
        f"about {(1 + shares[2]) / 2:.2%} of this population.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    z = np.linspace(-3.5, 3.5, 400)
    ax.plot(z, normal_pdf(z), color=BLUE, linewidth=2)
    for k in (-1, 1):
        ax.plot(k, normal_pdf(k), "o", color=ORANGE, markersize=9, zorder=3)
    ax.annotate("bends change here:\nat μ ± σ", (1, normal_pdf(1)), textcoords="offset points",
                xytext=(18, 18), fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_xlabel("z (standard deviations from μ)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.47)
    ax.grid(True)
    ax.set_title("Where is σ on the bell? The inflection points")

    ax = fig.add_subplot(gs[1, 1])
    for s, color in [(0.5, ORANGE), (1, BLUE), (2, AQUA)]:
        ax.plot(z * 2, normal_pdf(z * 2, 0, s), color=color, linewidth=2, label=f"σ = {s:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Same mean, different σ: the same rule holds")

    bars(fig.add_subplot(gs[1, 2]), ["std(X)\ndie", "std(2X − 5)", "Var(X)\ndie", "Var(2X − 5)"],
         [die_sd, 2 * die_sd, 35 / 12, 4 * 35 / 12], "Scaling: std × |a|, variance × a²", "",
         ylabel="value", ylim=14, label_fmt=lambda v: f"{v:.2f}", highlight=1)
    save(fig, name="06_standard_deviation.png")


# ================================================================ notebook 07
def sheet_07_sum_of_gaussians():
    mu_t, sd_t, mu_l, sd_l = 10.0, 2.0, 5.0, 1.0
    mu_r, var_r = mu_t + mu_l, sd_t ** 2 + sd_l ** 2
    rng = np.random.default_rng(7)
    sample = rng.normal(mu_t, sd_t, 100_000) + rng.normal(mu_l, sd_l, 100_000)

    fig = new_figure("The sum of independent Gaussians is Gaussian: means add, variances add",
                     "Response time R = processing T ~ N(10, 2²) + latency L ~ N(5, 1²), independent.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    x = np.linspace(0, 25, 600)
    ax.hist(sample, bins=80, density=True, color=BLUE_LIGHT, label="simulated T + L (100,000 draws)")
    ax.plot(x, normal_pdf(x, mu_l, sd_l), color=AQUA, linewidth=2.2, label="L ~ N(5, 1)")
    ax.plot(x, normal_pdf(x, mu_t, sd_t), color=ORANGE, linewidth=2.2, label="T ~ N(10, 4)")
    ax.plot(x, normal_pdf(x, mu_r, sqrt(var_r)), color=BLUE, linewidth=2.6, label="R ~ N(15, 5)")
    ax.set_xlabel("milliseconds")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("The two components and their sum")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$\mu_R = \mu_T + \mu_L = 10 + 5 = 15$",
        r"$\sigma_R^2 = \sigma_T^2 + \sigma_L^2 = 4 + 1 = 5$",
        r"$\sigma_R = \sqrt{5} \approx 2.24$, not 2 + 1 = 3",
        "",
        "General:  W = aX + bY  (independent)",
        r"$\mu_W = a\mu_X + b\mu_Y$",
        r"$\sigma_W^2 = a^2\sigma_X^2 + b^2\sigma_Y^2$",
        "",
        f"`simulated: mean {sample.mean():.2f}, std {sample.std():.2f}`",
    ], spacing=0.09)

    bars(fig.add_subplot(gs[1, 0]), ["σ_T + σ_L\n(wrong)", "√(σ_T² + σ_L²)", "simulated\nstd"],
         [sd_t + sd_l, sqrt(var_r), sample.std()], "Standard deviations do not add", "",
         ylabel="milliseconds", ylim=4, label_fmt=lambda v: f"{v:.2f}", highlight=0)

    ax = fig.add_subplot(gs[1, 1])
    z = np.linspace(-4, 4, 400)
    ax.plot(z, normal_pdf(z, 0, 1), color=ORANGE, linewidth=2.2, label="one measurement, σ")
    ax.plot(z, normal_pdf(z, 0, 1 / sqrt(2)), color=BLUE, linewidth=2.2, label="average of two, σ/√2")
    ax.set_xlabel("measurement error (in units of σ)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Averaging two measurements halves the variance")

    panel(fig.add_subplot(gs[1, 2]), "Two consequences", [
        "Difference D = T − L  (a = 1, b = −1):",
        r"$\mu_D = 5, \quad \sigma_D^2 = 4 + (-1)^2 \cdot 1 = 5$",
        "",
        "Subtracting noise adds as much spread",
        "as adding it.",
        "",
        "Average W = ½X + ½Y of two independent",
        "measurements: same center, variance σ²/2.",
        "That is why experiments are repeated.",
        "",
        "Variances add only for independent variables.",
    ], spacing=0.08)
    save(fig, name="07_sum_of_gaussians.png")


# ================================================================ notebook 08
def sheet_08_standardizing():
    mu, var = moments(DIE, DIE_PMF)[:2]
    sigma = sqrt(var)
    centered = DIE - mu
    z = centered / sigma
    assert np.isclose(z.mean(), 0) and np.isclose(np.sqrt(np.mean(z ** 2)), 1)

    fig = new_figure("Standardizing: Z = (X − μ) / σ has mean 0 and standard deviation 1",
                     "Step 1: subtract the mean (center).  Step 2: divide by the standard deviation (scale). The shape never changes.")
    gs = grid(fig, top=0.84, wspace=0.3)
    steps = [(DIE, f"Die roll X: μ = 3.5, σ = {sigma:.2f}", BLUE_LIGHT),
             (centered, "Centered X − 3.5: μ = 0, σ unchanged", BLUE),
             (z, f"Standardized Z = (X − 3.5) / {sigma:.2f}: μ = 0, σ = 1", ORANGE)]
    for col, (values, title, color) in enumerate(steps):
        ax = fig.add_subplot(gs[0, col])
        ax.set_ylim(0, 0.3)
        ax.bar(values, DIE_PMF, width=0.25 if col == 2 else 0.4, color=color, zorder=2)
        center, spread = values.mean(), np.sqrt(np.mean((values - values.mean()) ** 2))
        ax.plot(center, 0, marker="^", markersize=15, color=INK, clip_on=False, zorder=5)
        ax.annotate("", xy=(center - spread, 0.23), xytext=(center + spread, 0.23),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
        ax.text(center, 0.245, "±1 std", ha="center", fontsize=10)
        for v in values:
            ax.text(v, 1 / 6 + 0.008, f"{v:.2f}" if col == 2 else f"{v:g}", ha="center", va="bottom",
                    fontsize=8.5 if col == 2 else 9, rotation=90 if col == 2 else 0)
        ax.set_xlim(-3.2, 6.8)
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.grid(axis="y", zorder=0)
        ax.set_title(title, fontsize=11.5)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(0, 110, 600)
    for name, m, s, score, color in [("Exam A", 70, 8, 82, BLUE), ("Exam B", 50, 15, 74, ORANGE)]:
        ax.plot(x, normal_pdf(x, m, s), color=color, linewidth=2.2, label=f"{name}: μ = {m}, σ = {s}")
        ax.axvline(score, color=color, linestyle="--", linewidth=1.5)
        ax.text(score + 1, normal_pdf(m, m, s) * 0.8, f"{score}", color=color, fontsize=11, weight="bold")
    ax.set_xlabel("raw score")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Raw scores: 82 on A looks better than 74 on B")

    ax = fig.add_subplot(gs[1, 1])
    zz = np.linspace(-3.5, 3.5, 400)
    ax.plot(zz, normal_pdf(zz), color=INK2, linewidth=2, label="both exams after standardizing")
    for value, color, name in [((82 - 70) / 8, BLUE, "you"), ((74 - 50) / 15, ORANGE, "friend")]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.8)
        ax.text(value + (0.08 if name == "friend" else -0.08), 0.36, f"{name}\nz = {value:.1f}", color=color,
                fontsize=11, weight="bold", ha="left" if name == "friend" else "right")
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.set_title("z-scores: the friend did slightly better")

    panel(fig.add_subplot(gs[1, 2]), "The recipe", [
        r"$Z = \dfrac{X - \mu}{\sigma}$",
        "",
        "",
        "Centering: E[X − μ] = μ − μ = 0.",
        "Scaling: std(X / σ) = σ / σ = 1.",
        "",
        "z = 1.5 means 1.5 std above the mean.",
        "If X is normal, Z ~ N(0, 1), so one table",
        "serves every normal: |z| > 2 happens",
        f"only about {1 - normal_within(2):.0%} of the time.",
    ], spacing=0.08)
    save(fig, name="08_standardizing.png")


# ================================================================ notebook 10
def sheet_10_skewness():
    lottery = (np.array([-1, 99]), np.array([0.99, 0.01]))
    insurance = (np.array([1, -99]), np.array([0.99, 0.01]))
    lot, ins = moments(*lottery), moments(*insurance)
    e_x3 = np.sum(lottery[1] * lottery[0].astype(float) ** 3)
    assert np.isclose(lot[0], 0) and np.isclose(lot[1], 99) and np.isclose(lot[2], -ins[2])

    fig = new_figure("Skewness: the third moment tells mirror-image games apart",
                     "skew(X) = E[((X − μ)/σ)³].  The lottery and the insurance have the same mean and variance.")
    gs = grid(fig, top=0.84, wspace=0.3)

    for col, ((values, pmf), title, color, stats) in enumerate([
            (lottery, "Lottery: lose 1 almost always, win 99 rarely", BLUE, lot),
            (insurance, "Insurance: win 1 almost always, lose 99 rarely", ORANGE, ins)]):
        ax = fig.add_subplot(gs[0, col])
        ax.bar(values, pmf, width=4, color=color, zorder=2)
        ax.plot(values, pmf, "o", color=color, markersize=7, zorder=3)
        for v, p in zip(values, pmf):
            ax.text(v, p + 0.04, f"{v:+d} dollar{'s' if abs(v) != 1 else ''}\np = {p}", ha="center", fontsize=10.5)
        ax.set_xlim(-110, 110)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("payoff (dollars)")
        ax.set_ylabel("probability")
        ax.grid(axis="y", zorder=0)
        ax.set_title(title, fontsize=12)
        ax.text(0.72 if col == 0 else 0.28, 0.6, f"skewness = {stats[2]:+.2f}", transform=ax.transAxes,
                ha="center", fontsize=12, weight="bold", color=color)

    panel(fig.add_subplot(gs[0, 2]), "Moments one, two, three", [
        "`             E[X]  Var   E[X³]   skew`",
        f"`lottery      {lot[0]:4.0f}  {lot[1]:4.0f}  {e_x3:+6.0f}  {lot[2]:+5.2f}`",
        f"`insurance    {ins[0]:4.0f}  {ins[1]:4.0f}  {-e_x3:+6.0f}  {ins[2]:+5.2f}`",
        "",
        "Same mean, same variance: only the",
        "third moment differs, and it flips sign.",
        "",
        r"$\mathrm{skew} = \dfrac{E[X^3]}{\sigma^3} = \dfrac{9702}{99^{3/2}} \approx 9.85$",
    ], spacing=0.095)

    rng = np.random.default_rng(42)
    lottery_outcomes = rng.choice([-1, 99], p=[0.99, 0.01], size=2000)
    insurance_outcomes = rng.choice([1, -99], p=[0.99, 0.01], size=2000)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.cumsum(lottery_outcomes), color=BLUE, linewidth=1.6, label="lottery player")
    ax.plot(np.cumsum(insurance_outcomes), color=ORANGE, linewidth=1.6, label="insurance company")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative winnings (dollars)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    ax.set_title("Living the games: slow bleed vs. rare shocks")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(0, 10, 500)
    gamma2 = x * np.exp(-x)
    ax.plot(x, gamma2, color=BLUE, linewidth=2.2, label="right tail: skew +1.41")
    ax.plot(10 - x, gamma2, color=ORANGE, linewidth=2.2, label="left tail: skew −1.41")
    xs = np.linspace(0, 10, 500)
    ax.plot(xs, normal_pdf(xs, 5, sqrt(2)), color=INK2, linewidth=1.6, linestyle="--", label="symmetric: skew 0")
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The sign of the skew points to the long tail")

    panel(fig.add_subplot(gs[1, 2]), "Reading skewness", [
        "Positive: a long right tail (rare big wins).",
        "Negative: a long left tail (rare big losses).",
        "Zero: symmetric. In the ±1 coin game the",
        "odd moments cancel in pairs.",
        "",
        "The warm-up variable (−2, 0, 1 with",
        "p = 1/3, 1/6, 1/2) has skew ≈ −0.52:",
        "the −2 sits far out on the left.",
        "",
        "Skewness is scale-free: dollars or cents,",
        "the number stays the same.",
    ], spacing=0.08)
    save(fig, name="10_skewness.png")


# ================================================================ notebook 11
def sheet_11_kurtosis():
    game1 = (np.array([-1.0, 1.0]), np.array([0.5, 0.5]))
    game2 = (np.array([-10.0, -0.1, 0.1, 10.0]), np.array([1, 100, 100, 1]) / 202)
    g1, g2 = moments(*game1), moments(*game2)
    assert np.isclose(g1[1], 1) and np.isclose(g2[1], 1) and abs(g2[3] - 99.01) < 0.01

    fig = new_figure("Kurtosis: the fourth moment measures tail thickness",
                     "kurt(X) = E[((X − μ)/σ)⁴].  Two games with the same mean, variance and skewness.")
    gs = grid(fig, top=0.84, wspace=0.3)

    for col, ((values, pmf), title, color) in enumerate([
            (game1, "Game 1: ±1 dollar, 50/50", BLUE),
            (game2, "Game 2: ±10 cents usually, ±10 dollars rarely", ORANGE)]):
        ax = fig.add_subplot(gs[0, col])
        ax.bar(values, pmf, width=0.35, color=color, zorder=2)
        ax.plot(values, pmf, "o", color=color, markersize=6, zorder=3)
        if col == 0:
            ax.text(0, 0.54, "−1 and +1, p = 1/2 each", ha="center", fontsize=10.5)
        else:
            ax.text(0, 0.54, "−0.1 and +0.1, p = 50/101 each", ha="center", fontsize=10.5)
            for v in (-10, 10):
                ax.text(v, 0.04, f"{v:+d}\np = 1/202", ha="center", fontsize=10)
        ax.set_xlim(-11.5, 11.5)
        ax.set_ylim(0, 0.68)
        ax.set_xlabel("payoff (dollars)")
        ax.set_ylabel("probability")
        ax.grid(axis="y", zorder=0)
        ax.set_title(title, fontsize=12)

    panel(fig.add_subplot(gs[0, 2]), "All four descriptors", [
        "`          mean  var   skew  kurtosis`",
        f"`game 1    {g1[0]:4.1f}  {g1[1]:4.2f}  {g1[2]:4.1f}  {g1[3]:8.2f}`",
        f"`game 2    {g2[0]:4.1f}  {g2[1]:4.2f}  {g2[2]:4.1f}  {g2[3]:8.2f}`",
        "",
        "Mean, variance and skewness all match.",
        "Only the kurtosis sees that game 2 spends",
        "its variance on rare ±10 shocks.",
        "",
        "Toolkit: center (mean), spread (variance),",
        "asymmetry (skewness), tails (kurtosis).",
    ], spacing=0.09)

    rng = np.random.default_rng(11)
    traces = gs[1, 0].subgridspec(2, 1, hspace=0.25)
    for row, ((values, pmf), color, name) in enumerate([(game1, BLUE, "game 1"), (game2, ORANGE, "game 2")]):
        outcomes = rng.choice(values, p=pmf, size=3000)
        ax = fig.add_subplot(traces[row])
        ax.plot(outcomes, color=color, linewidth=0.6)
        ax.set_ylim(-11, 11)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True)
        if row == 0:
            ax.set_title("Same variance, opposite texture")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("round")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-4.5, 4.5, 900)
    half = sqrt(3)
    b = 1 / sqrt(2)
    ax.plot(x, np.where(np.abs(x) <= half, 1 / (2 * half), 0), color=AQUA, linewidth=2, label="uniform: 1.8")
    ax.plot(x, normal_pdf(x), color=BLUE, linewidth=2, label="normal: 3")
    ax.plot(x, np.exp(-np.abs(x) / b) / (2 * b), color=ORANGE, linewidth=2, label="Laplace: 6")
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("x (all with variance 1)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10, title="kurtosis", title_fontsize=10)
    ax.grid(True)
    ax.set_title("The kurtosis ladder at equal variance")

    panel(fig.add_subplot(gs[1, 2]), "Reading kurtosis", [
        "1: the minimum (a two-point coin flip).",
        "1.8 uniform, 3 normal, 6 Laplace.",
        "Higher = more of the variance comes from",
        "rare, large deviations.",
        "",
        "Risk report: two portfolios, both with",
        "mean 0% and std 1% a day. Kurtosis 3 vs 30:",
        "the second is calmer on most days but has",
        "rare violent moves.",
    ], spacing=0.085)
    save(fig, name="11_kurtosis.png")


# ================================================================ notebook 03
def sheet_03_expected_value_of_a_function():
    e_x = np.sum(DIE * DIE_PMF)
    e_x2 = np.sum(DIE ** 2 * DIE_PMF)
    e_lin = np.sum((2 * DIE - 5) * DIE_PMF)
    assert np.isclose(e_x2, 91 / 6) and np.isclose(e_lin, 2 * e_x - 5) and np.isclose(e_lin, 2)
    rng = np.random.default_rng(42)
    simulated = (rng.integers(1, 7, size=100_000) ** 2).mean()

    fig = new_figure("E[g(X)]: keep the probabilities, transform the values",
                     "E[g(X)] = Σ g(x) · p(x).  For a linear g: E[aX + b] = a · E[X] + b.  In general E[g(X)] ≠ g(E[X]).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(0, 0.25)
    stems(ax, DIE, DIE_PMF)
    fulcrum(ax, e_x, f"balances at {e_x:g}")
    ax.set_xticks(DIE)
    ax.set_xlabel("die roll x")
    ax.set_ylabel("probability")
    ax.set_title("The roll X: six bars of 1/6")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, 0.25)
    stems(ax, DIE ** 2, DIE_PMF, width=1.4)
    ax.axvline(e_x ** 2, color=INK2, linestyle=":", linewidth=1.8, zorder=3)
    ax.text(e_x ** 2 - 0.6, 0.215, f"(E[X])² = {e_x ** 2:g}", ha="right", fontsize=10.5, color=INK2)
    fulcrum(ax, e_x2, f"balances at {e_x2:.2f}")
    ax.set_xticks(DIE ** 2)
    ax.set_xlabel("payoff x²")
    ax.set_ylabel("probability")
    ax.set_title("The payoff X²: same bars, moved")

    panel(fig.add_subplot(gs[0, 2]), "The square game", [
        "Your friend pays the square of the roll.",
        "Fair price = the expected payoff:",
        r"$E[X^2] = \frac{1}{6}(1 + 4 + 9 + 16 + 25 + 36) = \frac{91}{6} \approx 15.17$",
        "",
        "Plugging in the mean is wrong:",
        r"$(E[X])^2 = 3.5^2 = 12.25 \neq 15.17$",
        "",
        f"`100,000 simulated games: {simulated:.2f}`",
        "",
        "g(x) happens exactly when x does, so it",
        "inherits p(x): no new distribution needed.",
    ], spacing=0.08)

    rows = gs[1, 0].subgridspec(3, 1, hspace=0.25)
    for r, (values, color, name, rule) in enumerate([
            (DIE, BLUE, "X", "3.5"), (2 * DIE, AQUA, "2X", "2 · 3.5 = 7"),
            (2 * DIE - 5, ORANGE, "2X − 5", "2 · 3.5 − 5 = 2")]):
        mean = np.sum(values * DIE_PMF)
        ax = fig.add_subplot(rows[r])
        ax.bar(values, DIE_PMF, width=0.5, color=color, zorder=2)
        ax.plot(mean, 0, marker="^", markersize=13, color=INK, clip_on=False, zorder=5)
        ax.text(4.5, 0.3, f"{name}: mean = {rule}", fontsize=10.5, weight="bold", color=color,
                ha="center", va="center")
        ax.set_xlim(-4, 13)
        ax.set_ylim(0, 0.36)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks(range(-3, 13))
        ax.grid(axis="x", zorder=0)
        if r == 0:
            ax.set_title("Linearity: the mean follows ax + b")
        if r < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("value (each bar has probability 1/6)")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(0.6, 6.4, 300)
    ax.plot(x, x ** 2, color=BLUE, linewidth=2, label="g(x) = x²")
    ax.plot(DIE, DIE ** 2, "o", color=BLUE, markersize=7, zorder=3)
    ax.plot(x, 2 * x - 5, color=AQUA, linewidth=2, label="g(x) = 2x − 5")
    ax.plot(DIE, 2 * DIE - 5, "o", color=AQUA, markersize=6, zorder=3)
    ax.plot([e_x, e_x], [e_x ** 2, e_x2], color=ORANGE, linewidth=2.5, zorder=4)
    ax.plot(e_x, e_x2, "o", color=ORANGE, markersize=10, zorder=5)
    ax.plot(e_x, e_x ** 2, "o", color=INK2, markersize=9, zorder=5)
    ax.text(e_x - 0.2, e_x2 + 1.5, f"E[X²] = {e_x2:.2f}", ha="right", fontsize=10.5, color=ORANGE, weight="bold")
    ax.text(e_x + 0.25, e_x ** 2 - 2.5, f"(E[X])² = {e_x ** 2:g}", ha="left", fontsize=10.5, color=INK2, weight="bold")
    ax.text(4.1, -3.6, "line: E[g(X)] = g(E[X]) = 2", fontsize=10, color=AQUA, weight="bold")
    ax.set_ylim(-5, 40)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("A curved g opens a gap, a straight one does not")

    panel(fig.add_subplot(gs[1, 2]), "Linearity of expectation", [
        r"$E[aX + b] = a\,E[X] + b$  for any distribution",
        r"$\sum (ax + b)\,p(x) = a \sum x\,p(x) + b \sum p(x)$",
        "",
        "Double game, net win 2X − 5:",
        r"$E[2X - 5] = 2 \cdot 3.5 - 5 = 2$ dollars per play",
        "",
        r"Special cases: $E[aX] = a\,E[X]$ and $E[b] = b$.",
        "",
        f"The gap E[X²] − (E[X])² = {frac_label(e_x2 - e_x ** 2, 12)} is the",
        "variance, the subject of the next notebook.",
    ], spacing=0.08)
    save(fig, name="03_expected_value_of_a_function.png")


# ================================================================ notebook 04
def sheet_04_sum_of_expectations():
    coin, die = np.array([0, 1]), DIE
    totals = np.arange(1, 8)
    p_total = np.array([sum(0.5 / 6 for c in coin for d in die if c + d == t) for t in totals])
    assert np.isclose(np.sum(totals * p_total), 0.5 + 3.5)

    people = ["Aisha", "Beto", "Cameron"]
    handouts = list(permutations(range(3)))
    hits = np.array([[int(h[i] == i) for i in range(3)] for h in handouts])
    matrix = np.column_stack([hits, hits.sum(axis=1)])
    assert np.allclose(hits.mean(axis=0), 1 / 3) and np.isclose(matrix[:, 3].mean(), 1)

    rng = np.random.default_rng(42)
    ns = [2, 3, 5, 10, 30, 100, 300, 1000]
    sim_means = [sum(np.sum(rng.permutation(n) == np.arange(n)) for _ in range(5000)) / 5000 for n in ns]

    fig = new_figure("Expectations add: E[X + Y] = E[X] + E[Y], always",
                     "Any distributions, independent or not. Split a hard count into 0/1 pieces and add their expectations.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(0, 0.24)
    stems(ax, totals, p_total, label_fmt=lambda v: frac_label(v, 12))
    fulcrum(ax, 4, "balances at 0.5 + 3.5 = 4")
    ax.set_xticks(totals)
    ax.set_xlabel("coin (1 dollar on heads) + die roll")
    ax.set_ylabel("probability")
    ax.set_title("Two-part game: total payoff")

    ax = fig.add_subplot(gs[0, 1])
    rows = [" ".join(people[j][0] for j in h) for h in handouts]
    labels = [[str(v) for v in row] for row in matrix]
    heat(ax, matrix, rows, people + ["matches"], "Name game, n = 3: all 6 handouts",
         "names given to A, B, C", "", labels=labels, vmax=3, highlight_cols=(3,))
    for j, avg in enumerate(matrix.mean(axis=0)):
        ax.text(j + 0.5, 6.45, f"avg {frac_label(avg)}", ha="center", va="top", fontsize=10.5,
                color=ORANGE if j == 3 else INK2, weight="bold")

    panel(fig.add_subplot(gs[0, 2]), "The rule", [
        r"$E[X_1 + \cdots + X_n] = E[X_1] + \cdots + E[X_n]$",
        "No independence needed.",
        "",
        "Name game: matches = A + B + C, where",
        "A = 1 if Aisha gets her own name, else 0.",
        r"$E[A] = P(\mathrm{Aisha\ matches}) = 1/3$, so",
        r"$E[\mathrm{matches}] = 1/3 + 1/3 + 1/3 = 1$",
        "",
        "A, B, C are dependent (two matches force",
        "the third), and the rule does not care.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    match_values, match_probs = np.arange(4), np.array([2, 3, 0, 1]) / 6
    ax.set_ylim(0, 0.65)
    stems(ax, match_values, match_probs, label_fmt=lambda v: frac_label(v) if v else "")
    fulcrum(ax, 1, "balances at 1")
    ax.annotate("impossible: two right\nforces all three", (2, 0.01), xytext=(2.1, 0.33),
                ha="center", fontsize=10, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xticks(match_values)
    ax.set_xlabel("number of matches")
    ax.set_ylabel("probability")
    ax.set_title("Matches for 3 people")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ns, sim_means, "o-", color=BLUE, linewidth=2, markersize=7, label="simulated, 5,000 games each")
    ax.axhline(1, color=ORANGE, linestyle="--", linewidth=1.8, label="theory: n · 1/n = 1")
    ax.set_xscale("log")
    ax.set_ylim(0, 2)
    ax.set_xlabel("number of people n (log scale)")
    ax.set_ylabel("expected matches")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("E[matches] = 1 for every n")

    panel(fig.add_subplot(gs[1, 2]), "8 billion names", [
        "Each of n people gets their own name",
        "with probability 1/n, so",
        r"$E[\mathrm{matches}] = n \cdot \frac{1}{n} = 1$",
        "for 3 people or for 8,000,000,000.",
        "Brute force needs n! handouts; this needs",
        "one line.",
        "",
        "Total dependence is fine too: a die roll X",
        "and Y = 7 − X give",
        r"$E[X + Y] = 3.5 + 3.5 = 7$  (X + Y is always 7).",
    ], spacing=0.08)
    save(fig, name="04_sum_of_expectations.png")


# ================================================================ notebook 09
def raw_moment(values, pmf, k):
    return np.sum(np.asarray(pmf, float) * np.asarray(values, float) ** k)


def central_moment(values, pmf, k):
    mu = raw_moment(values, pmf, 1)
    return np.sum(np.asarray(pmf, float) * (np.asarray(values, float) - mu) ** k)


def sheet_09_moments():
    values, pmf = np.array([-2, 0, 1]), np.array([1 / 3, 1 / 6, 1 / 2])
    raw = [raw_moment(values, pmf, k) for k in range(1, 5)]
    central = [central_moment(values, pmf, k) for k in range(1, 5)]
    assert np.allclose(raw, [-1 / 6, 11 / 6, -13 / 6, 35 / 6]) and np.isclose(central[1], 65 / 36)
    die_raw = [raw_moment(DIE, DIE_PMF, k) for k in range(1, 5)]
    assert np.allclose(die_raw, [3.5, 91 / 6, 73.5, 2275 / 6])

    sym = (np.array([-1.0, 0.0, 1.0]), np.array([0.25, 0.5, 0.25]))
    lop = (np.array([-1 / (2 * np.sqrt(2)), np.sqrt(2)]), np.array([0.8, 0.2]))
    stats = [(raw_moment(*d, 1), central_moment(*d, 2), central_moment(*d, 3)) for d in (sym, lop)]
    assert np.allclose(stats[0][:2], stats[1][:2]) and np.isclose(stats[0][2], 0)

    fig = new_figure("Moments: E[X], E[X²], E[X³], … , E[Xᵏ]",
                     "Same probabilities, values raised to the k-th power. Centered at μ, moments 2, 3, 4 give variance, skewness, kurtosis.")
    gs = grid(fig, top=0.84, wspace=0.3)

    row = gs[0, :2].subgridspec(1, 4, wspace=0.45)
    for k in range(1, 5):
        ax = fig.add_subplot(row[k - 1])
        contributions = pmf * values.astype(float) ** k
        ax.bar(np.arange(3), contributions, width=0.6, zorder=2,
               color=[ORANGE if c < 0 else BLUE for c in contributions])
        ax.axhline(0, color=INK2, linewidth=1)
        for p, c in zip(np.arange(3), contributions):
            if abs(c) > 1e-12:
                ax.text(p, c + (0.15 if c > 0 else -0.15), frac_label(c), ha="center",
                        va="bottom" if c > 0 else "top", fontsize=9.5)
        ax.set_xticks(np.arange(3), [str(v) for v in values])
        ax.set_ylim(-3.6, 6.4)
        ax.set_xlabel("x")
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"k = {k}:  sum = {frac_label(raw[k - 1])}", fontsize=11.5)
        if k == 1:
            ax.set_ylabel(r"$p(x) \cdot x^k$")

    panel(fig.add_subplot(gs[0, 2]), "The warm-up variable", [
        "x = −2, 0, 1 with p = 1/3, 1/6, 1/2.",
        "Bars at left: each value's share p(x)·xᵏ.",
        "`k    raw E[X^k]    central E[(X-mu)^k]`",
        *[f"`{k}    {frac_label(raw[k - 1]):>9}    {(central[k - 1] if abs(central[k - 1]) > 1e-12 else 0.0):>12.4f}`"
          for k in range(1, 5)],
        "",
        "Variance from the first two moments:",
        r"$E[X^2] - E[X]^2 = \frac{11}{6} - \frac{1}{36} = \frac{65}{36} \approx 1.81$",
        "",
        "Odd powers keep the sign of the −2;",
        "even powers make it the dominant term.",
    ], spacing=0.07, fontsize=12)

    for col, ((vals, probs), (m1, var, m3), color, name) in enumerate([
            (sym, stats[0], BLUE, "A: symmetric"), (lop, stats[1], ORANGE, "B: lopsided")]):
        ax = fig.add_subplot(gs[1, col])
        ax.bar(vals, probs, width=0.22, color=color, zorder=2)
        for v, p in zip(vals, probs):
            ax.text(v, p + 0.02, f"x = {v:.3g}\np = {p:g}", ha="center", va="bottom", fontsize=9.5)
        fulcrum(ax, 0, color=INK)
        ax.text(0.97, 0.95, f"mean {m1:.0f},  variance {var:.1f}\n3rd central moment {round(m3, 2) + 0.0:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, weight="bold", color=color)
        ax.set_xlim(-1.6, 1.9)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("value")
        ax.set_ylabel("probability")
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"{name}: same mean and variance", fontsize=12)

    panel(fig.add_subplot(gs[1, 2]), "Raw vs. central moments", [
        r"raw: $E[X^k]$        central: $E[(X - \mu)^k]$",
        "",
        "1st central = 0, always (deviations cancel).",
        "2nd central = the variance.",
        "3rd central → skewness (asymmetry).",
        "4th central → kurtosis (tail weight).",
        "",
        "A and B agree on moments 1 and 2;",
        "only the third tells them apart.",
        "",
        "`die: E[X^k] = " + ", ".join(f"{m:.2f}" for m in die_raw) + "`",
    ], spacing=0.08)
    save(fig, name="09_moments.png")


# ================================================================ lesson data for notebooks 12-16
NEWSPAPER = np.array([8.7, 14.2, 18.3, 18.4, 23.2, 25.9, 29.7, 35.2, 51.2, 54.7, 65.9, 75.0])

_Q_GRID = np.linspace(-6, 6, 200_001)
_Q_CDF = np.cumsum(normal_pdf(_Q_GRID)) * (_Q_GRID[1] - _Q_GRID[0])


def normal_quantile(p):
    """Standard-normal quantile by inverting a numerically integrated CDF (as in the notebooks)."""
    return _Q_GRID[np.searchsorted(_Q_CDF, p)]


def quartiles(data):
    return np.quantile(data, [0.25, 0.5, 0.75], method="midpoint")


def kde(x, data, sigma):
    return np.mean([normal_pdf(x, xi, sigma) for xi in data], axis=0)


def hide_y(ax):
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


# ================================================================ notebook 12
def sheet_12_quantiles():
    q1, q2, q3 = quartiles(NEWSPAPER)
    q90 = np.quantile(NEWSPAPER, 0.9, method="midpoint")
    assert np.allclose([q1, q2, q3, q90], [18.35, 27.8, 52.95, 60.3])
    q1_linear = np.quantile(NEWSPAPER, 0.25)

    fig = new_figure("Quantiles: the value with k% of the data to its left",
                     "The k% quantile q satisfies P(X ≤ q) = k/100. The quartiles Q1, Q2 (the median) and Q3 cut the data into four equal parts.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    edges = [0, q1, q2, q3, 82]
    for k in range(4):
        ax.axvspan(edges[k], edges[k + 1], color=BLUE_LIGHT if k % 2 == 0 else SURFACE, alpha=0.45, zorder=0)
        ax.text((edges[k] + edges[k + 1]) / 2 if k < 3 else (q3 + 82) / 2, -0.6, "3 points\n(25%)",
                ha="center", va="center", fontsize=10, color=INK2)
    ax.plot(NEWSPAPER, np.zeros(12), "o", color=BLUE, markersize=11, zorder=3)
    for v in NEWSPAPER:
        nudge = {18.3: (-0.4, "right"), 18.4: (0.4, "left")}.get(v, (0, "center"))
        ax.text(v + nudge[0], -0.2, f"{v:g}", ha=nudge[1], va="top", fontsize=9, color=INK2)
    for q, name, color, style in [(q1, "Q1", ORANGE, "--"), (q2, "median", INK, "-"),
                                  (q3, "Q3", ORANGE, "--"), (q90, "90%", AQUA, ":")]:
        ax.axvline(q, color=color, linestyle=style, linewidth=2, zorder=2)
        ax.text(q, 0.72, f"{name}\n{q:g}", ha="center", va="bottom", fontsize=11, weight="bold", color=color,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_xlim(0, 82)
    ax.set_ylim(-0.8, 1.2)
    hide_y(ax)
    ax.set_xlabel("newspaper ad budget (thousands of dollars), sorted")
    ax.set_title("The 12 newspaper budgets cut into four groups of three")

    panel(fig.add_subplot(gs[0, 2]), "Cutting between data points", [
        "Sort, then average the two values that",
        "straddle the cut (the course's convention):",
        "`Q1  = (18.3 + 18.4) / 2 = 18.35`",
        "`Q2  = (25.9 + 29.7) / 2 = 27.8`",
        "`Q3  = (51.2 + 54.7) / 2 = 52.95`",
        "`q.9 = (54.7 + 65.9) / 2 = 60.3`",
        "",
        "Software conventions differ: numpy's",
        f"default (linear) gives Q1 = {q1_linear:g}.",
        "",
        f"IQR = Q3 − Q1 = {q3 - q1:.1f}: the middle 50%.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(-4, 4, 800)
    q20 = normal_quantile(0.2)
    ax.plot(x, normal_pdf(x), color=BLUE, linewidth=2)
    ax.fill_between(x[x <= q20], normal_pdf(x[x <= q20]), color=ORANGE, alpha=0.7)
    ax.axvline(q20, color=INK, linestyle=":", linewidth=1.8)
    ax.annotate("area = 20%", (q20 - 0.45, 0.06), xytext=(-3.6, 0.2), fontsize=11, weight="bold",
                color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.text(q20 - 0.15, 0.36, f"$q_{{0.2}}$ = {q20:.2f}", ha="right", fontsize=11.5)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x (standard normal)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.set_title("For a distribution: an area under the PDF")

    ax = fig.add_subplot(gs[1, 1])
    x = np.linspace(-3.5, 3.5, 800)
    ax.plot(x, np.interp(x, _Q_GRID, _Q_CDF), color=BLUE, linewidth=2)
    for p, name in [(0.25, "Q1"), (0.5, "median"), (0.75, "Q3")]:
        q = normal_quantile(p)
        ax.plot([-3.5, q, q], [p, p, 0], ":", color=ORANGE, linewidth=1.6)
        ax.plot(q, p, "o", color=ORANGE, markersize=7, zorder=3)
        ax.text(-3.35, p + 0.015, f"{name} = {q:.3f}", va="bottom", fontsize=10.5, color=INK)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("value q")
    ax.set_ylabel("P(X ≤ q)")
    ax.grid(True)
    ax.set_title("Read the CDF backwards: probability → value")

    rng = np.random.default_rng(42)
    mu, sigma = 27.8, 15
    ns = [12, 100, 1000, 100_000]
    sample_q = np.array([np.quantile(rng.normal(mu, sigma, size=n), [0.25, 0.5, 0.75]) for n in ns])
    ax = fig.add_subplot(gs[1, 2])
    for j, (name, color) in enumerate([("Q1", ORANGE), ("median", BLUE), ("Q3", AQUA)]):
        true = mu + normal_quantile([0.25, 0.5, 0.75][j]) * sigma
        ax.axhline(true, color=color, linestyle="--", linewidth=1.4)
        ax.plot(ns, sample_q[:, j], "o-", color=color, linewidth=2, markersize=7, label=f"{name} ({true:.1f})")
    ax.set_xscale("log")
    ax.set_ylim(5, 58)
    ax.set_xlabel("sample size n from N(27.8, 15²), log scale")
    ax.set_ylabel("sample quantile")
    ax.legend(loc="upper center", fontsize=9.5, ncol=3, columnspacing=0.8, handlelength=1.5)
    ax.grid(True)
    ax.set_title("Sample quartiles settle (dashed = true)")
    save(fig, name="12_quantiles.png")


# ================================================================ notebook 13
def sheet_13_box_plots():
    mn, mx = NEWSPAPER.min(), NEWSPAPER.max()
    q1, q2, q3 = quartiles(NEWSPAPER)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    assert np.isclose(iqr, 34.6) and np.isclose(lo_fence, -33.55) and np.isclose(hi_fence, 104.85)
    full = dict(whislo=0.3, q1=12.75, med=25.75, q3=45.1, fliers=[100.9, 114.0])
    full["whishi"] = full["q3"] + 1.5 * (full["q3"] - full["q1"])
    assert np.isclose(full["whishi"], 93.625)

    fig = new_figure("Box plots: five numbers that show center, spread, skew and outliers",
                     "Box from Q1 to Q3 (the IQR) with the median inside. Whiskers reach toward 1.5 · IQR beyond the box but stop at the data; points past them are outliers.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    ax.add_patch(plt.Rectangle((q1, 0.8), q3 - q1, 0.4, facecolor=BLUE_LIGHT, edgecolor=INK, linewidth=1.5, zorder=2))
    ax.plot([q2, q2], [0.8, 1.2], color=ORANGE, linewidth=3.5, zorder=3)
    ax.plot([mn, q1], [1, 1], color=INK, linewidth=1.5)
    ax.plot([q3, mx], [1, 1], color=INK, linewidth=1.5)
    for end in (mn, mx):
        ax.plot([end, end], [0.92, 1.08], color=INK, linewidth=2)
    ax.plot(NEWSPAPER, np.full(12, 0.5), "o", color=BLUE, markersize=8, alpha=0.8)
    for fence, label, side in [(lo_fence, f"lower fence\nQ1 − 1.5·IQR = {lo_fence:g}", 1),
                               (hi_fence, f"upper fence\nQ3 + 1.5·IQR = {hi_fence:g}", -1)]:
        ax.axvline(fence, color=MUTED, linestyle=":", linewidth=1.8)
        ax.text(fence + 2 * side, 1.42, label, ha="left" if side > 0 else "right", fontsize=10, color=INK2)
    ax.text(q1, 0.72, f"Q1 = {q1:g}", ha="center", va="top", fontsize=10.5)
    ax.text(q3, 0.72, f"Q3 = {q3:g}", ha="center", va="top", fontsize=10.5)
    ax.text(q2, 1.27, f"median = {q2:g}", ha="center", fontsize=10.5, weight="bold", color=ORANGE)
    ax.text(mn, 1.13, f"min {mn:g}", ha="center", fontsize=10)
    ax.text(mx, 1.13, f"max {mx:g}", ha="center", fontsize=10)
    ax.annotate("", xy=(q1, 0.86), xytext=(q3, 0.86), arrowprops=dict(arrowstyle="<->", color=INK2))
    ax.text((q1 + q3) / 2 + 8, 0.9, f"IQR = {iqr:.1f}", ha="center", fontsize=10.5, color=INK2)
    ax.text(85, 0.5, "the 12 data points", va="center", fontsize=10, color=BLUE)
    ax.set_xlim(-45, 118)
    ax.set_ylim(0.35, 1.65)
    hide_y(ax)
    ax.set_xlabel("newspaper ad budget (thousands of dollars)")
    ax.set_title("Anatomy on the 12 budgets: the fences lie beyond the data, so the whiskers stop at min and max", fontsize=12)

    panel(fig.add_subplot(gs[0, 2]), "Reading this box plot", [
        "Five numbers: 8.7, 18.35, 27.8, 52.95, 75",
        f"IQR = {q3:g} − {q1:g} = {iqr:.1f}",
        f"Fences: {lo_fence:g} and {hi_fence:g}",
        "All 12 points are inside: no outliers.",
        "",
        f"Skew: the median is {q2 - q1:.2f} above Q1",
        f"but {q3 - q2:.2f} below Q3, and the upper",
        f"whisker ({mx - q3:.2f}) beats the lower ({q1 - mn:.2f}):",
        "right-skewed.",
        "",
        "1.5 · IQR is a convention, not a law.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    ax.bxp([dict(label="", **full)], vert=False, patch_artist=True, widths=0.45,
           boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=INK), medianprops=dict(color=ORANGE, linewidth=3),
           flierprops=dict(marker="o", markerfacecolor=ORANGE, markeredgecolor=ORANGE, markersize=8))
    ax.axvline(full["whishi"], color=MUTED, linestyle=":", linewidth=1.5)
    ax.text(full["whishi"] - 2, 1.38, f"fence {full['whishi']:g}", ha="right", fontsize=10, color=INK2)
    ax.text(107.5, 0.72, "2 outliers", ha="center", fontsize=10.5, weight="bold", color=ORANGE)
    ax.text(full["med"], 1.3, f"median {full['med']:g}", ha="center", fontsize=10)
    ax.set_ylim(0.5, 1.55)
    hide_y(ax)
    ax.set_xlim(-5, 122)
    ax.grid(axis="x")
    ax.set_xlabel("newspaper ad budget (thousands of dollars)")
    ax.set_title("All 200 rows: the upper whisker stops at the fence")

    rng = np.random.default_rng(42)
    samples = [rng.normal(50, 10, size=2000), rng.uniform(20, 80, size=2000), rng.exponential(15, size=2000)]
    ax = fig.add_subplot(gs[1, 1])
    ax.boxplot(samples, tick_labels=["normal", "uniform", "exponential"], patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=INK), medianprops=dict(color=ORANGE, linewidth=2.5),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED, markeredgecolor="none"))
    ax.set_ylabel("value")
    ax.grid(axis="y")
    ax.set_title("Shape from the box: symmetric vs. right-skewed")

    rng = np.random.default_rng(42)
    bimodal = np.concatenate([rng.normal(30, 4, size=1000), rng.normal(70, 4, size=1000)])
    unimodal = rng.normal(50, 16.5, size=2000)
    rows = gs[1, 2].subgridspec(2, 1, hspace=0.15)
    for r, (data, name) in enumerate([(bimodal, "bimodal"), (unimodal, "unimodal")]):
        ax = fig.add_subplot(rows[r])
        heights, _, _ = ax.hist(data, bins=50, range=(0, 100), density=True, color=BLUE_LIGHT)
        top = heights.max()
        ax.boxplot(data, vert=False, positions=[top * 1.55], widths=top * 0.35, manage_ticks=False,
                   patch_artist=True, boxprops=dict(facecolor=SURFACE, edgecolor=INK),
                   medianprops=dict(color=ORANGE, linewidth=2.5), showfliers=False)
        ax.set_ylim(0, top * 1.85)
        ax.set_xlim(0, 100)
        hide_y(ax)
        ax.text(1, top * 0.9, name, fontsize=10.5, weight="bold", color=INK2)
        if r == 0:
            ax.set_title("A box plot cannot see the two clusters")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("value")
    save(fig, name="13_box_plots.png")


# ================================================================ notebook 14
def mixture_sample(rng, n):
    which = rng.random(n) < 0.6
    return np.where(which, rng.normal(3, 1, n), rng.normal(7, 1.2, n))


def sheet_14_kde():
    n = len(NEWSPAPER)
    wide = np.linspace(-150, 250, 8001)
    areas = [trapezoid(kde(wide, NEWSPAPER, s), wide) for s in (2, 5, 10, 25)]
    assert np.allclose(areas, 1, atol=1e-4)

    fig = new_figure("Kernel density estimation: a little Gaussian on every point, averaged",
                     "f̂(x) = (1/n) Σ φ(x; xᵢ, σ).  Area 1 by construction; the bandwidth σ trades detail against smoothness.")
    gs = grid(fig, top=0.84, wspace=0.3)
    x = np.linspace(-20, 110, 1000)

    ax = fig.add_subplot(gs[0, 0])
    for xi in NEWSPAPER:
        ax.plot(x, normal_pdf(x, xi, 6) / n, color=BLUE, linewidth=1.2, alpha=0.8)
    ax.plot(NEWSPAPER, np.zeros(n), "o", color=INK, markersize=6, clip_on=False, zorder=3)
    ax.set_ylim(0, 0.036)
    ax.set_xlabel("newspaper budget (thousands of dollars)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.set_title("Step 1: a mountain of area 1/n on each point")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(NEWSPAPER, bins=8, density=True, color=BLUE_LIGHT, edgecolor=SURFACE, label="histogram (area 1, blocky)")
    estimate = kde(x, NEWSPAPER, 6)
    ax.plot(x, estimate, color=ORANGE, linewidth=2.5, label="KDE, σ = 6 (area 1, smooth)")
    ax.fill_between(x, estimate, color=ORANGE, alpha=0.12)
    ax.plot(NEWSPAPER, np.zeros(n), "o", color=INK, markersize=6, clip_on=False, zorder=3)
    ax.set_ylim(0, 0.036)
    ax.set_xlabel("newspaper budget (thousands of dollars)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Step 2: sum them; compare with a histogram")

    s2 = NEWSPAPER.var()
    panel(fig.add_subplot(gs[0, 2]), "Why it is a genuine density", [
        r"$\hat f(x) = \frac{1}{n} \sum \varphi(x;\ x_i, \sigma)$",
        "",
        "with φ the normal PDF, mean xᵢ, std σ.",
        "",
        "Each Gaussian has area 1, so their average",
        "has area 1, and it is never negative.",
        "`sigma:  2       5       10      25`",
        "`area:   " + "  ".join(f"{a:.4f}" for a in areas) + "`",
        "",
        "It is a mixture of n Gaussians: its mean is",
        f"the sample mean ({NEWSPAPER.mean():.2f}), its variance",
        "s² + σ²: smearing adds σ² of spread.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    xb = np.linspace(-30, 120, 1000)
    for s, color in [(1.5, AQUA), (6, ORANGE), (25, BLUE)]:
        ax.plot(xb, kde(xb, NEWSPAPER, s), color=color, linewidth=2, label=f"σ = {s:g}")
    ax.plot(NEWSPAPER, np.zeros(n), "|", color=INK, markersize=14, clip_on=False)
    ax.set_ylim(0, 0.065)
    ax.set_xlabel("newspaper budget (thousands of dollars)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10, title="bandwidth", title_fontsize=10)
    ax.grid(True)
    ax.set_title("Small σ memorizes, large σ blurs")

    rng = np.random.default_rng(42)
    true_x = np.linspace(-1, 11, 500)
    true_pdf = 0.6 * normal_pdf(true_x, 3, 1) + 0.4 * normal_pdf(true_x, 7, 1.2)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(true_x, true_pdf, color=INK, linewidth=2.5, label="true PDF")
    errors = []
    for size, color in [(12, AQUA), (100, BLUE), (10_000, ORANGE)]:
        data = mixture_sample(rng, size)
        bandwidth = 1.06 * data.std() * size ** (-1 / 5)
        est = kde(true_x, data, bandwidth)
        errors.append((size, bandwidth, np.max(np.abs(est - true_pdf))))
        ax.plot(true_x, est, color=color, linewidth=1.8, label=f"KDE, n = {size:,}")
    ax.set_ylim(0, 0.38)
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("More data: the mountains melt into the truth")

    panel(fig.add_subplot(gs[1, 2]), "Choosing σ, and what data buys", [
        "Silverman's rule of thumb:",
        r"$\sigma \approx 1.06\, s\, n^{-1/5}$",
        "",
        "`      n   bandwidth   max |KDE - true|`",
        *[f"`{size:>7,}   {bw:>9.3f}   {err:>16.4f}`" for size, bw, err in errors],
        "",
        "Twelve points give a rough estimate;",
        "thousands give a smooth curve close",
        "to the real PDF.",
    ], spacing=0.08)
    save(fig, name="14_kernel_density_estimation.png")


# ================================================================ notebook 15
def style_violin(parts, color=BLUE_LIGHT, edge=BLUE):
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(edge)
        body.set_alpha(1)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(INK)


def sheet_15_violin_plots():
    q1, q2, q3 = quartiles(NEWSPAPER)
    mean = NEWSPAPER.mean()
    assert np.allclose([q1, q2, q3], [18.35, 27.8, 52.95])

    fig = new_figure("Violin plots: a mirrored KDE with the box-plot statistics inside",
                     "Body width = estimated density (the shape); inner marks = median, quartiles and whiskers (the summary).")
    gs = grid(fig, top=0.84, wspace=0.3)

    rng = np.random.default_rng(42)
    ax = fig.add_subplot(gs[0, 0])
    style_violin(ax.violinplot(NEWSPAPER, positions=[0], showextrema=False, widths=0.9))
    ax.vlines(0, NEWSPAPER.min(), NEWSPAPER.max(), color=INK, linewidth=1.2)
    ax.vlines(0, q1, q3, color=INK, linewidth=9)
    ax.plot(0, q2, "o", color="white", markersize=8, zorder=3)
    ax.plot(0, mean, "D", color=ORANGE, markersize=7, zorder=3)
    ax.plot(rng.uniform(-0.06, 0.06, size=12) + 0.42, NEWSPAPER, "o", color=AQUA, markersize=6)
    for y, text in [(q3, f"Q3 = {q3:g}"), (q1, f"Q1 = {q1:g}"), (q2, f"median = {q2:g}"),
                    (mean, f"mean = {mean:.1f}")]:
        ax.text(-0.12, y, text, ha="right", va="center", fontsize=10,
                color=ORANGE if text.startswith("mean") else INK)
    ax.text(0.42, 80, "data", ha="center", fontsize=10, color=AQUA, weight="bold")
    ax.annotate("KDE outline,\nmirrored", (-0.2, 62), xytext=(-0.5, 78), ha="center", fontsize=10,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xlim(-0.75, 0.65)
    ax.set_ylim(0, 90)
    ax.set_xticks([])
    ax.set_ylabel("budget (thousands of dollars)")
    ax.grid(axis="y")
    ax.set_title("Anatomy: the 12 newspaper budgets")

    tv = np.array([7.8, 66.1, 44.5, 100.4, 151.5, 180.8, 199.8, 214.7, 218.4, 228.3, 240.1, 293.6])
    radio = np.array([2.6, 5.7, 8.6, 10.8, 16.9, 24.6, 26.7, 33.5, 36.9, 39.6, 41.3, 45.9])
    ax = fig.add_subplot(gs[0, 1])
    parts = ax.violinplot([NEWSPAPER, tv, radio], showmedians=True)
    style_violin(parts)
    for body, color in zip(parts["bodies"], [BLUE_LIGHT, "#f6c3ad", "#a9e2cc"]):
        body.set_facecolor(color)
    ax.set_xticks([1, 2, 3], ["newspaper", "TV", "radio"])
    ax.set_ylabel("budget (thousands of dollars)")
    ax.grid(axis="y")
    ax.set_title("Comparing groups: center, spread and shape")

    panel(fig.add_subplot(gs[0, 2]), "How to read a violin", [
        "1. Compute the KDE of the data.",
        "2. Draw it sideways and mirror it: the",
        "   width at each height is the density.",
        "3. Mark median, quartiles and whiskers inside.",
        "",
        "Fat regions: the data concentrates there.",
        "Two bulges: two clusters (a box plot",
        "would hide them).",
        "Bulge low, long thin neck up: right skew.",
        "",
        "Inherits the KDE's bandwidth choice.",
    ], spacing=0.08)

    rng = np.random.default_rng(42)
    bimodal = np.concatenate([rng.normal(30, 4, size=500), rng.normal(70, 4, size=500)])
    sigma_b = 1.06 * bimodal.std() * len(bimodal) ** (-1 / 5)
    grid_y = np.linspace(5, 95, 500)
    trio = gs[1, :2].subgridspec(1, 3, wspace=0.3)
    ax = fig.add_subplot(trio[0])
    ax.boxplot(bimodal, widths=0.5, patch_artist=True, boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=INK),
               medianprops=dict(color=ORANGE, linewidth=2.5))
    ax.set_xticks([])
    ax.set_ylabel("value")
    ax.set_title("box plot: clusters hidden", fontsize=12)
    ax = fig.add_subplot(trio[1])
    ax.plot(kde(grid_y, bimodal, sigma_b), grid_y, color=ORANGE, linewidth=2)
    ax.set_xlabel("density")
    ax.set_title("KDE: shape, no statistics", fontsize=12)
    ax = fig.add_subplot(trio[2])
    style_violin(ax.violinplot(bimodal, showmedians=True, widths=0.8))
    ax.set_xticks([])
    ax.set_title("violin: both", fontsize=12)
    for ax in fig.axes[-3:]:
        ax.set_ylim(5, 95)
        ax.grid(axis="y")
    fig.text(0.355, 0.455, "One bimodal dataset (two clusters at 30 and 70), three plots", ha="center",
             fontsize=13, weight="bold")

    ax = fig.add_subplot(gs[1, 2])
    data = np.sort(NEWSPAPER)
    grid_v = np.linspace(-10, 100, 500)
    for pos, s in enumerate([2, 6, 20]):
        density = kde(grid_v, data, s)
        half = density / density.max() * 0.38
        ax.fill_betweenx(grid_v, pos - half, pos + half, color=BLUE_LIGHT, edgecolor=BLUE)
        ax.plot(np.full(12, pos), data, "o", color=INK, markersize=3.5)
    ax.set_xticks([0, 1, 2], ["σ = 2", "σ = 6", "σ = 20"])
    ax.set_ylabel("budget (thousands of dollars)")
    ax.grid(axis="y")
    ax.set_title("Same 12 points, three bandwidths")
    save(fig, name="15_violin_plots.png")


# ================================================================ notebook 16
def qq_points(data):
    z = (data - data.mean()) / data.std()
    p = (np.arange(1, len(data) + 1) - 0.5) / len(data)
    return normal_quantile(p), np.sort(z)


def qq_axes(ax, data, title, lim, markersize=4, color=BLUE):
    tq, sq = qq_points(data)
    ax.plot([-lim, lim], [-lim, lim], color=ORANGE, linewidth=2, zorder=1)
    ax.plot(tq, sq, "o", color=color, markersize=markersize, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.grid(True)
    ax.set_title(title, fontsize=12)
    return tq, sq


def sheet_16_qq_plots():
    z = (NEWSPAPER - NEWSPAPER.mean()) / NEWSPAPER.std()
    skew = np.mean(z ** 3)
    assert skew > 0.5

    fig = new_figure("QQ plots: compare your data with a normal, quantile by quantile",
                     "Standardize, sort, and plot against standard-normal quantiles at (i − 0.5)/n. Normal data hugs y = x; the bend says what is wrong.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    tq, sq = qq_axes(ax, NEWSPAPER, "Newspaper budgets (n = 12): a bowl", 2.6, markersize=8)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("theoretical quantiles (standard normal)")
    ax.set_ylabel("sample quantiles (standardized)")
    inset = ax.inset_axes([0.62, 0.08, 0.35, 0.3])
    inset.hist(NEWSPAPER, bins=6, color=BLUE_LIGHT, edgecolor=SURFACE)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("histogram", fontsize=9, pad=3)

    rng = np.random.default_rng(42)
    sales_like = rng.normal(14, 5, size=200)
    ax = fig.add_subplot(gs[0, 1])
    qq_axes(ax, sales_like, "Sales-like N(14, 5²), n = 200: on the line", 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("theoretical quantiles (standard normal)")
    ax.set_ylabel("sample quantiles (standardized)")
    inset = ax.inset_axes([0.62, 0.08, 0.35, 0.3])
    inset.hist(sales_like, bins=20, color=BLUE_LIGHT, edgecolor=SURFACE)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("histogram", fontsize=9, pad=3)

    panel(fig.add_subplot(gs[0, 2]), "The recipe", [
        r"1. Standardize: $z_i = (x_i - \bar x) / s$",
        "2. Sort: the i-th value is the sample's",
        "   (i − 0.5)/n quantile.",
        "3. Take the standard normal's quantiles",
        "   at the same levels.",
        "4. Scatter theory (x) against sample (y).",
        "",
        "Newspaper: the largest budgets rise above",
        "the line, the small ones bunch up:",
        f"right skew (skewness {skew:.2f}), not normal.",
    ], spacing=0.08)

    rng = np.random.default_rng(42)
    n = 500
    guide = [("normal: on the line", rng.normal(0, 1, n)),
             ("right-skewed: bowl", rng.exponential(1, n)),
             ("heavy tails: S through the line", rng.standard_t(3, n)),
             ("light tails: the other S", rng.uniform(0, 1, n))]
    row = gs[1, :].subgridspec(1, 4, wspace=0.25)
    for k, (name, data) in enumerate(guide):
        ax = fig.add_subplot(row[k])
        qq_axes(ax, data, name, 3.4, markersize=3)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel("theoretical quantiles")
        if k == 0:
            ax.set_ylabel("sample quantiles")
    fig.text(0.5, 0.455, "A field guide (n = 500 each): every departure from normality has a signature",
             ha="center", fontsize=13, weight="bold")
    save(fig, name="16_qq_plots.png")


if __name__ == "__main__":
    sheet_01_mean()
    sheet_02_median_mode()
    sheet_03_expected_value_of_a_function()
    sheet_04_sum_of_expectations()
    sheet_05_variance()
    sheet_06_standard_deviation()
    sheet_07_sum_of_gaussians()
    sheet_08_standardizing()
    sheet_09_moments()
    sheet_10_skewness()
    sheet_11_kurtosis()
    sheet_12_quantiles()
    sheet_13_box_plots()
    sheet_14_kde()
    sheet_15_violin_plots()
    sheet_16_qq_plots()
