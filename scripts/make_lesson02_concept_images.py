#!/usr/bin/env python3
"""Concept sheets for lesson 02 (describing distributions).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data and parameters the notebook uses.
Images are written to coursera/02_discribing_distribution/img/.

Run from anywhere:  python scripts/make_lesson02_concept_images.py
"""
from functools import partial
from math import comb, erf, sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, frac_label, bars, grouped_bars, panel)
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


if __name__ == "__main__":
    sheet_01_mean()
    sheet_02_median_mode()
    sheet_05_variance()
    sheet_06_standard_deviation()
    sheet_07_sum_of_gaussians()
    sheet_08_standardizing()
    sheet_10_skewness()
    sheet_11_kurtosis()
