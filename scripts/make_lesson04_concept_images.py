#!/usr/bin/env python3
"""Concept sheets for lesson 04 (sampling and point estimation).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data, parameters and random seeds the notebook uses.
Images are written to coursera/04_sampling_and_point_estimation/img/.

Run from anywhere:  python scripts/make_lesson04_concept_images.py
"""
from functools import partial
from math import comb, erf, sqrt

import numpy as np
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, frac_label, heat, bars, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "04_sampling_and_point_estimation"
save = partial(_save, folder=LESSON / "img")


# ================================================================ helpers
def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def binomial_pmf(n, p):
    k = np.arange(n + 1)
    return k, np.array([comb(n, i) * p ** i * (1 - p) ** (n - i) for i in k])


def step_hist(ax, data, bins, color, label, fill=False):
    ax.hist(data, bins=bins, density=True, histtype="stepfilled" if fill else "step",
            color=color if fill else None, edgecolor=color, linewidth=1.8, alpha=0.35 if fill else 1,
            label=label)


def mean_line(ax, x, label, color=ORANGE, style="--"):
    ax.axvline(x, color=color, linestyle=style, linewidth=2, label=label, zorder=3)


# ================================================================ notebook 01
def sheet_01_population_sample():
    rng = np.random.default_rng(42)
    population = rng.normal(166, 9, size=10_000)
    rng = np.random.default_rng(7)
    sample = rng.choice(population, size=100, replace=True)
    ten = np.array([150, 155, 158, 160, 163, 167, 170, 174, 178, 185])
    sorted_first4 = np.sort(ten)[:4]
    rng = np.random.default_rng(3)
    random_means = np.array([rng.choice(ten, size=4, replace=True).mean() for _ in range(10_000)])

    fig = new_figure("Population and sample: estimate the whole from a random part",
                     "Population: everyone you care about (N).  Sample: the few you actually measure (n), "
                     "chosen at random, independently, by the same rule.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(population, bins=50, density=True, color=BLUE_LIGHT, label="population, N = 10,000")
    ax.plot(sample, np.full(sample.size, -0.002), "|", color=BLUE, markersize=12, label="sample, n = 100")
    mean_line(ax, population.mean(), f"population mean {population.mean():.2f}", INK2, "-")
    mean_line(ax, sample.mean(), f"sample mean {sample.mean():.2f}")
    ax.set_ylim(-0.005, 0.07)
    ax.set_xlabel("height (cm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.02))
    ax.grid(axis="y")
    ax.set_title("Statistopia: 100 people stand in for 10,000")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(random_means, bins=np.arange(148, 188, 1.25), density=True, color=BLUE_LIGHT,
            label="random samples of 4 (10,000 repeats)")
    mean_line(ax, ten.mean(), f"true mean {ten.mean():.0f}", INK2, "-")
    mean_line(ax, sorted_first4.mean(), f"first 4 of sorted line: {sorted_first4.mean():.2f}", ORANGE, "-")
    ax.set_ylim(0, 0.115)
    ax.set_xlabel("sample mean of 4 islanders (cm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y")
    ax.set_title("Random samples center on the truth; sorted ones don't")

    panel(fig.add_subplot(gs[0, 2]), "What makes a good sample", [
        "Random: every member has the same chance.",
        "The sorted pick (150, 155, 158, 160) is",
        f"biased low: {sorted_first4.mean():.2f} vs the true 166.",
        "",
        "Independent: sample with replacement, so",
        "one draw never changes the next one's odds.",
        "",
        "Identically distributed: the same rule for",
        "every draw (don't sample only one district).",
    ], spacing=0.09)

    bars(fig.add_subplot(gs[1, 0]), ["without\nreplacement", "with\nreplacement"], [1 / 4, 1 / 5],
         "P(2nd draw is the giant | 1st was not)", "", ylim=0.35, label_fmt=lambda v: frac_label(v),
         highlight=0)

    rng = np.random.default_rng(5)
    ax = fig.add_subplot(gs[1, 1])
    for town, shift, color in [("only Southburg", -6, AQUA), ("whole island", 0, BLUE), ("only Northville", 6, ORANGE)]:
        means = rng.choice(population + shift, size=(2000, 100), replace=True).mean(axis=1)
        step_hist(ax, means, 40, color, town, fill=True)
    mean_line(ax, population.mean(), "population mean", INK2, "-")
    ax.set_ylim(0, 0.68)
    ax.set_xlabel("sample mean of 100 people (cm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.grid(axis="y")
    ax.set_title("Sampling one district gives a biased estimate")

    panel(fig.add_subplot(gs[1, 2]), "Takeaways", [
        "A random sample's mean errs, but in no",
        "preferred direction; bigger samples err less.",
        "",
        "A biased rule errs the same way every time,",
        "and no sample size fixes that.",
        "",
        "Machine learning: the training data is a",
        "sample. A cat classifier trained only on",
        "orange cats has not seen the population.",
    ], spacing=0.09)
    save(fig, name="01_population_and_sample.png")


# ================================================================ notebook 02
def sheet_02_sample_mean():
    population = np.array([151, 153, 155, 157, 159, 161, 163, 165, 167, 169])
    mu, sigma = population.mean(), population.std()
    samples = {"x̄₁ random, n = 6": np.array([153, 155, 159, 163, 165, 169]),
               "x̄₂ six shortest": np.sort(population)[:6],
               "x̄₃ random, n = 2": np.array([155, 161])}
    assert mu == 160

    fig = new_figure("The sample mean x̄ estimates the population mean μ",
                     "Statistopia: 10 heights from 151 to 169 cm, μ = 160. Two morals: randomness matters, and size matters.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(population, np.zeros(10), "o", color=BLUE_LIGHT, markersize=11, markeredgecolor=BLUE)
    ax.axvline(mu, color=INK2, linewidth=2)
    ax.text(mu + 0.3, -0.22, "μ = 160", ha="left", fontsize=11, weight="bold")
    for y, (name, s), color in zip([1.2, 0.85, 0.5], samples.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(s.mean(), y, "v", color=color, markersize=12)
        right = s.mean() > mu
        ax.text(s.mean() + (0.5 if right else -0.5), y, f"{name}: {s.mean():.2f}", ha="left" if right else "right",
                va="center", fontsize=9.5, color=color, weight="bold")
    ax.set_ylim(-0.35, 1.45)
    ax.set_yticks([])
    ax.set_xlim(149, 171)
    ax.set_xlabel("height (cm)")
    ax.set_title("Three estimates of the same μ")

    rng = np.random.default_rng(0)
    ax = fig.add_subplot(gs[0, 1])
    for n, color in [(2, AQUA), (6, BLUE)]:
        means = population[rng.integers(0, 10, size=(100_000, n))].mean(axis=1)
        step_hist(ax, means, np.arange(150, 171, 1), color, f"random, n = {n}: std {means.std():.2f}", fill=True)
    mean_line(ax, mu, "μ = 160", INK2, "-")
    mean_line(ax, np.sort(population)[:6].mean(), "six shortest: always 156", ORANGE, "--")
    ax.set_ylim(0, 0.24)
    ax.set_xlabel("sample mean (cm)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y")
    ax.set_title("Random x̄ centers on μ; bigger n is tighter")

    panel(fig.add_subplot(gs[0, 2]), "Two morals", [
        "Randomness: the six shortest people always",
        "give 156, off by 4 every single time.",
        "",
        "Size: one unlucky person moves a mean of 2",
        "a lot and a mean of 6 only a little.",
        "",
        r"Spread of x̄:  $\sigma / \sqrt{n}$",
        f"(here σ = {sigma:.2f}, so n = 6 gives {sigma / sqrt(6):.2f})",
    ], spacing=0.095)

    rng = np.random.default_rng(1)
    ns = np.array([1, 2, 6, 10, 30, 100])
    sim = [population[rng.integers(0, 10, size=(50_000, n))].mean(axis=1).std() for n in ns]
    ax = fig.add_subplot(gs[1, 0])
    grid_n = np.linspace(1, 100, 300)
    ax.plot(grid_n, sigma / np.sqrt(grid_n), color=BLUE, linewidth=2, label="σ / √n")
    ax.plot(ns, sim, "o", color=ORANGE, markersize=8, label="simulated std of x̄")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n (log scale)")
    ax.set_ylabel("std of x̄ (cm)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Size buys accuracy: error shrinks like 1/√n")

    rng = np.random.default_rng(4)
    draws = population[rng.integers(0, 10, size=300)]
    running = np.cumsum(draws) / np.arange(1, 301)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(1, 301), running, color=BLUE, linewidth=1.8)
    ax.axhline(mu, color=INK2, linewidth=1.5)
    ax.fill_between(np.arange(1, 301), mu - 2 * sigma / np.sqrt(np.arange(1, 301)),
                    mu + 2 * sigma / np.sqrt(np.arange(1, 301)), color=BLUE_LIGHT, alpha=0.5,
                    label="μ ± 2σ/√n")
    ax.set_ylim(150, 170)
    ax.set_xlabel("people sampled so far")
    ax.set_ylabel("running sample mean (cm)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Watching an estimate improve")

    panel(fig.add_subplot(gs[1, 2]), "Proportions work the same way", [
        "4 of the 10 islanders are taller than",
        "162 cm: p = 0.4.",
        "",
        "Sample {153, 155, 159, 163, 165, 169}:",
        "3 of 6 qualify, so p̂ = 0.5.",
        "",
        "A proportion is the mean of 0/1 values,",
        "so everything above carries over",
        "(next notebook).",
    ], spacing=0.09)
    save(fig, name="02_sample_mean.png")


# ================================================================ notebook 03
def sheet_03_sample_proportion():
    vehicles = np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    p = vehicles.mean()
    se_poll = sqrt(0.52 * 0.48 / 1000)

    fig = new_figure("The sample proportion p̂: the mean of a 0/1 variable",
                     "Statistopia: 4 of 10 islanders ride bicycles, p = 0.4. A sample of 6 gives p̂ on a ladder of 0, 1/6, …, 1.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for i, v in enumerate(vehicles):
        r, c = divmod(i, 5)
        ax.add_patch(Rectangle((c, 1 - r), 0.88, 0.88, facecolor=ORANGE if v else BLUE_LIGHT, linewidth=0))
        ax.text(c + 0.44, 1 - r + 0.44, "bike\n1" if v else "car\n0", ha="center", va="center", fontsize=12,
                color="white" if v else INK)
    ax.set_xlim(-0.1, 5)
    ax.set_ylim(-0.6, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(2.45, -0.35, "p = mean of the 0/1 values = 4/10 = 0.4", ha="center", fontsize=12, weight="bold")
    ax.set_title("The population: 10 islanders")

    ax = fig.add_subplot(gs[0, 1])
    k, pmf = binomial_pmf(6, p)
    ax.bar(k / 6, pmf, width=0.05, color=BLUE, zorder=2)
    for x, y in zip(k / 6, pmf):
        ax.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=9)
    mean_line(ax, p, "p = 0.4 (not a rung!)")
    ax.set_xticks(k / 6, [frac_label(v / 6, 6) for v in k])
    ax.set_xlabel("p̂ with n = 6")
    ax.set_ylabel("probability")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Sampling distribution of p̂: 6 × p̂ ~ Binomial(6, 0.4)")

    panel(fig.add_subplot(gs[0, 2]), "Why it behaves like x̄", [
        "Code bike = 1, car = 0. Then the sample",
        "proportion is the sample mean:",
        "p̂ = (number of bikes) / n.",
        "",
        "Unbiased:  E[p̂] = p",
        r"Spread:  $\mathrm{std}(\hat{p}) = \sqrt{p(1-p)/n}$",
        "",
        "The video's sample: 2 bikes in 6,",
        "p̂ = 33.3%, off by 6.7 points.",
    ], spacing=0.09)

    ladders = gs[1, 0].subgridspec(3, 1, hspace=0.35)
    for row, (n, width) in enumerate([(6, 0.03), (20, 0.02), (100, 0.006)]):
        k, pmf = binomial_pmf(n, p)
        ax = fig.add_subplot(ladders[row])
        ax.bar(k / n, pmf, width=width, color=BLUE, zorder=2)
        ax.axvline(p, color=ORANGE, linestyle="--", linewidth=1.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_yticks([])
        ax.text(0.99, 0.75, f"n = {n}: std {sqrt(p * (1 - p) / n):.3f}", transform=ax.transAxes, ha="right",
                fontsize=10)
        if row == 0:
            ax.set_title("Bigger n: a finer ladder, a tighter spread")
        if row < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("p̂  (dashed: p = 0.4)")

    ax = fig.add_subplot(gs[1, 1])
    ps = np.linspace(0, 1, 300)
    ax.plot(ps, np.sqrt(ps * (1 - ps) / 100), color=BLUE, linewidth=2.2)
    ax.plot(0.5, sqrt(0.25 / 100), "o", color=ORANGE, markersize=9)
    ax.annotate("hardest at p = 0.5", (0.5, 0.05), textcoords="offset points", xytext=(0, 10), ha="center",
                fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_ylim(0, 0.062)
    ax.set_xlabel("true proportion p")
    ax.set_ylabel("std of p̂ (n = 100)")
    ax.grid(True)
    ax.set_title("Which proportions are hardest to estimate?")

    panel(fig.add_subplot(gs[1, 2]), "An election poll", [
        "True support p = 0.52, poll of n = 1,000:",
        r"$\sqrt{0.52 \times 0.48 / 1000}$" + f" ≈ {se_poll:.3f}",
        "",
        "p̂ lands within about two standard errors,",
        f"±{2 * se_poll:.1%}, 95% of the time.",
        "",
        "That is the \"±3% margin of error\" in",
        "news polls.",
    ], spacing=0.095)
    save(fig, name="03_sample_proportion.png")


# ================================================================ notebook 04
def sheet_04_sample_variance():
    values = [1, 2, 3]
    pairs = [(a, b) for a in values for b in values]
    naive = [((a - (a + b) / 2) ** 2 + (b - (a + b) / 2) ** 2) / 2 for a, b in pairs]
    s2 = [2 * v for v in naive]
    sigma2 = np.var(values)
    assert np.isclose(np.mean(naive), 1 / 3) and np.isclose(np.mean(s2), 2 / 3)

    fig = new_figure("The sample variance divides by n − 1",
                     "Measuring deviations from x̄ instead of μ makes them too small. Dividing by n − 1 instead of n fixes the bias.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(9)
    ax.bar(x - 0.2, naive, width=0.38, color=BLUE_LIGHT, label="naive ÷ n", zorder=2)
    ax.bar(x + 0.2, s2, width=0.38, color=BLUE, label="s² ÷ (n − 1)", zorder=2)
    ax.axhline(sigma2, color=ORANGE, linewidth=2, label="σ² = 2/3")
    ax.axhline(np.mean(naive), color=INK2, linestyle=":", linewidth=1.8, label="average naive = 1/3")
    ax.set_xticks(x, [f"({a},{b})" for a, b in pairs], fontsize=9)
    ax.set_ylim(0, 2.9)
    ax.set_xlabel("all 9 samples of two draws from the hat {1, 2, 3}")
    ax.set_ylabel("variance estimate")
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    ax.grid(axis="y", zorder=0)
    ax.set_title("The hat game: average s² = 2/3, average naive = 1/3")

    rng = np.random.default_rng(0)
    population = rng.normal(160, 8, size=100_000)
    pop_var = population.var()
    ns = [2, 3, 5, 10, 30]
    ax = fig.add_subplot(gs[0, 1])
    sim_naive, sim_s2 = [], []
    for n in ns:
        samples = population[rng.integers(0, len(population), size=(40_000, n))]
        sim_naive.append(samples.var(axis=1, ddof=0).mean())
        sim_s2.append(samples.var(axis=1, ddof=1).mean())
    grid_n = np.linspace(2, 30, 200)
    ax.plot(grid_n, pop_var * (grid_n - 1) / grid_n, color=BLUE_LIGHT, linewidth=2, label="σ²(n − 1)/n")
    ax.plot(ns, sim_naive, "o", color=BLUE_LIGHT, markeredgecolor=BLUE, markersize=8, label="average naive (sim)")
    ax.plot(ns, sim_s2, "s", color=BLUE, markersize=8, label="average s² (sim)")
    ax.axhline(pop_var, color=ORANGE, linewidth=2, label=f"σ² = {pop_var:.1f}")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("average estimate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True)
    ax.set_title("The bias is the factor (n − 1)/n")

    panel(fig.add_subplot(gs[0, 2]), "The two estimators", [
        r"naive:  $\frac{1}{n}\sum (x_i - \bar{x})^2$   (too small)",
        r"$s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$   (unbiased)",
        "",
        "Sample (1, 3): x̄ = 2, naive 1, s² = 2.",
        "Sample (2, 2): both 0. Identical values",
        "cannot show any spread.",
        "",
        "numpy: np.var(x) is naive,",
        "np.var(x, ddof=1) is s².",
    ], spacing=0.09)

    rng = np.random.default_rng(1)
    mu = population.mean()
    sample = rng.choice(population, size=5, replace=True)
    centers = np.linspace(mu - 12, mu + 12, 400)
    ssd = [np.sum((sample - c) ** 2) for c in centers]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(centers, ssd, color=BLUE, linewidth=2)
    mean_line(ax, sample.mean(), f"x̄ = {sample.mean():.1f}: the minimum", ORANGE, "--")
    mean_line(ax, mu, f"μ = {mu:.1f}", INK2, ":")
    ax.set_xlabel("center c")
    ax.set_ylabel("Σ (xᵢ − c)²")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Why: deviations from x̄ are as small as possible")

    rng = np.random.default_rng(2)
    samples = population[rng.integers(0, len(population), size=(40_000, 5))]
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 250, 60)
    step_hist(ax, samples.var(axis=1, ddof=0), bins, BLUE_LIGHT, "naive (n = 5)", fill=True)
    step_hist(ax, samples.var(axis=1, ddof=1), bins, BLUE, "s² (n = 5)")
    mean_line(ax, pop_var, f"σ² = {pop_var:.1f}", ORANGE, "-")
    ax.set_xlabel("variance estimate")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y")
    ax.set_title("The two estimators' distributions (n = 5)")

    panel(fig.add_subplot(gs[1, 2]), "When the divisor matters", [
        "Huddled heights A (mean 160):",
        "naive 1.7,  s² = 2.125",
        "Spread heights B (mean 160):",
        "naive 50.8,  s² = 63.5",
        "",
        "n = 3, Σ = 12: naive 4 vs s² = 6 (50% apart).",
        "n = 1000: 0.1% apart.",
        "Essential in principle, invisible at scale.",
    ], spacing=0.09)
    save(fig, name="04_sample_variance.png")


# ================================================================ notebook 05
def sheet_05_law_of_large_numbers():
    faces = np.array([1, 2, 3, 4])
    pair_avg = np.array([[(a + b) / 2 for b in faces] for a in faces])
    assert pair_avg.mean() == 2.5

    fig = new_figure("The law of large numbers: sample averages converge to the expected value",
                     "Throw a four-sided die twice and average. Draw such pair-averages at random: their running mean heads to μ = 2.5.")
    gs = grid(fig, top=0.8, wspace=0.3)

    heat(fig.add_subplot(gs[0, 0]), pair_avg, faces, faces, "All 16 pair-averages: mean 2.5", "throw 1",
         "throw 2", labels=[[f"{v:.1f}" for v in row] for row in pair_avg], vmax=4.6)

    rng = np.random.default_rng(10)
    draws = rng.choice(faces, size=(50, 2)).mean(axis=1)
    running = np.cumsum(draws) / np.arange(1, 51)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(1, 51), draws, "o", color=BLUE_LIGHT, markersize=6, label="each pair-average")
    ax.plot(np.arange(1, 51), running, color=BLUE, linewidth=2.2, label="running mean")
    ax.axhline(2.5, color=ORANGE, linestyle="--", linewidth=1.8, label="μ = 2.5")
    ax.set_ylim(0.8, 4.2)
    ax.set_xlabel("draws")
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True)
    ax.set_title(f"The video's experiment: {running[-1]:.2f} after 50 draws")

    panel(fig.add_subplot(gs[0, 2]), "The law", [
        "For i.i.d. draws with mean μ:",
        r"$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \;\to\; \mu$   as n grows",
        "",
        "",
        "",
        "Conditions:",
        "1. random sampling (no biased rule)",
        "2. independent draws",
        "3. a finite expected value",
        "",
        "Each new draw moves an n-point average",
        "by at most |x − x̄| / (n + 1).",
    ], spacing=0.085)

    rng = np.random.default_rng(0)
    paths = rng.choice(faces, size=(25, 2000, 2)).mean(axis=2)
    run_paths = np.cumsum(paths, axis=1) / np.arange(1, 2001)
    ax = fig.add_subplot(gs[1, 0])
    for path in run_paths:
        ax.plot(np.arange(1, 2001), path, color=BLUE, alpha=0.35, linewidth=1)
    ax.axhline(2.5, color=ORANGE, linewidth=2)
    ax.set_xscale("log")
    ax.set_ylim(1, 4)
    ax.set_xlabel("draws (log scale)")
    ax.set_ylabel("running mean")
    ax.grid(True)
    ax.set_title("25 running averages, one destination")

    rng = np.random.default_rng(1)
    n = 100_000
    population = rng.normal(160, 8, size=100_000)
    mu = population.mean()
    tall_only = rng.choice(population[population > np.median(population)], size=n)
    cauchy = rng.standard_cauchy(n)
    breakers = gs[1, 1].subgridspec(2, 1, hspace=0.6)
    ax = fig.add_subplot(breakers[0])
    ax.plot(np.cumsum(tall_only) / np.arange(1, n + 1), color=ORANGE, linewidth=1.5)
    ax.axhline(mu, color=INK2, linewidth=1.5)
    ax.text(n, mu + 0.6, f"μ = {mu:.1f}", ha="right", fontsize=9.5, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(155, 172)
    ax.set_ylabel("cm", fontsize=10)
    ax.grid(True)
    ax.set_title("Lawbreakers: biased rule (only the tall half)", fontsize=11.5)
    ax = fig.add_subplot(breakers[1])
    ax.plot(np.cumsum(cauchy) / np.arange(1, n + 1), color=ORANGE, linewidth=1.2)
    ax.axhline(0, color=INK2, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("draws (log scale)")
    ax.grid(True)
    ax.set_title("No mean at all (Cauchy): never settles", fontsize=11.5)

    panel(fig.add_subplot(gs[1, 2]), "Why it matters", [
        "Every \"simulate 100,000 times and take",
        "the average\" in this course relies on it.",
        "",
        "The law is the bridge from probability",
        "(expected values) to statistics (averages).",
        "",
        "Break a condition and it fails: a biased",
        "rule converges to the wrong number, and",
        "a distribution without a mean never settles.",
    ], spacing=0.09)
    save(fig, name="05_law_of_large_numbers.png")


# ================================================================ notebook 06
def sheet_06_clt_discrete():
    fig = new_figure("The central limit theorem (discrete): coin counts become Gaussian",
                     "Heads in n fair flips is Binomial(n, 0.5), a sum of n Bernoullis. As n grows it approaches N(np, np(1 − p)).")
    gs = grid(fig, top=0.84, wspace=0.3)

    seq = gs[0, :2].subgridspec(1, 5, wspace=0.35)
    for i, n in enumerate([1, 2, 3, 4, 10]):
        k, pmf = binomial_pmf(n, 0.5)
        ax = fig.add_subplot(seq[i])
        ax.bar(k, pmf, width=0.7, color=BLUE, zorder=2)
        ax.set_xticks(k if n <= 4 else k[::2])
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", zorder=0)
        ax.set_title(f"n = {n}\nμ = {n * 0.5:g}, σ² = {n * 0.25:g}", fontsize=11)
        ax.set_xlabel("heads", fontsize=10)
        if i == 0:
            ax.set_ylabel("probability")
        else:
            ax.set_yticklabels([])

    panel(fig.add_subplot(gs[0, 2]), "The parameters", [
        "Each flip: mean p, variance p(1 − p).",
        "n independent flips add up:",
        "",
        "μ = np,  σ² = np(1 − p)",
        "",
        "n = 4: PMF (1, 4, 6, 4, 1)/16, mean 2,",
        "variance 1, matching np and np(1 − p).",
    ], spacing=0.1)

    rng = np.random.default_rng(42)
    heads = rng.integers(0, 2, size=(100_000, 100)).sum(axis=1)
    k, pmf = binomial_pmf(100, 0.5)
    within = pmf[(k >= 40) & (k <= 60)].sum()
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(heads, bins=np.arange(25, 76) - 0.5, density=True, color=BLUE_LIGHT, label="100,000 experiments")
    x = np.linspace(25, 75, 400)
    ax.plot(x, normal_pdf(x, 50, 5), color=ORANGE, linewidth=2.2, label="N(50, 25)")
    ax.axvspan(40, 60, color=AQUA, alpha=0.12, label=f"40–60 heads: {within:.1%}")
    ax.set_xlabel("heads in 100 flips")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Flip 100 coins, count, repeat")

    biased = gs[1, 1].subgridspec(1, 2, wspace=0.3)
    for i, n in enumerate([5, 50]):
        k, pmf = binomial_pmf(n, 0.2)
        mu, sd = n * 0.2, sqrt(n * 0.2 * 0.8)
        ax = fig.add_subplot(biased[i])
        ax.bar(k, pmf, width=0.7, color=BLUE, zorder=2)
        x = np.linspace(-1, 5.5 if n == 5 else 22, 300)
        ax.plot(x, normal_pdf(x, mu, sd), color=ORANGE, linewidth=2)
        ax.set_xlim(-1, 5.5 if n == 5 else 22)
        ax.grid(axis="y", zorder=0)
        skew = (1 - 0.4) / sqrt(n * 0.16)
        ax.set_title(f"p = 0.2, n = {n}\nskew {skew:.2f}", fontsize=11)
        ax.set_xlabel("heads", fontsize=10)
        if i == 0:
            ax.set_ylabel("probability")

    panel(fig.add_subplot(gs[1, 2]), "Using the approximation", [
        "n = 100 fair flips: X ≈ N(50, 25), σ = 5.",
        "About 95% of the time: 40 to 60 heads",
        f"(μ ± 2σ). Exact binomial sum: {within:.1%}.",
        "",
        "A biased coin still turns Gaussian, just",
        "more slowly: skewness (1 − 2p)/√(np(1 − p))",
        "shrinks like 1/√n.",
        "",
        "Rule of thumb: np and n(1 − p) ≥ 5–10.",
    ], spacing=0.09)
    save(fig, name="06_clt_discrete.png")


# ================================================================ notebook 07
def sheet_07_clt_continuous():
    a, b = 0, 15
    mu, sigma2 = (a + b) / 2, (b - a) ** 2 / 12
    p_7_8 = normal_cdf(0.5 / sqrt(sigma2 / 30)) - normal_cdf(-0.5 / sqrt(sigma2 / 30))
    assert abs(p_7_8 - 0.47) < 0.01

    fig = new_figure("The central limit theorem (continuous): averages of any shape become a bell",
                     "Waiting time ~ Uniform(0, 15): μ = 7.5, σ² = 18.75. The average of n waits ≈ N(μ, σ²/n).")
    gs = grid(fig, top=0.84, wspace=0.3)

    rng = np.random.default_rng(42)
    gallery = gs[0, :2].subgridspec(1, 5, wspace=0.25)
    x = np.linspace(0, 15, 400)
    for i, n in enumerate([1, 2, 3, 4, 5]):
        y = rng.uniform(a, b, size=(20_000, n)).mean(axis=1)
        ax = fig.add_subplot(gallery[i])
        ax.hist(y, bins=40, density=True, color=BLUE_LIGHT)
        ax.plot(x, normal_pdf(x, mu, sqrt(sigma2 / n)), color=ORANGE, linewidth=2)
        ax.set_ylim(0, 0.24)
        ax.set_xticks([0, 7.5, 15])
        ax.grid(axis="y")
        ax.set_title(f"n = {n}", fontsize=11.5)
        ax.set_xlabel("average wait", fontsize=10)
        if i == 0:
            ax.set_ylabel("density")
        else:
            ax.set_yticklabels([])

    lines = ["`  n    Var(Yₙ) = 18.75/n   std`"]
    for n in [1, 2, 3, 4, 5, 30]:
        lines.append(f"`{n:>3}    {sigma2 / n:>15.4g}   {sqrt(sigma2 / n):.2f}`")
    lines += ["", "Mean stays at 7.5 (linearity); the variance", "shrinks as σ²/n (independence)."]
    panel(fig.add_subplot(gs[0, 2]), "Flat → triangle → bell", lines, spacing=0.08)

    rng = np.random.default_rng(1)
    z = np.linspace(-4, 4, 300)
    ax = fig.add_subplot(gs[1, 0])
    for n, color in [(1, ORANGE), (2, AQUA), (30, BLUE)]:
        y = rng.uniform(a, b, size=(200_000, n)).mean(axis=1)
        step_hist(ax, (y - mu) / sqrt(sigma2 / n), 80, color, f"n = {n}")
    ax.plot(z, normal_pdf(z), color=INK, linewidth=2.2, label="N(0, 1)")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("z = (Yₙ − μ) / (σ/√n)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Standardizing collapses the family onto N(0, 1)")

    rng = np.random.default_rng(3)
    ax = fig.add_subplot(gs[1, 1])
    for n, color in [(5, ORANGE), (30, BLUE)]:
        y = rng.lognormal(1.5, 1.0, size=(100_000, n)).mean(axis=1)
        zz = (y - y.mean()) / y.std()
        step_hist(ax, zz, 120, color, f"n = {n}: skew {np.mean(zz ** 3):+.2f}")
    ax.plot(z, normal_pdf(z), color=INK, linewidth=2.2, label="N(0, 1)")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("standardized average")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("A skewed (lognormal) population needs more n")

    panel(fig.add_subplot(gs[1, 2]), "Using it", [
        "Average 30 calls: Y₃₀ ≈ N(7.5, 0.625).",
        "P(7 < Y₃₀ < 8): z from −0.63 to +0.63,",
        f"so about {p_7_8:.0%}. No 30-dimensional integral.",
        "",
        "Why n = 2 is a triangle: the sum of two",
        "uniforms convolves two rectangles. Each",
        "extra average rounds the corners further.",
        "",
        "Sum form (ΣXᵢ − nμ)/(√n σ): the same theorem.",
    ], spacing=0.085)
    save(fig, name="07_clt_continuous.png")


if __name__ == "__main__":
    sheet_01_population_sample()
    sheet_02_sample_mean()
    sheet_03_sample_proportion()
    sheet_04_sample_variance()
    sheet_05_law_of_large_numbers()
    sheet_06_clt_discrete()
    sheet_07_clt_continuous()
