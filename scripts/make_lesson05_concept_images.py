#!/usr/bin/env python3
"""Concept sheets for lesson 05 (point estimation: MLE, regularization, MAP).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data, parameters and random seeds the notebook uses.
Images are written to coursera/05_point_estimation/img/.

Run from anywhere:  python scripts/make_lesson05_concept_images.py
"""
from functools import partial
from math import log, sqrt

import numpy as np
from matplotlib.ticker import FuncFormatter

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA, SEQ,
                       new_figure, grid, bars, grouped_bars, panel)
from viz_style import save as _save

LESSON = REPO / "coursera" / "05_point_estimation"
save = partial(_save, folder=LESSON / "img")
P = np.linspace(0.001, 0.999, 1999)


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def trapezoid(y, x):
    return (getattr(np, "trapezoid", None) or np.trapz)(y, x)


def peak_scaled(curve):
    return curve / curve.max()


# ================================================================ notebook 01
def sheet_01_intro_mle():
    scenarios = {"movie": 0.7, "board games": 0.3, "nap": 0.05}
    coins = {"A (p = 0.3)": 0.3, "B (p = 0.5)": 0.5, "C (p = 0.7)": 0.7}
    hht = [p ** 2 * (1 - p) for p in coins.values()]
    ttt = [(1 - p) ** 3 for p in coins.values()]
    assert np.argmax(hht) == 2 and np.argmax(ttt) == 0

    fig = new_figure("Maximum likelihood: pick the scenario that makes the evidence most probable",
                     "Choose the explanation θ that maximizes P(evidence | θ). Different evidence, different winner.")
    gs = grid(fig, top=0.84, wspace=0.3)

    bars(fig.add_subplot(gs[0, 0]), list(scenarios), list(scenarios.values()),
         "Popcorn on the floor: P(popcorn | scenario)", "scenario", ylim=0.9,
         label_fmt=lambda v: f"{v:g}", highlight=0)

    grouped_bars(fig.add_subplot(gs[0, 1]), list(coins),
                 [("evidence HHT: p²(1 − p)", hht, BLUE), ("evidence TTT: (1 − p)³", ttt, ORANGE)],
                 "Which coin produced the flips?", "coin", ylim=0.45, label_fmt=lambda v: f"{v:.3f}")

    panel(fig.add_subplot(gs[0, 2]), "The principle", [
        "Evidence: popcorn on the floor.",
        "Scenarios: movie, board games, nap.",
        "MLE picks the movie: 0.7 is the largest",
        "P(popcorn | scenario).",
        "",
        "The three numbers need not sum to 1: they",
        "come from three different distributions,",
        "all evaluated at the same evidence.",
        "",
        "HHT → coin C wins.  TTT → coin A wins.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(P, P ** 2 * (1 - P), color=BLUE, linewidth=2.4, label="L(p) for HHT = p²(1 − p)")
    ax.plot(P, (1 - P) ** 3, color=ORANGE, linewidth=2.4, label="L(p) for TTT = (1 − p)³")
    for p, v in zip(coins.values(), hht):
        ax.plot(p, v, "o", color=BLUE, markersize=8, zorder=3)
    ax.plot(2 / 3, (2 / 3) ** 2 / 3, "*", color=INK, markersize=16, zorder=4)
    ax.annotate("best over all p: 2/3 = the share of heads", (2 / 3, (2 / 3) ** 2 / 3), textcoords="offset points",
                xytext=(14, 10), fontsize=10.5, weight="bold")
    ax.set_xlabel("heads probability p")
    ax.set_ylabel("likelihood")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True)
    ax.set_title("The likelihood curve: every coin at once (dots: coins A, B, C)")

    panel(fig.add_subplot(gs[1, 2]), "In machine learning", [
        "Scenario → model (its parameters).",
        "Evidence → the training data.",
        "Training → find the parameters that",
        "make the data most probable.",
        "",
        "Next notebooks: the best coin is k/n,",
        "the best Gaussian uses the sample mean",
        "and variance, and the best regression",
        "line is least squares.",
    ], spacing=0.09)
    save(fig, name="01_intro_mle.png")


# ================================================================ notebook 02
def sheet_02_mle_bernoulli():
    heads, tails = 8, 2
    coins = {"coin 1 (p = 0.7)": 0.7, "coin 2 (p = 0.5)": 0.5, "coin 3 (p = 0.3)": 0.3}
    lik = [p ** heads * (1 - p) ** tails for p in coins.values()]
    L = P ** heads * (1 - P) ** tails
    logL = heads * np.log(P) + tails * np.log(1 - P)
    assert abs(P[np.argmax(L)] - 0.8) < 1e-3 and abs(P[np.argmax(logL)] - 0.8) < 1e-3

    fig = new_figure("MLE for a coin: with k heads in n flips, the best p is k/n",
                     "Data: 8 heads and 2 tails. Maximize L(p) = p⁸(1 − p)², or more easily its log.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(3), lik, width=0.6, color=[ORANGE, BLUE, BLUE], zorder=2)
    for i, v in enumerate(lik):
        ax.text(i, v * 1.4, f"{v:.2g}", ha="center", fontsize=10.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.05)
    ax.set_xticks(range(3), list(coins), fontsize=10)
    ax.set_ylabel("P(8H 2T | coin), log scale")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Three coins: coin 1 wins by 5.3× and 161×")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(P, L, color=BLUE, linewidth=2.4)
    for p, v in zip(coins.values(), lik):
        ax.plot(p, v, "o", color=BLUE, markersize=8)
    ax.axvline(0.8, color=ORANGE, linestyle="--", linewidth=1.8)
    ax.text(0.79, L.max() * 1.03, "p̂ = 0.8", ha="right", fontsize=11, color=ORANGE, weight="bold")
    ax.set_xlabel("p")
    ax.set_ylabel("L(p) = p⁸(1 − p)²")
    ax.set_ylim(0, L.max() * 1.15)
    ax.grid(True)
    ax.set_title("All coins at once: the peak is at 0.8")

    panel(fig.add_subplot(gs[0, 2]), "The derivation", [
        "Take the log (same peak, easier algebra):",
        "log L(p) = 8 log p + 2 log(1 − p)",
        "",
        "Set the derivative to zero:",
        r"$\dfrac{8}{p} - \dfrac{2}{1-p} = 0 \;\Rightarrow\; 8(1-p) = 2p \;\Rightarrow\; \hat{p} = 0.8$",
        "",
        "",
        "In general:  p̂ = k / n,",
        "the sample proportion of heads.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(P, logL, color=BLUE, linewidth=2.4)
    ax.axvline(0.8, color=ORANGE, linestyle="--", linewidth=1.8)
    ax.set_ylim(-20, 0)
    ax.set_xlabel("p")
    ax.set_ylabel("log L(p)")
    ax.grid(True)
    ax.set_title("Log-likelihood: different heights, same peak")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(P, heads / P - tails / (1 - P), color=BLUE, linewidth=2.4)
    ax.axhline(0, color=INK2, linewidth=1.2)
    ax.plot(0.8, 0, "o", color=ORANGE, markersize=10, zorder=3)
    ax.annotate("derivative = 0 at p = 0.8", (0.8, 0), textcoords="offset points", xytext=(-12, 22),
                ha="right", fontsize=10.5, color=ORANGE, weight="bold")
    ax.set_ylim(-60, 60)
    ax.set_xlabel("p")
    ax.set_ylabel("d/dp log L")
    ax.grid(True)
    ax.set_title("The derivative crosses zero at the maximum")

    panel(fig.add_subplot(gs[1, 2]), "Other datasets, same formula", [
        "3 heads in 10  →  p̂ = 0.3",
        "5 heads in 10  →  p̂ = 0.5",
        "10 heads in 10 →  p̂ = 1",
        "",
        "\"This coin always lands heads\" after",
        "10 flips is overconfident: MLE trusts",
        "small data completely. MAP (notebook 08)",
        "adds a prior to soften that.",
    ], spacing=0.09)
    save(fig, name="02_mle_bernoulli.png")


# ================================================================ notebook 03
def sheet_03_mle_gaussian():
    data = np.array([1.0, -1.0])
    mus = {-1: AQUA, 0: ORANGE, 1: BLUE}
    sigmas = {0.5: AQUA, 1: ORANGE, 2: BLUE}
    x = np.linspace(-4, 4, 400)

    fig = new_figure("MLE for a Gaussian: the best μ and σ² are the data's own mean and variance",
                     "Data: the two points 1 and −1. The likelihood of a Gaussian is the product of its heights at the data.")
    gs = grid(fig, top=0.84, wspace=0.3)

    for col, (title, curves) in enumerate([("Contest of means (σ = 1)", [(m, 1, c) for m, c in mus.items()]),
                                           ("Contest of widths (μ = 0)", [(0, s, c) for s, c in sigmas.items()])]):
        ax = fig.add_subplot(gs[0, col])
        for mu, s, color in curves:
            heights = normal_pdf(data, mu, s)
            name = f"N({mu}, {s:g}²)"
            ax.plot(x, normal_pdf(x, mu, s), color=color, linewidth=2.2,
                    label=f"{name}: product {heights.prod():.3f}")
            ax.vlines(data, 0, heights, color=color, linewidth=3, alpha=0.6)
        ax.plot(data, [0, 0], "o", color=INK, markersize=9, zorder=4)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("x (data at −1 and 1)")
        ax.set_ylabel("density")
        ax.legend(loc="upper left", fontsize=9.5)
        ax.grid(True)
        ax.set_title(title)

    panel(fig.add_subplot(gs[0, 2]), "The result", [
        "Winners: μ = 0 and σ = 1, exactly the",
        "data's own moments:",
        r"$\hat{\mu} = \bar{x}, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum (x_i - \bar{x})^2$",
        "",
        "For {1, −1}: mean 0, variance (1 + 1)/2 = 1.",
        "",
        "MLE divides by n, not n − 1:",
        "s² would give 2 here.",
    ], spacing=0.095)

    mu_grid, sigma_grid = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(0.3, 3, 271))
    surface = normal_pdf(1, mu_grid, sigma_grid) * normal_pdf(-1, mu_grid, sigma_grid)
    ax = fig.add_subplot(gs[1, :2])
    image = ax.contourf(mu_grid, sigma_grid, surface, levels=20, cmap=SEQ)
    ax.contour(mu_grid, sigma_grid, surface, levels=10, colors="white", linewidths=0.6, alpha=0.6)
    peak = np.unravel_index(np.argmax(surface), surface.shape)
    ax.plot(mu_grid[peak], sigma_grid[peak], "*", color=ORANGE, markersize=18)
    ax.annotate(f"peak at μ = {mu_grid[peak]:.2f}, σ = {sigma_grid[peak]:.2f}", (mu_grid[peak], sigma_grid[peak]),
                textcoords="offset points", xytext=(16, -4), fontsize=11, weight="bold", color="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("likelihood")
    colorbar.outline.set_visible(False)
    ax.set_xlabel("μ")
    ax.set_ylabel("σ")
    ax.set_title("Both parameters at once: the likelihood surface over (μ, σ)")

    panel(fig.add_subplot(gs[1, 2]), "Why balance wins", [
        "N(−1, 1) nails one point (height 0.399)",
        "but misses the other (0.054):",
        "product 0.399 × 0.054 = 0.022.",
        "",
        "N(0, 1) gives 0.242 to each point:",
        "product 0.242² = 0.059.",
        "",
        "A product rewards explaining all the",
        "data adequately over some of it perfectly.",
    ], spacing=0.09)
    save(fig, name="03_mle_gaussian.png")


# ================================================================ notebook 04
def sheet_04_mle_regression():
    rng = np.random.default_rng(42)
    m_true, b_true = 1.2, 2.0
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y = m_true * x + b_true + rng.normal(0, 1, size=len(x))
    lines = {"model 1": (0.2, 4.5), "model 2": (1.2, 2.0), "model 3": (2.2, -0.5)}

    def sse(m, b):
        return np.sum((y - (m * x + b)) ** 2)

    def loglik(m, b, sigma=1.0):
        d = y - (m * x + b)
        return np.sum(-0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)))

    fit_m, fit_b = np.polyfit(x, y, 1)

    fig = new_figure("MLE for linear regression is least squares",
                     "Each point is a sample from a Gaussian standing on the line. Maximizing their joint likelihood "
                     "minimizes Σ d², the squared vertical distances.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, 5, 100)
    ax.plot(xs, m_true * xs + b_true, color=INK2, linewidth=2, label="the line (road)")
    for xi, yi in zip(x, y):
        center = m_true * xi + b_true
        t = np.linspace(-3, 3, 100)
        ax.plot(xi + 0.9 * normal_pdf(t), center + t, color=BLUE, linewidth=1.5)
        ax.plot([xi, xi], [center - 3, center + 3], color=BLUE_LIGHT, linewidth=1)
    ax.plot(x, y, "o", color=ORANGE, markersize=9, zorder=4, label="data (houses)")
    ax.set_xlim(0, 5.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The generative story: N(0, 1) noise around the line")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x, y, "o", color=INK, markersize=9, zorder=4)
    for (name, (m, b)), color in zip(lines.items(), [AQUA, ORANGE, BLUE]):
        ax.plot(xs, m * xs + b, color=color, linewidth=2.2,
                label=f"{name}: SSE {sse(m, b):.1f}, log L {loglik(m, b):.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Smallest SSE = largest likelihood")

    panel(fig.add_subplot(gs[0, 2]), "The collapse", [
        "One point, distance d from the line:",
        r"$\dfrac{1}{\sqrt{2\pi}}\, e^{-d^2/2}$",
        "",
        "",
        "All points (independent):",
        r"$L = \prod_i \dfrac{1}{\sqrt{2\pi}}\, e^{-d_i^2/2} \;\propto\; e^{-\sum_i d_i^2/2}$",
        "",
        "",
        "Take the log, drop constants:",
        "maximize L  ⇔  minimize Σ dᵢ²",
    ], spacing=0.095)

    ms, bs = np.meshgrid(np.linspace(-1, 3, 300), np.linspace(-2, 6, 300))
    D = y[None, None, :] - (ms[..., None] * x[None, None, :] + bs[..., None])
    sse_grid = np.sum(D ** 2, axis=-1)
    ax = fig.add_subplot(gs[1, 0])
    image = ax.contourf(ms, bs, np.log(sse_grid), levels=20, cmap=SEQ.reversed())
    best = np.unravel_index(np.argmin(sse_grid), sse_grid.shape)
    ax.plot(ms[best], bs[best], "*", color=ORANGE, markersize=18, label="min SSE = max likelihood")
    ax.plot(fit_m, fit_b, "o", color="white", markersize=6, markeredgecolor=INK, label=f"np.polyfit ({fit_m:.2f}, {fit_b:.2f})")
    ax.set_xlabel("slope m")
    ax.set_ylabel("intercept b")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("One surface, two readings")

    ax = fig.add_subplot(gs[1, 1])
    m_slice = np.linspace(-0.5, 3, 300)
    for sigma, color in [(0.5, AQUA), (1, BLUE), (2, ORANGE)]:
        ll = np.array([loglik(m, fit_b, sigma) for m in m_slice])
        ax.plot(m_slice, ll - ll.max(), color=color, linewidth=2.2, label=f"noise σ = {sigma:g}")
    ax.axvline(fit_m, color=INK2, linestyle="--", linewidth=1.5)
    ax.set_ylim(-12, 0.8)
    ax.set_xlabel(f"slope m (intercept fixed at {fit_b:.2f})")
    ax.set_ylabel("log L − max")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True)
    ax.set_title("The noise level changes the width, not the winner")

    panel(fig.add_subplot(gs[1, 2]), "The identity in numbers", [
        "For any two lines:",
        r"$\dfrac{L_1}{L_2} = e^{-(\mathrm{SSE}_1 - \mathrm{SSE}_2)/2}$",
        "",
        "",
        "2 units less SSE = e ≈ 2.72× more likely.",
        "",
        "Two points (0, 1), (1, 3): y = 2x + 1 hits",
        "both (L ≈ 0.159); y = x + 1 misses one by",
        "1 (L ≈ 0.0965).",
    ], spacing=0.085)
    save(fig, name="04_mle_regression.png")


# ================================================================ notebook 05
def sheet_05_regularization():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 3, 11)
    y = 2 * x ** 2 - 4 * x + 5 + rng.normal(0, 1.6, 11)
    fits = {deg: np.polyfit(x, y, deg) for deg in (1, 2, 10)}
    video = {"linear": (10.0, 16.0), "quadratic": (2.0, 20.0), "degree 10": (0.1, 246.0)}
    rng_test = np.random.default_rng(7)
    x_test = rng_test.uniform(-2, 3, 200)
    y_test = 2 * x_test ** 2 - 4 * x_test + 5 + rng_test.normal(0, 1.6, 200)

    fig = new_figure("Regularization: penalize complexity so the simplest good fit wins",
                     "New loss = squared-error loss + λ · (sum of squared coefficients, constant term excluded).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-2, 3, 400)
    ax.plot(x, y, "o", color=INK, markersize=8, zorder=4)
    for deg, color, name in [(1, AQUA, "linear: underfit"), (2, BLUE, "quadratic: good fit"),
                             (10, ORANGE, "degree 10: overfit")]:
        ax.plot(xs, np.polyval(fits[deg], xs), color=color, linewidth=2.2, label=name)
    ax.set_ylim(-5, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Three models, one dataset")

    ax = fig.add_subplot(gs[0, 1])
    names = list(video)
    loss = [v[0] for v in video.values()]
    pen = [v[1] for v in video.values()]
    ax.bar(names, loss, color=BLUE, width=0.55, label="loss", zorder=2)
    ax.bar(names, pen, bottom=loss, color=ORANGE, width=0.55, label="L2 penalty", zorder=2)
    for i, (l, p) in enumerate(zip(loss, pen)):
        ax.text(i, l + p + 4, f"{l:g} + {p:g} = {l + p:g}", ha="center", fontsize=10.5,
                weight="bold" if i == 1 else "normal")
    ax.set_ylim(0, 290)
    ax.set_ylabel("loss + penalty")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", zorder=0)
    ax.set_title("The video's table: the quadratic wins (22)")

    panel(fig.add_subplot(gs[0, 2]), "The L2 penalty", [
        "Sum of squared coefficients, constant",
        "term excluded (shifting a curve up or",
        "down adds no wiggle):",
        "",
        "4x + 3:         4² = 16",
        "2x² − 4x + 5:  2² + 4² = 20",
        "degree 10:      16 + 81 + 4 + 9 + 36 + 100 = 246",
        "",
        "Raw loss crowns the degree-10 wiggle (0.1).",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    lam = np.logspace(-4, 2, 400)
    for (name, (l, p)), color in zip(video.items(), [AQUA, BLUE, ORANGE]):
        ax.plot(lam, l + lam * p, color=color, linewidth=2.2, label=name)
    for cross, text in [(1.9 / 226, "λ ≈ 0.0084"), (2.0, "λ = 2")]:
        ax.axvline(cross, color=INK2, linestyle=":", linewidth=1.3)
        ax.text(cross * 1.15, 300, text, fontsize=10, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("loss + λ · penalty (log scale)")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("The λ dial: each model has its regime")

    train = [np.mean((y - np.polyval(fits[d], x)) ** 2) for d in (1, 2, 10)]
    test = [np.mean((y_test - np.polyval(fits[d], x_test)) ** 2) for d in (1, 2, 10)]
    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(3)
    ax.bar(pos - 0.2, train, width=0.38, color=BLUE_LIGHT, label="training loss", zorder=2)
    ax.bar(pos + 0.2, test, width=0.38, color=BLUE, label="test loss (fresh data)", zorder=2)
    for xi, v in zip(np.r_[pos - 0.2, pos + 0.2], train + test):
        ax.text(xi, v + 0.5, f"{v:.2f}", ha="center", fontsize=9.5)
    ax.set_ylim(0, 31)
    ax.set_xticks(pos, ["linear", "quadratic", "degree 10"])
    ax.set_ylabel("mean squared error")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Why: the wiggle fails on new data")

    panel(fig.add_subplot(gs[1, 2]), "Reading λ", [
        "λ = 0: pure loss, degree 10 wins (overfit).",
        "λ ≈ 0.0084: the quadratic overtakes it,",
        "since 2 + 20λ = 0.1 + 246λ.",
        "λ = 1 (the video): quadratic, 22.",
        "λ > 2: the linear model wins (16 < 20).",
        "",
        "The penalty stands in for generalization:",
        "degree 10 has zero training error but",
        "nearly 3× the quadratic's test error.",
    ], spacing=0.09)
    save(fig, name="05_regularization.png")


# ================================================================ notebook 06
def sheet_06_mle_to_map():
    scenarios = {"movies": (0.70, 0.300), "board games": (0.30, 0.200), "nap": (0.05, 0.400),
                 "popcorn contest": (0.99, 0.001)}
    names = list(scenarios)
    lik = np.array([v[0] for v in scenarios.values()])
    prior = np.array([v[1] for v in scenarios.values()])
    joint = lik * prior
    threshold = 0.7 * 0.3 / 0.99
    assert np.argmax(lik) == 3 and np.argmax(joint) == 0

    fig = new_figure("From MLE to MAP: multiply the likelihood by the prior",
                     "The popcorn contest explains the evidence best, but it almost never happens. "
                     "Maximize P(evidence | scenario) · P(scenario) instead.")
    gs = grid(fig, top=0.84, wspace=0.3)

    grouped_bars(fig.add_subplot(gs[0, 0]), ["movies", "games", "nap", "contest"],
                 [("likelihood P(popcorn | s)", lik, BLUE_LIGHT), ("prior P(s)", prior, BLUE)],
                 "Likelihood vs. prior", "", ylim=1.2, label_fmt=lambda v: f"{v:g}")
    bars(fig.add_subplot(gs[0, 1]), ["movies", "games", "nap", "contest"], joint,
         "Joint = likelihood × prior: movies wins", "", ylabel="P(popcorn and scenario)", ylim=0.26,
         label_fmt=lambda v: f"{v:.3g}", highlight=0)

    panel(fig.add_subplot(gs[0, 2]), "The repair", [
        "MLE: the contest wins, 0.99 vs 0.7.",
        "",
        "Multiply by how likely each scenario is:",
        "movies   0.7 × 0.3   = 0.21",
        "contest  0.99 × 0.001 = 0.00099",
        "Movies wins by a factor of about 212.",
        "",
        "P(B | A) P(A) = P(A ∩ B): we are",
        "maximizing a joint probability.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    priors = np.linspace(0, 0.4, 300)
    ax.plot(priors, 0.99 * priors, color=ORANGE, linewidth=2.2, label="contest: 0.99 × P(contest)")
    ax.axhline(0.21, color=BLUE, linewidth=2.2, label="movies: 0.7 × 0.3 = 0.21")
    ax.axvline(threshold, color=INK2, linestyle=":", linewidth=1.4)
    ax.text(threshold + 0.005, 0.33, f"flip at P(contest) ≈ {threshold:.3f}", fontsize=10.5, color=INK2)
    ax.set_xlabel("prior P(contest)")
    ax.set_ylabel("joint probability")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("How likely must the contest be to win?")

    nights = np.arange(0, 31)
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(nights, 0.3 * 0.7 ** nights, "o-", color=BLUE, markersize=4, linewidth=2, label="movies every night")
    ax.semilogy(nights, 0.001 * 0.99 ** nights, "o-", color=ORANGE, markersize=4, linewidth=2,
                label="contest every night")
    cross = log(0.3 / 0.001) / log(0.99 / 0.7)
    ax.axvline(cross, color=INK2, linestyle=":", linewidth=1.4)
    ax.text(cross + 0.5, 0.1, f"evidence wins after\n{int(np.ceil(cross))} popcorn nights", fontsize=10, color=INK2)
    ax.set_xlabel("evenings in a row that ended with popcorn")
    ax.set_ylabel("joint probability (log scale)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Enough evidence overrides the prior")

    panel(fig.add_subplot(gs[1, 2]), "The moral", [
        "Priors don't forbid unlikely scenarios;",
        "they demand proportionate evidence.",
        "",
        "Same logic as the rare-disease test:",
        "a 99%-sensitive test usually flags a",
        "healthy person when the disease is rare.",
        "",
        "Ignoring the prior is the base-rate",
        "fallacy; multiplying by it is the cure.",
    ], spacing=0.09)
    save(fig, name="06_mle_to_map.png")


# ================================================================ notebook 07
def sheet_07_frequentist_vs_bayesian():
    heads, n = 8, 10
    strong, weak = 100, 2

    def bayes_answer(a, k, flips):
        return (a + k) / (2 * a + flips)

    frequentist = heads / n
    assert round(bayes_answer(strong, heads, n), 3) == 0.514 and round(bayes_answer(weak, heads, n), 2) == 0.71

    # the notebook's convergence run (seed 0) and tournament (seed 1)
    rng = np.random.default_rng(0)
    flips = rng.random(10_000) < 0.8
    ks = np.cumsum(flips)
    ns = np.arange(1, 10_001)
    freq_est = ks / ns
    bayes_est = (strong + ks) / (2 * strong + ns)

    rng = np.random.default_rng(1)

    def tournament(coin_sampler, trials=100_000, n_flips=10):
        ps = coin_sampler(trials)
        k = rng.binomial(n_flips, ps)
        return np.mean((k / n_flips - ps) ** 2), np.mean(((strong + k) / (2 * strong + n_flips) - ps) ** 2)

    fair_mse = tournament(lambda t: np.clip(rng.normal(0.5, 0.03, t), 0.01, 0.99))
    wild_mse = tournament(lambda t: rng.uniform(0.05, 0.95, t))
    assert fair_mse[1] < fair_mse[0] and wild_mse[0] < wild_mse[1]

    fig = new_figure("Frequentist vs. Bayesian: same 10 flips, two answers",
                     "A coin found in the street shows 8 heads in 10 tosses. One camp uses only the evidence, "
                     "the other updates a prior belief that coins are fair.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(P, belief(0, heads, n - heads), color=BLUE, linewidth=2.4, label="frequentist: likelihood of 8H 2T")
    ax.plot(P, belief(strong, 0, 0), color=MUTED, linewidth=1.8, linestyle="--", label="Bayesian prior (a = 100)")
    ax.plot(P, belief(strong, heads, n - heads), color=ORANGE, linewidth=2.4, label="Bayesian updated belief")
    ax.axvline(frequentist, color=BLUE, linestyle=":", linewidth=1.3)
    ax.axvline(bayes_answer(strong, heads, n), color=ORANGE, linestyle=":", linewidth=1.3)
    ax.text(frequentist + 0.015, 1.08, "0.8", color=BLUE, fontsize=11, weight="bold")
    ax.text(bayes_answer(strong, heads, n) + 0.015, 1.08, f"{bayes_answer(strong, heads, n):.3f}",
            color=ORANGE, fontsize=11, weight="bold")
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("heads probability p")
    ax.set_ylabel("curve (peak = 1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("The bar story")

    ax = fig.add_subplot(gs[0, 1])
    strengths = np.array([0, 1, 2, 5, 10, 30, 100, 300])
    answers = bayes_answer(strengths, heads, n)
    ax.plot(strengths, answers, "o-", color=BLUE, linewidth=2.2, markersize=7, zorder=3)
    for a, color in [(weak, AQUA), (strong, ORANGE)]:
        ax.plot(a, bayes_answer(a, heads, n), "o", color=color, markersize=11, zorder=4)
    ax.annotate(f"weak prior a = 2: {bayes_answer(weak, heads, n):.2f}", (weak, bayes_answer(weak, heads, n)),
                textcoords="offset points", xytext=(10, 6), fontsize=10, color=AQUA, weight="bold")
    ax.annotate(f"strong prior a = 100: {bayes_answer(strong, heads, n):.3f}",
                (strong, bayes_answer(strong, heads, n)), textcoords="offset points", xytext=(-12, -16),
                ha="right", va="top", fontsize=10, color=ORANGE, weight="bold")
    ax.axhline(0.8, color=MUTED, linestyle=":", linewidth=1.2)
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(450, 0.81, "frequentist 0.8", ha="right", va="bottom", fontsize=9.5, color=INK2)
    ax.text(0, 0.505, "prior center 0.5", ha="left", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.3, 450)
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("prior strength a (0 = no prior)")
    ax.set_ylabel("estimate after 8H 2T")
    ax.grid(True)
    ax.set_title("The prior-strength dial")

    panel(fig.add_subplot(gs[0, 2]), "Two meanings of probability", [
        "Frequentist: long-run frequency over",
        "endless repetitions. No prior; find the",
        "model that maximizes P(data | model).",
        "",
        "Bayesian: degree of belief. Start from a",
        "prior P(model) and update it with data.",
        "",
        "Everything up to here (MLE included)",
        "was frequentist.",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(ns, freq_est, color=BLUE, linewidth=1.8, label="frequentist k/n")
    ax.semilogx(ns, bayes_est, color=ORANGE, linewidth=2.2, label="Bayesian (a = 100)")
    ax.axhline(0.8, color=INK, linestyle="--", linewidth=1, label="true p = 0.8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of tosses (log scale)")
    ax.set_ylabel("estimate of p")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title(f"The camps converge: {freq_est[-1]:.3f} vs {bayes_est[-1]:.3f} at 10,000")

    ax = fig.add_subplot(gs[1, 1])
    pos = np.arange(2)
    for offset, idx, color, label in [(-0.2, 0, BLUE, "frequentist"), (0.2, 1, ORANGE, "Bayesian (a = 100)")]:
        values = [fair_mse[idx], wild_mse[idx]]
        ax.bar(pos + offset, values, width=0.36, color=color, label=label, zorder=2)
        for p, v in zip(pos, values):
            ax.text(p + offset, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(pos, ["fair-ish coins\n(prior is right)", "wild coins\n(prior is wrong)"])
    ax.set_ylim(0, max(wild_mse) * 1.25)
    ax.set_ylabel("mean squared error, 10 tosses")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", zorder=0)
    ax.set_title("Small n: a good prior helps, a wrong one hurts")

    w_prior = 2 * strong / (2 * strong + n)
    panel(fig.add_subplot(gs[1, 2]), "Priors as phantom flips", [
        r"$\hat{p}_{\mathrm{Bayes}} = \dfrac{a + k}{2a + n}$",
        "",
        "",
        "= weighted average of 0.5 and k/n,",
        f"weights {w_prior:.3f} and {1 - w_prior:.3f} for a = 100, n = 10.",
        "",
        "a = 100 means 200 fair flips already seen,",
        "so 10 real ones barely move the belief.",
        "Priors are initial conditions, not destinies.",
    ], spacing=0.085)
    save(fig, name="07_frequentist_vs_bayesian.png")


# ================================================================ notebook 08
BAYESIANS = [("conservative", 1500, ORANGE), ("moderate", 9, BLUE), ("agnostic (flat)", 0, AQUA)]


def belief(a, heads, tails):
    log_curve = (a + heads) * np.log(P) + (a + tails) * np.log(1 - P)
    return np.exp(log_curve - log_curve.max())


def sheet_08_map():
    def map_estimate(a, k, n):
        return (a + k) / (2 * a + n)

    maps = [map_estimate(a, 8, 10) for _, a, _ in BAYESIANS]
    assert [round(m, 3) for m in maps] == [0.501, 0.607, 0.8]

    fig = new_figure("Maximum a posteriori (MAP): the peak of the posterior belief",
                     "Three Bayesians with different priors about a coin see the same flips. "
                     "Curves are scaled to peak at 1.")
    gs = grid(fig, top=0.84, wspace=0.3)
    for col, (heads, tails, title) in enumerate([(0, 0, "Priors (before any flip)"), (1, 0, "After one head"),
                                                 (8, 2, "After 8 heads, 2 tails")]):
        ax = fig.add_subplot(gs[0, col])
        for (name, a, color), m in zip(BAYESIANS, maps):
            ax.plot(P, belief(a, heads, tails), color=color, linewidth=2.2, label=name)
            if col == 2:
                ax.plot(m, 1, "o", color=color, markersize=8, zorder=3)
                shift = {1500: -0.03, 9: -0.02, 0: 0.03}.get(a, 0)
                ax.text(m + shift, 1.05, f"{m:.3f}", ha="right" if shift < 0 else ("left" if shift > 0 else "center"),
                        fontsize=10, color=color, weight="bold")
        ax.set_ylim(0, 1.18)
        ax.set_xlabel("heads probability p")
        ax.set_ylabel("belief (peak = 1)")
        ax.grid(True)
        ax.set_title(title)
        if col == 0:
            ax.legend(loc="center right", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 0])
    a_grid = np.logspace(-1, 4, 300)
    ax.semilogx(a_grid, map_estimate(a_grid, 8, 10), color=BLUE, linewidth=2.2)
    for (name, a, color), m in zip(BAYESIANS, maps):
        ax.plot(max(a, 0.1), m, "o", color=color, markersize=9, zorder=3)
        ax.annotate(f"{name}: {m:.3f}", (max(a, 0.1), m), textcoords="offset points",
                    xytext=(8, 6) if a != 1500 else (-8, 10), ha="left" if a != 1500 else "right",
                    fontsize=10, color=color, weight="bold")
    ax.axhline(0.8, color=MUTED, linestyle=":", linewidth=1.2)
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    ax.set_xlabel("prior strength a (phantom fair flips per side, log scale)")
    ax.set_ylabel("MAP estimate for 8H 2T")
    ax.grid(True)
    ax.set_title("From MLE (0.8) to the prior (0.5)")

    panel(fig.add_subplot(gs[1, 1]), "Priors as phantom flips", [
        "Prior ∝ pᵃ(1 − p)ᵃ: a fair heads and",
        "a fair tails already \"seen\".",
        "Posterior after k heads in n flips:",
        "∝ pᵃ⁺ᵏ (1 − p)ᵃ⁺ⁿ⁻ᵏ, peak at",
        r"$\hat{p}_{\mathrm{MAP}} = \dfrac{a + k}{2a + n}$",
        "",
        "",
        "a = 1500: 0.501   a = 9: 0.607   a = 0: 0.8",
    ], spacing=0.09)

    panel(fig.add_subplot(gs[1, 2]), "Flat prior ⇒ MAP = MLE", [
        "The agnostic's prior is constant, so",
        "maximizing prior × likelihood is just",
        "maximizing the likelihood.",
        "",
        "After one head their belief is the line",
        "p itself, peaking at 1 (the MLE edge case).",
        "",
        "The curve says more than its peak: the",
        "moderate's 0.607 is \"0.6, give or take 0.1\".",
    ], spacing=0.09)
    save(fig, name="08_map.png")


# ================================================================ notebook 09
def sheet_09_bayesian_updating():
    coin_types = np.array([0.5, 0.8])
    prior = np.array([0.75, 0.25])

    def update(belief_now, outcome):
        likelihood = np.where(outcome == 1, coin_types, 1 - coin_types)
        joint = likelihood * belief_now
        return joint / joint.sum()

    after_h = update(prior, 1)
    after_t = update(prior, 0)
    after_ht = update(after_h, 0)
    assert round(after_h[0], 3) == 0.652 and round(after_t[0], 3) == 0.882 and round(after_ht[0], 3) == 0.824

    fig = new_figure("Updating beliefs with Bayes: posterior ∝ likelihood × prior",
                     "A mystery coin is fair (heads 0.5) or biased (heads 0.8). Most coins are fair: prior 0.75 vs 0.25.")
    gs = grid(fig, top=0.84, wspace=0.3)

    joint = coin_types * prior
    steps = ["prior", "likelihood\nof heads", "joint", "posterior"]
    grouped_bars(fig.add_subplot(gs[0, 0]), steps,
                 [("fair coin", [prior[0], coin_types[0], joint[0], after_h[0]], BLUE),
                  ("biased coin", [prior[1], coin_types[1], joint[1], after_h[1]], ORANGE)],
                 "Anatomy of one update (heads)", "", ylim=1.0, label_fmt=lambda v: f"{v:.3g}")

    bars(fig.add_subplot(gs[0, 1]), ["prior", "after heads", "after tails"], [prior[0], after_h[0], after_t[0]],
         "P(coin is fair): heads pushes down, tails up", "", ylabel="P(fair)", ylim=1.05,
         label_fmt=lambda v: f"{v:.3f}", highlight=0)

    panel(fig.add_subplot(gs[0, 2]), "The four parts", [
        r"$P(\mathrm{fair} \mid H) = \dfrac{P(H \mid \mathrm{fair})\; P(\mathrm{fair})}{P(H)}$",
        "",
        "",
        "evidence P(H) = 0.5 × 0.75 + 0.8 × 0.25 = 0.575",
        "posterior = 0.5 × 0.75 / 0.575 = 0.652",
        "",
        "The evidence is the same for both coins:",
        "it rescales, never re-ranks (why MAP",
        "can ignore it).",
    ], spacing=0.085)

    rng = np.random.default_rng(0)
    flips = (rng.random(50) < 0.8).astype(int)
    belief_now = prior.copy()
    trajectory = [belief_now[0]]
    for f in flips:
        belief_now = update(belief_now, f)
        trajectory.append(belief_now[0])
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(range(51), trajectory, color=BLUE, linewidth=2, zorder=2)
    for i, f in enumerate(flips, start=1):
        ax.plot(i, trajectory[i], "o", color=ORANGE if f else AQUA, markersize=5, zorder=3)
    ax.axhline(0.75, color=MUTED, linestyle=":", linewidth=1.3)
    ax.text(50, 0.77, "prior 0.75", ha="right", fontsize=10, color=INK2)
    ax.plot([], [], "o", color=ORANGE, label="heads")
    ax.plot([], [], "o", color=AQUA, label="tails")
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("flips observed (the coin is actually biased, p = 0.8)")
    ax.set_ylabel("P(coin is fair | data)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title(f"Chained updates: P(fair) = {trajectory[10]:.3f} after 10 flips, {trajectory[50]:.4f} after 50")

    panel(fig.add_subplot(gs[1, 2]), "Chaining", [
        "Yesterday's posterior is today's prior:",
        f"0.75 → heads → {after_h[0]:.3f} → tails → {after_ht[0]:.3f}",
        "",
        "The order does not matter: HT and TH",
        "end at the same belief.",
        "",
        "Same recipe for a continuous θ:",
        "posterior density ∝ likelihood × prior",
        "density.",
    ], spacing=0.09)
    save(fig, name="09_bayesian_updating.png")


# ================================================================ notebook 10
def sheet_10_bayesian_worked_example():
    theta = np.linspace(0.001, 0.999, 999)          # the notebook's grid

    def density(heads, tails):
        curve = theta ** heads * (1 - theta) ** tails
        return curve / trapezoid(curve, theta)

    def map_of(curve):
        return theta[np.argmax(curve)]

    uniform = density(0, 0)
    post_10 = density(8, 2)
    post_20 = density(14, 6)
    batch_2 = density(6, 4)
    assert round(map_of(post_10), 2) == 0.8 and round(map_of(post_20), 2) == 0.7 and round(map_of(batch_2), 2) == 0.6
    assert abs(post_10.max() - 495 * 0.8 ** 8 * 0.2 ** 2) < 0.01

    fig = new_figure("A complete Bayesian worked example: flat prior to firm belief",
                     "A mystery coin: first 8 heads and 2 tails, then 6 heads and 4 tails. "
                     "Each posterior becomes the next prior.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for curve, label, color in [(uniform, "uniform prior: flat, no peak", MUTED),
                                (post_10, "after 10 flips: 495 θ⁸(1 − θ)²", BLUE),
                                (post_20, "after 20 flips: ∝ θ¹⁴(1 − θ)⁶", ORANGE)]:
        ax.plot(theta, curve, color=color, linewidth=2.3, label=label)
        ax.fill_between(theta, curve, color=color, alpha=0.12)
    for curve, color in [(post_10, BLUE), (post_20, ORANGE)]:
        m = map_of(curve)
        ax.plot(m, curve.max(), "o", color=color, markersize=8, zorder=3)
        ax.text(m, curve.max() + 0.12, f"MAP {m:.1f}", ha="center", fontsize=10.5, color=color, weight="bold")
    ax.set_ylim(0, 5.2)
    ax.set_xlabel("θ (heads probability)")
    ax.set_ylabel("posterior density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Flat → peak at 0.8 → sharper peak at 0.7")

    ax = fig.add_subplot(gs[0, 1])
    for curve, label, color, style in [(batch_2, "batch 2 alone (6H 4T): 0.6", AQUA, "-"),
                                       (post_10, "prior = batch-1 posterior: 0.8", BLUE, "--"),
                                       (post_20, "posterior (product): 0.7", ORANGE, "-")]:
        ax.plot(theta, curve, color=color, linewidth=2.3, linestyle=style, label=label)
        ax.axvline(map_of(curve), color=color, linestyle=":", linewidth=1.2)
    ax.set_ylim(0, 5.2)
    ax.set_xlabel("θ (heads probability)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("The prior pulls batch 2's 0.6 up to 0.7")

    panel(fig.add_subplot(gs[0, 2]), "The recipe", [
        "posterior ∝ likelihood × prior",
        "Batch 1:  θ⁸(1 − θ)² × 1",
        "Batch 2:  θ⁶(1 − θ)⁴ × θ⁸(1 − θ)² = θ¹⁴(1 − θ)⁶",
        "",
        "Exponents add, so MAP = heads / flips:",
        "8/10 = 0.8, then 14/20 = 0.7.",
        "",
        "The constant 495 = 11!/(8! 2!) rescales the",
        "curve but never moves its peak.",
    ], spacing=0.09)

    # the notebook's 'data waters down the prior' run (seed 42, 400 flips)
    rng = np.random.default_rng(42)
    stream = (rng.random(400) < 0.8).astype(int)
    h = np.cumsum(stream)
    t = np.arange(1, 401) - h
    log_t, log_1t = np.log(theta), np.log(1 - theta)
    ax = fig.add_subplot(gs[1, :2])
    finals = []
    for (label, a, b), color in zip([("flat prior (MAP = MLE here)", 0, 0), ("mildly wrong prior ∝ θ⁹(1 − θ)⁹", 9, 9),
                                     ("badly wrong prior ∝ θ²(1 − θ)¹⁸", 2, 18)], [MUTED, BLUE, ORANGE]):
        traj = theta[np.argmax((a + h)[:, None] * log_t + (b + t)[:, None] * log_1t, axis=1)]
        finals.append(traj[-1])
        ax.plot(np.arange(1, 401), traj, color=color, linewidth=2.2, label=label)
    ax.plot(np.arange(1, 401), h / np.arange(1, 401), color=INK, linewidth=1, linestyle="--",
            label="MLE (running share of heads)")
    ax.axhline(0.8, color=INK, linestyle=":", linewidth=1, label="true θ = 0.8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of flips observed")
    ax.set_ylabel("MAP estimate of θ")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title(f"Data waters down the prior: after 400 flips all three MAPs are within "
                 f"{max(finals) - min(finals):.2f} of each other")

    panel(fig.add_subplot(gs[1, 2]), "What the example proves", [
        "Flat prior ⇒ MAP = MLE: a constant",
        "never moves an argmax.",
        "",
        "Chunking doesn't matter: one update of",
        "20 flips, two of 10, or 20 of one all",
        "end at θ¹⁴(1 − θ)⁶.",
        "",
        "Priors matter most with little data,",
        "and wrong priors cost many flips to undo.",
    ], spacing=0.09)
    save(fig, name="10_bayesian_worked_example.png")


# ================================================================ notebook 11
def sheet_11_map_regularization():
    rng = np.random.default_rng(3)
    gauss_const = 1 / np.sqrt(2 * np.pi)
    x_pts = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    y_pts = 1.2 * x_pts + 0.4 * x_pts ** 2 + rng.normal(0, 0.5, size=x_pts.size)
    X = np.column_stack([x_pts, x_pts ** 2])
    ridge = np.linalg.solve(X.T @ X + np.eye(2), X.T @ y_pts)
    mle = np.linalg.lstsq(X, y_pts, rcond=None)[0]
    x_data = np.linspace(0, 3, 12)
    y_data = 1.0 * x_data + 0.5 * x_data ** 2 + rng.normal(0, 0.6, size=x_data.size)
    contest = {}
    for degree in (1, 2, 10):
        coeffs = np.polyfit(x_data, y_data, degree)
        residuals = y_data - np.polyval(coeffs, x_data)
        contest[degree] = (np.sum(np.log(gauss_const) - 0.5 * residuals ** 2),
                           np.sum(np.log(gauss_const) - 0.5 * coeffs ** 2))
    best_joint = max(contest, key=lambda d: sum(contest[d]))
    priors = {"y = 2x": [2], "y = 2x + 0.5x²": [2, 0.5], "degree 10,\nall |aᵢ| = 2": [2] * 10}
    prior_values = [np.prod(gauss_const * np.exp(-0.5 * np.array(c, float) ** 2)) for c in priors.values()]

    fig = new_figure("MAP with a Gaussian prior on the coefficients is L2-regularized regression",
                     "log(likelihood × prior) = −½ Σ dⱼ² − ½ Σ aᵢ² + constant.  Maximizing it minimizes loss + L2 penalty.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    degrees = list(contest)
    pos = np.arange(3)
    for offset, idx, color, label in [(-0.27, 0, BLUE_LIGHT, "log-likelihood (fit)"),
                                      (0.0, 1, ORANGE, "log-prior (simplicity)"),
                                      (0.27, 2, BLUE, "sum: log joint")]:
        values = [contest[d][0] + contest[d][1] if idx == 2 else contest[d][idx] for d in degrees]
        ax.bar(pos + offset, values, width=0.26, color=color, label=label, zorder=2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_yscale("symlog")
    ax.set_xticks(pos, [f"degree {d}" for d in degrees])
    ax.set_ylabel("log probability (symlog)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", zorder=0)
    ax.set_title(f"The model contest: degree {best_joint} wins the joint")

    a1, a2 = np.meshgrid(np.linspace(-1, 3, 300), np.linspace(-1, 3, 300))
    pred = x_pts[:, None, None] * a1 + (x_pts ** 2)[:, None, None] * a2
    log_lik = -0.5 * ((y_pts[:, None, None] - pred) ** 2).sum(axis=0)
    log_post = log_lik - 0.5 * (a1 ** 2 + a2 ** 2)
    ax = fig.add_subplot(gs[0, 1])
    levels = np.linspace(log_post.max() - 12, log_post.max(), 13)
    ax.contourf(a1, a2, log_post, levels=levels, cmap=SEQ, extend="min")
    ax.contour(a1, a2, log_lik, levels=np.linspace(log_lik.max() - 12, log_lik.max(), 7), colors=INK2,
               linewidths=0.8, linestyles="--")
    ax.plot(*mle, "s", color=INK, markersize=9, label=f"MLE ({mle[0]:.2f}, {mle[1]:.2f})")
    ax.plot(*ridge, "*", color=ORANGE, markersize=17, label=f"MAP = ridge ({ridge[0]:.2f}, {ridge[1]:.2f})")
    ax.plot(0, 0, "o", color="white", markeredgecolor=INK, markersize=8, label="prior center (0, 0)")
    ax.annotate("", xy=ridge, xytext=mle, arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.8))
    ax.set_xlabel("a₁")
    ax.set_ylabel("a₂")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("The prior pulls the peak toward 0")

    panel(fig.add_subplot(gs[0, 2]), "The dictionary", [
        "likelihood P(data | model)  ↔  squared loss",
        "prior P(model)                  ↔  L2 penalty",
        "MAP                               ↔  regularized fit",
        "MLE (flat prior)               ↔  plain least squares",
        "",
        "With noise N(0, σ_d²) and coefficients",
        "N(0, σ_a²):",
        r"$\lambda = \sigma_d^2 / \sigma_a^2$",
    ], spacing=0.095)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(range(3), prior_values, width=0.6, color=[BLUE, BLUE, ORANGE], zorder=2)
    for i, v in enumerate(prior_values):
        ax.text(i, v * 2.5, f"{v:.2g}", ha="center", fontsize=10.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1)
    ax.set_xticks(range(3), list(priors), fontsize=10)
    ax.set_ylabel("P(model), log scale")
    ax.grid(axis="y", zorder=0)
    ax.set_title("Each extra or bigger coefficient costs a factor")

    lam = np.logspace(-2, 2, 200)
    paths = np.array([np.linalg.solve(X.T @ X + l * np.eye(2), X.T @ y_pts) for l in lam])
    ax = fig.add_subplot(gs[1, 1])
    ax.semilogx(lam, paths[:, 0], color=BLUE, linewidth=2.2, label="a₁")
    ax.semilogx(lam, paths[:, 1], color=ORANGE, linewidth=2.2, label="a₂")
    ax.axvline(1, color=INK2, linestyle=":", linewidth=1.3)
    ax.text(1.1, paths[:, 0].max() * 0.95, "λ = 1: σ_a = σ_d", fontsize=10, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("λ = σ_d² / σ_a²  (narrower prior → larger λ)")
    ax.set_ylabel("MAP coefficient")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True)
    ax.set_title("Prior width is the λ dial")

    panel(fig.add_subplot(gs[1, 2]), "The derivation", [
        "Model y = a₁x + a₂x², five points missed",
        "by d₁ … d₅, coefficients ~ N(0, 1):",
        "",
        "log(likelihood × prior)",
        "  = −½(d₁² + … + d₅²) − ½(a₁² + a₂²) + const",
        "",
        "Multiply by −2 and minimize:",
        "Σ dⱼ² + (a₁² + a₂²) = loss + L2 penalty.",
    ], spacing=0.09)
    save(fig, name="11_map_regularization.png")


if __name__ == "__main__":
    sheet_01_intro_mle()
    sheet_02_mle_bernoulli()
    sheet_03_mle_gaussian()
    sheet_04_mle_regression()
    sheet_05_regularization()
    sheet_06_mle_to_map()
    sheet_07_frequentist_vs_bayesian()
    sheet_08_map()
    sheet_09_bayesian_updating()
    sheet_10_bayesian_worked_example()
    sheet_11_map_regularization()
