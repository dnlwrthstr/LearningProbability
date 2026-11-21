#!/usr/bin/env python3
"""Concept sheets for lesson 07 (hypothesis testing).

One page per notebook that summarizes its central idea. Every number is
computed here from the same data, parameters and random seeds the notebook uses.
Images are written to coursera/07_hypothesis_testing/img/.

Run from anywhere:  python scripts/make_lesson07_concept_images.py
"""
from functools import partial
from math import comb, erf, gamma, pi, sqrt

import numpy as np
import sympy as sp

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, bars, grouped_bars, heat, panel, trapezoid)
from viz_style import save as _save

LESSON = REPO / "coursera" / "07_hypothesis_testing"
save = partial(_save, folder=LESSON / "img")

ORANGE_LIGHT = "#f8c9b3"     # the lesson 06 tint of ORANGE for secondary fills

# the height example shared by notebooks 03-06
MU0, SIGMA, N_HEIGHTS = 66.7, 3.0, 10
SE_HEIGHTS = SIGMA / sqrt(N_HEIGHTS)
X_OBS, X_OBS_LEFT = 68.442, 64.252
HEIGHTS = np.array([64.12, 71.15, 66.41, 70.17, 73.13, 65.10, 69.16, 67.39, 71.97, 65.82])


# ================================================================ helpers
def z_score(p):
    """Critical value z_p: the point with probability p to its left."""
    return float(sp.sqrt(2) * sp.erfinv(2 * sp.Float(p) - 1))


def Phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def normal_pdf(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def t_pdf(t, df):
    coeff = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    return coeff * (1 + np.asarray(t, float) ** 2 / df) ** (-(df + 1) / 2)


def t_tail(t_value, df):
    """P(T > t_value) by numeric integration, as in the notebooks."""
    grid_t = np.linspace(0, abs(t_value), 20_000)
    tail = 0.5 - trapezoid(t_pdf(grid_t, df), grid_t)
    return tail if t_value >= 0 else 1 - tail


def t_score(p, df):
    """t critical value by bisection on the numerically integrated density."""
    target, lo, hi = p - 0.5, 0.0, 200.0
    for _ in range(100):
        mid = (lo + hi) / 2
        grid_t = np.linspace(0, mid, 4001)
        if trapezoid(t_pdf(grid_t, df), grid_t) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def p_value_heights(x_bar, tail):
    z = (x_bar - MU0) / SE_HEIGHTS
    return {"right": 1 - Phi(z), "left": Phi(z), "two": 2 * (1 - Phi(abs(z)))}[tail]


def crit_heights(alpha, tail):
    if tail == "right":
        return MU0 + z_score(1 - alpha) * SE_HEIGHTS
    if tail == "left":
        return MU0 + z_score(alpha) * SE_HEIGHTS
    half = z_score(1 - alpha / 2) * SE_HEIGHTS
    return MU0 - half, MU0 + half


def power_heights(mu, alpha=0.05, n=N_HEIGHTS):
    se = SIGMA / np.sqrt(n)
    k = MU0 + z_score(1 - alpha) * se
    z = (k - np.asarray(mu, float)) / se
    return 1 - 0.5 * (1 + np.vectorize(erf)(z / sqrt(2)))


def tail_plot(ax, x, y, region, color=ORANGE, base=INK2):
    ax.plot(x, y, color=base, linewidth=1.8, zorder=2)
    ax.fill_between(x, y, where=region, color=color, alpha=0.75, zorder=1, linewidth=0)


def welch(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return (m1 - m2) / se, df, se, v1, v2


# ================================================================ notebook 02
def sheet_02_type_errors():
    p_ham, p_spam, n_phrases = 0.001, 0.40, 6
    rng = np.random.default_rng(6)          # the notebook's seed and draw order
    n_emails = 100_000
    is_spam = rng.random(n_emails) < 0.05
    counts = np.where(is_spam, rng.binomial(n_phrases, p_spam, n_emails),
                      rng.binomial(n_phrases, p_ham, n_emails))
    reject = counts >= 2
    cells = np.array([[(~is_spam & ~reject).sum(), (~is_spam & reject).sum()],
                      [(is_spam & ~reject).sum(), (is_spam & reject).sum()]])
    assert cells.tolist() == [[95081, 1], [1184, 3734]]
    rates = cells / cells.sum(axis=1, keepdims=True)

    def ham_tail(k):
        return sum(comb(n_phrases, j) * p_ham ** j * (1 - p_ham) ** (n_phrases - j)
                   for j in range(k, n_phrases + 1))

    tails = [ham_tail(k) for k in range(1, 5)]
    assert abs(tails[0] - 5.99e-3) < 1e-5 and abs(tails[1] - 1.50e-5) < 1e-7

    fig = new_figure("Two ways to be wrong, and α caps the first one",
                     "Type I = reject a true H₀ (ham to the spam box); Type II = keep a false H₀ "
                     "(spam in the inbox). α = the largest Type I probability you accept.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0].subgridspec(2, 1, height_ratios=[0.13, 1], hspace=0)[1])
    labels = [[f"{cells[i, j]:,}\n{rates[i, j]:.4%}" if (i, j) == (0, 1) else f"{cells[i, j]:,}\n{rates[i, j]:.1%}"
               for j in range(2)] for i in range(2)]
    heat(ax, rates, ["H₀ true\n(ham)", "H₁ true\n(spam)"], ["keep H₀", "reject H₀"],
         "100,000 emails through the k ≥ 2 rule", "", "", labels=labels, vmax=1.0,
         highlight_cells=[(0, 1), (1, 0)])
    ax.set_title("100,000 emails through the k ≥ 2 rule", pad=26)
    ax.text(1.5, 0.86, "TYPE I", ha="center", fontsize=10, weight="bold", color=ORANGE)
    ax.text(0.5, 1.86, "TYPE II", ha="center", fontsize=10, weight="bold", color=ORANGE)

    panel(fig.add_subplot(gs[0, 1]), "The definitions", [
        "Type I (false positive): reject H₀",
        "although H₀ is true.",
        "Type II (false negative): keep H₀",
        "although H₁ is true.",
        "",
        r"$\alpha = \max\; P(\mathrm{reject}\ H_0 \mid H_0\ \mathrm{true})$",
        "",
        "Typical: α = 0.05, stricter α = 0.01.",
        "α = 0: never reject, every spam passes.",
        "α = 1: always reject, every ham is lost.",
    ], spacing=0.083)

    ax = fig.add_subplot(gs[0, 2])
    score = np.linspace(-4, 7, 700)
    f0, f1 = normal_pdf(score), normal_pdf(score, 3)
    thr = z_score(0.95)
    beta_05 = Phi(thr - 3)
    ax.plot(score, f0, color=BLUE, linewidth=2, label="score under H₀: N(0, 1)")
    ax.plot(score, f1, color=AQUA, linewidth=2, label="score under H₁: N(3, 1)")
    ax.fill_between(score, f0, where=score >= thr, color=ORANGE, alpha=0.8, linewidth=0)
    ax.fill_between(score, f1, where=score <= thr, color=AQUA, alpha=0.3, linewidth=0)
    ax.axvline(thr, color=INK2, linestyle="--", linewidth=1.3)
    ax.annotate("Type I: α = 0.05", xy=(2.0, 0.02), xytext=(3.7, 0.47), color=ORANGE, fontsize=10.5,
                weight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.annotate(f"Type II: {beta_05:.1%}", xy=(1.3, 0.04), xytext=(-4, 0.3), color=AQUA,
                fontsize=10.5, weight="bold", arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.text(thr + 0.1, 0.56, f"threshold {thr:.2f}", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("evidence score")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y")
    ax.set_title("One threshold cuts both curves")

    ax = fig.add_subplot(gs[1, 0])
    alphas = np.logspace(-4, np.log10(0.5), 200)
    betas = [Phi(z_score(1 - a) - 3) for a in alphas]
    ax.plot(alphas, betas, color=BLUE, linewidth=2.2)
    for a, offset, ha in [(0.05, (8, 6), "left"), (0.01, (8, 4), "left"), (0.001, (8, 4), "left")]:
        b = Phi(z_score(1 - a) - 3)
        ax.plot(a, b, "o", color=ORANGE, markersize=7, zorder=3)
        ax.annotate(f"α = {a:g}: Type II = {b:.1%}", (a, b), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=10)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("α, the Type I ceiling (log scale)")
    ax.set_ylabel("Type II probability")
    ax.grid(True)
    ax.set_title("The see-saw: less Type I, more Type II")

    ax = fig.add_subplot(gs[1, 1])
    ks = np.arange(1, 5)
    ax.bar(ks, tails, width=0.6, color=[ORANGE, BLUE, BLUE, BLUE], zorder=2)
    for k, v in zip(ks, tails):
        ax.text(k, v * 1.6, f"{v:.1e}", ha="center", fontsize=10)
    for a, label in [(0.05, "α = 0.05"), (1e-3, "α = 10⁻³"), (1e-6, "α = 10⁻⁶")]:
        ax.axhline(a, color=INK2, linestyle=":", linewidth=1.2)
        ax.text(4.62, a * 1.25, label, ha="right", fontsize=9.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1)
    ax.set_xlim(0.5, 4.7)
    ax.set_xticks(ks, [f"k ≥ {k}" for k in ks])
    ax.set_xlabel("rule: reject when count of trigger phrases ≥ k")
    ax.set_ylabel("P(Type I) = P(count ≥ k | ham)", labelpad=8)
    ax.grid(axis="y")
    ax.set_title("α picks the threshold k")

    panel(fig.add_subplot(gs[1, 2]), "Which error hurts more?", [
        "Email: losing a good message (Type I).",
        "That is why ham was made H₀.",
        "Trial (H₀ innocent): Type I is severe.",
        "Smoke alarm (H₀ no fire): Type II.",
        "",
        "Smallest k whose ham tail fits under α:",
        "α = 0.05 → k = 1,   α = 10⁻³ → k = 2,",
        "α = 10⁻⁶ → k = 3.",
        "",
        "You never learn which cell one email is in;",
        "only the long-run rates are under control.",
    ], spacing=0.083)
    save(fig, name="02_type_errors.png")


# ================================================================ notebook 03
def sheet_03_three_tails():
    x_obs = HEIGHTS.mean()
    assert round(x_obs, 3) == X_OBS
    se = SE_HEIGHTS
    rng = np.random.default_rng(70)          # the notebook's seed and draw order
    simulated = rng.normal(MU0, se, size=100_000)
    frac_beyond = (simulated >= x_obs).mean()
    assert f"{frac_beyond:.4f}" == "0.0333"
    z95, z975 = z_score(0.95), z_score(0.975)

    fig = new_figure("Hypotheses about μ, a test statistic X̄, and three kinds of tail",
                     "1970s mean height 66.7 in; ten 18-year-olds today average 68.442 in. "
                     "H₀: μ = 66.7, and H₁ points right, left, or both ways (σ = 3 assumed known).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(simulated, bins=80, density=True, color=BLUE_LIGHT, edgecolor="none",
            label="X̄ under H₀ (100,000 sims)")
    xs = np.linspace(MU0 - 4 * se, MU0 + 4 * se, 400)
    ax.plot(xs, normal_pdf(xs, MU0, se), color=BLUE, linewidth=1.8, label="N(66.7, 3²/10)")
    ax.axvline(x_obs, color=ORANGE, linewidth=2.5, label="observed x̄ = 68.442")
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("sample mean x̄ (inches)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title(f"Statistic vs observed: {frac_beyond:.2%} reach 68.442")

    sub = gs[0, 1:].subgridspec(1, 3, wspace=0.12)
    x_axis = np.linspace(MU0 - 4 * se, MU0 + 4 * se, 600)
    dens = normal_pdf(x_axis, MU0, se)
    setups = [("Right-tailed: μ > 66.7", x_axis >= MU0 + z95 * se, "reject for large x̄",
               [(69.4, "5%")]),
              ("Left-tailed: μ < 66.7", x_axis <= MU0 - z95 * se, "reject for small x̄",
               [(64.0, "5%")]),
              ("Two-tailed: μ ≠ 66.7", np.abs(x_axis - MU0) >= z975 * se,
               "reject for large |x̄ − 66.7|", [(63.9, "2.5%"), (69.5, "2.5%")])]
    for k, (title, region, caption, tags) in enumerate(setups):
        ax = fig.add_subplot(sub[0, k])
        tail_plot(ax, x_axis, dens, region)
        ax.fill_between(x_axis, dens, where=~region, color=BLUE_LIGHT, linewidth=0, zorder=0)
        ax.axvline(MU0, color=INK2, linestyle=":", linewidth=1)
        ax.text(MU0, 0.47, caption, ha="center", fontsize=9.5, color=INK2)
        for x_tag, tag in tags:
            ax.text(x_tag, 0.07, tag, ha="center", fontsize=10, weight="bold", color=ORANGE)
        ax.set_ylim(0, 0.52)
        ax.set_yticks([] if k else [0, 0.2, 0.4])
        ax.set_xlabel("x̄ (inches)")
        ax.set_title(title, fontsize=12)
        if k == 0:
            ax.set_ylabel("density under H₀")

    panel(fig.add_subplot(gs[1, 0]), "Writing the hypotheses", [
        "Always about the population parameter:",
        "✓  H₀: μ = 66.7",
        "✓  H₁: μ > 66.7",
        "✗  H₁: x̄ > 66.7   (x̄ is evidence)",
        "✗  H₁: μ > 68.442   (data in the claim)",
        "✗  H₀: S² = 9   (write σ² = 9)",
        "",
        "Test statistic: the random variable X̄.",
        "Observed statistic: its value, 68.442.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(x_axis, dens, color=INK2, linewidth=1.8)
    ax.fill_between(x_axis, dens, where=x_axis >= MU0 + z95 * se, color=ORANGE, alpha=0.75,
                    linewidth=0, label=f"right-tailed region (x̄ > {MU0 + z95 * se:.2f})")
    ax.fill_between(x_axis, dens, where=x_axis <= MU0 - z95 * se, color=AQUA, alpha=0.6,
                    linewidth=0, label=f"left-tailed region (x̄ < {MU0 - z95 * se:.2f})")
    ax.axvline(x_obs, color=BLUE, linewidth=2.5, label="observed 68.442")
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("x̄ (inches)")
    ax.set_ylabel("density under H₀")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Same evidence, different verdicts")

    panel(fig.add_subplot(gs[1, 2]), "Before testing, and the errors", [
        "Data: representative, randomized, and",
        "ideally n ≥ 30 (here n = 10).",
        "",
        "Right: Type I = claim an increase that",
        "isn't there; Type II = miss a real one.",
        "Left: the same for a decrease.",
        "Two-tailed: claim / miss any change.",
        "",
        "Pick the tail from the question,",
        "before looking at the data.",
    ], spacing=0.083)
    save(fig, name="03_three_tails.png")


# ================================================================ notebook 04
def sheet_04_p_values():
    se = SE_HEIGHTS
    p_right, p_two, p_left = (p_value_heights(X_OBS, "right"), p_value_heights(X_OBS, "two"),
                              p_value_heights(X_OBS_LEFT, "left"))
    z_obs = (X_OBS - MU0) / se
    z_left = (X_OBS_LEFT - MU0) / se
    assert (round(p_right, 4), round(p_two, 4), round(p_left, 4)) == (0.0332, 0.0663, 0.0049)
    assert round(z_obs, 3) == 1.836 and round(z_left, 2) == -2.58

    rng = np.random.default_rng(44)          # the notebook's seed and draw order
    null_xbars = rng.normal(MU0, se, size=1_000_000)
    sim = [(null_xbars >= X_OBS).mean(), (np.abs(null_xbars - MU0) > 1.742).mean(),
           (null_xbars <= X_OBS_LEFT).mean()]
    assert [round(v, 4) for v in sim] == [0.0332, 0.0661, 0.0048]

    fig = new_figure("The p-value: how surprising is the sample if H₀ is true?",
                     "p = P(statistic as extreme as observed or more, in H₁'s direction | H₀).  "
                     "Reject H₀ when p < α.  Heights: X̄ ~ N(66.7, 3²/10) under H₀.")
    gs = grid(fig, top=0.84, wspace=0.3)

    sub = gs[0, :2].subgridspec(1, 3, wspace=0.12)
    x_axis = np.linspace(MU0 - 4 * se, MU0 + 4 * se, 700)
    dens = normal_pdf(x_axis, MU0, se)
    cases = [("Right-tailed, x̄ = 68.442", x_axis >= X_OBS, [X_OBS], p_right, "reject"),
             ("Two-tailed, x̄ = 68.442", np.abs(x_axis - MU0) >= 1.742, [X_OBS, MU0 - 1.742],
              p_two, "don't reject"),
             ("Left-tailed, x̄ = 64.252", x_axis <= X_OBS_LEFT, [X_OBS_LEFT], p_left, "reject")]
    for k, (title, region, marks, p, verdict) in enumerate(cases):
        ax = fig.add_subplot(sub[0, k])
        tail_plot(ax, x_axis, dens, region)
        for m in marks:
            ax.axvline(m, color=ORANGE, linewidth=1.5)
        ax.axvline(MU0, color=INK2, linestyle=":", linewidth=1)
        ax.text(MU0, 0.47, f"p = {p:.4f}", ha="center", fontsize=12, weight="bold",
                color=ORANGE if p < 0.05 else INK2)
        ax.text(MU0, 0.43, f"→ {verdict}", ha="center", fontsize=10.5, color=INK2)
        ax.set_ylim(0, 0.52)
        ax.set_yticks([] if k else [0, 0.2, 0.4])
        ax.set_xlabel("x̄ (inches)")
        ax.set_title(title, fontsize=12)
        if k == 0:
            ax.set_ylabel("density under H₀")

    panel(fig.add_subplot(gs[0, 2]), "The three formulas", [
        r"right:  $P(\bar{X} > t_{obs} \mid \mu_0)$",
        r"left:    $P(\bar{X} < t_{obs} \mid \mu_0)$",
        r"two:    $P(|\bar{X} - \mu_0| > |t_{obs} - \mu_0|)$",
        "",
        "p < α  →  reject H₀",
        "p ≥ α  →  evidence too weak",
        "",
        "Two-tailed = 2 × right-tailed here,",
        "so the verdict flips: 0.0663 > 0.05.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    xb_axis = np.linspace(65.5, 70, 300)
    ax.plot(xb_axis, [p_value_heights(x, "right") for x in xb_axis], color=BLUE, linewidth=2.2,
            label="right-tailed p-value")
    threshold = crit_heights(0.05, "right")
    ax.axhline(0.05, color=INK2, linestyle=":", linewidth=1.3, label="α = 0.05")
    ax.axvline(threshold, color=MUTED, linestyle="--", linewidth=1.3,
               label=f"threshold {threshold:.2f}")
    ax.plot(X_OBS, p_right, "o", color=ORANGE, markersize=8, zorder=3,
            label=f"observed: p = {p_right:.4f}")
    ax.set_yscale("log")
    ax.set_xlabel("observed sample mean x̄ (inches)")
    ax.set_ylabel("p-value (log scale)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)
    ax.set_title("Every possible observation, priced")

    grouped_bars(fig.add_subplot(gs[1, 1]), ["right\nx̄ = 68.442", "two\nx̄ = 68.442", "left\nx̄ = 64.252"],
                 [("exact (normal tail)", [p_right, p_two, p_left], BLUE),
                  ("1,000,000 null sims", sim, AQUA)],
                 "A p-value is a tail area: just count", "", ylim=0.09,
                 label_fmt=lambda v: f"{v:.4f}")

    panel(fig.add_subplot(gs[1, 2]), "By hand, and in z units", [
        "SE = 3/√10 = 0.949",
        r"$z_{obs} = \dfrac{68.442 - 66.7}{0.949} = 1.836$",
        "",
        f"P(Z > 1.836) = {1 - Phi(z_obs):.4f}: same p, new ruler.",
        "",
        f"Left sample: z = {z_left:.2f}, p = {p_left:.4f}",
        "< 0.01 < 0.05: rejects at both levels.",
        "",
        "Standardizing changes the units, not the test.",
    ], spacing=0.085)
    save(fig, name="04_p_values.png")


# ================================================================ notebook 05
def sheet_05_critical_values():
    se = SE_HEIGHTS
    k05, k01 = crit_heights(0.05, "right"), crit_heights(0.01, "right")
    k_left = crit_heights(0.05, "left")
    lo, hi = crit_heights(0.05, "two")
    assert (round(k05, 2), round(k01, 2), round(k_left, 2)) == (68.26, 68.91, 65.14)
    assert (round(lo, 2), round(hi, 2)) == (64.84, 68.56)

    xbars = np.linspace(64, 71, 2001)
    p_values = np.array([p_value_heights(x, "right") for x in xbars])
    agree = ((p_values < 0.05) == (xbars > k05)).all()
    first_reject = xbars[(xbars > k05).argmax()]
    assert agree and round(first_reject, 3) == 68.263

    fig = new_figure("Critical values: the decision rule you can write before the data",
                     "k_α = the least extreme observation that still rejects (its p-value is exactly α). "
                     "Reject when the observed statistic passes k_α.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x_axis = np.linspace(MU0 - 4 * se, MU0 + 4 * se, 700)
    dens = normal_pdf(x_axis, MU0, se)
    ax.plot(x_axis, dens, color=INK2, linewidth=1.8)
    ax.fill_between(x_axis, dens, where=x_axis >= k05, color=ORANGE_LIGHT, linewidth=0,
                    label=f"α = 0.05: reject beyond {k05:.2f}")
    ax.fill_between(x_axis, dens, where=x_axis >= k01, color=ORANGE, linewidth=0,
                    label=f"α = 0.01: reject beyond {k01:.2f}")
    for k in (k05, k01):
        ax.axvline(k, color=ORANGE, linestyle="--", linewidth=1.3)
    ax.axvline(X_OBS, color=BLUE, linewidth=2.5, label="observed 68.442")
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("x̄ (inches)")
    ax.set_ylabel("density under H₀")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Stricter α, farther threshold")

    ax = fig.add_subplot(gs[0, 1])
    rows = [("right-tailed", [(k05, 71.5)]), ("left-tailed", [(62, k_left)]),
            ("two-tailed", [(62, lo), (hi, 71.5)])]
    for i, (name, zones) in enumerate(rows):
        y = 2 - i
        ax.plot([62, 71.5], [y, y], color=BLUE_LIGHT, linewidth=10, solid_capstyle="butt")
        for a, b in zones:
            ax.plot([a, b], [y, y], color=ORANGE, linewidth=10, solid_capstyle="butt")
            edge = a if b == 71.5 else b
            ax.text(edge, y + 0.22, f"{edge:.2f}", ha="center", fontsize=10, weight="bold")
    for x, label in [(X_OBS, "68.442"), (X_OBS_LEFT, "64.252")]:
        ax.axvline(x, color=INK, linewidth=1.2, linestyle=":")
        ax.text(x, -0.62, label, ha="center", fontsize=10)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.set_ylim(-0.8, 2.6)
    ax.set_xlim(62, 71.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("x̄ (inches); orange = reject H₀ (α = 0.05)")
    ax.set_title("The rules for all three tails")

    panel(fig.add_subplot(gs[0, 2]), "Critical values are quantiles", [
        "right:  quantile 1 − α of the H₀ curve",
        "left:    quantile α",
        "two:    quantiles α/2 and 1 − α/2",
        "",
        r"$k_\alpha = \mu_0 + z_{1-\alpha} \cdot \sigma/\sqrt{n}$",
        "",
        "k₀.₀₅ = 66.7 + 1.645 × 0.949 = 68.26",
        "k₀.₀₁ = 66.7 + 2.326 × 0.949 = 68.91",
        "68.442: rejects at 0.05, not at 0.01.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xbars, p_values, color=BLUE, linewidth=2.2, label="p-value of the observation")
    ax.axhline(0.05, color=INK2, linestyle=":", linewidth=1.3, label="α = 0.05")
    ax.axvline(k05, color=ORANGE, linestyle="--", linewidth=1.5, label=f"k₀.₀₅ = {k05:.2f}")
    ax.axvspan(k05, xbars[-1], color=ORANGE, alpha=0.1, linewidth=0)
    ax.text(69.8, 0.2, "reject zone:\np < α and x̄ > k", ha="center", fontsize=10, color=ORANGE)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(64, 71)
    ax.set_xlabel("observed sample mean x̄")
    ax.set_ylabel("right-tailed p-value (log scale)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True)
    ax.set_title("One rule, two dashboards")

    ax = fig.add_subplot(gs[1, 1])
    alphas = np.logspace(-3, np.log10(0.3), 200)
    ax.plot(alphas, [crit_heights(a, "right") for a in alphas], color=BLUE, linewidth=2.2,
            label="critical value k_α")
    ax.axhline(X_OBS, color=INK2, linestyle=":", linewidth=1.3, label="observed 68.442")
    p_obs = p_value_heights(X_OBS, "right")
    ax.plot(p_obs, X_OBS, "o", color=ORANGE, markersize=8, zorder=3)
    ax.annotate(f"α = p = {p_obs:.4f}", (p_obs, X_OBS), textcoords="offset points", xytext=(10, 10),
                fontsize=10.5, weight="bold", color=ORANGE)
    for a, k in [(0.05, k05), (0.01, k01)]:
        ax.plot(a, k, "o", color=BLUE, markersize=6, zorder=3)
        ax.annotate(f"{k:.2f}", (a, k), textcoords="offset points", xytext=(6, 4), fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("significance level α (log scale)")
    ax.set_ylabel("k_α (inches)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    ax.set_title("p = the smallest α that still rejects")

    panel(fig.add_subplot(gs[1, 2]), "Why bother with k_α?", [
        "1. Set in advance from the design alone:",
        "   n, the null value, and σ.",
        "",
        "2. Always agrees with the p-value rule:",
        f"   {len(xbars):,} observations, identical decisions;",
        f"   first rejecting x̄ = {first_reject:.3f}.",
        "",
        "3. A fixed rule makes Type II error",
        "   probabilities computable (next notebook).",
    ], spacing=0.085)
    save(fig, name="05_critical_values.png")


# ================================================================ notebook 06
def sheet_06_power():
    se = SE_HEIGHTS
    k = crit_heights(0.05, "right")
    beta70 = 1 - float(power_heights(70))
    pow68, pow70 = float(power_heights(68)), float(power_heights(70))
    assert round(beta70, 4) == 0.0334 and round(pow68, 3) == 0.392 and round(pow70, 3) == 0.967
    rng = np.random.default_rng(66)          # the notebook's seed and draw order
    sim = rng.normal(70, se, size=200_000)
    beta_sim = (sim <= 68.26).mean()
    assert round(beta_sim, 4) == 0.0337
    # smallest n with alpha = beta(68) = 0.05
    n_design = next(n for n in range(2, 500) if float(power_heights(68, n=n)) >= 0.95)
    assert n_design == 58

    fig = new_figure("Type II errors and power: β is a whole curve, not a number",
                     "Rule from notebook 05: reject iff x̄ > 68.26.  If the truth is μ, then "
                     "X̄ ~ N(μ, 3²/10):  β(μ) = P(X̄ ≤ 68.26 | μ),  power(μ) = 1 − β(μ).")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    x_axis = np.linspace(63, 74, 800)
    f_null, f_alt = normal_pdf(x_axis, MU0, se), normal_pdf(x_axis, 70, se)
    ax.plot(x_axis, f_null, color=BLUE, linewidth=2, label="under H₀: μ = 66.7")
    ax.plot(x_axis, f_alt, color=AQUA, linewidth=2, label="if truth is μ = 70")
    ax.fill_between(x_axis, f_null, where=x_axis >= k, color=ORANGE, alpha=0.8, linewidth=0)
    ax.fill_between(x_axis, f_alt, where=x_axis <= k, color=AQUA, alpha=0.6, linewidth=0)
    ax.axvline(k, color=INK2, linestyle="--", linewidth=1.3)
    ax.text(k, 0.5, f"k = {k:.2f}", ha="center", fontsize=10.5)
    ax.annotate("α = 0.05", xy=(68.6, 0.02), xytext=(69.3, 0.17), fontsize=10.5, color=ORANGE,
                weight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.annotate(f"β(70) = {beta70:.4f}", xy=(67.95, 0.006), xytext=(63.2, 0.17), fontsize=10.5,
                color=AQUA, weight="bold", arrowprops=dict(arrowstyle="-", color=AQUA))
    ax.set_ylim(0, 0.56)
    ax.set_xlabel("x̄ (inches)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("One threshold, two worlds")

    ax = fig.add_subplot(gs[0, 1])
    mu_axis = np.linspace(MU0, 71.5, 300)
    ax.plot(mu_axis, power_heights(mu_axis), color=BLUE, linewidth=2.5)
    ax.axhline(0.05, color=ORANGE, linestyle=":", linewidth=1.3)
    ax.text(66.8, 0.09, "starts at α = 0.05", fontsize=10, color=ORANGE)
    for mu_mark in (68.0, 70.0):
        p_mark = float(power_heights(mu_mark))
        ax.plot([mu_mark, mu_mark], [0, p_mark], color=MUTED, linestyle=":", linewidth=1)
        ax.plot([mu_mark, mu_mark], [p_mark, 1], color=AQUA, linewidth=3)
        ax.plot(mu_mark, p_mark, "o", color=INK, markersize=6, zorder=3)
        ax.annotate(f"power({mu_mark:.0f}) = {p_mark:.3f}", (mu_mark, p_mark),
                    textcoords="offset points", xytext=(8, -14), fontsize=10)
        ax.annotate(f"β = {1 - p_mark:.3f}", (mu_mark, (1 + p_mark) / 2), textcoords="offset points",
                    xytext=(-8, 0), ha="right", fontsize=10, color=AQUA, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("true population mean μ")
    ax.set_ylabel("power = P(reject H₀ | μ)")
    ax.grid(True)
    ax.set_title("The power curve")

    lines = ["`true μ    β       power`"]
    for mu in (67.0, 68.0, 69.0, 70.0, 71.0):
        p = float(power_heights(mu))
        lines.append(f"`{mu:5.1f}   {1 - p:6.3f}   {p:6.3f}`")
    lines += ["",
              r"$\beta(70) = \Phi\left(\dfrac{68.26 - 70}{0.949}\right) = \Phi(-1.834)$",
              "",
              "",
              f"Simulated β(70): {beta_sim:.4f} (200,000 draws).",
              "β depends on the design, never the data."]
    panel(fig.add_subplot(gs[0, 2]), "β and power by alternative", lines,
          spacing=0.078)

    ax = fig.add_subplot(gs[1, 0])
    for alpha, color in [(0.10, AQUA), (0.05, BLUE), (0.01, ORANGE)]:
        ax.plot(mu_axis, power_heights(mu_axis, alpha=alpha), color=color, linewidth=2.2,
                label=f"α = {alpha}: power(70) = {float(power_heights(70, alpha=alpha)):.3f}")
        ax.plot(MU0, alpha, "o", color=color, markersize=6)
    ax.axvline(70, color=MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("true population mean μ")
    ax.set_ylabel("power")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Fixed n: bigger α lifts the whole curve")

    ax = fig.add_subplot(gs[1, 1])
    for n, color in [(10, BLUE), (30, AQUA), (100, ORANGE)]:
        ax.plot(mu_axis, power_heights(mu_axis, n=n), color=color, linewidth=2.2,
                label=f"n = {n}: power(68) = {float(power_heights(68, n=n)):.3f}")
    ax.axvline(68, color=MUTED, linestyle=":", linewidth=1)
    ax.axhline(0.05, color=INK2, linestyle=":", linewidth=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("true population mean μ")
    ax.set_ylabel("power")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True)
    ax.set_title("Fixed α = 0.05: more data, more power")

    panel(fig.add_subplot(gs[1, 2]), "Reading the curve", [
        "Type I lives at one point, μ = 66.7;",
        "Type II at every μ > 66.7.",
        "",
        "Height at μ₀ is exactly α; the gap to 1",
        "is β. Far alternatives are easy catches,",
        "near ones nearly invisible.",
        "",
        "At fixed n, α and β trade off. Choose n",
        f"freely and both shrink: n = {n_design} gives",
        "α = 0.05 and β(68) ≤ 0.05.",
    ], spacing=0.083)
    save(fig, name="06_power.png")


# ================================================================ notebook 07
def sheet_07_steps():
    se = SE_HEIGHTS
    p_heights = p_value_heights(X_OBS, "right")
    coffee = np.array([249.1, 246.0, 250.3, 245.7, 248.9, 247.2, 251.0, 244.8,
                       249.6, 246.5, 248.0, 247.3, 250.1, 245.9, 246.8, 247.6])
    z_coffee = (coffee.mean() - 250.0) / (4.0 / sqrt(len(coffee)))
    p_coffee = 2 * (1 - Phi(abs(z_coffee)))
    assert round(p_heights, 4) == 0.0332 and round(coffee.mean(), 3) == 247.8
    assert round(z_coffee, 3) == -2.2 and round(p_coffee, 4) == 0.0278

    rng = np.random.default_rng(77)          # the notebook's seed and draw order
    n_sims = 100_000
    upper = np.vectorize(lambda z: 1 - Phi(z))
    p_h0 = upper((rng.normal(MU0, se, n_sims) - MU0) / se)
    p_h1 = upper((rng.normal(68.0, se, n_sims) - MU0) / se)
    keep_h0, keep_h1 = (p_h0 >= 0.05).mean(), (p_h1 >= 0.05).mean()
    assert f"{keep_h0:.1%}" == "95.0%" and f"{keep_h1:.1%}" == "60.5%"

    def beta68(alpha, n):
        se_n = SIGMA / sqrt(n)
        return Phi((MU0 + z_score(1 - alpha) * se_n - 68.0) / se_n)

    n_needed = 10
    while beta68(0.001, n_needed) > 0.10:
        n_needed += 1
    assert round(beta68(0.05, 10), 3) == 0.608 and round(beta68(0.001, 10), 3) == 0.957
    assert n_needed == 102

    lik_h0 = normal_pdf(X_OBS, MU0, se)
    lik_h1 = normal_pdf(X_OBS, 68.0, se)

    def posterior(prior_h0):
        return lik_h0 * prior_h0 / (lik_h0 * prior_h0 + lik_h1 * (1 - prior_h0))

    assert round(posterior(0.5), 3) == 0.171

    fig = new_figure("Four steps to a verdict, and two ways to misread it",
                     "State → design → compute → decide.  Then read it honestly: p is not P(H₀ true), "
                     "and 'fail to reject' is not 'H₀ is true'.")
    gs = grid(fig, top=0.84, wspace=0.3)

    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_subplot(gs[0, :2])
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    steps = [("1. STATE", ["H₀: μ = 66.7", "H₁: μ > 66.7"], ["H₀: μ = 250", "H₁: μ ≠ 250"]),
             ("2. DESIGN", ["statistic X̄, α = 0.05", "σ = 3, n = 10"],
              ["statistic X̄, α = 0.05", "σ = 4, n = 16"]),
             ("3. COMPUTE", [f"x̄ = {X_OBS}", f"z = {(X_OBS - MU0) / se:.3f}"],
              [f"x̄ = {coffee.mean():.1f}", f"z = {z_coffee:.1f}"]),
             ("4. DECIDE", [f"p = {p_heights:.4f} < 0.05", "reject: taller"],
              [f"p = {p_coffee:.4f} < 0.05", "reject: drifted"])]
    for k, (head, heights_lines, coffee_lines) in enumerate(steps):
        x0 = k + 0.04
        ax.add_patch(FancyBboxPatch((x0, 0.02), 0.84, 0.86, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    facecolor="#eaf2fc", edgecolor=BLUE, linewidth=1.4))
        ax.text(x0 + 0.42, 0.80, head, ha="center", va="center", fontsize=13, weight="bold",
                color=BLUE)
        ax.text(x0 + 0.42, 0.68, "heights", ha="center", fontsize=9.5, color=MUTED, style="italic")
        for j, line in enumerate(heights_lines):
            ax.text(x0 + 0.42, 0.60 - 0.085 * j, line, ha="center", va="center", fontsize=11)
        ax.text(x0 + 0.42, 0.33, "coffee machine", ha="center", fontsize=9.5, color=MUTED,
                style="italic")
        for j, line in enumerate(coffee_lines):
            ax.text(x0 + 0.42, 0.25 - 0.085 * j, line, ha="center", va="center", fontsize=11)
        if k < 3:
            ax.annotate("", xy=(k + 1.03, 0.45), xytext=(k + 0.89, 0.45),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    ax.text(2.0, 0.97, "Steps 1-2 are fixed before any data is seen", ha="center", va="center",
            fontsize=13, weight="bold")

    panel(fig.add_subplot(gs[0, 2]), "The error facts, kept fresh", [
        "Type I: reject H₀ although it is true.",
        "Type II: keep H₀ although H₁ is true.",
        "α = the maximum Type I probability.",
        "",
        "Entanglement at fixed n = 10, μ = 68:",
        f"α = 0.05   → β = {beta68(0.05, 10):.1%}",
        f"α = 0.001 → β = {beta68(0.001, 10):.1%}",
        f"α = 0.001 with β ≤ 10% needs n = {n_needed}.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 41)
    ax.hist(p_h0, bins=bins, density=True, color=BLUE, alpha=0.55,
            label=f"H₀ true (μ = 66.7): keep {keep_h0:.1%}")
    ax.hist(p_h1, bins=bins, density=True, color=ORANGE, alpha=0.55,
            label=f"H₀ false (μ = 68): keep {keep_h1:.1%}")
    ax.axvline(0.05, color=INK, linestyle="--", linewidth=1.3)
    ax.text(0.07, 10.6, "← reject", fontsize=10, color=INK2)
    ax.set_ylim(0, 15)
    ax.set_xlabel("p-value (100,000 simulated studies each)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    ax.set_title("Both worlds land in 'fail to reject'")

    ax = fig.add_subplot(gs[1, 1])
    priors = np.linspace(0.01, 0.99, 300)
    ax.plot(priors, posterior(priors), color=BLUE, linewidth=2.2,
            label="P(H₀ | x̄ = 68.442), H₀: μ = 66.7 vs μ = 68")
    ax.axhline(p_heights, color=ORANGE, linewidth=2, linestyle="--",
               label=f"p-value = {p_heights:.4f} (needs no prior)")
    ax.plot(0.5, posterior(0.5), "o", color=BLUE, markersize=8, zorder=3)
    ax.annotate(f"50/50 prior: {posterior(0.5):.3f}", (0.5, posterior(0.5)),
                textcoords="offset points", xytext=(10, -16), fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("prior probability of H₀")
    ax.set_ylabel("probability")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("p is not the probability H₀ is true")

    panel(fig.add_subplot(gs[1, 2]), "Sentence repair", [
        "✗  'p = 0.03: a 3% chance H₀ is true.'",
        "✓  'Under H₀, data this extreme turn",
        "     up only 3% of the time.'",
        "",
        "✗  'p = 0.4, so the machine is fine.'",
        "✓  'Not enough evidence it drifted.'",
        "",
        "✗  'We accept H₀.'",
        "✓  'We fail to reject H₀.'",
        "",
        "A non-rejection persuades only when",
        "the test had high power.",
    ], spacing=0.075)
    save(fig, name="07_steps_and_interpretation.png")


# ================================================================ notebook 08
def sheet_08_t_statistic():
    n = len(HEIGHTS)
    x_bar, s = HEIGHTS.mean(), HEIGHTS.std(ddof=1)
    t_obs = (x_bar - MU0) / (s / sqrt(n))
    z_obs = (x_bar - MU0) / SE_HEIGHTS
    t_crit, z_crit = t_score(0.95, n - 1), z_score(0.95)
    p_t, p_wrong = t_tail(t_obs, n - 1), 1 - Phi(t_obs)
    assert round(s, 3) == 3.113 and round(t_obs, 3) == 1.770 and round(z_obs, 3) == 1.836
    assert round(t_crit, 3) == 1.833 and round(p_t, 4) == 0.0553 and round(p_wrong, 4) == 0.0384

    rng = np.random.default_rng(88)          # the notebook's seed and draw order
    samples = rng.normal(MU0, SIGMA, size=(60_000, n))
    t_stats = (samples.mean(axis=1) - MU0) / (samples.std(axis=1, ddof=1) / sqrt(n))
    beyond3 = np.mean(np.abs(t_stats) > 3)
    assert round(beyond3, 4) == 0.0141
    realized = {}
    for n_ in (5, 10, 30):
        sims = rng.normal(0, 1, size=(100_000, n_))
        ts = sims.mean(axis=1) / (sims.std(axis=1, ddof=1) / sqrt(n_))
        realized[n_] = (ts > 1.645).mean()
    assert [f"{v:.1%}" for v in realized.values()] == ["8.7%", "6.7%", "5.5%"]

    fig = new_figure("The t-statistic: swap σ for s, and pay for it in the tails",
                     "T = (X̄ − μ) / (S/√n) follows Student's t with ν = n − 1 degrees of freedom: "
                     "heavier tails than N(0, 1), Gaussian-like by ν ≈ 30.")
    gs = grid(fig, top=0.84, wspace=0.3)

    t_axis = np.linspace(-5, 5, 600)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_axis, normal_pdf(t_axis), color=INK, linewidth=2.4, label="N(0, 1)")
    for df, color in [(1, ORANGE), (5, "#f2a07c"), (9, AQUA), (30, BLUE)]:
        ax.plot(t_axis, t_pdf(t_axis, df), color=color, linewidth=1.6, label=f"t, ν = {df}")
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("One parameter: ν = n − 1")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(t_stats, bins=160, range=(-5, 5), density=True, color=BLUE_LIGHT, edgecolor="none",
            label="60,000 simulated T, n = 10")
    ax.plot(t_axis, t_pdf(t_axis, n - 1), color=AQUA, linewidth=2, label="t, ν = 9")
    ax.plot(t_axis, normal_pdf(t_axis), color=INK, linewidth=1.4, linestyle="--", label="N(0, 1)")
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Simulated T follows t₉, not the normal")
    ax.text(4.9, 0.30, f"P(|T| > 3)\nsimulated  {beyond3:.4f}\nt₉            {2 * t_tail(3, 9):.4f}"
            f"\nnormal      {2 * (1 - Phi(3)):.4f}", ha="right", va="top", fontsize=9.5, color=INK2)

    panel(fig.add_subplot(gs[0, 2]), "What changed from z", [
        r"$Z = \dfrac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$   (σ known)",
        "",
        "",
        r"$T = \dfrac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$   (σ unknown)",
        "",
        "",
        r"$S = \sqrt{\frac{1}{n-1}\sum (X_i - \bar{X})^2}$",
        "",
        "",
        "S is random too: samples that underestimate",
        "σ inflate T, hence the heavier tails.",
        "ν depends on n only, never on μ or σ.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_axis, t_pdf(t_axis, 9), color=INK2, linewidth=1.8)
    ax.fill_between(t_axis, t_pdf(t_axis, 9), where=t_axis >= t_crit, color=ORANGE, alpha=0.8,
                    linewidth=0, label=f"reject: t > t₀.₉₅,₉ = {t_crit:.3f}")
    ax.axvline(z_crit, color=MUTED, linestyle="--", linewidth=1.4,
               label=f"z bar {z_crit:.3f} (too lenient)")
    ax.axvline(t_obs, color=BLUE, linewidth=2.2, label=f"observed t = {t_obs:.3f}")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("t")
    ax.set_ylabel("density under H₀: t₉")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title(f"Heights: p = {p_t:.4f}, fail to reject")

    bars(fig.add_subplot(gs[1, 1]), ["n = 5", "n = 10", "n = 30"], list(realized.values()),
         "z cutoff on a t statistic inflates α", "sample size", ylabel="actual Type I rate",
         ylim=0.11, label_fmt=lambda v: f"{v:.1%}", highlight=0)
    ax = fig.axes[-1]
    ax.axhline(0.05, color=INK, linestyle=":", linewidth=1.3, label="intended α = 5%")
    ax.legend(loc="upper right", fontsize=9.5)

    panel(fig.add_subplot(gs[1, 2]), "The heights, both ways", [
        f"x̄ = {x_bar:.3f},  s = {s:.3f},  ν = {n - 1}",
        f"t = 1.742 / ({s:.3f}/√10) = {t_obs:.3f}",
        f"z = 1.742 / (3/√10) = {z_obs:.3f}",
        "",
        "Nearly the same number; the curve differs:",
        f"t₉ tail: p = {p_t:.4f} ≥ 0.05 → keep H₀",
        f"normal tail would claim p = {p_wrong:.4f}",
        "",
        "Match the curve to the statistic.",
    ], spacing=0.085)
    save(fig, name="08_t_statistic.png")


# ================================================================ notebook 09
def sheet_09_t_tests():
    n = len(HEIGHTS)
    s = HEIGHTS.std(ddof=1)
    se_t = s / sqrt(n)
    t_r = (HEIGHTS.mean() - MU0) / se_t
    t_l = (X_OBS_LEFT - MU0) / se_t
    p_r, p_2, p_l = t_tail(t_r, n - 1), 2 * t_tail(abs(t_r), n - 1), 1 - t_tail(t_l, n - 1)
    assert (round(t_r, 3), round(t_l, 3)) == (1.770, -2.487)
    assert (round(p_r, 4), round(p_2, 4), round(p_l, 4)) == (0.0553, 0.1106, 0.0173)
    z_world = [p_value_heights(X_OBS, "right"), p_value_heights(X_OBS, "two"),
               p_value_heights(X_OBS_LEFT, "left")]
    t_world = [p_r, p_2, p_l]

    t_crit = 1.8331
    s_star = (X_OBS - MU0) * sqrt(n) / t_crit
    t30 = (HEIGHTS.mean() - MU0) / (s / sqrt(30))
    p30 = t_tail(t30, 29)
    assert round(s_star, 3) == 3.005 and round(t30, 3) == 3.065 and round(p30, 4) == 0.0023

    rng = np.random.default_rng(99)          # the notebook's seed and draw order
    true_samples = rng.normal(68.4, 3.0, size=(100_000, 10))
    xbars, svals = true_samples.mean(axis=1), true_samples.std(axis=1, ddof=1)
    power_z = ((xbars - MU0) / (3.0 / sqrt(10)) > z_score(0.95)).mean()
    power_t = ((xbars - MU0) / (svals / sqrt(10)) > t_crit).mean()
    assert f"{power_z:.1%}" == "55.8%" and f"{power_t:.1%}" == "50.4%"

    fig = new_figure("Three t-tests: same data, honest σ, and one verdict flips",
                     f"σ unknown → T = (X̄ − 66.7)/(s/√10) ~ t₉ under H₀, with s = {s:.3f}.  "
                     "Every curve now sits around 0, and every p-value grows.")
    gs = grid(fig, top=0.84, wspace=0.3)

    sub = gs[0, :2].subgridspec(1, 3, wspace=0.12)
    t_axis = np.linspace(-4.5, 4.5, 700)
    dens = t_pdf(t_axis, n - 1)
    cases = [(f"Right-tailed, t = {t_r:.2f}", t_axis >= t_r, [t_r], p_r, "don't reject"),
             (f"Two-tailed, |t| = {t_r:.2f}", np.abs(t_axis) >= t_r, [t_r, -t_r], p_2,
              "don't reject"),
             (f"Left-tailed, t = {t_l:.2f}", t_axis <= t_l, [t_l], p_l, "reject")]
    for k, (title, region, marks, p, verdict) in enumerate(cases):
        ax = fig.add_subplot(sub[0, k])
        tail_plot(ax, t_axis, dens, region)
        for m in marks:
            ax.axvline(m, color=ORANGE, linewidth=1.5)
        ax.axvline(0, color=INK2, linestyle=":", linewidth=1)
        ax.text(0, 0.49, f"p = {p:.4f}", ha="center", fontsize=12, weight="bold",
                color=ORANGE if p < 0.05 else INK2, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.text(0, 0.45, f"→ {verdict}", ha="center", fontsize=10.5, color=INK2, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
        ax.set_ylim(0, 0.54)
        ax.set_yticks([] if k else [0, 0.2, 0.4])
        ax.set_xlabel("t")
        ax.set_title(title, fontsize=12)
        if k == 0:
            ax.set_ylabel("density under H₀: t₉")

    panel(fig.add_subplot(gs[0, 2]), "The arithmetic", [
        f"s/√n = {s:.3f}/√10 = {se_t:.3f}",
        f"right:  t = 1.742 / {se_t:.3f} = {t_r:.3f}",
        f"two:    p = 2 · P(T > |t|) = {p_2:.4f}",
        f"left:   t = −2.448 / {se_t:.3f} = {t_l:.3f}",
        "",
        "Always take |t| for the two-tailed p:",
        "t = −1.77 gives the same p as +1.77.",
        "",
        f"Left still rejects at 0.05, no longer at 0.01.",
    ], spacing=0.088)

    ax = fig.add_subplot(gs[1, 0])
    grouped_bars(ax, ["right\nx̄ = 68.442", "two\nx̄ = 68.442", "left\nx̄ = 64.252"],
                 [("z-world (σ = 3 known)", z_world, BLUE), ("t-world (s estimated)", t_world, ORANGE)],
                 "Every p grows; one verdict flips", "", ylim=0.14,
                 label_fmt=lambda v: f"{v:.4f}")
    ax.set_ylabel("p-value")
    ax.axhline(0.05, color=INK, linestyle=":", linewidth=1.3)
    ax.text(2.45, 0.052, "α = 0.05", ha="right", va="bottom", fontsize=9.5, color=INK2)
    ax.text(0, 0.075, "FLIP", ha="center", fontsize=11, weight="bold", color=ORANGE)

    ax = fig.add_subplot(gs[1, 1])
    z_r = (X_OBS - MU0) / SE_HEIGHTS
    norm = normal_pdf(t_axis)
    ax.plot(t_axis, norm, color=BLUE, linewidth=1.8,
            label=f"z-world: N(0,1), z = {z_r:.3f}, p = {z_world[0]:.4f}")
    ax.plot(t_axis, dens, color=ORANGE, linewidth=1.8,
            label=f"t-world: t₉, t = {t_r:.3f}, p = {p_r:.4f}")
    ax.fill_between(t_axis, norm, where=t_axis >= z_r, color=BLUE, alpha=0.35, linewidth=0)
    ax.fill_between(t_axis, dens, where=t_axis >= t_r, color=ORANGE, alpha=0.35, linewidth=0)
    ax.axvline(z_r, color=BLUE, linestyle="--", linewidth=1.2)
    ax.axvline(t_r, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.set_xlim(-4, 4.5)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("standardized statistic")
    ax.set_ylabel("density under H₀")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title("Same evidence, fatter tail, opposite verdict")

    panel(fig.add_subplot(gs[1, 2]), "What would have saved it?", [
        f"A tighter sample: s ≤ {s_star:.3f}, i.e. the",
        "z-world's σ = 3 sat right on the edge.",
        "",
        f"More data: same x̄ and s at n = 30 gives",
        f"t = {t30:.3f} on t₂₉, p = {p30:.4f}.",
        "",
        f"Power against μ = 68.4 at n = 10:",
        f"z-test {power_z:.1%},  t-test {power_t:.1%}",
        f"→ honesty costs {power_z - power_t:.1%} of power.",
    ], spacing=0.085)
    save(fig, name="09_t_tests.png")


# ================================================================ notebook 10
def sheet_10_two_sample():
    us = HEIGHTS
    arg = np.array([64.75, 66.17, 71.61, 68.64, 61.82, 66.67, 64.84, 67.12, 61.92])
    t_obs, df, se, vx, vy = welch(us.mean(), us.std(ddof=1), len(us),
                                  arg.mean(), arg.std(ddof=1), len(arg))
    p_right = t_tail(t_obs, df)
    p_two = 2 * t_tail(abs(t_obs), df)
    p_left = 1 - p_right
    assert round(t_obs, 3) == 1.745 and round(df, 1) == 16.8 and round(se, 3) == 1.429
    assert round(p_right, 4) == 0.0496 and round(p_two, 4) == 0.0992
    t_crit = t_score(0.95, df)

    rng = np.random.default_rng(10)          # the notebook's seed and draw order
    trials = 60_000
    sim_x = rng.normal(67, 3.11, size=(trials, 10))
    sim_y = rng.normal(67, 3.11, size=(trials, 9))
    sim_t = (sim_x.mean(axis=1) - sim_y.mean(axis=1)) / np.sqrt(
        sim_x.var(axis=1, ddof=1) / 10 + sim_y.var(axis=1, ddof=1) / 9)
    type1 = (sim_t > 1.7396).mean()
    assert round(type1, 3) == 0.050
    jitters = [rng.normal(0, 0.04, size=len(us)), rng.normal(0, 0.04, size=len(arg))]

    fig = new_figure("The two-sample t-test: is the US mean height above Argentina's?",
                     "H₀: μ_US − μ_Arg = 0.  Statistic: (x̄ − ȳ) / √(s²ₓ/nₓ + s²ᵧ/nᵧ), "
                     "t-distributed with Welch degrees of freedom ν.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for i, (data, label, color) in enumerate([(us, "US", BLUE), (arg, "Argentina", ORANGE)]):
        ax.scatter(data, np.full(len(data), i) + jitters[i], s=60, color=color, alpha=0.8,
                   edgecolor=SURFACE, zorder=3)
        ax.plot([data.mean()] * 2, [i - 0.25, i + 0.25], color=color, linewidth=3)
        ax.text(data.mean(), i + 0.33, f"mean {data.mean():.3f}", ha="center", fontsize=10.5,
                color=color, weight="bold")
    ax.set_yticks([0, 1], [f"US\nn = {len(us)}", f"Arg.\nn = {len(arg)}"])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlabel("height (inches)")
    ax.grid(axis="x")
    ax.set_title(f"The data: a {us.mean() - arg.mean():.3f}-inch gap")

    panel(fig.add_subplot(gs[0, 1]), "The statistic and its ν", [
        f"x̄ = {us.mean():.3f}, sₓ = {us.std(ddof=1):.3f};  ȳ = {arg.mean():.3f}, sᵧ = {arg.std(ddof=1):.3f}",
        f"SE = √({vx:.3f} + {vy:.3f}) = {se:.3f}",
        f"t = {us.mean() - arg.mean():.3f} / {se:.3f} = {t_obs:.3f}",
        "",
        r"$\nu = \dfrac{(s_x^2/n_x + s_y^2/n_y)^2}{(s_x^2/n_x)^2/(n_x-1) + (s_y^2/n_y)^2/(n_y-1)}$",
        "",
        "",
        "",
        f"ν = {df:.1f}: between min(n) − 1 = 8 and",
        "nₓ + nᵧ − 2 = 17. It need not be a whole",
        "number; let software compute it.",
    ], spacing=0.08)

    ax = fig.add_subplot(gs[0, 2])
    t_axis = np.linspace(-4.5, 4.5, 600)
    dens = t_pdf(t_axis, df)
    ax.plot(t_axis, dens, color=INK2, linewidth=1.8)
    ax.fill_between(t_axis, dens, where=t_axis >= t_obs, color=ORANGE, alpha=0.8, linewidth=0,
                    label=f"right-tailed p = {p_right:.4f}")
    ax.fill_between(t_axis, dens, where=t_axis <= -t_obs, color=ORANGE_LIGHT, linewidth=0,
                    label=f"+ mirror: two-tailed p = {p_two:.4f}")
    ax.axvline(t_obs, color=ORANGE, linewidth=1.8)
    ax.axvline(t_crit, color=INK, linestyle="--", linewidth=1.1, label=f"α = 0.05 cutoff t = {t_crit:.3f}")
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("t")
    ax.set_ylabel(f"density of t with ν = {df:.1f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title(f"t_obs = {t_obs:.3f}: a squeaker")

    ax = fig.add_subplot(gs[1, 0])
    ax.hist(sim_t, bins=150, range=(-5, 5), density=True, color=BLUE_LIGHT, edgecolor="none",
            label="Welch statistics, H₀ true (60,000)")
    ax.plot(t_axis, t_pdf(t_axis, 16.8), color=ORANGE, linewidth=2, label="t density, ν = 16.8")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    ax.set_title(f"Welch's ν checks out: Type I rate {type1:.3f}")

    bars(fig.add_subplot(gs[1, 1]), ["right\nμ_US > μ_Arg", "two\nμ_US ≠ μ_Arg", "left\nμ_US < μ_Arg"],
         [p_right, p_two, p_left], "Three tails, three verdicts", "", ylabel="p-value", ylim=1.15,
         label_fmt=lambda v: f"{v:.4f}", highlight=0)
    ax = fig.axes[-1]
    ax.axhline(0.05, color=INK, linestyle=":", linewidth=1.3)
    ax.text(0.5, 0.45, "dotted line: α = 0.05", ha="center", fontsize=10, color=INK2)

    panel(fig.add_subplot(gs[1, 2]), "Assumptions and verdict", [
        "1. distinct individuals in the groups",
        "2. independent measurements",
        "3. normal populations",
        "",
        r"$\bar{X} - \bar{Y} \sim N(\mu_x - \mu_y,\ \sigma_x^2/n_x + \sigma_y^2/n_y)$",
        "",
        f"Right-tailed p = {p_right:.4f} < 0.05: reject,",
        "US taller, by half a thousandth.",
        f"Two-tailed p = {p_two:.4f}: don't reject.",
    ], spacing=0.085)
    save(fig, name="10_two_sample_t_test.png")


# ================================================================ notebook 11
def sheet_11_paired():
    before = np.array([86.6, 82.5, 83.0, 80.3, 77.6, 82.3, 88.4, 83.3, 91.7, 83.6])
    after = np.array([83.7, 83.5, 81.5, 79.6, 76.8, 81.1, 89.0, 82.3, 91.3, 79.6])
    d = before - after
    n = len(d)
    s_d = d.std(ddof=1)
    t_obs = d.mean() / (s_d / sqrt(n))
    p_right = t_tail(t_obs, n - 1)
    p_two = 2 * p_right
    assert round(d.mean(), 2) == 1.09 and round(s_d, 3) == 1.486 and round(t_obs, 3) == 2.320
    assert round(p_right, 4) == 0.0227 and round(p_two, 4) == 0.0455
    t_ind, df_ind, se_ind, _, _ = welch(before.mean(), before.std(ddof=1), n,
                                        after.mean(), after.std(ddof=1), n)
    p_ind = t_tail(t_ind, df_ind)
    assert round(se_ind, 3) == 1.886 and round(t_ind, 3) == 0.578 and round(p_ind, 4) == 0.2852
    se_pair = s_d / sqrt(n)
    rng = np.random.default_rng(21)          # the notebook's seed: its only draw is this jitter
    jitter = rng.normal(0, 0.03, size=n)

    fig = new_figure("The paired t-test: subtract within pairs, then run a one-sample test",
                     "Ten people weighed before and after a training program.  d = before − after;  "
                     "H₀: μ_D = 0 vs H₁: μ_D > 0;  T = D̄ / (S_D/√n) ~ t with n − 1 = 9 df.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    lost = before > after
    for b, a, l in zip(before, after, lost):
        ax.plot([0, 1], [b, a], "o-", color=BLUE if l else ORANGE, linewidth=1.8, markersize=6,
                alpha=0.85)
    ax.set_xticks([0, 1], ["before", "after"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("weight (kg)")
    ax.grid(axis="y")
    ax.set_title(f"{lost.sum()} lost weight (blue), {n - lost.sum()} gained (orange)")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(d, jitter, s=80, color=[BLUE if v > 0 else ORANGE for v in d], edgecolor=SURFACE,
               zorder=3)
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvline(d.mean(), color=BLUE, linestyle="--", linewidth=2)
    ax.text(d.mean() + 0.08, 0.2, f"d̄ = {d.mean():.2f} kg", fontsize=11, weight="bold", color=BLUE)
    ax.set_ylim(-0.25, 0.25)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("difference dᵢ = before − after (kg)")
    ax.grid(axis="x")
    ax.set_title("The reduced dataset: one sample of ten")

    panel(fig.add_subplot(gs[0, 2]), "The arithmetic", [
        "d = " + ", ".join(f"{v:.1f}" for v in d[:5]) + ",",
        "      " + ", ".join(f"{v:.1f}" for v in d[5:]),
        "",
        f"d̄ = {d.mean():.2f},   s_d = {s_d:.3f},   n = {n}",
        f"s_d/√n = {se_pair:.3f}",
        r"$t_{obs} = \dfrac{1.09}{0.470} = 2.320$",
        "",
        "",
        f"right-tailed p = {p_right:.4f} < 0.05: reject",
        f"two-tailed p = {p_two:.4f}: also rejects",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    t_axis = np.linspace(-4.5, 4.5, 600)
    dens = t_pdf(t_axis, n - 1)
    tail_plot(ax, t_axis, dens, t_axis >= t_obs)
    ax.axvline(t_obs, color=ORANGE, linewidth=1.8)
    ax.text(t_obs + 0.15, 0.12, f"p = {p_right:.4f}", fontsize=11, weight="bold", color=ORANGE)
    ax.text(t_obs + 0.15, 0.09, f"t = {t_obs:.3f}", fontsize=10.5, color=INK2)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("t")
    ax.set_ylabel("density of t with 9 df")
    ax.grid(True)
    ax.set_title("The right tail beyond t_obs")

    ax = fig.add_subplot(gs[1, 1])
    bars(ax, [f"paired\nt = {t_obs:.3f}, p = {p_right:.4f}",
              f"as independent groups\nt = {t_ind:.3f}, p = {p_ind:.4f}"],
         [se_pair, se_ind], "Same 20 numbers, two standard errors", "",
         ylabel="standard error of the mean difference (kg)", ylim=2.3,
         label_fmt=lambda v: f"{v:.3f}", highlight=0)

    panel(fig.add_subplot(gs[1, 2]), "The punchline", [
        "Before and after are the same people,",
        "so the two-sample test's independence",
        "assumption fails by design.",
        "",
        "Differences dᵢ turn the problem into",
        "one sample: the paired t-test IS the",
        "one-sample t-test on the dᵢ.",
        "",
        "Pairing removes the person-to-person",
        "spread from the SE: 0.470 vs 1.886.",
    ], spacing=0.083)
    save(fig, name="11_paired_t_test.png")


# ================================================================ notebook 12
def sheet_12_ab_testing():
    t_obs, df, se_t, v1, v2 = welch(50, 10, 80, 55, 15, 20)
    p1 = 1 - t_tail(t_obs, df)
    assert round(t_obs, 3) == -1.414 and round(df, 2) == 23.39 and round(p1, 3) == 0.085
    x, n_a, y, n_b = 20, 80, 8, 20
    p_pool = (x + y) / (n_a + n_b)
    se_z = sqrt(p_pool * (1 - p_pool) * (n_a + n_b) / (n_a * n_b))
    z_obs = (x / n_a - y / n_b) / se_z
    p2 = Phi(z_obs)
    assert round(z_obs, 3) == -1.336 and round(p2, 3) == 0.091 and round(se_z, 4) == 0.1122

    rng = np.random.default_rng(12)          # the notebook's seed and draw order
    trials = 100_000
    x_sim = rng.binomial(80, 0.28, trials)
    y_sim = rng.binomial(20, 0.28, trials)
    pp = (x_sim + y_sim) / 100
    z_sim = (x_sim / 80 - y_sim / 20) / np.sqrt(pp * (1 - pp) * 100 / 1600)
    type1 = (z_sim < z_score(0.05)).mean()
    assert round(type1, 3) == 0.053

    fig = new_figure("A/B testing: random split, pick a metric, then the matching test",
                     "80 customers see design A, 20 see the new design B.  Both metrics favor B; "
                     "neither difference clears α = 0.05 (left-tailed: H₁ says B is better).")
    gs = grid(fig, top=0.84, wspace=0.3)

    panel(fig.add_subplot(gs[0, 0]), "The methodology", [
        "1. Propose two variations (A, B).",
        "2. Randomly split customers; send the",
        "    smaller share to the untested B.",
        "3. Choose the metric: a mean? a rate?",
        "4. Measure, then apply the test that",
        "    fits the metric:",
        "    mean purchase → Welch t-test",
        "    conversion rate → two-proportion z-test",
    ], spacing=0.09)

    ax = fig.add_subplot(gs[0, 1])
    sems = [10 / sqrt(80), 15 / sqrt(20)]
    ax.bar([0, 1], [50, 55], yerr=[1.96 * s for s in sems], color=[BLUE, ORANGE], width=0.55,
           capsize=10, error_kw=dict(ecolor=INK, linewidth=1.4), zorder=2)
    for i, (m, s) in enumerate(zip([50, 55], sems)):
        ax.text(i + 0.32, m, f"{m} ± {1.96 * s:.1f}", va="center", fontsize=10.5)
    ax.set_xticks([0, 1], ["A (n = 80, s = 10)", "B (n = 20, s = 15)"])
    ax.set_ylim(40, 65)
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylabel("mean purchase (dollars), ±1.96 SE")
    ax.grid(axis="y", zorder=0)
    ax.set_title(f"Experiment 1: purchases, p = {p1:.3f}")

    ax = fig.add_subplot(gs[0, 2])
    rates = [x / n_a, y / n_b]
    rate_se = [sqrt(r * (1 - r) / n) for r, n in zip(rates, [n_a, n_b])]
    ax.bar([0, 1], rates, yerr=[1.96 * s for s in rate_se], color=[BLUE, ORANGE], width=0.55,
           capsize=10, error_kw=dict(ecolor=INK, linewidth=1.4), zorder=2)
    for i, (r, s) in enumerate(zip(rates, rate_se)):
        ax.text(i + 0.32, r, f"{r:.0%} ± {1.96 * s:.0%}", va="center", fontsize=10.5)
    ax.set_xticks([0, 1], ["A (20 of 80)", "B (8 of 20)"])
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylabel("conversion rate, ±1.96 SE")
    ax.grid(axis="y", zorder=0)
    ax.set_title(f"Experiment 2: conversions, p = {p2:.3f}")

    ax = fig.add_subplot(gs[1, 0])
    t_axis = np.linspace(-4.5, 4.5, 600)
    tail_plot(ax, t_axis, t_pdf(t_axis, df), t_axis <= t_obs)
    ax.axvline(t_obs, color=ORANGE, linewidth=1.8)
    ax.axvline(t_score(0.05, df) if False else -t_score(0.95, df), color=INK, linestyle="--",
               linewidth=1.1)
    ax.text(t_obs - 0.15, 0.3, f"t = {t_obs:.3f}\np = {p1:.3f}", ha="right", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.text(-t_score(0.95, df) - 0.12, 0.42, f"cutoff {-t_score(0.95, df):.3f}", ha="right",
            fontsize=9.5, color=INK2)
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("t")
    ax.set_ylabel(f"density of t with ν = {df:.2f}")
    ax.grid(True)
    ax.set_title("Welch t-test: left tail, don't reject")

    ax = fig.add_subplot(gs[1, 1])
    z_axis = np.linspace(-4, 4, 400)
    ax.hist(z_sim, bins=120, range=(-4, 4), density=True, color=BLUE_LIGHT, edgecolor="none",
            label="pooled z, H₀ true (100,000 sims)")
    ax.plot(z_axis, normal_pdf(z_axis), color=INK2, linewidth=1.8, label="N(0, 1)")
    ax.fill_between(z_axis, normal_pdf(z_axis), where=z_axis <= z_obs, color=ORANGE, alpha=0.8,
                    linewidth=0)
    ax.axvline(z_obs, color=ORANGE, linewidth=1.8)
    ax.text(z_obs - 0.15, 0.3, f"z = {z_obs:.3f}\np = {p2:.3f}", ha="right", fontsize=10.5,
            color=ORANGE, weight="bold")
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("z")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    ax.set_title(f"Two-proportion z-test: simulated Type I = {type1:.3f}")

    panel(fig.add_subplot(gs[1, 2]), "The two statistics", [
        r"$t = \dfrac{50 - 55}{\sqrt{10^2/80 + 15^2/20}} = \dfrac{-5}{3.536} = -1.414$",
        "",
        "",
        "",
        f"pooled p̂ = (20 + 8)/(80 + 20) = {p_pool:.2f}",
        "",
        r"$z = \dfrac{0.25 - 0.40}{\sqrt{0.28 \cdot 0.72 \cdot 100/1600}} = -1.336$",
        "",
        "",
        "Both p > 0.05: B trends better, but the",
        "small B group (20) can't prove it yet.",
    ], spacing=0.07)
    save(fig, name="12_ab_testing.png")


if __name__ == "__main__":
    sheet_02_type_errors()
    sheet_03_three_tails()
    sheet_04_p_values()
    sheet_05_critical_values()
    sheet_06_power()
    sheet_07_steps()
    sheet_08_t_statistic()
    sheet_09_t_tests()
    sheet_10_two_sample()
    sheet_11_paired()
    sheet_12_ab_testing()
