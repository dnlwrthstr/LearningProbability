#!/usr/bin/env python3
"""Regenerate the illustrated-solution images for lesson 03 (multiple variables).

Every number drawn here is computed from the same data the notebooks use:
the children table (notebooks 04 and 05), two fair dice, and the call-center
sample, which is rebuilt by running notebook 03's own setup cells.
Images are written to coursera/03_probability_distribution_with_multiple_variables/img/.

Run from anywhere:  python scripts/make_lesson03_solution_images.py
"""
import contextlib
import io
import json
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from viz_style import (REPO, SURFACE, INK, INK2, MUTED, GRID, BLUE, ORANGE, AQUA, SEQ,
                       new_figure, grid, frac_label, heat, bars, grouped_bars, panel,
                       trapezoid, density_image)
from viz_style import save as _save

LESSON = REPO / "coursera" / "03_probability_distribution_with_multiple_variables"
save = partial(_save, folder=LESSON / "img")

# ---------------------------------------------------------------- data
AGES = [7, 8, 9, 10]
HEIGHTS = [45, 46, 47, 48, 49, 50]
CHILDREN = np.array([[0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.3, 0.1],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.1]])
P_AGE = CHILDREN.sum(axis=1)
P_HEIGHT = CHILDREN.sum(axis=0)
FACES = list(range(1, 7))
SUMS = list(range(2, 13))


def load_call_center():
    """Run notebook 03's code cells up to the exercises; return (big_wait, big_rate)."""
    nb = json.loads((LESSON / "03_continuous_joint_distribution.ipynb").read_text())
    namespace = {}
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if source.startswith("# Exercise"):
            break
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source.replace("plt.show()", "plt.close('all')"), "nb03", "exec"), namespace)
    plt.close("all")
    return namespace["big_wait"], namespace["big_rate"]



# ================================================================ notebook 03 (correct originals, restyled)
def fig_03_ex3():
    grid_1d = np.linspace(0, 1, 1001)
    xg, yg = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    volume = trapezoid(trapezoid(xg * yg, grid_1d, axis=1), grid_1d)
    c = 1 / volume
    corner = np.where((xg <= 0.5) & (yg <= 0.5), c * xg * yg, 0)
    p_corner = trapezoid(trapezoid(corner, grid_1d, axis=1), grid_1d)
    assert abs(c - 4) < 1e-6 and abs(p_corner - 1 / 16) < 2e-3

    fig = new_figure("Exercise 3: normalizing f(x, y) = c·xy on the unit square",
                     "Find c so the total volume is 1, then compute P(X ≤ ½, Y ≤ ½). Is it ¼?")
    gs = grid(fig, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    density_image(fig, ax, c * xg * yg, "Normalized density f(x, y) = 4xy", vmax=4)
    ax.add_patch(Rectangle((0, 0), 0.5, 0.5, fill=False, edgecolor=ORANGE, linewidth=3))
    ax.text(0.25, 0.25, "X ≤ ½, Y ≤ ½", ha="center", va="center", fontsize=10.5, weight="bold",
            color=INK, bbox=dict(facecolor="white", edgecolor=ORANGE, boxstyle="round,pad=0.3"))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = fig.add_subplot(gs[0, 1])
    for y_fixed, color in [(1.0, BLUE), (0.5, ORANGE), (0.25, AQUA)]:
        ax.plot(grid_1d, c * grid_1d * y_fixed, color=color, linewidth=2, label=f"y = {y_fixed:g}")
    ax.set_title("Cross-sections: f(x, y) for fixed y")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, y)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.2)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=10)

    panel(fig.add_subplot(gs[0, 2]), "Step 1: find c", [
        r"$\int_0^1 \int_0^1 xy\ dx\,dy = \int_0^1 \dfrac{y}{2}\ dy = \dfrac{1}{4}$",
        "",
        "",
        r"$c \cdot \dfrac{1}{4} = 1 \quad \Rightarrow \quad c = 4$",
        "",
        "",
        r"So $f_{XY}(x, y) = 4xy$ for $0 \leq x, y \leq 1$.",
    ], spacing=0.1)

    ax = fig.add_subplot(gs[1, 0])
    bars(ax, ["share of area", "probability"], [0.25, p_corner],
         "Corner X ≤ ½, Y ≤ ½", "", ylim=0.35, label_fmt=lambda v: frac_label(v, 16), highlight=1)

    panel(fig.add_subplot(gs[1, 1]), "Step 2: the corner", [
        r"$P(X \leq \frac{1}{2},\ Y \leq \frac{1}{2}) = \int_0^{1/2}\!\int_0^{1/2} 4xy\ dx\,dy$",
        "",
        r"$= 4 \cdot \dfrac{(1/2)^2}{2} \cdot \dfrac{(1/2)^2}{2} = \dfrac{1}{16}$",
        "",
        "",
        "Numerical check (trapezoid rule):",
        f"`integral of xy = {volume:.4f}  ->  c = {c:.1f}`",
        f"`P(corner)      = {p_corner:.4f}  (exact 0.0625)`",
    ], spacing=0.095)

    panel(fig.add_subplot(gs[1, 2]), "Why not ¼?", [
        "The corner covers ¼ of the area, but the",
        "density is lowest there: it rises toward",
        "(1, 1). Low density over ¼ of the area",
        "holds only 1/16 of the volume.",
        "",
        "Only a flat (uniform) density turns",
        "share of area into probability.",
    ], spacing=0.09)
    save(fig, name="03_ex3_normalizing.png")


def fig_03_ex6(big_wait, big_rate):
    rng = np.random.default_rng(5)
    x_pos = np.clip(rng.normal(5, 2.6, 100_000), 0, 10)
    y_pos = np.clip(0.8 * x_pos + rng.normal(1, 1.4, 100_000), 0, 10)
    datasets = [("Call center", big_wait, big_rate, "X = waiting time", "Y = rating"),
                ("Constructed: Y = 0.8X + noise", x_pos, y_pos, "X", "Y")]

    def cov_corr(x, y):
        cov = np.mean((x - x.mean()) * (y - y.mean()))
        return cov, cov / (x.std() * y.std())

    stats = [cov_corr(x, y) for _, x, y, _, _ in datasets]

    fig = new_figure("Exercise 6: positive vs negative dependence",
                     "The direction of the high-density ridge shows the sign of the relationship between X and Y.")
    gs = grid(fig, wspace=0.32)
    sample = np.random.default_rng(0).choice(100_000, size=4000, replace=False)
    for col, (name, x, y, xlabel, ylabel) in enumerate(datasets):
        density, _, _ = np.histogram2d(x, y, bins=40, range=[[0, 10], [0, 10]], density=True)
        ax = fig.add_subplot(gs[0, col])
        density_image(fig, ax, density, f"{name}: joint density", vmax=np.percentile(density, 99),
                      extent=(0, 10, 0, 10), label="density (capped at 99th pct)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax = fig.add_subplot(gs[1, col])
        ax.scatter(x[sample], y[sample], s=5, color=BLUE, alpha=0.3, linewidths=0)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(f"{name}: 4,000 sampled points", fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    panel(fig.add_subplot(gs[0, 2]), "Reading the ridges", [
        "Call center: the mass sits top-left and",
        "bottom-right. Long waits go with low",
        "ratings → negative relationship.",
        "",
        "Constructed: the ridge rises from",
        "bottom-left to top-right. Large X goes",
        "with large Y → positive relationship.",
    ], spacing=0.095)
    panel(fig.add_subplot(gs[1, 2]), "Preview: covariance", [
        "Computed from all 100,000 points:",
        "",
        f"Call center:   Cov = {stats[0][0]:.2f},  corr = {stats[0][1]:.2f}",
        f"Constructed:  Cov = {stats[1][0]:.2f},  corr = {stats[1][1]:.2f}",
        "",
        "Downward ridge → Cov < 0.",
        "Upward ridge → Cov > 0.",
        "Covariance puts a number on this next.",
    ], spacing=0.09)
    save(fig, name="03_ex6_joint_distribution_neg_pos.png")


# ================================================================ notebook 04 (correct originals, restyled)
def fig_04_ex3():
    counts = np.zeros((6, 6))
    for a in FACES:
        for b in FACES:
            counts[max(a, b) - 1, a - 1] += 1
    marginal = counts.sum(axis=1) / 36
    formula = np.array([(2 * m - 1) / 36 for m in FACES])
    assert np.allclose(marginal, formula)
    labels = [[f"{int(v)}/36" if v else "0" for v in row] for row in counts]

    fig = new_figure("Exercise 3: joint of the first die and the maximum",
                     "A = first die, M = max(A, B) of two fair dice. Sum each row to get the marginal of M.")
    gs = grid(fig)
    heat(fig.add_subplot(gs[0, :2]), counts / 36, FACES, FACES, "Joint P(M = m, A = a)",
         "M = maximum", "A = first die", labels=labels, vmax=6 / 36 * 1.2)
    panel(fig.add_subplot(gs[0, 2]), "Reading the table", [
        "If a < m, the second die must show m:",
        "1 outcome → 1/36.",
        "",
        "If a = m, the second die can be 1, …, m:",
        "m outcomes → m/36 (the diagonal).",
        "",
        "If a > m: impossible, since the maximum",
        "is at least the first die → 0.",
    ], spacing=0.095)
    grouped_bars(fig.add_subplot(gs[1, :2]), FACES,
                 [("from the table (row sums)", marginal, BLUE), ("formula (2m − 1)/36", formula, ORANGE)],
                 "Marginal P(M = m): table vs formula", "M = maximum", ylim=0.4,
                 label_fmt=lambda v: f"{round(v * 36)}/36")
    panel(fig.add_subplot(gs[1, 2]), "The direct formula", [
        "M ≤ m means both dice are at most m:",
        r"$P(M \leq m) = \dfrac{m^2}{36}$",
        "",
        "",
        r"$P(M = m) = \dfrac{m^2 - (m-1)^2}{36} = \dfrac{2m - 1}{36}$",
        "",
        "",
        "Row sums 1, 3, 5, 7, 9, 11 (in 36ths)",
        "match the formula, and they add up to 36.",
    ], spacing=0.085)
    save(fig, name="04_ex3_marginal_distribution.png")


def fig_04_ex4():
    grid_1d = np.linspace(0, 1, 501)
    xg, yg = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    cases = [("Uniform f = 1", np.ones_like(xg)), ("f = 4xy", 4 * xg * yg)]
    marginals = {name: (trapezoid(f, grid_1d, axis=1), trapezoid(f, grid_1d, axis=0)) for name, f in cases}
    checks = {name: [fx[i] for i in (125, 250, 375)] for name, (fx, _) in marginals.items()}
    assert np.allclose(checks["f = 4xy"], [0.5, 1.0, 1.5], atol=1e-3)

    fig = new_figure("Exercise 4: marginals of a joint density on the unit square",
                     "Integrate out one variable to get each marginal, then check whether the joint factors.")
    gs = grid(fig, wspace=0.32)
    for col, (name, f) in enumerate(cases):
        ax = fig.add_subplot(gs[0, col])
        density_image(fig, ax, f, f"{name}: joint density", vmax=4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fx, fy = marginals[name]
        ax = fig.add_subplot(gs[1, col])
        ax.plot(grid_1d, fx, color=BLUE, linewidth=2.5, label=r"$f_X(x)$")
        ax.plot(grid_1d, fy, color=ORANGE, linewidth=2, linestyle="--", label=r"$f_Y(y)$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.2)
        ax.grid(True)
        ax.set_xlabel("x (or y)")
        ax.set_ylabel("marginal density")
        ax.set_title(f"{name}: marginals " + ("(both flat at 1)" if col == 0 else "(both 2x)"))
        ax.legend(loc="upper left", fontsize=10)

    panel(fig.add_subplot(gs[0, 2]), "Integrate out the other variable", [
        r"Uniform:  $f_X(x) = \int_0^1 1\ dy = 1$",
        "",
        r"4xy:  $f_X(x) = \int_0^1 4xy\ dy = 2x$",
        "",
        r"          $f_Y(y) = \int_0^1 4xy\ dx = 2y$",
        "",
        "",
        "Numerical marginal f_X at x = 0.25, 0.5, 0.75:",
        "`uniform: " + ", ".join(f"{v:.3f}" for v in checks["Uniform f = 1"]) + "`",
        "`4xy:     " + ", ".join(f"{v:.3f}" for v in checks["f = 4xy"]) + "`",
    ], spacing=0.1)
    panel(fig.add_subplot(gs[1, 2]), "Independent?", [
        r"Uniform:  $f_X(x)\,f_Y(y) = 1 \cdot 1 = 1 = f_{XY}$",
        "",
        r"4xy:  $f_X(x)\,f_Y(y) = 2x \cdot 2y = 4xy = f_{XY}$",
        "",
        "Both joints factor into their marginals,",
        "so in both cases X and Y are independent,",
        "even though 4xy is far from flat.",
    ], spacing=0.1)
    save(fig, name="04_ex4_marginals_unit_squares.png")


def fig_04_ex6():
    joint3 = np.full((6, 6, 6), 1 / 216)
    two_dice = joint3.sum(axis=2)
    one_die = two_dice.sum(axis=1)
    assert np.allclose(two_dice, 1 / 36) and np.allclose(one_die, 1 / 6)

    fig = new_figure("Exercise 6: marginalizing a three-variable joint",
                     "Three fair dice: sum out X₃, then X₂, one variable at a time.")
    gs = grid(fig, wspace=0.32)
    heat(fig.add_subplot(gs[0, 0]), joint3[:, :, 0], FACES, FACES,
         "One layer of P(X₁, X₂, X₃): 1/216", "X₁", "X₂", labels=[["1/216"] * 6] * 6,
         vmax=1 / 216 * 2.5, fontsize=9)
    heat(fig.add_subplot(gs[0, 1]), two_dice, FACES, FACES, "Sum over X₃: P(X₁, X₂) = 1/36",
         "X₁", "X₂", labels=[[frac_label(v) for v in row] for row in two_dice], vmax=1 / 36 * 2.5)
    bars(fig.add_subplot(gs[0, 2]), FACES, one_die, "Sum over X₂: P(X₁) = 1/6", "X₁", ylim=0.25)

    panel(fig.add_subplot(gs[1, :2]), "Step by step", [
        r"The joint is a 6 × 6 × 6 cube: six layers ($X_3$ = 1, …, 6) like the left table, every cell 1/216.",
        "",
        r"Sum out $X_3$:  $P(x_1, x_2) = \sum_{x_3=1}^{6} \dfrac{1}{216} = \dfrac{6}{216} = \dfrac{1}{36}$",
        "",
        "",
        "",
        r"Sum out $X_2$:  $P(x_1) = \sum_{x_2=1}^{6} \dfrac{1}{36} = \dfrac{1}{6}$",
        "",
        "",
        "",
        "`shape (6, 6, 6)  --sum axis 2-->  (6, 6)  --sum axis 1-->  (6,)`",
        f"`all cells 1/36? {np.allclose(two_dice, 1 / 36)}     P(X1) = 1/6 each? {np.allclose(one_die, 1 / 6)}`",
    ], spacing=0.075)
    panel(fig.add_subplot(gs[1, 2]), "Key takeaways", [
        "To remove a variable, sum its",
        "probabilities over all its values.",
        "",
        "Each sum removes one dimension.",
        "",
        "Chaining sums walks a joint down to",
        "any subset of its variables, in any order.",
    ], spacing=0.09)
    save(fig, name="04_ex6_3_variable_joint.png")


# ================================================================ notebook 03
def fig_03_ex4(big_wait, big_rate):
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(big_rate)
    sample = rng.choice(len(big_wait), size=4000, replace=False)

    def summary(x, y):
        cov = np.mean((x - x.mean()) * (y - y.mean()))
        return [x.mean(), y.mean(), x.var(), y.var(), cov, cov / (x.std() * y.std())]

    original, mixed = summary(big_wait, big_rate), summary(big_wait, shuffled)

    fig = new_figure("Exercise 4: same marginals, different joint",
                     "Shuffling Y breaks the pairing but leaves both histograms, means and variances unchanged. "
                     "Scatter plots show 4,000 of the 100,000 customers.")
    gs = grid(fig, top=0.84)
    for col, (y, title) in enumerate([(big_rate, "Original: two corners"),
                                      (shuffled, "Y shuffled: four corners")]):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(big_wait[sample], y[sample], s=5, color=BLUE, alpha=0.3, linewidths=0)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X = waiting time (min)")
        ax.set_ylabel("Y = rating")
        ax.grid(True)

    hist_gs = gs[0, 2].subgridspec(2, 1, hspace=0.9)
    for k, (values, name) in enumerate([(big_wait, "X (waiting time)"), (big_rate, "Y (rating)")]):
        ax = fig.add_subplot(hist_gs[k])
        ax.hist(values, bins=40, range=(0, 10), color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"Marginal of {name}", fontsize=12)
        ax.set_ylabel("customers")
        ax.set_xlim(0, 10)
        ax.grid(axis="y")
    ax.set_xlabel("identical before and after the shuffle", color=MUTED, fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    names = ["E[X]", "E[Y]", "Var(X)", "Var(Y)", "Cov(X, Y) *", "Correlation *"]
    cells = [[n, f"{a:.3f}", f"{b:.3f}"] for n, a, b in zip(names, original, mixed)]
    cells[-2][2] = cells[-1][2] = "≈ 0"
    table = ax.table(cellText=cells, colLabels=["", "original", "shuffled"], colWidths=[0.4, 0.3, 0.3],
                     cellLoc="center", bbox=[0.0, 0.02, 1.0, 0.9])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0 or c == 0:
            cell.set_text_props(weight="bold", color=INK)
        if r >= 5:
            cell.set_facecolor("#fdf0ea")
    ax.set_title("Summary numbers", pad=4)
    ax.text(0.5, -0.04, "* joint summaries, a preview of the covariance notebook",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)

    panel(fig.add_subplot(gs[1, 1:]), "What this shows", [
        "Means and variances are computed one variable at a time, so they",
        "cannot tell the two-corner data from the four-corner data.",
        "",
        "Original: short waits go with high ratings, long waits with low ratings.",
        f"That is a negative relationship: Cov = {original[4]:.2f}, correlation = {original[5]:.2f}.",
        "",
        "Shuffled: the pairing is random, so Cov ≈ 0 and correlation ≈ 0.",
        "",
        "Same marginals  ≠  same joint distribution.",
        "To see the relationship you need a joint summary (covariance, correlation,",
        "the joint table or density, or the conditional distributions).",
        "",
        "The spikes at 0 and 10 in the histograms come from the notebook clipping values to [0, 10].",
    ], spacing=0.075)
    save(fig, name="03_ex4_marginals_vs_joint_distribution.png")


def fig_03_ex5(big_wait, big_rate):
    density, x_edges, y_edges = np.histogram2d(big_wait, big_rate, bins=60,
                                               range=[[0, 10], [0, 10]], density=True)
    cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    total = (density * cell_area).sum()
    region = (x_centers <= 3)[:, None] & (y_centers >= 7)[None, :]
    p_grid = (density[region] * cell_area).sum()
    p_count = np.mean((big_wait <= 3) & (big_rate >= 7))

    fig = new_figure("Exercise 5: probability = volume under the empirical density",
                     "60 × 60 histogram of 100,000 call-center customers, with the region X ≤ 3, Y ≥ 7 highlighted.")
    gs = grid(fig, rows=2, cols=2, width_ratios=[1.25, 1], left=0.06, wspace=0.18, top=0.86)

    ax = fig.add_subplot(gs[:, 0])
    cap = np.percentile(density, 99)
    image = ax.imshow(density.T, origin="lower", extent=[0, 10, 0, 10], cmap=SEQ, aspect="equal",
                      vmin=0, vmax=cap)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("estimated density (scale capped at the 99th percentile)")
    colorbar.outline.set_visible(False)
    ax.axvline(3, color=INK2, linestyle="--", linewidth=1.2)
    ax.axhline(7, color=INK2, linestyle="--", linewidth=1.2)
    ax.add_patch(Rectangle((0, 7), 3, 3, fill=False, edgecolor=ORANGE, linewidth=3.5))
    ax.text(1.5, 6.6, f"X ≤ 3, Y ≥ 7\nP ≈ {p_grid:.4f}", ha="center", va="top", fontsize=12,
            weight="bold", color=INK, bbox=dict(facecolor="white", edgecolor=ORANGE, boxstyle="round,pad=0.35"))
    ax.set_xlabel("X = waiting time (min)")
    ax.set_ylabel("Y = rating")
    ax.set_title("Empirical joint density (np.histogram2d, density=True)")
    for spine in ax.spines.values():
        spine.set_visible(False)

    panel(fig.add_subplot(gs[0, 1]), "From density to probability", [
        r"Each cell is $\Delta x \times \Delta y = \dfrac{10}{60} \times \dfrac{10}{60} = \dfrac{1}{36} \approx 0.0278$",
        "",
        r"Probability of a cell  =  density $\times$ area  =  $f_{ij}\,\Delta x\,\Delta y$",
        "",
        r"$P(X \leq 3,\ Y \geq 7) \approx \sum_{\mathrm{cells\ in\ region}} f_{ij}\,\Delta x\,\Delta y$",
        "",
        "",
        "A cell counts as inside when its center satisfies",
        r"$x \leq 3$ and $y \geq 7$ (the orange box).",
        "",
        "Dark strips along the edges: the notebook clips",
        "values to [0, 10], so the overflow piles up there.",
    ], spacing=0.095)

    panel(fig.add_subplot(gs[1, 1]), "Results (computed from the data)", [
        f"`total volume  = sum(density * area) = {total:.4f}`",
        f"`corner volume = {p_grid:.4f}`",
        f"`direct count  = {p_count:.4f}`",
        "",
        "The total volume is 1, so this is a valid density.",
        "The corner holds about 23% of customers: it sits on",
        "the top-left cluster (short wait, high rating).",
        "",
        "Density × area is exactly each cell's share of points,",
        "so integrating and counting agree.",
    ], spacing=0.085)
    save(fig, name="03_ex5_joint_density.png")


# ================================================================ notebook 04
def fig_04_ex5():
    product = np.outer(P_AGE, P_HEIGHT)
    diff = np.abs(product - CHILDREN)
    i_max, j_max = np.unravel_index(diff.argmax(), diff.shape)
    assert (AGES[i_max], HEIGHTS[j_max]) == (9, 49)

    fig = new_figure("Exercise 5: two different joints, same marginals",
                     "The independence product table keeps both marginals of the children table but loses the dependence.")
    gs = grid(fig)
    heat(fig.add_subplot(gs[0, 0]), CHILDREN, AGES, HEIGHTS, "True joint P(age, height)",
         "age", "height (inches)", vmax=0.3)
    heat(fig.add_subplot(gs[0, 1]), product, AGES, HEIGHTS, "Product P(age) · P(height)",
         "age", "height (inches)", vmax=0.3)
    heat(fig.add_subplot(gs[0, 2]), diff, AGES, HEIGHTS, "|product − true|",
         "age", "height (inches)", vmax=0.3, highlight_cells=[(i_max, j_max)])

    bars(fig.add_subplot(gs[1, 0]), AGES, P_AGE, "Marginal of age (same for both)", "age",
         ylim=0.55, label_fmt=lambda v: f"{v:.1f}")
    bars(fig.add_subplot(gs[1, 1]), HEIGHTS, P_HEIGHT, "Marginal of height (same for both)",
         "height (inches)", ylim=0.55, label_fmt=lambda v: f"{v:.1f}")
    panel(fig.add_subplot(gs[1, 2]), "What it shows", [
        "Row sums and column sums of both",
        "tables are identical, by construction.",
        "",
        f"Largest gap at (age, height) = (9, 49):",
        f"product {product[i_max, j_max]:.2f}  vs  true {CHILDREN[i_max, j_max]:.2f}",
        "",
        "From the product table you would conclude",
        "a 7-year-old is as likely to be 50 inches",
        "as anyone. The true joint says P(7, 50) = 0:",
        "age and height move together.",
        "",
        "Marginals alone cannot restore that.",
    ], spacing=0.075)
    save(fig, name="04_ex5_different_joints_same_marginals.png")


# ================================================================ notebook 05
def fig_05_ex1():
    row = AGES.index(7)
    slice_ = CHILDREN[row]
    conditional = slice_ / slice_.sum()

    fig = new_figure("Exercise 1: P(Y | X = 7) for the children table",
                     "Height distribution among the 7-year-olds: slice the row, sum it, normalize.")
    gs = grid(fig)
    heat(fig.add_subplot(gs[0, :2]), CHILDREN, AGES, HEIGHTS, "Joint P(X = age, Y = height)",
         "X = age", "Y = height (inches)", vmax=0.3, highlight_rows=[row])
    panel(fig.add_subplot(gs[0, 2]), "Three steps", [
        "1. Slice: take the row X = 7",
        "    (0.1, 0.2, 0, 0, 0, 0)",
        "",
        r"2. Sum: $P(X = 7) = 0.1 + 0.2 = 0.3$",
        "",
        "3. Normalize: divide the row by 0.3",
        r"    $P(Y = y \mid X = 7) = \dfrac{P(X = 7,\ Y = y)}{P(X = 7)}$",
    ], spacing=0.1)
    bars(fig.add_subplot(gs[1, 0]), HEIGHTS, slice_, "Slice: P(X = 7, Y = y)",
         "height (inches)", ylim=1.0, label_fmt=lambda v: f"{v:.1f}")
    bars(fig.add_subplot(gs[1, 1]), HEIGHTS, conditional, "Normalized: P(Y = y | X = 7)",
         "height (inches)", color=AQUA, ylim=1.0)
    panel(fig.add_subplot(gs[1, 2]), "Result", [
        r"$P(Y \mid X = 7) = \left(\dfrac{1}{3},\ \dfrac{2}{3},\ 0,\ 0,\ 0,\ 0\right)$",
        "",
        "",
        "One third of the 7-year-olds are 45 inches,",
        "two thirds are 46 inches.",
        "",
        "The slice sums to 0.3; the conditional sums",
        "to 1. Same shape, rescaled.",
    ], spacing=0.1)
    save(fig, name="05_ex1_compute_conditional_probability.png")


def fig_05_ex2():
    col = HEIGHTS.index(50)
    column = CHILDREN[:, col]
    conditional = column / column.sum()

    fig = new_figure("Exercise 2: the reverse conditional P(X | Y = 50)",
                     "Age distribution among the 50-inch children: slice a column this time.")
    gs = grid(fig)
    heat(fig.add_subplot(gs[0, :2]), CHILDREN, AGES, HEIGHTS, "Joint P(X = age, Y = height)",
         "X = age", "Y = height (inches)", vmax=0.3, highlight_cols=[col])
    panel(fig.add_subplot(gs[0, 2]), "Three steps", [
        "1. Slice: take the column Y = 50",
        "    ages 7, 8, 9, 10:  (0, 0, 0.1, 0.1)",
        "",
        r"2. Sum: $P(Y = 50) = 0.1 + 0.1 = 0.2$",
        "",
        "3. Normalize: divide the column by 0.2",
        r"    $P(X = x \mid Y = 50) = \dfrac{P(X = x,\ Y = 50)}{P(Y = 50)}$",
    ], spacing=0.1)
    bars(fig.add_subplot(gs[1, 0]), AGES, column, "Slice: P(X = x, Y = 50)", "age",
         ylim=1.0, label_fmt=lambda v: f"{v:.1f}")
    bars(fig.add_subplot(gs[1, 1]), AGES, conditional, "Normalized: P(X = x | Y = 50)", "age",
         color=AQUA, ylim=1.0)
    panel(fig.add_subplot(gs[1, 2]), "Result", [
        r"$P(X \mid Y = 50) = (0,\ 0,\ 0.5,\ 0.5)$",
        "",
        "Half of the 50-inch children are 9,",
        "half are 10.",
        "",
        "Exercise 1 fixed an age and sliced a row;",
        "this fixes a height and slices a column.",
        r"In general $P(X \mid Y) \neq P(Y \mid X)$.",
    ], spacing=0.1)
    save(fig, name="05_ex2_compute_conditional_probability.png")


def fig_05_ex3():
    conditionals = CHILDREN / P_AGE[:, None]
    rebuilt = P_AGE[:, None] * conditionals
    assert np.allclose(conditionals.sum(axis=1), 1) and np.allclose(rebuilt, CHILDREN)
    cond_labels = [[frac_label(v, 12) for v in row] for row in conditionals]

    fig = new_figure("Exercise 3: conditionals sum to 1, and rebuild the joint",
                     "Divide each row by P(X = x), check every row sums to 1, then multiply back.")
    gs = grid(fig, wspace=0.32)
    heat(fig.add_subplot(gs[0, 0]), CHILDREN, AGES, HEIGHTS, "Joint P(X, Y)",
         "X = age", "Y = height (inches)", vmax=0.3)
    ax = fig.add_subplot(gs[0, 1])
    heat(ax, conditionals, AGES, HEIGHTS, "Conditionals P(Y | X = x)", "X = age",
         "Y = height (inches)", labels=cond_labels, vmax=1.0)
    for i, total in enumerate(conditionals.sum(axis=1)):
        ax.text(len(HEIGHTS) + 0.25, i + 0.5, f"Σ = {total:g}", va="center", fontsize=11,
                color=INK, weight="bold")
    heat(fig.add_subplot(gs[0, 2]), rebuilt, AGES, HEIGHTS, "Rebuilt P(X) · P(Y | X)",
         "X = age", "Y = height (inches)", vmax=0.3)

    panel(fig.add_subplot(gs[1, :2]), "How it works", [
        r"Divide each row by its sum $P(X = x)$ = (0.3, 0.2, 0.4, 0.1):",
        r"    $P(Y = y \mid X = x) = \dfrac{P(X = x,\ Y = y)}{P(X = x)}$",
        "",
        "",
        "Each conditional row now sums to 1 (middle table). Example: for X = 7,",
        "(0.1, 0.2) / 0.3 = (1/3, 2/3).",
        "",
        r"Run the multiplication rule backwards:  $P(X = x,\ Y = y) = P(X = x) \cdot P(Y = y \mid X = x)$",
        "",
        f"`max |rebuilt − joint| = {np.abs(rebuilt - CHILDREN).max():g}   → exact match`",
    ], spacing=0.1)
    panel(fig.add_subplot(gs[1, 2]), "Key takeaways", [
        "Every conditional is a valid",
        "distribution: it sums to 1.",
        "",
        "P(X) together with P(Y | X)",
        "holds exactly the same",
        "information as the joint.",
        "",
        "The same works the other way:",
        "P(Y) and P(X | Y).",
    ], spacing=0.085)
    save(fig, name="05_ex3_conditionals_sum_to_1.png")


def fig_05_ex4():
    joint = np.array([[1 / 36 if 1 <= s - x <= 6 else 0.0 for s in SUMS] for x in FACES])
    p_sum = joint.sum(axis=0)
    col = SUMS.index(7)
    conditional_7 = joint[:, col] / joint.sum(axis=1)
    assert np.allclose(conditional_7, 1 / 6)
    labels = [[frac_label(v) for v in row] for row in joint]

    fig = new_figure("Exercise 4: P(S = 7 | X = x) for every x",
                     "X = first die, S = sum of two fair dice. Why is the answer the same for every x?")
    gs = grid(fig, width_ratios=[1.2, 1, 1])
    heat(fig.add_subplot(gs[0, :2]), joint, FACES, SUMS, "Joint P(X = x, S = s): a band six cells wide",
         "X = first die", "S = sum", labels=labels, vmax=1 / 36 * 1.6, highlight_cols=[col],
         fontsize=10)
    panel(fig.add_subplot(gs[0, 2]), "Why 1/6 every time", [
        "Row x has 1/36 in columns x+1, …, x+6:",
        "the second die can show any of 6 faces.",
        "",
        "The S = 7 column is 1/36 in every row,",
        "because for each x exactly one second",
        "face (7 − x) makes the sum 7.",
        "",
        r"$P(S = 7 \mid X = x) = \dfrac{1/36}{6/36} = \dfrac{1}{6}$",
    ], spacing=0.095)
    bars(fig.add_subplot(gs[1, 0]), SUMS, p_sum, "Marginal P(S = s)", "S = sum",
         highlight=col, ylim=0.25)
    ax = fig.add_subplot(gs[1, 1])
    bars(ax, FACES, conditional_7, "P(S = 7 | X = x)", "X = first die", color=AQUA, ylim=0.35)
    ax.axhline(p_sum[col], color=INK2, linestyle="--", linewidth=1.2)
    ax.text(-0.4, 0.31, "dashed line: P(S = 7) = 1/6", ha="left", fontsize=10, color=INK2)
    panel(fig.add_subplot(gs[1, 2]), "The other direction", [
        "The six pairs that sum to 7:",
        "(1,6) (2,5) (3,4) (4,3) (5,2) (6,1)",
        "",
        r"$P(X = x \mid S = 7) = \dfrac{1/36}{6/36} = \dfrac{1}{6}$",
        "",
        "",
        "Knowing X never changes the chance of",
        "S = 7, so knowing S = 7 cannot change the",
        "distribution of X. Not true for other sums:",
        r"$P(S = 2 \mid X = 6) = 0$.",
    ], spacing=0.09)
    save(fig, name="05_ex4_symmetry_sum_given_first_die.png")


def fig_05_ex5():
    dice_conditionals = np.full((6, 6), 1 / 6)
    child_conditionals = CHILDREN / P_AGE[:, None]
    row9 = AGES.index(9)

    fig = new_figure("Exercise 5: independence via conditionals",
                     "Two dice: every P(Y | X = x) equals the marginal. Children: the conditionals change with age.")
    gs = grid(fig, wspace=0.3)
    heat(fig.add_subplot(gs[0, 0]), dice_conditionals, FACES, FACES,
         "Dice: P(Y | X = x), every row the same", "X = first die", "Y = second die",
         labels=[["1/6"] * 6] * 6, vmax=1.0)
    heat(fig.add_subplot(gs[0, 1]), child_conditionals, AGES, HEIGHTS,
         "Children: P(Y | X = x), rows differ", "X = age", "Y = height (inches)",
         labels=[[frac_label(v, 12) for v in row] for row in child_conditionals], vmax=1.0,
         highlight_rows=[row9])
    panel(fig.add_subplot(gs[0, 2]), "Why the two tests agree", [
        "Definition of conditional probability:",
        r"$P(Y = y \mid X = x) = \dfrac{P(X = x,\ Y = y)}{P_X(x)}$",
        "",
        "",
        r"If $P(Y = y \mid X = x) = P_Y(y)$ for every x,",
        r"multiply by $P_X(x)$:  $P(x, y) = P_X(x)\,P_Y(y)$.",
        "",
        r"If $P(x, y) = P_X(x)\,P_Y(y)$, divide by $P_X(x)$",
        "and you get the conditional back.",
    ], spacing=0.1)
    grouped_bars(fig.add_subplot(gs[1, 0]), FACES,
                 [("P(Y | X = 4)", dice_conditionals[3], BLUE), ("marginal P(Y)", np.full(6, 1 / 6), ORANGE)],
                 "Dice: conditional = marginal", "Y = second die", ylim=0.4)
    grouped_bars(fig.add_subplot(gs[1, 1]), HEIGHTS,
                 [("P(Y | X = 9)", child_conditionals[row9], BLUE), ("marginal P(Y)", P_HEIGHT, ORANGE)],
                 "Children: conditional ≠ marginal", "Y = height (inches)", ylim=1.0,
                 label_fmt=lambda v: f"{v:g}")
    panel(fig.add_subplot(gs[1, 2]), "Result", [
        "Dice: P(Y | X = x) = 1/6 for every x and y",
        "→ independent.",
        "",
        "Children, X = 9:",
        "P(Y | X = 9) = (0, 0, 0, 0, 0.75, 0.25)",
        "marginal P(Y) = (0.1, 0.2, 0.2, 0, 0.3, 0.2)",
        "→ not independent.",
        "",
        "Independence = conditioning changes nothing.",
    ], spacing=0.09)
    save(fig, name="05_ex5_independence_via_conditionals.png")


if __name__ == "__main__":
    wait, rate = load_call_center()
    fig_03_ex3()
    fig_03_ex4(wait, rate)
    fig_03_ex5(wait, rate)
    fig_03_ex6(wait, rate)
    fig_04_ex3()
    fig_04_ex4()
    fig_04_ex5()
    fig_04_ex6()
    fig_05_ex1()
    fig_05_ex2()
    fig_05_ex3()
    fig_05_ex4()
    fig_05_ex5()
