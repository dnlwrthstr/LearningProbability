#!/usr/bin/env python3
"""Regenerate the illustrated-solution images in img/ from the notebooks' own data.

Every number drawn here is computed from the same data the notebooks use:
the children table (notebooks 04 and 05), two fair dice, and the call-center
sample, which is rebuilt by running notebook 03's own setup cells.

Run from anywhere:  python make_solution_images.py
"""
import contextlib
import io
import json
import textwrap
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

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
    nb = json.loads((HERE / "03_continuous_joint_distribution.ipynb").read_text())
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


# ---------------------------------------------------------------- helpers
def new_figure(title, subtitle):
    fig = plt.figure(figsize=(16, 10))
    fig.text(0.5, 0.965, title, ha="center", va="center", fontsize=22, weight="bold")
    fig.text(0.5, 0.925, subtitle, ha="center", va="center", fontsize=13, color=INK2)
    return fig


def grid(fig, rows=2, cols=3, **kw):
    params = dict(top=0.79, bottom=0.06, left=0.05, right=0.975, hspace=0.5, wspace=0.28)
    params.update(kw)
    return fig.add_gridspec(rows, cols, **params)


def save(fig, name):
    fig.savefig(IMG / name, dpi=100)
    plt.close(fig)
    print("wrote", name)


def frac_label(value, max_den=36):
    if abs(value) < 1e-12:
        return "0"
    f = Fraction(value).limit_denominator(max_den)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def heat(ax, matrix, rows, cols, title, row_name, col_name, labels=None, vmax=None,
         highlight_rows=(), highlight_cols=(), highlight_cells=(), fontsize=11):
    """Annotated heatmap table: one-hue sequential fill, 2px surface gaps, orange highlights."""
    matrix = np.asarray(matrix, float)
    vmax = vmax or matrix.max() or 1.0
    n_rows, n_cols = matrix.shape
    ax.pcolormesh(np.arange(n_cols + 1), np.arange(n_rows + 1), matrix, cmap=SEQ,
                  vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, cols)
    ax.set_yticks(np.arange(n_rows) + 0.5, rows)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(col_name, labelpad=6)
    ax.set_ylabel(row_name)
    ax.set_title(title, pad=34)
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            text = labels[i][j] if labels is not None else (f"{value:.2f}" if value else "0")
            if abs(value) < 1e-12:
                color = MUTED
            elif value / vmax > 0.55:
                color = "white"
            else:
                color = INK
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fontsize, color=color)
    box = dict(fill=False, edgecolor=ORANGE, linewidth=3, clip_on=False)
    for i in highlight_rows:
        ax.add_patch(Rectangle((0, i), n_cols, 1, **box))
    for j in highlight_cols:
        ax.add_patch(Rectangle((j, 0), 1, n_rows, **box))
    for i, j in highlight_cells:
        ax.add_patch(Rectangle((j, i), 1, 1, **box))


def bars(ax, x, y, title, xlabel, ylabel="probability", color=BLUE, ylim=None,
         label_fmt=frac_label, highlight=None):
    positions = np.arange(len(x))
    colors = [ORANGE if highlight is not None and k == highlight else color for k in range(len(x))]
    ax.bar(positions, y, width=0.62, color=colors, zorder=2)
    ax.set_xticks(positions, x)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    top = ylim if ylim else max(y) * 1.25
    ax.set_ylim(0, top)
    for p, v in zip(positions, y):
        if v > 0:
            ax.text(p, v + top * 0.02, label_fmt(v), ha="center", va="bottom", fontsize=10, color=INK)


def grouped_bars(ax, x, series, title, xlabel, ylim=1.0, label_fmt=lambda v: frac_label(v, 12)):
    positions = np.arange(len(x))
    width = 0.38
    for k, (name, values, color) in enumerate(series):
        offset = (k - 0.5) * width
        ax.bar(positions + offset, values, width=width - 0.04, color=color, label=name, zorder=2)
        for p, v in zip(positions, values):
            if v > 0:
                ax.text(p + offset, v + ylim * 0.02, label_fmt(v), ha="center",
                        va="bottom", fontsize=9, color=INK)
    ax.set_xticks(positions, x)
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper left", fontsize=10)


def panel(ax, heading, lines, fontsize=12.5, top=1.0, spacing=0.085):
    """Text card: a bold heading, then one entry per line (mathtext allowed)."""
    ax.axis("off")
    ax.add_patch(Rectangle((-0.03, -0.02), 1.06, top + 0.04, transform=ax.transAxes,
                           facecolor="#f2f1ed", edgecolor="none", zorder=0, clip_on=False))
    ax.text(0.0, top - 0.04, heading, transform=ax.transAxes, fontsize=14.5,
            weight="bold", va="top")
    y = top - 0.04 - 0.11
    for line in lines:
        if line == "":
            y -= spacing * 0.5
            continue
        is_code = line.startswith("`")
        text = line.strip("`")
        ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=INK if is_code else INK2,
                fontsize=fontsize - (1 if is_code else 0),
                family="monospace" if is_code else None)
        y -= spacing


def wrap(text, width=48):
    return textwrap.wrap(text, width)


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
    save(fig, "03_ex4_marginals_vs_joint_distribution.png")


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
    save(fig, "03_ex5_joint_density.png")


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
    save(fig, "04_ex5_different_joints_same_marginals.png")


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
    save(fig, "05_ex1_compute_conditional_probability.png")


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
    save(fig, "05_ex2_compute_conditional_probability.png")


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
    save(fig, "05_ex3_conditionals_sum_to_1.png")


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
    save(fig, "05_ex4_symmetry_sum_given_first_die.png")


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
    save(fig, "05_ex5_independence_via_conditionals.png")


if __name__ == "__main__":
    wait, rate = load_call_center()
    fig_03_ex4(wait, rate)
    fig_03_ex5(wait, rate)
    fig_04_ex5()
    fig_05_ex1()
    fig_05_ex2()
    fig_05_ex3()
    fig_05_ex4()
    fig_05_ex5()
