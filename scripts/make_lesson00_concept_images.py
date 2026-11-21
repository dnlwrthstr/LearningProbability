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
    sheet_05_inclusion_exclusion()
    sheet_07_birthday_problem()
    sheet_09_probability_tree()
    sheet_10_bayes_disease_test()
    sheet_11_bayes_formula()
    sheet_12_bayes_spam()
    sheet_14_naive_bayes()
