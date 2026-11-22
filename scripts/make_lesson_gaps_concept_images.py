#!/usr/bin/env python3
"""Concept sheets for two notebooks that had none yet.

- coursera/00_introduction_to_probability/15_probability_in_machine_learning.ipynb
- coursera/07_hypothesis_testing/01_intro_to_hypothesis_testing.ipynb

One page per notebook that summarizes its central idea. Every number is computed here from
the same data, parameters and random seeds the notebook uses, with asserts against the values
the notebook prints. Images are written to each lesson's img/ folder.

Run from anywhere:  /opt/anaconda3/bin/python scripts/make_lesson_gaps_concept_images.py
"""
from math import comb

import numpy as np
from matplotlib.patches import FancyBboxPatch

from viz_style import (REPO, INK, INK2, MUTED, BLUE, BLUE_LIGHT, ORANGE, AQUA,
                       new_figure, grid, bars, grouped_bars, panel)
from viz_style import save as _save

LESSON00 = REPO / "coursera" / "00_introduction_to_probability"
LESSON07 = REPO / "coursera" / "07_hypothesis_testing"

ORANGE_LIGHT = "#f8c9b3"
AQUA_LIGHT = "#a9e3cd"


# ================================================================ helpers
def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


def flow_row(ax, y, boxes, colors, heading):
    """Three rounded boxes joined by arrows, with a bold heading above them."""
    xs = [0.0, 3.55, 7.1]
    width, height = 2.9, 1.25
    for x, text, color in zip(xs, boxes, colors):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor=MUTED, linewidth=1.2))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.5)
    for x0, x1 in [(xs[0] + width, xs[1]), (xs[1] + width, xs[2])]:
        ax.annotate("", xy=(x1 - 0.06, y + height / 2), xytext=(x0 + 0.08, y + height / 2),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=1.8))
    ax.text(0.0, y + height + 0.3, heading, fontsize=11.5, weight="bold", color=INK)


# ================================================================ lesson 00, notebook 15
def sheet_15_probability_in_ml():
    words = {                                         # P(word | happy), P(word | sad)
        "wonderful": (0.30, 0.02),
        "love": (0.25, 0.05),
        "awful": (0.02, 0.35),
        "tears": (0.04, 0.20),
        "sunshine": (0.15, 0.03),
    }
    p_happy = 0.5

    def sides(sentence):
        happy, sad = p_happy, 1 - p_happy
        steps = [(happy, sad)]
        for w in sentence:
            happy *= words[w][0]
            sad *= words[w][1]
            steps.append((happy, sad))
        return happy / (happy + sad), steps

    sentences = [["love", "sunshine"], ["awful", "tears"], ["love", "awful"]]
    posts = [sides(s)[0] for s in sentences]
    assert [round(p, 3) for p in posts] == [0.962, 0.011, 0.222]   # the notebook's output
    spam_both = 0.2 * 0.7 * 0.75                       # notebook 14, quoted in section 2
    ham_both = 0.8 * 0.125 * 0.1
    assert round(spam_both / (spam_both + ham_both), 3) == 0.913
    _, steps = sides(["love", "awful"])
    ratios = {w: ph / ps for w, (ph, ps) in words.items()}
    assert close(ratios["wonderful"], 15, 1e-9)

    fig = new_figure("Machine learning is mostly conditional probability",
                     "Supervised models evaluate P(label | evidence); generative models choose the output "
                     "that maximizes a probability.  Naive Bayes: prior × word likelihoods, normalized.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    flow_row(ax, 2.75, ["evidence\npixels, words,\nsymptoms", "model\ncomputes\nP(label | ev.)",
                        "prediction\nP(cat | pixels)\n= 0.9"], [BLUE_LIGHT, AQUA_LIGHT, ORANGE_LIGHT],
             "Supervised: evaluate a conditional probability")
    flow_row(ax, 0.0, ["goal\na face,\na story", "model\nmaximizes\nP(face | pixels)",
                       "generated\noutput\n(StyleGAN)"], [BLUE_LIGHT, AQUA_LIGHT, ORANGE_LIGHT],
             "Generative: maximize a probability")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.2, 4.6)
    ax.axis("off")
    ax.set_title("Two ways ML uses probability")

    ax = fig.add_subplot(gs[0, 1])
    labels = [" + ".join(s) for s in sentences]
    positions = np.arange(len(sentences))
    colors = [AQUA if p > 0.5 else ORANGE for p in posts]
    ax.bar(positions, posts, width=0.6, color=colors, zorder=2)
    for x, p in zip(positions, posts):
        ax.text(x, p + 0.025, f"{p:.3f} → {'HAPPY' if p > 0.5 else 'SAD'}", ha="center", fontsize=10,
                weight="bold", color=INK)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(2.45, 0.52, "decision line 0.5", ha="right", fontsize=9.5, color=INK2)
    ax.set_xticks(positions, [lbl.replace(" + ", "\n+ ") for lbl in labels])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("P(happy | words)")
    ax.grid(axis="y", zorder=0)
    ax.set_title("A toy sentiment classifier (prior 0.5)")

    panel(fig.add_subplot(gs[0, 2]), "The Bayes pipeline is a classifier", [
        "1. Prior P(label), counted from labeled",
        "   data: that is training.",
        "2. Event / features (words, pixels):",
        "   that is the input.",
        "3. Posterior P(label | features):",
        "   that is the prediction.",
        "",
        "Same shape, different features:",
        "P(spam | lottery, winning) = 0.105 / 0.115",
        f"   = {spam_both / (spam_both + ham_both):.3f}   (notebook 14)",
    ], spacing=0.082)

    ax = fig.add_subplot(gs[1, 0])
    names = list(words)
    vals = [ratios[w] for w in names]
    ax.barh(np.arange(len(names)), [abs(v - 1) for v in vals], left=[min(v, 1) for v in vals],
            color=[AQUA if v > 1 else ORANGE for v in vals], height=0.6, zorder=2)
    ax.set_xscale("log")
    ax.axvline(1, color=INK2, linewidth=1.2)
    for i, (w, v) in enumerate(zip(names, vals)):
        ph, ps = words[w]
        text = f"{ph:.2f} / {ps:.2f} = {v:.3g}"
        ax.text(v * 1.15 if v > 1 else 1.15, i, text, va="center", ha="left", fontsize=9.5)
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_ylim(len(names) - 0.4, -1.2)
    ax.set_xlim(1 / 300, 300)
    ax.set_xlabel("P(word | happy) / P(word | sad), log scale")
    ax.grid(axis="x", zorder=0)
    ax.text(250, -0.85, "pushes happy →", ha="right", fontsize=9.5, color=AQUA, weight="bold")
    ax.text(1 / 250, -0.85, "← pushes sad", ha="left", fontsize=9.5, color=ORANGE, weight="bold")
    ax.set_title("Each word is a likelihood ratio")

    ax = fig.add_subplot(gs[1, 1])
    grouped_bars(ax, ["prior", "× love", "× awful"],
                 [("happy side", [s[0] for s in steps], AQUA), ("sad side", [s[1] for s in steps], ORANGE)],
                 "\"love + awful\": multiply both sides", "", ylim=0.62, label_fmt=lambda v: f"{v:.4g}")
    ax.legend(loc="upper right", fontsize=10)
    happy_end, sad_end = steps[-1]
    ax.text(2.0, 0.2, f"{happy_end:.4f} / ({happy_end:.4f} + {sad_end:.5f})\n= {posts[2]:.3f}  → SAD",
            ha="center", fontsize=10.5, weight="bold", color=INK)

    panel(fig.add_subplot(gs[1, 2]), "One question, many tasks", [
        "spam filter → P(spam | words, sender, ...)",
        "sentiment → P(happy | words)",
        "image → P(cat | pixel₁, ..., pixelₙ)",
        "diagnosis → P(healthy | symptoms, ...)",
        "",
        "Bigger models change the features and",
        "the machinery, not the question.",
        "Generative models (StyleGAN faces, text)",
        "search for the most probable output.",
    ], spacing=0.085)
    _save(fig, LESSON00 / "img", "15_probability_in_machine_learning.png")


# ================================================================ lesson 07, notebook 01
def sheet_01_intro_hypothesis_testing():
    n_triggers, p_ham, p_spam = 6, 0.001, 0.40
    rng = np.random.default_rng(8)                    # the notebook's seed and draw order
    n_emails = 10_000
    is_spam = rng.random(n_emails) < 0.05
    counts = np.where(is_spam, rng.binomial(n_triggers, p_spam, size=n_emails),
                      rng.binomial(n_triggers, p_ham, size=n_emails))
    reject = counts >= 2
    n_ham, n_spam = int((~is_spam).sum()), int(is_spam.sum())
    ham_flagged = int((reject & ~is_spam).sum())
    spam_flagged = int((reject & is_spam).sum())
    spam_missed = int((~reject & is_spam).sum())
    assert (n_ham, n_spam, int(reject.sum()), ham_flagged, spam_missed) == (9558, 442, 325, 0, 117)
    assert f"{spam_missed / n_spam:.2%}" == "26.47%"

    p_email_ham = p_ham ** 6
    p_email_spam = p_spam ** 6
    assert f"{p_email_ham:.2e}" == "1.00e-18" and f"{p_email_spam:.4f}" == "0.0041"

    def pmf(p, k):
        return comb(n_triggers, k) * p ** k * (1 - p) ** (n_triggers - k)

    ks = np.arange(0, 7)
    pmf_ham = np.array([pmf(p_ham, k) for k in ks])
    pmf_spam = np.array([pmf(p_spam, k) for k in ks])
    tails = [pmf_ham[k:].sum() for k in range(1, 7)]
    assert [f"{t:.2e}" for t in tails[:3]] == ["5.99e-03", "1.50e-05", "2.00e-08"]
    assert close(0.001 ** 3, 1e-9, 1e-15)             # Example 3 in the text

    fig = new_figure("Evidence can reject H₀, but never prove it",
                     "H₀: the email is ham (the safe default),  H₁: it is spam.  Reject H₀ only when the "
                     "evidence would be very unlikely under H₀; otherwise you merely fail to reject it.")
    gs = grid(fig, top=0.84, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    w = 0.38
    ax.bar(ks - w / 2, pmf_ham, w, color=BLUE, label="under H₀: ham (p = 0.001)", zorder=2)
    ax.bar(ks + w / 2, pmf_spam, w, color=ORANGE, label="under H₁: spam (p = 0.40)", zorder=2)
    ax.axvline(1.5, color=INK, linestyle="--", linewidth=1.3)
    ax.set_yscale("log")
    ax.set_ylim(1e-19, 1e7)
    ax.set_yticks([10.0 ** e for e in range(-18, 1, 3)])
    ax.set_xticks(ks)
    ax.set_xlabel("trigger phrases in the email (of 6);  dashed: reject H₀ at 2+")
    ax.set_ylabel("probability (log scale)")
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("One observable, two stories")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["P(≥1)", "P(≥2)", "P(≥3)", "P(≥4)", "P(≥5)", "all 6"]
    ax.bar(np.arange(6), tails, width=0.62, color=[BLUE] * 5 + [ORANGE], zorder=2)
    for i, t in enumerate(tails):
        ax.text(i, t * 3, f"{t:.1e}", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-20, 1)
    ax.set_xticks(np.arange(6), labels)
    ax.set_ylabel("P(this many phrases | ham)")
    ax.set_xlabel("evidence observed")
    ax.grid(axis="y", zorder=0)
    ax.set_title("How unlikely is the evidence under H₀?")

    panel(fig.add_subplot(gs[0, 2]), "The video's email: all 6 phrases", [
        "earn extra cash, risk free, dear friend,",
        "act immediately, apply now, winner",
        "",
        f"`P(this | H₀: ham)  = 0.001⁶ = {p_email_ham:.2e}`",
        f"`P(this | H₁: spam) = 0.4⁶   = {p_email_spam:.4f}`",
        "",
        "Under H₀ a once-in-10¹⁸ coincidence,",
        "under H₁ unremarkable: reject H₀,",
        "the email goes to the spam folder.",
    ], spacing=0.085)

    ax = fig.add_subplot(gs[1, 0])
    rows = [("spam\nfolder", 0, spam_flagged), ("inbox", n_ham, spam_missed)]
    for i, (name, ham_part, spam_part) in enumerate(rows):
        ax.barh(i, ham_part, color=BLUE, height=0.55, zorder=2)
        ax.barh(i, spam_part, left=ham_part, color=ORANGE, height=0.55, zorder=2)
    ax.text(spam_flagged + 150, 0, f"reject H₀: {ham_flagged} ham + {spam_flagged} spam = {reject.sum()}",
            va="center", fontsize=10.5)
    ax.text(n_ham / 2, 1, f"{n_ham:,} ham", va="center", ha="center", color="white", fontsize=11,
            weight="bold")
    ax.annotate(f"+ {spam_missed} spam slipped by\n({spam_missed / n_spam:.2%} of spam)",
                (n_ham + spam_missed, 1.2), xytext=(4300, 1.62), fontsize=10, color=ORANGE,
                weight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE))
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.set_ylim(1.95, -0.55)
    ax.set_xlim(0, 10_500)
    ax.set_xlabel(f"emails  (simulated day: {n_emails:,}, seed 8, 5% spam)")
    ax.grid(axis="x", zorder=0)
    ax.set_title("A day of email under the 2+ rule")

    ax = fig.add_subplot(gs[1, 1])
    hist_ham = np.array([((counts == k) & ~is_spam).sum() for k in ks])
    hist_spam = np.array([((counts == k) & is_spam).sum() for k in ks])
    ax.bar(ks - w / 2, hist_ham, w, color=BLUE, label="truly ham", zorder=2)
    ax.bar(ks + w / 2, hist_spam, w, color=ORANGE, label="truly spam", zorder=2)
    for k in ks:
        for offset, v in [(-w / 2, hist_ham[k]), (w / 2, hist_spam[k])]:
            if v:
                ax.text(k + offset, v * 1.25, f"{v:,}", ha="center", fontsize=8.5)
    ax.axvline(1.5, color=INK, linestyle="--", linewidth=1.3)
    ax.axvspan(-0.6, 1.5, color=BLUE_LIGHT, alpha=0.25, zorder=0)
    ax.axvspan(1.5, 6.6, color=ORANGE_LIGHT, alpha=0.25, zorder=0)
    ax.text(0.45, 1.5e5, "inbox: 'not shown\nto be spam'", ha="center", fontsize=9.5, color=INK2)
    ax.text(4.0, 2.5e5, "spam folder", ha="center", fontsize=9.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2e6)
    ax.set_xlim(-0.6, 6.6)
    ax.set_xticks(ks)
    ax.set_xlabel("trigger phrases counted")
    ax.set_ylabel("emails (log scale)")
    ax.legend(loc="center right", fontsize=9.5)
    ax.set_title("Every email starts as ham")

    panel(fig.add_subplot(gs[1, 2]), "Pairs and the four sentences", [
        "`H₀ (baseline)          H₁ (claim)`",
        "`email is ham           email is spam`",
        "`drug has no effect     drug has an effect`",
        "`coin is fair p = 0.5   coin is biased`",
        "",
        "✓ reject H₀ → 'the email is spam'",
        "✓ fail to reject → 'not enough evidence'",
        "✗ fail to reject → 'proven ham'",
        "✗ reject H₀ → 'H₁ needed no evidence'",
        "H₀ = side whose wrongful rejection costs more.",
    ], spacing=0.08, fontsize=12)
    _save(fig, LESSON07 / "img", "01_intro_to_hypothesis_testing.png")


if __name__ == "__main__":
    sheet_15_probability_in_ml()
    sheet_01_intro_hypothesis_testing()
