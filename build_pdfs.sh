#!/usr/bin/env bash
# Build printable / shareable PDFs from the Markdown references.
#
#   FORMULAS.pdf        even margins, the compact reading/printing copy
#   FORMULAS-notes.pdf  wide outer margin for handwritten annotation on a tablet
#   PYTHON.pdf          the implementation companion
#
# Requires: pandoc and a LaTeX distribution with xelatex (MacTeX on macOS).
#   brew install pandoc && brew install --cask mactex-no-gui
#
# Usage: ./build_pdfs.sh            (KEEP_TMP=1 to inspect the generated .tex/.log)

set -euo pipefail
cd "$(dirname "$0")"

REPO="https://github.com/dnlwrthstr/LearningProbability/blob/main"
NOTES_MARGIN="5cm"   # Do not widen past ~5.3cm: the two widest display equations
                     # (MLE->MAP summary, alpha/beta/power) then cross the divider.

TMP="$(mktemp -d)"
if [ -z "${KEEP_TMP:-}" ]; then
    trap 'rm -rf "$TMP"' EXIT
else
    trap 'echo "build files kept in $TMP"' EXIT
fi

# ---------------------------------------------------------------- preprocessing
# Rewrites a throwaway copy so the sources stay clean for GitHub/PyCharm:
#   1. relative links -> absolute GitHub URLs (a shared PDF has no repo beside it)
#   2. Unicode with no Latin Modern glyph -> LaTeX math, so nothing silently drops
cat > "$TMP/preprocess.py" <<'PY'
import re
import sys

src, dst, repo = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src, encoding="utf-8").read()

text = text.replace("](./", f"]({repo}/")

SYMBOLS = {
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$",
    "⇔": r"$\Leftrightarrow$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "×": r"$\times$",
    "²": r"$^2$",
}


def substitute(chunk):
    for char, macro in SYMBOLS.items():
        # Pandoc refuses to close inline math on a "$" that is followed by a
        # digit, so "≈68%" needs the space: without it \approx lands in text
        # mode and xelatex reports a missing glyph.
        chunk = re.sub(re.escape(char) + r"(?=\d)", lambda _, m=macro: m + " ", chunk)
        chunk = chunk.replace(char, macro)
    return chunk


out, in_fence = [], False
for line in text.split("\n"):
    if line.lstrip().startswith("```"):
        in_fence = not in_fence
        out.append(line)
        continue
    if in_fence:                       # code is copied byte-for-byte
        out.append(line)
        continue
    parts = re.split(r"(`[^`]*`)", line)   # and so are inline code spans
    out.append("".join(p if p.startswith("`") else substitute(p) for p in parts))

open(dst, "w", encoding="utf-8").write("\n".join(out))
PY

# --------------------------------------------------------------------- preamble
cat > "$TMP/header-common.tex" <<'TEX'
\usepackage{amsmath,amssymb}
\usepackage{longtable,booktabs,array}
\usepackage{fancyhdr}
\usepackage{etoolbox}

% needspace.sty is not in BasicTeX, so inline the one macro we need: reserve
% vertical space or break the page first. Without it a longtable can start at
% the foot of a page and leave its header stranded there with no rows.
\makeatletter
\newcommand\tableneedspace[1]{%
  \penalty-100\begingroup\setlength{\dimen@}{#1}%
  \vskip\dimen@\penalty9999\vskip-\dimen@\vskip\z@skip\endgroup}
\makeatother
\BeforeBeginEnvironment{longtable}{\tableneedspace{6\baselineskip}}

% Pandoc marks table heads with \endhead, i.e. "repeat on every page", which
% makes longtable strand a duplicate header when a table sits near a page
% boundary. These are short reference tables, so print the head once.
\let\endhead\endfirsthead

% Wide display equations and table cells shrink instead of running off the page.
\setlength{\emergencystretch}{3em}
\AtBeginEnvironment{longtable}{\small}

% Several tables carry stacked fractions that are taller than a default row,
% so without extra row height they collide with the line above.
\renewcommand{\arraystretch}{1.45}
\setlength{\extrarowheight}{3pt}

% The pages are mostly display equations; the default skips around them leave the
% reference looking airier than a lookup sheet wants, and cost a whole page.
\setlength{\abovedisplayskip}{6pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{6pt plus 2pt minus 2pt}
\setlength{\abovedisplayshortskip}{4pt plus 2pt}
\setlength{\belowdisplayshortskip}{4pt plus 2pt}

% Code is set unhighlighted (framed.sty is absent from BasicTeX). \small keeps
% the longest line in PYTHON.md (87 chars) inside the text block; verbatim does
% not wrap, so anything wider would run into the margin.
\makeatletter
\renewcommand{\verbatim@font}{\ttfamily\small}
\makeatother

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\doctitle}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
TEX

# The annotation copy adds a hairline marking off the writing column, so the
# blank space reads as deliberate rather than as a bad margin.
{
    cat "$TMP/header-common.tex"
    cat <<TEX
\usepackage{eso-pic}
\AddToShipoutPictureBG{%
  \AtPageLowerLeft{%
    \put(\LenToUnit{\dimexpr 2cm+\textwidth+0.9cm\relax},%
         \LenToUnit{1.7cm})%
    {\color{black!20}\rule{0.4pt}{\textheight}}}}
TEX
} > "$TMP/header-notes.tex"

# ----------------------------------------------------------------------- render
# render <jobname> <source.md> <output.pdf> <right-margin> <header.tex> <title.tex>
render() {
    local job="$1" md="$2" out="$3" right="$4" header="$5" titletex="$6"

    pandoc "$md" \
        --from=markdown+tex_math_dollars+pipe_tables+gfm_auto_identifiers \
        --standalone \
        --variable=documentclass:article \
        --variable=papersize:a4 \
        --variable=geometry:top=1.7cm \
        --variable=geometry:bottom=1.7cm \
        --variable=geometry:left=2cm \
        --variable="geometry:right=$right" \
        --variable=fontsize:10pt \
        --variable=linkcolor:RoyalBlue \
        --variable=urlcolor:RoyalBlue \
        --variable=colorlinks:true \
        --no-highlight \
        --include-in-header="$titletex" \
        --include-in-header="$header" \
        --output="$TMP/$job.tex"

    # Three passes: longtable needs a second one to settle its page breaks (a
    # single pass strands a repeated table header), and hyperref needs one to
    # resolve the contents links. Driving xelatex directly guarantees the runs.
    for pass in 1 2 3; do
        (cd "$TMP" && xelatex -interaction=batchmode -halt-on-error "$job.tex" >/dev/null 2>&1) || {
            echo "xelatex failed on pass $pass of $job; log tail:" >&2
            tail -30 "$TMP/$job.log" >&2
            exit 1
        }
    done

    cp "$TMP/$job.pdf" "$out"

    # Page count from the log, not Spotlight metadata, which lags a rebuild.
    local pages
    pages=$(sed -n 's/.*Output written on .* (\([0-9]*\) pages.*/\1/p' "$TMP/$job.log" | tail -1)
    echo "Wrote $out ($(du -h "$out" | cut -f1 | tr -d ' '), $pages pages)"

    # Surface layout problems instead of hiding them.
    if grep -q 'Overfull \\hbox' "$TMP/$job.log"; then
        echo "  note: $(grep -c 'Overfull \\hbox' "$TMP/$job.log") overfull hbox warning(s) — check wide equations/tables"
    fi
}

# build_doc <source.md> <basename> <running-title> <notes: yes|no>
build_doc() {
    local src="$1" base="$2" title="$3" want_notes="$4"

    python3 "$TMP/preprocess.py" "$src" "$TMP/$base.md" "$REPO"
    printf '\\newcommand{\\doctitle}{%s}\n' "$title" > "$TMP/$base-title.tex"

    render "$base-std" "$TMP/$base.md" "$base.pdf" 2cm \
        "$TMP/header-common.tex" "$TMP/$base-title.tex"

    if [ "$want_notes" = "yes" ]; then
        render "$base-notes" "$TMP/$base.md" "$base-notes.pdf" "$NOTES_MARGIN" \
            "$TMP/header-notes.tex" "$TMP/$base-title.tex"
    fi
}

build_doc FORMULAS.md FORMULAS 'Probability --- Formula \& Notation Reference' yes
build_doc PYTHON.md   PYTHON   'Probability --- Python Implementation Reference' no
