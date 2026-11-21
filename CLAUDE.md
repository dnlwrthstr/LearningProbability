# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A personal learning project for probability. The primary workflow is transcribing course videos (starting with Coursera's Introduction to Probability) into educational Jupyter notebooks. There is no application code, build system, or test suite — the deliverables are notebooks.

## Structure

- `coursera/00_introduction_to_probability/` — notebooks for the Coursera course, one numbered directory per course/module.
- `GUIDELINES.md` — the authoritative spec for how every notebook must be written. Read it before creating or editing a notebook.
- `scripts/` — matplotlib generators for the images embedded in notebooks (`make_lesson00_concept_images.py`, `make_lesson02_concept_images.py`, `make_lesson03_solution_images.py`, `make_lesson04_concept_images.py`, `make_lesson05_concept_images.py`, `make_lesson06_concept_images.py`), sharing one look via `viz_style.py`. Each script computes every number from the notebook's own data and writes into that lesson's `img/` folder. Run with the Anaconda Python, e.g. `/opt/anaconda3/bin/python scripts/make_lesson00_concept_images.py`.

## Creating notebooks

Follow GUIDELINES.md. The essentials:

- Standard section order: Introduction → Theory (markdown + LaTeX) → Worked Examples → Python Examples → Visualizations → Exercises (basic/intermediate/advanced) → Solutions → Conclusion.
- Explain each topic in layers: intuition first, then formal definition, worked examples, visuals, code, exercises.
- Default libraries: numpy, pandas, matplotlib, sympy. Keep notebooks self-contained with minimal dependencies.
- Code cells should run independently where practical; readable variable names, no dense one-liners.
- Plots use matplotlib with figsize, title, axis labels, grid, and legends.
- Before finishing: verify the notebook JSON is valid, all cells render, and code runs.

When transcribing a video, the goal is not a verbatim transcript — restructure the content into the notebook format above so the learner can understand, see, and apply the concept.

## Environment

Python 3 with Jupyter. Run notebooks with `jupyter notebook`.

Two Python environments touch the notebooks: Anaconda (numpy 1.x) executes them during authoring, and the user's PyCharm virtualenv (numpy 2.x) runs them interactively. Write numpy code that works in both — e.g. use `np.trapezoid` with the numpy<2 fallback shim `if not hasattr(np, "trapezoid"): np.trapezoid = np.trapz` in the imports cell, and avoid APIs removed in numpy 2.0 (`np.trapz`, `np.in1d`, `np.row_stack`).

## Git

Notebook outputs are stripped on commit via an nbstripout git filter (`.gitattributes` + local git config). In a fresh clone the filter must be re-registered once with `nbstripout --install --attributes .gitattributes`. Never commit notebooks with outputs; local working copies may keep them.
