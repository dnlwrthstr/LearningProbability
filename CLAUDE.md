# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A personal learning project for probability. The primary workflow is transcribing course videos (starting with Coursera's Introduction to Probability) into educational Jupyter notebooks. There is no application code, build system, or test suite — the deliverables are notebooks.

## Structure

- `coursera/00_introduction_to_probability/` — notebooks for the Coursera course, one numbered directory per course/module.
- `GUIDELINES.md` — the authoritative spec for how every notebook must be written. Read it before creating or editing a notebook.

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
