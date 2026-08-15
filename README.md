# Learning Probability

A collection of Jupyter notebooks dedicated to learning and practicing Probability. Course videos are transcribed and restructured into educational notebooks with theory, worked examples, Python code, visualizations, and exercises.

## Project Structure

### [Coursera](./coursera/)
Notebooks following Coursera probability courses, one numbered directory per course/module:
- **00 Introduction to Probability**

More courses and topics will be added over time.

### Root Files
- `GUIDELINES.md`: The specification every notebook follows (structure, teaching style, code and visualization standards).
- [`FORMULAS.md`](./FORMULAS.md): Course-wide formula and notation reference — every formula from the notebooks in one place, with critical-value tables, a "which test do I use" guide, and a scipy/numpy cookbook.
- [`PYTHON.md`](./PYTHON.md): The implementation companion — how each formula is actually computed in the notebooks, using numpy/sympy/stdlib only (no scipy), plus the recurring simulation, enumeration and plotting idioms.
- `FORMULAS.pdf` / `FORMULAS-notes.pdf` / `PYTHON.pdf`: Printable typeset versions of the two references — `FORMULAS.pdf` with even margins (15 pages), `FORMULAS-notes.pdf` with a 5 cm blank outer column for handwritten annotation on a tablet (16 pages), and `PYTHON.pdf` (14 pages). Rebuild all three after editing either Markdown file with `./build_pdfs.sh` (needs `pandoc` and a LaTeX install providing `xelatex`).

## Getting Started

You need a Python 3 environment. Install the dependencies:

```bash
pip install -r requirements.txt
```

Notebook outputs are stripped automatically on commit. After cloning, register the git filter once:

```bash
nbstripout --install --attributes .gitattributes
```

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

## License
This project is for educational purposes.
