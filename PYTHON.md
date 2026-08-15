# Python Implementation Reference

The companion to [FORMULAS.md](./FORMULAS.md): where that file says *what* a formula is, this one says **how the course computes it**. Every snippet is the idiom actually used across the notebooks in [`coursera/`](./coursera/), so you can lift code straight into a new notebook.

> **No scipy.** The notebooks deliberately stay on `numpy`, `pandas`, `matplotlib`, `sympy` and the standard library (see `requirements.txt`). That is why distributions are built from `math.erf`, `math.gamma` and numeric integration rather than looked up in `scipy.stats`. [§17](#17-scipy-equivalents) maps every course helper to its scipy one-liner for when you do have it.

---

## Contents

1. [The toolkit](#1-the-toolkit)
2. [The standard preamble](#2-the-standard-preamble)
3. [Five core techniques](#3-five-core-techniques)
4. [Probability rules and Bayes](#4-probability-rules-and-bayes)
5. [Combinatorics](#5-combinatorics)
6. [Discrete distributions (PMF)](#6-discrete-distributions-pmf)
7. [Continuous distributions (PDF and CDF)](#7-continuous-distributions-pdf-and-cdf)
8. [Sampling from a distribution](#8-sampling-from-a-distribution)
9. [Describing a distribution](#9-describing-a-distribution)
10. [Quantiles, KDE and shape diagnostics](#10-quantiles-kde-and-shape-diagnostics)
11. [Several variables at once](#11-several-variables-at-once)
12. [LLN and CLT as simulations](#12-lln-and-clt-as-simulations)
13. [Maximum likelihood](#13-maximum-likelihood)
14. [MAP and Bayesian updating](#14-map-and-bayesian-updating)
15. [Confidence intervals](#15-confidence-intervals)
16. [Hypothesis tests](#16-hypothesis-tests)
17. [scipy equivalents](#17-scipy-equivalents)
18. [Plotting conventions](#18-plotting-conventions)
19. [Gotchas](#19-gotchas)

---

## 1. The toolkit

| Library | Used for | Typical calls |
|---|---|---|
| **numpy** | arrays, vectorized math, simulation, numeric integration | `default_rng`, `linspace`, `trapezoid`, `cumsum`, `searchsorted` |
| **matplotlib** | every visualization | `subplots`, `plot`, `hist`, `axvline`, `fill_between` |
| **sympy** | symbolic derivations and exact special functions | `diff`, `solve`, `erf`, `erfinv` |
| **math** | scalar special functions | `comb`, `factorial`, `erf`, `gamma`, `sqrt`, `pi` |
| **itertools** | enumerating finite sample spaces | `product`, `permutations`, `combinations` |
| **fractions** | exact (non-floating-point) probabilities | `Fraction` |
| **pandas** | joint-distribution tables, tabular data | `DataFrame`, `crosstab`, `.sum(axis=)` |

Division of labour worth internalising: **`math.erf` for a scalar CDF, `sympy.erfinv` for a quantile, `math.gamma` for the t and chi-squared densities, `np.trapezoid` for any tail area with no closed form.**

---

## 2. The standard preamble

Every notebook opens with some subset of this:

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# numpy < 2 fallback: np.trapezoid was called np.trapz
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

rng = np.random.default_rng(42)
```

Three conventions carried by those five lines:

- **`np.trapezoid`, never `np.trapz`.** Two environments run these notebooks (Anaconda with numpy 1.x, the PyCharm venv with numpy 2.x); `np.trapz` was removed in numpy 2.0, so the shim keeps one spelling working in both.
- **A seeded `Generator`, never `np.random.seed`.** `default_rng(42)` gives reproducible output without touching global state. All draws go through `rng`.
- **A fixed seed per notebook**, so printed numbers match the prose.

---

## 3. Five core techniques

Almost every computation in the course is one of these five moves.

### 3.1 Enumerate the sample space — exact answers by counting

For a finite experiment, build every outcome and count. This *is* the classical definition, executed literally.

```python
from itertools import product

sample_space = list(product("HT", repeat=3))              # all 8 outcomes
favorable = [o for o in sample_space if all(c == "H" for c in o)]
p = len(favorable) / len(sample_space)                     # 1/8 = 0.125
```

Same move for dice, with a condition instead of a filter:

```python
two_dice = list(product(range(1, 7), repeat=2))            # 36 outcomes
p_sum_7 = sum(1 for a, b in two_dice if a + b == 7) / len(two_dice)   # 6/36
```

### 3.2 Exact arithmetic with `Fraction`

Floating point hides the structure of a Bayes computation; `Fraction` keeps it visible.

```python
from fractions import Fraction

p_a = Fraction(1, 10_000)          # P(sick)
p_b_given_a = Fraction(99, 100)    # P(+ | sick)
p_b_given_not_a = Fraction(1, 100) # P(+ | healthy)

numerator = p_a * p_b_given_a
denominator = numerator + (1 - p_a) * p_b_given_not_a
posterior = numerator / denominator          # Fraction(1, 102) — exact, reduced
print(posterior, float(posterior))           # 1/102 0.00980392...
```

### 3.3 Simulate, then watch it converge

The law of large numbers is what licenses simulation as a check on theory, so nearly every notebook ends with one.

```python
n_trials = 10_000
flips = rng.integers(0, 2, size=(n_trials, 2))
both_heads = np.all(flips == 1, axis=1)

running_estimate = np.cumsum(both_heads) / np.arange(1, n_trials + 1)
print(f"after {n_trials}: {running_estimate[-1]:.4f}   exact: 0.25")
```

`np.cumsum(hits) / np.arange(1, n+1)` is **the** running-average idiom — it appears wherever a curve should settle onto a theoretical line.

### 3.4 Numeric integration with `np.trapezoid`

Continuous probability is area, and most of the areas here have no elementary closed form.

```python
grid = np.linspace(-6, 6, 20_000)
density = normal_pdf(grid)

total = np.trapezoid(density, grid)                       # ~1.0: it is a density
mask = (grid >= -1) & (grid <= 1)
inside = np.trapezoid(density[mask], grid[mask])          # ~0.6827
```

Two uses throughout: **checking** that a density integrates to 1, and **computing** a probability as the area over a sub-range.

### 3.5 Symbolic derivation with sympy

Where the course does calculus on a slide, the notebook does the same calculus in sympy rather than asserting the result.

```python
p = sp.symbols("p", positive=True)
k, n = sp.symbols("k n", positive=True)

logL = k * sp.log(p) + (n - k) * sp.log(1 - p)
p_hat = sp.solve(sp.Eq(sp.diff(logL, p), 0), p)[0]        # k/n
second = sp.simplify(sp.diff(logL, p, 2))                  # negative => a maximum
```

---

## 4. Probability rules and Bayes

### The rules, as functions

$$P(A') = 1 - P(A) \qquad P(A\cup B) = P(A)+P(B)-P(A\cap B) \qquad P(B\mid A) = \frac{P(A\cap B)}{P(A)}$$

```python
def p_complement(p_a):
    return 1 - p_a

def p_union(p_a, p_b, p_both=0.0):
    """Inclusion-exclusion; p_both=0 is the disjoint case."""
    return p_a + p_b - p_both

def p_conditional(sample_space, event, condition):
    """Conditioning = shrinking the sample space to `condition`."""
    restricted = [o for o in sample_space if condition(o)]
    both = [o for o in restricted if event(o)]
    return len(both) / len(restricted)
```

The counting version of conditional probability makes the "shrink the sample space" idea literal:

```python
two_dice = list(product(range(1, 7), repeat=2))
p_conditional(two_dice, lambda o: sum(o) == 10, lambda o: o[0] == 6)   # 1/6
```

### Bayes' theorem

$$P(A \mid B) = \frac{P(A)\,P(B\mid A)}{P(A)\,P(B\mid A) + P(A')\,P(B\mid A')}$$

```python
def bayes(p_a, p_b_given_a, p_b_given_not_a):
    numerator = p_a * p_b_given_a
    evidence = numerator + (1 - p_a) * p_b_given_not_a
    return numerator / evidence

bayes(0.0001, 0.99, 0.01)      # 0.0098... the disease-test surprise
```

### Naive Bayes

$$P(\text{spam}\mid w_1,\dots,w_n) \propto P(\text{spam})\prod_i P(w_i \mid \text{spam})$$

```python
import math

def naive_bayes(p_spam, p_word_given_spam, p_word_given_ham):
    """Each list holds P(word | class) for the words that appeared."""
    spam_score = p_spam * math.prod(p_word_given_spam)
    ham_score = (1 - p_spam) * math.prod(p_word_given_ham)
    return spam_score / (spam_score + ham_score)

naive_bayes(0.2, [0.7, 0.75], [0.125, 0.1])    # 0.913...
```

`math.prod` is the code form of the product in the formula. With many words, sum logs instead (see [§19](#19-gotchas)).

---

## 5. Combinatorics

$$\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$$

```python
from math import comb, factorial, perm

comb(5, 2)                                    # 10   — order ignored
perm(5, 2)                                    # 20   — order matters
factorial(5) // (factorial(2) * factorial(3)) # 10   — the formula, spelled out
comb(10, 3) == comb(10, 7)                    # True — symmetry
```

Prefer `math.comb` over a hand-rolled factorial ratio: it is exact for large `n` and never overflows through an intermediate factorial.

---

## 6. Discrete distributions (PMF)

A PMF is just **values + probabilities**, two aligned arrays.

### Binomial

$$p(x) = \binom{n}{x}p^x(1-p)^{n-x}$$

```python
def binomial_pmf(n, p):
    """Returns (values, probabilities) for X ~ Binomial(n, p)."""
    xs = np.arange(n + 1)
    probs = np.array([comb(n, int(x)) * p**x * (1 - p)**(n - x) for x in xs])
    return xs, probs

xs, probs = binomial_pmf(5, 0.5)
probs.sum()          # 1.0 — the normalization requirement
probs[2]             # 0.3125 = P(exactly 2 heads in 5 fair flips)
```

### Bernoulli, and building a PMF by enumeration

```python
def bernoulli_pmf(p):
    return np.array([0, 1]), np.array([1 - p, p])

# The same distribution, derived by counting instead of by formula:
from collections import Counter
counts = Counter(sum(o) for o in product([0, 1], repeat=5))
xs = np.array(sorted(counts))
probs = np.array([counts[x] for x in xs]) / 2**5     # matches binomial_pmf(5, 0.5)
```

### Checking the two PMF requirements

```python
assert np.all(probs >= 0)
assert np.isclose(probs.sum(), 1.0)
```

### Simulation check

```python
draws = rng.binomial(n=5, p=0.5, size=100_000)
empirical = np.bincount(draws, minlength=6) / len(draws)
np.abs(empirical - probs).max()        # < 0.01 — agrees to sampling noise
```

---

## 7. Continuous distributions (PDF and CDF)

### Uniform

$$f(x) = \frac{1}{b-a} \text{ on } [a,b] \qquad F(x) = \frac{x-a}{b-a}$$

```python
def uniform_pdf(x, a, b):
    x = np.asarray(x, dtype=float)
    return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)

def uniform_cdf(x, a, b):
    x = np.asarray(x, dtype=float)
    return np.clip((x - a) / (b - a), 0.0, 1.0)
```

`np.where` for the piecewise density and `np.clip` for the flat–ramp–flat CDF keep both vectorized.

### Normal — the key one

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \qquad F(x) = \tfrac12\left[1 + \operatorname{erf}\!\left(\frac{x-\mu}{\sigma\sqrt2}\right)\right]$$

```python
from math import erf, sqrt

def normal_pdf(x, mu=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def normal_cdf(x, mu=0.0, sigma=1.0):
    """Scalar CDF via the error function — no numeric integration needed."""
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))

normal_cdf(1.96) - normal_cdf(-1.96)      # 0.95000...
```

**The inverse — critical values — comes from sympy**, because `math` has no `erfinv`:

$$z_p = \sqrt{2}\;\operatorname{erf}^{-1}(2p - 1)$$

```python
def z_score(p):
    """Critical value z_p: the point with probability p to its left."""
    return float(sp.sqrt(2) * sp.erfinv(2 * sp.Float(p) - 1))

z_score(0.975)      # 1.959963...  the 95% two-tailed critical value
z_score(0.95)       # 1.644853...  the 5% one-tailed critical value
```

A variant used in the confidence-interval notebooks takes the confidence level directly:

```python
def z_for_confidence(confidence):
    """z such that a standard normal lands in [-z, z] with this probability."""
    return float(sp.sqrt(2) * sp.erfinv(sp.Float(confidence)))

z_for_confidence(0.95)      # 1.959963...
```

### Standardizing

$$Z = \frac{X-\mu}{\sigma}$$

```python
z = (x - x.mean()) / x.std()          # sample version
z = (x - mu) / sigma                  # population version
```

### Chi-squared

$$f(x) = \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}$$

```python
from math import gamma

def chi2_pdf(x, k):
    x = np.asarray(x, dtype=float)
    return np.where(x > 0,
                    x ** (k / 2 - 1) * np.exp(-x / 2) / (2 ** (k / 2) * gamma(k / 2)),
                    0.0)
```

The one-degree-of-freedom CDF also has the closed form $2\Phi(\sqrt{w}) - 1$:

```python
def chi2_cdf_1dof(w):
    return 2 * normal_cdf(np.sqrt(w)) - 1
```

### Student's t

$$f(t) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)}\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

```python
from math import gamma, sqrt, pi

def t_pdf(t, df):
    coeff = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    return coeff * (1 + np.asarray(t, dtype=float) ** 2 / df) ** (-(df + 1) / 2)
```

There is no closed-form t CDF here, so the tail is integrated numerically — the single most reusable helper in the whole course:

```python
def t_tail(t_value, df):
    """P(T > t_value) for T ~ t_df, by numeric integration."""
    grid = np.linspace(0, abs(t_value), 20_000)
    cdf_half = np.trapezoid(t_pdf(grid, df), grid)   # area from 0 to |t|
    tail = 0.5 - cdf_half                # symmetry: half the mass is left of 0
    return tail if t_value >= 0 else 1 - tail

t_tail(2.262, df=9)      # ~0.025 — reproduces the standard t table
```

Inverting it — a t critical value — means accumulating the density into a CDF and reading it backwards:

```python
def t_critical(p, df, hi=20.0, points=200_000):
    """The p-quantile of t_df: the t with P(T <= t) = p."""
    grid = np.linspace(0, hi, points)
    pdf = t_pdf(grid, df)
    areas = (pdf[1:] + pdf[:-1]) / 2 * np.diff(grid)     # trapezoid strips
    # start at 0.5: the grid begins at the median, where half the mass is already left
    cdf = 0.5 + np.concatenate([[0.0], np.cumsum(areas)])
    return float(np.interp(p, cdf, grid))                # read the CDF horizontally

t_critical(0.975, df=9)      # 2.2622  — matches the printed t table
t_critical(0.95, df=9)       # 1.8331
t_critical(0.975, df=120)    # 1.9799  — closing in on z = 1.96
```

Note the shape: **integrate once into a CDF array, then `np.interp` backwards.** That is the same inverse-transform move as [§8](#8-sampling-from-a-distribution), reused for quantiles.

---

## 8. Sampling from a distribution

### Discrete — the strip method

Stack the probabilities into `[0, 1]` and see which strip a uniform draw lands in.

```python
values = np.array(["red", "green", "blue"])
probs = np.array([0.5, 0.3, 0.2])
edges = np.cumsum(probs)                       # [0.5, 0.8, 1.0]

def sample_one(u):
    return values[np.searchsorted(edges, u)]

u = rng.random(10_000)
draws = values[np.searchsorted(edges, u)]      # vectorized: all 10,000 at once
```

`np.cumsum` builds the strips, `np.searchsorted` does the lookup.

### Continuous — inverse transform

$$U \sim \text{Uniform}(0,1), \qquad X = F^{-1}(U)$$

```python
grid = np.linspace(-5, 5, 10_000)
cdf_grid = np.array([normal_cdf(x) for x in grid])

def normal_inverse_cdf(u):
    return np.interp(u, cdf_grid, grid)        # horizontal read-off the CDF

samples = normal_inverse_cdf(rng.random(200_000))
samples.mean(), samples.std()                  # ~0, ~1
```

`np.interp` is the whole trick: reading the CDF curve horizontally instead of vertically.

### Or use the generator directly

```python
rng.normal(loc=0, scale=1, size=1000)
rng.integers(1, 7, size=1000)          # a die; high is exclusive
rng.binomial(n=10, p=0.5, size=1000)
rng.uniform(0, 15, size=1000)
rng.exponential(scale=2.0, size=1000)
rng.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2])
```

---

## 9. Describing a distribution

Given `values` and `pmf` as aligned arrays, every descriptor is one weighted sum.

$$E[g(X)] = \sum_x g(x)p(x) \qquad \mathrm{Var}(X) = E[X^2]-(E[X])^2$$

```python
def expected_value(values, pmf, g=lambda x: x):
    """E[g(X)] — transform the values, keep the probabilities."""
    return float(np.sum(g(np.asarray(values)) * np.asarray(pmf)))

def variance(values, pmf):
    mu = expected_value(values, pmf)
    return expected_value(values, pmf, g=lambda x: x**2) - mu**2      # shortcut form

def moment(values, pmf, k):
    return expected_value(values, pmf, g=lambda x: x**k)

def central_moment(values, pmf, k):
    mu = expected_value(values, pmf)
    return expected_value(values, pmf, g=lambda x: (x - mu) ** k)
```

Skewness and kurtosis are central moments divided by the right power of sigma:

$$\text{skew} = E\left[\left(\tfrac{X-\mu}{\sigma}\right)^3\right] \qquad \text{kurt} = E\left[\left(\tfrac{X-\mu}{\sigma}\right)^4\right]$$

```python
def skewness(values, pmf):
    sigma = np.sqrt(variance(values, pmf))
    return central_moment(values, pmf, 3) / sigma**3

def kurtosis(values, pmf):
    sigma = np.sqrt(variance(values, pmf))
    return central_moment(values, pmf, 4) / sigma**4      # normal = 3, not 0
```

### The continuous counterpart

Replace the sum with `np.trapezoid`:

```python
grid = np.linspace(-10, 10, 20_000)
f = normal_pdf(grid, mu=2, sigma=3)

mean = np.trapezoid(grid * f, grid)                    # ~2
var = np.trapezoid((grid - mean) ** 2 * f, grid)       # ~9
```

### From a sample instead of a distribution

```python
x.mean()
x.std(ddof=1)          # sample standard deviation s  (n-1)
np.var(x, ddof=1)      # sample variance s^2
np.median(x)
```

---

## 10. Quantiles, KDE and shape diagnostics

### Quantiles and the five-number summary

```python
q1, q2, q3 = np.percentile(data, [25, 50, 75])
iqr = q3 - q1
lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = data[(data < lower_fence) | (data > upper_fence)]
```

Quantiles **of a distribution** are the inverse CDF, found by scanning a grid:

```python
def normal_quantile(p, grid=np.linspace(-5, 5, 100_000)):
    cdf = np.array([normal_cdf(x) for x in grid])
    return grid[np.searchsorted(cdf, p)]
```

### Kernel density estimation

$$\hat f(x) = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-x_i)^2}{2\sigma^2}}$$

```python
def gaussian(x, center, sigma):
    return normal_pdf(x, mu=center, sigma=sigma)

def kde(x, data, sigma):
    """One bump per data point, averaged."""
    return np.mean([gaussian(x, xi, sigma) for xi in data], axis=0)

grid = np.linspace(data.min() - 3, data.max() + 3, 1_000)
estimate = kde(grid, data, sigma=5)
np.trapezoid(estimate, grid)         # ~1.0 — a genuine density
```

`sigma` is the bandwidth dial: small = spiky, large = over-smoothed.

### QQ plot by hand

```python
n = len(data)
probs = (np.arange(1, n + 1) - 0.5) / n         # plotting positions
theoretical = np.array([normal_quantile(p) for p in probs])
empirical = np.sort((data - data.mean()) / data.std(ddof=1))

plt.scatter(theoretical, empirical)
plt.plot([-3, 3], [-3, 3], "r--")               # straight line => normal
```

---

## 11. Several variables at once

A joint PMF is a 2-D array, or a `DataFrame` when the labels matter.

```python
joint = pd.crosstab(df["age"], df["height"], normalize=True)
```

### Marginals — sum out the other variable

$$P_X(x) = \sum_y P_{XY}(x,y)$$

```python
p_x = joint.sum(axis=1)      # sum across columns -> marginal of the row variable
p_y = joint.sum(axis=0)      # sum down rows      -> marginal of the column variable
```

With a plain numpy array: `joint.sum(axis=1)` and `joint.sum(axis=0)`.

### Conditionals — slice, then normalize

$$P_{Y\mid X=x}(y) = \frac{P_{XY}(x,y)}{P_X(x)}$$

```python
def conditional(joint, x):
    row = joint.loc[x]
    return row / row.sum()       # renormalize the slice so it sums to 1
```

### Independence check

```python
outer = np.outer(p_x, p_y)                       # what the joint WOULD be
independent = np.allclose(joint.values, outer)   # cell-by-cell factorization
```

### Covariance and correlation

$$\mathrm{Cov}(X,Y) = E[XY]-E[X]E[Y] \qquad \rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}$$

```python
def covariance(x, y):
    """Population version: divide by n."""
    return np.mean((x - x.mean()) * (y - y.mean()))

def correlation(x, y):
    return covariance(x, y) / (x.std() * y.std())

# library versions (note: these use n-1)
np.cov(x, y)                # 2x2 covariance matrix
np.corrcoef(x, y)[0, 1]     # correlation
```

Correlation is scale-free, so `covariance` changes with units while `correlation` does not — a useful assertion to run when demonstrating the point.

### Multivariate Gaussian

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\tfrac12(\mathbf{x}-\vec\mu)^T\Sigma^{-1}(\mathbf{x}-\vec\mu)\right)$$

```python
def multivariate_normal_pdf(x, mu, sigma):
    x, mu = np.atleast_2d(x), np.asarray(mu)
    n = len(mu)
    diff = x - mu
    inv, det = np.linalg.inv(sigma), np.linalg.det(sigma)
    quad = np.einsum("ij,jk,ik->i", diff, inv, diff)      # the quadratic form, per row
    return np.exp(-0.5 * quad) / ((2 * np.pi) ** (n / 2) * np.sqrt(det))
```

`np.linalg.inv` / `det` are the code for the two matrix pieces; `np.einsum` evaluates the quadratic form for every row at once. Sampling: `rng.multivariate_normal(mu, sigma, size=n)`.

---

## 12. LLN and CLT as simulations

### Law of large numbers

$$\bar X_n \longrightarrow \mu$$

```python
draws = rng.exponential(scale=2.0, size=50_000)         # any distribution works
running_mean = np.cumsum(draws) / np.arange(1, len(draws) + 1)

plt.plot(running_mean)
plt.axhline(2.0, color="red", linestyle="--")           # the true mean
```

### Central limit theorem

$$\bar X_n \approx N\!\left(\mu, \tfrac{\sigma^2}{n}\right)$$

```python
n, reps = 30, 20_000
samples = rng.exponential(scale=2.0, size=(reps, n))    # rows = independent samples
means = samples.mean(axis=1)                            # one sample mean per row

print(means.mean(), 2.0)                                # matches mu
print(means.std(ddof=1), 2.0 / np.sqrt(n))              # matches sigma/sqrt(n)

standardized = (means - 2.0) / (2.0 / np.sqrt(n))
plt.hist(standardized, bins=60, density=True)
plt.plot(grid := np.linspace(-4, 4, 400), normal_pdf(grid))
```

**The `size=(reps, n)` + `.mean(axis=1)` pattern is the CLT workhorse**: one row per repeated experiment, collapse each row to its statistic.

---

## 13. Maximum likelihood

### Symbolically — the derivation, not just the answer

$$\hat p = \frac{k}{n}$$

```python
p = sp.symbols("p", positive=True)
k, n = sp.symbols("k n", positive=True)

logL = k * sp.log(p) + (n - k) * sp.log(1 - p)
p_hat = sp.solve(sp.Eq(sp.diff(logL, p), 0), p)[0]      # k/n
sp.simplify(sp.diff(logL, p, 2))                         # < 0 => a maximum
```

### Numerically — the likelihood curve and its peak

```python
grid = np.linspace(0.001, 0.999, 10_000)
likelihood = grid**8 * (1 - grid)**2                     # 8 heads, 2 tails
grid[np.argmax(likelihood)]                              # 0.8 = 8/10
```

`np.argmax` over a parameter grid is the course's default optimizer — no solver required, and it makes the "contest between models" framing literal.

### Gaussian MLE

$$\hat\mu = \bar x, \qquad \hat\sigma^2 = \tfrac1n\sum_i (x_i-\bar x)^2$$

```python
mu_hat = data.mean()
sigma2_hat = data.var(ddof=0)      # ddof=0: the MLE divides by n, and is biased
```

### MLE = least squares for regression

$$\text{maximize } L \iff \text{minimize } \sum_i d_i^2$$

```python
def squared_error(slope, x, y):
    return np.sum((y - slope * x) ** 2)

def log_likelihood(slope, x, y, sigma=1.0):
    d = y - slope * x
    return np.sum(-0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)))

slopes = np.linspace(0, 4, 2_000)
best_ls = slopes[np.argmin([squared_error(s, x, y) for s in slopes])]
best_ml = slopes[np.argmax([log_likelihood(s, x, y) for s in slopes])]
assert np.isclose(best_ls, best_ml)      # the same point, two readings
```

---

## 14. MAP and Bayesian updating

$$\text{posterior} \propto \text{likelihood} \times \text{prior} \qquad \hat p_{\text{MAP}} = \frac{a+k}{2a+n}$$

```python
grid = np.linspace(0.001, 0.999, 10_000)

def posterior(grid, k, n, a=0):
    """Unnormalized: proportional is enough to locate the mode."""
    likelihood = grid**k * (1 - grid) ** (n - k)
    prior = grid**a * (1 - grid) ** a          # a=0 is the flat prior
    return likelihood * prior

post = posterior(grid, k=8, n=10, a=0)
grid[np.argmax(post)]                          # 0.8 — flat prior => MAP = MLE
grid[np.argmax(posterior(grid, 8, 10, a=9))]   # ~0.607 — the moderate Bayesian
```

Normalize only when you need to plot or integrate it:

```python
post /= np.trapezoid(post, grid)
np.trapezoid(post, grid)          # 1.0
```

### Sequential updating — yesterday's posterior is today's prior

```python
def update(post, grid, heads, tails):
    return post * grid**heads * (1 - grid) ** tails

belief = np.ones_like(grid)                 # flat prior
belief = update(belief, grid, 8, 2)         # first batch
belief = update(belief, grid, 6, 4)         # second batch
grid[np.argmax(belief)]                     # 0.7 = 14/20 — same as all at once
```

---

## 15. Confidence intervals

$$\text{CI} = \bar x \pm z_{1-\alpha/2}\frac{\sigma}{\sqrt n}$$

The six steps, exactly as the course narrates them:

```python
def confidence_interval(x_bar, sigma, n, level=0.95):
    alpha = 1 - level                              # 2. significance
    critical = z_score(1 - alpha / 2)              # 3. critical value
    standard_error = sigma / np.sqrt(n)            # 4. standard error
    margin = critical * standard_error             # 5. margin of error
    return x_bar - margin, x_bar + margin          # 6. the interval

confidence_interval(172.0, sigma=10, n=49)         # (169.2, 174.8)
```

### Unknown sigma — the t-interval

```python
def t_confidence_interval(x_bar, s, n, level=0.95):
    critical = t_critical(1 - (1 - level) / 2, df=n - 1)
    margin = critical * s / np.sqrt(n)
    return x_bar - margin, x_bar + margin
```

### Proportions

$$\hat p \pm z_{1-\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}$$

```python
def proportion_interval(successes, n, level=0.95):
    p_hat = successes / n
    margin = z_score(1 - (1 - level) / 2) * np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - margin, p_hat + margin

proportion_interval(24, 30)          # (0.657, 0.943)
```

### Sample size planning

$$n \ge \left(\frac{z\,\sigma}{\text{MOE}}\right)^2$$

```python
def required_n(sigma, moe, level=0.95):
    z = z_score(1 - (1 - level) / 2)
    return math.ceil((z * sigma / moe) ** 2)

required_n(sigma=25, moe=3)          # 267 — always round UP
```

### Verifying the 95% promise

The definition of a confidence interval is a claim about repetition, so test it by repeating:

```python
mu, sigma, n, reps = 170.0, 10.0, 49, 10_000
samples = rng.normal(mu, sigma, size=(reps, n))
lo, hi = zip(*[confidence_interval(s.mean(), sigma, n) for s in samples])
coverage = np.mean((np.array(lo) <= mu) & (mu <= np.array(hi)))    # ~0.95
```

---

## 16. Hypothesis tests

### z-test (sigma known)

$$Z = \frac{\bar x - \mu_0}{\sigma/\sqrt n}$$

```python
def z_test(x_bar, sigma, n, mu0, tail="two", alpha=0.05):
    z_obs = (x_bar - mu0) / (sigma / np.sqrt(n))
    right = 1 - normal_cdf(z_obs)
    p = {"right": right, "left": 1 - right, "two": 2 * min(right, 1 - right)}[tail]
    return z_obs, p, ("reject H0" if p < alpha else "fail to reject H0")
```

### t-test (sigma unknown) — the course's main workhorse

$$T = \frac{\bar x - \mu_0}{s/\sqrt n} \sim t_{n-1}$$

```python
def t_test(x_bar, s, n, mu0, tail="two", alpha=0.05):
    t_obs = (x_bar - mu0) / (s / np.sqrt(n))
    df = n - 1
    p = {"right": t_tail(t_obs, df),
         "left": 1 - t_tail(t_obs, df),
         "two": 2 * t_tail(abs(t_obs), df)}[tail]
    return t_obs, p, ("reject H0" if p < alpha else "fail to reject H0")

t_test(68.442, s=3.113, n=10, mu0=66.7, tail="right")   # t=1.77, p=0.055
```

Note how the three tails are a **dictionary lookup on one tail function** — the shared structure of the three tests made explicit.

### Two-sample (Welch)

$$T = \frac{\bar x - \bar y}{\sqrt{\frac{s_x^2}{n_x} + \frac{s_y^2}{n_y}}}, \qquad \nu = \frac{\left(\frac{s_x^2}{n_x}+\frac{s_y^2}{n_y}\right)^2}{\frac{(s_x^2/n_x)^2}{n_x-1}+\frac{(s_y^2/n_y)^2}{n_y-1}}$$

```python
def welch_t_test(x_bar, s_x, n_x, y_bar, s_y, n_y, tail="two"):
    a, b = s_x**2 / n_x, s_y**2 / n_y
    se = np.sqrt(a + b)
    t_obs = (x_bar - y_bar) / se
    df = (a + b) ** 2 / (a**2 / (n_x - 1) + b**2 / (n_y - 1))     # Welch-Satterthwaite
    p = {"right": t_tail(t_obs, df),
         "left": 1 - t_tail(t_obs, df),
         "two": 2 * t_tail(abs(t_obs), df)}[tail]
    return t_obs, df, p

welch_t_test(68.442, 3.113, 10, 65.949, 3.106, 9, tail="right")   # t=1.745, df=16.8
```

### Paired — a one-sample test on the differences

```python
def paired_t_test(before, after, tail="two"):
    d = np.asarray(before) - np.asarray(after)          # collapse pairs to differences
    return t_test(d.mean(), d.std(ddof=1), len(d), mu0=0, tail=tail)

# The course's ten participants reduce to d-bar = 1.09, s_d = 1.485, n = 10:
t_test(1.09, s=1.485, n=10, mu0=0, tail="right")        # t = 2.321, p = 0.0227
```

The whole test is `t_test` applied to one derived array — which is exactly the claim that a paired test *is* a one-sample test in disguise.

### Two proportions (pooled z) — the A/B test

$$Z = \frac{\frac{X}{n_A}-\frac{Y}{n_B}}{\sqrt{\hat p(1-\hat p)\left(\frac{n_A+n_B}{n_A n_B}\right)}}, \qquad \hat p = \frac{X+Y}{n_A+n_B}$$

```python
def two_proportion_z_test(x, n_a, y, n_b, tail="two"):
    p_pooled = (x + y) / (n_a + n_b)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (n_a + n_b) / (n_a * n_b))
    z_obs = (x / n_a - y / n_b) / se
    right = 1 - normal_cdf(z_obs)
    p = {"right": right, "left": 1 - right, "two": 2 * min(right, 1 - right)}[tail]
    return z_obs, p

# Group A: 20 conversions out of 80.  Group B: 8 out of 20.
two_proportion_z_test(20, 80, 8, 20, tail="left")     # z = -1.336, p = 0.091
```

The arguments are **counts and sizes, not rates** — passing `0.25` instead of `20` silently produces a meaningless pooled estimate.

### Power and Type II error

$$\beta(\mu) = P(\text{fail to reject} \mid \mu), \qquad \text{power} = 1-\beta$$

```python
def power_at(mu_true, mu0, sigma, n, alpha=0.05):
    k = mu0 + z_score(1 - alpha) * sigma / np.sqrt(n)     # right-tailed threshold
    beta = normal_cdf(k, mu=mu_true, sigma=sigma / np.sqrt(n))
    return 1 - beta

power_at(70, mu0=66.7, sigma=3, n=10)      # ~0.967
```

---

## 17. scipy equivalents

The course avoids scipy on purpose — implementing `normal_cdf` from `erf` is part of the teaching. In production code you would use the library. This table is the bridge (and matches [FORMULAS.md §15](./FORMULAS.md#15-python-cookbook)).

| Task | Course implementation | scipy one-liner |
|---|---|---|
| Normal PDF | `normal_pdf(x, mu, sigma)` | `stats.norm.pdf(x, mu, sigma)` |
| Normal CDF | `0.5*(1+erf(z/sqrt(2)))` | `stats.norm.cdf(x, mu, sigma)` |
| Normal quantile | `sp.sqrt(2)*sp.erfinv(2*p-1)` | `stats.norm.ppf(p)` |
| Binomial PMF | `comb(n,x)*p**x*(1-p)**(n-x)` | `stats.binom.pmf(x, n, p)` |
| Chi-squared PDF | closed form with `math.gamma` | `stats.chi2.pdf(x, k)` |
| t PDF | closed form with `math.gamma` | `stats.t.pdf(t, df)` |
| t tail | `t_tail(t, df)` via `np.trapezoid` | `stats.t.sf(t, df)` |
| t critical value | cumulative trapezoid + `np.interp` | `stats.t.ppf(1-alpha/2, df)` |
| One-sample t-test | `t_test(...)` | `stats.ttest_1samp(x, mu0)` |
| Welch two-sample | `welch_t_test(...)` | `stats.ttest_ind(x, y, equal_var=False)` |
| Paired t-test | `paired_t_test(...)` | `stats.ttest_rel(x, y)` |
| KDE | `kde(x, data, sigma)` | `stats.gaussian_kde(data)` |
| QQ plot | manual quantile pairing | `stats.probplot(x, dist="norm", plot=plt)` |
| Skew / kurtosis | central moments / sigma**k | `stats.skew(x)` / `stats.kurtosis(x, fisher=False)` |

scipy's `ttest_*` return **two-tailed** p-values; halve them for a one-tailed test after checking the sign.

---

## 18. Plotting conventions

Every figure in the course follows the same recipe (per `GUIDELINES.md`):

```python
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Theoretical PDF")
plt.hist(samples, bins=50, density=True, alpha=0.5, label="Simulated")
plt.axvline(mu, color="red", linestyle="--", label=f"mean = {mu}")
plt.title("What the reader should conclude")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

Recurring devices:

- **`density=True` on histograms** so simulated bars and a theoretical PDF share one y-axis.
- **`plt.axvline` / `axhline`** with a dashed red line for a theoretical landmark (a mean, a threshold, a critical value).
- **`plt.fill_between`** to shade a tail area — the visual form of a p-value.
- **`plt.subplots(1, 3, figsize=(15, 4))`** for side-by-side comparisons, then `plt.suptitle`.
- **`plt.grid(True, alpha=0.3)`** everywhere; grids stay faint enough not to compete with the data.

---

## 19. Gotchas

- **`np.trapezoid`, not `np.trapz`.** Removed in numpy 2.0; keep the shim at the top of every notebook. Same story for `np.in1d` and `np.row_stack`.
- **`ddof` is the bias switch.** `np.std(x)` / `np.var(x)` divide by `n` (the population/MLE form); pass `ddof=1` for the sample estimators `s` and `s²`. `np.cov` already defaults to `n-1`.
- **`rng.integers(1, 7)` excludes the 7.** The upper bound is exclusive, unlike the old `np.random.randint` mental model people carry over.
- **Kurtosis conventions differ.** The course uses the raw standardized fourth moment, where the normal scores **3**. `scipy.stats.kurtosis` defaults to *excess* kurtosis (normal = 0); pass `fisher=False` to compare.
- **Quantile conventions differ.** `np.percentile` interpolates by default, so hand-computed quartiles on small samples may not match exactly; that is convention, not error.
- **Products of many small probabilities underflow.** Naive Bayes over hundreds of words needs logs: sum `np.log(p)` and compare, or use `math.fsum`.
- **Normalize a posterior only when you must.** For a MAP estimate the proportional form is enough; the constant matters only for plotting or integrating.
- **A grid search is only as good as its grid.** `np.argmax` over `np.linspace(0, 1, 10_000)` resolves to about 1e-4; tighten the grid before trusting more digits.
- **Seed everything.** `rng = np.random.default_rng(42)` at the top; never `np.random.seed`, which mutates global state other cells depend on.
