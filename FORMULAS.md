# Formula & Notation Reference

A single-page lookup for every formula and piece of notation in the course notebooks under [`coursera/`](./coursera/).
Each entry says **what it is**, **when to use it**, and links to the notebook that derives it.

> Math renders in GitHub and in any MathJax-enabled Markdown preview. Currency is written as `10 USD` to avoid dollar-sign collisions with math delimiters.

---

## Contents

1. [Notation](#1-notation)
2. [Probability rules](#2-probability-rules)
3. [Counting and combinatorics](#3-counting-and-combinatorics)
4. [Bayes' theorem](#4-bayes-theorem)
5. [Random variables: PMF, PDF, CDF](#5-random-variables-pmf-pdf-cdf)
6. [Named distributions](#6-named-distributions)
7. [Describing a distribution](#7-describing-a-distribution)
8. [Several variables at once](#8-several-variables-at-once)
9. [Sampling, LLN, CLT](#9-sampling-lln-clt)
10. [Point estimation: MLE, MAP, regularization](#10-point-estimation-mle-map-regularization)
11. [Confidence intervals](#11-confidence-intervals)
12. [Hypothesis testing](#12-hypothesis-testing)
13. [Critical-value tables](#13-critical-value-tables)
14. [Which procedure do I use?](#14-which-procedure-do-i-use)
15. [Python cookbook](#15-python-cookbook)
16. [Traps and misreadings](#16-traps-and-misreadings)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $S$ | sample space — the set of all possible outcomes |
| $A, B, E$ | events (subsets of $S$); $\lvert A\rvert$ = number of outcomes in $A$ |
| $A'$ (or $A^c$, $\neg A$) | complement of $A$ — "not $A$" |
| $A \cup B$, $A \cap B$ | union ("or"), intersection ("and") |
| $P(A \mid B)$ | probability of $A$ **given** $B$ |
| $X, Y$ | random variables; $x, y$ their realized values |
| $p(x)$, $p_X(x)$ | probability **mass** function (discrete) |
| $f(x)$, $f_X(x)$ | probability **density** function (continuous) |
| $F(x)$, $F_X(x)$ | cumulative distribution function, $P(X \le x)$ |
| $\varphi$, $\Phi$ | standard-normal PDF and CDF |
| $\mu$, $\sigma$, $\sigma^2$ | **population** mean, standard deviation, variance |
| $\bar{x}$, $s$, $s^2$ | **sample** mean, standard deviation, variance |
| $p$, $\hat{p}$ | population proportion, sample proportion |
| $N$, $n$ | population size, sample size |
| $\theta$ | a generic parameter being estimated |
| $\hat{\theta}$ | an estimate of $\theta$ |
| $\Sigma$ | covariance matrix; $\lvert\Sigma\rvert$ its determinant |
| $\rho_{XY}$ | correlation coefficient |
| $\alpha$, $\beta$ | Type I error rate (significance), Type II error rate |
| $z_{1-\alpha/2}$, $t_{\nu,\,1-\alpha/2}$ | critical values of the normal / $t$ distribution |
| $\nu$, df | degrees of freedom |
| $\lambda$ | regularization strength |

**Greek = population/truth (fixed, unknown). Latin = sample/estimate (random, computed).**

---

## 2. Probability rules

Module: [`00_introduction_to_probability`](./coursera/00_introduction_to_probability/)

### Classical definition (equally likely outcomes)

$$P(E) = \frac{\text{favorable outcomes}}{\text{total outcomes}} = \frac{|E|}{|S|}$$

Only valid when every outcome in $S$ is equally likely. → [01](./coursera/00_introduction_to_probability/01_what_is_probability.ipynb)

### Complement rule

$$P(A') = 1 - P(A)$$

Use whenever "at least one …" is easier to count as "none …". → [03](./coursera/00_introduction_to_probability/03_complement_rule.ipynb)

### Sum rule (union)

$$P(A \cup B) = P(A) + P(B) \qquad \text{(disjoint: } A \cap B = \varnothing\text{)}$$

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) \qquad \text{(general — inclusion–exclusion)}$$

Subtract the overlap or you double-count it. → [04](./coursera/00_introduction_to_probability/04_sum_of_probabilities_disjoint_events.ipynb), [05](./coursera/00_introduction_to_probability/05_sum_of_probabilities_joint_events.ipynb)

### Product rule (intersection)

$$P(A \cap B) = P(A)\,P(B \mid A) \qquad \text{(always true)}$$

$$P(A \cap B) = P(A)\,P(B) \qquad \text{(independent events only)}$$

$$P(A_1 \cap \dots \cap A_n) = P(A_1)\cdots P(A_n) \qquad \text{(mutually independent)}$$

→ [06](./coursera/00_introduction_to_probability/06_independence_product_rule.ipynb), [09](./coursera/00_introduction_to_probability/09_dependent_events_probability_trees.ipynb)

### Conditional probability

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)} = \frac{\text{outcomes in } A \text{ and } B}{\text{outcomes in } A}$$

Conditioning **shrinks the sample space** to $A$. → [08](./coursera/00_introduction_to_probability/08_conditional_probability.ipynb)

**Independence test:** $A, B$ independent $\iff P(B \mid A) = P(B) \iff P(A\cap B) = P(A)P(B)$.

### Law of total probability

$$P(B) = P(A)\,P(B \mid A) + P(A')\,P(B \mid A')$$

The denominator of Bayes' theorem. Generalizes to any partition $\{A_i\}$: $P(B) = \sum_i P(A_i)P(B\mid A_i)$.

### The birthday problem (template for "at least one match")

$$P(\text{shared birthday}) = 1 - \underbrace{\frac{365}{365}\cdot\frac{364}{365}\cdots\frac{365-n+1}{365}}_{\text{all distinct}}$$

Tipping point past 50%: $n = 23$. → [07](./coursera/00_introduction_to_probability/07_birthday_problem.ipynb)

---

## 3. Counting and combinatorics

Module: [`01_probability_distribution`](./coursera/01_probability_distribution/)

| Quantity | Formula | Counts |
|---|---|---|
| Factorial | $k! = k(k-1)\cdots 2\cdot 1$, with $0! = 1$ | orderings of $k$ items |
| Ordered selections (permutations) | $n(n-1)\cdots(n-k+1) = \dfrac{n!}{(n-k)!}$ | picking $k$ from $n$, **order matters** |
| Binomial coefficient | $\displaystyle\binom{n}{k} = \frac{n!}{k!\,(n-k)!}$ | picking $k$ from $n$, **order ignored** |

Derivation: count ordered selections, then divide by the $k!$ orderings each set was counted with. → [04](./coursera/01_probability_distribution/04_binomial_coefficient.ipynb)

Useful: $\binom{n}{0} = \binom{n}{n} = 1$, $\binom{n}{k} = \binom{n}{n-k}$, row $n$ of Pascal's triangle.

---

## 4. Bayes' theorem

Module: [`00_introduction_to_probability`](./coursera/00_introduction_to_probability/), revisited in [`05_point_estimation`](./coursera/05_point_estimation/)

$$\boxed{\;P(A \mid B) = \frac{P(A)\,P(B \mid A)}{P(A)\,P(B \mid A) + P(A')\,P(B \mid A')}\;}$$

Named parts:

$$\underbrace{P(A \mid B)}_{\text{posterior}} = \frac{\overbrace{P(B \mid A)}^{\text{likelihood}} \cdot \overbrace{P(A)}^{\text{prior}}}{\underbrace{P(B)}_{\text{evidence}}}
\qquad\qquad \text{posterior} = \text{prior} \times \frac{\text{likelihood}}{\text{evidence}}$$

**When to use:** you know $P(\text{evidence} \mid \text{cause})$ but need $P(\text{cause} \mid \text{evidence})$ — medical tests, spam filters, any diagnosis.
**Key lesson:** a rare prior can swamp a very accurate test (99% accurate test, 1-in-10,000 disease → posterior ≈ 0.0098). → [10](./coursera/00_introduction_to_probability/10_bayes_theorem_disease_test.ipynb), [11](./coursera/00_introduction_to_probability/11_bayes_theorem_formula.ipynb)

### Naive Bayes

Assume the features are conditionally independent given the class:

$$P(\text{spam} \mid w_1,\dots,w_n) = \frac{P(\text{spam})\prod_{i=1}^{n} P(w_i \mid \text{spam})}{P(\text{spam})\prod_i P(w_i \mid \text{spam}) + P(\text{ham})\prod_i P(w_i \mid \text{ham})}$$

The assumption is false but makes the counts feasible — direct counting of word combinations runs out of data. → [14](./coursera/00_introduction_to_probability/14_naive_bayes.ipynb)

---

## 5. Random variables: PMF, PDF, CDF

Module: [`01_probability_distribution`](./coursera/01_probability_distribution/)

| | Discrete | Continuous |
|---|---|---|
| Function | PMF $p(x) = P(X = x)$ | PDF $f(x)$ — a **density**, not a probability |
| Non-negativity | $p(x) \ge 0$ | $f(x) \ge 0$ |
| Normalization | $\sum_x p(x) = 1$ | $\int_{-\infty}^{\infty} f(x)\,dx = 1$ |
| Probability of a value | $p(x)$ | $P(X = x) = 0$ always |
| Probability of a range | $\sum_{a \le x \le b} p(x)$ | $P(a \le X \le b) = \int_a^b f(x)\,dx$ (an **area**) |
| CDF | $F(x) = \sum_{t \le x} p(t)$ — a staircase | $F(x) = \int_{-\infty}^x f(t)\,dt$ — smooth |

→ [02](./coursera/01_probability_distribution/02_probability_mass_function.ipynb), [07](./coursera/01_probability_distribution/07_probability_density_function.ipynb), [08](./coursera/01_probability_distribution/08_cumulative_distribution_function.ipynb)

### CDF properties (both cases)

$$F_X(x) = P(X \le x), \qquad F(-\infty) = 0, \qquad F(+\infty) = 1, \qquad F \text{ is non-decreasing}$$

$$P(a < X \le b) = F(b) - F(a), \qquad f(x) = \frac{d}{dx}F(x) \;\;\text{(continuous)}$$

### Sampling from a distribution — inverse transform

Draw $U \sim \text{Uniform}(0,1)$, then $X = F^{-1}(U)$ has CDF $F$. Discrete version: stack the probabilities into $[0,1]$ and see which strip $U$ lands in. → [12](./coursera/01_probability_distribution/12_sampling_from_a_distribution.ipynb)

---

## 6. Named distributions

| Distribution | PMF / PDF | Mean | Variance | Use for |
|---|---|---|---|---|
| **Bernoulli**$(p)$ | $p(1)=p,\; p(0)=1-p$ | $p$ | $p(1-p)$ | one yes/no trial |
| **Binomial**$(n,p)$ | $\displaystyle\binom{n}{x}p^x(1-p)^{n-x}$ | $np$ | $np(1-p)$ | successes in $n$ independent trials |
| **Uniform**$(a,b)$ | $\dfrac{1}{b-a}$ on $[a,b]$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ | "any value in a window, equally likely" |
| **Normal**$(\mu,\sigma^2)$ | $\dfrac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | sums/averages of many effects |
| **Chi-squared** $\chi^2_k$ | $\dfrac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}$, $x>0$ | $k$ | $2k$ | sums of $k$ squared standard normals |
| **Student's** $t_\nu$ | heavier-tailed bell, → normal as $\nu\to\infty$ | $0$ ($\nu>1$) | $\frac{\nu}{\nu-2}$ ($\nu>2$) | means when $\sigma$ is unknown |

### Uniform, in full

$$f(x) = \begin{cases}\dfrac{1}{b-a} & a \le x \le b \\[2mm] 0 & \text{otherwise}\end{cases}
\qquad\qquad
F(x) = \begin{cases}0 & x < a \\[1mm] \dfrac{x-a}{b-a} & a \le x \le b \\[1mm] 1 & x > b\end{cases}$$

→ [09](./coursera/01_probability_distribution/09_uniform_distribution.ipynb)

### Normal, in full

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \qquad X \sim N(\mu, \sigma^2)$$

**Standardization** turns any normal into the standard normal:

$$Z = \frac{X - \mu}{\sigma} \sim N(0,1)$$

**68–95–99.7 rule:** ≈68% within $\mu \pm \sigma$, ≈95% within $\mu \pm 2\sigma$, ≈99.7% within $\mu \pm 3\sigma$. → [10](./coursera/01_probability_distribution/10_normal_distribution.ipynb), [06](./coursera/02_discribing_distribution/06_standard_deviation.ipynb)

### Chi-squared, from the ground up

$W = Z^2$ with $Z \sim N(0,1)$:

$$F_W(w) = 2\Phi(\sqrt{w}) - 1, \qquad f_W(w) = \frac{e^{-w/2}}{\sqrt{2\pi w}}, \quad w > 0$$

$$Z_1^2 + \dots + Z_k^2 \sim \chi^2_k, \qquad E[\chi^2_k] = k, \qquad \mathrm{Var}(\chi^2_k) = 2k$$

→ [11](./coursera/01_probability_distribution/11_chi_squared_distribution.ipynb)

---

## 7. Describing a distribution

Module: [`02_discribing_distribution`](./coursera/02_discribing_distribution/)

### Expected value (mean)

$$E[X] = \sum_x x\,p(x) \qquad\qquad E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx$$

The **balance point** of the distribution — a weighted average, not the halfway split. → [01](./coursera/02_discribing_distribution/01_mean_expected_value.ipynb)

### Expected value of a function

$$E[g(X)] = \sum_x g(x)\,p(x)$$

Transform the **values**, keep the **probabilities**. Note $E[g(X)] \ne g(E[X])$ in general — equality holds only for linear $g$. → [03](./coursera/02_discribing_distribution/03_expected_value_of_a_function.ipynb)

### Linearity of expectation

$$E[aX + b] = a\,E[X] + b \qquad\qquad E[X_1 + \dots + X_n] = E[X_1] + \dots + E[X_n]$$

**Sums of expectations hold even for dependent variables** — the workhorse for decomposing a hard count into indicator variables. → [04](./coursera/02_discribing_distribution/04_sum_of_expectations.ipynb)

### Variance and standard deviation

$$\mathrm{Var}(X) = E\big[(X - \mu)^2\big] = \sum_x (x-\mu)^2 p(x)$$

$$\boxed{\;\mathrm{Var}(X) = E[X^2] - (E[X])^2\;} \qquad \text{(shortcut — use this to compute)}$$

$$\mathrm{std}(X) = \sigma = \sqrt{\mathrm{Var}(X)}$$

Scaling rules:

$$\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X), \qquad \mathrm{std}(aX + b) = |a|\,\mathrm{std}(X)$$

Shifting does nothing to spread; scaling by $a$ multiplies variance by $a^2$. Standard deviation carries the **units of the data**; variance does not. → [05](./coursera/02_discribing_distribution/05_variance.ipynb), [06](./coursera/02_discribing_distribution/06_standard_deviation.ipynb)

### Variance of a sum (independent variables only)

$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) \qquad \text{(independent)}$$

**Variances add; standard deviations do not.** For dependent variables see [§8](#8-several-variables-at-once).

### Sums of Gaussians

If $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$ are independent, then for $W = aX + bY$:

$$W \sim N\big(a\mu_X + b\mu_Y,\;\; a^2\sigma_X^2 + b^2\sigma_Y^2\big)$$

A linear combination of independent Gaussians is Gaussian. Note $b = -1$ (a difference) still **adds** the variances. → [07](./coursera/02_discribing_distribution/07_sum_of_gaussians.ipynb)

### Standardizing (z-scores)

$$Z = \frac{X - \mu}{\sigma} \qquad\Longrightarrow\qquad E[Z] = 0,\; \mathrm{std}(Z) = 1$$

Center, then scale. The **shape survives** — only location and spread change. Use it to compare values from different scales. → [08](./coursera/02_discribing_distribution/08_standardizing_a_distribution.ipynb)

### Moments

$$\text{$k$-th raw moment} = E[X^k] = \sum_i p_i x_i^k \qquad\qquad \text{$k$-th central moment} = E\big[(X-\mu)^k\big]$$

→ [09](./coursera/02_discribing_distribution/09_moments_of_a_distribution.ipynb)

### The four descriptors

| # | Name | Formula | Reads |
|---|---|---|---|
| 1 | Mean | $\mu = E[X]$ | where it sits |
| 2 | Variance | $\sigma^2 = E[(X-\mu)^2]$ | how wide it is |
| 3 | **Skewness** | $E\!\left[\left(\dfrac{X-\mu}{\sigma}\right)^{3}\right]$ | which side the tail leans (`+` = right tail) |
| 4 | **Kurtosis** | $E\!\left[\left(\dfrac{X-\mu}{\sigma}\right)^{4}\right]$ | how heavy the tails are (normal = 3) |

Both are **standardized** moments, so both are scale-free. → [10](./coursera/02_discribing_distribution/10_skewness.ipynb), [11](./coursera/02_discribing_distribution/11_kurtosis.ipynb)

### Median, mode, quantiles

- **Median** = the middle value ($q_{0.5}$); resistant to outliers, unlike the mean.
- **Mode** = the most frequent value / the PDF's peak.
- **$k$% quantile** $q_{k/100}$: the value with $P(X \le q_{k/100}) = k/100$ — i.e. the **inverse CDF**.
- Quartiles: $Q_1 = q_{0.25}$, $Q_2 = q_{0.5}$ (median), $Q_3 = q_{0.75}$.

Right-skewed data: mode < median < mean. Symmetric unimodal: all three coincide. → [02](./coursera/02_discribing_distribution/02_median_and_mode.ipynb), [12](./coursera/02_discribing_distribution/12_quantiles.ipynb)

### Box plots

$$\mathrm{IQR} = Q_3 - Q_1 \qquad\qquad \text{fences: } Q_1 - 1.5\cdot\mathrm{IQR} \;\;\text{and}\;\; Q_3 + 1.5\cdot\mathrm{IQR}$$

Five-number summary: $(\min, Q_1, Q_2, Q_3, \max)$. Points beyond the fences are flagged as outliers. A box plot **cannot show multimodality**. → [13](./coursera/02_discribing_distribution/13_box_plots.ipynb)

### Kernel density estimation

$$\hat f(x) = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}}\,e^{-\frac{(x - x_i)^2}{2\sigma^2}}$$

One small Gaussian bump per data point, averaged. $\sigma$ is the **bandwidth**: too small → spiky, too large → over-smoothed. It integrates to 1, so it is a genuine density. Underlies violin plots. → [14](./coursera/02_discribing_distribution/14_kernel_density_estimation.ipynb), [15](./coursera/02_discribing_distribution/15_violin_plots.ipynb)

### QQ plots

Plot sample quantiles against theoretical (normal) quantiles. Straight line → normal. Upward-curving ends → heavy tails; a curve → skew. More sensitive than a histogram for judging normality. → [16](./coursera/02_discribing_distribution/16_qq_plots.ipynb)

---

## 8. Several variables at once

Module: [`03_probability_distribution_with_multiple_variables`](./coursera/03_probability_distribution_with_multiple_variables/)

### Joint distribution

$$P_{XY}(x,y) = P(X = x,\; Y = y) \qquad\qquad \sum_{x,y} P_{XY}(x,y) = 1$$

Continuous: $f_{XY}(x,y) \ge 0$, $\iint f_{XY}\,dx\,dy = 1$, and $P\big((X,Y) \in R\big) = \iint_R f_{XY}\,dx\,dy$.

**If independent:** $P_{XY}(x,y) = P_X(x)\,P_Y(y)$ — the joint table factorizes cell by cell. → [01](./coursera/03_probability_distribution_with_multiple_variables/01_joint_distribution.ipynb), [02](./coursera/03_probability_distribution_with_multiple_variables/02_joint_distribution_of_independent_variables.ipynb)

### Marginal distribution — sum out the other variable

$$P_X(x_i) = \sum_j P_{XY}(x_i, y_j) \qquad\qquad P_Y(y_j) = \sum_i P_{XY}(x_i, y_j)$$

$$f_X(x) = \int f_{XY}(x,y)\,dy \qquad\qquad f_Y(y) = \int f_{XY}(x,y)\,dx$$

Marginals **lose** the dependence information: different joints can share the same marginals. → [04](./coursera/03_probability_distribution_with_multiple_variables/04_marginal_distribution.ipynb)

### Conditional distribution — slice, then normalize

$$P_{Y\mid X=x}(y) = \frac{P_{XY}(x,y)}{P_X(x)} \qquad\qquad f_{Y\mid X=x}(y) = \frac{f_{XY}(x,y)}{f_X(x)}$$

Take the row/slice at $X = x$, divide by its total so it sums to 1. → [05](./coursera/03_probability_distribution_with_multiple_variables/05_conditional_distribution.ipynb)

### Covariance

$$\mathrm{Cov}(X,Y) = E\big[(X-\mu_X)(Y-\mu_Y)\big] = \sum_{x,y} P_{XY}(x,y)(x-\mu_X)(y-\mu_Y)$$

$$\boxed{\;\mathrm{Cov}(X,Y) = E[XY] - E[X]\,E[Y]\;} \qquad \text{(shortcut)}$$

From raw data: $\mathrm{Cov}(X,Y) = \dfrac{1}{n}\sum_i (x_i - \bar{x})(y_i - \bar{y})$ (use $n-1$ for the unbiased sample version, `np.cov`'s default).

Sign reading: $>0$ they rise together, $<0$ one rises as the other falls, $\approx 0$ no **linear** relation. Note $\mathrm{Cov}(X,X) = \mathrm{Var}(X)$. → [06](./coursera/03_probability_distribution_with_multiple_variables/06_covariance.ipynb), [07](./coursera/03_probability_distribution_with_multiple_variables/07_covariance_of_a_distribution.ipynb)

### Covariance matrix

$$\Sigma = \begin{bmatrix}\mathrm{Var}(X) & \mathrm{Cov}(X,Y)\\ \mathrm{Cov}(X,Y) & \mathrm{Var}(Y)\end{bmatrix}, \qquad\qquad \Sigma_{ij} = \mathrm{Cov}(X_i, X_j), \quad \Sigma_{ii} = \mathrm{Var}(X_i)$$

Symmetric, variances on the diagonal. For $n$ variables it holds $n(n+1)/2$ distinct numbers. → [08](./coursera/03_probability_distribution_with_multiple_variables/08_covariance_matrix.ipynb)

### Correlation coefficient

$$\rho_{XY} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y} = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X)}\sqrt{\mathrm{Var}(Y)}}, \qquad -1 \le \rho \le 1$$

Covariance of the **standardized** variables — immune to units and shifts, so magnitudes are comparable across pairs. Anscombe's quartet: same $\rho$, four different stories — always plot the data. → [09](./coursera/03_probability_distribution_with_multiple_variables/09_correlation_coefficient.ipynb)

### Multivariate Gaussian

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}\,|\Sigma|^{1/2}}\,\exp\left(-\tfrac12 (\mathbf{x}-\vec{\mu})^{T}\Sigma^{-1}(\mathbf{x}-\vec{\mu})\right)$$

Univariate ↔ multivariate dictionary: $\mu \to \vec{\mu}$, $\sigma^2 \to \Sigma$, $\frac{(x-\mu)^2}{\sigma^2} \to (\mathbf{x}-\vec{\mu})^T\Sigma^{-1}(\mathbf{x}-\vec{\mu})$, $\sigma \to |\Sigma|^{1/2}$.

Diagonal $\Sigma$ ⇔ independent components ⇔ the density is a product of 1-D bells. Off-diagonal terms tilt the level curves from circles toward a line. → [10](./coursera/03_probability_distribution_with_multiple_variables/10_multivariate_gaussian.ipynb)

---

## 9. Sampling, LLN, CLT

Module: [`04_sampling_and_point_estimation`](./coursera/04_sampling_and_point_estimation/)

### Sample statistics

| Quantity | Population | Sample estimator |
|---|---|---|
| Mean | $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$ | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ |
| Proportion | $p = \frac{\text{items with trait}}{N}$ | $\hat{p} = \frac{x}{n}$ |
| Variance | $\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$ | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ |

**Why $n-1$:** using $\bar{x}$ (rather than the unknown $\mu$) makes the deviations artificially small, so dividing by $n$ **underestimates** $\sigma^2$. Dividing by $n-1$ removes exactly that bias: $E[s^2] = \sigma^2$. → [02](./coursera/04_sampling_and_point_estimation/02_sample_mean.ipynb), [03](./coursera/04_sampling_and_point_estimation/03_sample_proportion.ipynb), [04](./coursera/04_sampling_and_point_estimation/04_sample_variance.ipynb)

A sample must be **i.i.d.**: independent (sample with replacement) and identically distributed (same rule every draw, no sorted/biased subgroups). → [01](./coursera/04_sampling_and_point_estimation/01_population_and_sample.ipynb)

### Sampling distribution of the mean

$$E[\bar{X}] = \mu \qquad\qquad \mathrm{Var}(\bar{X}) = \frac{\sigma^2}{n} \qquad\qquad \sigma_{\bar{X}} = \underbrace{\frac{\sigma}{\sqrt{n}}}_{\textbf{standard error}}$$

Accuracy improves like $1/\sqrt{n}$: **four times the data buys half the error**.

### Law of large numbers

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \;\longrightarrow\; E[X] = \mu \qquad \text{as } n \to \infty$$

Conditions: i.i.d. sample and a finite mean. This is what licenses simulation as a way to check theory. → [05](./coursera/04_sampling_and_point_estimation/05_law_of_large_numbers.ipynb)

### Central limit theorem

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \;\xrightarrow{\;n\to\infty\;}\; N(0,1) \qquad\qquad \frac{\sum_i X_i - n\mu}{\sqrt{n}\,\sigma} \;\xrightarrow{\;n\to\infty\;}\; N(0,1)$$

The **average of any** i.i.d. sample with finite variance becomes normal, whatever the population's shape. Skewed populations need larger $n$.

Discrete case (binomial as a sum of Bernoullis):

$$X \approx N\big(np,\; np(1-p)\big) \qquad \text{for large } n$$

→ [06](./coursera/04_sampling_and_point_estimation/06_central_limit_theorem_discrete.ipynb), [07](./coursera/04_sampling_and_point_estimation/07_central_limit_theorem_continuous.ipynb)

---

## 10. Point estimation: MLE, MAP, regularization

Module: [`05_point_estimation`](./coursera/05_point_estimation/)

### The one-line summary of the whole lesson

$$\underbrace{\arg\max_{\theta} P(\text{data}\mid\theta)}_{\text{MLE}} \;\longrightarrow\; \underbrace{\arg\max_{\theta} P(\text{data}\mid\theta)P(\theta)}_{\text{MAP}} = \arg\max_{\theta}\Big[\log P(\text{data}\mid\theta) + \underbrace{\log P(\theta)}_{\text{regularizer}}\Big]$$

### Maximum likelihood estimation

$$\hat\theta_{\text{MLE}} = \arg\max_{\theta} \; L(\theta), \qquad L(\theta) = P(\text{data} \mid \theta)$$

**Recipe:** write $L(\theta)$ as a product → take $\log$ (products become sums) → differentiate → set to 0 → solve.

| Model | Likelihood | MLE |
|---|---|---|
| Bernoulli, $k$ heads in $n$ | $L(p) = p^k(1-p)^{n-k}$ | $\hat p = \dfrac{k}{n}$ |
| Gaussian | $\prod_i \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$ | $\hat\mu = \bar{x}$, $\;\hat\sigma^2 = \frac{1}{n}\sum_i(x_i-\bar{x})^2$ |
| Linear regression, Gaussian noise | $\prod_i \frac{1}{\sqrt{2\pi}}e^{-\frac12 d_i^2}$ | minimize $\sum_i d_i^2$ |

$$\textbf{maximize } L \quad\Longleftrightarrow\quad \textbf{minimize } \sum_i d_i^2 \qquad \text{(MLE = least squares)}$$

→ [02](./coursera/05_point_estimation/02_mle_bernoulli.ipynb), [03](./coursera/05_point_estimation/03_mle_gaussian.ipynb), [04](./coursera/05_point_estimation/04_mle_linear_regression.ipynb)

Note $\hat\sigma^2_{\text{MLE}}$ divides by $n$ — it is the **biased** estimator; $s^2$ (divide by $n-1$) is the unbiased one.

### Regularization (L2 / ridge)

$$\text{regularized error} = \underbrace{\sum_j d_j^2}_{\text{fit}} + \lambda \underbrace{\sum_{i\ge1} a_i^2}_{\text{L2 penalty on coefficients}}$$

Large $\lambda$ → simpler model (underfit risk); small $\lambda$ → complex model (overfit risk). Judge on **test** loss, not train loss. → [05](./coursera/05_point_estimation/05_regularization.ipynb)

### Maximum a posteriori (MAP)

$$\hat\theta_{\text{MAP}} = \arg\max_\theta \; P(\text{data}\mid\theta)\,P(\theta) = \arg\max_\theta \; \text{posterior}$$

MAP = the **mode of the posterior**. With a fair-centered Beta-type prior $\propto p^a(1-p)^a$ ("$2a$ phantom flips") and $k$ heads in $n$ tosses:

$$\hat{p}_{\text{MAP}} = \frac{a + k}{2a + n}$$

- $a = 0$ (flat prior) ⇒ $\hat p = k/n$ = **the MLE**.
- Larger $a$ ⇒ the estimate is dragged toward the prior's center.
- More data melts any fixed prior.

→ [08](./coursera/05_point_estimation/08_maximum_a_posteriori.ipynb)

### Bayesian updating

$$f_{\Theta\mid X}(\theta\mid x) = \frac{p_{X\mid\Theta}(x\mid\theta)\,f_\Theta(\theta)}{p_X(x)} \qquad\Longrightarrow\qquad \text{posterior} \;\propto\; \text{likelihood} \times \text{prior}$$

The denominator is a constant in $\theta$, so it can be dropped when you only need the **mode**. **Yesterday's posterior is today's prior** — updating in chunks equals updating all at once:

$$\underbrace{\theta^{6}(1-\theta)^{4}}_{\text{new data}} \times \underbrace{\theta^{8}(1-\theta)^{2}}_{\text{old posterior}} = \theta^{14}(1-\theta)^{6} \;\Rightarrow\; \hat\theta_{\text{MAP}} = \tfrac{14}{20} = 0.7$$

→ [09](./coursera/05_point_estimation/09_bayesian_updating.ipynb), [10](./coursera/05_point_estimation/10_bayesian_worked_example.ipynb)

### The unifying identity

With a Gaussian prior on the coefficients and Gaussian noise on the data,

$$\boxed{\;\arg\max\; P(\text{data}\mid\text{model})\,P(\text{model}) \;=\; \arg\min\; \sum_j d_j^2 + \lambda\sum_i a_i^2\;}, \qquad \lambda = \frac{\sigma_d^2}{\sigma_a^2}$$

**L2-regularized regression *is* MAP estimation.** Taking logs turns the products into sums and the two penalties fall out. → [11](./coursera/05_point_estimation/11_map_mle_regularization.ipynb)

### Frequentist vs Bayesian

| | Frequentist | Bayesian |
|---|---|---|
| Probability means | long-run frequency | degree of belief |
| Parameter $\theta$ is | fixed, unknown | a random variable with a distribution |
| Uses a prior | no | yes |
| Point estimate | MLE | MAP (or posterior mean) |

They converge as data accumulates. → [07](./coursera/05_point_estimation/07_frequentist_vs_bayesian.ipynb)

---

## 11. Confidence intervals

Module: [`06_confidence_intervals`](./coursera/06_confidence_intervals/)

### The general shape

$$\text{CI} = \text{point estimate} \;\pm\; \underbrace{(\text{critical value}) \times (\text{standard error})}_{\text{margin of error (MOE)}}$$

### Mean, $\sigma$ **known** (z-interval)

$$\text{SE} = \frac{\sigma}{\sqrt{n}}, \qquad \text{MOE} = z_{1-\alpha/2}\cdot\frac{\sigma}{\sqrt{n}}, \qquad \text{CI} = \bar{x} \pm z_{1-\alpha/2}\frac{\sigma}{\sqrt{n}}$$

The flip that makes it work — the same inequality, rearranged:

$$P\!\left(\bar{X} - z_{1-\alpha/2}\tfrac{\sigma}{\sqrt{n}} \le \mu \le \bar{X} + z_{1-\alpha/2}\tfrac{\sigma}{\sqrt{n}}\right) = 1-\alpha$$

Assumptions: normal population (or $n$ large enough for the CLT), i.i.d. sample, known $\sigma$. → [03](./coursera/06_confidence_intervals/03_margin_of_error.ipynb), [04](./coursera/06_confidence_intervals/04_calculating_confidence_intervals.ipynb)

### Mean, $\sigma$ **unknown** (t-interval)

$$T = \frac{\bar{X}-\mu}{s/\sqrt{n}} \sim t_{\nu}, \qquad \nu = n - 1, \qquad \text{CI} = \bar{x} \pm t_{\nu,\,1-\alpha/2}\cdot\frac{s}{\sqrt{n}}$$

Estimating $\sigma$ by $s$ adds extra uncertainty → **fatter tails** → wider intervals. $t_\nu \to N(0,1)$ as $\nu$ grows. → [07](./coursera/06_confidence_intervals/07_t_distribution.ipynb)

### Proportion

$$\text{CI} = \hat{p} \;\pm\; z_{1-\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}$$

The standard error comes from the Bernoulli variance $p(1-p)$, and it **peaks at $p = 0.5$** — fifty-fifty questions are the hardest to poll. → [08](./coursera/06_confidence_intervals/08_confidence_intervals_for_proportions.ipynb)

### Sample size planning

$$\boxed{\;n \;\ge\; \left(\frac{z_{1-\alpha/2}\cdot\sigma}{\text{MOE}}\right)^{2}\;} \qquad\qquad n \;\ge\; \frac{z_{1-\alpha/2}^{2}\;p(1-p)}{\text{MOE}^{2}} \quad \text{(proportions)}$$

Both precision and confidence are paid for **quadratically**: halving the margin costs 4× the data. Use $p = 0.5$ for the worst-case poll size. Always round **up**. → [05](./coursera/06_confidence_intervals/05_choosing_sample_size.ipynb)

### The two dials

| Dial | Effect on the interval |
|---|---|
| $n \uparrow$ | narrower (via $\sigma/\sqrt{n}$) |
| confidence $\uparrow$ (e.g. 95% → 99%) | wider (via a larger $z$) |

→ [02](./coursera/06_confidence_intervals/02_changing_the_interval.ipynb)

### Confidence is not probability

"95% confident" describes **the procedure**, not this one interval: repeat the sampling many times and 95% of the intervals produced will contain $\mu$. Once computed, an interval either contains the fixed $\mu$ or it doesn't — probability 1 or 0. → [06](./coursera/06_confidence_intervals/06_confidence_vs_probability.ipynb)

---

## 12. Hypothesis testing

Module: [`07_hypothesis_testing`](./coursera/07_hypothesis_testing/)

### The four steps

1. State $H_0$ (status quo, always with $=$) and $H_1$ (the claim to be proven).
2. Choose $\alpha$ (usually 0.05) and the test statistic.
3. Compute the observed statistic and its p-value (or compare against critical values).
4. Decide: $p < \alpha$ → reject $H_0$; $p \ge \alpha$ → **fail to reject** $H_0$.

→ [07](./coursera/07_hypothesis_testing/07_steps_and_interpretation.ipynb)

### Errors

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | **Type I error** (prob. $\alpha$) — false positive | correct (prob. = power) |
| **Fail to reject** | correct | **Type II error** (prob. $\beta$) — false negative |

$$\alpha = \max P(\text{reject } H_0 \mid H_0 \text{ true}), \qquad \beta(\mu) = P(\text{fail to reject} \mid \mu), \qquad \text{power}(\mu) = 1 - \beta(\mu)$$

Lowering $\alpha$ raises $\beta$ at fixed $n$ — the only escape is a larger sample. The power curve starts at $\alpha$ (at the boundary of $H_0$) and rises toward 1 as the true $\mu$ moves away. → [02](./coursera/07_hypothesis_testing/02_type_errors_and_significance.ipynb), [06](./coursera/07_hypothesis_testing/06_type_ii_errors_and_power.ipynb)

### The three tails

| Alternative | Name | p-value | Reject when |
|---|---|---|---|
| $H_1: \mu > \mu_0$ | right-tailed | $P(T \ge t_{obs})$ | $t_{obs} > t_{1-\alpha}$ |
| $H_1: \mu < \mu_0$ | left-tailed | $P(T \le t_{obs})$ | $t_{obs} < t_{\alpha}$ |
| $H_1: \mu \ne \mu_0$ | two-tailed | $2\,P(T \ge \lvert t_{obs}\rvert)$ | $\lvert t_{obs}\rvert > t_{1-\alpha/2}$ |

The two-tailed p-value is **double** the one-tailed one — the same data can reject in one tail and fail in two. → [03](./coursera/07_hypothesis_testing/03_right_left_two_tailed_tests.ipynb), [04](./coursera/07_hypothesis_testing/04_p_values.ipynb)

### p-value vs critical value

$$p\text{-value} = P(\text{a statistic at least this extreme} \mid H_0 \text{ true})$$

Critical value = the threshold on the original scale, computable **before** seeing the data, e.g. right-tailed:

$$k_\alpha = \mu_0 + z_{1-\alpha}\cdot\frac{\sigma}{\sqrt{n}} \qquad\Rightarrow\qquad \text{reject if } \bar{x} > k_\alpha$$

The two methods always agree. → [05](./coursera/07_hypothesis_testing/05_critical_values.ipynb)

### The test statistics

| Test | Statistic | Distribution under $H_0$ |
|---|---|---|
| One-sample, $\sigma$ known | $Z = \dfrac{\bar{X}-\mu_0}{\sigma/\sqrt{n}}$ | $N(0,1)$ |
| One-sample, $\sigma$ unknown | $T = \dfrac{\bar{X}-\mu_0}{S/\sqrt{n}}$ | $t_{n-1}$ |
| Two-sample (Welch) | $T = \dfrac{(\bar{X}-\bar{Y}) - (\mu_X-\mu_Y)}{\sqrt{\dfrac{s_x^2}{n_x} + \dfrac{s_y^2}{n_y}}}$ | $t_\nu$ (Welch $\nu$, below) |
| Paired | $T = \dfrac{\bar{D}-\mu_D}{S_D/\sqrt{n}}$, $\;D_i = X_i - Y_i$ | $t_{n-1}$ |
| Two proportions (pooled) | $Z = \dfrac{\frac{X}{n_A} - \frac{Y}{n_B}}{\sqrt{\hat p(1-\hat p)\left(\frac{n_A+n_B}{n_A n_B}\right)}}$, $\;\hat p = \frac{X+Y}{n_A+n_B}$ | $\approx N(0,1)$ |

with $S = \sqrt{\frac{1}{n-1}\sum_i (X_i - \bar{X})^2}$.

**Welch degrees of freedom** (let software compute it):

$$\nu = \frac{\left(\frac{s_x^2}{n_x} + \frac{s_y^2}{n_y}\right)^{2}}{\frac{(s_x^2/n_x)^2}{n_x-1} + \frac{(s_y^2/n_y)^2}{n_y-1}}$$

Two-sample sampling distribution: $\bar{X}-\bar{Y} \sim N\!\left(\mu_X-\mu_Y,\; \frac{\sigma_x^2}{n_x} + \frac{\sigma_y^2}{n_y}\right)$ — variances add.

→ [08](./coursera/07_hypothesis_testing/08_t_statistic.ipynb), [09](./coursera/07_hypothesis_testing/09_t_tests.ipynb), [10](./coursera/07_hypothesis_testing/10_two_sample_t_test.ipynb), [11](./coursera/07_hypothesis_testing/11_paired_t_test.ipynb), [12](./coursera/07_hypothesis_testing/12_ab_testing.ipynb)

### Paired vs two-sample

If each observation in group A is naturally matched with one in group B (before/after on the same subject), take the differences $d_i = x_i - y_i$ and run a **one-sample** t-test on them. Pairing removes between-subject variation, so it detects effects the two-sample test misses. → [11](./coursera/07_hypothesis_testing/11_paired_t_test.ipynb)

### A/B testing

Hypothesis testing applied to a product experiment: split users into A (control) and B (variant), define the metric, then pick the matching test — continuous metric (purchase amount) → **two-sample t-test**; binary metric (conversion) → **pooled two-proportion z-test**. → [12](./coursera/07_hypothesis_testing/12_ab_testing.ipynb)

---

## 13. Critical-value tables

### Normal (z)

| Confidence | $\alpha$ | Two-tailed $z_{1-\alpha/2}$ | One-tailed $z_{1-\alpha}$ |
|---|---|---|---|
| 80% | 0.20 | 1.282 | 0.842 |
| 90% | 0.10 | 1.645 | 1.282 |
| **95%** | **0.05** | **1.960** | **1.645** |
| 98% | 0.02 | 2.326 | 2.054 |
| 99% | 0.01 | 2.576 | 2.326 |

### Student's t

| df ($\nu$) | $t_{0.95}$ (one-tailed 5%) | $t_{0.975}$ (two-tailed 5%) | $t_{0.995}$ (two-tailed 1%) |
|---|---|---|---|
| 1 | 6.314 | 12.706 | 63.657 |
| 2 | 2.920 | 4.303 | 9.925 |
| 5 | 2.015 | 2.571 | 4.032 |
| 9 | 1.833 | 2.262 | 3.250 |
| 10 | 1.812 | 2.228 | 3.169 |
| 15 | 1.753 | 2.131 | 2.947 |
| 20 | 1.725 | 2.086 | 2.845 |
| 30 | 1.697 | 2.042 | 2.750 |
| 60 | 1.671 | 2.000 | 2.660 |
| 120 | 1.658 | 1.980 | 2.617 |
| $\infty$ | 1.645 | 1.960 | 2.576 |

---

## 14. Which procedure do I use?

**Estimating a single mean**

- $\sigma$ known, normal population or large $n$ → **z-interval / z-test**
- $\sigma$ unknown (the usual case) → **t-interval / t-test**, df $= n-1$

**Estimating a proportion** → $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$

**Comparing two groups**

- independent groups, continuous outcome → **two-sample (Welch) t-test**
- matched pairs / before-after → **paired t-test** (one-sample t-test on the differences)
- independent groups, binary outcome → **pooled two-proportion z-test**

**Finding a parameter from data**

- no prior beliefs → **MLE**
- prior beliefs, or you want regularization → **MAP** (equivalently, penalized MLE)

**Checking normality** → **QQ plot** (more sensitive than a histogram)

**Seeing a distribution's shape** → histogram, **KDE**, box plot (summary), violin plot (both)

---

## 15. Python cookbook

> These are the **scipy** one-liners, for when you have it. The course notebooks deliberately do *not* use scipy — they build each distribution from `math.erf`, `math.gamma` and `np.trapezoid`. For those implementations, and how every formula above is coded, see [PYTHON.md](./PYTHON.md).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# numpy<2 compatibility shim used across the notebooks
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
```

| Task | Code |
|---|---|
| Binomial coefficient | `math.comb(n, k)` |
| Binomial PMF / CDF | `stats.binom.pmf(k, n, p)` / `stats.binom.cdf(k, n, p)` |
| Normal PDF / CDF / quantile | `stats.norm.pdf(x, mu, sd)` / `.cdf(...)` / `.ppf(q, mu, sd)` |
| t CDF / critical value | `stats.t.cdf(t, df)` / `stats.t.ppf(1 - alpha/2, df)` |
| Chi-squared | `stats.chi2.pdf(x, k)` |
| Sample mean / std | `x.mean()` / `x.std(ddof=1)` ← `ddof=1` gives $s$ |
| Variance ($n-1$) | `np.var(x, ddof=1)` |
| Quantiles / IQR | `np.percentile(x, [25, 50, 75])` |
| Skewness / kurtosis | `stats.skew(x)` / `stats.kurtosis(x, fisher=False)` |
| Covariance matrix | `np.cov(X, Y)` (uses $n-1$) |
| Correlation | `np.corrcoef(X, Y)` / `df.corr()` |
| One-sample t-test | `stats.ttest_1samp(x, popmean)` |
| Two-sample t-test (Welch) | `stats.ttest_ind(x, y, equal_var=False)` |
| Paired t-test | `stats.ttest_rel(x, y)` |
| Sampling | `rng = np.random.default_rng(42)`; `rng.normal(mu, sd, n)`, `rng.binomial(n, p, size)`, `rng.choice(...)` |
| KDE | `stats.gaussian_kde(x)` |
| QQ plot | `stats.probplot(x, dist="norm", plot=plt)` |
| Numeric area under a curve | `np.trapezoid(y, x)` |

Note: `stats.ttest_*` return **two-tailed** p-values — halve them for a one-tailed test (after checking the sign of the statistic).

---

## 16. Traps and misreadings

- **$P(A\cup B) = P(A)+P(B)$ only if disjoint.** Otherwise subtract $P(A\cap B)$.
- **$P(A\cap B) = P(A)P(B)$ only if independent.** Otherwise use $P(A)P(B\mid A)$.
- **A rare prior beats an accurate test.** Always run the numbers through Bayes rather than trusting "99% accurate".
- **A PDF value is not a probability.** It can exceed 1; only areas are probabilities. $P(X=x) = 0$ for continuous $X$.
- **$E[g(X)] \ne g(E[X])$** unless $g$ is linear.
- **Standard deviations do not add; variances do** (and only for independent variables).
- **Variance of a difference adds too**: $\mathrm{Var}(X-Y) = \mathrm{Var}(X)+\mathrm{Var}(Y)$ when independent.
- **Divide by $n-1$** for a sample variance; MLE's $\hat\sigma^2$ (divide by $n$) is biased low.
- **Correlation is not causation, and $\rho$ only measures *linear* association.** Anscombe's quartet.
- **Same marginals ≠ same joint.** Marginalizing destroys dependence information.
- **"95% confident" is about the procedure, not this interval.** $\mu$ is not random.
- **The p-value is not $P(H_0 \text{ true})$**, and it is not the probability that the result was chance.
- **"Fail to reject $H_0$" is not "$H_0$ is true"** — a large p-value can simply mean too little data.
- **Choose the tail before seeing the data.** Switching to a one-tailed test after a two-tailed test fails is p-hacking.
- **Use `ddof=1`** in numpy for sample standard deviations; the default `ddof=0` divides by $n$.
