# Image corrections

Eight of the 13 illustrated-solution images in this folder contain errors and are **not embedded** in the notebooks. Each entry below lists what is wrong and the exact data the regenerated image must use. All numbers come from executing the notebooks' own solution cells.

Once an image is regenerated and checked, re-embed it as a markdown cell directly after its solution cell:

```html
**Exercise K, illustrated.** A one-page visual summary of the solution above.

<img src="img/FILENAME.png" alt="Illustrated solution to Exercise K" width="900">
```

Already correct and embedded: `03_ex3`, `03_ex6`, `04_ex3`, `04_ex4`, `04_ex6`.

---

## Shared data

### Children table (notebooks 04 and 05)

Ages **7–10**, heights **45–50 inches**. Not ages 5–8, and not heights 45/50/55/60.

| age \ height | 45 | 46 | 47 | 48 | 49 | 50 | P(age) |
|---|---|---|---|---|---|---|---|
| 7  | 0.1 | 0.2 | 0   | 0 | 0   | 0   | **0.3** |
| 8  | 0   | 0   | 0.2 | 0 | 0   | 0   | **0.2** |
| 9  | 0   | 0   | 0   | 0 | 0.3 | 0.1 | **0.4** |
| 10 | 0   | 0   | 0   | 0 | 0   | 0.1 | **0.1** |
| **P(height)** | **0.1** | **0.2** | **0.2** | **0** | **0.3** | **0.2** | **1.0** |

Conditionals P(height | age), each row sums to 1:

| age | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|
| 7  | 1/3 | 2/3 | 0 | 0 | 0    | 0    |
| 8  | 0   | 0   | 1 | 0 | 0    | 0    |
| 9  | 0   | 0   | 0 | 0 | 0.75 | 0.25 |
| 10 | 0   | 0   | 0 | 0 | 0    | 1    |

### Call-center data (notebook 03)

100,000 customers: X = waiting time, Y = rating, both on [0, 10]. The relationship is **negative**: the data sits in two corners, **top-left** (short wait, high rating) and **bottom-right** (long wait, low rating).

| quadrant | share |
|---|---|
| X < 5, Y ≥ 5 (top-left)     | 0.459 |
| X ≥ 5, Y < 5 (bottom-right) | 0.437 |
| X < 5, Y < 5                | 0.043 |
| X ≥ 5, Y ≥ 5                | 0.062 |

Both marginals are broad and bimodal, with a shallow dip near 5. There is real mass across the whole range, not two narrow spikes. Counts per unit-width bin from 0 to 10:

- X: 9361, 10685, 12472, 10333, 7328, 7238, 10243, 12198, 10667, 9475
- Y: 8610, 9430, 11689, 10559, 7683, 6853, 8826, 11288, 11231, 13831

---

## Notebook 03: continuous joint distributions

### `03_ex4_marginals_vs_joint_distribution.png` (Exercise 4)

**Wrong:**
- Panel 1 shows a **positive** relationship (small X with small Y). The real data is negative (see above).
- Cov(X, Y) = 8.717 and correlation = 0.993 are invented.
- The marginal histograms show two narrow spikes with nothing in between, and their y-axis (counts up to 600) implies about 5,000 points instead of 100,000.

**Correct values:**

| | original | shuffled |
|---|---|---|
| E[X] | 4.996 | 4.996 |
| E[Y] | 5.274 | 5.274 |
| Var(X) | 8.778 | 8.778 |
| Var(Y) | 9.338 | 9.338 |
| Cov(X, Y) | **≈ −6.70** | ≈ 0.00 |
| Correlation | **≈ −0.74** | ≈ 0.00 |

- Original scatter: two clouds, top-left and bottom-right.
- Shuffled scatter: four clouds, one in each corner.
- The notebook's solution prints only the means and variances. If covariance is shown, label it as a preview of the next notebook.

### `03_ex5_joint_density.png` (Exercise 5)

**Wrong:**
- The heatmap shows a single blob centered near (5, 5). The real density is concentrated in the top-left and bottom-right corners.
- The corner probability is given as **0.0832**. The correct value is **0.2334**.

**Correct values:**
- Grid: 60 × 60 bins on [0, 10]², so Δx = Δy = 1/6 and cell area = 1/36 ≈ 0.02778. The image already has this right.
- Total volume = sum(density × cell area) = **1.0000**.
- Region X ≤ 3, Y ≥ 7: grid volume = **0.2334**, direct count = **0.2334**.
- The highlighted top-left region should look **dense**, since it covers part of one of the two main clusters.

---

## Notebook 04: marginal distributions

### `04_ex5_different_joints_same_marginals.png` (Exercise 5)

**Wrong:**
- Uses an invented children table (ages 5–8, heights 45/50/55/60). Use the shared children table above.
- The invented product table also has a multiplication slip: cell (7, 55) shows 0.055, but 0.25 × 0.26 = 0.065.

**Correct values:**
- Marginals: P(age) = (0.3, 0.2, 0.4, 0.1) for ages 7–10. P(height) = (0.1, 0.2, 0.2, 0, 0.3, 0.2) for heights 45–50.

Independence product table P(age) · P(height):

| age \ height | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|
| 7  | 0.03 | 0.06 | 0.06 | 0 | 0.09 | 0.06 |
| 8  | 0.02 | 0.04 | 0.04 | 0 | 0.06 | 0.04 |
| 9  | 0.04 | 0.08 | 0.08 | 0 | 0.12 | 0.08 |
| 10 | 0.01 | 0.02 | 0.02 | 0 | 0.03 | 0.02 |

Absolute difference |product − true|:

| age \ height | 45 | 46 | 47 | 48 | 49 | 50 |
|---|---|---|---|---|---|---|
| 7  | 0.07 | 0.14 | 0.06 | 0 | 0.09 | 0.06 |
| 8  | 0.02 | 0.04 | 0.16 | 0 | 0.06 | 0.04 |
| 9  | 0.04 | 0.08 | 0.08 | 0 | **0.18** | 0.02 |
| 10 | 0.01 | 0.02 | 0.02 | 0 | 0.03 | 0.08 |

- Largest difference at (age, height) = **(9, 49)**: product 0.12 vs true 0.30.
- Row and column sums of both tables match.
- Wrong conclusion drawn from the product table: "a 7-year-old is as likely to be 50 inches tall as anyone". Under the true joint, P(7, 50) = 0.

---

## Notebook 05: conditional distributions

All five images below must use the shared children table above. Exercise 4 is the exception: it uses the two-dice table.

### `05_ex1_compute_conditional_probability.png` (Exercise 1)

**Wrong:** the answer and the age-7 row are correct, but the other rows are invented (ages 5, 6, 8) and the table **sums to 1.10**.

**Correct:**
- Slice for age 7: (0.1, 0.2, 0, 0, 0, 0). Row sum P(X = 7) = 0.3.
- P(Y | X = 7) = (1/3, 2/3, 0, 0, 0, 0) ≈ (0.333, 0.667, 0, 0, 0, 0).

### `05_ex2_compute_conditional_probability.png` (Exercise 2)

**Wrong:** the answer is correct, but the table is invented (ages 5–10) and **sums to 1.15**. The column for height 49 is labeled 0.10, but its entries add up to 0.15.

**Correct:**
- Column for height 50, by age 7, 8, 9, 10: (0, 0, 0.1, 0.1). Column sum P(Y = 50) = 0.2.
- P(X | Y = 50) = (0, 0, 0.5, 0.5): half of the 50-inch children are 9 and half are 10.
- Only ages 7–10 exist, so the bar chart should have 4 bars, not 6.

### `05_ex3_conditionals_sum_to_1.png` (Exercise 3)

**Wrong:**
- Uses the same broken table as `05_ex2` (sums to 1.15, and the height-49 total is wrong).
- Its example P(Y | X = 7) = (0, ½, ½, 0, 0, 0) contradicts Exercise 1.

**Correct:**
- Use the shared children table and the conditionals table above.
- Worked example for X = 7: (1/3, 2/3, 0, 0, 0, 0).
- The reconstruction P(X = x) · P(Y | X = x) gives back the joint exactly. The code and output panel in the image are already correct.

### `05_ex4_symmetry_sum_given_first_die.png` (Exercise 4)

**Wrong:** panel 2 shows only one 1/36 per row, along a diagonal. With that table, the S = 7 column holds only one nonzero cell, which contradicts panel 3.

**Correct joint table P(X = x, S = s):** row x has **1/36 in every column s = x+1, …, x+6** and 0 elsewhere. It's a band six cells wide.

| X \ S | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 | 1/36 |
| **P(S)** | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | 6/36 | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

- The S = 7 column is 1/36 in **every** row. That's why P(S = 7 | X = x) = (1/36) / (6/36) = 1/6 for all x.
- Panels 3–6 of the current image are correct.

### `05_ex5_independence_via_conditionals.png` (Exercise 5)

**Wrong:** the dice half is correct. The children half uses the invented table:
- It shows P(Y | X = 9) = (0, 0, 0, .25, .25, .50) and P_Y = (.10, .30, .25, .15, .10, .20).

**Correct:**
- P(Y | X = 9) = **(0, 0, 0, 0, 0.75, 0.25)**.
- Marginal P_Y = **(0.1, 0.2, 0.2, 0, 0.3, 0.2)**.
- These differ, so age and height are not independent. The image's conclusion is right; only the numbers need fixing.
- The equivalence argument in panel 3 and the takeaways in panel 4 are correct.
