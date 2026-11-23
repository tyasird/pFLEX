# Spec: half-max AUC (early precision, global PRA, pFLEX 2.0)

## Why

Global AUPRC averages precision over all of recall, so the long low-precision
tail (thousands of TPs near background) decides the value. A dataset that is
clearly better among its top predictions can lose to one with a long, weak
tail. The half-max AUC only scores the high-precision head of the curves, in
one TP window that is the same for all datasets.

## Scope

Global PRA only. Standard pFLEX AUPRC (`pra()`, `pr_auc`,
`corrected_pr_auc`) is unchanged: it is still computed over linear recall.
Per-module AUPRC and the scatter plots are unchanged.

## Axis and weighting

The score is computed on the **linear TP axis**: every TP in the window has
the same weight. There is no log scaling and no other weighting. PR plots use a
log TP axis for reading only; the numbers do not follow the plot. Any weighting
(e.g. giving earlier TPs more weight) is a separate method change that has to be
proposed, described and benchmarked before it is implemented.

## Definition

Input: the score-sorted global pair table of every dataset from `pra()`.
Datasets are compared over the same genes (`analysis_genes="shared"`), so they
have the same number of true positives `P`, pairs `N` and background.

1. **Curve points.** One point per true positive: for the k-th TP found at rank
   `r_k`, `p(k) = k / r_k`.
2. **Start for max and crossing.** `K0 = 1` by default: both are searched from
   the first TP. Option `k0=10` starts the search at the 10th TP, where the log
   plot starts (the first TPs are noisy). The area always starts at TP 1.
3. **Background.** `p_bg = P / N`, the precision of a random ranking. The same
   for every dataset; if it differs, a warning is logged and the reference
   dataset's value is used for the threshold.
4. **Max (one for all datasets).** The highest `p(k)`, `k >= K0`, of any
   dataset (default `max_window = 1`: the plain max). Option `max_window = 3`:
   the highest precision held for 3 consecutive TPs (the max of the rolling
   minimum over 3 TPs), so a single early hit cannot set it. The dataset it
   comes from is the reference (`threshold_from`). Option `max_rank = 3`: the
   3rd highest precision instead of the highest, so one or two lucky early
   hits cannot set it (a curve at 1, 1, 1, ... still gives 1). Default 1.
5. **Threshold (one for all datasets).**
   `p_threshold = p_bg + f × (p_max - p_bg)`, with `f` = 1/2 (half max,
   **default**), 3/4 or 1/4; `fraction="half"`, `"three_quarter"`, `"quarter"`
   or a number in (0, 1). Example: max 0.8, bg 0.1 → 3/4: 0.625, 1/2: 0.45,
   1/4: 0.275. A lower fraction cuts later.
6. **Crossing (per dataset).** `k_cross` = the first `k >= K0` with
   `p(k) < p_threshold`, interpolated linearly between the last TP above and
   the first TP below. Later recoveries are ignored. `K0` if the curve starts
   below the threshold, `P` if it never falls below.
7. **Window (one for all datasets).** `K = max k_cross` over the datasets
   (`cut_from` names the dataset). Every dataset is scored over TP 1 to K.
8. **Score.**

   ```text
   area           = ∫ p(k) dk   from k = 1 to k = K     (trapezoid, linear TP axis)
   mean_precision = area / (K - 1)                     (the average precision over the window)
   score          = (mean_precision - p_bg) / (1 - p_bg)
   ```

   Measured from the background: a random ranking scores 0, a perfect ranking
   1, worse than random below 0 (not clipped). The window and the background
   are the same for every dataset.

A dataset that falls below the threshold before `K` keeps its lower precision
up to `K` in the average, so dropping early lowers its score.

## API

- `half_max_auc(prediction, k0=1, threshold=None, k_end=None, max_window=1, fraction=0.5, max_rank=1)`:
  one ranked label vector. Without `threshold` it uses the dataset's own
  threshold; without `k_end` its own crossing as the window end.
- `half_max_auc_scores(fraction=0.5, k0=1, max_window=1, max_rank=1)`: reads every cached
  `pra` result, computes the shared threshold and window, scores all datasets,
  saves each under `half_max_auc` and returns `{dataset: result}`. Plots and
  `save_results_to_csv()` call it, so scores always reflect the current set of
  datasets.

Each result holds:

```text
score            (mean precision - background) / (1 - background)
mean_precision   average precision over TP 1 to K
area             trapezoid area under precision, TP 1 to K (linear TP axis)
p_max, k_max     this dataset's own max (from K0, held max_window TPs) and its TP
p_bg             background P / N
fraction         3/4, 1/2 or 1/4
p_threshold      shared threshold;   threshold_from: reference dataset
k_cross          this dataset's own crossing
k_cut            shared window end K; cut_from: dataset it comes from
n_tp (P), n_pairs (N), k0, max_window, max_rank, reason
```

## Edge cases

- `P < K0` (fewer than 10 TPs): score NaN with a reason.
- No dataset with precision above background: all scores NaN.
- Every dataset below the threshold from `K0`: `K = K0`, the window is TP 1 to
  `K0`.
- Window end beyond a dataset's last TP (only with different gene sets): cut
  at its last TP, with a reason.
- Point exactly on the threshold counts as above.

## Plotting

- `plot_precision_recall_curve(half_max=True, half_max_fraction=0.5)`: on the
  usual log-TP plot, the shared threshold (grey dashed), each dataset's
  crossing (dot) and the window end K (grey dotted). No area is shaded there,
  because the score is not an area on a log axis.
- `plot_half_max_auc_scores(fraction=0.5)`: one bar per dataset
  (`half_max_auc_values`, `three_quarter_max_auc_values`,
  `quarter_max_auc_values`).
- `plot_half_max_explainer(fraction=0.5)` (optional): step-by-step figure on
  the linear TP axis, with the scored area, the formula and a results table.
- `save_results_to_csv()` writes `csv/half_max_auc.csv` (default fraction).
