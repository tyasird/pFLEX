# pFLEX

![PyPI](https://img.shields.io/badge/pypi-v1.1-orange)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Tests](https://github.com/tyasird/pFLEX/actions/workflows/tests.yml/badge.svg)](https://github.com/tyasird/pFLEX/actions/workflows/tests.yml)
![Build system](https://img.shields.io/badge/build-hatchling-blue)
![Lint](https://img.shields.io/badge/lint-ruff-46a2f1)
![CLI](https://img.shields.io/badge/CLI-pFLEX-brightgreen)

**Abstract**

Genetic networks derived from omics data are a powerful tool for systematic gene function prediction. Performance evaluation of such predictions is crucial to judge the data and computational pipeline for network construction, but unbalanced functional standards often cause hidden evaluation biases. To visualize and mitigate such biases, we previously developed the R package FLEX. Here, we present the pFLEX genetic network benchmarking tool as Python library with new and improved functionality. pFLEX improves overall runtime 4.1 to 15.8-fold. It offers additional evaluation metrics that allow for easy comparison of precision recall performance at the module or pathway resolution between genetic networks. We demonstrate the utility of pFLEX for evaluating tissue-specific co-essentiality networks and data normalization strategies of the Cancer Dependency Map, as well as for cell line-specific Perturb-Seq-derived networks. This illustrates the requirement for biological module-resolved precision recall metrics in pFLEX for sensitive and fast evaluation of genetic networks.

---

## Features

- Precision-recall curve generation for ranked gene lists
- Evaluation using CORUM-derived modules, GO terms, and pathways
- Module-level resolution analysis and visualization
- Half-max AUC: an early-precision score that compares datasets on the top of their ranking
- Easy integration into CRISPR screen workflows
- Packaged DepMap example inputs filtered to CORUM genes

---

## Installation

pFLEX is developed and tested with Python 3.10. We recommend installing it in a dedicated Python 3.10 environment to keep the package and its scientific Python dependencies separate from other projects.

Create `venv`:

```bash
conda create -n p310 python=3.10
conda activate p310
pip install uv
```

Install pFLEX via pip:

```bash
uv pip install pflex
```

or:

```bash
pip install pflex
```

or install pFLEX via git to develop the package locally:

```bash
git clone https://github.com/tyasird/pFLEX.git
cd pFLEX
uv pip install -e .
```

---

## Usage

Full documentation is available at [https://tyasird.github.io/pFLEX/](https://tyasird.github.io/pFLEX/).

### Input Data

pFLEX expects each input dataset as a matrix with genes in rows and screens, samples, or cell lines in columns.

| Gene | ACH-000014 | ACH-000219 | ACH-000274 |
| --- | ---: | ---: | ---: |
| A2M | -0.125 | -0.215 | 0.065 |
| AATF | 0.042 | -0.088 | -0.016 |
| BCL6 | -0.019 | 0.112 | -0.074 |

CSV, Excel, and Parquet files are supported. Parquet is recommended for larger matrices.

The packaged example inputs are real DepMap 25Q2 tissue subsets filtered to genes present in CORUM:

- `skin_cell_lines_corum_genes.parquet`: 3,465 genes x 75 cell lines
- `soft_tissue_cell_lines_corum_genes.parquet`: 3,465 genes x 46 cell lines

Use `flex.example_input_path()` to resolve packaged example inputs:

```python
import pflex as flex

inputs = {
    "Skin": {
        "path": flex.example_input_path("skin_cell_lines_corum_genes.parquet"),
        "sort": "high",
        "color": "#4E79A7",
    },
    "Soft Tissue": {
        "path": flex.example_input_path("soft_tissue_cell_lines_corum_genes.parquet"),
        "sort": "high",
        "color": "#F28E2B",
    },
}
```

### Configuration

```python
config = {
    "functional_standard": "CORUM",
    "min_genes_in_module": 2,
    "min_genes_per_module_analysis": 2,
    "output_folder": "output",
    "analysis_genes": "shared",
    "jaccard": True,
    "preprocessing": {
        "fill_na": True,
    },
    "corr_function": "numpy_without_mask",
    "per_module": {
        "n_jobs": 8,
    },
    "plotting": {
        "save_plot": True,
        "output_type": "pdf",
        "figure_scale": 0.75,  # plot-area size; 0.75 = three panels across A4, 0.55 = four
        "titles": False,       # default plot titles
        "captions": False,     # small grey description under each plot
    },
}
```

Common choices:

- `functional_standard`: `"CORUM"`, `"GOBP"`, `"PATHWAY"`, or a custom `.csv` path
- `analysis_genes`: `"shared"` or `"dataset_specific"`
- `sort`: `"high"` or `"low"` per input dataset
- `preprocessing.fill_na`: fill missing values with gene means
- `corr_function`: `"numpy"`, `"numpy_without_mask"`, `"numba"`, or `"pandas"`
- `per_module.n_jobs`: worker count for per-module analysis

### Analysis Flow

```python
flex.initialize(config)
data, common_genes = flex.load_datasets(inputs)
terms, _ = flex.load_functional_standard()

for name, dataset in data.items():
    corr = flex.perform_corr(dataset, config["corr_function"])
    flex.pra(name, corr, is_corr=True)
    flex.pra_per_module(name, corr, is_corr=True)
    flex.module_contributions(name)
    flex.mpr_prepare(name)

flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_modules()
flex.plot_per_module_scatter()
flex.plot_per_module_scatter_by_size(n_top=10)
flex.plot_module_contributions()
flex.plot_mpr_summary()
flex.save_results_to_csv()
```

`mpr_prepare()` computes only the standard, unfiltered module-coverage mPR
curve. `plot_mpr_summary()` plots that curve and the corresponding dataset-level
mPR AUC values; it does not produce a duplicate global precision-recall plot.

Figures are made at their final print size: 8 pt Arial text, thin lines and
editable text in PDF/SVG. `plotting.figure_scale` sets the plot-area size
(default `0.75`, three panels across A4; `0.55` fits four), and
`plotting.titles` / `plotting.captions` switch the default titles and short grey
descriptions on. Plots can be combined into one lettered multi-panel figure:

```python
flex.plot_panels([
    flex.plot_precision_recall_curve,
    (flex.plot_per_module_scatter, {"pair": ("Skin", "Soft Tissue")}),
    flex.plot_auc_scores,
    flex.plot_mpr_module_coverage_curve,
], ncols=2)
```

The predefined complex-filter comparisons are optional and computed lazily:

```python
for name in data:
    flex.mpr_filter(name)
    flex.globalpr_filter(name)

# Compare all complexes, removal of mtRibo / ETC I, and removal of
# small high-AUPRC complexes.
mpr_ax, mpr_filter_auc = flex.plot_mpr_filter()
globalpr_ax = flex.plot_globalpr_filter()
```

The calculation and plotting calls are intentionally separate. Calling a
filter plotting function before its matching calculation raises an error that
identifies the required preparation function.

### Early precision: half-max AUC

Global AUPRC averages precision over all of recall, so the long tail of
low-precision pairs decides most of its value. A dataset that is clearly better
among its top predictions can still get a lower AUPRC than one with a long,
weak tail. The half-max AUC scores only the high-precision head of the PR
curves, in one TP window that is the same for all datasets. Standard AUPRC is
not changed.

**How it is computed** (for the datasets run through `pra()`, compared over the
same genes):

1. Each true positive is one point: the k-th true pair found at rank r has
   precision p(k) = k / r.
2. Background `bg = P / N` (true pairs / all pairs), the precision of a random
   ranking. The same for all datasets.
3. Max = the highest precision of any dataset.
4. Threshold = `bg + fraction × (max − bg)`; `fraction` is 1/2 by default (half
   max). One threshold for all datasets.
5. For each dataset, the TP where its precision first falls below the
   threshold. K = the latest of these TPs.
6. Window: TP 1 to K, the same for every dataset.
7. Score = (average precision over TP 1 to K − bg) / (1 − bg). The average is
   taken on the linear TP axis (every TP has the same weight, no weighting). A
   random ranking scores 0, a perfect ranking 1. A dataset that drops below the
   threshold before K keeps its low precision in the average.

**Usage** (after `pra()` has been run for every dataset):

```python
# Scores for all datasets, as {dataset: result dict}
scores = flex.half_max_auc_scores()                        # half max (default)
scores_3q = flex.half_max_auc_scores(fraction="three_quarter")
scores_q = flex.half_max_auc_scores(fraction="quarter")

for name, res in scores.items():
    print(name, round(res["score"], 3), "window TP 1 ->", round(res["k_cut"]))

# One bar per dataset
flex.plot_half_max_auc_scores()                            # saved as half_max_auc_values
flex.plot_half_max_auc_scores(fraction="quarter")          # quarter_max_auc_values

# PR curves with the threshold, each dataset's crossing and the window end K
flex.plot_precision_recall_curve(half_max=True)

# Optional step-by-step figure for sharing: curves on the linear TP axis with
# the scored area, the formula and a results table
flex.plot_half_max_explainer()                             # half_max_explainer
flex.plot_half_max_explainer(dataset_names=["Skin", "Soft Tissue"], fraction="three_quarter")

# The scores are also written to csv/half_max_auc.csv (default settings)
flex.save_results_to_csv()
```

**Parameters** (accepted by `half_max_auc_scores`, `plot_half_max_auc_scores`
and `plot_half_max_explainer`):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `fraction` | `0.5` | Threshold position between background and max: `"half"` / `0.5`, `"three_quarter"` / `0.75`, `"quarter"` / `0.25`, or any number in (0, 1). Higher = stricter threshold, shorter window. |
| `max_rank` | `1` | Use the n-th highest precision as the max. `max_rank=3` takes the 3rd highest, so one or two lucky early hits cannot set the threshold (a curve at 1, 1, 1, ... still gives 1). |
| `max_window` | `1` | Require the max to be held for this many consecutive TPs. `max_window=3`: the highest precision kept for 3 TPs in a row. |
| `k0` | `1` | TP where the search for the max and for the crossings starts. `k0=10` ignores the first 9 TPs for these two steps (the area still starts at TP 1). |

`plot_precision_recall_curve(half_max=True)` takes the same settings as
`half_max_fraction`, `half_max_rank`, `half_max_window` and `half_max_k0`.
`plot_half_max_explainer` also takes `dataset_names` (which datasets to show,
in this order; the threshold and K are still computed from all datasets),
`title` and `filename`.

**Result fields** (one dict per dataset):

| Field | Meaning |
| --- | --- |
| `score` | (mean precision − background) / (1 − background); random 0, perfect 1 |
| `mean_precision` | average precision over TP 1 to K |
| `area` | area under the precision curve from TP 1 to K (linear TP axis) |
| `p_bg` | background P / N |
| `p_max`, `k_max` | this dataset's max precision and the TP where it is reached |
| `p_threshold`, `threshold_from` | the shared threshold and the dataset whose max set it |
| `k_cross` | the TP where this dataset first falls below the threshold |
| `k_cut`, `cut_from` | the shared window end K and the dataset it comes from |
| `fraction`, `max_rank`, `max_window`, `k0` | the settings used |
| `n_tp`, `n_pairs` | true pairs (P) and all pairs (N) |
| `reason` | why the score is missing, if it is NaN (e.g. fewer TPs than `k0`) |

For one ranked label vector outside the pipeline,
`flex.half_max_auc(prediction, threshold=None, k_end=None, ...)` returns the
same fields; without `threshold` and `k_end` it uses the dataset's own
threshold and crossing. The full definition is in
[docs/half-max-auc-spec.md](docs/half-max-auc-spec.md).

See the [User Guide](https://tyasird.github.io/pFLEX/user-guide/) for a detailed explanation of every input field, configuration key, function, return value, and output.

---

## Quickstart

```python
import pflex as flex

inputs = {
    "Skin": {
        "path": flex.example_input_path("skin_cell_lines_corum_genes.parquet"),
        "sort": "high",
        "color": "#4E79A7",
    },
    "Soft Tissue": {
        "path": flex.example_input_path("soft_tissue_cell_lines_corum_genes.parquet"),
        "sort": "high",
        "color": "#F28E2B",
    },
}

config = {
    "functional_standard": "CORUM",
    "output_folder": "output",
    "analysis_genes": "shared",
    "jaccard": True,
    "preprocessing": {
        "fill_na": True,
    },
    "corr_function": "numpy_without_mask",
}

flex.initialize(config)
data, _ = flex.load_datasets(inputs)

for name, dataset in data.items():
    corr = flex.perform_corr(dataset, config["corr_function"])
    flex.pra(name, corr, is_corr=True)

flex.plot_precision_recall_curve()
flex.plot_auc_scores()
```

For a runnable full workflow, see [src/pflex/examples/basic_usage.py](src/pflex/examples/basic_usage.py).

---

## License

MIT
