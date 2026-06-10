# User Guide

This guide explains the example workflow in detail: how the input matrix should look, what each configuration key means, and what each pFLEX function returns or stores.

## Input Matrix Format

!!! danger "Input orientation"
    Genes must be rows and samples, screens, or cell lines must be columns. If this matrix is transposed, pFLEX will build and benchmark sample-sample relationships instead of gene-gene relationships.

pFLEX expects one matrix per dataset. Values are the measurement used to build a gene-gene network, such as dependency scores across cell lines.

| Gene | ACH-000014 | ACH-000219 | ACH-000274 | ACH-000304 |
| --- | ---: | ---: | ---: | ---: |
| A2M | -0.125 | -0.215 | 0.065 | -0.014 |
| AATF | 0.042 | -0.088 | -0.016 | 0.031 |
| BCL6 | -0.019 | 0.112 | -0.074 | -0.053 |
| HDAC4 | 0.078 | -0.031 | -0.091 | 0.045 |

pFLEX accepts `.csv`, `.xlsx`, `.parquet`, and `.p` files; `.p` files are read as Parquet files. Parquet files are recommended for larger matrices because they load faster and preserve the DataFrame structure cleanly.

## Example Inputs

`example_input_path()` returns the installed package path for an example input file. It does not load the data by itself. `load_datasets()` reads the file later.

```python
skin_path = flex.example_input_path("skin_cell_lines_corum_genes.parquet")
```

The example `inputs` dictionary names each dataset and tells pFLEX where to find the matrix:

```python
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

Input fields:

- `path`: a CSV, Excel, Parquet, or `.p` path, or a `pandas.DataFrame`.
- `sort`: use `"high"` when higher pair scores should rank first; use `"low"` when lower scores are better.
- `color`: optional plot color for this dataset.

## Configuration

Initialize pFLEX once at the start of a run:

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
        "output_type": "png",
    },
    "logging": {
        "visible_levels": ["DONE", "INFO", "WARNING"],
    },
}

flex.initialize(config)
```

Configuration keys:

- `functional_standard`: built-in values are `"CORUM"`, `"GOBP"`, and `"PATHWAY"`. You can also provide a custom `.csv` path with a `Genes` column containing semicolon-separated gene symbols.
- `min_genes_in_module`: minimum number of genes a functional-standard term must contain before dataset-specific filtering.
- `min_genes_per_module_analysis`: minimum term size for per-module analysis.
- `output_folder`: folder where plots and exported CSV files are written.
- `analysis_genes`: `"shared"` uses only genes common to all loaded datasets; `"dataset_specific"` evaluates each dataset using its own available genes.
- `jaccard`: when `True`, removes terms with identical used-gene sets after filtering.
- `preprocessing.fill_na`: when `True`, fills missing values with each gene row's mean.
- `corr_function`: correlation backend. Use `"numpy_without_mask"` for the fastest run when the matrix has no missing values, `"numpy"` for masked NumPy correlation when missing values may be present, `"numba"` for a compiled missing-value-aware implementation, or `"pandas"` for pandas pairwise-complete correlation that most closely matches R-style missing-value handling. In practice, `"numpy_without_mask"` is fastest but does not handle missing values, `"numpy"` and `"numba"` handle missing values faster than pandas, and `"pandas"` is usually slowest.
- `per_module.n_jobs`: worker count for per-module analysis.
- `plotting.save_plot`: when `True`, saves figures to `output_folder`.
- `plotting.output_type`: figure extension, such as `"png"` or `"pdf"`.
- `logging.visible_levels`: log levels to print during the run. Available values are `"STARTED"`, `"PROGRESS"`, `"DONE"`, `"INFO"`, `"WARNING"`, and `"ERROR"`. A quiet run can use `["DONE", "WARNING"]`; a more verbose run can use `["STARTED", "PROGRESS", "DONE", "INFO", "WARNING"]`.

## Loading

```python
data, common_genes = flex.load_datasets(inputs)
terms, _ = flex.load_functional_standard()
```

`load_datasets(inputs)`:

- reads each input matrix
- applies configured preprocessing
- stores dataset colors and sort direction for later plotting
- returns `data`, a dictionary of loaded DataFrames
- returns `common_genes`, the genes shared by all datasets

`load_functional_standard()`:

- loads the configured functional standard
- creates an `all_genes` list for each term
- applies `min_genes_in_module`
- stores the term table for later analysis
- returns the loaded term DataFrame and a placeholder second value

## Analysis Functions

```python
for name, dataset in data.items():
    corr = flex.perform_corr(dataset, config["corr_function"])
    flex.pra(name, corr, is_corr=True)
    flex.pra_per_module(name, corr, is_corr=True)
    flex.module_contributions(name)
    flex.mpr_prepare(name)
```

`perform_corr(dataset, corr_function)`:

- takes a gene-by-sample matrix
- returns a gene-by-gene correlation matrix

`pra(name, matrix, is_corr=True)`:

- runs global precision-recall analysis over ranked gene pairs
- returns the pairwise PRA table
- stores PRA results, AUPRC, and corrected AUPRC in the run cache

`pra_per_module(name, matrix, is_corr=True)`:

- evaluates precision-recall performance for each functional-standard term
- returns a term-level result table
- stores per-module AUPRC values for plotting and export

`module_contributions(name)`:

- estimates which terms contribute most to true-positive gene pairs
- returns a contribution table
- stores contribution data for plots and CSV export

`mpr_prepare(name)`:

- prepares module-level precision-recall data
- stores true-positive curves, coverage curves, filter metadata, and mPR AUC summaries
- must be run before the mPR plotting functions
- accepts optional thresholds such as `size_th`, `auprc_th`, `tp_th`, and `percent_th` for the module-level filtering logic

## Plotting and Export

```python
flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_modules()
flex.plot_per_module_scatter(n_top=10)
flex.plot_per_module_scatter_by_size(n_top=10)
flex.plot_module_contributions()
flex.plot_mpr_summary()
flex.save_results_to_csv()
```

Plotting functions read cached analysis results from the current run. If `plotting.save_plot` is `True`, figures are written to `output_folder`.

Global plots:

- `plot_precision_recall_curve(line_width=2.0, hide_minor_ticks=True)` shows precision against the cumulative number of true-positive gene pairs. This is the main global performance curve; higher precision at the same true-positive count means better ranking.
- `plot_auc_scores()` shows one bar per dataset using the global AUPRC value saved by `pra()`. Higher AUPRC means the ranked gene-pair network recovers functional-standard pairs earlier in the ranking.

Per-module plots:

- `plot_significant_modules()` counts how many functional-standard terms pass AUPRC thresholds `0.1`, `0.2`, `0.3`, `0.4`, and `0.5`. It returns the threshold-by-dataset count table, which helps compare how many terms are strongly recovered in each dataset.
- `plot_per_module_scatter(n_top=10, ...)` compares per-module AUPRC values between pairs of datasets. Each point is one functional-standard term. Points near the diagonal perform similarly in both datasets; points far from the diagonal are terms that are stronger in one dataset.
- `plot_per_module_scatter_by_size(n_labels=10, n_top=10, ...)` plots each term's per-module AUPRC against the number of genes used for that term. It helps distinguish compact high-performing terms from broader modules.
- `plot_module_contributions(min_pairs=10, min_precision_cutoff=0.5, num_module_to_show=10, ...)` shows which terms contribute most to true-positive gene pairs across precision cutoffs. It requires `module_contributions(name)` first and helps identify which biological terms drive the global precision-recall curve.

mPR preparation and plots:

- `mpr_prepare(name, size_th=30, auprc_th=0.4, tp_th=1, percent_th=0.1, use_corrected=True)` prepares module-level PR data for one dataset. It filters terms for module-level analysis, stores true-positive curves, module-coverage curves, filtered variants, and mPR AUC values.
- `plot_mpr_tp_multi(dataset_names=None, colors=None, linewidth=1.8, show_filters=("all", "no_mtRibo_ETCI", "no_small_highAUPRC"))` shows true positives versus precision for one or more datasets after `mpr_prepare()`. This plot shows how many true-positive gene pairs are recovered as the precision cutoff changes.
- `plot_mpr_modules_multi(dataset_names=None, colors=None, linewidth=1.8, show_filters=("all", "no_mtRibo_ETCI", "no_small_highAUPRC"), show_markers="auto")` shows how many functional-standard terms are covered at each precision cutoff. This plot focuses on term coverage, not pair counts.
- `plot_mpr_summary(dataset_names=None, colors=None, variants="unfiltered", save=True, linewidth=1.8, show_markers="auto")` creates the standard mPR true-positive, module-coverage, and mPR AUC summary plots in one call.
- Newer mPR functions also accept `variants`, with values `"unfiltered"`, `"without_mt_ribo_etci"`, `"without_small_high_auprc"`, or `"all"`.

Export:

- `save_results_to_csv()` exports cached tables such as global AUPRC values, per-module results, contribution tables, and mPR summaries to `output_folder/csv`.
- Plot files are saved directly under `output_folder` when `plotting.save_plot` or the function-level `save` argument is enabled.

## Temporary Files

pFLEX creates a `.tmp` folder in the current working directory while an analysis is running. `flex.initialize(...)` removes any existing `.tmp` folder at the start of a new analysis so cached results from an older run do not mix with the new run.
