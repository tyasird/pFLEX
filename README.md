# pFLEX

![PyPI](https://img.shields.io/badge/pypi-v1.1-orange)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
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
        "output_type": "png",
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
flex.plot_per_module_scatter(n_top=10)
flex.plot_per_module_scatter_by_size(n_top=10)
flex.plot_module_contributions()
flex.plot_mpr_summary()
flex.save_results_to_csv()
```

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
