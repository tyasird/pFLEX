# pflex

![PyPI](https://img.shields.io/badge/pypi-v1.0-orange)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Build system](https://img.shields.io/badge/build-hatchling-blue)
![Lint](https://img.shields.io/badge/lint-ruff-46a2f1)
![CLI](https://img.shields.io/badge/CLI-pflex-brightgreen)

**pflex** is a benchmarking toolkit for evaluating CRISPR screen results against biological gold standards. It provides precision-recall analysis using reference gene sets from CORUM protein complexes, Gene Ontology Biological Processes (GO-BP), KEGG pathways, and other curated resources. The toolkit computes gene-level and complex-level performance metrics, helping researchers systematically assess the biological relevance and resolution of their CRISPR screening data.

---

## Features

- Precision-recall curve generation for ranked gene lists
- Evaluation using CORUM complexes, GO terms, pathways
- Complex-level resolution analysis and visualization
- Easy integration into CRISPR screen workflows

---

## Installation

Suggested to use Python version `3.10` with `virtual env`.

Create `venv`:

```bash
conda create -n p310 python=3.10
conda activate p310
pip install uv
```

Install pflex via pip:

```bash
uv pip install pflex
```

or:

```bash
pip install pflex
```

or install pflex via git to develop the package locally:

```bash
git clone https://github.com/tyasird/pflex.git
cd pflex
uv pip install -e .
```

---

## Quickstart

```python
import pflex as flex

inputs = {
    "Melanoma (63 Screens)": {
        "path": flex.get_example_data_path("melanoma_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#FF0000",
    },
    "Liver (24 Screens)": {
        "path": flex.get_example_data_path("liver_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#FFDD00",
    },
    "Neuroblastoma (37 Screens)": {
        "path": flex.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#FFDDDD",
    },
}

default_config = {
    "min_genes_in_complex": 0,
    "min_genes_per_complex_analysis": 3,
    "output_folder": "CORUM",
    "gold_standard": "CORUM",
    "color_map": "BuGn",
    "jaccard": False,
    "use_common_genes": False,
    "plotting": {
        "save_plot": True,
        "output_type": "png",
    },
    "preprocessing": {
        "fill_na": True,
        "normalize": False,
    },
    "corr_function": "numpy",
    "logging": {
        "visible_levels": ["DONE"],
    },
}

flex.initialize(default_config)

data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_gold_standard()

for name, dataset in data.items():
    pra = flex.pra(name, dataset, is_corr=False)
    fpc = flex.pra_percomplex(name, dataset, is_corr=False)
    cc = flex.complex_contributions(name)
    flex.mpr_prepare(name)

flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_complexes()
flex.plot_percomplex_scatter(n_top=20)
flex.plot_percomplex_scatter_bysize()
flex.plot_complex_contributions()
flex.plot_mpr_tp_multi()
flex.plot_mpr_complexes_multi()
flex.save_results_to_csv()
```

---

## Examples

- [src/pflex/examples/basic_usage.py](src/pflex/examples/basic_usage.py)

---

## License

MIT
