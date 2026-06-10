# Getting Started

## Run an Example

!!! info "Example data"
    These files are included only to demonstrate how the script and pFLEX functions work. They are not required dependencies for using the package with your own data. Instead of shipping artificial dummy matrices, pFLEX provides small real DepMap 25Q2 tissue subsets. The matrices are filtered to CORUM genes so the examples stay biologically meaningful while keeping the package size reasonable.

pFLEX includes two real DepMap 25Q2 example inputs filtered to genes present in CORUM:

- `skin_cell_lines_corum_genes.parquet`: 3,465 genes x 75 cell lines
- `soft_tissue_cell_lines_corum_genes.parquet`: 3,465 genes x 46 cell lines

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
    "min_genes_in_complex": 2,
    "min_genes_per_complex_analysis": 2,
    "output_folder": "output",
    "analysis_genes": "shared",
    "jaccard": True,
    "preprocessing": {
        "fill_na": True,
    },
    "corr_function": "numpy_without_mask",
    "per_complex": {
        "n_jobs": 8,
    },
    "plotting": {
        "save_plot": True,
        "output_type": "png",
    },
}

flex.initialize(config)
data, common_genes = flex.load_datasets(inputs)
terms, _ = flex.load_functional_standard()

for name, dataset in data.items():
    corr = flex.perform_corr(dataset, config["corr_function"])
    flex.pra(name, corr, is_corr=True)
    flex.pra_percomplex(name, corr, is_corr=True)
    flex.complex_contributions(name)
    flex.mpr_prepare(name)

flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_complexes()
flex.plot_percomplex_scatter(n_top=10)
flex.plot_percomplex_scatter_bysize(n_top=10)
flex.plot_complex_contributions()
flex.plot_mpr_summary()
flex.save_results_to_csv()
```

The same workflow is available in `src/pflex/examples/basic_usage.py`.
