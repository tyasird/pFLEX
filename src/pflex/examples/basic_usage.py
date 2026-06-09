"""
Basic usage example of the pflex package.
Demonstrates initialization, data loading, analysis, and plotting.
"""
#%%
import pflex as flex

inputs = {
    "Melanoma (63 Screens)": {
        "path": flex.get_example_data_path("melanoma_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#4E79A7",
    },
    "Liver (24 Screens)": {
        "path": flex.get_example_data_path("liver_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#F28E2B",
    },
    "Neuroblastoma (37 Screens)": {
        "path": flex.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"),
        "sort": "high",
        "color": "#59A14F",
    },
}



default_config = {
    "min_genes_in_complex": 2,
    "min_genes_per_complex_analysis": 2,
    "output_folder": "output_test",
    "gold_standard": "GOBP",
    "color_map": "RdYlBu",
    "jaccard": True,
    "analysis_genes": "shared",  # or "dataset_specific" (genes present per dataset)
    "plotting": {
        "save_plot": True,
        "output_type": "png",
    },
    "preprocessing": {
        "fill_na": True,
        "normalize": False,
    },
    "corr_function": "numpy_without_mask",
    "per_complex": {
        "n_jobs": 8,
        "chunk_size": 400,
        "max_nbytes": "100M",
    },
    "logging": {  
        "visible_levels": ["DONE", "INFO", "WARNING"]
        # "PROGRESS", "STARTED", ,"INFO","WARNING"
    }
}

# Initialize logger, config, and output folder
flex.initialize(default_config)

# Load datasets and gold standard terms
data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_gold_standard()

# Run analysis
for name, dataset in data.items():
    # Calculate correlation once and reuse it for global and per-complex PRA.
    corr = flex.perform_corr(dataset, default_config["corr_function"])
    pra = flex.pra(name, corr, is_corr=True)
    fpc = flex.pra_percomplex(name, corr, is_corr=True)
    cc = flex.complex_contributions(name)

    # Optional mPR analysis. This can be slow on large datasets.
    # flex.mpr_prepare(name)
    


#%%
# Generate plots
flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_complexes()
#%%
flex.plot_percomplex_scatter(n_top=10)
flex.plot_percomplex_scatter_bysize(n_top=10)
#flex.plot_complex_contributions()

# Optional mPR summary plot. Requires flex.mpr_prepare(name) above.
# flex.plot_mpr_summary(variants="unfiltered")

#%%
# Save results to CSV
# flex.save_results_to_csv()

#how many cpu I have?
import multiprocessing
print(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
