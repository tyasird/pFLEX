"""
Basic usage example of the pflex package.
Demonstrates initialization, data loading, analysis, and plotting.
"""
#%%
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



default_config = {
    "min_genes_in_module": 2,
    "min_genes_per_module_analysis": 2,
    "output_folder": "output_test",
    "functional_standard": "CORUM",
    "color_map": "RdYlBu",
    "jaccard": True,
    "analysis_genes": "shared",  # or "dataset_specific" (genes present per dataset)
    "plotting": {
        "save_plot": True,
        "output_type": "png",
    },
    "preprocessing": {
        "fill_na": True,
    },
    "corr_function": "numpy_without_mask",
    "per_module": {
        "n_jobs": 8,
    },
    "logging": {  
        "visible_levels": ["DONE", "INFO", "WARNING"]
        # "PROGRESS", "STARTED", ,"INFO","WARNING"
    }
}

# Initialize logger, config, and output folder
flex.initialize(default_config)

# Load datasets and functional standard terms
data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_functional_standard()

# Run analysis
for name, dataset in data.items():
    # Calculate correlation once and reuse it for global and per-module PRA.
    corr = flex.perform_corr(dataset, default_config["corr_function"])
    pra = flex.pra(name, corr, is_corr=True)
    fpc = flex.pra_per_module(name, corr, is_corr=True)
    cc = flex.module_contributions(name)
    flex.mpr_prepare(name)


#%%
# Generate plots
flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_modules()
flex.plot_per_module_scatter(n_top=10)
flex.plot_per_module_scatter_by_size(n_top=10)
flex.plot_module_contributions()
flex.plot_mpr_summary()

#%%
# Save results to CSV
flex.save_results_to_csv()

# %%
