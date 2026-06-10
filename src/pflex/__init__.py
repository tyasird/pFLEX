from .logging_config import log
from .utils import dsave, dload
from .preprocessing import example_input_path, get_example_data_path, load_datasets,  get_common_genes, filter_matrix_by_genes, load_functional_standard, filter_duplicate_terms
from .analysis import initialize, prepare_terms_for_dataset, pra, pra_per_module, fast_corr, perform_corr, is_symmetric, binary, has_mirror_of_first_pair, convert_full_to_half_matrix, drop_mirror_pairs, quick_sort, module_contributions, save_results_to_csv, update_matploblib_config, mpr_prepare
from .plotting import (
    adjust_text_positions, plot_precision_recall_curve, plot_aggregated_pra, plot_iqr_pra, plot_all_runs_pra, plot_per_module_scatter,
    plot_per_module_scatter_by_size, plot_module_contributions, plot_significant_modules, plot_auc_scores,
    plot_mpr_tp, plot_mpr_modules, plot_mpr_tp_multi, plot_mpr_modules_multi, plot_mpr_modules_auc_scores,
    plot_mpr_true_positive_curve, plot_mpr_module_coverage_curve, plot_mpr_module_auc_scores, plot_mpr_summary
)

def main():
    import argparse
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("pflex")
    except PackageNotFoundError:
        package_version = "unknown"

    parser = argparse.ArgumentParser(
        prog="pflex",
        description="pflex benchmarking toolkit. Use the Python API to run analyses.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pflex {package_version}",
    )
    parser.parse_args()
    parser.print_help()

__all__ = [ "log", "example_input_path", "get_example_data_path", "fast_corr",
    "initialize", "dsave", "dload", "load_datasets", "get_common_genes",
    "filter_matrix_by_genes", "load_functional_standard", "filter_duplicate_terms", "pra", "pra_per_module",
    "prepare_terms_for_dataset",
    "perform_corr", "is_symmetric", "binary", "has_mirror_of_first_pair", "convert_full_to_half_matrix",
    "drop_mirror_pairs", "quick_sort", "module_contributions", "adjust_text_positions", "plot_precision_recall_curve",
    "plot_aggregated_pra", "plot_iqr_pra", "plot_all_runs_pra", "plot_per_module_scatter", "plot_per_module_scatter_by_size", "plot_module_contributions",
    "plot_significant_modules", "plot_auc_scores", "plot_mpr_modules_auc_scores", "plot_mpr_module_auc_scores", "save_results_to_csv", "update_matploblib_config",
    "mpr_prepare", "plot_mpr_tp", "plot_mpr_modules",
    "plot_mpr_tp_multi", "plot_mpr_modules_multi", "plot_mpr_true_positive_curve",
    "plot_mpr_module_coverage_curve", "plot_mpr_summary", "main"
]
