# Standard library imports
import gc
import os
import re
import shutil
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

# Third-party imports
from art import tprint
from bitarray import bitarray
from joblib import Parallel, delayed, dump, load
from numba import njit, prange
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

# Local/application-specific imports
from .logging_config import log
from .preprocessing import (
    filter_matrix_by_genes,
    filter_duplicate_terms,
    load_functional_standard,
)
from .utils import dsave, dload, _sanitize, normalize_analysis_genes

import matplotlib as mpl

def deep_update(source, overrides):
    """Recursively update the source dict with the overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source

def initialize(config={}):

    user_overrides = config if isinstance(config, dict) else {}

    default_config = {
        "min_genes_in_complex": 3,
        "min_genes_per_complex_analysis": 2,
        "output_folder": "output",
        "functional_standard": "CORUM",
        "color_map": "RdYlBu",
        "jaccard": True,
        # Which genes are used for analysis (drives used_genes intersection)
        # - 'shared'           : use genes common to all datasets (common_genes)
        # - 'dataset_specific' : use genes present in each dataset individually
        "analysis_genes": "shared",
        "plotting": {
            "save_plot": True,
            "show_plot": True,
            "output_type": "pdf",
        },
        "preprocessing": {
            "normalize": False,
            "fill_na": False,
            "drop_na": False,
        },
        "corr_function": "numpy",
        "per_complex": {
            "n_jobs": 4,
            "chunk_size": 200,
            "max_nbytes": "100M",
        },
        "logging": {  # Added: Default logging config
            "visible_levels": ["DONE"]  # if needed #, "PROGRESS", "STARTED", "INFO"
        }
    }
    
    # Early merge to get user-overridden config (including logging.visible_levels)
    if config is not None:
        config = deep_update(default_config, config)
    else:
        config = default_config

    # Backward compatibility: if user provided legacy key but not the new one,
    # map it to analysis_genes. (We must look at the original overrides, because
    # defaults always include analysis_genes.)
    analysis_genes_provided = (
        isinstance(user_overrides, dict)
        and "analysis_genes" in user_overrides
        and user_overrides.get("analysis_genes") is not None
        and str(user_overrides.get("analysis_genes")).strip() != ""
    )
    if (
        isinstance(user_overrides, dict)
        and "use_common_genes" in user_overrides
        and not analysis_genes_provided
    ):
        config["analysis_genes"] = (
            "shared" if bool(user_overrides.get("use_common_genes")) else "dataset_specific"
        )

    config["analysis_genes"] = normalize_analysis_genes(config.get("analysis_genes"))
    
    # Extract visible_levels from the merged config and set logging visibility immediately (before any logs)
    visible_levels = config.get("logging", {}).get("visible_levels", ["DONE"])
    log.set_visible_levels(visible_levels)

    log.info("******************************************************************")
    log.info("pFLEX initialized")
    log.info("******************************************************************")
    log.started("Initialization")

    # Check and remove .tmp folder if it exists (clean slate to avoid overriding old results)
    tmp_folder = ".tmp"
    if os.path.exists(tmp_folder):
        log.info(f"Removing existing '{tmp_folder}' folder for a clean start.")
        shutil.rmtree(tmp_folder)
        log.done(f"'{tmp_folder}' folder removed successfully.")

    log.progress("Saving configuration settings.")   
        
    dsave(config, "config")
    update_matploblib_config(config)
    output_folder = config.get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)
    log.progress(f"Output folder '{output_folder}' ensured to exist.")
    log.done("Initialization completed. ")
    log.info("Input data: genes should be rows and samples should be columns.")
    #tprint("pFLEX", font="standard")

def update_matploblib_config(config=None, font_family="Arial", layout="single"):
    """
    Configure matplotlib settings optimized for Nature journal figures:
      - 7 pt fonts (labels, ticks, legend), 9 pt titles
      - Thin spines (0.5 pt), ticks out (left/bottom only), no minor ticks
      - No grid, clean minimalist look
      - Colorblind-friendly Tableau 10 color cycle
      - Illustrator-safe PDF export (Type 42)
      - Figure sizes: "single" (~89 mm), "double" (~183 mm), or custom (width, height) in inches
    
    Args:
        config (dict, optional): Configuration dict (e.g., {'color_map': 'RdYlBu'}).
        font_family (str): Preferred font (e.g., 'Arial', falls back to 'Helvetica').
        layout (str or tuple): 'single' (~89 mm), 'double' (~183 mm), or (width, height) in inches.
    """
    if config is None:
        config = {}
    # Fallback if chosen font missing
    requested_font_family = font_family
    try:
        from matplotlib.font_manager import findfont, FontProperties
        findfont(FontProperties(family=font_family))
    except Exception:
        font_family = "Helvetica"  # Nature prefers Helvetica if Arial unavailable
        log.warning(
            f"Font '{requested_font_family}' not found; falling back to '{font_family}'."
        )
    
    # Figure size presets (Nature: single ≈ 89 mm, double ≈ 183 mm at 25.4 mm/inch)
    if isinstance(layout, tuple):
        fig_w, fig_h = layout
    else:
        if layout == "double":
            fig_w = 7.2  # ~183 mm
            fig_h = 5.4  # Adjusted aspect
        else:  # "single"
            fig_w = 4.0  # Increased from 3.5" for more space (~102 mm)
            fig_h = 3.0  # Increased from 2.6" for better aspect (~76 mm)
    # Colorblind-friendly cycle (Tableau 10 adapted)
    cb_cycle = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ]
    mpl.rcParams.update({
        # --- Text & Fonts ---
        "text.usetex": False,  # Avoid LaTeX
        "font.family": [font_family],  # Explicit font
        "mathtext.fontset": "dejavusans",  # Disable mathtext
        "mathtext.default": "regular",  # Plain text
        "axes.unicode_minus": True,  # Proper minus signs
        # --- Sizes (7 pt baseline, adjusted for space) ---
        "font.size": 7,  # Reduced from 8 pt
        "axes.titlesize": 9,  # Reduced from 10 pt
        "axes.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # --- Lines & Markers ---
        "lines.linewidth": 1.5,  # Kept for data visibility
        "lines.markersize": 4.0,
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2,
        # --- Axes, Spines, Ticks ---
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "axes.facecolor": "none",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.prop_cycle": mpl.cycler(color=cb_cycle),
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.top": False,
        "ytick.right": False,
        # --- Grid ---
        "axes.grid": False,
        # --- Legend ---
        "legend.frameon": False,
        "legend.handlelength": 1.6,  # Slightly adjusted
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "legend.loc": "best",  # Dynamic placement to avoid overlap
        # --- Figure & Save ---
        "figure.dpi": 600,
        "figure.figsize": (fig_w, fig_h),
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,  # Increased for spacing
        "savefig.transparent": False,  # White background
        # --- PDF/SVG Export ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",
    })


def _sort_ascending_for_dataset(dataset_name):
    sorting = dload("input", "sorting")
    if not isinstance(sorting, dict):
        return False
    sort_order = str(sorting.get(dataset_name, "high")).strip().lower()
    return sort_order == "low"


def prepare_terms_for_dataset(dataset_name, matrix):
    """Prepare dataset-specific functional-standard terms and filtered matrix.

        This computes:
            - terms['used_genes'] as the intersection of terms['all_genes'] with either
                shared genes (config['analysis_genes']=='shared') or the dataset genes
                (config['analysis_genes']=='dataset_specific').
      - genes_present_in_terms_<dataset_name>

    Side effects:
      - stores dataset-specific terms and genes list under:
        dsave(..., 'common', f'terms_{dataset_name}')
        dsave(..., 'common', f'genes_present_in_terms_{dataset_name}')

    Returns:
      (terms, genes_present, matrix_filtered)
    """
    config = dload("config")
    if config is None:
        raise RuntimeError(
            "prepare_terms_for_dataset(): config not found. Run initialize() first."
        )

    terms_path_exists = any(
        Path(".tmp", "common", f"terms{ext}").exists()
        for ext in (".parquet", ".npy", ".pkl")
    )
    terms_data = dload("common", "terms") if terms_path_exists else None
    if terms_data is None or not isinstance(terms_data, pd.DataFrame):
        log.info("Functional standard terms not loaded; loading them now.")
        terms_data, _ = load_functional_standard()
        if terms_data is None or not isinstance(terms_data, pd.DataFrame):
            raise ValueError(
                "prepare_terms_for_dataset(): expected 'terms' to be a DataFrame, but got None or invalid type."
            )
    terms = terms_data.copy()

    analysis_genes = normalize_analysis_genes(config.get("analysis_genes"))

    if analysis_genes == "shared":
        common_genes = dload("common", "common_genes")
        if common_genes is None:
            raise ValueError(
                "prepare_terms_for_dataset(): common genes not found. "
                "Run get_common_genes() or set analysis_genes='dataset_specific'."
            )

        common_genes_list = list(common_genes)
        if len(common_genes_list) == 0:
            raise ValueError(
                "prepare_terms_for_dataset(): common genes is empty. "
                "Run get_common_genes() or set analysis_genes='dataset_specific'."
            )

        gene_universe = set(common_genes_list)
        log.info(f"Using shared genes approach: {len(gene_universe)} genes")
    else:
        gene_universe = set(matrix.index)
        log.info(
            f"Using dataset-specific approach for {dataset_name}: {len(gene_universe)} genes in dataset"
        )

    terms["used_genes"] = terms["all_genes"].apply(
        lambda genes: list(set(genes) & gene_universe)
    )

    min_genes_raw = config.get("min_genes_in_complex", 3)
    min_genes = int(min_genes_raw) if min_genes_raw is not None else 3
    terms["n_used_genes"] = terms["used_genes"].apply(len)
    terms = terms[terms["n_used_genes"] >= min_genes]

    if bool(config.get("jaccard", False)):
        before = len(terms)
        terms = filter_duplicate_terms(terms)
        log.done(
            f"After Jaccard duplicate used_genes filtering for {dataset_name}: "
            f"{len(terms)} terms ({before - len(terms)} removed)"
        )

    genes_present = list(
        set(gene for genes_list in terms["used_genes"] for gene in genes_list)
    )
    log.info(f"Genes present in terms for {dataset_name}: {len(genes_present)}")

    matrix_filtered = filter_matrix_by_genes(matrix, genes_present)

    dsave(terms, "common", f"terms_{dataset_name}")
    dsave(genes_present, "common", f"genes_present_in_terms_{dataset_name}")

    return terms, genes_present, matrix_filtered

def pra(dataset_name, matrix, is_corr=False):
    log.info(f"******************** {dataset_name} ********************")
    log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")
    config = dload("config")
    ascending = _sort_ascending_for_dataset(dataset_name)

    if not is_corr:
        matrix = perform_corr(matrix, config.get("corr_function"))

    terms, _genes_present, matrix = prepare_terms_for_dataset(dataset_name, matrix)
    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")
    df = quick_sort(df, ascending=ascending)

    log.started("Building gene-to-pair indices")
    gold_pair_to_complex = _build_gold_pair_to_complex(terms)  
    log.done("Gene-to-pair indices built.")
    
    log.started("Precomputing complex IDs")
    df = _precompute_complex_ids(df, gold_pair_to_complex)
    log.done("Complex IDs precomputed.")

    df["prediction"] = df["complex_ids"].astype(bool).astype(int)
    df["complex_id"] = df["complex_ids"].apply(
        lambda s: list(map(int, s.split(";"))) if s else []
    )

    if df["prediction"].sum() == 0:
        log.info("No true positives found in dataset.")
        pr_auc = np.nan
        df["tp"] = 0
        df["precision"] = np.nan
        df["recall"] = np.nan
    else:
        tp = df["prediction"].cumsum()
        df["tp"] = tp
        precision = tp / (np.arange(len(df)) + 1)
        recall = tp / tp.iloc[-1]
        df["precision"] = precision
        df["recall"] = recall
        pr_auc = metrics.auc(recall, precision) if len(recall) >= 2 else np.nan
    
    log.info(f"AUPRC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
    dsave(df, "pra", dataset_name)
    dsave(pr_auc, "pr_auc", dataset_name)
    dsave(_corrected_auc(df), "corrected_pr_auc", dataset_name)

    log.done(f"Global PRA completed for {dataset_name}")
    return df

# --------------------------------------------------------------------------
# helper functions for PRA per-complex analysis
# --------------------------------------------------------------------------

def _corrected_auc(df: pd.DataFrame) -> float:
    if df.empty or "precision" not in df.columns or "recall" not in df.columns:
        return np.nan
    valid = df[["precision", "recall"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2:
        return np.nan
    return np.trapz(valid["precision"], valid["recall"]) - valid["precision"].iloc[-1]

def _build_gene_to_pair_indices(pairwise_df):
    indices = pairwise_df.index.values
    genes = pd.concat([pairwise_df['gene1'], pairwise_df['gene2']], ignore_index=True)
    stacked_indices = np.concatenate([indices, indices])
    idx_series = pd.Series(stacked_indices, index=range(len(genes)))
    gene_to_pair_indices = defaultdict(list)
    for gene, group in idx_series.groupby(genes):
        gene_to_pair_indices[gene] = group.values.tolist() 
    return gene_to_pair_indices

def _build_gold_pair_to_complex(terms):
    pair_map = defaultdict(set)
    for comp_id, genes in zip(terms.index, terms['used_genes']):
        genes = list(genes)
        if len(genes) < 2: continue
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                g1, g2 = sorted([genes[i], genes[j]])
                pair_map[(g1, g2)].add(comp_id)
    return pair_map

def _precompute_complex_ids(pairwise_df, gold_pair_to_complex):
    if not gold_pair_to_complex:
        pairwise_df['complex_ids'] = ''
        return pairwise_df
    
    # Precompute pairs as tuples
    g1 = pairwise_df['gene1']
    g2 = pairwise_df['gene2']
    pairs = [tuple(sorted((a, b))) for a, b in zip(g1, g2)]
    pairwise_df['complex_ids'] = [
        ';'.join(map(str, sorted(gold_pair_to_complex[p]))) 
        if p in gold_pair_to_complex else '' 
        for p in pairs
    ]
    return pairwise_df

def _dump_pairwise_memmap(df: pd.DataFrame, tag: str) -> Path:
    tmp_dir = Path(os.path.join(".tmp", "mmap"))  # Use .tmp/mmap/ for organization
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    path = tmp_dir / f".pairwise_{_sanitize(tag)}.pkl"          
    dump(df, path, compress=0)  
    return path 

# Global variables for worker processes (compatible with older joblib)
PAIRWISE_DF = None
GENE2IDX = None

def _init_worker_globals(memmap_path, gene_to_pair_indices):
    """Initialize global variables for worker processes"""
    global PAIRWISE_DF, GENE2IDX
    PAIRWISE_DF = load(memmap_path)        
    GENE2IDX = gene_to_pair_indices

def delete_memmap(memmap_path, log, wait_seconds=0.1):

    gc.collect()
    time.sleep(wait_seconds)

    try:
        os.remove(memmap_path)
        log.info(f"Cleaned up temporary memmap file: {memmap_path}")
    except OSError as e:
        log.warning(f"Original error: {e}")

# --------------------------------------------------------------------------
# Process each chunk of terms
# --------------------------------------------------------------------------
def _process_chunk(chunk_terms, min_genes, memmap_path, gene_to_pair_indices):
    """Process a chunk of terms - compatible with older joblib versions"""
    try:
        # Load data in each worker (compatible with older joblib)
        pairwise_df = load(memmap_path)
        local_auc_scores = {}
        local_corrected_auc_scores = {}

        for idx, row in chunk_terms.iterrows():
            gene_set = set(row.used_genes)
            if len(gene_set) < min_genes:
                continue

            candidate_indices = bitarray(len(pairwise_df))
            for g in gene_set:
                if g in gene_to_pair_indices:
                    candidate_indices[gene_to_pair_indices[g]] = True
            if not candidate_indices.any():
                continue

            selected = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
            sub_df   = pairwise_df.iloc[selected]

            complex_id = str(idx)
            pattern    = r'(?:^|;)' + re.escape(complex_id) + r'(?:;|$)'
            true_label = sub_df["complex_ids"].str.contains(pattern, regex=True).astype(int)
            mask       = (sub_df["complex_ids"] == "") | (true_label == 1)
            preds      = true_label[mask]

            if preds.sum() == 0:
                continue

            tp_cum   = preds.cumsum()
            precision = tp_cum / (np.arange(len(preds)) + 1)
            recall    = tp_cum / tp_cum.iloc[-1]
            if len(recall) >= 2 and recall.iloc[-1] != 0:
                # Compute regular AUC
                local_auc_scores[idx] = metrics.auc(recall, precision)
                # Compute corrected AUC using the same logic as _corrected_auc function
                local_corrected_auc_scores[idx] = np.trapz(precision, recall) - precision.iloc[-1]

        return {'auc': local_auc_scores, 'corrected_auc': local_corrected_auc_scores}
    
    except Exception as e:
        # Return error info for debugging
        return {'error': str(e), 'chunk_size': len(chunk_terms)}

def pra_percomplex(dataset_name, matrix, is_corr=False, chunk_size=None, n_jobs=None):
    log.started(f"*** Per-complex PRA started - {dataset_name} ***")
    config = dload("config")
    ascending = _sort_ascending_for_dataset(dataset_name)
    per_complex_config = config.get("per_complex", {})
    if not isinstance(per_complex_config, dict):
        per_complex_config = {}
    chunk_size_value = (
        chunk_size if chunk_size is not None else per_complex_config.get("chunk_size", 200)
    )
    n_jobs_value = n_jobs if n_jobs is not None else per_complex_config.get("n_jobs", 4)
    max_nbytes = per_complex_config.get("max_nbytes", "100M")

    try:
        effective_chunk_size = int(chunk_size_value)
        effective_n_jobs = int(n_jobs_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "per-complex chunk_size and n_jobs must be integer-compatible values."
        ) from exc

    if effective_chunk_size <= 0:
        raise ValueError("per-complex chunk_size must be greater than 0.")
    if effective_n_jobs <= 0:
        raise ValueError("per-complex n_jobs must be greater than 0.")

    if not is_corr:
        matrix = perform_corr(matrix, config.get("corr_function"))

    # Prefer terms prepared by pra(); if absent, prepare them here so direct
    # pra_percomplex() calls use the same dataset-specific gene universe.
    terms = dload("common", f"terms_{dataset_name}")
    genes_present = dload("common", f"genes_present_in_terms_{dataset_name}")

    if not isinstance(terms, pd.DataFrame) or genes_present is None:
        log.warning(
            f"No dataset-specific terms found for {dataset_name}; preparing them now."
        )
        terms, genes_present, matrix = prepare_terms_for_dataset(dataset_name, matrix)
    else:
        matrix = filter_matrix_by_genes(matrix, genes_present)

    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")
    df = quick_sort(df, ascending=ascending)
    pairwise_df = df.copy()
    pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
    pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")
    
    # Use helper functions for precomputations
    log.started("Building gene-to-pair indices")
    gene_to_pair_indices = _build_gene_to_pair_indices(pairwise_df)
    log.done("Building gene-to-pair indices") 
    
    log.started("Building gold pair to complex mapping")
    gold_pair_to_complex = _build_gold_pair_to_complex(terms)  # Now serial
    log.done("Building gold pair to complex mapping") 
    
    log.started("Precomputing complex IDs")
    pairwise_df = _precompute_complex_ids(pairwise_df, gold_pair_to_complex)
    log.done("Precomputing complex IDs")  # 

    chunks = [
        terms.iloc[i:i + effective_chunk_size]
        for i in range(0, len(terms), effective_chunk_size)
    ]
    min_genes = config["min_genes_per_complex_analysis"]

    if not chunks:
        terms["auc_score"] = pd.Series(dtype=float)
        terms["corrected_auc_score"] = pd.Series(dtype=float)
        dsave(terms, "pra_percomplex", dataset_name)
        log.done("Per-complex PRA completed with no eligible terms.")
        return terms

    log.info('Dumping pairwise_df to memmap')
    memmap_path = _dump_pairwise_memmap(pairwise_df, dataset_name)
    log.done('Dumping pairwise_df to memmap')

    # Initialize results variable
    results = None
    
    try:
        # Compatible parallel execution for older joblib versions
        log.started("Processing chunks in parallel")
        actual_n_jobs = min(effective_n_jobs, len(chunks))
        log.info(
            "Per-complex parallel config: "
            f"n_jobs={actual_n_jobs}, requested_n_jobs={effective_n_jobs}, "
            f"chunk_size={effective_chunk_size}, chunks={len(chunks)}, "
            f"max_nbytes={max_nbytes}"
        )

        # Use a more conservative approach with older joblib
        results = Parallel(
            n_jobs=actual_n_jobs,
            temp_folder=os.path.dirname(memmap_path),     
            max_nbytes=max_nbytes,
            verbose=1  # Show progress
        )(delayed(_process_chunk)(chunk, min_genes, memmap_path, gene_to_pair_indices) 
          for chunk in tqdm(chunks, desc="Per-complex PRA"))
        
        log.done("Processing chunks in parallel")
        
    except Exception as e:
        log.error(f"Error during parallel processing: {e}")
        log.error(f"Error type: {type(e).__name__}")
        # Still try to clean up the memmap file
        try:
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
                log.info(f"Cleaned up temporary memmap file after error: {memmap_path}")
        except OSError as cleanup_error:
            log.warning(f"Failed to remove memmap file after error {memmap_path}: {cleanup_error}")
        raise  # Re-raise the original exception
    
    finally:
        # Ensure cleanup happens regardless of success or failure
        try:
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
                log.info(f"Cleaned up temporary memmap file: {memmap_path}")
        except OSError as e:
            log.warning(f"Failed to remove memmap file {memmap_path}: {e}")

    # Merge results with enhanced error handling
    auc_scores = {}
    corrected_auc_scores = {}
    errors = []
    if results:
        for i, res in enumerate(results):
            if isinstance(res, dict):
                if 'error' in res:
                    log.error(f"Error in chunk {i}: {res['error']}")
                    errors.append(f"chunk {i}: {res['error']}")
                elif 'auc' in res and 'corrected_auc' in res:
                    # New format with both AUC types
                    auc_scores.update(res['auc'])
                    corrected_auc_scores.update(res['corrected_auc'])
                else:
                    # Fallback for old format (backward compatibility)
                    auc_scores.update(res)
            elif isinstance(res, tuple) and len(res) >= 2 and res[0] is None:
                log.error(f"Chunk {i} error: {res[1]}")
                errors.append(f"chunk {i}: {res[1]}")
            else:
                log.warning(f"Unexpected result type from chunk {i}: {type(res)} - {res}")
                errors.append(f"chunk {i}: unexpected result type {type(res)}")

    if errors:
        preview = "; ".join(errors[:3])
        extra = f" ({len(errors) - 3} more)" if len(errors) > 3 else ""
        raise RuntimeError(f"Per-complex PRA failed in worker chunks: {preview}{extra}")
    
    # Add the computed AUPRC values to the terms DataFrame.
    terms["auc_score"] = pd.Series(auc_scores)
    terms["corrected_auc_score"] = pd.Series(corrected_auc_scores)
    dsave(terms, "pra_percomplex", dataset_name)
    log.done(f"Per-complex PRA completed.")
    return terms

def complex_contributions(name):
    log.info(f"Computing complex contributions (Greedy) for dataset: {name}")
    pra = dload("pra", name)
    terms = dload("common", f"terms_{name}")
    if not isinstance(terms, pd.DataFrame):
        # Fallback for backward compatibility
        terms = dload("common", "terms")
    if not isinstance(pra, pd.DataFrame) or pra.empty:
        raise RuntimeError(f"complex_contributions(): PRA data for dataset '{name}' not found.")
    if not isinstance(terms, pd.DataFrame) or terms.empty:
        raise RuntimeError(f"complex_contributions(): terms for dataset '{name}' not found.")
    
    # Respect the dataset's score direction: high scores by default, low scores if configured.
    pra = pra.sort_values(
        by='score',
        ascending=_sort_ascending_for_dataset(name),
    ).reset_index(drop=True)
    
    # Compute cumulative TP and precision (matches R's TP.count = cumsum(true), Precision = TP / (1:n))
    pra['cumTP'] = pra['prediction'].cumsum()
    pra['rank'] = pra.index + 1
    pra['precision'] = pra['cumTP'] / pra['rank']
    
    # R-style precision thresholds (matches c( min, seq(0.1, max, 0.025) ) rounded)
    prec_min = pra['precision'].min()
    prec_max = pra['precision'].max()
    precision_cutoffs = [round(prec_min, 3)]
    cutoffs_range = np.arange(0.1, prec_max + 0.001, 0.025)
    precision_cutoffs += [round(t, 3) for t in cutoffs_range if t > prec_min]
    thresholds = sorted(set(precision_cutoffs))  # Ensure unique and sorted
    
    # Precompute positives for faster access
    pos_mask = pra['prediction'] == 1
    positives = pra[pos_mask].reset_index(drop=True)
    
    # Compute global unique ordered IDs for initial tie-breaking (appearance order from all positives)
    global_row_to_cids = []
    for ids in positives['complex_id']:
        if isinstance(ids, str):
            cleaned = [str(int(float(i.strip()))) for i in ids.split(';') if i.strip()]
        else:
            cleaned = [str(int(i)) for i in ids if pd.notnull(i)]
        global_row_to_cids.append(cleaned)
    all_global_ids = [cid for cids in global_row_to_cids for cid in cids]
    global_unique_ordered = list(OrderedDict.fromkeys(all_global_ids))
    
    results = {}
    valid_thresholds = []  # Track valid like R's ind.valid.precision
    
    # Progress bar for the main loop (thresholds)
    with tqdm(total=len(thresholds), desc="Processing thresholds", unit="thresh") as pbar:
        for thresh_idx, t in enumerate(thresholds):
            # Check if valid (matches R's ind.valid.precision)
            if not (prec_min <= t <= prec_max):
                pbar.update(1)
                continue
            valid_thresholds.append(thresh_idx)  # Track for later sorting
            
            # Find rightmost k where precision >= t (matches R's cand.ind[length(cand.ind)])
            cand_mask = pra['precision'] >= t
            if not cand_mask.any():
                pbar.update(1)
                continue
            k = pra.index[cand_mask].max()
            tp_target = pra.loc[k, 'cumTP']
            if tp_target <= 0:
                pbar.update(1)
                continue
            
            # Find first ind where cumTP == tp_target (matches R's tmp.ind[1])
            matching_inds = pra[pra['cumTP'] == tp_target].index
            if matching_inds.empty:
                pbar.update(1)
                continue
            ind = matching_inds.min()  # First (smallest) like R
            
            # Get top (ind+1) rows, filter to prediction==1 and non-null complex_id
            tmp = pra.iloc[0:ind + 1]
            tmp = tmp[(tmp['prediction'] == 1) & tmp['complex_id'].notnull()].reset_index(drop=True)
            if tmp.empty:
                pbar.update(1)
                continue
            
            # Build row_to_cids as list of lists (str for consistency, matches R strsplit)
            row_to_cids = []
            for ids in tmp['complex_id']:
                if isinstance(ids, str):
                    cleaned = [str(int(float(i.strip()))) for i in ids.split(';') if i.strip()]
                else:
                    cleaned = [str(int(i)) for i in ids if pd.notnull(i)]
                row_to_cids.append(cleaned)
            
            N = len(row_to_cids)
            cid_to_rows = defaultdict(list)
            for row_idx in range(N):
                for cid in row_to_cids[row_idx]:
                    cid_to_rows[cid].append(row_idx)
            
            current_size = {cid: len(lst) for cid, lst in cid_to_rows.items()}
            covered = np.zeros(N, dtype=bool)
            remaining_rows = list(range(N))  # Track remaining for tie-breaking
            final_contrib = {}
            is_first = True  # Flag for initial greedy step
            
            while current_size:
                if not current_size:
                    break
                max_contrib = max(current_size.values())
                candidates = [cid for cid, cnt in current_size.items() if cnt == max_contrib]
                
                if len(candidates) == 1:
                    top_cid = candidates[0]
                else:
                    if is_first:
                        # Initial tie-break: first in global appearance order (matches R's global matrix row order)
                        positions = {cid: global_unique_ordered.index(cid) for cid in candidates if cid in global_unique_ordered}
                        top_cid = min(positions, key=positions.get)
                    else:
                        # Subsequent: first in local remaining appearance order
                        all_ids = [cid for ri in remaining_rows for cid in row_to_cids[ri]]
                        unique_ordered = list(OrderedDict.fromkeys(all_ids))
                        positions = {cid: unique_ordered.index(cid) for cid in candidates if cid in unique_ordered}
                        top_cid = min(positions, key=positions.get)  # Earliest appearance
                
                contrib = current_size[top_cid]
                if contrib <= 0:
                    current_size.pop(top_cid, None)
                    continue
                
                # Cover the remaining rows for top_cid
                for row_idx in cid_to_rows[top_cid]:
                    if not covered[row_idx]:
                        covered[row_idx] = True
                        for cid in row_to_cids[row_idx]:
                            if cid in current_size:
                                current_size[cid] -= 1
                                if current_size[cid] <= 0:
                                    current_size.pop(cid, None)
                
                # Update remaining_rows (remove covered)
                remaining_rows = [ri for ri in remaining_rows if not covered[ri]]
                
                final_contrib[top_cid] = contrib
                is_first = False  # Only first time is special
            
            # Store for this threshold
            for cid, count in final_contrib.items():
                if cid not in results:
                    results[cid] = [0] * len(thresholds)
                results[cid][thresh_idx] = count
            
            pbar.update(1)  # Update progress after processing threshold
    
    # Build result DataFrame (index=cid as str)
    r = pd.DataFrame(results, index=thresholds).T
    r.index = r.index.astype(str)
    
    # Filter to non-zero first (matches R's nonzero.cont.ind)
    r = r[r.sum(axis=1) > 0]
    
    # Intersect with terms IDs, preserving terms order 
    gold_ids = set(r.index)
    common_ids = [str(id) for id in terms.index if str(id) in gold_ids]
    r = r.loc[common_ids]
    
    # Map Names and insert as first column
    t = pd.Series(terms['Name'].values, index=terms.index.astype(str))
    r.insert(0, 'Name', r.index.map(t))
    
    # Set all column names: Name + Precision_*
    precision_cols = [f"Precision_{t}" for t in thresholds]
    r.columns = ['Name'] + precision_cols
    
    # Sort by the last valid precision column descending, stable sort (matches R's stable order)
    if valid_thresholds:
        last_valid_col = f"Precision_{thresholds[valid_thresholds[-1]]}"
        r = r.sort_values(by=last_valid_col, ascending=False, kind='stable')
    
    # De-duplicate by Name, keeping first (matches R's !duplicated(Name) after function)
    r = r[~r['Name'].duplicated(keep='first')].reset_index(drop=True)
    
    dsave(r, "complex_contributions", name)
    log.info(f"Complex contribution (Greedy) completed for dataset: {name}")
    return r

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def perform_corr(df, corr_func):
    if corr_func not in {"numpy", "numpy_without_mask","pandas","numba"}:
        raise ValueError("corr_func must be 'numpy', 'numpy_without_mask', 'numba', or 'pandas'")

    log.started(f"Performing correlation using '{corr_func}' method.")
    
    x_axis, y_axis = df.shape
    log.info(f"Data shape: {x_axis} features, {y_axis} samples")

    if corr_func == "numpy":
        M    = np.ma.masked_invalid(df.values)
        corr = np.ma.corrcoef(M)
        arr  = corr.filled(np.nan)
        df_corr = pd.DataFrame(arr, index=df.index, columns=df.index)
        np.fill_diagonal(df_corr.values, np.nan)
        # check shape is x_axis x x_axis
        if df_corr.shape != (x_axis, x_axis):
            raise ValueError(f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {df_corr.shape}")
        log.done("Correlation.")
        return df_corr
    
    elif corr_func == "numpy_without_mask":
        corr = np.corrcoef(df.values)
        df_corr = pd.DataFrame(corr, index=df.index, columns=df.index)
        np.fill_diagonal(df_corr.values, np.nan)
        if df_corr.shape != (x_axis, x_axis):
            raise ValueError(f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {df_corr.shape}")
        log.done("Correlation.")
        return df_corr
    
    
    elif corr_func == "numba":
        corr = fast_corr(df)
        np.fill_diagonal(corr.values, np.nan)
        if corr.shape != (x_axis, x_axis):
            raise ValueError(f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {corr.shape}")
        log.done("Correlation using Numba.")
        return corr   
    
    else:
        # Compute correlations and modify diagonal in-place
        corr = df.T.corr()
        np.fill_diagonal(corr.values, np.nan)
        if corr.shape != (x_axis, x_axis):
            raise ValueError(f"Correlation matrix shape mismatch: expected ({x_axis}, {x_axis}), got {corr.shape}")
        return corr

def fast_corr(df):
    @njit(parallel=True)
    def compute_corr(data):
        m, n = data.shape
        corr = np.full((n, n), np.nan, dtype=np.float64)
        # Compute off-diagonal (upper triangle, parallel over i)
        for i in prange(n):
            for j in range(i + 1, n):
                sum_x = 0.0
                sum_y = 0.0
                sum_xx = 0.0
                sum_yy = 0.0
                sum_xy = 0.0
                count = 0
                for k in range(m):
                    x = data[k, i]
                    y = data[k, j]
                    if not np.isnan(x) and not np.isnan(y):
                        sum_x += x
                        sum_y += y
                        sum_xx += x * x
                        sum_yy += y * y
                        sum_xy += x * y
                        count += 1
                if count >= 2:
                    # Sample variance/covariance (div by count-1)
                    var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
                    var_y = (sum_yy - (sum_y ** 2) / count) / (count - 1)
                    cov = (sum_xy - (sum_x * sum_y) / count) / (count - 1)
                    denom = np.sqrt(var_x * var_y)
                    if denom > 0:  # Avoid div-by-zero (e.g., constant cols -> nan)
                        r = cov / denom
                    else:
                        r = np.nan
                else:
                    r = np.nan
                corr[i, j] = r
                corr[j, i] = r  # Symmetric
        # Compute diagonal in parallel
        for i in prange(n):
            sum_x = 0.0
            sum_xx = 0.0
            count = 0
            for k in range(m):
                x = data[k, i]
                if not np.isnan(x):
                    sum_x += x
                    sum_xx += x * x
                    count += 1
            if count >= 2:
                var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
                if var_x > 0:
                    corr[i, i] = 1.0
                else:
                    corr[i, i] = np.nan  # Constant column
            else:
                corr[i, i] = np.nan
        return corr
    
    df_numeric = df.select_dtypes(include=np.number)
    data = df_numeric.to_numpy().T
    corr_matrix = compute_corr(data)
    corr_df = pd.DataFrame(corr_matrix, index=df_numeric.index, columns=df_numeric.index)
    return corr_df

def is_symmetric(df):
    return np.allclose(df, df.T, equal_nan=True)

def binary(corr):
    log.started("Converting correlation matrix to pair-wise format.")
    if is_symmetric(corr):
        corr = convert_full_to_half_matrix(corr)
    
    stack = corr.stack().rename_axis(index=['gene1', 'gene2']).\
            reset_index().rename(columns={0: 'score'})
    if stack.empty:
        log.done("Pair-wise conversion.")
        return stack
    if has_mirror_of_first_pair(stack):
        log.info("Mirror pairs detected. Dropping them to ensure unique gene pairs.")
        stack = drop_mirror_pairs(stack)
    log.done("Pair-wise conversion.")
    return stack

def has_mirror_of_first_pair(df):
    g1, g2 = df.iloc[0]['gene1'], df.iloc[0]['gene2']
    mirror_exists = ((df['gene1'] == g2) & (df['gene2'] == g1)).iloc[1:].any()
    return mirror_exists

def convert_full_to_half_matrix(df):
    if not is_symmetric(df):
        raise ValueError("Matrix must be symmetric to convert to half matrix.")

    log.started("Converting full correlation matrix to upper triangle (half-matrix) format.")
    arr = df.values.copy()
    arr[np.tril_indices_from(arr)] = np.nan  # zero-based lower triangle + diagonal → NaN
    log.done("Matrix conversion.")
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def drop_mirror_pairs(df):
    log.started("Dropping mirror pairs to ensure unique gene pairs (Optimized).")
    gene_pairs = np.sort(df[["gene1", "gene2"]].to_numpy(), axis=1)
    df.loc[:, ["gene1", "gene2"]] = gene_pairs
    df = df.loc[~df.duplicated(subset=["gene1", "gene2"], keep="first")]
    log.done("Mirror pairs are dropped.")
    return df

def quick_sort(df, ascending=False):
    log.started(f"Pair-wise matrix is sorting based on the 'score' column: ascending:{ascending}")
    order = 1 if ascending else -1
    sorted_df = df.iloc[np.argsort(order * df["score"].values)].reset_index(drop=True)
    log.done("Pair-wise matrix sorting.")
    return sorted_df

def save_results_to_csv(categories = ["complex_contributions", "pr_auc", "pra_percomplex", "mpr_complexes_auc"]):

    config = dload("config")  # Load config to get output folder
    output_folder = Path(config.get("output_folder", "output"))
    output_folder = output_folder / "csv"  # Create a subfolder for results
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists
      
    for category in categories:
        data = dload(category)  # Load the data for this category
        if data is None:
            log.warning(f"No data found for category '{category}'. Skipping save.")
            continue

        if category == "mpr_complexes_auc" and isinstance(data, dict):
            # Dict[dataset_name -> Dict[variant_key -> auc]]
            try:
                df = pd.DataFrame.from_dict(data, orient="index")
                df.index.name = "Dataset"
                csv_path = output_folder / f"{category}.csv"
                df.to_csv(csv_path, index=True)
                log.info(f"Saved '{category}' to {csv_path}")
            except Exception as e:
                log.warning(f"Failed to convert and save '{category}': {e}")
            continue
        
        if category == "pr_auc" and isinstance(data, dict):
            # Special handling: Convert dict to DataFrame (assuming keys are indices, values are data)
            # If values are scalars, create a simple DF with 'Dataset' and 'AUC' columns
            try:
                df = pd.DataFrame.from_dict(data, orient='index', columns=['AUC'])
                df.index.name = 'Dataset'
                txt_path = output_folder / f"{category}.txt"
                df.to_csv(txt_path, sep='\t', index=True)  # Save as tab-delimited TXT
                log.info(f"Saved '{category}' as tabular TXT to {txt_path}")
            except Exception as e:
                log.warning(f"Failed to convert and save '{category}' as TXT: {e}")
            continue  # Skip to next category after handling pr_auc
        
        if isinstance(data, dict):
            # If it's a dict of datasets, save each as a separate CSV
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    csv_path = output_folder / f"{category}_{key}.csv"
                    df.to_csv(csv_path, index=False)
                    log.info(f"Saved '{category}_{key}' to {csv_path}")
                else:
                    log.warning(f"Skipping non-DataFrame item '{key}' in '{category}'.")
        elif isinstance(data, pd.DataFrame):
            # If it's a single DataFrame, save it directly
            csv_path = output_folder / f"{category}.csv"
            data.to_csv(csv_path, index=False)
            log.info(f"Saved '{category}' to {csv_path}")
        else:
            log.warning(f"Unsupported data type for '{category}'. Expected DataFrame or dict of DataFrames. Skipping.")

    log.done("Results saved to CSV files in the output folder.")

# -----------------------------------------------------------------------------
# mPR preparation (module-level precision–recall, Fig. 1E / 1F)
# -----------------------------------------------------------------------------


def _mpr_get_mtRibo_ETCI_ids(terms_like):
    """
    Identify mitochondrial ribosome and ETC I complexes to remove.

    Rule (matching the FLEX paper):
      - Name contains 'Respiratory chain complex I (holoenzyme)'
      - OR Name contains '55S'
    """
    if "Name" not in terms_like.columns:
        raise KeyError("mpr_prepare(): expected a 'Name' column in the CORUM terms.")

    name = terms_like["Name"].astype(str)
    mask = name.str.contains(
        "Respiratory chain complex I \\(holoenzyme\\)", case=False, regex=True
    ) | name.str.contains("55S", case=False, regex=False)

    return set(terms_like.index[mask])


def _mpr_get_small_high_auprc_ids(
    pra_percomplex, size_th=30, auprc_th=0.4, use_corrected=True
):
    """
    Identify complexes that are small and have high per-complex AUPRC.

    Small: full CORUM size (Length) < size_th
    High AUPRC: per-complex AUPRC >= auprc_th
    """
    if "Length" not in pra_percomplex.columns:
        raise KeyError(
            "mpr_prepare(): expected a 'Length' column in the per-complex table."
        )

    if use_corrected and "corrected_auc_score" in pra_percomplex.columns:
        score_col = "corrected_auc_score"
    elif "auc_score" in pra_percomplex.columns:
        score_col = "auc_score"
    else:
        raise KeyError(
            "mpr_prepare(): expected 'corrected_auc_score' or 'auc_score' in the per-complex table."
        )

    size_mask = pra_percomplex["Length"] < size_th
    score_mask = pra_percomplex[score_col] >= auprc_th

    mask = size_mask & score_mask
    return set(pra_percomplex.index[mask])


# -------------------------------------------------------------------------
# Helpers implementing the FLEX stepwise module-level PR logic
# -------------------------------------------------------------------------

def _mpr_build_pairs(pra, removed_ids=None, ascending=False):
    """
    Build a Pairs.in.data-like table for mPR / stepwise contributions.

    Rows containing filtered positive complex IDs are removed from the ranking,
    matching the FLEX stepwise module-level precision-recall behavior.

    Input:
      pra : DataFrame with at least columns
            - 'score'       : ranking score
            - 'complex_id'  : complex annotations
      removed_ids : set[int] of complexes to remove

    Output:
      DataFrame with columns:
        - predicted   : score
        - true        : 0/1
        - complex_ids : list[int] per row
    """
    if "complex_id" not in pra.columns and "complex_ids" not in pra.columns:
        raise RuntimeError(
            "mpr_prepare(): expected a 'complex_id' or 'complex_ids' column in 'pra'."
        )

    removed_ids = set(int(x) for x in (removed_ids or []))

    df = pra.copy()

    # Normalize complex-ID column name
    if "complex_id" in df.columns:
        cid_col = "complex_id"
    else:
        cid_col = "complex_ids"

    if "score" not in df.columns:
        raise RuntimeError("mpr_prepare(): expected a 'score' column in 'pra'.")

    def normalize_ids(cell):
        """Parse complex IDs from various formats."""
        if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
            return [int(x) for x in cell if pd.notnull(x)]
        elif isinstance(cell, str):
            if not cell:
                return []
            parts = [p for p in cell.split(";") if p]
            return [int(float(p)) for p in parts]
        elif pd.isna(cell):
            return []
        else:
            try:
                return [int(cell)]
            except Exception:
                return []

    def should_remove(cell):
        """Check if this row should be removed (contains any removed_id AND is a TP)."""
        ids = normalize_ids(cell)
        if not ids:
            return False  # Not a TP, keep it
        # Remove if ANY of the IDs is in removed_ids
        return any(cid in removed_ids for cid in ids)

    # Build output DataFrame
    out = pd.DataFrame(index=df.index)
    out["predicted"] = df["score"].astype(float)
    out["complex_ids"] = df[cid_col].apply(normalize_ids)
    out["true"] = out["complex_ids"].apply(lambda ids: 1 if len(ids) > 0 else 0)
    
    # Remove rows that are TPs and contain a removed complex ID.
    if removed_ids:
        should_remove_mask = df[cid_col].apply(should_remove)
        remove_mask = should_remove_mask & (out["true"] == 1)
        out = out[~remove_mask].copy()
    
    # Also filter the complex_ids to remove the removed IDs (for stepwise contributions)
    if removed_ids:
        out["complex_ids"] = out["complex_ids"].apply(
            lambda ids: [cid for cid in ids if cid not in removed_ids]
        )

    # Sort by the dataset's configured score direction.
    out = out.sort_values("predicted", ascending=ascending).reset_index(drop=True)
    return out


def _mpr_precision_cutoffs_from_pairs(pairs, step=0.025):
    """
    Choose precision cutoffs similar to FLEX:
      - start at min positive precision
      - add grid 0.10, 0.125, 0.15, ... up to max precision
    """
    true = pairs["true"].to_numpy(dtype=int)
    n = len(true)
    if n == 0 or true.sum() == 0:
        return np.array([1.0], dtype=float)

    tp_cum = true.cumsum()
    denom = np.arange(n, dtype=float) + 1.0
    precision = tp_cum / denom

    pos_prec = precision[true == 1]
    min_p = float(pos_prec.min())
    max_p = float(precision.max())

    cuts = [round(min_p, 3)]

    v = 0.10
    while v <= max_p + 1e-9:
        if v > min_p:
            cuts.append(round(v, 3))
        v += step

    cuts = sorted(set(cuts))
    return np.array(cuts, dtype=float)


def _mpr_stepwise_contributions(pairs, precision_cutoffs, ascending=False):
    """
    Greedy, stepwise TP allocation per complex at each precision cutoff.

    Input:
      pairs : DataFrame with columns
              - predicted (float)
              - true (0/1)
              - complex_ids : list[int]
      precision_cutoffs : 1D array of precision thresholds

    Output:
      contrib_df : DataFrame [complex_id x cutoff] with TP counts
    """
    pairs = pairs.copy()
    pairs = pairs.sort_values("predicted", ascending=ascending).reset_index(drop=True)

    true = pairs["true"].to_numpy(dtype=int)
    n = len(true)
    if n == 0 or true.sum() == 0:
        return pd.DataFrame()

    tp_cum = true.cumsum()
    denom = np.arange(n, dtype=float) + 1.0
    precision = tp_cum / denom

    complex_lists = []
    for cell in pairs["complex_ids"].tolist():
        if isinstance(cell, (list, tuple, np.ndarray, pd.Series)):
            complex_lists.append([int(x) for x in cell if pd.notnull(x)])
        elif pd.isna(cell):
            complex_lists.append([])
        else:
            try:
                complex_lists.append([int(cell)])
            except Exception:
                complex_lists.append([])

    all_cids = sorted({cid for cids in complex_lists for cid in cids})
    if not all_cids:
        return pd.DataFrame()

    cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}
    n_cids = len(all_cids)
    n_cut = len(precision_cutoffs)

    contrib = np.zeros((n_cids, n_cut), dtype=float)

    pos_prec = precision[true == 1]
    prec_min = float(pos_prec.min())
    prec_max = float(precision.max())

    for j, cutoff in enumerate(precision_cutoffs):
        if cutoff < prec_min or cutoff > prec_max:
            continue

        cand_mask = precision >= cutoff
        if not np.any(cand_mask & (true == 1)):
            continue

        k = np.where(cand_mask)[0][-1]
        tp_target = tp_cum[k]
        i_end = np.where(tp_cum == tp_target)[0][0]

        rows = np.arange(i_end + 1, dtype=int)
        tp_rows = rows[true[rows] == 1]
        if tp_rows.size == 0:
            continue

        cid_to_rows = {}
        for r in tp_rows:
            for cid in complex_lists[r]:
                cid_to_rows.setdefault(cid, set()).add(r)

        covered = set()

        while True:
            best_cid = None
            best_size = 0
            for cid, rset in cid_to_rows.items():
                size = len(rset - covered)
                if size > best_size:
                    best_size = size
                    best_cid = cid

            if best_cid is None or best_size == 0:
                break

            new_rows = cid_to_rows[best_cid] - covered
            covered.update(new_rows)
            row_idx = cid_to_idx[best_cid]
            contrib[row_idx, j] = float(len(new_rows))

    contrib_df = pd.DataFrame(
        contrib,
        index=pd.Index(all_cids, name="complex_id"),
        columns=precision_cutoffs,
    )
    return contrib_df








def _mpr_module_coverage(contrib_df, terms, tp_th=1, percent_th=0.1):
    """
    Convert stepwise contribution matrix to "#covered complexes" per cutoff.

    contrib_df : rows = complex_id (int), columns = precision_cutoffs (float)
    terms      : CORUM 'terms' table (index = complex_id)
    
    A complex is covered at a precision cutoff if:
    1. It contributes at least tp_th TP pairs (stepwise)
    2. The fraction of contributed pairs vs total possible pairs > percent_th
       (matches R behavior: x > percent_th)
    """
    if contrib_df.empty:
        return np.zeros(0, dtype=float)

    precision_cutoffs = np.asarray(contrib_df.columns, dtype=float)
    data = contrib_df.to_numpy(dtype=float)
    n_cut = data.shape[1]

    n_pairs = np.zeros(data.shape[0], dtype=float)
    for i, cid in enumerate(contrib_df.index):
        cid_int = int(cid)
        if cid_int not in terms.index:
            n_pairs[i] = 0.0
            continue
        row = terms.loc[cid_int]

        n_genes = None

        # Prefer used_genes (genes actually in the dataset) for a fair coverage
        # fraction. This matters for GOBP/PATHWAY where all_genes >> used_genes.
        if "used_genes" in row.index:
            genes = row["used_genes"]
            if isinstance(genes, (list, np.ndarray)) and len(genes) > 0:
                n_genes = len(genes)
        if n_genes is None and "n_used_genes" in row.index:
            try:
                v = int(row["n_used_genes"])
                if v > 0:
                    n_genes = v
            except (ValueError, TypeError):
                pass

        # Fallback: all_genes (how it's stored in preprocessing)
        if n_genes is None and "all_genes" in row.index:
            genes = row["all_genes"]
            if isinstance(genes, (list, np.ndarray)):
                n_genes = len(genes)
            elif isinstance(genes, str):
                n_genes = len([g for g in genes.split(";") if g])

        # Fallback to Genes column (original string from CORUM)
        if n_genes is None and "Genes" in row.index:
            genes_str = row["Genes"]
            if isinstance(genes_str, str):
                n_genes = len([g for g in genes_str.split(";") if g])
        
        # Fallback to n_all_genes (computed during preprocessing)
        if n_genes is None and "n_all_genes" in row.index:
            try:
                n_genes = int(row["n_all_genes"])
            except (ValueError, TypeError):
                n_genes = None
        
        # Fallback to Length column (from original CORUM file)
        if n_genes is None and "Length" in row.index:
            try:
                n_genes = int(row["Length"])
            except (ValueError, TypeError):
                n_genes = None

        if n_genes is None or n_genes < 2:
            n_pairs[i] = 0.0
        else:
            n_pairs[i] = n_genes * (n_genes - 1) / 2.0

    coverage = np.zeros(n_cut, dtype=float)

    for j in range(n_cut):
        tps = data[:, j]
        mask = (tps >= tp_th) & (n_pairs > 0)
        if not np.any(mask):
            coverage[j] = 0.0
            continue

        frac = np.zeros_like(tps)
        frac[mask] = tps[mask] / n_pairs[mask]
        # Note: Using > (strict inequality) to match R code behavior
        covered = (tps >= tp_th) & (frac > percent_th)
        coverage[j] = float(covered.sum())

    return coverage


def _mpr_complexes_auc(
    coverage: np.ndarray,
    precision_cutoffs: np.ndarray,
    max_complexes: float = 200.0,
) -> float:
    """Compute AUC for the Fig. 1F-style mPR curve (#complexes vs precision).

    The plot uses:
      x = #covered complexes (capped at `max_complexes`, shown on a log axis)
      y = precision cutoff

    We compute a normalized AUC by integrating precision over the *normalized*
    coverage axis:
        AUC = integral y d(x/max_complexes)

    This yields a score in [0, 1] (or NaN if insufficient data).
    """
    cov = np.asarray(coverage, dtype=float)
    prec = np.asarray(precision_cutoffs, dtype=float)

    if cov.size == 0 or prec.size == 0:
        return 0.0

    # Match plot_mpr_complex_coverage_curve(): only count cov>0 (log-x cannot show 0)
    mask = (
        np.isfinite(cov)
        & np.isfinite(prec)
        & (cov > 0)
        & (cov <= max_complexes)
        & (prec >= 0)
        & (prec <= 1.0)
    )
    if not np.any(mask):
        return 0.0

    x_cov = cov[mask]
    y = prec[mask]

    # x-axis is log-scaled in the plot; normalize so cov=1 -> 0, cov=max_complexes -> 1
    # (This matches the plot's tick hack where 1 is labeled as "0".)
    x = np.log10(x_cov) / np.log10(float(max_complexes))

    # Sort by x and collapse duplicate x values by taking max y (upper envelope)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_unique = np.unique(x)
    if x_unique.size != x.size:
        y = np.array([float(np.nanmax(y[x == xv])) for xv in x_unique], dtype=float)
        x = x_unique

    if x.size < 2:
        return 0.0

    return float(np.trapz(y, x))










def mpr_prepare(
    name,
    size_th=30,
    auprc_th=0.4,
    tp_th=1,
    percent_th=0.1,
    use_corrected=True,
):
    """
    Prepare data for Fig. 1E (TP vs precision) and Fig. 1F (mPR) for dataset `name`.

    Stores an 'mpr' object with:
      - precision_cutoffs
      - tp_curves[label]         : full PR (TP vs precision) per filter
      - coverage_curves[label]   : #covered complexes per cutoff per filter
      - filters metadata
    """
    pra = dload("pra", name)
    pra_percomplex = dload("pra_percomplex", name)
    terms = dload("common", f"terms_{name}")
    if not isinstance(terms, pd.DataFrame):
        # Fallback for backward compatibility
        terms = dload("common", "terms")

    if pra is None or not isinstance(pra, pd.DataFrame) or pra.empty:
        raise RuntimeError(
            f"mpr_prepare(): PRA data for dataset '{name}' not found "
            "(dload('pra', name))."
        )
    if pra_percomplex is None or not isinstance(pra_percomplex, pd.DataFrame) or pra_percomplex.empty:
        raise RuntimeError(
            f"mpr_prepare(): per-complex PRA data for dataset '{name}' not found "
            "(dload('pra_percomplex', name))."
        )
    if terms is None or not isinstance(terms, pd.DataFrame) or terms.empty:
        raise RuntimeError(
            "mpr_prepare(): functional standard 'terms' table not found (dload('common', 'terms'))."
        )

    ascending = _sort_ascending_for_dataset(name)

    # Sort by the dataset's configured score direction.
    if "score" in pra.columns:
        pra = pra.sort_values("score", ascending=ascending).reset_index(drop=True)
    else:
        pra = pra.reset_index(drop=True)

    # filters
    mtRibo_ids = _mpr_get_mtRibo_ETCI_ids(pra_percomplex)
    small_hi_ids = _mpr_get_small_high_auprc_ids(
        pra_percomplex,
        size_th=size_th,
        auprc_th=auprc_th,
        use_corrected=use_corrected,
    )

    filter_sets = {
        "all": set(),
        "no_mtRibo_ETCI": set(mtRibo_ids),
        "no_small_highAUPRC": set(small_hi_ids),
    }

    tp_curves = {}
    coverage_curves = {}
    complexes_auc = {}
    precision_cutoffs = None

    for label, removed in filter_sets.items():
        # 1) Build pairs table after removing complexes in `removed`
        pairs = _mpr_build_pairs(pra, removed_ids=removed, ascending=ascending)

        true = pairs["true"].to_numpy(dtype=int)
        n = len(true)
        if n == 0 or true.sum() == 0:
            tp_curves[label] = {
                "tp": np.array([], dtype=float),
                "precision": np.array([], dtype=float),
            }
            coverage_curves[label] = np.zeros(0, dtype=float)
            complexes_auc[label] = float("nan")
            continue

        tp_cum = true.cumsum()
        denom = np.arange(n, dtype=float) + 1.0
        precision = tp_cum / denom

        # full PR: only positions where we add a TP
        mask_tp = true == 1
        tp_full = tp_cum[mask_tp]
        prec_full = precision[mask_tp]
        tp_curves[label] = {"tp": tp_full, "precision": prec_full}

        # common precision grid from 'all'
        if precision_cutoffs is None:
            precision_cutoffs = _mpr_precision_cutoffs_from_pairs(pairs)

        contrib_df = _mpr_stepwise_contributions(
            pairs,
            precision_cutoffs,
            ascending=ascending,
        )
        cov = _mpr_module_coverage(
            contrib_df,
            terms,
            tp_th=tp_th,
            percent_th=percent_th,
        )
        # precision_cutoffs are sorted ascending (low → high).
        # Coverage must be non-increasing in that direction: a more permissive
        # threshold (lower precision) should never yield fewer covered terms.
        # The independent greedy allocation per cutoff can violate this, so
        # enforce monotonicity by propagating the max from right to left.
        if cov.size > 0:
            cov = np.maximum.accumulate(cov[::-1])[::-1]
        coverage_curves[label] = cov
        complexes_auc[label] = _mpr_complexes_auc(
            cov,
            precision_cutoffs,
            max_complexes=200.0,
        )

    mpr_data = {
        "precision_cutoffs": precision_cutoffs,
        "tp_curves": tp_curves,
        "coverage_curves": coverage_curves,
        "complexes_auc": complexes_auc,
        "filters": {
            "no_mtRibo_ETCI": sorted(mtRibo_ids),
            "no_small_highAUPRC": sorted(small_hi_ids),
            "size_th": size_th,
            "auprc_th": auprc_th,
            "percent_th": percent_th,
            "tp_th": tp_th,
            "use_corrected": bool(use_corrected),
        },
    }

    dsave(mpr_data, "mpr", name)

    # Convenience: store AUCs as their own category for easy export / plotting.
    dsave(complexes_auc, "mpr_complexes_auc", name)
