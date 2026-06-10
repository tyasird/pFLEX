import os
import pandas as pd
import numpy as np
from .utils import dsave, dload, normalize_analysis_genes
from tqdm import tqdm
from .logging_config import log
tqdm.pandas()
# Locate package data path once at module level
from importlib.metadata import distribution
from importlib.resources import files
import json
from pathlib import Path


def return_package_dir():
    try:
        # Get the distribution
        dist = distribution('pflex')

        # Check for direct_url.json
        try:
            direct_url_text = dist.read_text('direct_url.json')
        except FileNotFoundError:
            direct_url_text = None

        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            if direct_url.get('dir_info', {}).get('editable'):
                # Editable install detected
                project_url = direct_url['url']
                # Remove 'file:///' prefix and handle Windows paths
                project_root = project_url.removeprefix('file:///').replace('/', os.sep)
                # Assuming src layout: project_root/src/pflex
                package_dir = os.path.join(project_root, 'src', 'pflex')
            else:
                # Non-editable
                package_dir = str(files('pflex'))
        else:
            # No direct_url, assume non-editable
            package_dir = str(files('pflex'))
            
    except Exception: # PackageNotFoundError or other issues
        # Fallback to local directory relative to this file
        # precise location: src/pflex/preprocessing.py -> package dir is parent
        package_dir = str(Path(__file__).parent)

    return package_dir



def example_input_path(filename: str):
    return files("pflex.data").joinpath("example_input").joinpath(filename)


def get_example_data_path(filename: str):
    return example_input_path(filename)


def _load_file(filepath, ext):
    loaders = {
        ".csv": lambda f: pd.read_csv(f, index_col=0),
        ".xlsx": lambda f: pd.read_excel(f, index_col=0),
        ".parquet": pd.read_parquet,
        ".p": pd.read_parquet
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loaders[ext](filepath)


def load_datasets(files, continue_with_common_genes=False):
    config = dload("config")
    preprocessing = config["preprocessing"]
    analysis_genes_raw = config.get("analysis_genes", "")
    analysis_genes_missing = (
        analysis_genes_raw is None or str(analysis_genes_raw).strip() == ""
    )
    analysis_genes = normalize_analysis_genes(
        analysis_genes_raw,
        legacy_use_common_genes=(
            config.get("use_common_genes") if analysis_genes_missing else None
        ),
    )
    data_dict= {}     

    for filename, meta in files.items():
        if isinstance(meta, pd.DataFrame):
            df = meta
        elif isinstance(meta, dict):
            filepath = meta["path"]
            if isinstance(filepath, pd.DataFrame):
                df = filepath
            else:
                ext = os.path.splitext(filepath)[1]
                df = _load_file(filepath, ext)
        else:
            raise ValueError(f"Unsupported data structure for '{filename}': {type(meta)}")

        df.index = df.index.str.split().str[0]
        if preprocessing.get('normalize'):
            log.info(f"{filename}: Normalization.")
            df = (df - df.mean()) / df.std(ddof=0)

        if preprocessing.get('drop_na'):
            log.info(f"{filename}: Dropping missing values.")
            df = df.dropna(how="any")

        if preprocessing.get('fill_na'):
            log.info(f"{filename}: Filling missing values with column mean.")
            #df = df.T.fillna(df.mean(axis=1)).T
            df = data_imputation(df)

            
        data_dict[filename] = df

    common_genes = get_common_genes(data_dict)

    # Apply common gene filtering only when analysis_genes='shared' (or forced by arg)
    if analysis_genes == "shared" or continue_with_common_genes:
        log.info(f"Applying common gene filtering: {len(common_genes)} genes")
        for filename, df in data_dict.items():
            if df.index.isin(common_genes).any():
                data_dict[filename] = df.loc[common_genes]
    else:
        log.info(
            f"Skipping common gene filtering (analysis_genes='dataset_specific'). Common genes found: {len(common_genes)}"
        )
    
    dsave({
        "datasets": data_dict,
        "sorting": {
            k: v.get("sort", "high") if isinstance(v, dict) else "high"
            for k, v in files.items()
        },
        "colors": {
            k: v.get("color", None) if isinstance(v, dict) else None
            for k, v in files.items()
        }
    }, "input")
    log.done(f"Datasets loaded.")
    return data_dict  , common_genes




def drop_bad_samples(df, max_na=0.1):
    total_elements = df.shape[0] * df.shape[1]
    percent_nan = np.isnan(df.values).sum() / total_elements if total_elements > 0 else 0
    has_nan_per_sample = np.isnan(df.values).any(axis=0) # how many samples has NA.

    log.info(f"Total: {total_elements}, Percent NaN: {percent_nan:.2%}, Samples with NaN: {np.sum(has_nan_per_sample)} / {df.shape[1]}")

    num_genes = df.shape[0]  # E.g., 1178 (total rows/genes)
    na_per_sample = np.isnan(df.values).sum(axis=0) / num_genes  # Fraction NA per sample (column)
    good_samples = na_per_sample <= max_na  # Keep if <=10% NA
    data_filtered = df.loc[:, good_samples]  # Drop bad samples (those >10% NA)

    log.info(f"Filtered samples: {data_filtered.shape[1]} (removed {df.shape[1] - data_filtered.shape[1]} samples with >{max_na*100:.0f}% NAs)")
    return data_filtered



def data_imputation(df):
    log.info("Imputing missing values with gene means ...")
    gene_means = np.nanmean(df.values, axis=1)  # 1D array: means per gene
    data_values = df.values.copy()
    rows, cols = np.where(np.isnan(data_values))
    if len(rows) > 0:
        data_values[rows, cols] = np.take(gene_means, rows)

    df = pd.DataFrame(data_values, index=df.index, columns=df.columns)
    log.info(f"Data after imputation: {df.shape[0]} genes, {df.shape[1]} samples")
    return df




def get_common_genes(datasets):
    log.started("Finding common genes across datasets.")
    gene_sets = [set(df.index) for df in datasets.values()]
    common_genes = set.intersection(*gene_sets)
    log.done(f"Common genes found: {len(common_genes)}")
    dsave(common_genes, "common", "common_genes")
    return list(common_genes)


def filter_matrix_by_genes(matrix, genes_present_in_terms):
    log.started("Filtering matrix using genes present in terms.")
    genes = matrix.index.intersection(genes_present_in_terms)
    matrix = matrix.loc[genes, genes]
    log.done(f"Filtering matrix: {matrix.shape}")
    return matrix
    



def load_functional_standard():
    
    package_dir = return_package_dir()
    data_dir_path = os.path.join(package_dir, 'data')
    
    config = dload("config") 
    analysis_genes_raw = config.get("analysis_genes", "")
    analysis_genes_missing = (
        analysis_genes_raw is None or str(analysis_genes_raw).strip() == ""
    )
    analysis_genes = normalize_analysis_genes(
        analysis_genes_raw,
        legacy_use_common_genes=(
            config.get("use_common_genes") if analysis_genes_missing else None
        ),
    )

    functional_standard_source = config["functional_standard"]
    jaccard_enabled = bool(config.get("jaccard", False))
    log.done(
        f"Loading functional standard: {functional_standard_source}, Min module size: {config['min_genes_in_module']}, "
        f"Jaccard filtering: {jaccard_enabled} (exact duplicate used_genes after dataset filtering), "
        f"analysis_genes: {analysis_genes}"
    )

    # Define functional standard file paths for predefined sources
    functional_standard_files = {
        "CORUM": "functional_standard/CORUM.parquet",
        "GOBP": "functional_standard/GOBP.parquet",
        "PATHWAY": "functional_standard/PATHWAY.parquet"
    }
    
    if functional_standard_source in functional_standard_files:
        # Load predefined functional standard from package resources
        filename = functional_standard_files[functional_standard_source]
        filename_path = Path(data_dir_path).joinpath(filename)
        if not filename_path.exists():  # Check if the file exists
            raise ValueError(f"Invalid functional standard type: {functional_standard_source}. File not found.")
        terms = pd.read_parquet(filename_path)  # type: ignore
    elif Path(functional_standard_source).suffix.lower() == '.csv':
        # Load user-provided custom functional standard from CSV file
        filename_path = Path(functional_standard_source)
        if not filename_path.exists():
            raise ValueError(f"Custom functional standard CSV file not found: {functional_standard_source}")
        log.done(f"Loading custom functional standard from CSV: {functional_standard_source}")
        terms = pd.read_csv(filename_path)  
    else:
        raise ValueError(
            f"Invalid functional standard source: {functional_standard_source}. "
            f"Must be one of {list(functional_standard_files.keys())} or a path to a .csv file."
        )

    # Store raw functional standard for later per-dataset filtering
    terms["all_genes"] = terms["Genes"].apply(lambda x: list(set(x.split(";"))))
    log.done(f"Functional standard loaded with {len(terms)} terms")

    # Basic filtering by minimum module size (before gene filtering)
    terms["n_all_genes"] = terms["all_genes"].apply(len) 
    terms = terms[terms["n_all_genes"] >= config['min_genes_in_module']]
    log.done(f"After min_genes_in_module filtering: {len(terms)} terms")

    # if there is column called "ID", set it as index
    if "ID" in terms.columns:
        terms = terms.set_index("ID")

    dsave(terms, "common", "terms")
    log.done("Functional standard loading completed.")
    return terms, None  # Return None for genes_present_in_terms - will be computed per dataset





def filter_duplicate_terms(terms: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for exact-duplicate filtering.

    This removes exact duplicate `used_genes` sets and keeps the smallest ID.
    """
    log.started("Filtering duplicate terms using exact used_genes sets.")
    before = len(terms)
    terms = terms.copy()
    terms["gene_set"] = terms["used_genes"].map(lambda x: frozenset(x))
    grouped = terms.groupby("gene_set", sort=False)

    id_values = terms["ID"] if "ID" in terms.columns else terms.index
    keep_ids = set(id_values)
    for _, group in grouped:
        if len(group) <= 1:
            continue
        cluster = group["ID"].values if "ID" in group.columns else group.index.values
        sorted_ids = sorted(cluster)
        keep_ids.difference_update(sorted_ids[1:])

    if "ID" in terms.columns:
        filtered = terms[terms["ID"].isin(keep_ids)].copy()
    else:
        filtered = terms[terms.index.isin(keep_ids)].copy()
    filtered.drop(columns=["gene_set"], inplace=True, errors="ignore")
    log.done(f"{before - len(filtered)} terms removed due to identical gene sets.")
    return filtered
