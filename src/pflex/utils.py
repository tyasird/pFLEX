import os
import re
import tempfile
import joblib
import numpy as np
import pandas as pd

from .logging_config import log

# Constants
TMP_ROOT = ".tmp"
VALID_EXTS = {".parquet", ".npy", ".pkl"}  # Removed .feather

ANALYSIS_GENES_ALIASES = {
    "shared": "shared",
    "common": "shared",
    "dataset_specific": "dataset_specific",
    "dataset": "dataset_specific",
}


def normalize_analysis_genes(value=None, legacy_use_common_genes=None):
    """Normalize analysis_genes config values while keeping legacy names working."""
    if value is None or str(value).strip() == "":
        if legacy_use_common_genes is not None:
            return "shared" if bool(legacy_use_common_genes) else "dataset_specific"
        value = "shared"

    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ANALYSIS_GENES_ALIASES:
        raise ValueError(
            "config['analysis_genes'] must be one of "
            "['shared', 'dataset_specific'] "
            "(legacy aliases ['common', 'dataset'] are also supported), "
            f"got {value!r}"
        )
    return ANALYSIS_GENES_ALIASES[key]

# Helper to sanitize names (make filesystem-safe)
def _sanitize(name):
    if not name:
        return "data"
    # Replace forbidden/problematic chars with '_', collapse multiples, strip edges
    safe = re.sub(r'[<>:"/\\|?*$,\s]+', '_', str(name).strip())
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe if safe else "data"

# Helper to get safe path
def _safe_path(category, name=None, ext=".pkl"):
    safe_category = _sanitize(category)
    dir_path = os.path.join(TMP_ROOT, safe_category)
    os.makedirs(dir_path, exist_ok=True)
    safe_name = _sanitize(name) if name else "data"
    return os.path.join(dir_path, f"{safe_name}{ext}")

# Save function - Parquet for DataFrames
def dsave(data, category, name=None, path=None):
    # If data is dict and no name, recurse on each item
    if name is None and isinstance(data, dict):
        for k, v in data.items():
            dsave(v, category, k)
        return

    # Choose format based on type
    if isinstance(data, pd.DataFrame):
        ext = ".parquet"
        save_func = lambda p: data.to_parquet(p)
    elif isinstance(data, np.ndarray):
        ext = ".npy"
        save_func = lambda p: np.save(p, data, allow_pickle=False)
    else:
        ext = ".pkl"
        save_func = lambda p: joblib.dump(data, p, compress=0)

    target = _safe_path(category, name, ext)

    # Atomic save: Write to temp file, then rename
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(target), delete=False, suffix=ext) as tf:
        tmp_path = tf.name
        tf.close()
        save_func(tmp_path)
    os.replace(tmp_path, target)

# Load function - Parquet for DataFrames
def dload(category, name=None, path=None):
    dir_path = os.path.join(TMP_ROOT, _sanitize(category))

    if not os.path.exists(dir_path):
        return {} if name is None else None

    if name is None:
        # Load all in category as dict
        out = {}
        for filename in os.listdir(dir_path):
            if not any(filename.endswith(ext) for ext in VALID_EXTS):
                continue
            k = os.path.splitext(filename)[0]
            full_path = os.path.join(dir_path, filename)
            try:
                if filename.endswith(".parquet"):
                    out[k] = pd.read_parquet(full_path)
                elif filename.endswith(".npy"):
                    out[k] = np.load(full_path, mmap_mode="r")
                elif filename.endswith(".pkl"):
                    out[k] = joblib.load(full_path, mmap_mode="r")
            except (EOFError, ValueError, OSError):
                log.warning(f"'{full_path}' is corrupted. Skipping...")
                os.remove(full_path)
        return out

    # Load specific name - try extensions in order
    for ext in VALID_EXTS:
        target = _safe_path(category, name, ext)
        if os.path.exists(target):
            try:
                if ext == ".parquet":
                    return pd.read_parquet(target)
                elif ext == ".npy":
                    return np.load(target, mmap_mode="r")
                elif ext == ".pkl":
                    return joblib.load(target, mmap_mode="r")
            except (EOFError, ValueError, OSError) as e:
                log.warning(f"'{target}' is corrupted ({e}). Trying next format...")
                os.remove(target)
                continue
    
    log.warning(f"No valid file found for {category}/{name}")
    return None
