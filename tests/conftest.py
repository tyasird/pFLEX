import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pflex as flex

# Three complexes with four genes each, plus four genes outside any complex.
COMPLEXES = {
    1: ["A1", "A2", "A3", "A4"],
    2: ["B1", "B2", "B3", "B4"],
    3: ["C1", "C2", "C3", "C4"],
}
NOISE_GENES = ["N1", "N2", "N3", "N4"]


@pytest.fixture(autouse=True)
def isolated_workdir(tmp_path, monkeypatch):
    """pFLEX keeps intermediate results in ./.tmp, so every test gets its own folder."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture
def standard_csv(tmp_path):
    """Small custom functional standard in the format pFLEX accepts."""
    rows = [
        {"ID": cid, "Name": f"Complex {cid}", "Genes": ";".join(genes), "Length": len(genes)}
        for cid, genes in COMPLEXES.items()
    ]
    path = tmp_path / "standard.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_dataset(seed=0, signal=1.0, n_samples=40):
    """Genes of one complex share a profile; noise genes are random.

    With signal=1.0 all within-complex pairs rank above all other pairs.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for genes in COMPLEXES.values():
        base = rng.normal(size=n_samples)
        for gene in genes:
            rows[gene] = signal * base + 0.1 * rng.normal(size=n_samples)
    for gene in NOISE_GENES:
        rows[gene] = rng.normal(size=n_samples)
    return pd.DataFrame(rows).T


@pytest.fixture
def config(standard_csv):
    return {
        "functional_standard": str(standard_csv),
        "min_genes_in_module": 3,
        "corr_function": "numpy",
        "output_folder": "output",
        "per_module": {"n_jobs": 1},
        "plotting": {"save_plot": True, "show_plot": False, "output_type": "png"},
    }


def run_pipeline(config, inputs):
    flex.initialize(config)
    data, _ = flex.load_datasets(inputs)
    flex.load_functional_standard()
    for name, dataset in data.items():
        corr = flex.perform_corr(dataset, config["corr_function"])
        flex.pra(name, corr, is_corr=True)
        flex.pra_per_module(name, corr, is_corr=True)
        flex.module_contributions(name)
        flex.mpr_prepare(name)
    return data
