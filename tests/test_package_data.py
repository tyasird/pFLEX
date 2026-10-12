"""Bundled functional standards and example data."""

import pandas as pd
import pytest

import pflex as flex


def test_example_data_loads():
    skin = pd.read_parquet(flex.example_input_path("skin_cell_lines_corum_genes.parquet"))
    soft = pd.read_parquet(flex.example_input_path("soft_tissue_cell_lines_corum_genes.parquet"))

    assert skin.shape == (3465, 75)
    assert soft.shape == (3465, 46)
    assert skin.index.equals(soft.index)


@pytest.mark.parametrize(
    "standard, n_terms",
    [("CORUM", 2916), ("GOBP", 4828), ("PATHWAY", 1329)],
)
def test_builtin_functional_standards(standard, n_terms):
    flex.initialize({"functional_standard": standard, "min_genes_in_module": 1})

    terms, _ = flex.load_functional_standard()

    assert len(terms) == n_terms
    assert terms["all_genes"].map(len).min() >= 1


def test_example_benchmark_smoke(tmp_path):
    """Short run on a slice of the real example data with CORUM."""
    skin = pd.read_parquet(flex.example_input_path("skin_cell_lines_corum_genes.parquet"))
    soft = pd.read_parquet(flex.example_input_path("soft_tissue_cell_lines_corum_genes.parquet"))
    genes = skin.index[:600]

    flex.initialize({
        "functional_standard": "CORUM",
        "corr_function": "numpy_without_mask",
        "preprocessing": {"fill_na": True},
        "per_module": {"n_jobs": 1},
        "plotting": {"save_plot": False, "show_plot": False},
    })
    data, _ = flex.load_datasets({
        "Skin": {"path": skin.loc[genes], "sort": "high"},
        "Soft Tissue": {"path": soft.loc[genes], "sort": "high"},
    })
    flex.load_functional_standard()
    for name, dataset in data.items():
        corr = flex.perform_corr(dataset, "numpy_without_mask")
        flex.pra(name, corr, is_corr=True)
        flex.pra_per_module(name, corr, is_corr=True)

    for name in ("Skin", "Soft_Tissue"):
        auprc = flex.dload("pr_auc", name)
        per_module = flex.dload("pra_per_module", name)
        assert 0 < auprc < 1
        assert len(per_module) > 0
        assert per_module["auc_score"].between(0, 1).all()
