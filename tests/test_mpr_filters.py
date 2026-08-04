import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import pflex
from pflex import analysis, plotting


def _analysis_inputs():
    pra = pd.DataFrame(
        {
            "score": [0.99, 0.90, 0.80, 0.70, 0.60, 0.50],
            "module_id": ["1", "", "2", "", "3", ""],
        }
    )
    per_module = pd.DataFrame(
        {
            "Name": [
                "Respiratory chain complex I (holoenzyme)",
                "Ordinary large complex",
                "Small strong complex",
            ],
            "Length": [40, 40, 10],
            "corrected_auc_score": [0.2, 0.2, 0.8],
            "auc_score": [0.2, 0.2, 0.8],
        },
        index=[1, 2, 3],
    )
    terms = pd.DataFrame(
        {
            "used_genes": [
                ["A", "B", "C"],
                ["D", "E", "F"],
                ["G", "H", "I"],
            ]
        },
        index=[1, 2, 3],
    )
    return pra, per_module, terms


def _install_analysis_storage(monkeypatch):
    pra, per_module, terms = _analysis_inputs()
    storage = {
        ("pra", "dataset"): pra,
        ("pra_per_module", "dataset"): per_module,
        ("common", "terms_dataset"): terms,
        ("input", "sorting"): {"dataset": "high"},
    }

    def fake_dload(category, name=None):
        if name is None:
            return {
                stored_name: value
                for (stored_category, stored_name), value in storage.items()
                if stored_category == category
            }
        return storage.get((category, name))

    def fake_dsave(value, category, name=None):
        storage[(category, name)] = value

    monkeypatch.setattr(analysis, "dload", fake_dload)
    monkeypatch.setattr(analysis, "dsave", fake_dsave)
    return storage


def test_mpr_prepare_computes_only_unfiltered_module_coverage(monkeypatch):
    storage = _install_analysis_storage(monkeypatch)

    result = analysis.mpr_prepare("dataset")

    assert set(result) == {
        "precision_cutoffs",
        "coverage_curve",
        "modules_auc",
        "parameters",
    }
    assert "coverage_curves" not in result
    assert "tp_curves" not in result
    assert storage[("mpr_modules_auc", "dataset")] == result["modules_auc"]


def test_filter_analyses_are_lazy_and_use_three_clear_variant_names(monkeypatch):
    storage = _install_analysis_storage(monkeypatch)
    base = analysis.mpr_prepare("dataset")

    mpr_result = analysis.mpr_filter("dataset")
    global_result = analysis.globalpr_filter("dataset")

    expected = set(analysis.FILTER_VARIANTS)
    assert set(mpr_result["coverage_curves"]) == expected
    assert set(mpr_result["modules_auc"]) == expected
    assert set(global_result["curves"]) == expected
    assert mpr_result["coverage_curves"]["all_complexes"] is base["coverage_curve"]
    assert storage[("mpr_filter", "dataset")] is mpr_result
    assert storage[("globalpr_filter", "dataset")] is global_result


def test_filter_plot_requires_explicit_calculation(monkeypatch):
    def fake_dload(category, name=None):
        if category == "config":
            return {
                "output_folder": ".",
                "plotting": {"save_plot": False, "show_plot": False},
            }
        return {}

    monkeypatch.setattr(plotting, "dload", fake_dload)

    with pytest.raises(RuntimeError, match=r"Run mpr_filter\(name\)"):
        plotting.plot_mpr_filter(save=False)
    with pytest.raises(RuntimeError, match=r"Run globalpr_filter\(name\)"):
        plotting.plot_globalpr_filter(save=False)


def test_new_filter_plot_functions_render_prepared_results(monkeypatch):
    storage = _install_analysis_storage(monkeypatch)
    analysis.mpr_prepare("dataset")
    analysis.mpr_filter("dataset")
    analysis.globalpr_filter("dataset")

    config = {
        "output_folder": ".",
        "color_map": "tab10",
        "plotting": {"save_plot": False, "show_plot": False, "output_type": "png"},
    }

    def fake_plot_dload(category, name=None):
        if category == "config":
            return config
        if category == "input" and name == "colors":
            return {}
        if name is None:
            return {
                stored_name: value
                for (stored_category, stored_name), value in storage.items()
                if stored_category == category
            }
        return storage.get((category, name))

    monkeypatch.setattr(plotting, "dload", fake_plot_dload)

    mpr_ax, auc_values = plotting.plot_mpr_filter(save=False)
    globalpr_ax = plotting.plot_globalpr_filter(save=False)
    base_ax = plotting.plot_mpr_module_coverage_curve(save=False)

    assert list(auc_values.columns) == list(analysis.FILTER_VARIANTS)
    assert list(auc_values.index) == ["dataset"]
    assert mpr_ax.get_ylabel() == "Precision"
    assert globalpr_ax.get_xlabel() == "Number of true positives"
    assert base_ax.get_xlabel() == "# modules"


def test_summary_contains_only_mpr_coverage_and_auc(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plotting,
        "plot_mpr_module_coverage_curve",
        lambda **kwargs: calls.append("coverage"),
    )
    monkeypatch.setattr(
        plotting,
        "plot_mpr_module_auc_scores",
        lambda **kwargs: calls.append("auc") or pd.Series({"dataset": 0.5}),
    )

    result = plotting.plot_mpr_summary(save=False)

    assert calls == ["coverage", "auc"]
    assert result.loc["dataset"] == 0.5
    assert not hasattr(pflex, "plot_mpr_true_positive_curve")
    assert not hasattr(pflex, "plot_mpr_tp")
