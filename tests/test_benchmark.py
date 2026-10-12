"""End-to-end benchmark on a synthetic network with a known answer."""

import numpy as np
import pandas as pd
import pytest

import pflex as flex

from conftest import COMPLEXES, make_dataset, run_pipeline

N_POSITIVE_PAIRS = sum(len(g) * (len(g) - 1) // 2 for g in COMPLEXES.values())


@pytest.fixture
def perfect_run(config):
    run_pipeline(config, {"perfect": {"path": make_dataset(seed=0), "sort": "high"}})
    return config


def test_global_pr_of_perfect_network(perfect_run):
    pra = flex.dload("pra", "perfect")

    # Only genes in the standard are kept: 12 genes -> 66 pairs, 18 positives.
    assert len(pra) == 12 * 11 // 2
    assert pra["prediction"].sum() == N_POSITIVE_PAIRS
    # Every within-complex pair ranks above every other pair.
    assert pra["prediction"].iloc[:N_POSITIVE_PAIRS].all()
    assert pra["precision"].iloc[N_POSITIVE_PAIRS - 1] == 1.0
    # The trapezoid area starts at the first TP (recall 1/18), as in FLEX/R.
    assert flex.dload("pr_auc", "perfect") == pytest.approx(1 - 1 / N_POSITIVE_PAIRS)


def test_global_pr_of_random_network_is_near_prevalence(config):
    rng = np.random.default_rng(3)
    genes = [g for genes in COMPLEXES.values() for g in genes]
    random_data = pd.DataFrame(rng.normal(size=(len(genes), 40)), index=genes)

    run_pipeline(config, {"random": {"path": random_data, "sort": "high"}})

    auprc = flex.dload("pr_auc", "random")
    perfect_auprc = 1.0
    prevalence = N_POSITIVE_PAIRS / (len(genes) * (len(genes) - 1) / 2)
    assert auprc < perfect_auprc - 0.3
    assert auprc == pytest.approx(prevalence, abs=0.25)


def test_per_module_auprc_of_perfect_network(perfect_run):
    per_module = flex.dload("pra_per_module", "perfect")

    assert len(per_module) == len(COMPLEXES)
    assert set(per_module.index) == set(COMPLEXES)
    # Six pairs per complex, all ranked first: area from recall 1/6 to 1.
    assert per_module["auc_score"].to_numpy() == pytest.approx([5 / 6] * len(COMPLEXES))


def test_contribution_diversity_assigns_each_tp_pair_once(perfect_run):
    contributions = flex.dload("module_contributions", "perfect")

    assert isinstance(contributions, pd.DataFrame)
    # At the lowest precision cutoff all 18 TP pairs are assigned, 6 per complex.
    first_cutoff = contributions.columns[1]  # column 0 is the module name
    assert contributions[first_cutoff].sum() == N_POSITIVE_PAIRS
    assert set(contributions[first_cutoff]) == {6}


def test_mpr_counts_all_complexes_as_recovered(perfect_run):
    mpr = flex.dload("mpr", "perfect")

    coverage = np.asarray(mpr["coverage_curve"])
    assert coverage.max() == len(COMPLEXES)
    # The curve is monotone: lower precision never recovers fewer modules.
    assert np.all(np.diff(coverage) <= 0)


def test_mpr_coverage_rule():
    """A module is recovered with >= 1 TP pair covering more than 10% of its pairs."""
    from pflex.analysis import _mpr_module_coverage

    # Module 1: 5 genes -> 10 pairs. Module 2: 10 genes -> 45 pairs.
    terms = pd.DataFrame(
        {"used_genes": [[f"a{i}" for i in range(5)], [f"b{i}" for i in range(10)]]},
        index=[1, 2],
    )
    contributions = pd.DataFrame(
        {0.9: [1, 4], 0.5: [2, 5]},  # TP pairs per module at two precision cutoffs
        index=[1, 2],
    )

    coverage = _mpr_module_coverage(contributions, terms)

    # 0.9: module 1 has 1/10 (not > 10%), module 2 has 4/45 -> 0 recovered.
    # 0.5: module 1 has 2/10 (20%), module 2 has 5/45 (11%) -> 2 recovered.
    assert list(coverage) == [0, 2]


def test_two_networks_and_plots(config):
    inputs = {
        "Network A": {"path": make_dataset(seed=0), "sort": "high", "color": "#4E79A7"},
        "Network B": {"path": make_dataset(seed=5, signal=0.03), "sort": "high", "color": "#F28E2B"},
    }
    run_pipeline(config, inputs)

    flex.plot_precision_recall_curve()
    flex.plot_auc_scores()
    flex.plot_per_module_scatter(n_top=2)
    flex.plot_module_contributions()
    flex.plot_mpr_summary()
    flex.save_results_to_csv()

    output = config["output_folder"]
    import os

    files = os.listdir(output)
    assert any(f.endswith(".png") for f in files)
    assert any(f.endswith(".csv") for f in files) or any(
        os.path.isdir(os.path.join(output, f)) for f in files
    )
    assert flex.dload("pr_auc", "Network_A") > flex.dload("pr_auc", "Network_B")
