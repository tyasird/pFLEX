import numpy as np
import pandas as pd
import pytest

import pflex as flex

from conftest import make_dataset


@pytest.mark.parametrize("method", ["numpy", "numpy_without_mask", "pandas", "numba"])
def test_correlation_methods_match_numpy(method):
    data = make_dataset(seed=1)
    expected = np.corrcoef(data.values)
    np.fill_diagonal(expected, np.nan)

    corr = flex.perform_corr(data, method)

    assert corr.shape == (len(data), len(data))
    assert list(corr.index) == list(data.index)
    np.testing.assert_allclose(corr.values, expected, atol=1e-10)


@pytest.mark.parametrize("method", ["numba", "pandas"])
def test_pairwise_complete_correlation_with_missing_values(method):
    data = make_dataset(seed=2)
    data.iloc[0, :5] = np.nan

    corr = flex.perform_corr(data, method)
    expected = data.T.corr()  # Pearson on pairwise complete observations

    off_diagonal = ~np.eye(len(data), dtype=bool)
    np.testing.assert_allclose(
        corr.values[off_diagonal], expected.values[off_diagonal], atol=1e-10
    )


def test_numpy_correlation_with_missing_values_is_finite_and_symmetric():
    data = make_dataset(seed=2)
    data.iloc[0, :5] = np.nan

    corr = flex.perform_corr(data, "numpy")

    off_diagonal = ~np.eye(len(data), dtype=bool)
    assert np.isfinite(corr.values[off_diagonal]).all()
    assert flex.is_symmetric(corr)


def test_unknown_correlation_method_raises():
    with pytest.raises(ValueError):
        flex.perform_corr(make_dataset(), "spearman")


def test_binary_returns_each_gene_pair_once():
    data = make_dataset()
    corr = flex.perform_corr(data, "numpy")

    pairs = flex.binary(corr)

    n = len(data)
    assert len(pairs) == n * (n - 1) // 2
    pair_keys = {frozenset(p) for p in zip(pairs["gene1"], pairs["gene2"])}
    assert len(pair_keys) == len(pairs)
    assert not (pairs["gene1"] == pairs["gene2"]).any()


def test_quick_sort_orders_scores():
    df = pd.DataFrame({"score": [0.2, 0.9, -0.5, 0.4]})

    assert list(flex.quick_sort(df)["score"]) == [0.9, 0.4, 0.2, -0.5]
    assert list(flex.quick_sort(df, ascending=True)["score"]) == [-0.5, 0.2, 0.4, 0.9]
