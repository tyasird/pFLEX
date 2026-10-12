"""Which modules the per-module AUPRC scatter highlights."""

import pandas as pd
import pytest

from pflex.plotting import _select_scatter_modules

# x and y AUPRC of six modules.
SCORES = pd.DataFrame(
    {
        "x": [0.95, 0.90, 0.80, 0.10, 0.50, 0.85],
        "y": [0.10, 0.85, 0.20, 0.70, 0.45, 0.80],
    },
    index=["x_far", "both_best", "x_mid", "y_far", "diagonal", "both_second"],
)


def test_delta_picks_largest_difference_in_each_direction():
    x_only, y_only, shared = _select_scatter_modules(SCORES, "delta", 2, 0)

    assert list(x_only) == ["x_far", "x_mid"]
    assert list(y_only) == ["y_far"]  # only one module is better in y
    assert len(shared) == 0


def test_rank_picks_highest_auprc_per_dataset():
    x_only, y_only, shared = _select_scatter_modules(SCORES, "rank", 2, 0)

    assert list(x_only) == ["x_far", "both_best"]
    assert list(y_only) == ["y_far"]
    assert len(shared) == 0


@pytest.mark.parametrize("highlight", ["delta", "rank"])
def test_shared_are_good_in_both_and_not_reused(highlight):
    x_only, y_only, shared = _select_scatter_modules(SCORES, highlight, 3, 2)

    assert list(shared) == ["both_best", "both_second"]
    groups = [set(x_only), set(y_only), set(shared)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
