"""Half-max AUC: plain average precision over a shared TP window, linear TP axis
(docs/half-max-auc-spec.md)."""

import math
import os

import numpy as np
import pytest
from sklearn import metrics

import pflex as flex
from pflex.analysis import half_max_auc

from conftest import COMPLEXES, make_dataset, run_pipeline

trapz = getattr(np, "trapezoid", None) or np.trapz


def labels_from_ranks(tp_ranks, n_pairs):
    """Binary label vector in rank order with positives at the given 1-based ranks."""
    labels = np.zeros(n_pairs, dtype=int)
    labels[np.asarray(tp_ranks) - 1] = 1
    return labels


def curve_from_precision(precision_of_k, n_tp=1000):
    """TP ranks giving roughly the requested precision for the k-th TP."""
    ranks, last = [], 0
    for k in range(1, n_tp + 1):
        last = max(math.ceil(k / precision_of_k(k)), last + 1)
        ranks.append(last)
    return ranks


def auprc(labels):
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return metrics.auc(tp / tp[-1], precision)


def shared_scores(*labels_list, k0=10, fraction=0.5):
    """What half_max_auc_scores does, on label vectors: one threshold, one window."""
    own = [half_max_auc(lab, k0) for lab in labels_list]
    best = max(own, key=lambda r: r["p_max"])
    threshold = best["p_bg"] + fraction * (best["p_max"] - best["p_bg"])
    k_end = max(half_max_auc(lab, k0, threshold=threshold)["k_cross"] for lab in labels_list)
    return [half_max_auc(lab, k0, threshold=threshold, k_end=k_end) for lab in labels_list]


def test_hand_computed_own_window():
    # p(1)=1, p(2)=1, p(3)=3/10; P=3, N=100 -> bg=0.03, threshold=0.515.
    res = half_max_auc(labels_from_ranks([1, 2, 10], 100), k0=1, max_window=1)

    k_cut = 2 + (1 - 0.515) / (1 - 0.3)          # linear interpolation between TP 2 and 3
    area = 1.0 * 1 + (k_cut - 2) * (1.0 + 0.515) / 2   # trapezoid on the linear TP axis
    mean = area / (k_cut - 1)

    assert res["p_max"] == 1.0
    assert res["p_bg"] == pytest.approx(0.03)
    assert res["p_threshold"] == pytest.approx(0.515)
    assert res["k_cross"] == res["k_cut"] == pytest.approx(k_cut)
    assert res["area"] == pytest.approx(area)
    assert res["mean_precision"] == pytest.approx(mean)
    assert res["score"] == pytest.approx((mean - 0.03) / (1 - 0.03))


def test_window_end_given_includes_the_part_below_threshold():
    # Same ranking, window to TP 3: the low last point is averaged in.
    labels = labels_from_ranks([1, 2, 10], 100)
    res = half_max_auc(labels, k0=1, threshold=0.515, k_end=3, max_window=1)

    area = 1.0 + (1.0 + 0.3) / 2
    assert res["k_cut"] == 3
    assert res["area"] == pytest.approx(area)
    assert res["mean_precision"] == pytest.approx(area / 2)


def test_every_tp_has_the_same_weight():
    """Linear axis: mean precision is the trapezoid average over TP 1..K, no 1/k weighting."""
    labels = labels_from_ranks(curve_from_precision(lambda k: 0.9 if k <= 50 else 0.4), 1_000_000)
    res = half_max_auc(labels, threshold=0.5, k_end=400)

    k = np.arange(1, 401)
    p = k / (np.flatnonzero(labels)[:400] + 1)  # k-th TP at rank r: precision k / r
    assert res["mean_precision"] == pytest.approx(trapz(p, k) / 399)


def test_perfect_ranking_scores_one():
    res = half_max_auc(labels_from_ranks(range(1, 101), 1000))

    assert res["k_cut"] == 100
    assert res["mean_precision"] == pytest.approx(1.0)
    assert res["score"] == pytest.approx(1.0)


def test_random_level_ranking_scores_zero():
    # Precision exactly at background (every 10th pair is a TP, P/N = 0.1).
    labels = labels_from_ranks(range(10, 1001, 10), 1000)
    res = half_max_auc(labels, threshold=0.5, k_end=100)

    assert res["p_bg"] == pytest.approx(0.1)
    assert res["score"] == pytest.approx(0.0, abs=1e-9)


def test_first_crossing_ignores_a_recovery():
    # crossing="first" (pFLEX 2.0): drops below at k=3, recovery to 20/27 is ignored.
    ranks = [1, 2, 10] + list(range(11, 28))
    res = half_max_auc(labels_from_ranks(ranks, 100), k0=1, max_window=1, crossing="first")

    assert res["p_threshold"] == pytest.approx(0.6)
    assert 2 < res["k_cross"] < 3


def test_last_crossing_takes_the_largest_tp_above_threshold():
    # Same ranking: falls below at k=3 (3/10), recovers to 20/27; never falls below again.
    ranks = [1, 2, 10] + list(range(11, 28))
    res = half_max_auc(labels_from_ranks(ranks, 100), k0=1, crossing="last")

    assert res["p_threshold"] == pytest.approx(0.6)
    assert res["k_cross"] == 20  # still above at the last TP
    assert res["crossing"] == "last"


def test_last_crossing_curve_that_starts_low_is_not_cut_at_its_start():
    # p: 1/5, 2/6, 3/7, 4/8, 5/9, ... rises above 0.5, then falls below for good.
    ranks = list(range(5, 45)) + [45 + 10 * i for i in range(1, 40)]
    labels = labels_from_ranks(ranks, 2000)
    k = np.arange(1, len(ranks) + 1)
    p = k / np.array(ranks)
    j = int(np.flatnonzero(p >= 0.5)[-1])  # last TP at or above
    expected = k[j] + (p[j] - 0.5) / (p[j] - p[j + 1])

    first = half_max_auc(labels, threshold=0.5, k0=1, crossing="first")
    last = half_max_auc(labels, threshold=0.5, k0=1)  # default: last crossing

    assert first["k_cross"] == 1  # starts below: the first-crossing rule cuts at TP 1
    assert last["k_cross"] == pytest.approx(expected)
    assert last["k_cross"] > 30


def test_last_crossing_never_above_is_first_tp():
    res = half_max_auc(labels_from_ranks(range(10, 1001, 10), 1000), threshold=0.5)

    assert res["k_cross"] == 10  # the first TP searched (k0=10)


def test_invalid_crossing():
    with pytest.raises(ValueError):
        half_max_auc(labels_from_ranks([1, 2, 10], 100), crossing="middle")


def test_max_starts_at_k0():
    # A precision-1 head below k0 must not set the max.
    ranks = list(range(1, 6)) + [10 * k for k in range(6, 40)]
    res = half_max_auc(labels_from_ranks(ranks, 1000), k0=10)

    assert res["k_max"] >= 10
    assert res["p_max"] == pytest.approx(0.1, abs=0.06)


def test_max_ignores_a_single_spike():
    # 10th TP at rank 12 (0.83), then precision falls: 11/20, 12/28, 13/36, ...
    ranks = list(range(1, 10)) + [12] + [12 + 8 * (k - 10) for k in range(11, 40)]
    labels = labels_from_ranks(ranks, 1000)

    assert half_max_auc(labels, k0=10)["p_max"] == pytest.approx(10 / 12)  # default: plain max
    # Option: highest precision held for 3 consecutive TPs: min(10/12, 11/20, 12/28).
    res = half_max_auc(labels, k0=10, max_window=3)
    assert res["p_max"] == pytest.approx(12 / 28)
    assert res["k_max"] == 10


def test_defaults_max_from_tenth_tp_and_last_crossing():
    # Precision 1 at TP 1-9 (ranks 1-9), lower afterwards: the default max skips them.
    ranks = list(range(1, 10)) + [12] + [12 + 8 * (k - 10) for k in range(11, 40)]
    labels = labels_from_ranks(ranks, 1000)
    res = half_max_auc(labels)

    assert res["k0"] == 10 and res["max_window"] == 1 and res["crossing"] == "last"
    assert res["p_max"] == pytest.approx(10 / 12)
    assert res["k_max"] == 10
    # k0=1 (pFLEX 2.0): the lucky head sets the max.
    assert half_max_auc(labels, k0=1)["p_max"] == 1.0


def test_high_head_beats_long_mediocre_tail():
    """AUPRC rewards a long moderate tail; the score only looks at the window up to K."""
    n_pairs = 1_000_000
    head = labels_from_ranks(curve_from_precision(lambda k: 0.9 if k <= 200 else 0.01, 2000), n_pairs)
    tail = labels_from_ranks(curve_from_precision(lambda k: 0.6 if k <= 150 else 0.3, 2000), n_pairs)

    res_head, res_tail = shared_scores(head, tail)

    assert auprc(tail) > auprc(head)
    assert res_head["k_cut"] == res_tail["k_cut"] == pytest.approx(res_head["k_cross"])
    assert res_head["score"] > res_tail["score"]


def test_dominated_curve_never_scores_higher():
    """Same curve plus a short early peak must not score lower."""
    n_pairs = 100_000
    peaked = labels_from_ranks(curve_from_precision(lambda k: 0.95 if k <= 15 else 0.45), n_pairs)
    flat = labels_from_ranks(curve_from_precision(lambda k: 0.45), n_pairs)

    res_peaked, res_flat = shared_scores(peaked, flat)

    assert res_peaked["score"] >= res_flat["score"]


def test_curve_below_threshold_from_start():
    flat = labels_from_ranks(curve_from_precision(lambda k: 0.3), 100_000)

    own = half_max_auc(flat, threshold=0.5, k0=1)
    from_10 = half_max_auc(flat, threshold=0.5)  # default k0=10
    shared = half_max_auc(flat, threshold=0.5, k_end=500)

    # Never above: crossing at the first TP searched, TP 1 with k0=1 (no window of its own), TP 10 by default.
    assert own["k_cross"] == 1 and math.isnan(own["score"]) and own["reason"]
    assert from_10["k_cross"] == from_10["k_cut"] == 10
    assert np.isfinite(from_10["score"])
    # On a shared window: mean precision ~0.3 above background.
    assert 0 < shared["score"] < 0.3


def test_area_starts_at_first_tp():
    """The first TPs count in the area, although max and crossing start at k0."""
    # From TP 10 on both curves are the same: p(k) = k / (k + 9).
    tail = [k + 9 for k in range(10, 101)]
    good_head = labels_from_ranks(list(range(1, 10)) + tail, 1000)             # TP 1-9 perfect
    poor_head = labels_from_ranks([2 * k for k in range(1, 10)] + tail, 1000)  # TP 1-9 at 0.5

    good = half_max_auc(good_head, threshold=0.5, k_end=100, k0=10)
    poor = half_max_auc(poor_head, threshold=0.5, k_end=100, k0=10)

    k = np.arange(1, 101)
    p = k / np.array(list(range(1, 10)) + tail)
    assert good["area"] == pytest.approx(trapz(p, k))
    assert good["mean_precision"] == pytest.approx(trapz(p, k) / 99)  # width K - 1
    assert good["p_max"] == poor["p_max"] and good["k_cross"] == poor["k_cross"]
    assert good["score"] > poor["score"]


def test_too_few_tps_is_nan():
    res = half_max_auc(labels_from_ranks(range(1, 6), 100), k0=10)

    assert math.isnan(res["score"])
    assert res["reason"]


def test_no_signal_above_background_is_nan():
    res = half_max_auc(np.ones(50, dtype=int))

    assert math.isnan(res["score"])
    assert res["reason"]


@pytest.mark.parametrize("fraction, value", [("three_quarter", 3 / 4), ("half", 1 / 2), ("quarter", 1 / 4),
                                             (0.75, 0.75), (0.5, 0.5), (0.25, 0.25), (1 / 3, 1 / 3)])
def test_threshold_fraction(fraction, value):
    labels = labels_from_ranks(curve_from_precision(lambda k: 0.9 if k <= 100 else 0.05), 100_000)

    res = half_max_auc(labels, fraction=fraction)

    assert res["fraction"] == pytest.approx(value)
    assert res["p_threshold"] == pytest.approx(res["p_bg"] + value * (res["p_max"] - res["p_bg"]))


def test_lower_fraction_cuts_later():
    labels = labels_from_ranks(curve_from_precision(lambda k: 0.9 * 100 / max(k, 100)), 100_000)

    cuts = [half_max_auc(labels, fraction=f)["k_cross"] for f in ("three_quarter", "half", "quarter")]

    assert cuts[0] < cuts[1] < cuts[2]


@pytest.mark.parametrize("fraction", ["third", "tenth", 0, 1, 1.5])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        half_max_auc(np.ones(50, dtype=int), fraction=fraction)


def test_scores_from_cached_pra(config):
    run_pipeline(config, {"perfect": {"path": make_dataset(seed=0), "sort": "high"}})
    n_pos = sum(len(g) * (len(g) - 1) // 2 for g in COMPLEXES.values())

    res = flex.half_max_auc_scores()["perfect"]

    assert res["n_tp"] == n_pos
    assert res["k_cut"] == n_pos
    assert res["area"] == pytest.approx(n_pos - 1)  # precision 1 from TP 1 to TP n_pos
    assert res["score"] == pytest.approx(1.0)  # perfect ranking


def test_all_datasets_share_one_threshold_and_one_window(config):
    inputs = {
        "clean": {"path": make_dataset(seed=0), "sort": "high"},
        "noisy": {"path": make_dataset(seed=5, signal=0.3), "sort": "high"},
    }
    run_pipeline(config, inputs)

    res = flex.half_max_auc_scores()
    best = max(res, key=lambda n: res[n]["p_max"])
    latest = max(res, key=lambda n: res[n]["k_cross"])

    expected = res[best]["p_bg"] + (res[best]["p_max"] - res[best]["p_bg"]) / 2
    assert res["clean"]["p_threshold"] == res["noisy"]["p_threshold"] == pytest.approx(expected)
    assert res["clean"]["threshold_from"] == res["noisy"]["threshold_from"] == best
    assert res["clean"]["k_cut"] == res["noisy"]["k_cut"] == res[latest]["k_cross"]
    assert res["clean"]["cut_from"] == res["noisy"]["cut_from"] == latest


def test_pr_curve_half_max_overlay(config):
    run_pipeline(config, {"perfect": {"path": make_dataset(seed=0), "sort": "high"}})
    hm = flex.half_max_auc_scores()["perfect"]

    ax = flex.plot_precision_recall_curve(half_max=True)

    # Log-scaled plot: lines only, no shaded area (the score is not an area on this axis).
    assert not ax.collections
    threshold_lines = [line for line in ax.get_lines()
                       if list(line.get_ydata()) == pytest.approx([hm["p_threshold"]] * 2)]
    assert len(threshold_lines) == 1
    cut_lines = [line for line in ax.get_lines()
                 if list(line.get_xdata()) == pytest.approx([hm["k_cut"], hm["k_cut"]])]
    assert len(cut_lines) == 1


def test_half_max_auc_bar_plot(config):
    inputs = {
        "Network A": {"path": make_dataset(seed=0), "sort": "high"},
        "Network B": {"path": make_dataset(seed=5, signal=0.5), "sort": "high"},
    }
    run_pipeline(config, inputs)

    scores = flex.plot_half_max_auc_scores()

    assert set(scores) == {"Network_A", "Network_B"}  # results use sanitized names
    assert scores["Network_A"] == flex.dload("half_max_auc", "Network_A")["score"]


def test_bar_plot_unsorted_keeps_dataset_order(config):
    # Weaker network first by name: sorted bars would put it second.
    inputs = {
        "SVD_10": {"path": make_dataset(seed=5, signal=0.3), "sort": "high"},
        "SVD_2": {"path": make_dataset(seed=0), "sort": "high"},
    }
    run_pipeline(config, inputs)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    flex.plot_half_max_auc_scores(sort=False, crossing="last", ax=ax)
    names = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)

    assert names == ["SVD_2", "SVD_10"]  # natural order, not by score


def test_explainer_with_last_crossing(config):
    inputs = {
        "clean": {"path": make_dataset(seed=0), "sort": "high"},
        "noisy": {"path": make_dataset(seed=5, signal=0.3), "sort": "high"},
    }
    run_pipeline(config, inputs)

    res = flex.plot_half_max_explainer(k0=10, crossing="last", sort=False)

    assert all(r["crossing"] == "last" and r["k0"] == 10 for r in res.values())


def test_fraction_in_scores_and_plots(config):
    inputs = {
        "clean": {"path": make_dataset(seed=0), "sort": "high"},
        "noisy": {"path": make_dataset(seed=5, signal=0.3), "sort": "high"},
    }
    run_pipeline(config, inputs)

    half = flex.half_max_auc_scores()
    quarter = flex.half_max_auc_scores(fraction="quarter")
    three_quarter = flex.half_max_auc_scores(fraction="three_quarter")
    assert quarter["clean"]["p_threshold"] < half["clean"]["p_threshold"] < three_quarter["clean"]["p_threshold"]

    flex.plot_half_max_auc_scores(fraction="three_quarter")
    res = flex.plot_half_max_explainer()
    flex.plot_half_max_explainer(fraction="quarter")
    files = os.listdir(config["output_folder"])
    assert set(res) == {"clean", "noisy"}
    assert "three_quarter_max_auc_values.png" in files
    assert "half_max_explainer.png" in files
    assert "quarter_max_explainer.png" in files


def test_max_rank_takes_nth_highest_precision():
    # TP 1-2 lucky hits (1.0), then the curve settles around 0.8.
    ranks = [1, 2] + curve_from_precision(lambda k: 0.8, 200)[2:]
    labels = labels_from_ranks(ranks, 10_000)
    k = np.arange(1, 201)
    p = k / (np.array(ranks) + 0)

    plain = half_max_auc(labels, k0=1)
    third = half_max_auc(labels, k0=1, max_rank=3)

    assert plain["p_max"] == 1.0
    assert third["p_max"] == pytest.approx(np.sort(p)[-3])
    assert third["p_max"] < 1.0 and third["max_rank"] == 3


def test_max_rank_keeps_one_when_precision_stays_one():
    res = half_max_auc(labels_from_ranks(range(1, 101), 1000), max_rank=3)

    assert res["p_max"] == 1.0
