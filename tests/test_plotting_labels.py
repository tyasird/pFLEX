import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pflex import plotting


def _fake_per_module(scores, names):
    index = [f"module_{i}" for i in range(len(scores))]
    return pd.DataFrame(
        {
            "Name": names,
            "auc_score": scores,
            "n_used_genes": [8 + i for i in range(len(scores))],
            "Genes": ["A,B,C"] * len(scores),
            "Length": [3] * len(scores),
            "used_genes": [["A", "B", "C"]] * len(scores),
        },
        index=index,
    )


def _install_common_plot_mocks(monkeypatch, rdict):
    def fake_dload(category, name=None):
        if category == "config":
            return {
                "output_folder": ".",
                "plotting": {
                    "save_plot": False,
                    "show_plot": False,
                    "output_type": "png",
                },
            }
        if category == "pra_per_module":
            return rdict
        if category == "input" and name == "colors":
            return {"Dataset A": "#1f77b4", "Dataset B": "#ff7f0e"}
        return {}

    monkeypatch.setattr(plotting, "dload", fake_dload)
    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    monkeypatch.setattr(plotting.plt, "close", lambda fig=None: None)


def _label_positions(ax):
    return [text.get_position() for text in ax.texts]


def _assert_labels_spread_horizontally(ax, min_unique_x):
    positions = _label_positions(ax)
    assert positions
    unique_x = {round(x, 3) for x, _ in positions}
    assert len(unique_x) >= min_unique_x

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for x, y in positions:
        assert x0 <= x <= x1
        assert y0 <= y <= y1


def test_plot_per_module_scatter_spreads_labels_when_points_share_x(monkeypatch):
    plt.close("all")
    names = [f"Shared x label {i}" for i in range(8)]
    shared_x_scores = [0.05] * len(names)
    varied_y_scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    _install_common_plot_mocks(
        monkeypatch,
        {
            "Dataset A": _fake_per_module(shared_x_scores, names),
            "Dataset B": _fake_per_module(varied_y_scores, names),
        },
    )

    plotting.plot_per_module_scatter(n_top=8)

    _assert_labels_spread_horizontally(plt.gcf().axes[0], min_unique_x=2)
    plt.close("all")


def test_plot_per_module_scatter_by_size_spreads_dense_label_columns(monkeypatch):
    plt.close("all")
    names = [f"Size plot label {i}" for i in range(10)]
    _install_common_plot_mocks(
        monkeypatch,
        {"Dataset A": _fake_per_module([0.05] * len(names), names)},
    )

    plotting.plot_per_module_scatter_by_size(n_labels=10)

    _assert_labels_spread_horizontally(plt.gcf().axes[0], min_unique_x=2)
    plt.close("all")


def test_shared_x_label_seed_positions_move_to_callout_columns(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(plotting, "adjust_text", lambda *args, **kwargs: None)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label_points = [
        (0.05, 0.95 - i * 0.02, f"Dense label {i}")
        for i in range(30)
    ]

    texts = plotting._place_scatter_labels(ax, label_points)

    distances = {
        round(abs(text.get_position()[0] - label_points[i][0]), 3)
        for i, text in enumerate(texts)
    }
    assert 2 <= len(distances) <= 4
    assert min(distances) >= 0.1
    plt.close("all")


def test_dense_shared_x_labels_use_non_overlapping_callout_columns(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(plotting, "adjust_text", lambda *args, **kwargs: None)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label_points = [
        (0.05, 0.95 - i * 0.02, f"Dense label {i}")
        for i in range(30)
    ]

    texts = plotting._place_scatter_labels(ax, label_points)

    rounded_x = [round(text.get_position()[0], 2) for text in texts]
    unique_x = sorted(set(rounded_x))
    assert 2 <= len(unique_x) <= 4

    for x in unique_x:
        ys = sorted(
            text.get_position()[1]
            for text in texts
            if round(text.get_position()[0], 2) == x
        )
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        assert all(gap >= 0.035 for gap in gaps)

    plt.close("all")


def test_plot_functions_use_thin_default_connector_lines(monkeypatch):
    captured = []

    def fake_helper(ax, label_points, **kwargs):
        captured.append(kwargs["connector_linewidth"])
        return []

    _install_common_plot_mocks(
        monkeypatch,
        {
            "Dataset A": _fake_per_module([0.05] * 8, [f"Label {i}" for i in range(8)]),
            "Dataset B": _fake_per_module([0.95 - i * 0.05 for i in range(8)], [f"Label {i}" for i in range(8)]),
        },
    )
    monkeypatch.setattr(plotting, "_place_scatter_labels", fake_helper)

    plotting.plot_per_module_scatter(n_top=8)
    plotting.plot_per_module_scatter_by_size(n_labels=8)

    assert captured == [0.2, 0.2, 0.2]


def test_external_mpr_legends_expand_canvas_and_do_not_overlap():
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(right=0.7)
    original_axes_width = ax.get_position().width * fig.get_figwidth()
    dataset_names = [
        f"Dataset with a deliberately long descriptive name {i}"
        for i in range(12)
    ]
    colors = [f"C{i % 10}" for i in range(len(dataset_names))]

    legend1, legend2 = plotting._add_vertical_legend(
        ax,
        dataset_names,
        colors,
        plotting.FILTER_VARIANTS,
        linewidth=1.8,
        fit_figure=True,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_boxes = [
        legend1.get_window_extent(renderer),
        legend2.get_window_extent(renderer),
    ]

    assert fig.get_figwidth() > 6
    assert ax.get_position().width * fig.get_figwidth() == pytest.approx(
        original_axes_width
    )
    assert max(box.x1 for box in legend_boxes) <= fig.bbox.x1
    assert min(box.y0 for box in legend_boxes) >= fig.bbox.y0
    assert legend2.get_window_extent(renderer).y1 < legend1.get_window_extent(renderer).y0
    plt.close("all")
