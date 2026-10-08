# Standard library
import functools
from itertools import combinations
from pathlib import Path
import re

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter, NullLocator

mpl.rcParams['text.usetex'] = False

# Local modules
from .utils import dload, _sanitize
from .logging_config import log
from .style import PUBLICATION_RC


def _bar_figsize(n_bars):
    """Bar chart plot area: narrow bars, width grows with the bar count."""
    return (0.2 + 0.4 * n_bars, 2.0)


def _value_bars(ax, names, values, colors, fmt="%.3f"):
    """Narrow bars with their value printed on top."""
    bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.6)
    ax.bar_label(bars, fmt=fmt, padding=1.5)
    finite = [v for v in values if np.isfinite(v)]
    ax.set_ylim(0, (max(finite) if finite else 1.0) * 1.15)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Precision-recall and mPR panels use a square plot area. The box aspect is
# physical, so it holds whether the TP axis ends at 10^2, 10^3 or 10^4.
CURVE_BOX_ASPECT = 1.0
CURVE_FIGSIZE = (2.0, 2.0)  # plot area at figure_scale 1
CAPTION_COLOR = "0.4"
CAPTION_SIZE = 6.5


def _caption(ax, text, below=None, x_ax=None):
    """Small grey description under the plot.

    The text hangs below the x axis of ``ax`` (or of the axes / legends in
    ``below``) and is re-positioned at every draw, so it follows layout changes
    in :func:`plot_panels`. ``x_ax`` sets the left edge (default ``ax``).
    """
    if not text or not _STYLE["captions"]:
        return None
    import textwrap
    from matplotlib.transforms import Bbox

    fig = ax.figure
    host = x_ax or ax
    width_pt = host.get_position().width * fig.get_figwidth() * 72
    wrapped = textwrap.fill(text, width=max(20, int(width_pt / (CAPTION_SIZE * 0.5))))
    anchors = below or [ax]

    def bottom(renderer):
        boxes = []
        for item in anchors:
            box = item.xaxis.get_tightbbox(renderer) if hasattr(item, "xaxis") else item.get_window_extent(renderer)
            if box is not None and box.height > 0:
                boxes.append(box)
        return Bbox.union(boxes) if boxes else host.get_window_extent(renderer)

    return host.annotate(
        wrapped, xy=(0, 0), xycoords=(host.transAxes, bottom),
        xytext=(0, -4), textcoords="offset points", ha="left", va="top",
        fontsize=CAPTION_SIZE, color=CAPTION_COLOR, annotation_clip=False,
    )


def _set_title(ax, text, **kwargs):
    """Axes title wrapped onto several lines when it is wider than the axes.

    Skipped unless titles are switched on (config ``plotting.titles`` or an
    explicit ``title=`` argument).
    """
    if not text or not _STYLE["titles"]:
        return ax.set_title("", **kwargs)
    import textwrap

    width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
    max_chars = max(16, int(width_pt / (PUBLICATION_RC["axes.titlesize"] * 0.52)))
    lines = [textwrap.fill(part, max_chars) for part in str(text).splitlines()]
    return ax.set_title("\n".join(lines), **kwargs)


def _artist_points(ax):
    """Display-space points covered by an axes' lines, fills and markers."""
    pts = []
    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy) == 0:
            continue
        xy = ax.transData.transform(xy)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) > 1:  # add points along long segments so a straight line counts too
            seg = np.hypot(*np.diff(xy, axis=0).T)
            extra = [np.linspace(a, b, int(n // 3) + 2) for a, b, n in zip(xy[:-1], xy[1:], seg) if n > 6]
            if extra:
                xy = np.vstack([xy] + extra)
        pts.append(xy)
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets):
            pts.append(coll.get_offset_transform().transform(offsets))
        for path in coll.get_paths():
            verts = coll.get_transform().transform(path.vertices)
            if len(verts) > 2:
                # fill_between polygon: sample its interior on a coarse grid
                from matplotlib.path import Path as MPath
                lo, hi = verts.min(axis=0), verts.max(axis=0)
                gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40))
                grid = np.column_stack([gx.ravel(), gy.ravel()])
                pts.append(grid[MPath(verts).contains_points(grid)])
                pts.append(verts)
    return np.vstack(pts) if pts else np.empty((0, 2))


def _place_legend(ax, handles=None, labels=None, above_ax=None, above_pad=0.5, **kwargs):
    """Legend in an empty corner of ``ax`` if one exists, otherwise above the plot.

    Corners are tried in the order upper right, lower left, upper left, lower right;
    a corner counts as empty when no line, fill or marker falls under the legend.
    """
    args = (handles, labels) if handles is not None else ()
    kwargs.setdefault("handlelength", 1.2)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = _artist_points(ax)
    pad = 2 * fig.dpi / 72
    for loc in ("upper right", "lower left", "upper left", "lower right"):
        legend = ax.legend(*args, loc=loc, **kwargs)
        box = legend.get_window_extent(renderer)
        inside = (
            (points[:, 0] > box.x0 - pad) & (points[:, 0] < box.x1 + pad)
            & (points[:, 1] > box.y0 - pad) & (points[:, 1] < box.y1 + pad)
        )
        if not inside.any():
            return legend
        legend.remove()
    # No free corner: horizontally above the plot, so the panel keeps its width.
    target = above_ax or ax
    legend = target.legend(*args, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                           borderaxespad=above_pad, columnspacing=1.0, **kwargs)
    if target.get_title():
        fig.canvas.draw()
        height_pt = legend.get_window_extent(renderer).height * 72 / fig.dpi
        target.set_title(target.get_title(), pad=height_pt + 8)
    return legend




def _trim_pr_xaxis(ax, min_precision=0.05, x_min=10):
    """End a log TP axis where every curve has dropped below ``min_precision``.

    PR curves trail off near zero precision for a decade or more; that tail
    adds width without information. ``ax`` may be a list of axes sharing x.
    """
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    ax = axes[0]
    x_max = 0.0
    for line in (line for a in axes for line in a.get_lines()):
        x, y = (np.asarray(v, dtype=float) for v in line.get_data())
        keep = np.isfinite(x) & np.isfinite(y) & (y >= min_precision)
        if keep.any():
            x_max = max(x_max, float(x[keep].max()))
    if x_max > x_min:
        # Label the next power of ten and let the axis (and curves) run on a
        # little past it, so the line does not stop exactly on the last tick.
        ax.set_xlim(x_min, 10 ** (np.ceil(np.log10(x_max * 1.05)) + 0.12))


# Settings for the plot call in progress (set by _publication_style).
_STYLE = {"scale": 0.75, "titles": False, "captions": False}


def _publication_style(func):
    """Run a plotting function under PUBLICATION_RC.

    Reads from ``config["plotting"]``:

    - ``dpi``: raster resolution (default 300);
    - ``figure_scale``: plot-area size relative to the design size (a 2-inch
      square for curves and scatters). Default 0.75: about 38 mm plot area,
      three panels across A4; 0.55 fits four. Text stays 8 pt; only the plot
      area changes;
    - ``titles`` / ``captions``: draw the default title / grey caption
      (default False). Passing ``title=`` or ``caption=`` shows it anyway.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rc = dict(PUBLICATION_RC)
        try:
            plot_config = (dload("config") or {}).get("plotting", {})
        except Exception:
            plot_config = {}
        rc["savefig.dpi"] = plot_config.get("dpi", 300)
        previous = dict(_STYLE)
        _STYLE.update(
            scale=float(plot_config.get("figure_scale", 0.75)),
            titles=bool(plot_config.get("titles", False)) or kwargs.get("title") is not None,
            captions=bool(plot_config.get("captions", False)) or kwargs.get("caption") is not None,
        )
        try:
            with plt.rc_context(rc):
                return func(*args, **kwargs)
        finally:
            _STYLE.clear()
            _STYLE.update(previous)
    return wrapper


def _marker_area(genes, point_scale=1.0):
    """Scatter marker area (pt^2) for a gene count.

    A fixed base (a 2.5 pt circle) plus an area proportional to the gene count,
    scaled with figure_scale. Every point stays visible and small modules still
    differ in size (a hard minimum would make 2-10 gene modules look identical).
    """
    area = np.asarray(genes, dtype=float) * 1.25 * point_scale * _STYLE["scale"] ** 2
    return 2.5 ** 2 + area


def _unit_ticks(ax, which="xy"):
    """Mark a 0-1 axis for tick fitting: a tick every 0.1, labels added once the
    axis length is known (see _fit_unit_ticks)."""
    from matplotlib.ticker import MultipleLocator

    for name in which:
        axis = ax.xaxis if name == "x" else ax.yaxis
        axis.set_minor_locator(MultipleLocator(0.1))
    # unlabelled ticks look like the labelled ones
    ax.tick_params(which="minor", length=PUBLICATION_RC["xtick.major.size"],
                   width=PUBLICATION_RC["xtick.major.width"])
    ax._pflex_unit_ticks = which


def _fit_unit_ticks(fig):
    """Label 0-1 axes at the finest step (0.1, 0.2, 0.5, 1) whose labels fit.

    Steps lie on the 0.1 tick grid, and both axes of a plot share the coarser
    of their two steps so a square plot reads the same way on x and y.
    """
    renderer = fig.canvas.get_renderer()
    axes = list(fig.axes) + [c for a in fig.axes for c in getattr(a, "child_axes", [])]
    size_pt = PUBLICATION_RC["xtick.labelsize"]

    def labels_for(step):
        values = np.round(np.arange(0, 1 + 1e-9, step), 2)
        return values, [f"{v:g}" for v in values]

    for ax in axes:
        if getattr(ax, "_pflex_log_ticks", False):
            _fit_log_ticks(ax, renderer, size_pt)
        which = getattr(ax, "_pflex_unit_ticks", "")
        if not which:
            continue
        box = ax.get_window_extent(renderer)
        steps = []
        for name in which:
            length_pt = (box.width if name == "x" else box.height) * 72 / fig.dpi
            for step in (0.1, 0.2, 0.5, 1.0):
                _, labels = labels_for(step)
                # x labels sit side by side (~0.55 em per char); y labels stack (one line each)
                need = (sum(len(t) for t in labels) * 0.55 * size_pt + len(labels) * 3
                        if name == "x" else len(labels) * size_pt * 1.4)
                if need <= length_pt:
                    break
            steps.append(step)
        values, labels = labels_for(max(steps))
        for name in which:
            (ax.set_xticks if name == "x" else ax.set_yticks)(values, labels)


def _fit_log_ticks(ax, renderer, size_pt):
    """Tick every decade on a log x axis; label all of them if they fit,
    otherwise every other one (10^1, 10^3, ...), keeping the unlabelled ticks."""
    lo, hi = ax.get_xlim()
    if lo <= 0 or hi <= lo:
        return
    powers = list(range(int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9)) + 1))
    if not powers:
        return
    length_pt = ax.get_window_extent(renderer).width * ax.figure.dpi ** -1 * 72
    # "10^3" is about 3 characters wide at 8 pt, plus a small gap
    every = 1 if len(powers) * (3 * 0.55 * size_pt + 4) <= length_pt else 2
    ticks = [10.0 ** k for k in powers]
    labels = [f"$10^{{{k}}}$" if i % every == 0 else "" for i, k in enumerate(powers)]
    ax.set_xticks(ticks, labels)


def _wrap_axis_labels(fig):
    """Break axis labels longer than their axis onto two lines (at the middle space).

    Also labels 0-1 axes now that their length is known.
    """
    fig.canvas.draw()
    _fit_unit_ticks(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = list(fig.axes)
    axes += [child for ax in fig.axes for child in getattr(ax, "child_axes", [])]
    for ax in axes:
        if not ax.get_visible() or not ax.axison:
            continue
        box = ax.get_window_extent(renderer)
        for label, length in ((ax.xaxis.label, box.width), (ax.yaxis.label, box.height)):
            text = label.get_text()
            if not text or "\n" in text or " " not in text:
                continue
            extent = label.get_window_extent(renderer)
            if max(extent.width, extent.height) <= length:
                continue
            spaces = [i for i, ch in enumerate(text) if ch == " "]
            cut = min(spaces, key=lambda i: abs(i - len(text) / 2))
            label.set_text(text[:cut] + "\n" + text[cut + 1:])


_LABEL_DROP = re.compile(r"\s*[\(\[].*?[\)\]]")


def _short_label(label, max_chars=12, short=True):
    """Shorten a module name for in-plot labels.

    With ``short=True`` the label is reduced to its first word after removing
    bracketed parts ("FA core complex (FANCA, ...)" -> "FA"), which keeps labels
    compact and lets related modules share one label.
    """
    label = "" if pd.isna(label) else str(label)
    if short:
        label = _LABEL_DROP.sub("", label).strip()
        label = label.split()[0].rstrip(",;:.") if label.split() else label
        if len(label) > max_chars and "-" in label:
            # Subunit lists like "COG5-COG6-COG7-COG8" -> "COG".
            label = label.split("-")[0].rstrip("0123456789") or label.split("-")[0]
    return label[: max_chars - 1] + "…" if len(label) > max_chars else label


def _rect_point_dist(box, cx, cy):
    """Distance from points (cx, cy) to a rectangle (x0, y0, x1, y1); 0 inside."""
    dx = np.maximum.reduce([box[0] - cx, np.zeros_like(cx), cx - box[2]])
    dy = np.maximum.reduce([box[1] - cy, np.zeros_like(cy), cy - box[3]])
    return np.hypot(dx, dy)


def _seg_point_dist(p, q, cx, cy):
    """Distance from points (cx, cy) to the segment p-q."""
    d = np.asarray(q, float) - np.asarray(p, float)
    denom = float(d @ d) or 1e-12
    t = np.clip(((cx - p[0]) * d[0] + (cy - p[1]) * d[1]) / denom, 0, 1)
    return np.hypot(cx - (p[0] + t * d[0]), cy - (p[1] + t * d[1]))


def _seg_box_hit(p, q, box, n=12):
    t = np.linspace(0.1, 1.0, n)
    xs = p[0] + t * (q[0] - p[0])
    ys = p[1] + t * (q[1] - p[1])
    return bool(np.any((xs > box[0]) & (xs < box[2]) & (ys > box[1]) & (ys < box[3])))


def _segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0
            and orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


def _nearest_on_box(box, x, y):
    return min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])


def _place_scatter_labels(
    ax,
    label_points,
    obstacle_points=None,
    obstacle_boxes=None,
    obstacle_lines=None,
    label_color="black",
    show_text_background=False,
    fontsize=8,
    connector_linewidth=0.25,
    merge_distance=15,
    n_restarts=30,
):
    """Place labels next to their points, ggrepel-style.

    Every label gets a connector to its point(s) and is chosen from a ring of
    candidate spots at growing distances. Text never covers a highlighted
    point or another label; covering grey points, running text over other
    connectors and crossing connectors are heavily penalised, and longer
    connectors cost more. Labels are fitted jointly (greedy start, then local
    search with a few seeded restarts), so the result is reproducible.

    Parameters
    ----------
    label_points : list of (x, y, text, size[, direction])
        Points to label in data coordinates; ``size`` is the scatter ``s``
        value (points^2). Points sharing a text within ``merge_distance``
        points of each other get one label with several connectors.
        ``direction`` (optional) is the preferred label direction in radians
        (0 = right, pi/2 = up) or a tuple of acceptable directions; spots
        away from it cost more, so labels go that way when there is room.
    obstacle_points : list of (x, y, size)
        Other drawn points that labels should avoid; covered only when no free
        spot exists.
    obstacle_boxes : list of matplotlib Bbox in display coordinates.
    obstacle_lines : list of ((x0, y0), (x1, y1)) in data coordinates.
    """
    if not label_points:
        return []

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0  # display pixels per typographic point
    to_px = ax.transData.transform
    to_data = ax.transData.inverted().transform
    ax_box = ax.get_window_extent(renderer)
    inner = (ax_box.x0 + 1 * pt, ax_box.y0 + 1 * pt, ax_box.x1 - 1 * pt, ax_box.y1 - 1 * pt)

    def as_px(points):
        if not points:
            return np.empty((0, 2)), np.empty(0)
        arr = np.array([[p[0], p[1]] for p in points], dtype=float)
        radius = np.sqrt(np.array([p[-1] for p in points], dtype=float)) / 2 * pt
        return to_px(arr), radius

    valid = [p for p in label_points if not (pd.isna(p[0]) or pd.isna(p[1]))]
    sig_xy, sig_r = as_px([(p[0], p[1], p[3]) for p in valid])
    prefer = [p[4] if len(p) > 4 else None for p in valid]
    bg_xy, bg_r = as_px([p for p in (obstacle_points or []) if not (pd.isna(p[0]) or pd.isna(p[1]))])
    lines_px = [(to_px(a), to_px(b)) for a, b in (obstacle_lines or [])]
    placed_boxes = [(b.x0, b.y0, b.x1, b.y1) for b in (obstacle_boxes or [])]

    # Merge nearby points that share a label text (e.g. several "FA" modules).
    groups = []
    for i, point in enumerate(valid):
        text = point[2]
        for group in groups:
            if group["text"] == text and np.max(
                np.hypot(*(sig_xy[group["members"]] - sig_xy[i]).T)
            ) <= merge_distance * pt:
                group["members"].append(i)
                break
        else:
            groups.append({"text": text, "members": [i]})

    bbox_props = dict(facecolor="white", edgecolor="none", pad=0.5) if show_text_background else None
    gap = 1.5 * pt
    angles = np.deg2rad(np.arange(0, 360, 15))
    radii = np.array([4, 7, 10, 14, 19, 25, 32, 40, 50, 65, 80, 100, 125, 150]) * pt
    min_link = 3 * pt  # every label gets a visible connector, however short
    fixed_boxes = np.array(placed_boxes, dtype=float).reshape(-1, 4)
    samples = np.linspace(0.1, 1.0, 12)
    n_keep = 120  # cheapest candidates per label kept for the joint search

    def static_candidates(group):
        """All candidate spots for one label with the cost that does not depend
        on the other labels (points, reference lines, connector length)."""
        members = group["members"]
        w, h = group["size"]
        center = sig_xy[members].mean(axis=0)
        reach = np.max(np.hypot(*(sig_xy[members] - center).T) + sig_r[members])
        rows = []
        for radius in radii:
            for angle in angles:
                c, s = np.cos(angle), np.sin(angle)
                ax_, ay_ = center + (reach + radius) * np.array([c, s])
                ha = "left" if c > 0.38 else "right" if c < -0.38 else "center"
                va = "bottom" if s > 0.38 else "top" if s < -0.38 else "center"
                x0 = ax_ if ha == "left" else ax_ - w if ha == "right" else ax_ - w / 2
                y0 = ay_ if va == "bottom" else ay_ - h if va == "top" else ay_ - h / 2
                if x0 < inner[0] or y0 < inner[1] or x0 + w > inner[2] or y0 + h > inner[3]:
                    continue
                padded = (x0 - gap, y0 - gap, x0 + w + gap, y0 + h + gap)
                if np.any(_rect_point_dist(padded, *sig_xy.T) < sig_r):
                    continue  # never hide a highlighted point under text
                r_pt = radius / pt
                cost = 0.5 * r_pt + 0.02 * r_pt ** 2  # short connectors are much preferred
                # Preferred direction: straight that way is free, diagonal costs a
                # little, sideways more, the opposite way most (but all far less
                # than crossing connectors, which stay the last resort).
                wanted = prefer[members[0]]
                if wanted is not None:
                    options = wanted if isinstance(wanted, (tuple, list)) else (wanted,)
                    cost += min(60 * (1 - np.cos(angle - a)) for a in options)
                if len(bg_xy):
                    # Each covered grey point costs more than a long connector or a
                    # crossing: labels cover grey points only when nothing else fits,
                    # and then as few as possible.
                    cost += 150 * np.count_nonzero(_rect_point_dist(padded, *bg_xy.T) < bg_r)
                for a, b in lines_px:
                    if _seg_box_hit(a, b, padded, n=400):
                        cost += 8
                links = []
                for m in members:
                    p = sig_xy[m]
                    q = np.array(_nearest_on_box(padded, *p))
                    links.append((p, q))
                    length = np.hypot(*(q - p))
                    if length - sig_r[m] < min_link:
                        cost += 1e3  # connector would be hidden under the point
                    cost += 0.5 * length / pt
                    crossed = np.setdiff1d(np.arange(len(sig_xy)), [m])
                    cost += 1e3 * np.count_nonzero(_seg_point_dist(p, q, *sig_xy[crossed].T) < sig_r[crossed])
                    if len(bg_xy):
                        cost += 50 * np.count_nonzero(_seg_point_dist(p, q, *bg_xy.T) < bg_r)
                    for a, b in lines_px:  # connector crossing a reference line (diagonal)
                        if _segments_intersect(p, q, a, b):
                            cost += 40
                rows.append((cost, (ax_, ay_), ha, va, padded, links))
        rows.sort(key=lambda row: row[0])
        rows = rows[:n_keep]
        if not rows:
            return None
        link_arr = np.array([[np.concatenate([p, q]) for p, q in row[5]] for row in rows])  # (N, m, 4)
        return {
            "cost": np.array([row[0] for row in rows]),
            "anchor": [row[1] for row in rows],
            "ha": [row[2] for row in rows],
            "va": [row[3] for row in rows],
            "box": np.array([row[4] for row in rows]),  # (N, 4)
            "links": link_arr,
            # sample points along each connector, (N, m, S, 2)
            "link_pts": link_arr[:, :, None, :2] + samples[None, None, :, None]
            * (link_arr[:, :, None, 2:] - link_arr[:, :, None, :2]),
        }

    def dynamic_cost(cand, boxes, links, link_pts):
        """Cost of every candidate against the labels already placed (vectorized)."""
        n = len(cand["cost"])
        cost = np.zeros(n)
        cb = cand["box"]
        if len(boxes):
            overlap = (
                (cb[:, None, 0] < boxes[None, :, 2]) & (boxes[None, :, 0] < cb[:, None, 2])
                & (cb[:, None, 1] < boxes[None, :, 3]) & (boxes[None, :, 1] < cb[:, None, 3])
            )
            cost += 1e5 * overlap.any(axis=1)
            # own connectors running through other labels
            pts = cand["link_pts"].reshape(n, -1, 2)
            inside = (
                (pts[:, :, None, 0] > boxes[None, None, :, 0]) & (pts[:, :, None, 0] < boxes[None, None, :, 2])
                & (pts[:, :, None, 1] > boxes[None, None, :, 1]) & (pts[:, :, None, 1] < boxes[None, None, :, 3])
            )
            cost += 1e4 * inside.any(axis=1).sum(axis=1)
        if len(links):
            # text over other labels' connectors
            px, py = link_pts[..., 0], link_pts[..., 1]  # (K, S)
            inside = (
                (px[None] > cb[:, None, None, 0]) & (px[None] < cb[:, None, None, 2])
                & (py[None] > cb[:, None, None, 1]) & (py[None] < cb[:, None, None, 3])
            )
            cost += 1e4 * inside.any(axis=2).sum(axis=1)
            # connector crossings
            own = cand["links"]  # (N, m, 4)
            a, b = own[:, :, None, :2], own[:, :, None, 2:]
            c, d = links[None, None, :, :2], links[None, None, :, 2:]

            def orient(p, q, r):
                return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                               - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

            cross = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
            cost += 1500 * cross.sum(axis=(1, 2))  # crossing connectors read badly
        return cost

    for group in groups:
        group["artist"] = ax.text(0, 0, group["text"], fontsize=fontsize, color=label_color,
                                  zorder=5, bbox=bbox_props)
        extent = group["artist"].get_window_extent(renderer)
        group["size"] = (extent.width, extent.height)
        group["cand"] = static_candidates(group)
        group["choice"] = None

    def context(exclude):
        boxes, links, pts = [fixed_boxes], [], []
        for other in groups:
            if other is exclude or other["choice"] is None:
                continue
            cand, i = other["cand"], other["choice"]
            boxes.append(cand["box"][i][None])
            links.append(cand["links"][i])
            pts.append(cand["link_pts"][i])
        return (
            np.concatenate(boxes),
            np.concatenate(links) if links else np.empty((0, 4)),
            np.concatenate(pts) if pts else np.empty((0, len(samples), 2)),
        )

    def total_cost(group, ctx):
        cand = group["cand"]
        return cand["cost"] + dynamic_cost(cand, *ctx)

    def solve(order):
        for group in groups:
            group["choice"] = None
        for group in order:
            if group["cand"] is not None:
                group["choice"] = int(np.argmin(total_cost(group, context(group))))
        for _ in range(20):
            moved = False
            for group in groups:
                if group["cand"] is None:
                    continue
                costs = total_cost(group, context(group))
                best = int(np.argmin(costs))
                if costs[best] < costs[group["choice"]] - 1e-6:
                    group["choice"], moved = best, True
            if not moved:
                break
        return {id(g): (total_cost(g, context(g))[g["choice"]] if g["cand"] is not None else np.inf)
                for g in groups}

    # Greedy pass with the most constrained labels first, then local search where
    # each label moves to its best spot given all the others. Penalties between
    # two labels are symmetric, so every move lowers the total and the search ends.
    order = sorted(groups, key=lambda g: -(g["cand"]["cost"][0] if g["cand"] is not None else np.inf))
    costs = solve(order)
    state = [g["choice"] for g in groups]
    # Ruin and recreate: a label stuck on a bad spot is placed first and the rest
    # re-fitted around it; keep that layout when the total cost drops.
    # Then a few random orders (fixed seed, so output is reproducible).
    rng = np.random.default_rng(0)
    for attempt in range(n_restarts):
        stuck = [g for g in order if costs[id(g)] >= 1e3 and np.isfinite(costs[id(g)])]
        if not stuck:
            break
        if attempt < len(stuck):
            trial_order = [stuck[attempt]] + [g for g in order if g is not stuck[attempt]]
        else:
            trial_order = [groups[i] for i in rng.permutation(len(groups))]
        trial = solve(trial_order)
        if sum(trial.values()) < sum(costs.values()):
            costs, order, state = trial, trial_order, [g["choice"] for g in groups]
    for group, choice in zip(groups, state):
        group["choice"] = choice

    texts = []
    skipped = []
    for group in groups:
        text, cand, i = group["artist"], group["cand"], group["choice"]
        if cand is None or costs[id(group)] >= 1e5:
            # No free spot: drop the label rather than stack it on another one.
            text.remove()
            skipped.append(group["text"])
            continue
        text.set_position(to_data(cand["anchor"][i]))
        text.set_ha(cand["ha"][i])
        text.set_va(cand["va"][i])
        texts.append(text)
        for link in cand["links"][i]:
            (x0, y0), (x1, y1) = to_data([link[:2], link[2:]])
            ax.plot([x0, x1], [y0, y1], color=label_color, linewidth=connector_linewidth,
                    zorder=1.5, solid_capstyle="butt")

    if skipped:
        log.warning(
            f"{len(skipped)} label(s) left out for lack of space: {', '.join(skipped)}. "
            "Lower n_labels or enlarge figsize."
        )
    return texts


def _unit_words():
    """("complex", "complexes") for CORUM, otherwise ("module", "modules")."""
    try:
        standard = str((dload("config") or {}).get("functional_standard", "")).upper()
    except Exception:
        standard = ""
    return ("complex", "complexes") if standard == "CORUM" else ("module", "modules")


def _tp_axis_label():
    """Label of the true-positive count axis of global PR plots."""
    return f"Co-{_unit_words()[0]}\nmembership (TP)"


def _display_name(key):
    """Dataset name as the user wrote it ("Soft Tissue"), not its file-safe key ("Soft_Tissue")."""
    names = dload("input", "names")
    if not isinstance(names, dict):
        colors = dload("input", "colors")
        names = {_sanitize(k): k for k in colors} if isinstance(colors, dict) else {}
    return names.get(_sanitize(key), key)


# Fixed room (inches) around the plot area for 8 pt ticks, two-line axis
# labels and legends; it does not shrink with figure_scale because text doesn't.
_MARGINS = {"left": 0.8, "right": 0.25, "bottom": 0.75, "top": 0.35}


def _start(ax, size):
    """Return ``(fig, ax, owns)``: a new standalone figure, or the panel axes passed in.

    ``size`` is the plot-area (axes box) size in inches at ``figure_scale`` 1;
    it is multiplied by ``figure_scale``. Text keeps its size, so the figure
    gets fixed margins for it (the saved file is cropped to its content).
    """
    if ax is None:
        scale = _STYLE["scale"]
        w, h = size[0] * scale, size[1] * scale
        m = _MARGINS
        W, H = w + m["left"] + m["right"], h + m["bottom"] + m["top"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([m["left"] / W, m["bottom"] / H, w / W, h / H])
        return fig, ax, True
    return ax.figure, ax, False


def _finish(fig, owns, config, filename, save=None):
    """Save, show and close a standalone figure. Panels are left to :func:`plot_panels`."""
    if not owns:
        return
    plot_config = config["plotting"]
    if plot_config.get("save_plot", False) if save is None else save:
        output_type = plot_config.get("output_type", "pdf")
        path = Path(config["output_folder"]) / f"{filename}.{output_type}"
        path.parent.mkdir(parents=True, exist_ok=True)
        _wrap_axis_labels(fig)
        fig.savefig(path, bbox_inches="tight", format=output_type)
    if plot_config.get("show_plot", True):
        _wrap_axis_labels(fig)
        plt.show()
    plt.close(fig)


def _after_layout(ax, owns, func):
    """Run ``func`` now for a standalone figure, or once :func:`plot_panels` has
    fixed the panel layout. Label placement works in screen space, so it must
    run after the axes have their final size."""
    if owns:
        func()
    else:
        ax._pflex_after_layout = getattr(ax, "_pflex_after_layout", []) + [func]


def _setup_log_tp_axis(ax, hide_minor_ticks):
    ax.set_xscale("log")
    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax._pflex_log_ticks = True  # every decade ticked; labels fitted in _fit_unit_ticks


def _finish_pr_axes(ax, min_precision, title=None):
    ax.set_box_aspect(CURVE_BOX_ASPECT)
    ax.set(xlabel=_tp_axis_label(), ylabel="Precision")
    _set_title(ax, title)
    ax.set_ylim(0, 1)
    _unit_ticks(ax, "y")
    _trim_pr_xaxis(ax, min_precision)
    _place_legend(ax)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


@_publication_style
def plot_precision_recall_curve(line_width=1.0, hide_minor_ticks=True, min_precision=0.05,
                                title="Global Precision–Recall (precision of top-ranked gene pairs)",
                                caption="Gene pairs ranked by score; a true positive is a pair "
                                        "sharing a module in {standard}.",
                                ax=None):
    pra = dload("pra")
    config = dload("config")
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    _setup_log_tp_axis(ax, hide_minor_ticks)

    if isinstance(pra, dict):
        names = list(pra.keys())
        colors = _plot_dataset_colors(names, None, config, dload("input", "colors"))
        for name, color in zip(names, colors):
            val = pra[name]
            val = val[val.tp > 10]
            ax.plot(val.tp, val.precision, c=color, label=_display_name(name), linewidth=line_width, alpha=0.9)
    else:
        pra = pra[pra.tp > 10]
        ax.plot(pra.tp, pra.precision, c="black", label="Precision Recall Curve", linewidth=line_width, alpha=0.9)

    _finish_pr_axes(ax, min_precision, title)
    _caption(ax, (caption or "").format(standard=config.get("functional_standard", "the reference set")))
    _finish(fig, owns, config, "precision_recall_curve")
    return ax


@_publication_style
def plot_aggregated_pra(agg_df, line_width=1.0, hide_minor_ticks=True, min_precision=0.05, ax=None):
    """
    Plots an aggregated Precision-Recall curve with mean line and min-max shading.
    agg_df should be indexed by 'tp' and contain 'mean', 'min', 'max' columns for precision.
    """
    config = dload("config")
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    _setup_log_tp_axis(ax, hide_minor_ticks)

    agg_df = agg_df[agg_df.index > 10]
    ax.fill_between(agg_df.index, agg_df['min'], agg_df['max'], color='gray', alpha=0.3, label='Range (Min-Max)')
    ax.plot(agg_df.index, agg_df['mean'], c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    _finish_pr_axes(ax, min_precision)
    _finish(fig, owns, config, "aggregated_precision_recall_curve")
    return ax


@_publication_style
def plot_iqr_pra(agg_df, line_width=1.0, hide_minor_ticks=True, min_precision=0.05, ax=None):
    """
    Plots an aggregated Precision-Recall curve with mean line and IQR (25-75%) shading.
    agg_df should be indexed by 'tp' and contain 'mean', '25%', '75%' columns for precision.
    """
    config = dload("config")
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    _setup_log_tp_axis(ax, hide_minor_ticks)

    agg_df = agg_df[agg_df.index > 10]
    ax.fill_between(agg_df.index, agg_df['25%'], agg_df['75%'], color='gray', alpha=0.3, label='IQR (25-75%)')
    ax.plot(agg_df.index, agg_df['mean'], c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    _finish_pr_axes(ax, min_precision)
    _finish(fig, owns, config, "aggregated_iqr_precision_recall_curve")
    return ax


@_publication_style
def plot_all_runs_pra(pra_list, mean_df=None, line_width=1.0, hide_minor_ticks=True, min_precision=0.05, ax=None):
    """
    Plots all individual Precision-Recall curves faintly, with an optional mean line.
    pra_list: list of dataframes (each with 'tp' and 'precision' columns) OR list of Series (if index is tp)
    mean_df: optional dataframe with 'mean' column indexed by tp
    """
    config = dload("config")
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    _setup_log_tp_axis(ax, hide_minor_ticks)

    for i, df in enumerate(pra_list):
        df_filtered = df[df['tp'] > 10] if 'tp' in df.columns else df[df.index > 10]
        x = df_filtered['tp'] if 'tp' in df_filtered.columns else df_filtered.index
        y = df_filtered['precision'] if 'precision' in df_filtered.columns else df_filtered.values
        # Only the first line gets a legend entry.
        ax.plot(x, y, c="gray", linewidth=0.3, alpha=0.3, label="Individual Runs" if i == 0 else None)

    if mean_df is not None:
        mean_df = mean_df[mean_df.index > 10]
        ax.plot(mean_df.index, mean_df['mean'], c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    _finish_pr_axes(ax, min_precision)
    _finish(fig, owns, config, "aggregated_all_runs_precision_recall_curve")
    return ax


@_publication_style
def plot_per_module_scatter(
    n_top=10,
    n_labels=None,
    sig_color='black',
    nonsig_color='white',
    label_color='black',
    border_color=None,
    border_width=0.25,
    nonsig_border_color='black',
    nonsig_border_width=0.25,
    show_labels=True,
    show_text_background=False,
    short_labels=True,
    label_map=None,
    max_label_chars=12,
    fontsize=8,
    figsize=(2.0, 2.0),
    point_scale=1.0,
    spine_offset=4,
    diagonal_color='0.6',
    size_legend=True,
    size_legend_genes=(5, 20, 100),
    labels_per_dataset=4,
    labels_in_both=2,
    title="Complex-level AUPRC comparison",
    caption="Colored and labelled: the {k} best modules specific to each dataset (of the "
            "top {n_top}); grey labelled: the {k_both} best in both. Circle area scales "
            "with module size.",
    pair=None,
    ax=None,
):
    """Compare per-module AUPRC between every pair of datasets.

    Modules in the top ``n_top`` of either dataset are filled with that
    dataset's colour (black when top in both); all others are drawn as light
    grey circles. Circle area scales with the number of genes in the module.

    The defaults produce a final-size figure (3 x 3 in, 8 pt Arial, 0.25 pt
    borders, axes offset from the data) whose text stays editable in PDF/SVG.

    Parameters
    ----------
    n_top : int
        Modules ranked in the top ``n_top`` of a dataset are candidates.
    labels_per_dataset : int
        Colour and label the best ``labels_per_dataset`` candidates that are
        top in only that dataset (default 4 per dataset).
    labels_in_both : int
        Label (in grey) the best ``labels_in_both`` modules top in both
        datasets, ranked by their mean AUPRC (default 2). Every coloured point
        is labelled; all other modules are grey.
    n_labels : deprecated
        Ignored; use ``labels_per_dataset`` / ``labels_in_both``.
    short_labels : bool
        Label with the first word of the module name ("FA core complex" ->
        "FA"); nearby modules with the same short name share one label.
    label_map : dict, optional
        Explicit label text per module name, applied instead of shortening.
    border_color : str, optional
        Border of highlighted points; defaults to their fill colour.
    point_scale : float
        Multiplier for circle area (area = genes * 1.25 * point_scale).
    spine_offset : float
        Points by which the x/y axis lines are moved away from the data.
    size_legend : bool
        Show reference circle sizes under the plot, above the caption.
    size_legend_genes : tuple of int
        Gene counts shown in the size legend (default 5, 20, 100).
    pair : tuple of two dataset names, optional
        Plot only this pair (x, y). Defaults to every pair, or the first pair
        when drawing into ``ax``.
    ax : matplotlib Axes, optional
        Draw into this axes (e.g. a panel from :func:`plot_panels`) instead of
        a new figure; nothing is saved.
    """
    config = dload("config")
    rdict = dload("pra_per_module")
    input_colors = dload("input", "colors")
    input_colors = {_sanitize(k): v for k, v in input_colors.items()} if input_colors else {}
    label_map = label_map or {}
    if n_labels is not None:
        log.warning(
            "plot_per_module_scatter: n_labels is ignored; use labels_per_dataset "
            "and labels_in_both."
        )

    if len(rdict) < 2:
        log.warning(
            "Skipping plot: at least two datasets are required for per-module scatter plot."
        )
        return

    if pair is not None:
        column_pairs = [tuple(_sanitize(name) for name in pair)]
    else:
        column_pairs = list(combinations(rdict.keys(), 2))
        if ax is not None:
            column_pairs = column_pairs[:1]
    df = pd.DataFrame()

    for i, (key, val) in enumerate(rdict.items()):
        val = val.rename(columns={"auc_score": key})
        if i == 0:
            df = val.copy().drop(columns=["Genes", "Length", "used_genes"], errors="ignore")
        else:
            df = pd.concat([df, val[key]], axis=1)

    n_genes = df['n_used_genes'] if 'n_used_genes' in df else pd.Series(1, index=df.index)
    sizes = pd.Series(_marker_area(n_genes, point_scale), index=df.index)

    def label_text(name):
        if name in label_map:
            return str(label_map[name])
        return _short_label(name, max_chars=max_label_chars, short=short_labels)

    panel = ax
    for pair in column_pairs:
        extreme_indices_0 = df[pair[0]].sort_values(ascending=False).head(n_top).index
        extreme_indices_1 = df[pair[1]].sort_values(ascending=False).head(n_top).index
        top_in_both = extreme_indices_0.intersection(extreme_indices_1)
        # Only the best few per group are coloured and labelled; the rest stay grey.
        significant_pair0_only = (
            df.loc[extreme_indices_0.difference(extreme_indices_1), pair[0]]
            .sort_values(ascending=False).head(labels_per_dataset).index
        )
        significant_pair1_only = (
            df.loc[extreme_indices_1.difference(extreme_indices_0), pair[1]]
            .sort_values(ascending=False).head(labels_per_dataset).index
        )
        significant_in_both = (
            df.loc[top_in_both, [pair[0], pair[1]]].mean(axis=1)
            .sort_values(ascending=False).head(labels_in_both).index
        )
        significant_indices = significant_pair0_only.union(significant_pair1_only).union(significant_in_both)

        bg_df = df.drop(index=significant_indices)
        fig, ax, owns = _start(panel, figsize)

        # Background: white circles with a grey hairline, largest drawn first so
        # the white fill never hides a smaller circle.
        bg_order = sizes[bg_df.index].sort_values(ascending=False).index
        ax.scatter(
            bg_df.loc[bg_order, pair[0]], bg_df.loc[bg_order, pair[1]],
            facecolors=nonsig_color, edgecolors=nonsig_border_color,
            s=sizes[bg_order], linewidth=nonsig_border_width,
            zorder=1,
        )

        def scatter_significant(indices, color, zorder=2):
            if len(indices) == 0:
                return
            ax.scatter(
                df.loc[indices, pair[0]], df.loc[indices, pair[1]],
                facecolors=color, edgecolors=border_color or color,
                s=sizes[indices], linewidth=border_width, zorder=zorder
            )

        # Dataset-specific significant modules use the dataset input color.
        scatter_significant(significant_pair0_only, input_colors.get(_sanitize(pair[0]), sig_color), zorder=2)
        scatter_significant(significant_pair1_only, input_colors.get(_sanitize(pair[1]), sig_color), zorder=2)
        # Modules top in both datasets look like the background (they sit near the
        # diagonal and are not a difference); they are still labelled.
        if len(significant_in_both):
            ax.scatter(
                df.loc[significant_in_both, pair[0]], df.loc[significant_in_both, pair[1]],
                facecolors=nonsig_color, edgecolors=nonsig_border_color,
                s=sizes[significant_in_both], linewidth=nonsig_border_width, zorder=2,
            )

        ax.plot([0, 1], [0, 1], linestyle='-', color=diagonal_color, linewidth=0.5, zorder=0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        _unit_ticks(ax, "xy")
        ax.set_xlabel(f"{_display_name(pair[0])}\nper-module AUPRC")
        ax.set_ylabel(f"{_display_name(pair[1])}\nper-module AUPRC")
        _set_title(ax, (title or "").format(x=_display_name(pair[0]), y=_display_name(pair[1])))

        # Nature style: no grid, open top/right spines, axes pulled away from the data.
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', spine_offset))
        ax.spines['bottom'].set_position(('outward', spine_offset))

        legend = None
        if size_legend:
            steps = [g for g in size_legend_genes if g <= n_genes.max()] or [int(n_genes.max())]
            handles = [
                Line2D([], [], linestyle='', marker='o', markerfacecolor=nonsig_color,
                       markeredgecolor=nonsig_border_color, markeredgewidth=nonsig_border_width,
                       markersize=float(np.sqrt(_marker_area(g, point_scale))), label=str(g))
                for g in steps
            ]
            # Under the x label (fixed gap in font sizes), left-aligned with the axes.
            from matplotlib.transforms import offset_copy
            # Left edge on the axes' left edge, below the offset spine, tick
            # labels and the x label: 36 pt plus one line per extra label line.
            below_pt = 36 + 10 * ax.get_xlabel().count("\n")
            legend = ax.legend(
                handles=handles, title="Genes in module", loc="upper left",
                bbox_to_anchor=(0.0, 0.0),
                bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=-below_pt, units="points"),
                ncol=len(handles), frameon=True, fancybox=True,
                handlelength=0.6, handletextpad=0.3, columnspacing=0.9,
                borderaxespad=0.0, borderpad=0.35,
                title_fontsize=fontsize, fontsize=fontsize, labelcolor=CAPTION_COLOR,
            )
            # Reads as a key, not data: thin light-grey frame, grey text, title
            # aligned with the first circle.
            legend._legend_box.align = "left"
            legend.get_title().set_color(CAPTION_COLOR)
            frame = legend.get_frame()
            frame.set_edgecolor("0.75")
            frame.set_linewidth(0.25)
            frame.set_facecolor("white")
            frame.set_boxstyle("round", pad=0.0, rounding_size=0.25)
        _caption(ax, (caption or "").format(n_top=n_top, k=labels_per_dataset, k_both=labels_in_both),
                 below=[ax] + ([legend] if legend else []))


        if show_labels:
            sig = df.loc[significant_indices, [pair[0], pair[1], "Name"]].dropna(subset=[pair[0], pair[1]])
            # Most informative first (placed first): high scores that differ between datasets.
            priority = sig[[pair[0], pair[1]]].max(axis=1) + (sig[pair[0]] - sig[pair[1]]).abs()
            order = priority.sort_values(ascending=False).index
            direction = {idx: 0.0 for idx in significant_pair0_only}          # x-specific: right
            direction.update({idx: np.pi / 2 for idx in significant_pair1_only})  # y-specific: up
            direction.update({idx: (3 * np.pi / 4, -np.pi / 4) for idx in significant_in_both})
            label_points = [
                (sig.loc[idx, pair[0]], sig.loc[idx, pair[1]], label_text(sig.loc[idx, "Name"]),
                 sizes[idx], direction.get(idx))
                for idx in order
            ]
            obstacles = [
                (df.loc[idx, pair[0]], df.loc[idx, pair[1]], sizes[idx])
                for idx in bg_df.index.append(significant_indices.difference(order))
            ]

            def place(ax=ax, label_points=label_points, obstacles=obstacles):
                ax.figure.canvas.draw()
                _place_scatter_labels(
                    ax,
                    label_points,
                    obstacle_points=obstacles,
                    obstacle_lines=[((0, 0), (1, 1))],
                    label_color=label_color,
                    show_text_background=show_text_background,
                    fontsize=fontsize,
                    connector_linewidth=0.25,
                )

            _after_layout(ax, owns, place)

        _finish(fig, owns, config, f"per_module_scatter_{pair[0]}_vs_{pair[1]}")
    return ax


@_publication_style
def plot_per_module_scatter_by_size(
    n_labels=5,
    n_top=10,
    sig_color='black',
    nonsig_color='white',
    label_color='black',
    border_color=None,
    border_width=0.25,
    nonsig_border_color='black',
    nonsig_border_width=0.25,
    show_labels=True,
    show_text_background=False,
    short_labels=True,
    label_map=None,
    max_label_chars=12,
    figsize=(2.0, 2.0),
    point_scale=1.0,
    spine_offset=4,
    title="Per-module AUPRC vs module size ({dataset})",
    caption="Filled: the {n_labels} modules with the highest AUPRC.",
    dataset=None,
    ax=None,
):
    """Per-module AUPRC against module size, one figure per dataset.

    The ``n_labels`` best modules are filled with the dataset colour and
    labelled; styling follows :func:`plot_per_module_scatter`. ``dataset``
    picks one dataset (default: all, or the first when drawing into ``ax``).
    """
    config = dload("config")
    rdict = dload("pra_per_module")
    input_colors = dload("input", "colors")
    input_colors = {_sanitize(k): v for k, v in input_colors.items()} if input_colors else {}
    label_map = label_map or {}

    keys = list(rdict.keys())
    if dataset is not None:
        keys = [_sanitize(dataset)]
    elif ax is not None:
        keys = keys[:1]

    def label_text(name):
        if name in label_map:
            return str(label_map[name])
        return _short_label(name, max_chars=max_label_chars, short=short_labels)

    def size(genes):
        return _marker_area(genes, point_scale)

    panel = ax
    for key in keys:
        per_module = rdict[key]
        dataset_color = input_colors.get(_sanitize(key), sig_color)
        sorted_pc = per_module.sort_values(by="auc_score", ascending=False, na_position="last")
        top_labels, rest = sorted_pc.head(n_labels), sorted_pc.iloc[n_labels:]

        fig, ax, owns = _start(panel, figsize)

        # Background: white circles with a grey hairline, largest drawn first.
        rest = rest.sort_values("n_used_genes", ascending=False)
        ax.scatter(
            rest.auc_score, rest.n_used_genes,
            facecolors=nonsig_color, edgecolors=nonsig_border_color,
            linewidth=nonsig_border_width, s=size(rest.n_used_genes),
            zorder=1
        )
        # Top modules are filled with the dataset colour.
        ax.scatter(
            top_labels.auc_score, top_labels.n_used_genes,
            facecolors=dataset_color, edgecolors=border_color or dataset_color,
            linewidth=border_width, s=size(top_labels.n_used_genes),
            zorder=2
        )

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_xlabel(f"{_display_name(key)} AUPRC")
        ax.set_ylabel("Genes in module")
        ax.set_box_aspect(1)
        _set_title(ax, (title or "").format(dataset=_display_name(key)))
        _caption(ax, (caption or "").format(n_labels=n_labels))
        ax.grid(visible=False, which='both', axis='both')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, sorted_pc.n_used_genes.max() * 1.05)
        _unit_ticks(ax, "x")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', spine_offset))
        ax.spines['bottom'].set_position(('outward', spine_offset))

        if show_labels:
            label_points = [
                (row.auc_score, row.n_used_genes, label_text(row.Name), float(_marker_area(row.n_used_genes, point_scale)))
                for _, row in top_labels.iterrows()
            ]
            obstacles = [
                (row.auc_score, row.n_used_genes, float(_marker_area(row.n_used_genes, point_scale)))
                for _, row in rest.iterrows()
            ]

            def place(ax=ax, label_points=label_points, obstacles=obstacles):
                _place_scatter_labels(
                    ax,
                    label_points,
                    obstacle_points=obstacles,
                    label_color=label_color,
                    show_text_background=show_text_background,
                    fontsize=PUBLICATION_RC["font.size"],
                    connector_linewidth=0.25,
                )

            _after_layout(ax, owns, place)

        _finish(fig, owns, config, f"per_module_scatter_by_modulesize_{key}")
    return ax


@_publication_style
def plot_module_contributions(
    min_pairs=10,
    min_precision_cutoff=0.5,
    num_module_to_show=10,
    y_lim=None,
    fig_title=None,
    fig_labs=('Fraction TP explained', 'Precision'),
    legend_rows=6,   # rows in the legend below the plot
    max_label_chars=18,
    title="Modules driving the global PR curve ({dataset})",
    caption="Share of true-positive gene pairs from each module across precision; "
            "grey: all other modules.",
    dataset=None,
    ax=None,
):
    """Share of true-positive pairs contributed by the top modules across precision.

    ``dataset`` picks one dataset (default: all, or the first when drawing into ``ax``).
    """
    config = dload("config")
    plot_data_dict = dload("module_contributions")

    keys = list(plot_data_dict.keys())
    if dataset is not None:
        keys = [_sanitize(dataset)]
    elif ax is not None:
        keys = keys[:1]

    panel = ax
    for key in keys:
        plot_data = plot_data_dict[key]
        s = plot_data.set_index('Name').sum()
        find_last_precision = s[s > min_pairs].index[-1]
        last_prec_value = float(find_last_precision.split('_')[1])

        plot_data = plot_data.drop_duplicates(subset='Name')
        cont_stepwise_anno = plot_data['Name']
        cont_stepwise_mat = plot_data.drop(columns=['Name'])
        tmp_TP = cont_stepwise_mat.sum(axis=0)
        Precision_ind = (tmp_TP >= min_pairs)
        cont_stepwise_mat = cont_stepwise_mat.loc[:, Precision_ind]
        tmp = cont_stepwise_mat.columns
        y = np.array([float(col.split('_')[1]) if isinstance(col, str) and '_' in col else col for col in tmp])
        x = cont_stepwise_mat.sum(axis=0)
        mx, nx = cont_stepwise_mat.shape[0], cont_stepwise_mat.shape[1]
        tmp = np.tile(x, (mx, 1))
        x = cont_stepwise_mat.values / tmp
        x_df = pd.DataFrame(x, index=cont_stepwise_anno, columns=cont_stepwise_mat.columns)

        ind_for_mean = y >= (last_prec_value - min_precision_cutoff)
        if sum(ind_for_mean) == 0:
            log.info("No values above 'min.precision.cutoff'"); return False
        if sum(ind_for_mean) == 1:
            log.info("Only one value above 'min.precision.cutoff'"); return False

        a = x_df.loc[:, ind_for_mean].mean(axis=1).sort_values()[-num_module_to_show:]
        subset = x_df.loc[a.index, :]

        # Categorical palette so neighbouring modules stay distinguishable; grey = others.
        palette = plt.get_cmap("tab10" if len(subset) <= 10 else "tab20").colors
        colors = [(0.6, 0.6, 0.6, 1.0)] + [palette[i % len(palette)] for i in range(len(subset))][::-1]
        others = pd.DataFrame(1 - subset.sum(axis=0), columns=['others']).T
        merged = pd.concat([others, subset], ignore_index=False)
        X = merged.to_numpy()
        x1 = np.zeros_like(X); x2 = np.zeros_like(X)
        for i in range(X.shape[0]):
            if i == 0:
                x2[i, :] = X[0, :]
            elif i == 1:
                x1[i, :] = X[0, :]
            else:
                x1[i, :] = X[:i, :].sum(axis=0)
            if i > 0:
                x2[i, :] = X[:i + 1, :].sum(axis=0)

        padding = 0.02
        lower = max(0, min(y) - padding)
        upper = last_prec_value + padding

        fig, ax, owns = _start(panel, (2.6, 1.8))
        ax.set_xlim(0, 1)
        _unit_ticks(ax, "x")
        ax.set_ylim(lower, upper)
        ax.set_xlabel(fig_labs[0])
        ax.set_ylabel(fig_labs[1])
        _set_title(ax, (title or "").format(dataset=_display_name(key)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i in range(X.shape[0]):
            ax.fill_betweenx(y, x1[i, :], x2[i, :], color=colors[i], edgecolor='white', linewidth=0.25)

        # Legend below the plot, largest contributor first, then "others".
        labels = [_short_label(lbl, max_chars=max_label_chars, short=False) for lbl in merged.index]
        handles = [patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        order = list(range(len(handles) - 1, 0, -1)) + [0]
        ncols = int(np.ceil(len(handles) / max(1, legend_rows)))
        legend = ax.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            # Anchored to the axes bottom with a fixed gap (in font sizes) that
            # clears tick labels and the x label, whatever the panel height.
            loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=ncols,
            title="Modules", handlelength=0.9, handletextpad=0.3,
            borderaxespad=3.4, labelspacing=0.25, columnspacing=0.8,
        )
        _caption(ax, caption, below=[legend])

        _finish(fig, owns, config, f"module_contributions_{key}")
    return ax


@_publication_style
def plot_significant_modules(title="Modules recovered at each AUPRC threshold",
                             caption="Number of modules whose per-module AUPRC is at least the threshold.",
                             ax=None):
    config = dload("config")
    pra_per_module = dload("pra_per_module")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    if not isinstance(pra_per_module, dict) or not pra_per_module:
        log.warning("No per-module PRA data found. Run pra_per_module() first.")
        return pd.DataFrame(index=thresholds)

    datasets = list(pra_per_module.keys())
    num_datasets = len(datasets)

    df = pd.DataFrame(index=thresholds)
    for key, module_data in pra_per_module.items():
        score_col = "corrected_auc_score" if "corrected_auc_score" in module_data.columns else "auc_score"
        df[key] = [module_data.query(f'{score_col} >= {t}').shape[0] for t in thresholds]

    # Square plot area like the other panels; each threshold group takes 60 %
    # of its slot, so bars get thinner rather than the plot wider.
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    ax.set_box_aspect(1)
    colors = _plot_dataset_colors(datasets, None, config, dload("input", "colors"))

    bar_width = 0.6 / num_datasets
    for i, dataset in enumerate(datasets):
        x = np.arange(len(thresholds)) + i * bar_width
        ax.bar(x, df[dataset], width=bar_width, color=colors[i], edgecolor='black', label=_display_name(dataset))

    ax.set_xticks(np.arange(len(thresholds)) + (num_datasets - 1) * bar_width / 2)
    ax.set_xticklabels([str(t) for t in thresholds], rotation=0, ha='center')
    ax.set_xlabel("Per-module\nAUPRC threshold")
    ax.set_ylabel(f"# {_unit_words()[1]}")
    _set_title(ax, title)

    # Nature style: no grid; open top/right spines
    ax.grid(False)
    for spine in ('right', 'top'):
        ax.spines[spine].set_visible(False)

    _place_legend(ax)
    _caption(ax, caption)
    _finish(fig, owns, config, "number_of_significant_modules")
    return df


@_publication_style
def plot_auc_scores(title="Area under the global PR curve per dataset",
                    caption="Area under the precision\u2013recall curve of all gene pairs.",
                    ax=None):
    config = dload("config")
    pra_dict = dload("pr_auc")

    sorted_items = sorted(pra_dict.items(), key=lambda x: x[1], reverse=True)
    datasets = [k for k, _ in sorted_items]
    auc_scores = [v for _, v in sorted_items]
    colors = _plot_dataset_colors(datasets, None, config, dload("input", "colors"))

    fig, ax, owns = _start(ax, _bar_figsize(len(datasets)))
    _value_bars(ax, [_display_name(d) for d in datasets], auc_scores, colors)
    ax.set_ylabel("AUPRC")
    _set_title(ax, title)
    _caption(ax, caption)
    _finish(fig, owns, config, "auprc_values")
    return pra_dict


DEFAULT_COLORS = [
    "#4E79A7",
    "#E15759",
    "#76B7B2",
    "#F28E2B",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

FILTER_VARIANTS = (
    "all_complexes",
    "without_mt_ribo_etci",
    "without_small_high_auprc",
)

FILTER_VARIANT_STYLES = {
    "all_complexes": {"linestyle": "-", "label": "All complexes"},
    "without_mt_ribo_etci": {"linestyle": "--", "label": "− mtRibo / ETC I"},
    "without_small_high_auprc": {"linestyle": ":", "label": "− small, high AUPRC"},
}


def _plot_dataset_names(category, dataset_names, prepare_function):
    stored = dload(category)
    available = stored if isinstance(stored, dict) else {}
    names = list(available) if dataset_names is None else [_sanitize(name) for name in dataset_names]
    missing = [name for name in names if name not in available]
    if not names or missing:
        detail = f" Missing datasets: {missing}." if missing else ""
        raise RuntimeError(
            f"No complete '{category}' results are available.{detail} "
            f"Run {prepare_function}(name) for each dataset first."
        )
    return names, {name: available[name] for name in names}


def _plot_dataset_colors(dataset_names, colors, config, input_colors):
    if colors is not None:
        return list(colors)
    input_colors = (
        {_sanitize(key): value for key, value in input_colors.items()}
        if input_colors
        else {}
    )
    cmap_name = config.get("color_map", "tab10")
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap("tab10")
    count = len(dataset_names)
    defaults = [
        cmap(i) if count <= 10 and cmap_name == "tab10"
        else cmap(float(i) / max(count - 1, 1))
        for i in range(count)
    ]
    return [
        input_colors.get(_sanitize(name), defaults[i])
        for i, name in enumerate(dataset_names)
    ]


MODULE_AXIS_MAX = 200
MODULE_AXIS_TICKS = (2, 20, 200)


def _configure_module_axis(ax):
    """Log module-count axis, always 1 to 200, labelled only at 2, 20 and 200.

    Curves that run past 200 modules are cut at the axis edge.
    """
    ax.set_xscale("log")
    ax.set_xlim(1, MODULE_AXIS_MAX)
    ax.set_xlabel(f"# {_unit_words()[1]} (TP)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)
    _unit_ticks(ax, "y")
    ax.set_xticks(MODULE_AXIS_TICKS, [str(t) for t in MODULE_AXIS_TICKS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_mpr_figure(fig, config, outname, default_stem):
    output_type = config["plotting"].get("output_type", "pdf")
    filename = outname or f"{default_stem}.{output_type}"
    path = Path(filename)
    if len(path.parts) == 1:
        path = Path(config["output_folder"]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    _wrap_axis_labels(fig)
    fig.savefig(path, bbox_inches="tight", format=output_type)


@_publication_style
def plot_mpr_module_auc_scores(save=None, outname=None, title="Area under the mPR curve per dataset",
                               caption="Area under the module-coverage (mPR) curve.", ax=None):
    """Plot the unfiltered mPR AUC score for each prepared dataset."""
    config = dload("config")
    plot_config = config["plotting"]
    stored = dload("mpr_modules_auc")
    if not isinstance(stored, dict) or not stored:
        raise RuntimeError(
            "No mPR AUC results found. Run mpr_prepare(name) for each dataset first."
        )
    values = {
        name: float(value)
        for name, value in stored.items()
        if np.isscalar(value)
    }
    scores = pd.Series(values, dtype=float).sort_values(ascending=False)
    if scores.empty:
        raise RuntimeError(
            "Stored mPR AUC results use an unsupported legacy format. "
            "Run mpr_prepare(name) again for each dataset."
        )

    colors = _plot_dataset_colors(list(scores.index), None, config, dload("input", "colors"))
    fig, ax, owns = _start(ax, _bar_figsize(len(scores)))
    _value_bars(ax, [_display_name(name) for name in scores.index], list(scores.values), colors, fmt="%.2f")
    ax.set_ylabel("mPR modules AUC")
    _set_title(ax, title)
    _caption(ax, caption)

    if owns:
        should_save = plot_config.get("save_plot", False) if save is None else bool(save)
        if should_save:
            _save_mpr_figure(fig, config, outname, "mpr_modules_auc")
        if plot_config.get("show_plot", True):
            plt.show()
        plt.close(fig)
    return scores


@_publication_style
def plot_mpr_module_coverage_curve(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.0,
    show_markers=False,
    marker_size=6,
    title="Modules recovered at each precision (mPR)",
    caption="Number of modules with at least one true-positive pair above each precision cutoff.",
):
    """Plot the unfiltered module-coverage mPR curve across datasets."""
    config = dload("config")
    dataset_names, stored = _plot_dataset_names(
        "mpr", dataset_names, "mpr_prepare"
    )
    colors = _plot_dataset_colors(
        dataset_names,
        colors,
        config,
        dload("input", "colors"),
    )
    fig, ax, owns = _start(ax, CURVE_FIGSIZE)
    ax.set_box_aspect(CURVE_BOX_ASPECT)
    _set_title(ax, title)

    _configure_module_axis(ax)
    for i, name in enumerate(dataset_names):
        data = stored[name]
        cutoffs = np.asarray(data["precision_cutoffs"], dtype=float)
        coverage = np.asarray(data["coverage_curve"], dtype=float)
        mask = coverage > 0  # beyond x_max the line is clipped at the axis edge
        if not mask.any():
            continue
        x_values = coverage[mask]
        y_values = cutoffs[mask]
        use_markers = x_values.size <= 10 if show_markers == "auto" else bool(show_markers)
        if x_values.size == 1:
            ax.scatter(x_values, y_values, color=colors[i], s=marker_size,
                       label=_display_name(name), zorder=3)
        else:
            ax.plot(
                x_values,
                y_values,
                color=colors[i],
                linewidth=linewidth,
                marker="o" if use_markers else None,
                markersize=2 if use_markers else None,
                label=_display_name(name),
            )

    _place_legend(ax)
    _caption(ax, caption)
    if save and owns:
        _save_mpr_figure(fig, config, outname, "mpr_modules_multi")
    return ax


def _filter_panels(host, title):
    """Split ``host`` into one side-by-side panel per complex filter.

    Filters become panel titles and datasets keep their usual colours, so no
    dashed/dotted line styles or long legend are needed.
    """
    host.set_axis_off()
    n, gap = len(FILTER_VARIANTS), 0.06
    width = (1 - gap * (n - 1)) / n
    axes = []
    for i, variant in enumerate(FILTER_VARIANTS):
        ax = host.inset_axes([i * (width + gap), 0, width, 1])
        ax.set_box_aspect(CURVE_BOX_ASPECT)
        ax.set_anchor("N")  # top-aligned, so the overall title sits a fixed gap above
        ax.set_title(FILTER_VARIANT_STYLES[variant]["label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i:
            ax.sharey(axes[0])
            ax.tick_params(labelleft=False)
        axes.append(ax)
    if title:
        _set_title(host, title, pad=18)
    return axes


FILTER_CAPTION = ("Left to right: all complexes; mitochondrial ribosome and ETC I complexes "
                  "removed; small complexes with high AUPRC removed.")


def _filter_legend(axes, host, dataset_names, colors, linewidth):
    """Dataset legend in an empty corner of the first panel, else above the panels."""
    handles = [
        Line2D([0], [0], color=colors[i % len(colors)], linewidth=linewidth)
        for i in range(len(dataset_names))
    ]
    # Fallback: above the row of panels, clear of the per-filter titles.
    return _place_legend(axes[0], handles, [_display_name(name) for name in dataset_names],
                         above_ax=host, above_pad=2.4)


@_publication_style
def plot_mpr_filter(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.0,
    show_markers=False,
    marker_size=6,
    title="Modules recovered under complex filters (mPR)",
    caption=FILTER_CAPTION,
):
    """Compare the three lazily computed complex-filter mPR variants, one panel per filter."""
    config = dload("config")
    dataset_names, stored = _plot_dataset_names(
        "mpr_filter", dataset_names, "mpr_filter"
    )
    colors = _plot_dataset_colors(
        dataset_names,
        colors,
        config,
        dload("input", "colors"),
    )
    fig, host, owns = _start(ax, (6.0, 2.0))
    axes = _filter_panels(host, title)

    auc_values = {}
    for ax, variant in zip(axes, FILTER_VARIANTS):
        _configure_module_axis(ax)
        for i, name in enumerate(dataset_names):
            data = stored[name]
            cutoffs = np.asarray(data["precision_cutoffs"], dtype=float)
            auc_values[name] = data["modules_auc"]
            coverage = np.asarray(data["coverage_curves"][variant], dtype=float)
            mask = coverage > 0  # beyond x_max the line is clipped at the axis edge
            if not mask.any():
                continue
            x_values = coverage[mask]
            y_values = cutoffs[mask]
            use_markers = x_values.size <= 10 if show_markers == "auto" else bool(show_markers)
            if x_values.size == 1:
                ax.scatter(x_values, y_values, color=colors[i], s=marker_size, zorder=3)
            else:
                ax.plot(
                    x_values,
                    y_values,
                    color=colors[i],
                    linewidth=linewidth,
                    marker="o" if use_markers else None,
                    markersize=2 if use_markers else None,
                )
    for i, ax in enumerate(axes):
        if i:
            ax.set_ylabel("")
        if i != len(axes) // 2:
            ax.set_xlabel("")
    _filter_legend(axes, host, dataset_names, colors, linewidth)
    _caption(host, caption, below=axes)
    if save and owns:
        _save_mpr_figure(fig, config, outname, "mpr_filter_comparison")
    return host, pd.DataFrame.from_dict(auc_values, orient="index")[list(FILTER_VARIANTS)]


@_publication_style
def plot_globalpr_filter(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.0,
    min_precision=0.05,
    title="Global precision–recall under complex filters",
    caption=FILTER_CAPTION,
):
    """Compare the three lazily computed complex-filter global PR variants, one panel per filter."""
    config = dload("config")
    dataset_names, stored = _plot_dataset_names(
        "globalpr_filter", dataset_names, "globalpr_filter"
    )
    colors = _plot_dataset_colors(
        dataset_names,
        colors,
        config,
        dload("input", "colors"),
    )
    fig, host, owns = _start(ax, (6.0, 2.0))
    axes = _filter_panels(host, title)

    xmax = 0.0
    for ax, variant in zip(axes, FILTER_VARIANTS):
        for i, name in enumerate(dataset_names):
            curve = stored[name]["curves"][variant]
            tp = np.asarray(curve["tp"], dtype=float)
            precision = np.asarray(curve["precision"], dtype=float)
            mask = np.isfinite(tp) & (tp > 0) & np.isfinite(precision) & (precision > 0)
            if not mask.any():
                continue
            xmax = max(xmax, float(tp[mask].max()))
            ax.plot(tp[mask], precision[mask], color=colors[i], linewidth=linewidth)

    axes[0].set_ylabel("Precision")
    axes[len(axes) // 2].set_xlabel(_tp_axis_label())
    axes[0].set_ylim(0.0, 1.05)
    for ax in axes:
        _unit_ticks(ax, "y")
    if xmax > 0:
        for ax in axes:
            _setup_log_tp_axis(ax, True)
        x_min = 10 if xmax > 10 else 1
        _trim_pr_xaxis(axes, min_precision, x_min=x_min)
        for ax in axes[1:]:
            ax.set_xlim(axes[0].get_xlim())
    _filter_legend(axes, host, dataset_names, colors, linewidth)
    _caption(host, caption, below=axes)
    if save and owns:
        _save_mpr_figure(fig, config, outname, "globalpr_filter_comparison")
    return host


@_publication_style
def plot_mpr_summary(
    dataset_names=None,
    colors=None,
    save=True,
    linewidth=1.0,
    show_markers=False,
    marker_size=6,
):
    """Plot unfiltered module-coverage mPR curves and their dataset AUCs."""
    plot_mpr_module_coverage_curve(
        dataset_names=dataset_names,
        colors=colors,
        save=save,
        linewidth=linewidth,
        show_markers=show_markers,
        marker_size=marker_size,
    )
    return plot_mpr_module_auc_scores(save=save)


def _fit_axes_in_cell(ax, cell, renderer):
    """Move/resize ``ax`` so everything it draws (ticks, labels, legends) lies
    inside ``cell`` = (x0, y0, x1, y1) in display pixels."""
    fig = ax.figure
    tight = ax.get_tightbbox(renderer)
    box = ax.get_window_extent(renderer)
    left, bottom = box.x0 - tight.x0, box.y0 - tight.y0
    right, top = tight.x1 - box.x1, tight.y1 - box.y1
    x0, y0 = cell[0] + left, cell[1] + bottom
    x1, y1 = cell[2] - right, cell[3] - top
    if x1 - x0 < 10 or y1 - y0 < 10:
        return  # content wider than the cell; leave it rather than invert the axes
    W, H = fig.bbox.width, fig.bbox.height
    ax.set_position([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


@_publication_style
def plot_panels(
    panels,
    ncols=2,
    panel_size=(3.4, 3.6),
    labels="abcdefghijklmnopqrstuvwxyz",
    outname="figure_panels",
    save=None,
    show=None,
):
    """Combine pflex plots into one multi-panel figure labelled a, b, c, ...

    Every panel gets a cell of ``panel_size`` inches; ticks, axis labels and
    legends are fitted inside the cell, so panels line up without manual work.

    Parameters
    ----------
    panels : list
        One entry per panel, filled row by row: a plot function such as
        ``flex.plot_auc_scores``, a ``(function, kwargs)`` tuple such as
        ``(flex.plot_per_module_scatter, {"pair": ("Skin", "Soft Tissue")})``,
        or ``(function, kwargs, span)`` for a panel ``span`` columns wide
        (the complex-filter plots need 2). ``None`` leaves a cell empty.
    ncols : int
        Panels per row.
    panel_size : (width, height)
        Design size of one cell in inches, multiplied by ``figure_scale``.
        3.4 x 3.6 fits a square plot with its legend; two columns fit a double
        journal column (~7 in).
    labels : str or list
        Panel letters, in order. Use ``""`` for none.
    outname : str
        File name (without extension) inside ``output_folder``.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> flex.plot_panels([
    ...     flex.plot_precision_recall_curve,
    ...     (flex.plot_per_module_scatter, {"n_top": 10}),
    ...     flex.plot_auc_scores,
    ...     flex.plot_mpr_module_coverage_curve,
    ...     (flex.plot_globalpr_filter, {}, 2),
    ... ], ncols=2)
    """
    import inspect

    config = dload("config")
    plot_config = config["plotting"]

    # Lay entries out row by row; a panel spanning k columns wraps to the
    # next row when it does not fit in the current one.
    slots, row, col = [], 0, 0
    for entry in panels:
        span = entry[2] if isinstance(entry, tuple) and len(entry) > 2 else 1
        span = max(1, min(int(span), ncols))
        if col + span > ncols:
            row, col = row + 1, 0
        slots.append((row, col, span))
        col += span
    nrows = max(1, (slots[-1][0] + 1) if slots else 1)
    pw, ph = panel_size[0] * _STYLE["scale"], panel_size[1] * _STYLE["scale"]
    letter_h = 0.15  # inches above each cell reserved for its letter
    fig = plt.figure(figsize=(pw * ncols, ph * nrows))
    dpi = fig.dpi

    placed = []
    letters = iter(labels)
    for entry, (row, col, span) in zip(panels, slots):
        if entry is None:
            continue
        if isinstance(entry, tuple):
            func, kwargs = entry[0], (entry[1] if len(entry) > 1 else {})
        else:
            func, kwargs = entry, {}
        kwargs = dict(kwargs)
        if "save" in inspect.signature(func).parameters:
            kwargs.setdefault("save", False)
        # Cell in display pixels, letter strip excluded.
        cell = (
            col * pw * dpi,
            (nrows - 1 - row) * ph * dpi,
            (col + span) * pw * dpi,
            ((nrows - row) * ph - letter_h) * dpi,
        )
        W, H = pw * ncols * dpi, ph * nrows * dpi
        ax = fig.add_axes([
            (cell[0] + 0.5 * dpi) / W, (cell[1] + 0.4 * dpi) / H,
            (cell[2] - cell[0] - 0.6 * dpi) / W, (cell[3] - cell[1] - 0.5 * dpi) / H,
        ])
        func(ax=ax, **kwargs)
        placed.append((ax, cell, next(letters, "")))

    # Tick labels change with axes size, so fit a few times until stable;
    # long axis labels are split onto two lines once the sizes are known.
    for attempt in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, cell, _ in placed:
            _fit_axes_in_cell(ax, cell, renderer)
        if attempt == 1:
            _wrap_axis_labels(fig)
    fig.canvas.draw()

    for ax, cell, letter in placed:
        for func in getattr(ax, "_pflex_after_layout", []):
            func()
        if letter:
            fig.text(cell[0] / fig.bbox.width, (cell[3] / dpi + letter_h) / (ph * nrows),
                     letter, fontweight="bold", ha="left", va="top")

    if plot_config.get("save_plot", False) if save is None else save:
        output_type = plot_config.get("output_type", "pdf")
        path = Path(config["output_folder"]) / f"{outname}.{output_type}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", format=output_type)
    if plot_config.get("show_plot", True) if show is None else show:
        plt.show()
    return fig
