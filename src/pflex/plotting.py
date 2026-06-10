# Standard library
from itertools import combinations
from pathlib import Path
import re

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

# Completely disable LaTeX and clear all font cache/references
import matplotlib as mpl
import matplotlib.font_manager as fm

# Disable LaTeX rendering completely
mpl.rcParams['text.usetex'] = False

# Reset all font-related parameters to system defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Bitstream Vera Serif', 'serif']
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Bitstream Vera Sans', 'sans-serif']
mpl.rcParams['font.cursive'] = ['Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT', 'Felipa', 'cursive']
mpl.rcParams['font.fantasy'] = ['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans', 'fantasy']
mpl.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Computer Modern Typewriter', 'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace']

# Remove any LaTeX-specific math font settings
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['mathtext.default'] = 'regular'

# Force font manager to rebuild with system fonts only
try:
    fm.fontManager.__init__()
except Exception:
    pass

# Local modules
from .utils import dload, _sanitize
from .logging_config import log

def plot_precision_recall_curve(line_width=2.0, hide_minor_ticks=True):
    pra = dload("pra")
    config = dload("config")
    plot_config = config["plotting"]
    input_colors = dload("input", "colors")  # Load individual color overrides
    
    # Sanitize color keys to match dataset keys
    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    # Get color map from config, default to "tab10" if not found
    cmap_name = config.get("color_map", "tab10")
    try:
        cmap = get_cmap(cmap_name)
    except ValueError:
        log.warning(f"Color map '{cmap_name}' not found. Falling back to 'tab10'.")
        cmap = get_cmap("tab10")

    # Increase figure width to accommodate external legend without squashing axes
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Adjust layout to make room for legend on the right
    plt.subplots_adjust(right=0.7)
    
    ax.set_xscale("log")

    # optionally hide minor ticks on the log axis
    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

    if isinstance(pra, dict):
        # Determine colors for each dataset
        dataset_names = list(pra.keys())
        num_datasets = len(dataset_names)
        
        # Generate default colors from colormap
        if num_datasets <= 10 and cmap_name == "tab10":
             default_colors = [cmap(i) for i in range(num_datasets)]
        else:
             default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

        for i, (key, val) in enumerate(pra.items()):
            # Use override color if available, otherwise use default from cmap
            color = input_colors.get(key) if input_colors else None
            if color is None:
                color = default_colors[i]
            
            val = val[val.tp > 10]
            ax.plot(val.tp, val.precision, c=color, label=key, linewidth=line_width, alpha=0.9)
    else:
        pra = pra[pra.tp > 10]
        ax.plot(pra.tp, pra.precision, c="black", label="Precision Recall Curve", linewidth=line_width, alpha=0.9)

    ax.set(title="",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    ax.set_ylim(0, 1)

    # Nature style: no grid, open top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_path = Path(config["output_folder"]) / f"precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()
    plt.close(fig)

def plot_aggregated_pra(agg_df, line_width=2.0, hide_minor_ticks=True):
    """
    Plots an aggregated Precision-Recall curve with mean line and min-max shading.
    agg_df should be indexed by 'tp' and contain 'mean', 'min', 'max' columns for precision.
    """
    config = dload("config")
    plot_config = config["plotting"]
    
    # Increase figure width to accommodate external legend without squashing axes
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Adjust layout to make room for legend on the right
    plt.subplots_adjust(right=0.7)
    
    ax.set_xscale("log")

    # optionally hide minor ticks on the log axis
    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Filter out very low TP counts if necessary, similar to plot_precision_recall_curve
    agg_df = agg_df[agg_df.index > 10]
    
    tp = agg_df.index
    mean_prec = agg_df['mean']
    min_prec = agg_df['min']
    max_prec = agg_df['max']

    # Plot shading
    ax.fill_between(tp, min_prec, max_prec, color='gray', alpha=0.3, label='Range (Min-Max)')
    
    # Plot mean line
    ax.plot(tp, mean_prec, c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    ax.set(title="",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    ax.set_ylim(0, 1)

    # Nature style: no grid, open top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_path = Path(config["output_folder"]) / f"aggregated_precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()
    plt.close(fig)

def plot_iqr_pra(agg_df, line_width=2.0, hide_minor_ticks=True):
    """
    Plots an aggregated Precision-Recall curve with mean line and IQR (25-75%) shading.
    agg_df should be indexed by 'tp' and contain 'mean', '25%', '75%' columns for precision.
    """
    config = dload("config")
    plot_config = config["plotting"]
    
    # Increase figure width to accommodate external legend without squashing axes
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Adjust layout to make room for legend on the right
    plt.subplots_adjust(right=0.7)
    
    ax.set_xscale("log")

    # optionally hide minor ticks on the log axis
    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Filter out very low TP counts
    agg_df = agg_df[agg_df.index > 10]
    
    tp = agg_df.index
    mean_prec = agg_df['mean']
    q25_prec = agg_df['25%']
    q75_prec = agg_df['75%']

    # Plot shading
    ax.fill_between(tp, q25_prec, q75_prec, color='gray', alpha=0.3, label='IQR (25-75%)')
    
    # Plot mean line
    ax.plot(tp, mean_prec, c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    ax.set(title="Precision-Recall (IQR)",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    ax.set_ylim(0, 1)

    # Nature style
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_path = Path(config["output_folder"]) / f"aggregated_iqr_precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()
    plt.close(fig)

def plot_all_runs_pra(pra_list, mean_df=None, line_width=2.0, hide_minor_ticks=True):
    """
    Plots all individual Precision-Recall curves faintly, with an optional mean line.
    pra_list: list of dataframes (each with 'tp' and 'precision' columns) OR list of Series (if index is tp)
    mean_df: optional dataframe with 'mean' column indexed by tp
    """
    config = dload("config")
    plot_config = config["plotting"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(right=0.7)
    
    ax.set_xscale("log")

    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Plot individual lines
    for i, df in enumerate(pra_list):
        # Ensure we filter low TPs same as others
        df_filtered = df[df['tp'] > 10] if 'tp' in df.columns else df[df.index > 10]
        
        x = df_filtered['tp'] if 'tp' in df_filtered.columns else df_filtered.index
        y = df_filtered['precision'] if 'precision' in df_filtered.columns else df_filtered.values
        
        # Only add label for the first line to avoid cluttering legend
        lbl = "Individual Runs" if i == 0 else None
        ax.plot(x, y, c="gray", linewidth=0.5, alpha=0.3, label=lbl)

    # Plot mean line if provided
    if mean_df is not None:
        mean_df = mean_df[mean_df.index > 10]
        ax.plot(mean_df.index, mean_df['mean'], c="black", label="Mean Precision", linewidth=line_width, alpha=0.9)

    ax.set(title="Precision-Recall (All Runs)",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=False)
    ax.set_ylim(0, 1)

    # Nature style
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_path = Path(config["output_folder"]) / f"aggregated_all_runs_precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()
    plt.close(fig)

def plot_percomplex_scatter(
    n_top=10,
    sig_color='black',
    nonsig_color='white',
    label_color='black',
    border_color='black',
    border_width=1.0,
    nonsig_border_color="#7F7F7F",
    nonsig_border_width=0.5,
    show_labels=True,
    show_text_background=False,
):
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")
    input_colors = dload("input", "colors")
    input_colors = {_sanitize(k): v for k, v in input_colors.items()} if input_colors else {}

    if len(rdict) < 2:
        log.warning(
            "Skipping plot: at least two datasets are required for per-complex scatter plot."
        )
        return

    column_pairs = list(combinations(rdict.keys(), 2))
    df = pd.DataFrame()

    for i, (key, val) in enumerate(rdict.items()):
        val = val.rename(columns={"auc_score": key})
        if i == 0:
            df = val.copy().drop(columns=["Genes", "Length", "used_genes"], errors="ignore")
        else:
            df = pd.concat([df, val[key]], axis=1)

    for pair in column_pairs:
        extreme_indices_0 = df[pair[0]].sort_values(ascending=False).head(n_top).index
        extreme_indices_1 = df[pair[1]].sort_values(ascending=False).head(n_top).index
        significant_indices = extreme_indices_0.union(extreme_indices_1)
        significant_in_both = extreme_indices_0.intersection(extreme_indices_1)
        significant_pair0_only = extreme_indices_0.difference(extreme_indices_1)
        significant_pair1_only = extreme_indices_1.difference(extreme_indices_0)

        bg_df  = df.drop(index=significant_indices)
        sig_df = df.loc[significant_indices]
        sig_sizes = (
            sig_df['n_used_genes']
            if 'n_used_genes' in sig_df
            else pd.Series(1, index=sig_df.index)
        ) * 8

        # Create square figure
        fig, ax = plt.subplots(figsize=(6, 6))

        # Background cloud: non-significant complexes are opaque white-filled circles.
        bg_sizes = (bg_df['n_used_genes'] if 'n_used_genes' in bg_df else pd.Series(1, index=bg_df.index)) * 5
        ax.scatter(
            bg_df[pair[0]], bg_df[pair[1]],
            facecolors=nonsig_color, edgecolors=nonsig_border_color,
            s=bg_sizes, linewidth=nonsig_border_width,
            zorder=0
        )

        def scatter_significant(indices, color, zorder=2):
            if len(indices) == 0:
                return
            point_df = df.loc[indices]
            point_sizes = (
                point_df['n_used_genes']
                if 'n_used_genes' in point_df
                else pd.Series(1, index=point_df.index)
            ) * 8
            ax.scatter(
                point_df[pair[0]], point_df[pair[1]],
                facecolors=color, edgecolors=color,
                s=point_sizes, linewidth=border_width, zorder=zorder
            )

        # Dataset-specific significant complexes use the dataset input color.
        scatter_significant(
            significant_pair0_only,
            input_colors.get(_sanitize(pair[0]), sig_color),
            zorder=2,
        )
        scatter_significant(
            significant_pair1_only,
            input_colors.get(_sanitize(pair[1]), sig_color),
            zorder=2,
        )
        # Complexes significant in both datasets stay black to avoid ambiguous color mixing.
        scatter_significant(significant_in_both, "black", zorder=3)

        if show_labels:
            # Improved label positioning with adaptive spacing
            coords = sorted(
                [(sig_df.loc[idx, pair[0]], sig_df.loc[idx, pair[1]], idx) for idx in sig_df.index],
                key=lambda c: (-c[1], -c[0])
            )
            
            # Calculate proper parameters for normalized coordinate system (0-1 range)
            max_y = 1.0  # Normalized plots use 0-1 range
            scale_factor = 1.0  # Standard scaling for normalized plots
            min_distance = 0.08  # Increased spacing for 0-1 range to avoid overlap

            adjusted_coords = adjust_text_positions_improved(
                coords, sig_sizes,
                min_distance=min_distance,
                max_y=max_y,
                scale_factor=scale_factor,
                y_threshold=0.8  # Points above this will have labels below
            )

            for x, adj_y, idx, direction in adjusted_coords:
                y = df.loc[idx, pair[1]]

                # Calculate connector line extension, but constrain within plot bounds
                line_extension_factor = 1.5  # Reduced from 2.5 to keep labels in bounds
                extended_adj_y = y + (adj_y - y) * line_extension_factor

                # Clip to ensure connector stays within 0-1 range
                extended_adj_y = max(0.02, min(extended_adj_y, 0.98))

                # Draw connector line
                ax.plot([x, x], [y, extended_adj_y],
                       color=label_color, linewidth=0.6, zorder=3)

                # Position text at the end of extended line with small offset
                text_y_offset = 0.01 if direction == "up" else -0.01
                final_text_y = extended_adj_y + text_y_offset

                # Final clip to ensure text stays within 0-1 range
                final_text_y = max(0.02, min(final_text_y, 0.98))

                bbox_props = dict(facecolor="white", edgecolor="none", pad=1) if show_text_background else None

                ax.text(
                    x, final_text_y,
                    df.loc[idx, 'Name'][:10] + '.' if len(df.loc[idx, 'Name']) > 10 else df.loc[idx, 'Name'],
                    fontsize=4,
                    ha='left',
                    va='bottom' if direction == "up" else 'top',
                    color=label_color,
                    linespacing=1,
                    zorder=4,
                    clip_on=True,  # Enable clipping to axes bounds
                    bbox=bbox_props
                )

        # Diagonal & axes cosmetics
        ax.plot([0, 1], [0, 1], linestyle='-', color='lightgray', linewidth=0.5, zorder=1)
        
        # Force square aspect ratio and exact 0-1 range
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        
        # Set explicit ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        ax.set_xlabel(f"{pair[0]} AUPRC")
        ax.set_ylabel(f"{pair[1]} AUPRC")
        #ax.set_title(f"{pair[0]} vs {pair[1]} - Comparison of complex performance")

        # Nature style: no grid, open top/right spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if plot_config["save_plot"]:
            output_type = plot_config["output_type"]
            output_path = Path(config["output_folder"]) / f"percomplex_scatter_{pair[0]}_vs_{pair[1]}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()

        plt.close(fig)

def adjust_text_positions_improved(coords, sizes, min_distance=0.08, max_y=1.0, scale_factor=1.0, y_threshold=0.8):
    """
    Enhanced text positioning with improved logic:
    - Points above y_threshold get labels below
    - Points below y_threshold get labels above
    - Better overlap detection and resolution
    """
    adjusted = []
    
    # Base offset for lines (increased even more for better visibility)
    base_offset = 0.1 * scale_factor  # Increased from 0.06 to 0.1
    
    # Sort coords by Y position (highest first) then by X
    coords_sorted = sorted(coords, key=lambda c: (-c[1], c[0]))
    
    # Track occupied regions to prevent overlap
    occupied_regions = []
    
    for x, y, idx in coords_sorted:
        # Determine direction based on Y position
        if y > y_threshold:
            direction = "down"
            adj_y = y - base_offset
        else:
            direction = "up"
            adj_y = y + base_offset
        
        # Check for overlaps with existing labels
        overlap_found = True
        attempts = 0
        max_attempts = 10
        
        while overlap_found and attempts < max_attempts:
            overlap_found = False
            
            for occ_x, occ_y, occ_height in occupied_regions:
                # Check if current label would overlap
                x_overlap = abs(x - occ_x) < 0.15  # Horizontal proximity threshold
                y_overlap = abs(adj_y - occ_y) < occ_height
                
                if x_overlap and y_overlap:
                    overlap_found = True
                    # Adjust position to avoid overlap
                    if direction == "up":
                        adj_y = occ_y + occ_height + 0.02
                    else:
                        adj_y = occ_y - occ_height - 0.02
                    break
            
            attempts += 1
        
        # Ensure within bounds
        if direction == "up":
            adj_y = min(adj_y, max_y - 0.05)
            adj_y = max(adj_y, y + base_offset)
        else:
            adj_y = max(adj_y, 0.05)
            adj_y = min(adj_y, y - base_offset)
        
        # Add to occupied regions (x, y, approximate height)
        occupied_regions.append((x, adj_y, 0.04))
        
        adjusted.append((x, adj_y, idx, direction))
    
    return adjusted

# Alternative simplified version that groups nearby points
def adjust_text_positions_grouped(coords, sizes, min_distance=0.08, max_y=1.0, scale_factor=1.0, y_threshold=0.8):
    """
    Group nearby points and stagger their labels to avoid overlap.
    """
    adjusted = []
    
    # Group points that are close together
    groups = []
    used = set()
    
    for i, (x1, y1, idx1) in enumerate(coords):
        if i in used:
            continue
            
        group = [(x1, y1, idx1)]
        used.add(i)
        
        # Find nearby points
        for j, (x2, y2, idx2) in enumerate(coords[i+1:], i+1):
            if j in used:
                continue
            
            # Check if points are close
            if abs(x1 - x2) < 0.1 and abs(y1 - y2) < 0.1:
                group.append((x2, y2, idx2))
                used.add(j)
        
        groups.append(group)
    
    # Process each group
    for group in groups:
        if len(group) == 1:
            # Single point - simple positioning
            x, y, idx = group[0]
            direction = "down" if y > y_threshold else "up"
            base_offset = 0.06 * scale_factor
            
            if direction == "up":
                adj_y = y + base_offset
            else:
                adj_y = y - base_offset
            
            adjusted.append((x, adj_y, idx, direction))
        else:
            # Multiple points - stagger them
            group_sorted = sorted(group, key=lambda p: p[1], reverse=True)
            
            for i, (x, y, idx) in enumerate(group_sorted):
                direction = "down" if y > y_threshold else "up"
                base_offset = 0.06 * scale_factor
                stagger_offset = i * 0.04 * scale_factor
                
                if direction == "up":
                    adj_y = y + base_offset + stagger_offset
                else:
                    adj_y = y - base_offset - stagger_offset
                
                # Ensure within bounds
                adj_y = max(0.05, min(adj_y, max_y - 0.05))
                
                adjusted.append((x, adj_y, idx, direction))
    
    return adjusted

def smart_direction_assignment(point_y, y_max, min_safe_distance=20.0):
    """Determine the best direction for label placement based on Y position."""
    lower_threshold = y_max / 3
    upper_threshold = 2 * y_max / 3
    
    if point_y < lower_threshold:
        return "up_only"
    elif point_y > upper_threshold:
        return "prefer_down"
    else:
        return "both_directions"

def group_points_by_y_proximity(coords, y_tolerance=5.0):
    """Group points that have similar Y values (within tolerance)."""
    groups = []
    remaining_coords = coords.copy()
    
    while remaining_coords:
        # Start a new group with the first remaining point
        seed_point = remaining_coords.pop(0)
        current_group = [seed_point]
        seed_y = seed_point[1]
        
        # Find all points within Y tolerance of the seed point
        i = 0
        while i < len(remaining_coords):
            if abs(remaining_coords[i][1] - seed_y) <= y_tolerance:
                current_group.append(remaining_coords.pop(i))
            else:
                i += 1
                
        groups.append(current_group)
    
    return groups

def adjust_text_positions(coords, sizes, min_distance=0.08, max_y=1.0, scale_factor=1.0):
    """Enhanced text positioning with adaptive spacing for dense clusters."""
    adjusted = []
    
    # Fix scaling issues - use data coordinates, not pixel scaling
    if max_y > 10:  # For gene count plots (large Y values)
        text_height = max_y * 0.02  # 2% of Y range
        min_safe_distance = max_y * 0.05  # 5% of Y range  
        y_tolerance = max_y * 0.02  # 2% of Y range for grouping
    else:  # For normalized plots (Y values 0-1)
        text_height = 0.04 * scale_factor
        min_safe_distance = 20 * scale_factor
        y_tolerance = 5 * scale_factor
    
    # Group points by Y proximity
    groups = group_points_by_y_proximity(coords, y_tolerance)
    
    for group in groups:
        group_size = len(group)
        
        # Calculate adaptive spacing based on cluster density
        density_multiplier = calculate_density_multiplier(group_size)
        
        if group_size == 1:
            # Single point - use original logic but with direction awareness
            x, y, idx = group[0]
            direction = smart_direction_assignment(y, max_y, min_safe_distance)
            
            # Use reasonable base offset relative to Y range
            if max_y > 10:  # Gene count plots
                base_offset = max(3, max_y * 0.03)  # 3% of Y range, minimum 3 units
            else:  # Normalized plots  
                base_offset = np.sqrt(sizes.loc[idx]) * 0.04 * scale_factor if idx in sizes else 0.04 * scale_factor
            
            if direction == "up_only" or direction == "both_directions":
                adj_y = y + base_offset
            elif direction == "prefer_down" and y - base_offset > min_safe_distance:
                adj_y = y - base_offset
            else:
                adj_y = y + base_offset
                
            # Ensure within bounds with proper limits
            adj_y = max(min_safe_distance, min(adj_y, max_y - text_height))
            
            # Additional safety check to prevent extreme values
            if adj_y < 0 or adj_y > max_y * 1.2:  # Allow 20% overflow for safety
                adj_y = y + base_offset  # Fallback to simple offset
            
            adjusted.append((x, adj_y, idx))
            
        else:
            # Multiple points with similar Y - use adaptive distribution
            group.sort(key=lambda p: p[0])  # Sort by X coordinate
            
            # Determine available directions for this Y level
            group_y = group[0][1]  # All have similar Y, use first as representative
            direction = smart_direction_assignment(group_y, max_y, min_safe_distance)
            
            # Calculate adaptive spacing and base offset
            adaptive_spacing = calculate_adaptive_spacing(
                group_size, min_distance, text_height, max_y, density_multiplier
            )
            adaptive_base_offset = calculate_adaptive_base_offset(
                group_size, max_y, scale_factor, density_multiplier
            )
            
            for i, (x, y, idx) in enumerate(group):
                if direction == "up_only":
                    # Stack all labels upward with adaptive spacing
                    adj_y = y + adaptive_base_offset + (i * adaptive_spacing)
                    
                elif direction == "prefer_down":
                    # Alternate down and up with adaptive spacing
                    if i % 2 == 0 and y - adaptive_base_offset - (i//2 * adaptive_spacing) > min_safe_distance:
                        # Even indices go down
                        adj_y = y - adaptive_base_offset - (i//2 * adaptive_spacing)
                    else:
                        # Odd indices or insufficient space below - go up
                        up_level = (i//2) if i % 2 == 0 else ((i+1)//2)
                        adj_y = y + adaptive_base_offset + (up_level * adaptive_spacing)
                        
                else:  # both_directions
                    # Alternate up and down with adaptive spacing
                    if i % 2 == 0:
                        # Even indices go up
                        adj_y = y + adaptive_base_offset + (i//2 * adaptive_spacing)
                    else:
                        # Odd indices go down (if safe)
                        potential_down = y - adaptive_base_offset - ((i+1)//2 * adaptive_spacing)
                        if potential_down > min_safe_distance:
                            adj_y = potential_down
                        else:
                            # Not safe to go down, stack upward instead
                            adj_y = y + adaptive_base_offset + (i//2 * adaptive_spacing)
                
                # Final bounds check with stricter limits
                adj_y = max(min_safe_distance, min(adj_y, max_y - text_height))
                
                # Additional safety check to prevent extreme values
                if adj_y < 0 or adj_y > max_y * 1.2:  # Allow 20% overflow for safety
                    adj_y = y + adaptive_base_offset  # Fallback to simple offset
                
                adjusted.append((x, adj_y, idx))
    
    return adjusted

def calculate_density_multiplier(group_size):
    """Calculate multiplier for spacing based on cluster density."""
    if group_size <= 3:
        return 1.0
    elif group_size <= 6:
        return 1.3
    elif group_size <= 10:
        return 1.6
    elif group_size <= 15:
        return 2.0
    elif group_size <= 20:
        return 2.5
    else:  # 20+ points
        return 3.0 + (group_size - 20) * 0.1  # Progressive scaling for very dense clusters

def calculate_adaptive_spacing(group_size, min_distance, text_height, max_y, density_multiplier):
    """Calculate adaptive vertical spacing between labels based on cluster density."""
    base_spacing = max(min_distance, text_height * 1.5)
    
    # Scale spacing based on density and coordinate system
    if max_y > 10:  # Gene count plots
        adaptive_spacing = base_spacing * density_multiplier * (max_y / 100.0)
        # Ensure minimum readable spacing for dense clusters
        adaptive_spacing = max(adaptive_spacing, max_y * 0.03)
    else:  # Normalized plots
        adaptive_spacing = base_spacing * density_multiplier
        # Ensure minimum readable spacing
        adaptive_spacing = max(adaptive_spacing, 0.05)
    
    return adaptive_spacing

def calculate_adaptive_base_offset(group_size, max_y, scale_factor, density_multiplier):
    """Calculate adaptive base offset (connector line height) based on cluster density."""
    if max_y > 10:  # Gene count plots
        base_offset = max(3, max_y * 0.03)
        # Increase connector line height for dense clusters
        adaptive_offset = base_offset * density_multiplier
        # Cap to reasonable maximum
        adaptive_offset = min(adaptive_offset, max_y * 0.15)
    else:  # Normalized plots
        base_offset = 0.04 * scale_factor
        # Increase connector line height for dense clusters
        adaptive_offset = base_offset * density_multiplier
        # Cap to reasonable maximum
        adaptive_offset = min(adaptive_offset, 0.2)
    
    return adaptive_offset

def cluster_nearby_points(coords, cluster_threshold=0.1):
    """
    Group nearby points into clusters to handle overlapping labels intelligently.
    """
    clusters = []
    used = set()
    
    for i, (x1, y1, idx1) in enumerate(coords):
        if i in used:
            continue
        
        cluster = [(x1, y1, idx1)]
        used.add(i)
        
        # Find all points within cluster threshold
        for j, (x2, y2, idx2) in enumerate(coords[i+1:], i+1):
            if j in used:
                continue
            
            # Calculate distance (normalized to data range)
            x_dist = abs(x1 - x2) / 1.0  # X range is 0-1
            y_dist = abs(y1 - y2) / max(coord[1] for coord in coords)  # Y range varies
            distance = np.sqrt(x_dist**2 + y_dist**2)
            
            if distance < cluster_threshold:
                cluster.append((x2, y2, idx2))
                used.add(j)
        
        clusters.append(cluster)
    
    return clusters

def position_cluster_labels(cluster, cluster_id, max_y, effective_max_y, label_color, ax, top_labels, show_text_background):
    """
    Position labels for a cluster of points using advanced anti-overlap strategies.
    """
    cluster_size = len(cluster)
    
    if cluster_size == 1:
        # Single point - simple positioning
        x, y, idx = cluster[0]
        base_offset = max_y * 0.08
        
        label_y = y + base_offset
        if label_y > effective_max_y:
            label_y = max(y - base_offset, max_y * 0.05)
        
        connector_end = min(label_y, effective_max_y)
        
        # Draw connector line
        ax.plot([x, x], [y, connector_end], 
               color=label_color, linewidth=0.6, zorder=3)
        
        # Position text
        text_x = x + 0.02 if x < 0.7 else x - 0.02
        ha = 'left' if x < 0.7 else 'right'
        
        bbox_props = dict(facecolor="white", edgecolor="none", pad=1) if show_text_background else None
        label_text = top_labels.loc[idx, 'Name'][:12] + '...' if len(top_labels.loc[idx, 'Name']) > 12 else top_labels.loc[idx, 'Name']
        
        ax.text(
            text_x, connector_end + (max_y * 0.01),
            label_text,
            fontsize=5, ha=ha, va='bottom',
            color=label_color, zorder=4,
            clip_on=True, bbox=bbox_props
        )
        
    elif cluster_size <= 3:
        # Small cluster - vertical stacking with dynamic font size
        cluster_center_x = np.mean([p[0] for p in cluster])
        cluster_center_y = np.mean([p[1] for p in cluster])
        
        base_offset = max_y * 0.1
        font_size = 4  # Smaller font for clusters
        
        for i, (x, y, idx) in enumerate(sorted(cluster, key=lambda p: p[1], reverse=True)):
            # Stagger vertically with increased spacing
            stagger = i * (max_y * 0.06)  # Increased spacing
            label_y = cluster_center_y + base_offset + stagger
            
            if label_y > effective_max_y:
                # Switch to downward stacking
                label_y = cluster_center_y - base_offset - stagger
                label_y = max(label_y, max_y * 0.05)
            
            connector_end = min(label_y, effective_max_y)
            
            # Draw connector line from original point
            ax.plot([x, x], [y, connector_end], 
                   color=label_color, linewidth=0.5, zorder=3)
            
            # Position text with alternating sides to reduce overlap
            side_offset = 0.03 if i % 2 == 0 else -0.03
            text_x = cluster_center_x + side_offset
            text_x = max(0.02, min(text_x, 0.98))  # Keep within bounds
            
            ha = 'left' if side_offset > 0 else 'right'
            
            bbox_props = dict(facecolor="white", edgecolor="none", pad=0.5) if show_text_background else None
            label_text = top_labels.loc[idx, 'Name'][:10] + '.' if len(top_labels.loc[idx, 'Name']) > 10 else top_labels.loc[idx, 'Name']
            
            ax.text(
                text_x, connector_end + (max_y * 0.005),
                label_text,
                fontsize=font_size, ha=ha, va='bottom',
                color=label_color, zorder=4,
                clip_on=True, bbox=bbox_props
            )
    
    else:
        # Large cluster - radial/column arrangement with smaller text
        cluster_center_x = np.mean([p[0] for p in cluster])
        cluster_center_y = np.mean([p[1] for p in cluster])
        
        base_offset = max_y * 0.12
        font_size = 3.5  # Even smaller font for dense clusters
        
        # Arrange in two columns to handle many labels
        left_column = cluster[:len(cluster)//2]
        right_column = cluster[len(cluster)//2:]
        
        for col_idx, column in enumerate([left_column, right_column]):
            side = -1 if col_idx == 0 else 1  # Left or right side
            
            for i, (x, y, idx) in enumerate(column):
                # Calculate position for this column
                stagger = i * (max_y * 0.05)  # Vertical spacing
                label_y = cluster_center_y + base_offset + stagger
                
                if label_y > effective_max_y:
                    # Switch to downward if too high
                    label_y = cluster_center_y - base_offset - stagger
                    label_y = max(label_y, max_y * 0.05)
                
                connector_end = min(label_y, effective_max_y)
                
                # Draw connector line
                ax.plot([x, x], [y, connector_end], 
                       color=label_color, linewidth=0.4, zorder=3)
                
                # Position text in columns
                text_x = cluster_center_x + (side * 0.04)  # Offset to left/right
                text_x = max(0.02, min(text_x, 0.98))  # Keep within bounds
                
                ha = 'right' if side < 0 else 'left'
                
                bbox_props = dict(facecolor="white", edgecolor="none", pad=0.3) if show_text_background else None
                label_text = top_labels.loc[idx, 'Name'][:8] + '.' if len(top_labels.loc[idx, 'Name']) > 8 else top_labels.loc[idx, 'Name']
                
                ax.text(
                    text_x, connector_end,
                    label_text,
                    fontsize=font_size, ha=ha, va='bottom',
                    color=label_color, zorder=4,
                    clip_on=True, bbox=bbox_props
                )

def plot_percomplex_scatter_bysize(
    n_labels=10,
    n_top=10,
    sig_color='black',
    nonsig_color='white',
    label_color='black',
    border_color='black',
    border_width=1.0,
    nonsig_border_color="#7F7F7F",
    nonsig_border_width=0.5,
    show_labels=True,
    show_text_background=False,
):
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")
    input_colors = dload("input", "colors")
    input_colors = {_sanitize(k): v for k, v in input_colors.items()} if input_colors else {}

    for key, per_complex in rdict.items():
        dataset_color = input_colors.get(_sanitize(key), sig_color)
        sorted_pc = per_complex.sort_values(by="auc_score", ascending=False, na_position="last")
        top_labels, rest = sorted_pc.head(n_labels), sorted_pc.iloc[n_labels:]

        # Calculate data range for appropriate figure sizing
        max_genes = sorted_pc.n_used_genes.max()
        
        # Use rectangular figure with appropriate aspect ratio
        # X-axis is 0-1, Y-axis varies based on gene count data
        aspect_ratio = max_genes / 100.0  # Scale based on gene count range
        fig_height = min(max(4, aspect_ratio), 8)  # Between 4-8 inches
        fig, ax = plt.subplots(figsize=(6, fig_height))

        # Background: non-significant complexes are opaque white-filled circles.
        ax.scatter(
            rest.auc_score, rest.n_used_genes,
            facecolors=nonsig_color, edgecolors=nonsig_border_color,
            linewidth=nonsig_border_width, s=rest.n_used_genes * 5,
            label="Other Complexes",
            zorder=0
        )

        # Top N/significant complexes are filled black circles.
        ax.scatter(
            top_labels.auc_score, top_labels.n_used_genes,
            facecolors=dataset_color, edgecolors=dataset_color,
            linewidth=border_width, s=top_labels.n_used_genes * 8,
            label=f"Top {n_labels} AUC Scores", zorder=2
        )

        max_y = sorted_pc.n_used_genes.max()
        plot_margin = max_y * 0.2 if show_labels else max_y * 0.05
        effective_max_y = max_y + plot_margin

        if show_labels:
            # Enhanced anti-overlap labeling system
            coords = [(row.auc_score, row.n_used_genes, idx) for idx, row in top_labels.iterrows()]

            # Cluster nearby points
            clusters = cluster_nearby_points(coords, cluster_threshold=0.15)

            # Position labels for each cluster
            for i, cluster in enumerate(clusters):
                position_cluster_labels(
                    cluster, i, max_y, effective_max_y,
                    label_color, ax, top_labels, show_text_background
                )

        # Set y-axis to show integer values only
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("AUPRC")
        ax.set_ylabel("Number of genes in the complex")

        # Configure axes with proper boundaries
        ax.grid(visible=False, which='both', axis='both')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, effective_max_y)  # Use effective max to include label space
        
        # Set explicit x-axis ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(x_ticks)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Manual spacing adjustment instead of tight_layout to avoid warnings
        plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95)

        if plot_config["save_plot"]:
            output_type = plot_config["output_type"]
            output_path = Path(config["output_folder"]) / f"percomplex_scatter_by_complexsize_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()
        plt.close(fig)

def plot_complex_contributions(
    min_pairs=10,
    min_precision_cutoff=0.5,
    num_complex_to_show=10,
    y_lim=None,
    fig_title=None,
    fig_labs=['Fraction of TP', 'Precision'],
    legend_rows=3,   # <— NEW: rows for legend layout (try 3 or 4)
):
    config = dload("config")
    plot_config = config["plotting"]
    plot_data_dict = dload("complex_contributions")

    for key, plot_data in plot_data_dict.items():
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

        a = x_df.loc[:, ind_for_mean].mean(axis=1).sort_values()[-num_complex_to_show:]
        subset = x_df.loc[a.index, :]

        cmap = plt.get_cmap()
        colors = cmap(np.linspace(0, 1, num_complex_to_show))
        colors = np.vstack(([0.5, 0.5, 0.5, 1.0], colors))  # 'others' + top K
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
        y_lim = (lower, upper)

        # Give legend a bit more room
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1.8]})
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(*y_lim)
        ax[0].set_xlabel(fig_labs[0])
        ax[0].set_ylabel(fig_labs[1])
        #ax[0].set_title(fig_title if fig_title else f"{key} - Contribution of complexes")
        for i in range(X.shape[0]):
            ax[0].fill_betweenx(y, x1[i, :], x2[i, :], color=colors[i], edgecolor='white')

        # Legend: multi-row, constrained to width
        def _short(s, n=14): return (s[:n-1] + '…') if len(s) > n else s
        labels  = [_short(lbl) for lbl in merged.index]
        handles = [patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax[1].axis('off')
        n_items = len(handles)
        ncols   = int(np.ceil(n_items / max(1, legend_rows)))  # spread across rows
        ax[1].legend(
            handles=handles,
            loc='center',
            ncol=ncols,
            frameon=False,
            title="Complexes",
            fontsize=6, title_fontsize=6,
            handlelength=0.9, handletextpad=0.25,
            borderaxespad=0.0,
            labelspacing=0.25, columnspacing=0.6,
            mode='expand'
        )

        plt.tight_layout()

        if plot_config["save_plot"]:
            output_type  = plot_config["output_type"]
            output_folder= Path(config["output_folder"])
            output_path  = output_folder / f"complex_contributions_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()
        plt.close(fig)

def plot_significant_complexes():
    config = dload("config")
    plot_config = config["plotting"]
    pra_percomplex = dload("pra_percomplex")
    input_colors = dload("input", "colors")
    
    # Sanitize color keys
    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    if not isinstance(pra_percomplex, dict) or not pra_percomplex:
        log.warning("No per-complex PRA data found. Run pra_percomplex() first.")
        return pd.DataFrame(index=thresholds)

    datasets = list(pra_percomplex.keys())
    num_datasets = len(datasets)

    if num_datasets == 0:
        return pd.DataFrame(index=thresholds)

    df = pd.DataFrame(index=thresholds)
    for key, complex_data in pra_percomplex.items():
        if "corrected_auc_score" in complex_data.columns:
            score_col = "corrected_auc_score"
        else:
            score_col = "auc_score"

        df[key] = [complex_data.query(f'{score_col} >= {t}').shape[0] for t in thresholds]

    fig, ax = plt.subplots()

    # Color logic
    cmap_name = config.get("color_map", "tab10")
    try:
        cmap = get_cmap(cmap_name)
    except ValueError:
        cmap = get_cmap("tab10")

    if num_datasets <= 10 and cmap_name == "tab10":
        default_colors = [cmap(i) for i in range(num_datasets)]
    else:
        default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

    bar_width = 0.8 / num_datasets
    for i, dataset in enumerate(datasets):
        # Use override color if available
        color = input_colors.get(dataset) if input_colors else None
        if color is None:
            color = default_colors[i]
            
        x = np.arange(len(thresholds)) + i * bar_width
        ax.bar(x, df[dataset], width=bar_width, color=color, edgecolor='black', label=dataset)

    ax.set_xticks(np.arange(len(thresholds)) + (num_datasets - 1) * bar_width / 2)
    ax.set_xticklabels([str(t) for t in thresholds], rotation=0, ha='center')

    #ax.set_title("Number of significant complexes above AUPRC thresholds")
    ax.set_xlabel("AUPRC thresholds")
    ax.set_ylabel("Number of complexes")

    # Nature style: no grid; open top/right spines
    ax.grid(False)
    for spine in ('right', 'top'):
        ax.spines[spine].set_visible(False)

    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_folder = Path(config["output_folder"])
        output_path = output_folder / f"number_of_significant_complexes.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()

    plt.close(fig)
    return df

def plot_auc_scores():
    config = dload("config")
    plot_config = config["plotting"]
    pra_dict = dload("pr_auc")
    input_colors = dload("input", "colors")
    
    # Sanitize color keys
    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    sorted_items = sorted(pra_dict.items(), key=lambda x: x[1], reverse=True)
    datasets = [k for k, _ in sorted_items]
    auc_scores = [v for _, v in sorted_items]

    fig, ax = plt.subplots()

    # Color logic
    cmap_name = config.get("color_map", "tab10")
    try:
        cmap = get_cmap(cmap_name)
    except ValueError:
        cmap = get_cmap("tab10")
        
    num_datasets = len(datasets)
    if num_datasets <= 10 and cmap_name == "tab10":
         default_colors = [cmap(i) for i in range(num_datasets)]
    else:
         default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

    # Assign colors strictly matching the sorted dataset order
    final_colors = []
    for i, dataset in enumerate(datasets):
        color = input_colors.get(dataset) if input_colors else None
        if color is None:
            color = default_colors[i]
        final_colors.append(color)

    ax.bar(datasets, auc_scores, color=final_colors, edgecolor="black")

    ax.set_ylim(0, max(auc_scores) + 0.01)
    #ax.set_title("AUPRC values for the datasets")
    ax.set_ylabel("AUPRC")
    plt.xticks(rotation=45, ha="right")

    # Hard-disable any grid/ruler
    ax.grid(visible=False, which='both', axis='both')
    ax.set_axisbelow(False)  # make sure nothing faint is drawn beneath
    # Open spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_folder = Path(config["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"auprc_values.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()

    plt.close(fig)
    return pra_dict


def plot_mpr_complex_auc_scores(variant: str = "unfiltered", save=None, outname=None):
    """Plot AUC values for the mPR complexes curve (Fig 1F-style).

    Requires `mpr_prepare()` to have been run for each dataset.

    Parameters
    ----------
    variant : str
        One of: "unfiltered", "without_mt_ribo_etci",
        "without_small_high_auprc".
    save : bool, optional
        Whether to save the figure. If None, uses config["plotting"]["save_plot"].
    outname : str, optional
        Output filename. If None, auto-generated.

    Returns
    -------
    pd.Series
        AUC values indexed by dataset name (sorted descending).
    """
    config = dload("config")
    plot_config = config["plotting"]
    mpr_auc_dict = dload("mpr_complexes_auc")
    input_colors = dload("input", "colors")

    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    if not isinstance(mpr_auc_dict, dict) or not mpr_auc_dict:
        log.warning(
            "No mPR complexes AUC data found. Run mpr_prepare() first (it stores 'mpr_complexes_auc')."
        )
        return pd.Series(dtype=float)

    variant_key = _normalize_mpr_variant(variant)

    # Build Series: dataset -> auc
    auc_by_dataset = {}
    for dataset, per_filter in mpr_auc_dict.items():
        if not isinstance(per_filter, dict):
            continue
        val = per_filter.get(variant_key)
        if val is None:
            continue
        try:
            auc_by_dataset[dataset] = float(val)
        except (TypeError, ValueError):
            continue

    if not auc_by_dataset:
        log.warning(
            f"No mPR complex AUC values found for variant '{variant}'. "
            f"Available variants: {list(PUBLIC_MPR_VARIANTS.keys())}"
        )
        return pd.Series(dtype=float)

    s = pd.Series(auc_by_dataset).sort_values(ascending=False)
    datasets = list(s.index)
    auc_scores = list(s.values)

    fig, ax = plt.subplots()

    # Color logic (match other bar plots)
    cmap_name = config.get("color_map", "tab10")
    try:
        cmap = get_cmap(cmap_name)
    except ValueError:
        cmap = get_cmap("tab10")

    num_datasets = len(datasets)
    if num_datasets <= 10 and cmap_name == "tab10":
        default_colors = [cmap(i) for i in range(num_datasets)]
    else:
        default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

    final_colors = []
    for i, dataset in enumerate(datasets):
        color = input_colors.get(dataset) if input_colors else None
        if color is None:
            color = default_colors[i]
        final_colors.append(color)

    ax.bar(datasets, auc_scores, color=final_colors, edgecolor="black")

    ymax = max([v for v in auc_scores if np.isfinite(v)], default=0.0)
    ax.set_ylim(0, ymax + 0.01)
    ax.set_ylabel("mPR complexes AUC")
    plt.xticks(rotation=45, ha="right")

    # Styling consistent with other plots
    ax.grid(visible=False, which="both", axis="both")
    ax.set_axisbelow(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    should_save = plot_config.get("save_plot", False) if save is None else bool(save)
    if should_save:
        output_type = plot_config.get("output_type", "pdf")
        output_folder = Path(config["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        if outname is None:
            outname = f"mpr_complexes_auc_{variant_key}.{output_type}"
        output_path = Path(outname)
        if len(output_path.parts) == 1:
            output_path = output_folder / outname
        plt.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()

    plt.close(fig)
    return s


def plot_mpr_complexes_auc_scores(filter_key: str = "all"):
    """Backward-compatible wrapper for plot_mpr_complex_auc_scores()."""
    return plot_mpr_complex_auc_scores(
        variant=_legacy_filter_to_variant(filter_key, default="unfiltered")
    )

# -----------------------------------------------------------------------------
# mPR plots (Fig. 1E and Fig. 1F)
# -----------------------------------------------------------------------------

def plot_mpr_complexes(name, ax=None, save=True, outname=None):
    """
    Fig. 1F-style module-level PR:
      x-axis: number of covered complexes (log)
      y-axis: precision cutoff
      x tick labels: 0, 2, 20, 200
    """
    mpr = dload("mpr", name)
    if mpr is None:
        raise RuntimeError(
            f"plot_mpr_complexes(): mPR data for dataset '{name}' not found. "
            "Run `mpr_prepare` first."
        )

    precision_cutoffs = np.asarray(mpr["precision_cutoffs"], dtype=float)
    coverage = mpr["coverage_curves"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    labels = [
        ("all", "all data"),
        ("no_mtRibo_ETCI", "no mtRibo, ETC I"),
        ("no_small_highAUPRC", "no small, high AUPRC"),
    ]
    styles = {
        "all": {"linewidth": 1.8},
        "no_mtRibo_ETCI": {"linewidth": 1.8, "linestyle": "--"},
        "no_small_highAUPRC": {"linewidth": 1.8, "linestyle": ":"},
    }

    for key, pretty in labels:
        if key not in coverage:
            continue
        cov = np.asarray(coverage[key], dtype=float)

        # keep only positive coverage up to 200 complexes
        mask = (cov > 0) & (cov <= 200)
        if not mask.any():
            continue

        cov_plot = cov[mask]
        prec_plot = precision_cutoffs[mask]

        ax.plot(cov_plot, prec_plot, label=pretty, **styles.get(key, {}))

    # log x-axis, show up to 200 complexes
    ax.set_xscale("log")
    ax.set_xlim(1, 200)  # 1 on log scale will be labelled as "0" below

    ax.set_xlabel("Number of covered complexes")
    ax.set_ylabel("Precision cutoff")
    ax.set_ylim(0.0, 1.05)

    # ticks at positions 1, 2, 20, 200 with labels 0, 2, 20, 200
    tick_positions = [1, 2, 20, 200]
    tick_labels = ["0", "2", "20", "200"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.legend(frameon=False)
    ax.set_title(f"[{name}] 19Q2 – mPR (#complexes vs precision)")

    if save:
        if outname is None:
            outname = f"mpr_complexes_{name}.pdf"
        fig.tight_layout()
        fig.savefig(outname)

    return ax

def plot_mpr_tp(name, ax=None, save=True, outname=None):
    """
    Plot Fig. 1E-style TP vs precision curves for a dataset.

    Uses the object created by `mpr_prepare(name)` and stored with key 'mpr'.
    """
    mpr = dload("mpr", name)
    if mpr is None:
        raise RuntimeError(
            f"plot_mpr_tp(): mPR data for dataset '{name}' not found. "
            "Run `mpr_prepare` first."
        )

    tp_curves = mpr["tp_curves"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    labels = [
        ("all", "all data"),
        ("no_mtRibo_ETCI", "no mtRibo, ETC I"),
        ("no_small_highAUPRC", "no small, high AUPRC"),
    ]
    styles = {
        "all": {"linewidth": 1.8},
        "no_mtRibo_ETCI": {"linewidth": 1.8, "linestyle": "--"},
        "no_small_highAUPRC": {"linewidth": 1.8, "linestyle": ":"},
    }

    xmax = 0.0
    for key, pretty in labels:
        if key not in tp_curves:
            continue
        data = tp_curves[key]
        tp = np.asarray(data["tp"], dtype=float)
        prec = np.asarray(data["precision"], dtype=float)
        mask = np.isfinite(tp) & (tp > 0) & np.isfinite(prec) & (prec > 0)
        if not mask.any():
            continue
        tp_plot = tp[mask]
        prec_plot = prec[mask]
        xmax = max(xmax, float(tp_plot.max()))
        ax.plot(tp_plot, prec_plot, label=pretty, **styles.get(key, {}))

    ax.set_xlabel("Number of true positives")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)

    if xmax > 0:
        ax.set_xscale("log")

        # If we have enough TPs, start at 10; otherwise fall back.
        if xmax > 10:
            ax.set_xlim(10, xmax * 1.05)
            logmin = 1  # 10^1 = 10
        else:
            ax.set_xlim(1, xmax * 1.05)
            logmin = 0  # 10^0 = 1

        logmax = int(np.ceil(np.log10(xmax)))
        logmax = max(logmax, logmin)
        xticks = [10 ** k for k in range(logmin, logmax + 1)]
        ax.set_xticks(xticks)

    ax.legend(frameon=False)
    ax.set_title(f"{name} – PR (TP vs precision)")

    if save:
        if outname is None:
            outname = f"mpr_tp_{name}.pdf"
        fig.tight_layout()
        fig.savefig(outname)

    return ax

# Default color palette (colorblind-friendly)
DEFAULT_COLORS = [
    "#4E79A7",  # blue
    "#E15759",  # red
    "#76B7B2",  # teal
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]

# Public mPR variant names map to the internal keys stored by mpr_prepare().
PUBLIC_MPR_VARIANTS = {
    "unfiltered": "all",
    "without_mt_ribo_etci": "no_mtRibo_ETCI",
    "without_small_high_auprc": "no_small_highAUPRC",
}
INTERNAL_MPR_VARIANTS = {v: k for k, v in PUBLIC_MPR_VARIANTS.items()}

# mPR variant line styles keyed by internal storage names.
MPR_VARIANT_STYLES = {
    "all": {"linestyle": "-", "label": "all data"},
    "no_mtRibo_ETCI": {"linestyle": "--", "label": "no mtRibo, ETC I"},
    "no_small_highAUPRC": {"linestyle": "dotted", "label": "no small, high AUPRC"},
}

# Compatibility alias for users who imported this internal constant.
FILTER_STYLES = MPR_VARIANT_STYLES


def _normalize_mpr_variant(variant):
    """Return the internal mPR variant key for one public variant name."""
    if variant in PUBLIC_MPR_VARIANTS:
        return PUBLIC_MPR_VARIANTS[variant]
    if variant in MPR_VARIANT_STYLES:
        if variant == "all":
            return PUBLIC_MPR_VARIANTS["unfiltered"]
        return variant
    raise ValueError(
        "Unknown mPR variant "
        f"{variant!r}. Use one of {list(PUBLIC_MPR_VARIANTS.keys())}."
    )


def _normalize_mpr_variants(variants):
    """Normalize public mPR variant names to internal storage keys."""
    if variants is None:
        raw_variants = ("all",)
    elif isinstance(variants, str):
        raw_variants = (variants,)
    else:
        try:
            raw_variants = tuple(variants)
        except TypeError:
            raw_variants = (variants,)

    out = []
    for variant in raw_variants:
        if variant == "all":
            out.extend(PUBLIC_MPR_VARIANTS.values())
        else:
            out.append(_normalize_mpr_variant(variant))

    # Preserve user order while removing duplicates.
    return tuple(dict.fromkeys(out))


def _legacy_filter_to_variant(filter_key, default=None):
    """Map old filter-key names to public variant names."""
    if filter_key is None:
        return default if default is not None else "all"
    mapping = {
        "all": "unfiltered",
        "no_mtRibo_ETCI": "without_mt_ribo_etci",
        "no_small_highAUPRC": "without_small_high_auprc",
    }
    return mapping.get(filter_key, filter_key)


def _legacy_filters_to_variants(show_filters):
    """Map old show_filters values to public variant names."""
    if show_filters is None:
        return "all"
    if isinstance(show_filters, str):
        return _legacy_filter_to_variant(show_filters)
    try:
        return tuple(_legacy_filter_to_variant(item) for item in show_filters)
    except TypeError:
        return (_legacy_filter_to_variant(show_filters),)


def _normalize_show_filters(show_filters):
    """Backward-compatible normalizer for old internal filter keys."""
    return _normalize_mpr_variants(_legacy_filters_to_variants(show_filters))

def plot_mpr_true_positive_curve(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.8,
    variants="unfiltered",
):
    """
    Plot mPR true-positive vs precision curves for multiple datasets.
    
    Can auto-detect datasets or use provided dataset names.
    Each dataset gets one color, each filter type gets one line style.
    Two legends: one for datasets (colors), one for filters (line styles).
    
    Parameters
    ----------
    dataset_names : list of str, optional
        Names of datasets to plot. If None, auto-detects available datasets.
    colors : list of str, optional
        Colors for each dataset. If None, uses default palette.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    save : bool
        Whether to save the figure
    outname : str, optional
        Output filename. If None, auto-generated.
    linewidth : float
        Line width for all curves
    variants : str or iterable of str
        Which mPR variants to show. Use "unfiltered",
        "without_mt_ribo_etci", "without_small_high_auprc", or "all".
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    config = dload("config")
    plot_config = config["plotting"]
    input_colors = dload("input", "colors")

    variant_keys = _normalize_mpr_variants(variants)
    
    # Sanitize color keys
    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    # Auto-detect datasets if none provided
    if dataset_names is None:
        # Get all available MPR datasets from the storage
        mpr_data_dict = dload("mpr")
        if isinstance(mpr_data_dict, dict) and mpr_data_dict:
            dataset_names = list(mpr_data_dict.keys())
        else:
            dataset_names = []
        
        if not dataset_names:
            log.warning("No mPR datasets found. Make sure to run mpr_prepare() first.")
            return None
    
    # Determine colors
    if colors is None:
        cmap_name = config.get("color_map", "tab10")
        try:
            cmap = get_cmap(cmap_name)
        except ValueError:
            cmap = get_cmap("tab10")

        num_datasets = len(dataset_names)
        if num_datasets <= 10 and cmap_name == "tab10":
            default_colors = [cmap(i) for i in range(num_datasets)]
        else:
            default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

        # Assign colors strictly matching the sorted dataset order
        final_colors = []
        for i, dataset in enumerate(dataset_names):
            color = input_colors.get(dataset) if input_colors else None
            # fallback to sanitized lookup just in case dataset_name is already sanitized but key isn't (or vice versa though we sanitized keys above)
            if color is None:
                 color = default_colors[i]
            final_colors.append(color)
        colors = final_colors
    
    if ax is None:
        # Increase width slightly
        fig, ax = plt.subplots(figsize=(6, 4))
        # Reserve space for legend on right
        plt.subplots_adjust(right=0.7)
    else:
        fig = ax.figure
    
    xmax = 0.0
    
    # Plot each dataset
    for i, name in enumerate(dataset_names):
        mpr = dload("mpr", name)
        if mpr is None:
            log.warning(f"mPR data for '{name}' not found, skipping.")
            continue
        
        # Check if mPR data has expected structure
        if "tp_curves" not in mpr:
            log.warning(f"mPR data for '{name}' missing 'tp_curves', skipping.")
            continue
            
        tp_curves = mpr["tp_curves"]
        color = colors[i % len(colors)]
        
        for variant_key in variant_keys:
            if variant_key not in tp_curves:
                continue
            
            data = tp_curves[variant_key]
            if not isinstance(data, dict) or "tp" not in data or "precision" not in data:
                log.warning(f"Invalid tp_curves data structure for '{name}' variant '{variant_key}', skipping.")
                continue
                
            tp = np.asarray(data["tp"], dtype=float)
            prec = np.asarray(data["precision"], dtype=float)
            
            mask = np.isfinite(tp) & (tp > 0) & np.isfinite(prec) & (prec > 0)
            if not mask.any():
                continue
            
            tp_plot = tp[mask]
            prec_plot = prec[mask]
            xmax = max(xmax, float(tp_plot.max()))
            
            style = MPR_VARIANT_STYLES.get(variant_key, {})
            ax.plot(
                tp_plot, 
                prec_plot, 
                color=color,
                linestyle=style.get("linestyle", "-"),
                linewidth=linewidth,
            )
    
    # Configure axes
    ax.set_xlabel("Number of true positives")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)
    
    if xmax > 0:
        ax.set_xscale("log")
        if xmax > 10:
            ax.set_xlim(10, xmax * 1.05)
            logmin = 1
        else:
            ax.set_xlim(1, xmax * 1.05)
            logmin = 0
        
        logmax = int(np.ceil(np.log10(xmax)))
        logmax = max(logmax, logmin)
        xticks = [10 ** k for k in range(logmin, logmax + 1)]
        ax.set_xticks(xticks)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create vertically stacked legends
    _add_vertical_legend(ax, dataset_names, colors, variant_keys, linewidth)
    
    # Save
    if save:
        output_type = plot_config.get("output_type", "pdf")
        if outname is None:
            outname = f"mpr_tp_multi.{output_type}"
        
        # Check if outname is just a filename or a full path
        outpath = Path(outname)
        if len(outpath.parts) == 1:
             # Just a filename, prepend configured output folder
             outpath = Path(config["output_folder"]) / outname

        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", format=output_type)
    
    return ax


def plot_mpr_tp_multi(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.8,
    show_filters=("all", "no_mtRibo_ETCI", "no_small_highAUPRC"),
):
    """Backward-compatible wrapper for plot_mpr_true_positive_curve()."""
    return plot_mpr_true_positive_curve(
        dataset_names=dataset_names,
        colors=colors,
        ax=ax,
        save=save,
        outname=outname,
        linewidth=linewidth,
        variants=_legacy_filters_to_variants(show_filters),
    )

def plot_mpr_complex_coverage_curve(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.8,
    variants="unfiltered",
    show_markers="auto",
    marker_size=20,
):
    """
    Plot mPR complex-coverage vs precision curves for multiple datasets.
    
    Can auto-detect datasets or use provided dataset names.
    Each dataset gets one color, each filter type gets one line style.
    Two legends: one for datasets (colors), one for filters (line styles).
    
    Parameters
    ----------
    dataset_names : list of str, optional
        Names of datasets to plot. If None, auto-detects available datasets.
    colors : list of str, optional
        Colors for each dataset. If None, uses default palette.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    save : bool
        Whether to save the figure
    outname : str, optional
        Output filename. If None, auto-generated.
    linewidth : float
        Line width for all curves
    variants : str or iterable of str
        Which mPR variants to show. Use "unfiltered",
        "without_mt_ribo_etci", "without_small_high_auprc", or "all".
    show_markers : bool or "auto"
        If True, draw markers on curves to make short curves visible.
        If "auto" (default), markers are drawn only for curves with <= 10 points.
    marker_size : int
        Scatter marker size (points^2) when markers are shown.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    config = dload("config")
    plot_config = config["plotting"]
    input_colors = dload("input", "colors")

    variant_keys = _normalize_mpr_variants(variants)
    
    # Sanitize color keys
    if input_colors:
        input_colors = {_sanitize(k): v for k, v in input_colors.items()}

    # Auto-detect datasets if none provided
    if dataset_names is None:
        # Get all available MPR datasets from the storage
        mpr_data_dict = dload("mpr")
        if isinstance(mpr_data_dict, dict) and mpr_data_dict:
            dataset_names = list(mpr_data_dict.keys())
        else:
            dataset_names = []
        
        if not dataset_names:
            log.warning("No mPR datasets found. Make sure to run mpr_prepare() first.")
            return None
    
    # Determine colors
    if colors is None:
        cmap_name = config.get("color_map", "tab10")
        try:
            cmap = get_cmap(cmap_name)
        except ValueError:
            cmap = get_cmap("tab10")

        num_datasets = len(dataset_names)
        if num_datasets <= 10 and cmap_name == "tab10":
            default_colors = [cmap(i) for i in range(num_datasets)]
        else:
            default_colors = [cmap(float(i) / max(num_datasets - 1, 1)) for i in range(num_datasets)]

        # Assign colors strictly matching the sorted dataset order
        final_colors = []
        for i, dataset in enumerate(dataset_names):
            color = input_colors.get(dataset) if input_colors else None
            # fallback to sanitized lookup just in case dataset_name is already sanitized but key isn't (or vice versa though we sanitized keys above)
            if color is None:
                 color = default_colors[i]
            final_colors.append(color)
        colors = final_colors
    
    if ax is None:
        # Increase width slightly
        fig, ax = plt.subplots(figsize=(6, 4))
        # Reserve space for legend on right
        plt.subplots_adjust(right=0.7)
    else:
        fig = ax.figure
    
    # First pass: determine max coverage across all datasets/filters for adaptive x-axis
    max_cov_global = 0
    _mpr_cache = {}
    for i, name in enumerate(dataset_names):
        mpr = dload("mpr", name)
        _mpr_cache[name] = mpr
        if mpr is not None:
            for variant_key in variant_keys:
                arr = mpr["coverage_curves"].get(variant_key)
                if arr is not None:
                    max_cov_global = max(max_cov_global, float(np.asarray(arr).max()))

    # Build adaptive x-axis limits and ticks
    import math
    if max_cov_global <= 200:
        # Original fixed range — keeps CORUM plots identical to before
        x_max_plot = 200
        tick_positions = [1, 2, 20, 200]
        tick_labels = ["0", "2", "20", "200"]
    else:
        # Round up to the next power of 10 so the max bar has breathing room
        x_max_plot = 10 ** math.ceil(math.log10(max_cov_global + 1))
        tick_positions = [1, 2]
        v = 10
        while v <= x_max_plot:
            tick_positions.append(v)
            v *= 10
        tick_labels = ["0"] + [str(t) for t in tick_positions[1:]]

    # Plot each dataset
    for i, name in enumerate(dataset_names):
        mpr = _mpr_cache[name]
        if mpr is None:
            log.warning(f"mPR data for '{name}' not found, skipping.")
            continue

        precision_cutoffs = np.asarray(mpr["precision_cutoffs"], dtype=float)
        coverage = mpr["coverage_curves"]
        color = colors[i % len(colors)]

        for variant_key in variant_keys:
            if variant_key not in coverage:
                continue

            cov = np.asarray(coverage[variant_key], dtype=float)

            # Keep only positive coverage within the visible x range
            mask = (cov > 0) & (cov <= x_max_plot)
            if not mask.any():
                continue

            cov_plot = cov[mask]
            prec_plot = precision_cutoffs[mask]

            style = MPR_VARIANT_STYLES.get(variant_key, {})

            # Decide marker visibility
            if show_markers == "auto":
                use_markers = (cov_plot.size <= 10)
            else:
                use_markers = bool(show_markers)

            if cov_plot.size == 1:
                # A single point is effectively invisible as a line; draw a marker.
                ax.scatter(cov_plot, prec_plot, color=color, s=marker_size, zorder=3)
            else:
                ax.plot(
                    cov_plot,
                    prec_plot,
                    color=color,
                    linestyle=style.get("linestyle", "-"),
                    linewidth=linewidth,
                    marker=("o" if use_markers else None),
                    markersize=(3 if use_markers else None),
                )

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlim(1, x_max_plot)
    ax.set_xlabel("# complexes")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.05)

    # Adaptive x-ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create vertically stacked legends
    _add_vertical_legend(ax, dataset_names, colors, variant_keys, linewidth)
    
    # Save
    if save:
        output_type = plot_config.get("output_type", "pdf")
        if outname is None:
            outname = f"mpr_complexes_multi.{output_type}"
        
        # Check if outname is just a filename or a full path
        outpath = Path(outname)
        if len(outpath.parts) == 1:
             # Just a filename, prepend configured output folder
             outpath = Path(config["output_folder"]) / outname

        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight", format=output_type)
    
    return ax


def plot_mpr_complexes_multi(
    dataset_names=None,
    colors=None,
    ax=None,
    save=True,
    outname=None,
    linewidth=1.8,
    show_filters=("all", "no_mtRibo_ETCI", "no_small_highAUPRC"),
    show_markers="auto",
    marker_size=20,
):
    """Backward-compatible wrapper for plot_mpr_complex_coverage_curve()."""
    return plot_mpr_complex_coverage_curve(
        dataset_names=dataset_names,
        colors=colors,
        ax=ax,
        save=save,
        outname=outname,
        linewidth=linewidth,
        variants=_legacy_filters_to_variants(show_filters),
        show_markers=show_markers,
        marker_size=marker_size,
    )


def plot_mpr_summary(
    dataset_names=None,
    colors=None,
    variants="unfiltered",
    save=True,
    linewidth=1.8,
    show_markers="auto",
    marker_size=20,
    auc_variant=None,
):
    """Generate the standard mPR summary plots and return complex AUC values."""
    plot_mpr_true_positive_curve(
        dataset_names=dataset_names,
        colors=colors,
        save=save,
        linewidth=linewidth,
        variants=variants,
    )
    plot_mpr_complex_coverage_curve(
        dataset_names=dataset_names,
        colors=colors,
        save=save,
        linewidth=linewidth,
        variants=variants,
        show_markers=show_markers,
        marker_size=marker_size,
    )

    if auc_variant is None:
        variant_keys = _normalize_mpr_variants(variants)
        auc_variant = INTERNAL_MPR_VARIANTS.get(variant_keys[0], "unfiltered")

    return plot_mpr_complex_auc_scores(variant=auc_variant, save=save)

def _add_vertical_legend(ax, dataset_names, colors, variant_keys, linewidth):
    """
    Add vertically stacked legends: Dataset on top, mPR variant below.
    """
    variant_keys = _normalize_show_filters(variant_keys)
    # Legend 1: Datasets (colors) - solid lines
    dataset_handles = []
    for i, name in enumerate(dataset_names):
        color = colors[i % len(colors)]
        handle = Line2D([0], [0], color=color, linewidth=linewidth, linestyle="-")
        dataset_handles.append(handle)
    
    # Legend 2: mPR variants (line styles) - black lines
    variant_handles = []
    variant_labels = []
    for variant_key in variant_keys:
        style = MPR_VARIANT_STYLES.get(variant_key, {})
        handle = Line2D(
            [0], [0], 
            color="black", 
            linewidth=linewidth, 
            linestyle=style.get("linestyle", "-")
        )
        variant_handles.append(handle)
        variant_labels.append(style.get("label", variant_key))
    
    # Position legends vertically with proper alignment
    # Dataset legend on upper right
    legend1 = ax.legend(
        dataset_handles, 
        dataset_names, 
        loc="upper left",
        frameon=False,
        title="Dataset",
        fontsize=7,
        title_fontsize=8,
        bbox_to_anchor=(1.05, 1.0)
    )
    ax.add_artist(legend1)
    
    # Filter legend below the dataset legend, aligned properly without title
    legend2 = ax.legend(
        variant_handles,
        variant_labels,
        loc="upper left",
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(1.05, 1.0 - len(dataset_names) * 0.06 - 0.1)
    )

def _add_dual_legend(ax, dataset_names, colors, variant_keys, linewidth):
    """
    Add two legends: one for datasets (colors), one for mPR variants (line styles).
    """
    variant_keys = _normalize_show_filters(variant_keys)
    # Legend 1: Datasets (colors) - solid lines
    dataset_handles = []
    for i, name in enumerate(dataset_names):
        color = colors[i % len(colors)]
        handle = Line2D([0], [0], color=color, linewidth=linewidth, linestyle="-")
        dataset_handles.append(handle)
    
    # Legend 2: mPR variants (line styles) - black lines
    variant_handles = []
    variant_labels = []
    for variant_key in variant_keys:
        style = MPR_VARIANT_STYLES.get(variant_key, {})
        handle = Line2D(
            [0], [0], 
            color="black", 
            linewidth=linewidth, 
            linestyle=style.get("linestyle", "-")
        )
        variant_handles.append(handle)
        variant_labels.append(style.get("label", variant_key))
    
    # Position legends
    # Dataset legend on upper right
    legend1 = ax.legend(
        dataset_handles, 
        dataset_names, 
        loc="upper right",
        frameon=False,
        title="Dataset",
        fontsize=7,
        title_fontsize=8,
    )
    ax.add_artist(legend1)
    
    # Filter legend on lower left or right depending on plot type
    legend2 = ax.legend(
        variant_handles,
        variant_labels,
        loc="lower left",
        frameon=False,
        title="Variant",
        fontsize=7,
        title_fontsize=8,
    )

