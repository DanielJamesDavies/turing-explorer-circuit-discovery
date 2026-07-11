"""Matplotlib styling for generated analysis figures ("Lab Bright" theme).

Single source of truth for figure aesthetics: the categorical palette, ink and
chrome roles, colormap names, figure-size tokens, and shared plotting helpers.
Analysis modules must take colors from here rather than hardcoding hex values.

The look: white ground, baseline-only axes with no tick marks, faint dashed
y-grid, left-aligned bold titles, bright blue-first categorical palette, and
direct value labels on bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

# Categorical palette in fixed slot order ("Punchy", reordered blue->red->teal).
# The ordering is the colorblind-safety mechanism (validated worst adjacent CVD
# deltaE 41.1 on white): assign slots in sequence for however many series a plot
# has, never re-order or skip.
CATEGORICAL = (
    "#0044ff",  # 1 blue
    "#fa1e4e",  # 2 red
    "#0ab5c9",  # 3 teal
    "#12c46a",  # 4 green
    "#b028ff",  # 5 purple
    "#ff3d8b",  # 6 pink
    "#d99400",  # 7 honey
    "#4a3aff",  # 8 indigo
)
BLUE = CATEGORICAL[0]
BLUE_LIGHT = "#8aa4ff"  # de-emphasized companion to BLUE (raw/background series)
SERIES2 = CATEGORICAL[:2]
SERIES3 = CATEGORICAL[:3]

# Ink and chrome roles.
INK = "#20242b"
INK_SECONDARY = "#4c525c"
INK_MUTED = "#6c727c"
GRID = "#e3e5e9"
BASELINE = "#6c727c"
SURFACE = "#ffffff"

# Continuous-magnitude colormaps: one blue->red scale for heatmaps/scales
# (blue at the low end, red at the high end, via a light neutral midpoint), and
# one many-class qualitative map reserved for cluster-identity scatters.
def _blue_red_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "analysis_blue_red", [BLUE, "#eef0f4", CATEGORICAL[1]]
    )


SEQUENTIAL_CMAP = _blue_red_cmap()
CLUSTER_CMAP = "tab20"

# Semantic color assignments shared across figure suites.
METHOD_COLORS = {
    "counterfactual_gradient": CATEGORICAL[0],
    "ablation_gradient": CATEGORICAL[1],
    "hybrid_gradient": CATEGORICAL[2],
}
NEG_MODE_COLORS = {
    "close": CATEGORICAL[0],
    "random": CATEGORICAL[1],
    "distant": CATEGORICAL[2],
}

# Blue ordinal ramp (light -> dark) for ordered categories: hop depth, rarity
# tiers, thresholds. Sample with ordinal_blues(n) so steps stay evenly spaced.
_BLUE_RAMP = (
    "#a3b8ff",
    "#8aa4ff",
    "#7090ff",
    "#557bff",
    "#2f60ff",
    "#0044ff",
    "#003be0",
    "#0031ba",
    "#002894",
)


def ordinal_blues(count: int) -> list[str]:
    """Evenly spaced light-to-dark blues for an ordered category scale."""

    if count <= 0:
        return []
    if count == 1:
        return [BLUE]
    last = len(_BLUE_RAMP) - 1
    return [_BLUE_RAMP[round(index * last / (count - 1))] for index in range(count)]


def tint(color: str, amount: float = 0.55) -> str:
    """Blend a hex color toward white; the light half of a two-tone pair."""

    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    mixed = (round(c + (255 - c) * amount) for c in (r, g, b))
    return "#" + "".join(f"{c:02x}" for c in mixed)


# Figure-size tokens.
FIGSIZE_SINGLE = (7.0, 4.5)
FIGSIZE_WIDE = (10.0, 5.0)
FIGSIZE_SQUARE = (8.0, 7.0)

# Bar geometry: single-series categorical bar width, leaving breathing room.
BAR_WIDTH = 0.62


def grouped_bar_geometry(n_series: int, *, group_width: float = 0.72, bar_gap: float = 0.16) -> tuple[float, list[float]]:
    """Bar width and per-series x-offsets for grouped bars with spacing.

    ``group_width`` is the fraction of each category slot the group occupies
    (the rest separates neighbouring groups); ``bar_gap`` is the fraction of
    each bar slot left empty between bars within a group.
    """

    slot = group_width / n_series
    width = slot * (1.0 - bar_gap)
    offsets = [(index - (n_series - 1) / 2.0) * slot for index in range(n_series)]
    return width, offsets


def panel_figsize(rows: int, cols: int) -> tuple[float, float]:
    """Figure size for a grid of equally weighted panels."""

    return (min(4.8 * cols + 1.2, 18.0), 3.8 * rows + 0.9)


def configure_matplotlib():
    """Configure Matplotlib for deterministic, headless plot generation."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "pdf.fonttype": 42,  # embed TrueType fonts so PDF text stays selectable/sharp
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.prop_cycle": plt.cycler(color=list(CATEGORICAL)),
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.labelsize": 11,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 12.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.2,
            "xtick.color": SURFACE,
            "ytick.color": SURFACE,
            "xtick.labelcolor": INK_MUTED,
            "ytick.labelcolor": INK_MUTED,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": (0, (4, 4)),
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.2,
            "patch.edgecolor": SURFACE,
            "patch.linewidth": 0.6,
            "patch.force_edgecolor": True,
        }
    )
    return plt


def save_figure(fig: Any, path: str | Path) -> Path:
    """Lay out, save, and close a figure; the single exit point for plots.

    Writes the requested raster file plus a vector ``.pdf`` sibling (for the
    paper, where vector figures stay sharp at any zoom and in print).
    """

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(fig, "_suptitle", None) is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def style_suptitle(fig: Any, title: str) -> None:
    """Apply the shared multi-panel figure title style (left-aligned)."""

    fig.suptitle(title, x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)


def round_bars(axis: Any, radius: float = 0.14) -> None:
    """Round only the free (outer) end of each bar column, baseline kept square.

    In a stacked bar this rounds just the outermost segment's top; interior
    segment boundaries stay flush (no mid-stack notch), and the baseline edge
    stays square. ``radius`` is a fraction of each bar's width; the corner
    radius is kept visually circular and capped at the bar height. Call after
    drawing bars (and after any axis-limit changes); not meant for histograms.
    """

    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    renderer = axis.figure.canvas.get_renderer()
    window = axis.get_window_extent(renderer)
    x_lo, x_hi = axis.get_xlim()
    y_lo, y_hi = axis.get_ylim()
    px_per_x = window.width / (x_hi - x_lo)
    px_per_y = window.height / (y_hi - y_lo)

    rects = [
        patch
        for patch in axis.patches
        if isinstance(patch, Rectangle) and patch.get_width() > 0 and patch.get_height() != 0
    ]
    # Group segments sharing a column; only the free-end segment of each stack
    # gets rounded so interior boundaries stay flush.
    columns: dict[float, list[Any]] = {}
    for patch in rects:
        key = round(patch.get_x() + patch.get_width() / 2.0, 6)
        columns.setdefault(key, []).append(patch)
    outer: list[Any] = []
    for patches in columns.values():
        positives = [p for p in patches if p.get_height() > 0]
        negatives = [p for p in patches if p.get_height() < 0]
        if positives:
            outer.append(max(positives, key=lambda p: p.get_y() + p.get_height()))
        if negatives:
            outer.append(min(negatives, key=lambda p: p.get_y() + p.get_height()))

    for patch in outer:
        width = patch.get_width()
        height = patch.get_height()
        rx = radius * width
        ry = rx * px_per_x / px_per_y  # visually circular corner
        if ry > abs(height):
            ry = abs(height)
            rx = ry * px_per_y / px_per_x
        x0 = patch.get_x()
        base = patch.get_y()
        tip = base + height
        direction = 1.0 if height > 0 else -1.0
        vertices = [
            (x0, base),
            (x0 + width, base),
            (x0 + width, tip - direction * ry),
            (x0 + width, tip),
            (x0 + width - rx, tip),
            (x0 + rx, tip),
            (x0, tip),
            (x0, tip - direction * ry),
            (x0, base),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.CLOSEPOLY,
        ]
        rounded = PathPatch(
            MplPath(vertices, codes),
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            linewidth=patch.get_linewidth(),
            zorder=patch.get_zorder(),
        )
        patch.remove()
        axis.add_patch(rounded)


def _handle_color(handle: Any) -> Any:
    if getattr(handle, "patches", None):  # BarContainer
        handle = handle.patches[0]
    if hasattr(handle, "get_facecolor"):
        color = handle.get_facecolor()
    elif hasattr(handle, "get_color"):
        color = handle.get_color()
    else:
        return INK
    if not isinstance(color, (str, tuple)) and hasattr(color, "__len__") and len(color) and hasattr(color[0], "__len__"):
        color = tuple(color[0])
    return color


def styled_legend(axis: Any, **kwargs: Any) -> Any:
    """Draw the axis legend with circular color keys instead of swatch boxes."""

    from matplotlib.lines import Line2D

    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return None
    proxies = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8.5,
            markerfacecolor=_handle_color(handle),
            markeredgecolor="none",
        )
        for handle in handles
    ]
    kwargs.setdefault("handlelength", 1.0)
    kwargs.setdefault("handletextpad", 0.4)
    return axis.legend(proxies, labels, **kwargs)


def styled_boxplot(
    axis: Any,
    data: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    edge: str = "ink",
) -> dict[str, Any]:
    """Draw a boxplot with the shared box/whisker/median styling.

    edge="ink" gives black box borders with muted grey whiskers;
    edge="match" colours each box's border, whiskers, caps, median line,
    mean marker, and outliers with lighter tints of that box's own colour.
    """

    safe_data = [list(values) if len(values) else [0.0] for values in data]
    plot = axis.boxplot(
        safe_data,
        patch_artist=True,
        tick_labels=list(labels),
        widths=0.5,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markersize": 4.0,
            "markerfacecolor": INK,
            "markeredgecolor": "none",
        },
        medianprops={"color": INK, "linewidth": 1.8},
        whiskerprops={"color": INK_MUTED, "linewidth": 1.4},
        capprops={"color": INK_MUTED, "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": INK_MUTED,
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
    )
    from matplotlib.colors import to_rgba

    fill_alpha = 0.3 if edge == "match" else 0.62
    for index, (patch, color) in enumerate(zip(plot["boxes"], colors)):
        patch.set_facecolor(to_rgba(color, fill_alpha))  # translucent fill
        edge_color = tint(color, 0.25) if edge == "match" else INK
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.4 if edge == "match" else 0.9)  # match whisker weight
        if edge == "match":
            for line in (
                plot["whiskers"][2 * index],
                plot["whiskers"][2 * index + 1],
                plot["caps"][2 * index],
                plot["caps"][2 * index + 1],
            ):
                line.set_color(edge_color)
            # Median and mean stay a step deeper than the edges so they
            # read clearly against the translucent fill.
            center_color = tint(color, 0.15)
            plot["medians"][index].set_color(center_color)
            if plot["means"]:
                plot["means"][index].set_markerfacecolor(center_color)
            if plot["fliers"]:
                plot["fliers"][index].set_markerfacecolor(edge_color)
    return plot


def value_labels(axis: Any, bars: Iterable[Any], fmt: str = ".2f") -> None:
    """Write each bar's value above it in ink, offset clear of the bar."""

    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            format(height, fmt),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="medium",
            color=INK,
        )


def annotate_bars(axis: Any, fmt: str = ".2f") -> None:
    """Write every bar's height above it (all patches on the axis)."""

    value_labels(axis, axis.patches, fmt)


def integer_ticks(axis: Any, which: str = "y") -> None:
    """Force integer tick locations on count axes."""

    from matplotlib.ticker import MaxNLocator

    if which in ("x", "both"):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if which in ("y", "both"):
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
