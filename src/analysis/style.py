"""Matplotlib styling for generated analysis figures."""

from __future__ import annotations


def configure_matplotlib():
    """Configure Matplotlib for deterministic, headless plot generation."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 11,
            "legend.frameon": False,
        }
    )
    return plt

