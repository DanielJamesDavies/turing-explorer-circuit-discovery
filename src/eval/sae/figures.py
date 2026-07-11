"""Render the SAE quality appendix figure from the eval CSVs.

Panels (each a line per component kind across layers 0-11):
  1. Explained variance -- from the reconstruction eval if available, else the
     training logs (which lack attention EV).
  2. Alive-latent fraction -- from the latent density table.
  3. CE recovered (%) -- only if the reconstruction eval ran with --ce.

Run from the repo root:
    PYTHONPATH=src python -m eval.sae.figures [--data DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analysis.style import (
    SERIES3,
    configure_matplotlib,
    panel_figsize,
    save_figure,
    styled_legend,
)

KINDS = ("attn", "mlp", "resid")
KIND_LABELS = {"attn": "Attention", "mlp": "MLP", "resid": "Residual"}
KIND_COLORS = dict(zip(KINDS, SERIES3))


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], field: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {kind: [] for kind in KINDS}
    for row in rows:
        value = row.get(field, "")
        if row.get("kind") in series and value not in ("", None):
            series[row["kind"]].append((int(row["layer"]), float(value)))
    for values in series.values():
        values.sort()
    return series


def _plot_panel(axis, series: dict[str, list[tuple[int, float]]], title: str, ylabel: str) -> bool:
    drew = False
    for kind in KINDS:
        points = series[kind]
        if not points:
            continue
        axis.plot(
            [layer for layer, _ in points],
            [value for _, value in points],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=KIND_COLORS[kind],
            label=KIND_LABELS[kind],
        )
        drew = True
    axis.set_title(title)
    axis.set_xlabel("Layer")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(0, 12))
    return drew


def render_sae_quality_figure(data_dir: Path, out_dir: Path) -> Path:
    recon_rows = _read_rows(data_dir / "tables" / "sae-reconstruction-eval.csv")
    log_rows = _read_rows(data_dir / "tables" / "sae-training-logs.csv")
    density_rows = _read_rows(data_dir / "tables" / "sae-latent-density.csv")

    ev_series = _series(recon_rows, "explained_variance") if recon_rows else _series(log_rows, "explained_variance")
    ev_source = "reconstruction eval" if recon_rows else "training logs"
    alive_series = _series(density_rows, "alive_fraction")
    ce_series = _series(recon_rows, "ce_recovered_pct") if recon_rows else {kind: [] for kind in KINDS}

    panels = 2 + (1 if any(ce_series[kind] for kind in KINDS) else 0)
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, panels, figsize=panel_figsize(1, panels))

    # Full natural scales: auto-zoomed axes exaggerate small differences and
    # make healthy SAEs look degraded.
    _plot_panel(axes[0], ev_series, "Reconstruction Quality", "Explained variance")
    axes[0].set_ylim(0.0, 1.04)
    styled_legend(axes[0], loc="lower left")
    _plot_panel(axes[1], alive_series, "Alive Latents", "Fraction of latents ever active")
    axes[1].set_ylim(0.0, 1.04)
    styled_legend(axes[1], loc="lower right")
    if panels == 3:
        _plot_panel(axes[2], ce_series, "Loss Recovered", "CE recovered (%)")
        axes[2].set_ylim(0.0, 104.0)
        styled_legend(axes[2], loc="lower left")

    print(f"[sae-figures] explained variance sourced from: {ev_source}")
    return save_figure(fig, out_dir / "figures" / "sae-quality.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the SAE quality figure.")
    parser.add_argument("--data", type=Path, default=Path("analysis-restyled/sae-eval"), help="Directory with the eval tables.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (defaults to --data).")
    args = parser.parse_args(argv)

    out_dir = args.out if args.out is not None else args.data
    path = render_sae_quality_figure(args.data, out_dir)
    print("wrote", path, "and", path.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
