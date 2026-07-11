"""Alive fractions and firing-rate densities per SAE from a run's latent_stats.pt.

Reads the collection pass's latent statistics checkpoint (read-only) and reports,
for each of the 36 SAEs: the fraction of latents that ever fired, firing-rate
quantiles, and a log10 firing-rate histogram for density plots.

Run from the repo root:
    PYTHONPATH=src python -m eval.sae.latent_density --run-root DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from analysis.io import write_csv, write_json

KINDS = ("attn", "mlp", "resid")
HIST_BINS = 40
HIST_RANGE = (-8.0, 0.0)  # log10 firing rate
TABLE_FIELDS = [
    "kind",
    "layer",
    "component",
    "tokens_seen",
    "alive_fraction",
    "median_firing_rate",
    "p10_firing_rate",
    "p90_firing_rate",
]


def load_latent_stats(run_root: Path) -> tuple[torch.Tensor, dict[int, int]]:
    path = run_root / "latent_stats.pt"
    if not path.exists():
        raise FileNotFoundError(f"latent stats not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    active_count = checkpoint["active_count"].to(torch.float64)
    component_steps = {int(comp): int(steps) for comp, steps in checkpoint["component_steps"].items()}
    return active_count, component_steps


def compute_density_rows(
    active_count: torch.Tensor,
    component_steps: dict[int, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    n_components = active_count.shape[0]
    n_kinds = len(KINDS)
    rows: list[dict[str, object]] = []
    histograms: dict[str, object] = {"bins": HIST_BINS, "range": list(HIST_RANGE), "by_component": {}}

    for component in range(n_components):
        layer, kind_idx = divmod(component, n_kinds)
        tokens_seen = component_steps.get(component, 0)
        counts = active_count[component]
        alive = counts > 0
        alive_fraction = float(alive.to(torch.float64).mean())
        if tokens_seen > 0 and bool(alive.any()):
            rates = counts[alive] / float(tokens_seen)
            quantiles = torch.quantile(rates, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
            log_rates = torch.log10(rates.clamp(min=10.0 ** HIST_RANGE[0]))
            hist = torch.histc(log_rates, bins=HIST_BINS, min=HIST_RANGE[0], max=HIST_RANGE[1])
            histograms["by_component"][str(component)] = [int(v) for v in hist.tolist()]
            p10, p50, p90 = (float(q) for q in quantiles)
        else:
            p10 = p50 = p90 = math.nan
        rows.append(
            {
                "kind": KINDS[kind_idx],
                "layer": layer,
                "component": component,
                "tokens_seen": tokens_seen,
                "alive_fraction": alive_fraction,
                "median_firing_rate": p50,
                "p10_firing_rate": p10,
                "p90_firing_rate": p90,
            }
        )
    return rows, histograms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute SAE latent density statistics.")
    parser.add_argument("--run-root", type=Path, required=True, help="Pipeline run root with latent_stats.pt.")
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/sae-eval"), help="Output directory.")
    args = parser.parse_args(argv)

    active_count, component_steps = load_latent_stats(args.run_root)
    rows, histograms = compute_density_rows(active_count, component_steps)
    table_path = write_csv(args.out / "tables" / "sae-latent-density.csv", rows, TABLE_FIELDS)
    summary_path = write_json(
        args.out / "summaries" / "sae-latent-density.json",
        {
            "run_root": str(args.run_root),
            "d_sae": int(active_count.shape[1]),
            "rows": rows,
            "log10_firing_rate_histograms": histograms,
        },
    )
    print("wrote", table_path)
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
