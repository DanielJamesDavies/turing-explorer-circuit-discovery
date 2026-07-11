"""One-command SAE quality report: CPU-only data steps plus the figure.

Parses the sae-system training logs, computes latent density statistics from a
run's latent_stats.pt, and renders the appendix figure. The GPU reconstruction
eval (``eval.sae.reconstruction_eval``) is separate; if its CSV already exists
in the output directory, the figure automatically uses it for explained
variance and CE recovered.

Run from the repo root:
    PYTHONPATH=src python -m eval.sae.report --run-root DIR [--log-root DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.io import write_csv, write_json

from .figures import render_sae_quality_figure
from .latent_density import compute_density_rows, load_latent_stats
from .latent_density import TABLE_FIELDS as DENSITY_FIELDS
from .training_logs import DEFAULT_LOG_ROOT, collect_training_log_rows
from .training_logs import TABLE_FIELDS as LOG_FIELDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the SAE quality report.")
    parser.add_argument("--run-root", type=Path, required=True, help="Pipeline run root with latent_stats.pt.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="sae-system directory.")
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/sae-eval"), help="Output directory.")
    args = parser.parse_args(argv)

    log_rows = collect_training_log_rows(args.log_root)
    write_csv(args.out / "tables" / "sae-training-logs.csv", log_rows, LOG_FIELDS)
    write_json(
        args.out / "summaries" / "sae-training-logs.json",
        {"log_root": str(args.log_root), "sae_count": len(log_rows), "rows": log_rows},
    )
    print(f"[sae-report] training logs parsed: {len(log_rows)} SAEs")

    active_count, component_steps = load_latent_stats(args.run_root)
    density_rows, histograms = compute_density_rows(active_count, component_steps)
    write_csv(args.out / "tables" / "sae-latent-density.csv", density_rows, DENSITY_FIELDS)
    write_json(
        args.out / "summaries" / "sae-latent-density.json",
        {
            "run_root": str(args.run_root),
            "d_sae": int(active_count.shape[1]),
            "rows": density_rows,
            "log10_firing_rate_histograms": histograms,
        },
    )
    print(f"[sae-report] latent density computed: {len(density_rows)} SAEs")

    figure_path = render_sae_quality_figure(args.out, args.out)
    print("[sae-report] figure:", figure_path, "and", figure_path.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
