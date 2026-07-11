"""Extract final training metrics for each SAE from the sae-system logs.

Log format: one line per logging step of alternating ``name value`` pairs, e.g.
``total_loss 0.17 dead_features 0 nrmse 0.42 explained_variance 0.82``.
The attention logs predate the richer format and carry only
``total_loss``/``dead_features``.

Run from the repo root:
    PYTHONPATH=src python -m eval.sae.training_logs [--log-root DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.io import write_csv, write_json

# Dev-machine location of the SAE training project (read-only); override with --log-root.
DEFAULT_LOG_ROOT = Path("X:/Projects/AIs/Turing/sae-system")
KIND_LOG_DIRS = {"attn": "log", "mlp": "log-sae-mlp", "resid": "log-sae-resid"}
METRIC_FIELDS = ["total_loss", "dead_features", "nrmse", "explained_variance"]
TABLE_FIELDS = ["kind", "layer", "log_steps", *METRIC_FIELDS]


def parse_log_line(line: str) -> dict[str, float]:
    parts = line.split()
    return {parts[i]: float(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}


def collect_training_log_rows(log_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, dirname in KIND_LOG_DIRS.items():
        log_dir = log_root / dirname
        if not log_dir.exists():
            raise FileNotFoundError(f"SAE training log directory not found: {log_dir}")
        for path in sorted(log_dir.glob("log_sae_layer_*.txt")):
            layer = int(path.stem.rsplit("_", 1)[-1])
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            final = parse_log_line(lines[-1])
            row: dict[str, object] = {"kind": kind, "layer": layer, "log_steps": len(lines)}
            for field in METRIC_FIELDS:
                row[field] = final.get(field, "")
            rows.append(row)
    rows.sort(key=lambda row: (str(row["kind"]), int(row["layer"])))  # type: ignore[arg-type]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract final SAE training metrics.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="sae-system directory.")
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/sae-eval"), help="Output directory.")
    args = parser.parse_args(argv)

    rows = collect_training_log_rows(args.log_root)
    table_path = write_csv(args.out / "tables" / "sae-training-logs.csv", rows, TABLE_FIELDS)
    summary_path = write_json(
        args.out / "summaries" / "sae-training-logs.json",
        {"log_root": str(args.log_root), "sae_count": len(rows), "rows": rows},
    )
    print("wrote", table_path)
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
