"""Recreate the TuringLLM training loss curve in the paper figure style.

Reads a turing-llm training log (lines of ``step split value``) and renders
train/val cross-entropy on a log scale with the GPT-2 reference level.

Run from the repo root:
    PYTHONPATH=src python -m debug.paper_loss_curve --log PATH [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.style import (
    BLUE,
    BLUE_LIGHT,
    FIGSIZE_WIDE,
    INK_MUTED,
    configure_matplotlib,
    save_figure,
    styled_legend,
)

GPT2_VAL_LOSS = 3.29  # OpenAI GPT-2 (124M) checkpoint val loss on this corpus setup


def load_log(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        step, split, value = parts
        if split == "train":
            train.append((int(step), float(value)))
        elif split == "val":
            val.append((int(step), float(value)))
    return train, val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the TuringLLM loss curve.")
    parser.add_argument("--log", type=Path, required=True, help="Training log file.")
    parser.add_argument("--out", type=Path, default=Path("paper/figures"), help="Output directory.")
    args = parser.parse_args(argv)

    train, val = load_log(args.log)
    if not train:
        raise SystemExit(f"no train entries found in {args.log}")

    plt = configure_matplotlib()
    fig, axis = plt.subplots(figsize=FIGSIZE_WIDE)
    axis.plot([s for s, _ in train], [v for _, v in train], color=BLUE, linewidth=1.6, label="Train loss")
    axis.plot(
        [s for s, _ in val],
        [v for _, v in val],
        color=BLUE_LIGHT,
        linewidth=1.8,
        linestyle="--",
        label="Validation loss",
    )
    axis.axhline(GPT2_VAL_LOSS, color=INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)), label="GPT-2 (124M) val loss")
    axis.set_yscale("log")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss (log scale)")
    axis.set_title("TuringLLM Training and Validation Loss")
    styled_legend(axis, loc="upper right")

    args.out.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, args.out / "loss-curve.png")
    print("wrote", path, "and", path.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
