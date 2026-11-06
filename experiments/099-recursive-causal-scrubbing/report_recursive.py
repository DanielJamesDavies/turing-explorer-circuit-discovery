"""Report the exact scope and context-level uncertainty of a recursive run."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def accuracy_interval(cases, seed=20261028):
    # References, not individual draws, are the resampling units. This interval
    # describes this very small selected panel, not arbitrary SAE targets.
    rng = np.random.default_rng(seed)
    samples = []
    for label in (0, 1):
        refs = sorted({r["reference"] for r in cases if r["label"] == label})
        accuracy = np.array([1 - np.mean([r["loss"] for r in cases if r["reference"] == ref]) for ref in refs])
        selection = rng.integers(0, len(refs), size=(4000, len(refs)))
        samples.append(accuracy[selection].mean(1))
    return np.quantile((samples[0] + samples[1]) / 2, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "out/recursive_two_cut")
    args = parser.parse_args()
    m = json.loads((args.run / "manifest.json").read_text())
    rows = json.loads((args.run / "results.json").read_text())
    classes = {str(label): m["labels"].count(label) for label in (0, 1)}
    lines = ["# Recursive two-cut diagnostic", "",
             f"Target: **{m['target']}**. Read position **{m['position']}**, selected from training firing counts only.", "",
             f"Donor dataset: {m['dataset_n']} held-out contexts ({classes['1']} firing, {classes['0']} silent). "
             f"Evaluated references: {len(m['base_ids'])}; {m['draws']} recursive draws per reference.", "",
             "**Hypothesis:** the WCM-selected residual groups at layers 0 and 1 each transmit the binary target-firing "
             "state. The native computations before, between, and after these boundaries are composite graph nodes. "
             "This is a coarse, teacher-labelled representational hypothesis, not a human semantic explanation and "
             "not recursive validation of every node in the full WCM circuit.", "",
             f"Selected residual group sizes: {m['site_counts']}. All other residual information, including SAE errors, "
             "is an explicitly resampled remainder. The natural-strength and fitted-gain arms both decompose the "
             "original model exactly as `r = phi_alpha(r) + (r - phi_alpha(r))`. The fitted-gain remainder therefore "
             "includes the compensating gain term.", "",
             "| Arm | Balanced firing agreement | Context-bootstrap 95% interval | Silent-reference errors | Firing-reference errors | Natural graph max absolute error |",
             "|---|---:|---|---:|---:|---:|"]
    summary = []
    for row in rows:
        acc = 1 - row["balanced_disagreement"]
        lo, hi = accuracy_interval(row["cases"])
        lines.append(f"| {row['arm']} | {acc:.1%} | [{lo:.1%}, {hi:.1%}] | "
                     f"{row['class_disagreement']['0']:.1%} | {row['class_disagreement']['1']:.1%} | "
                     f"{row['natural_max_abs_delta']:.3g} |")
        summary.append(dict(arm=row["arm"], balanced_accuracy=acc, context_bootstrap=[lo, hi],
                            noninput_support_min=min(t["noninput_support_min"] for t in row["coverage"]),
                            self_draw_fraction=float(np.mean([t["noninput_self_fraction"] for t in row["coverage"]]))))
    lines += ["", "## Checks and limits", "",
              "- Every unintervened graph was checked against the full inference path, using the same one-example numerical context. "
              "No clean firing labels changed.",
              "- Recursive output was replayed through an explicitly expanded expression tree for the first draw of each reference and arm.",
              "- The full-state control is required to preserve firing. The empty and active random controls show how informative the test is.",
              "- Conditional samples match the exact binary interpretation value; no nearest-neighbour matching or forced non-self sampling is used.",
              "- Input-node matches are necessarily singleton IDs; non-input donor diversity is reported separately in report.json.",
              "- Repeated donor draws are not independent examples. The interval resamples references within firing classes and is very uncertain with so few positive references.",
              "- This is a development case. Any cut-restoration or full-SAE additions selected after seeing its result are exploratory diagnostics.", ""]
    (args.run / "report.md").write_text("\n".join(lines))
    (args.run / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
