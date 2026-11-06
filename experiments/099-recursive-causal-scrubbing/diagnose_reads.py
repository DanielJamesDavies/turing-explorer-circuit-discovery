"""Separate loss of encoder-direction signal from post-Top-K firing loss.

Uses stored outputs only. Pre means ReLU(pre-activation), not signed preactivation.
"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import torch

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "out/panel18")
    parser.add_argument("--contexts", type=Path, default=HERE / "out/records/ctx_v2")
    args = parser.parse_args()
    rows = []
    lines = ["# Contrast-donor failure: encoder signal and Top-K firing", "",
             "Pre denotes ReLU(pre-activation). Ratios are at held-out activating anchors. "
             "Relative RMSE is divided by the clean anchor RMS, not by each context's activation. "
             "Raw pre ratios retain the background baseline; normalized pre ratios subtract it.", "",
             "| Target | Post-Top-K recovery | Raw pre / clean pre | Empty pre / clean pre | Normalized pre recovery | Pre relative RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for path in sorted((args.run / "targets").glob("*.json")):
        result = json.loads(path.read_text())
        rec = torch.load(args.contexts / (result["target"] + ".pt"), map_location="cpu", weights_only=False)[result["target"]]
        held = rec["strong"]["held"]
        anchors = rec["strong"]["arg"][held].numpy().astype(int)
        rs = [r for r in result["donor_results"] if r["background"] == "C"]
        def at(row, natural=False):
            field = "natural_raw" if natural else "raw"
            values = np.asarray(row["pre_relu"][field]).reshape(-1, 64)
            return values[np.arange(len(anchors)), anchors]
        draws = []
        for row in rs:
            if row["variant"] != "wcm":
                continue
            empty = next(r for r in rs if r["variant"] == "empty" and r["draw"] == row["draw"])
            actual, clean, baseline = at(row), at(row, True), at(empty)
            a, n, e = actual.mean(), clean.mean(), baseline.mean()
            draws.append(dict(post=row["recovery"], raw_pre_ratio=float(a / n) if n else None,
                              empty_pre_ratio=float(e / n) if n else None,
                              normalized_pre=float((a - e) / (n - e)) if n - e > 1e-9 else None,
                              pre_relative_rmse=float(np.sqrt(np.mean((actual - clean) ** 2) / np.mean(clean ** 2)))))
        summary = {name: statistics.median([d[name] for d in draws if d[name] is not None])
                   if any(d[name] is not None for d in draws) else None for name in draws[0]}
        rows.append(dict(target=result["target"], median=summary, draws=draws))
        fmt = lambda v: "undefined" if v is None else f"{v:.3f}"
        lines.append("| " + result["target"] + " | " + " | ".join(fmt(summary[k]) for k in
                     ("post", "raw_pre_ratio", "empty_pre_ratio", "normalized_pre", "pre_relative_rmse")) + " |")
    lines += ["", "These are exploratory diagnostics. A post/pre difference suggests competition but does not isolate it "
              "from baseline signal or other correlated changes. In particular, a high raw pre ratio can be uninformative "
              "when the empty circuit has a similar pre-activation.", ""]
    (args.run / "read_diagnostics.md").write_text("\n".join(lines))
    (args.run / "read_diagnostics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
