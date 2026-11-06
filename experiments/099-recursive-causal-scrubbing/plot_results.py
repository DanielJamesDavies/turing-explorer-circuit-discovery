"""Standalone scientific figures from saved results; no model access."""
import argparse
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=HERE / "out/first3")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((args.run / "manifest.json").read_text())
    rows = [json.loads(p.read_text()) for p in (args.run / "targets").glob("*.json")]
    rows = sorted(rows, key=lambda r: manifest["targets"].index(r["target"]))
    if not rows:
        raise ValueError("no completed targets")
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.52), 4.2), constrained_layout=True)
    x = np.arange(len(rows))
    ax.axhspan(0.8, 1.5, color="#e6e9ed", alpha=0.6, label="Production faithfulness band")
    ax.axhline(0, color="#707070", lw=0.7)
    ax.scatter(x - 0.23, [r["production"]["wcm"].get("freeN_topk_tk", np.nan) for r in rows],
               marker="s", color="#20252b", s=30, label="WCM: contrast-mean background")
    for offset, variant, color, marker, label in [
        (-0.08, "wcm", "#b33a3a", "o", "WCM: contrast donors"),
        (0.08, "wcm_pinned", "#27658c", "D", "Pinned WCM: contrast donors"),
        (0.23, "ig_a1", "#6c4387", "^", "Adapted IG: contrast donors")]:
        for j, r in enumerate(rows):
            vals = [d["recovery"] for d in r["donor_results"]
                    if d["background"] == "C" and d["variant"] == variant and d["recovery"] is not None]
            if not vals:
                continue
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            ax.errorbar(j + offset, med, yerr=[[med - q1], [q3 - med]], fmt=marker, color=color,
                        markersize=5, capsize=2, label=label if j == 0 else None)
    ax.set_xticks(x, [r["target"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("Target recovery (ratio of anchor means)")
    ax.set_title(f"Clean WCM pilot: mean backgrounds versus donor backgrounds (n={len(rows)})", fontsize=11)
    ax.legend(fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.015, "Donor points: median and interquartile range across draws. Pinned values are a diagnostic. "
             "This is background resampling, not recursive scrubbing.", fontsize=7)
    fig.savefig(args.run / "background_recovery.png", dpi=180, bbox_inches="tight")
    fig.savefig(args.run / "background_recovery.pdf", bbox_inches="tight")
    print(args.run / "background_recovery.png")


if __name__ == "__main__":
    main()
