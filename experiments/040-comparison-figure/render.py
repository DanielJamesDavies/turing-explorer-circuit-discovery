"""Render the three-arena comparison figure from points.jsonl.

2 rows x 3 columns: rows = zero-fill faithfulness and necessity (sup);
columns = the three arenas, each self-contained (no cross-arena
comparison). x = circuit size in nodes (log). The acceptance band
[0.8, 1.25] is shaded on the faithfulness row. Faint markers are
per-seed points; bold lines join per-method medians over log-clustered
sizes, so ranking sweeps and lambda sweeps read as frontiers and
single-size methods read as points.

  python render.py     ->  comparison.pdf / comparison.png
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ARENAS = [("gemma-tc", "Gemma-2-2B + GemmaScope transcoders\n(circuit-tracer's shipped scan)"),
          ("llama-tc", "Llama-3.2-1B + TopK skip-transcoders\n(converted weights)"),
          ("turingllm", "TuringLLM + home SAE bank\n(k=128 / 40,960)")]
STYLE = {  # method: (color, linestyle, zorder, label)
    "tri-amp":       ("#1657d6", "-",  9, "tri-amp (ours)"),
    "tri-mask":      ("#6f9df0", "--", 8, "tri-mask (ours)"),
    "ct-direct":     ("#e8862c", "-",  6, "circuit-tracer adjacency ranking"),
    "ct-published":  ("#c22f2f", "-",  6, "circuit-tracer (as published)"),
    "ct-rooted":     ("#8c1a1a", "-",  6, "circuit-tracer, seed-rooted"),
    "sfc":           ("#7b3fb8", "-",  6, "SFC attribution patching"),
    "abl-gradient":  ("#c22f2f", "-",  6, "ablation gradient"),
    "cf-gradient":   ("#e8862c", "-",  6, "cf gradient"),
    "restoration":   ("#b8792c", "-",  6, "restoration"),
    "ge-hier":       ("#7b3fb8", "-",  6, "Ge et al. hierarchical"),
    "sfc+amp":       ("#7b3fb8", ":",  5, "HYBRID: SFC selection + our alpha fit"),
    "ct-rooted+amp": ("#8c1a1a", ":",  5, "HYBRID: ct-rooted selection + our alpha fit"),
    "coact":         ("#7a6652", "-",  4, "co-activation"),
    "coact+amp":     ("#7a6652", ":",  4, "HYBRID: co-activation + our alpha fit"),
    "null":          ("#999999", "-",  3, "random null (fitted alpha)"),
    "support-null":  ("#bbbbbb", "-",  3, "support-matched null"),
}


def log_clusters(pts, ratio=1.7):
    pts = sorted(pts, key=lambda p: p["n"])
    groups, cur = [], [pts[0]]
    for p in pts[1:]:
        if p["n"] <= cur[-1]["n"] * ratio:
            cur.append(p)
        else:
            groups.append(cur); cur = [p]
    groups.append(cur)
    return groups


def med(v):
    v = sorted(v); return v[len(v) // 2]


def main():
    pts = [json.loads(l) for l in open(HERE / "points.jsonl")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey="row")
    for col, (arena, title) in enumerate(ARENAS):
        ap = [p for p in pts if p["arena"] == arena]
        by = defaultdict(list)
        for p in ap:
            by[p["method"]].append(p)
        for row, key in [(0, "f0"), (1, "sup")]:
            ax = axes[row][col]
            if row == 0:
                ax.axhspan(0.8, 1.25, color="#2a9d2a", alpha=0.10, lw=0)
            for m, mp in by.items():
                if m not in STYLE:
                    continue
                c, ls, z, _ = STYLE[m]
                mp = [p for p in mp if p.get(key) is not None]
                if not mp:
                    continue
                xs = [p["n"] for p in mp]
                ys = [min(max(p[key], -0.08), 1.55) for p in mp]
                ax.scatter(xs, ys, s=8, color=c, alpha=0.25, zorder=z, edgecolors="none")
                gs = log_clusters(mp)
                cx = [med([p["n"] for p in g]) for g in gs]
                cy = [min(max(med([p[key] for p in g]), -0.08), 1.55) for g in gs]
                # break the line where consecutive size clusters are more
                # than 40x apart (e.g. matched-size vs full-set arms):
                # joining them reads as a frontier that was never measured.
                seg_x, seg_y = [cx[0]], [cy[0]]
                first = True
                for i in range(1, len(cx)):
                    if cx[i] > seg_x[-1] * 40:
                        ax.plot(seg_x, seg_y, ls, color=c, lw=1.9, zorder=z + 10,
                                marker="o", ms=4.5,
                                label=(STYLE[m][3] if row == 0 and first else None))
                        first = False
                        seg_x, seg_y = [], []
                    seg_x.append(cx[i]); seg_y.append(cy[i])
                ax.plot(seg_x, seg_y, ls, color=c, lw=1.9, zorder=z + 10,
                        marker="o", ms=4.5,
                        label=(STYLE[m][3] if row == 0 and first else None))
            ax.set_xscale("log")
            ax.set_ylim(-0.1, 1.6)
            ax.grid(True, which="both", alpha=0.15)
            if row == 0:
                ax.set_title(title, fontsize=9.5)
            if col == 0:
                ax.set_ylabel(["zero-fill faithfulness", "necessity (sup)"][row])
            if row == 1:
                ax.set_xlabel("circuit size (nodes, log)")
        h, l = axes[0][col].get_legend_handles_labels()
        axes[1][col].legend(h, l, fontsize=6.0, loc="upper center",
                            bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.9)
    fig.suptitle("Faithfulness-size and necessity-size frontiers, per arena "
                 "(6-seed pilot; 22 seeds on TuringLLM). Methods are compared only within a "
                 "column; y values clipped to [-0.08, 1.55]; lines break across >40x size gaps; dotted = hybrid decomposition arms (external selection, our calibration), not shipped methods.", fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("comparison.%s" % ext), dpi=170)
    print("wrote comparison.pdf / comparison.png")


if __name__ == "__main__":
    main()
