"""Qualitative + quantitative anatomy of all six Gemma tri-amp
circuits. For every member: fitted alpha, rank in their direct-edge
ranking, rank in the SFC ranking, window-survival count in their
as-published pruned circuits, and the Neuronpedia label (from
neuronpedia_labels.jsonl -- run fetch_labels.py first).

Outputs:
  anatomy_L<l>_<f>.md      per-seed table, sorted by alpha
  anatomy_summary.md       cross-seed stats: Spearman(alpha, log rank),
                           coverage of our nodes in their rankings,
                           token-chain vs context split by rank bucket
  alpha_vs_rank.pdf/png    the quantitative version of the story

  python seed_anatomies.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

LABELS = {}
for line in open(HERE / "neuronpedia_labels.jsonl"):
    r = json.loads(line)
    LABELS[(r["layer"], r["feat"])] = r["label"]

SEEDS, OURS = [], {}
for line in open(HERE / "ours_gtc_members.jsonl"):
    r = json.loads(line)
    if r["arm"] != "triamp400":
        continue
    key = (int(r["layer"]), int(r["latent"]))
    SEEDS.append(key)
    OURS[key] = sorted(((int(l), int(f), a) for l, d in r["alphas"].items()
                        for f, a in d.items()), key=lambda t: -t[2])

RANKS = {"ct": {}, "sfc": {}}
for name, fn in (("ct", "theirs_gtc_nodes.jsonl"), ("sfc", "sfc_nodes.jsonl")):
    for line in open(HERE / fn):
        r = json.loads(line)
        RANKS[name][(int(r["layer"]), int(r["latent"]))] = {
            (int(l), int(f)): i + 1
            for i, (l, f, *_) in enumerate(r["ranking"])}
FREQ = {}
for line in open(HERE / "theirs_gtc_pruned.jsonl"):
    r = json.loads(line)
    FREQ[(int(r["layer"]), int(r["latent"]))] = {
        (int(l), int(f)): int(c)
        for l, f, c in r["ct_published"]["freq"]}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    summary = ["# Anatomy summary: six Gemma tri-amp circuits", "",
               "Label source: Neuronpedia gemmascope-transcoder-16k "
               "auto-interp, fetched 2026-08-26.", ""]
    stats_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    cols = ["#1657d6", "#2a9d2a", "#e8862c", "#7b3fb8", "#c22f2f", "#7a6652"]

    for si, key in enumerate(SEEDS):
        L, S = key
        mem = OURS[key]
        ct, sfc, fq = RANKS["ct"][key], RANKS["sfc"][key], FREQ[key]
        lab = lambda k: LABELS.get(k, "")
        lines = ["# Seed anatomy: L%d/%d" % key, "",
                 "Seed feature: %s" % (lab(key) or "(no label)"), "",
                 "tri-amp circuit n=%d. ct = rank in their direct-edge "
                 "ranking; sfc = rank in the SFC ranking; wins = window-"
                 "survival count (of 48) in their as-published pruning."
                 % len(mem), "",
                 "| lyr | feat | alpha | ct | sfc | wins | label |",
                 "|---|---|---|---|---|---|---|"]
        for l, f, a in mem:
            lines.append("| %d | %d | %.2f | %s | %s | %s | %s |" % (
                l, f, a, ct.get((l, f), "-"), sfc.get((l, f), "-"),
                fq.get((l, f), "-"), lab((l, f))))
        (HERE / ("anatomy_L%d_%d.md" % key)).write_text(
            "\n".join(lines), encoding="utf-8", newline="")

        al = [a for _, _, a in mem]
        ctr = [ct.get((l, f)) for l, f, _ in mem]
        in_ct = [i for i, r in enumerate(ctr) if r]
        rho_ct = spearman([al[i] for i in in_ct],
                          [math.log(ctr[i]) for i in in_ct])
        sfr = [sfc.get((l, f)) for l, f, _ in mem]
        in_sf = [i for i, r in enumerate(sfr) if r]
        rho_sf = spearman([al[i] for i in in_sf],
                          [math.log(sfr[i]) for i in in_sf])
        wins = [fq.get((l, f), 0) for l, f, _ in mem]
        rho_w = spearman(al, wins)
        # median their-rank of our top-alpha vs bottom-alpha quartile
        q = max(1, len(mem) // 4)
        top_r = sorted(r for r in ctr[:q] if r)
        bot_r = sorted(r for r in ctr[-q:] if r)
        med = lambda v: v[len(v) // 2] if v else None
        stats_rows.append((key, len(mem), sum(1 for r in ctr if r),
                           rho_ct, rho_sf, rho_w, med(top_r), med(bot_r)))
        axes[0].scatter([r for r in ctr if r],
                        [al[i] for i, r in enumerate(ctr) if r],
                        s=16, alpha=0.6, color=cols[si],
                        label="L%d/%d" % key, edgecolors="none")
        axes[1].scatter(wins, al, s=16, alpha=0.6, color=cols[si],
                        edgecolors="none")

    summary += ["## Cross-seed statistics", "",
                "| seed | n | in their 20k | Sp(alpha, log ct-rank) | "
                "Sp(alpha, log sfc-rank) | Sp(alpha, wins/48) | "
                "med ct-rank of top-alpha quartile | of bottom quartile |",
                "|---|---|---|---|---|---|---|---|"]
    for (key, n, cov, rc, rs, rw, tr, br) in stats_rows:
        summary.append("| L%d/%d | %d | %d | %.2f | %.2f | %.2f | %s | %s |"
                       % (key[0], key[1], n, cov, rc, rs, rw, tr, br))

    axes[0].set_xscale("log")
    axes[0].set_xlabel("rank in their direct-edge ranking (log)")
    axes[0].set_ylabel("our fitted alpha")
    axes[0].set_title("Amplification vs attribution rank", fontsize=10)
    axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("window-survival count in their pruning (of 48)")
    axes[1].set_ylabel("our fitted alpha")
    axes[1].set_title("Amplification vs their membership stability",
                      fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("Where the fit puts its weight: our alphas vs "
                 "circuit-tracer's importance signals (6 Gemma seeds)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("alpha_vs_rank.%s" % ext), dpi=170)

    (HERE / "anatomy_summary.md").write_text("\n".join(summary),
                                             encoding="utf-8", newline="")
    print("\n".join(summary))
    print("\nwrote anatomy_L*.md, anatomy_summary.md, alpha_vs_rank.pdf/png")


if __name__ == "__main__":
    main()
