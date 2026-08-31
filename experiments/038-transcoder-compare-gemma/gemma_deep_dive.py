"""Gemma-arena DEEP DIVE: everything the stored circuits can say
without touching the GPU, plus the consensus frontier once the
CT_FREQ_SWEEP arms exist in ours_gtc_rows.jsonl.

Inputs (all already on disk):
  ours_gtc_members.jsonl   our 6 tri-amp circuits, members + fitted alphas
  theirs_gtc_nodes.jsonl   ct direct-edge ranking (20k cap)
  sfc_nodes.jsonl          SFC attribution ranking
  theirs_gtc_pruned.jsonl  ct_published / ct_seed_rooted union + window freq
  ours_gtc_rows.jsonl      every scored arm (frontier panel reads *_f<t>)

Panels (gemma_deepdive.pdf/png):
  A  recall of OUR members in each external ordering, vs depth k
  B  window-consensus decay: nodes surviving >= t of 48 windows
  C  layer composition: ours vs their stable core / matched head / union
  D  their per-window circuit sizes (box) vs our fixed n
  E  our fitted amplitude spectrum per seed
  F  consensus frontier: f0 vs size for freq-threshold cuts (needs sweep)

  python gemma_deep_dive.py   ->  figure + deepdive_stats.md
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BAND = (0.8, 1.25)
SEEDS = []          # [(L, sl)] discovered from members file, plot order
OURS = {}           # (L,sl) -> {(layer, feat): alpha}


def load():
    for line in open(HERE / "ours_gtc_members.jsonl"):
        r = json.loads(line)
        if r["arm"] != "triamp400":
            continue
        key = (int(r["layer"]), int(r["latent"]))
        SEEDS.append(key)
        OURS[key] = {(int(l), int(f)): float(a)
                     for l, d in r["alphas"].items() for f, a in d.items()}
    rank = {}
    for name, fn in (("ct-direct", "theirs_gtc_nodes.jsonl"),
                     ("sfc", "sfc_nodes.jsonl")):
        rank[name] = {}
        for line in open(HERE / fn):
            r = json.loads(line)
            rank[name][(int(r["layer"]), int(r["latent"]))] = [
                (int(l), int(f)) for l, f, *_ in r["ranking"]]
    pruned = {}
    for line in open(HERE / "theirs_gtc_pruned.jsonl"):
        r = json.loads(line)
        pruned[(int(r["layer"]), int(r["latent"]))] = {
            arm: {"freqc": [(int(l), int(f), int(c)) for l, f, c in a["freq"]],
                  "sizes": a["size_per_window"]}
            for arm in ("ct_published", "ct_seed_rooted")
            if (a := r.get(arm))}
    rows = defaultdict(dict)
    for line in open(HERE / "ours_gtc_rows.jsonl"):
        r = json.loads(line)
        rows[(int(r["layer"]), int(r["latent"]))][r["arm"]] = r
    return rank, pruned, rows


def main():
    rank, pruned, rows = load()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    sm = "\n".join
    stats = ["# Gemma deep-dive statistics", ""]
    seed_lab = {k: "L%d/%d" % k for k in SEEDS}

    # ---- A: recall of our members in external orderings ----------------
    A = ax[0][0]
    srcs = [("ct-direct", "#e8862c"), ("sfc", "#7b3fb8"),
            ("ct_published freq", "#c22f2f"), ("ct_seed_rooted freq", "#8c1a1a")]
    stats += ["## A. Depth k needed to recover our circuit", "",
              "| seed | ours n | " + " | ".join(s for s, _ in srcs) + " |",
              "|---" * (len(srcs) + 2) + "|"]
    for key in SEEDS:
        ours = set(OURS[key]); need = []
        for name, col in srcs:
            if name.endswith("freq"):
                arm = name.split()[0]
                order = [(l, f) for l, f, _ in
                         sorted(pruned[key][arm]["freqc"], key=lambda x: -x[2])]
            else:
                order = rank[name][key]
            pos = {nf: i for i, nf in enumerate(order)}
            got = sorted(pos[m] + 1 for m in ours if m in pos)
            xs, ys = [1], [0.0]
            for i, kk in enumerate(got, 1):
                xs += [kk, kk]; ys += [ys[-1], i / len(ours)]
            xs.append(len(order)); ys.append(ys[-1])
            A.plot(xs, ys, color=col, alpha=0.35, lw=1.0)
            k90 = next((kk for i, kk in enumerate(got, 1)
                        if i / len(ours) >= 0.9), None)
            need.append("%s" % (k90 if k90 else "> %d" % len(order)))
        stats.append("| %s | %d | %s |" % (seed_lab[key], len(ours),
                                           " | ".join(need)))
    for name, col in srcs:
        A.plot([], [], color=col, lw=2, label=name)
    A.set_xscale("log"); A.set_xlabel("depth k in external ordering")
    A.set_ylabel("recall of our tri-amp members")
    A.set_title("A. How deep must each ordering go\nto recover our circuit?", fontsize=9.5)
    A.legend(fontsize=6.5, loc="upper left"); A.grid(alpha=0.15)

    # ---- B: window-consensus decay -------------------------------------
    B = ax[0][1]
    stats += ["", "## B. Window stability (share of union in >= t of 48 windows)",
              "", "| seed | arm | all 48 | >= 24 | union |", "|---|---|---|---|---|"]
    for key in SEEDS:
        for arm, col in (("ct_published", "#c22f2f"), ("ct_seed_rooted", "#8c1a1a")):
            fc = pruned[key][arm]["freqc"]
            ts = range(1, 49)
            ys = [sum(1 for _, _, c in fc if c >= t) for t in ts]
            B.plot(ts, ys, color=col, alpha=0.4, lw=1.1)
            stats.append("| %s | %s | %d (%.1f%%) | %d (%.1f%%) | %d |" % (
                seed_lab[key], arm, ys[-1], 100 * ys[-1] / ys[0],
                ys[23], 100 * ys[23] / ys[0], ys[0]))
    for arm, col in (("as published", "#c22f2f"), ("seed-rooted", "#8c1a1a")):
        B.plot([], [], color=col, lw=2, label=arm)
    B.set_yscale("log"); B.set_xlabel("window-survival threshold t (of 48)")
    B.set_ylabel("nodes surviving (log)")
    B.set_title("B. Membership churn across windows:\nthe pruned circuit is mostly window-local", fontsize=9.5)
    B.legend(fontsize=6.5); B.grid(alpha=0.15)

    # ---- C: layer composition ------------------------------------------
    C = ax[0][2]
    import numpy as np
    max_l = max(l for key in SEEDS for (l, f) in OURS[key])
    max_l = max(max_l, 6)
    sets = [("ours", "#1657d6",
             lambda key: list(OURS[key])),
            ("their stable core (t=48)", "#c22f2f",
             lambda key: [(l, f) for l, f, c in
                          pruned[key]["ct_published"]["freqc"] if c >= 48]),
            ("their matched head", "#e8862c",
             lambda key: [(l, f) for l, f, _ in sorted(
                 pruned[key]["ct_published"]["freqc"],
                 key=lambda x: -x[2])][:len(OURS[key])]),
            ("their union", "#7a6652",
             lambda key: [(l, f) for l, f, _ in
                          pruned[key]["ct_published"]["freqc"]])]
    w = 0.8 / len(sets)
    for i, (name, col, fn) in enumerate(sets):
        share = np.zeros(max_l + 1)
        for key in SEEDS:
            mem = fn(key)
            if not mem:
                continue
            h = np.bincount([l for l, _ in mem], minlength=max_l + 1)
            share += h / max(1, len(mem))
        share /= len(SEEDS)
        C.bar(np.arange(max_l + 1) + (i - len(sets) / 2 + .5) * w, share,
              width=w, color=col, label=name)
    C.set_xlabel("transcoder layer"); C.set_ylabel("mean share of set")
    C.set_title("C. Where the nodes live:\nlayer composition (mean over 6 seeds)", fontsize=9.5)
    C.legend(fontsize=6.5); C.grid(alpha=0.15, axis="y")

    # ---- D: their per-window sizes vs our n ----------------------------
    D = ax[1][0]
    pos, labels = [], []
    for i, key in enumerate(SEEDS):
        bp = D.boxplot([pruned[key]["ct_published"]["sizes"],
                        pruned[key]["ct_seed_rooted"]["sizes"]],
                       positions=[i * 3, i * 3 + 1], widths=0.8,
                       patch_artist=True, showfliers=False)
        for p, col in zip(bp["boxes"], ("#c22f2f", "#8c1a1a")):
            p.set_facecolor(col); p.set_alpha(0.5)
        D.plot([i * 3 - .5, i * 3 + 1.5], [len(OURS[key])] * 2,
               color="#1657d6", lw=2)
        pos.append(i * 3 + .5); labels.append(seed_lab[key])
    D.plot([], [], color="#1657d6", lw=2, label="ours (fixed n)")
    D.plot([], [], color="#c22f2f", lw=5, alpha=.5, label="published, per window")
    D.plot([], [], color="#8c1a1a", lw=5, alpha=.5, label="seed-rooted, per window")
    D.set_xticks(pos); D.set_xticklabels(labels, fontsize=7, rotation=20)
    D.set_yscale("log"); D.set_ylabel("circuit size (nodes, log)")
    D.set_title("D. Natural sizes: their pruning per window\nvs our learned circuit", fontsize=9.5)
    D.legend(fontsize=6.5); D.grid(alpha=0.15, axis="y")

    # ---- E: our amplitude spectrum -------------------------------------
    E = ax[1][1]
    stats += ["", "## E. Fitted amplitudes", "",
              "| seed | n | median alpha | IQR | max/min |", "|---|---|---|---|---|"]
    for i, key in enumerate(SEEDS):
        al = sorted(OURS[key].values())
        E.scatter([i] * len(al), al, s=14, alpha=0.55, color="#1657d6",
                  edgecolors="none")
        q1, q2, q3 = al[len(al)//4], al[len(al)//2], al[3*len(al)//4]
        stats.append("| %s | %d | %.2f | %.2f-%.2f | %.1fx |" % (
            seed_lab[key], len(al), q2, q1, q3, al[-1] / max(al[0], 1e-9)))
    E.axhline(1.0, color="#999", lw=1, ls="--")
    E.set_xticks(range(len(SEEDS)))
    E.set_xticklabels([seed_lab[k] for k in SEEDS], fontsize=7, rotation=20)
    E.set_yscale("log"); E.set_ylabel("fitted alpha (log)")
    E.set_title("E. Calibration is not uniform: fitted\namplitudes span an order of magnitude", fontsize=9.5)
    E.grid(alpha=0.15, axis="y")

    # ---- F: consensus frontier (from the sweep arms, if present) -------
    F = ax[1][2]
    F.axhspan(*BAND, color="#2a9d2a", alpha=0.10, lw=0)
    have_sweep = False
    for arm, col, lab in (("ct_published", "#c22f2f", "published, freq cut"),
                          ("ct_seed_rooted", "#8c1a1a", "seed-rooted, freq cut")):
        per_seed = []
        for key in SEEDS:
            pts = sorted(((r["n"], r["ampF0"]) for a, r in rows[key].items()
                          if a.startswith(arm + "_f")
                          and r.get("ampF0") is not None), key=lambda t: t[0])
            if pts:
                have_sweep = True
                F.plot(*zip(*pts), color=col, alpha=0.4, lw=1.1, marker="o", ms=2.5)
                per_seed.append(pts)
        F.plot([], [], color=col, lw=2, label=lab)
    for key in SEEDS:
        r = rows[key].get("triamp400")
        if r:
            F.scatter([r["n"]], [r["ampF0"]], color="#1657d6", s=42, zorder=9,
                      marker="*")
    F.scatter([], [], color="#1657d6", s=42, marker="*", label="ours (tri-amp)")
    F.set_xscale("log"); F.set_ylim(-0.05, 1.3)
    F.set_xlabel("circuit size (nodes, log)")
    F.set_ylabel("zero-fill faithfulness")
    F.set_title("F. Consensus frontier: their circuits cut by\ntheir own window-survival signal", fontsize=9.5)
    if not have_sweep:
        F.text(.5, .5, "sweep arms not scored yet\n(rerun after CT_FREQ_SWEEP)",
               ha="center", va="center", transform=F.transAxes, fontsize=9)
    F.legend(fontsize=6.5, loc="lower right"); F.grid(alpha=0.15)

    fig.suptitle("Gemma arena deep dive: circuit-tracer's pruned circuits vs our learned circuits, "
                 "on their shipped scan (6 seeds, 48 windows each)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("gemma_deepdive.%s" % ext), dpi=170)
    (HERE / "deepdive_stats.md").write_text(sm(stats), encoding="utf-8",
                                            newline="")
    print(sm(stats))
    print("\nwrote gemma_deepdive.pdf/png + deepdive_stats.md")


if __name__ == "__main__":
    main()
