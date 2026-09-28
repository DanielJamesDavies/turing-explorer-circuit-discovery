"""Merge the per-shard outputs of the protocol-v1 runs and write the summary (safe to run while shards are going).

Main run (out/main):
  - bookkeeping: targets done / skipped / failed, thin targets, per-stage timings and a throughput projection
    for the full 15,046-target run (DAN-75)
  - circuits: size distribution (share inside 10^2-10^3), faithfulness under Z / A / C on held-out strongest and
    mid-band contexts, necessity, sufficiency to induce, and the DAN-8 pass rule (Z, A, C all in [0.8, 1.5] AND
    necessity >= 0.9; vacuous denominators excluded and counted), by layer and by site kind
  - specificity (056): amplifier rate, sibling vs target faithfulness; near-threshold targets (clean rank >= K/2)
Sweep (out/sweep): per method x lambda medians, per-target natural-scale cost (unweighted / WCM nodes at a
    worst-of-3 deviation <= 0.3) by depth band, and the faithfulness-vs-size figure.

  PYTHONPATH=src python experiments/062-h100-protocol-v1/merge.py
      -> out/summary.md, out/targets.csv, out/sweep_rows.csv, out/faithfulness_vs_size.png
  env: OUT (default experiments/062-h100-protocol-v1/out)  GPUS (8, for the projection)
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out")))
GPUS = int(os.environ.get("GPUS", 8))
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
BAND = (0.8, 1.5)          # DAN-8 pass rule (2026-09-27): Z, A, C all in BAND and necessity >= NEC_MIN
NEC_MIN = 0.9
N_FULL = 15046
K_TOPK = 128


def read(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return pd.DataFrame(rows)


def depth_band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def passes(df):
    """The DAN-8 pass rule on the activation read: Z, A, C all in BAND and necessity >= NEC_MIN. A missing score
    fails. Apply to rows with vacuous denominators already excluded (vacuous())."""
    inb = ((df[HEAD] >= BAND[0]) & (df[HEAD] <= BAND[1])).all(axis=1)
    return inb & (df.phi_sup_blind_tk >= NEC_MIN)


def vacuous(df):
    return df.get("vacuous_tk", pd.Series(False, index=df.index)).fillna(False).astype(bool)


def headline(df, lines, title):
    if df.empty:
        return
    vac = vacuous(df)
    ok = df[~vac]
    worst = (ok[HEAD] - 1).abs().max(axis=1)
    n = df["n"]
    lines.append("\n### %s (%d circuits)\n" % (title, len(df)))
    lines.append("- nodes: median %d (p10 %d, p90 %d); inside 10^2-10^3: %.1f%%" % (
        n.median(), n.quantile(.1), n.quantile(.9), 100 * ((n >= 100) & (n <= 1000)).mean()))
    lines.append("- faithfulness medians Z / A / C: %.3f / %.3f / %.3f" % (
        df.free0_tk.median(), df.freeM_topk_tk.median(), df.freeN_topk_tk.median()))
    lines.append("- PASS (Z, A, C in [%.2f, %.2f] and necessity >= %.1f): %.1f%% of %d; vacuous denominators excluded: %d"
                 % (BAND[0], BAND[1], NEC_MIN, 100 * passes(ok).mean(), len(ok), int(vac.sum())))
    lines.append("- largest deviation max|faith - 1|: median %.3f (p25 %.3f, p75 %.3f)" % (
        worst.median(), worst.quantile(.25), worst.quantile(.75)))
    lines.append("- necessity median %.3f; sufficiency to induce median %.3f" % (
        df.phi_sup_blind_tk.median(), df.phi_cf_alpha_blind_tk.median()))


def table_by(df, col, lines):
    if df.empty:
        return
    d = df[~vacuous(df)].copy()
    d["passes"] = passes(d)
    t = d.groupby(col).agg(circuits=("seed", "count"), nodes=("n", "median"), free0=("free0_tk", "median"),
                           freeM=("freeM_topk_tk", "median"), freeN=("freeN_topk_tk", "median"),
                           passes=("passes", "mean"), necessity=("phi_sup_blind_tk", "median"),
                           induce=("phi_cf_alpha_blind_tk", "median")).round(3)
    lines.append("\n" + t.to_string())


def main_summary(lines):
    st = read(OUT / "main" / "status.*.jsonl")
    if st.empty:
        lines.append("\n## Main run: no rows yet"); return
    st = st.drop_duplicates("seed", keep="last")
    ev = read(OUT / "main" / "eval.*.jsonl")
    sp = read(OUT / "main" / "spec.*.jsonl")
    ok = st[st.get("ok") == True] if "ok" in st else st.iloc[0:0]  # noqa: E712
    lines.append("\n## Main run (protocol v1)\n")
    lines.append("- targets with a status row: %d; ok %d; skipped %s; failed %d; thin (no mid-band pool) %d" % (
        len(st), len(ok), st["skip"].value_counts().to_dict() if "skip" in st else {},
        int(st["error"].notna().sum()) if "error" in st else 0, int(ok.get("thin", pd.Series(dtype=bool)).sum())))
    tcols = [c for c in ("t_ctx", "t_fit", "t_eval", "t_spec") if c in ok]
    if tcols and len(ok):
        tot = ok[tcols].fillna(0).sum(axis=1)
        per_gpu_h = 3600 / tot.mean()
        lines.append("- seconds per target (median): %s; mean total %.0f s -> %.0f targets per GPU-hour" % (
            ", ".join("%s %.0f" % (c[2:], ok[c].median()) for c in tcols), tot.mean(), per_gpu_h))
        lines.append("- projection for the full %d-target run on %d GPUs: %.1f h (contexts + fit + eval + spec; "
                     "about %.1f h without specificity)" % (
                         N_FULL, GPUS, N_FULL / per_gpu_h / GPUS,
                         N_FULL / (3600 / (tot - ok.get("t_spec", 0).fillna(0)).mean()) / GPUS))
    if ev.empty:
        return
    ev = ev[ev.get("error").isna()] if "error" in ev else ev
    ev = ev[ev.get("skip").isna()] if "skip" in ev else ev
    ev = ev.drop_duplicates(["seed", "held"], keep="last")
    ev["layer"] = ev.seed.str.split(".").str[0].astype(int)
    ev["kind"] = ev.seed.str.split(".").str[1]
    ev["depth"] = ev.layer.map(depth_band)
    strong, mid = ev[ev.held == "strong"], ev[ev.held == "mid"]
    headline(strong, lines, "Held-out strongest contexts (primary)")
    headline(mid, lines, "Held-out mid-band contexts (reported)")
    lines.append("\n### By site kind (held-out strongest)")
    table_by(strong, "kind", lines)
    lines.append("\n### By layer (held-out strongest)")
    table_by(strong, "layer", lines)
    tgt = strong.set_index("seed")[["layer", "kind", "n"] + HEAD + ["phi_sup_blind_tk", "phi_cf_alpha_blind_tk"]]
    if not sp.empty:
        sp = sp[sp.get("error").isna()] if "error" in sp else sp
        sp = sp.drop_duplicates(["seed", "pi"], keep="last")
        flag = (sp.target_faith_pre >= 0.5) & (sp.sibling_faith_median >= 0.8 * sp.target_faith_pre)
        sp = sp.assign(amp_flag=flag)
        per = sp.groupby("seed").agg(amp_any=("amp_flag", "any"), sib_C=("sibling_faith_median", "median"),
                                     rank_clean=("rank_clean", "median"), lifted_C=("switched_on_circuit", "median"))
        tgt = tgt.join(per, how="left")
        tgt["near_threshold"] = tgt.rank_clean >= K_TOPK / 2
        inb = passes(tgt)
        lines.append("\n### Specificity and near-threshold targets (held-out strongest)\n")
        lines.append("- amplifier flag under any ablation method: %.1f%% of circuits (%d / %d)" % (
            100 * tgt.amp_any.mean(), int(tgt.amp_any.sum()), int(tgt.amp_any.notna().sum())))
        for lab, sub in (("near-threshold (clean rank >= %d)" % (K_TOPK // 2), tgt[tgt.near_threshold == True]),  # noqa: E712
                         ("other targets", tgt[tgt.near_threshold == False])):  # noqa: E712
            if len(sub):
                lines.append("- %s: %d targets, pass %.1f%%, amplifier rate %.1f%%" % (
                    lab, len(sub), 100 * inb[sub.index].mean(), 100 * sub.amp_any.mean()))
        lines.append("\nAmplifier rate by site kind: %s" % tgt.groupby("kind").amp_any.mean().round(3).to_dict())
    tgt.to_csv(OUT / "targets.csv")


def sweep_summary(lines):
    sw = read(OUT / "sweep" / "eval.*.jsonl")
    if sw.empty:
        return
    sw = sw[sw.get("error").isna()] if "error" in sw else sw
    sw = sw[sw.get("skip").isna()] if "skip" in sw else sw
    sw = sw.drop_duplicates(["seed", "arm"], keep="last")
    sw["layer"] = sw.seed.str.split(".").str[0].astype(int)
    sw["depth"] = sw.layer.map(depth_band)
    sw["dev"] = (sw[HEAD] - 1).abs().max(axis=1)
    sw["in_band"] = ((sw[HEAD] >= BAND[0]) & (sw[HEAD] <= BAND[1])).all(axis=1)
    sw["passes"] = passes(sw) & ~vacuous(sw)
    sw.to_csv(OUT / "sweep_rows.csv", index=False)
    lines.append("\n## Sweep: WCM vs unweighted circuit masking (held-out strongest)\n")
    t = sw.groupby(["weighted", "lam"]).agg(targets=("seed", "nunique"), nodes=("n", "median"),
                                             free0=("free0_tk", "median"), freeM=("freeM_topk_tk", "median"),
                                             freeN=("freeN_topk_tk", "median"), worst_dev=("dev", "median"),
                                             passes=("passes", "mean"), induce=("phi_cf_alpha_blind_tk", "median"))
    lines.append(t.round(3).to_string())
    # per-target natural-scale cost
    cost = []
    for s, g in sw.groupby("seed"):
        def smallest(w):
            x = g[(g.weighted == w) & (g.dev <= 0.3)]
            return x.n.min() if len(x) else np.nan
        cost.append(dict(seed=s, depth=g.depth.iloc[0], wcm=smallest(True), unw=smallest(False)))
    c = pd.DataFrame(cost)
    c["ratio"] = c.unw / c.wcm
    lines.append("\n### Natural-scale cost: smallest circuit with worst-of-3 deviation <= 0.3, per target\n")
    for dep, g in c.groupby("depth"):
        both = g.dropna(subset=["wcm", "unw"])
        lines.append("- %s: %d targets; WCM reaches %d, unweighted %d, both %d; median nodes WCM %s vs unweighted %s; "
                     "median ratio %s" % (dep, len(g), g.wcm.notna().sum(), g.unw.notna().sum(), len(both),
                                          "%.0f" % both.wcm.median() if len(both) else "-",
                                          "%.0f" % both.unw.median() if len(both) else "-",
                                          "%.1fx" % both.ratio.median() if len(both) else "-"))
    figure(sw)


def figure(sw):
    from analysis.style import BLUE, CATEGORICAL, INK_MUTED, configure_matplotlib, save_figure, style_suptitle, tint
    plt = configure_matplotlib()
    bands = ["L0-4", "L5-7", "L8-11"]
    fig, axes = plt.subplots(len(bands), 3, figsize=(15, 4.2 * len(bands)), sharey=True)
    for r, band in enumerate(bands):
        sub = sw[sw.depth == band]
        for cidx, (f, lab) in enumerate((("free0_tk", "zero ablation"), ("freeM_topk_tk", "mean ablation, activating"),
                                         ("freeN_topk_tk", "mean ablation, contrast"))):
            ax = axes[r, cidx]
            ax.axhspan(BAND[0], BAND[1], color=tint(INK_MUTED, 0.85), zorder=0)
            ax.axhline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (3, 3)), zorder=1)
            for weighted, col, name in ((True, BLUE, "WCM (fitted coefficients)"), (False, CATEGORICAL[1], "unweighted (α = 1)")):
                s = sub[sub.weighted == weighted].groupby("lam")
                if not len(s):
                    continue
                med = s.agg(n=("n", "median"), y=(f, "median"), lo=(f, lambda v: v.quantile(.25)),
                            hi=(f, lambda v: v.quantile(.75))).sort_values("n")
                ax.fill_between(med.n, med.lo, med.hi, color=tint(col, 0.75), lw=0, zorder=2)
                ax.plot(med.n, med.y, marker="o", color=col, label=name, zorder=3)
            ax.set_xscale("log"); ax.set_ylim(-0.1, 1.6)
            ax.set_title("%s — %s (%d targets)" % (band, lab, sub.seed.nunique()) if cidx == 0 else lab)
            if r == len(bands) - 1:
                ax.set_xlabel("nodes (median over targets)")
        axes[r, 0].set_ylabel("faithfulness (held-out)")
    axes[0, 0].legend(loc="lower right")
    style_suptitle(fig, "Faithfulness against circuit size by depth: WCM vs unweighted circuit masking")
    save_figure(fig, OUT / "faithfulness_vs_size.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Protocol-v1 H100 runs: summary (%s)" % pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")]
    main_summary(lines)
    sweep_summary(lines)
    txt = "\n".join(lines)
    (OUT / "summary.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
