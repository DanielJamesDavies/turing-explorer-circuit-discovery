"""PAPER MAIN RESULT (DAN-75): the full protocol-v1 run on TuringLLM, under the DAN-8 pass rule.

  figures paper/figures/failure-anatomy.pdf (Figure 2): one outcome per circuit by layer and site kind (figure_v4)
          paper/figures/budget-curves.pdf (Figure 3): share of the sweep's targets with a passing circuit of at most N
            latents, WCM (8 penalties) vs unweighted (5), plus the full run at the production penalty (figure_budget)
          paper/figures/circuit-sizes.pdf (appendix): the distribution of circuit sizes
          paper/figures/induce.pdf (appendix): sufficiency to induce
          (2026-10-06; the old main-result.pdf, bars by band x kind + size histogram, is no longer written)
  prints  every number the paper quotes from the run (headline, by kind, by layer and band, sizes, near-threshold,
          specificity, mid-band, sufficiency to induce), so the text can be checked against one output

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/paper_main.py
  FIG=v2   draft of the new main-result figure (pass rate vs size per depth band + layer x kind heatmap), written to
           OUT/main-result-v2.png only, for review before it replaces main-result.pdf
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import merge as M  # noqa: E402  (reads OUT at import)
from analysis.style import (INK, INK_SECONDARY, SURFACE, configure_matplotlib, grouped_bar_geometry,  # noqa: E402
                            round_bars, save_figure, styled_legend, tint)

OUT = M.OUT
FIGS = ROOT / "paper" / "figures"
BANDS = (("L0-4", "0–4"), ("L5-7", "5–7"), ("L8-11", "8–11"))
# site kinds: validated trio (blue / honey / pink; all pairs pass, worst dE 12.8 colour-blind), no red so pink is
# not read as C; "all" in ink. Values are printed on every bar (the contrast relief the validator asks for).
KINDS = (("resid", "residual stream", "#0044ff"), ("mlp", "MLP", "#d99400"), ("attn", "attention", "#ff3d8b"))


def load():
    ev = M.read(OUT / "main" / "eval.*.jsonl")
    ev = ev[ev.get("error").isna()] if "error" in ev else ev
    ev = ev[ev.get("skip").isna()] if "skip" in ev else ev
    ev = ev.drop_duplicates(["seed", "held"], keep="last")
    ev["layer"] = ev.seed.str.split(".").str[0].astype(int)
    ev["kind"] = ev.seed.str.split(".").str[1]
    ev["depth"] = ev.layer.map(M.depth_band)
    tg = pd.read_csv(OUT / "targets.csv", index_col="seed")          # merge.py's per-target table (amp, rank)
    strong = ev[ev.held == "strong"].set_index("seed").join(tg[["amp_any", "rank_clean"]], how="left")
    strong = strong[~M.vacuous(strong)].copy()
    strong["passes"] = M.passes(strong)
    strong["near"] = strong.rank_clean >= M.K_TOPK / 2
    mid = ev[ev.held == "mid"].set_index("seed")
    mid = mid[~M.vacuous(mid)].copy()
    mid["passes"] = M.passes(mid)
    return strong, mid


def numbers(s, mid):
    pct = lambda x: 100 * float(np.mean(x))
    n = s["n"]
    print("== headline (held-out strongest, activation read, DAN-8 rule)")
    print("circuits %d (non-vacuous) | PASS %.1f%% | median nodes %d (p10 %d, p90 %d) | in 10^2-10^3 %.1f%%" % (
        len(s), pct(s.passes), n.median(), n.quantile(.1), n.quantile(.9), pct((n >= 100) & (n <= 1000))))
    worst = (s[M.HEAD] - 1).abs().max(axis=1)
    print("largest deviation median %.3f | Z/A/C medians %.3f / %.3f / %.3f | necessity >= 0.9: %.1f%%" % (
        worst.median(), *[s[h].median() for h in M.HEAD], pct(s.phi_sup_blind_tk >= .9)))
    print("passing circuits: median nodes %d; in 10^2-10^3 %.1f%%" % (
        s[s.passes].n.median(), pct((s[s.passes].n >= 100) & (s[s.passes].n <= 1000))))
    print("\n== pass rate by site kind: " + " | ".join("%s %.1f%% (n %d)" % (k, pct(s[s.kind == k].passes),
                                                                            (s.kind == k).sum()) for k, _, _ in KINDS))
    print("== by depth band: " + " | ".join("%s %.1f%%" % (b, pct(s[s.depth == b].passes)) for b, _ in BANDS))
    print("== by layer: " + ", ".join("L%d %.0f%%" % (l, pct(g.passes)) for l, g in s.groupby("layer")))
    print("== band x kind:")
    for b, lab in BANDS:
        print("   %-6s " % lab + " | ".join("%s %.1f%%" % (k, pct(s[(s.depth == b) & (s.kind == k)].passes))
                                          for k, _, _ in KINDS))
    near, rest = s[s.near == True], s[s.near == False]                    # noqa: E712
    print("\n== near-threshold (clean rank >= %d): %d targets (%.1f%%), %.0f%% attention, pass %.1f%% | others pass %.1f%%"
          % (M.K_TOPK // 2, len(near), 100 * len(near) / len(s), pct(near.kind == "attn"), pct(near.passes),
             pct(rest.passes)))
    amp = s.amp_any == True                                               # noqa: E712
    print("== amplifier flag: %.1f%% overall; " % pct(amp) + ", ".join(
        "%s %.1f%%" % (k, pct(amp[s.kind == k])) for k, _, _ in KINDS) + "; among PASSING circuits %.1f%%" % pct(amp[s.passes]))
    print("== pass AND not amplifier: %.1f%%" % pct(s.passes & ~amp))
    print("\n== mid-band held-out (reported): pass %.1f%% of %d; Z/A/C medians %.3f / %.3f / %.3f; induce median %.3f"
          % (pct(mid.passes), len(mid), *[mid[h].median() for h in M.HEAD], mid.phi_cf_alpha_blind_tk.median()))
    ind = s.phi_cf_alpha_blind_tk
    print("== sufficiency to induce (strongest): median %.3f | in [0.5, 2] %.1f%% | by kind: %s" % (
        ind.median(), pct(ind.between(.5, 2)),
        ", ".join("%s %.2f" % (k, s[s.kind == k].phi_cf_alpha_blind_tk.median()) for k, _, _ in KINDS)))


def figure(s):
    plt = configure_matplotlib()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.1), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28})
    groups = [lab for _, lab in BANDS] + ["all"]
    width, offsets = grouped_bar_geometry(len(KINDS))
    x = np.arange(len(groups))
    for (k, name, color), off in zip(KINDS, offsets):
        vals = [100 * s[(s.depth == b) & (s.kind == k)].passes.mean() for b, _ in BANDS] + \
               [100 * s[s.kind == k].passes.mean()]
        bars = ax.bar(x + off, vals, width, color=color, label=name, zorder=2)
        for bar in bars:
            ax.annotate("%.0f" % bar.get_height(), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set(xticks=x, xticklabels=["layers " + g if g != "all" else "all layers" for g in groups],
           ylabel="circuits that pass (%)", ylim=(0, 100))
    ax.set_title("Pass rate (all circuits %.0f%%)" % (100 * s.passes.mean()), fontsize=11.5)
    styled_legend(ax, loc="upper right", fontsize=8.5)
    round_bars(ax)

    n = s["n"].clip(lower=1)
    bins = np.logspace(0, np.log10(n.max()) + 0.05, 45)
    bx.axvspan(100, 1000, color=tint(INK_SECONDARY, 0.88), zorder=0)
    bx.hist(n, bins=bins, color=tint("#0044ff", 0.25), edgecolor=SURFACE, linewidth=0.6, zorder=2)
    bx.set(xscale="log", xlabel="circuit size (upstream latents)", ylabel="circuits", xlim=(max(1, n.min() * 0.8), None))
    bx.text(bx.get_xlim()[0] * 1.3, bx.get_ylim()[1] * 0.93, "shaded: $10^{2}$–$10^{3}$\n%.0f%% of circuits" % (
        100 * ((n >= 100) & (n <= 1000)).mean()), ha="left", va="top", fontsize=9, color=INK)   # empty left side
    bx.set_title("Circuit size (median %d)" % s["n"].median(), fontsize=11.5)
    bx.grid(axis="x", visible=False)
    return fig


def frontier_data():
    """Pass rate and median size per (depth band, lambda) for WCM: the 190-target sweep (lambda 4e-3 .. 2.5e-4), and the
    DAN-131 subsample (out_bands: 16 + 16 targets at layers 0-7; out_depth: 32 at 8-11) followed across EVERY lambda,
    its sweep rows plus its weaker-penalty fits (1e-4, 5e-5, 2.5e-5). The subsample is a subset of the sweep."""
    import json
    import depth_test as D
    sw = pd.read_csv(OUT / "sweep_rows.csv")
    sw = sw[sw.weighted]
    low = []
    for d in ("out_bands", "out_depth"):
        for line in open(HERE / d / "sweep" / "eval.shard0.jsonl"):
            r = json.loads(line)
            if "error" not in r and "skip" not in r:
                low.append(r)
    low = pd.DataFrame(low).drop_duplicates(["seed", "arm"], keep="last")
    low = low[~low.arm.str.contains("_s")]                           # the 800-step arm is not a penalty point
    low = low[~D.vacuous(low)].copy()
    low["passes"] = D.passes(low)
    low["lam"] = low.arm.str[2:].astype(float)
    low["depth"] = low.seed.map(lambda k: M.depth_band(int(k.split(".")[0])))
    sub = set(low.seed)
    agg = dict(n=("n", "median"), passes=("passes", "mean"), targets=("seed", "nunique"))
    full = sw.groupby(["depth", "lam"]).agg(**agg).reset_index()
    subc = pd.concat([sw[sw.seed.isin(sub)][["seed", "depth", "lam", "n", "passes"]],
                      low[["seed", "depth", "lam", "n", "passes"]]]).groupby(["depth", "lam"]).agg(**agg).reset_index()
    return full, subc


def figure_v2(s):
    """MAIN-RESULT FIGURE, v2 (2026-10-04 draft; preview only until Daniel approves). Drawn at print size (7 in).
    Top: pass rate against median circuit size per depth band, across sparsity penalties (the production penalty is one
    operating point on a trade-off). Solid = the 190-target sweep; dashed = the DAN-131 subsample of the same targets
    followed down to lambda 2.5e-5; ring = the full run (~15k targets) at the production penalty.
    Bottom: pass rate at the production penalty by layer x site kind, value in every cell."""
    from matplotlib.colors import LinearSegmentedColormap
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.8, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.4, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    full, subc = frontier_data()
    fig = plt.figure(figsize=(7.0, 4.15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.78], hspace=0.62, wspace=0.12)
    light = tint("#0044ff", 0.45)
    for j, (band, lab) in enumerate(BANDS):
        ax = fig.add_subplot(gs[0, j])
        f, c = full[full.depth == band].sort_values("n"), subc[subc.depth == band].sort_values("n")
        ax.plot(c.n, 100 * c.passes, color=light, linestyle=(0, (3, 2)), marker="o", markersize=2.8,
                markerfacecolor=SURFACE, markeredgewidth=0.9, linewidth=1.1, zorder=2,
                label="same %d targets, weaker penalties" % c.targets.max())
        ax.plot(f.n, 100 * f.passes, color="#0044ff", marker="o", markersize=3.0, linewidth=1.4, zorder=3,
                label="%d-target sweep" % f.targets.max())
        prod = s[s.depth == band]
        ax.plot(prod.n.median(), 100 * prod.passes.mean(), marker="o", markersize=7.5, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.1, linestyle="none", zorder=4,
                label="full run, production penalty")
        ax.annotate("%.0f%%" % (100 * prod.passes.mean()), (prod.n.median(), 100 * prod.passes.mean()),
                    xytext=(-7, 5), textcoords="offset points", ha="right", va="bottom", fontsize=6.4, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.4), zorder=5)
        top = c.sort_values("n").iloc[-1]
        ax.annotate("%.0f%%" % (100 * top.passes), (top.n, 100 * top.passes), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.4, color=INK)
        ax.set(xscale="log", ylim=(0, 100), xlim=(70, 2e4))
        ax.set_xlabel("circuit size (median latents)")
        if j == 0:
            ax.set_ylabel("targets that pass (%)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(axis="x", visible=False)
        ax.set_title("(%s) Layers %s" % ("abc"[j], lab), fontsize=7.8)
        if j == 0:                                                   # one legend row above the three panels
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color="#0044ff", marker="o", markersize=3.0, linewidth=1.4,
                              label="190-target sweep"),
                       Line2D([0], [0], color=light, linestyle=(0, (3, 2)), marker="o", markersize=2.8,
                              markerfacecolor=SURFACE, linewidth=1.1, label="64-target subsample, to weaker penalties"),
                       Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="none", markeredgecolor=INK,
                              linestyle="none", label="full run, production penalty")]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=6.4,
                       handlelength=2.2, columnspacing=2.0, frameon=False)

    hx = fig.add_subplot(gs[1, :])
    kinds = [(k, name) for k, name, _ in KINDS]
    grid = np.full((len(kinds), 13), np.nan)
    counts = np.zeros_like(grid)
    for i, (k, _) in enumerate(kinds):
        for l in range(12):
            g = s[(s.kind == k) & (s.layer == l)]
            if len(g):
                grid[i, l], counts[i, l] = 100 * g.passes.mean(), len(g)
        grid[i, 12] = 100 * s[s.kind == k].passes.mean()
    cmap = LinearSegmentedColormap.from_list("pass", ["#f4f6fb", "#0044ff"])
    cols = list(range(12)) + [12.6]                                  # the "all layers" column, set apart
    for i in range(len(kinds)):
        for jj, x in enumerate(cols):
            v = grid[i, jj]
            if np.isnan(v):
                hx.add_patch(plt.Rectangle((x - 0.47, i - 0.45), 0.94, 0.9, facecolor=SURFACE, edgecolor=tint(INK, 0.85),
                                           linewidth=0.6, linestyle=(0, (2, 2))))
                hx.text(x, i, "—", ha="center", va="center", fontsize=6.4, color=INK_SECONDARY)
                continue
            hx.add_patch(plt.Rectangle((x - 0.47, i - 0.45), 0.94, 0.9, facecolor=cmap(v / 100), edgecolor="none"))
            hx.text(x, i, "%.0f" % v, ha="center", va="center", fontsize=6.6,
                    color="white" if v >= 55 else INK, fontweight="bold" if jj == 12 else "normal")
    hx.set_xlim(-0.6, 13.2); hx.set_ylim(len(kinds) - 0.5, -0.5)
    hx.set_xticks(cols); hx.set_xticklabels([str(l) for l in range(12)] + ["all"])
    hx.set_yticks(range(len(kinds))); hx.set_yticklabels([name for _, name in kinds])
    hx.set_xlabel("target layer")
    hx.grid(False)
    for sp in hx.spines.values():
        sp.set_visible(False)
    hx.set_title("(d) Pass rate (%) at the production penalty, by target layer and site kind", fontsize=7.8)
    return fig


def figure_v3(s):
    """MAIN-RESULT FIGURE, v3 draft (Daniel, 2026-10-04): one panel per ablation method (Z, A, C), one line per target
    layer coloured red (layer 0) -> purple (layer 11). x = median circuit size across the 190-target sweep of sparsity
    penalties (WCM); y = median faithfulness under that ablation on its own scale (1 = the target fully reproduced,
    above 1 = overshoot), with the pass band [0.8, 1.5] shaded. Preview only."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.8, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.4, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    sw = pd.read_csv(OUT / "sweep_rows.csv")
    sw = sw[sw.weighted].copy()
    # red -> purple rainbow (Daniel, 2026-10-04), from the Lab Bright palette plus an orange; gold, not yellow, so the
    # middle layers stay visible on white
    cmap = LinearSegmentedColormap.from_list("depth", ["#fa1e4e", "#ff7a1a", "#d99400", "#12c46a", "#0ab5c9",
                                                       "#0044ff", "#b028ff"])
    norm = Normalize(0, 11)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (col, name) in zip(axes, zip(M.HEAD, ("(a) Zero ablation (Z)", "(b) Activating-mean ablation (A)",
                                                   "(c) Contrast-mean ablation (C)"))):
        ax.axhspan(0.8, 1.5, color=tint("#12c46a", 0.9), zorder=0, linewidth=0)
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for layer, g in sw.groupby("layer"):
            c = g.groupby("lam").agg(n=("n", "median"), f=(col, "median")).sort_values("n")
            ax.plot(c.n, c.f, color=cmap(norm(layer)), marker="o", markersize=2.2, linewidth=1.1, zorder=3)
        ax.set(xscale="log", ylim=(0, 1.6))
        ax.set_xlabel("circuit size (median latents)")
        ax.grid(axis="x", visible=False)
        ax.set_title(name, fontsize=7.8)
    axes[0].set_ylabel("faithfulness (median)")
    axes[0].text(axes[0].get_xlim()[0] * 1.3, 1.47, "pass band", fontsize=6.2, color="#0e9e55", va="top")
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.025, pad=0.015, ticks=[0, 3, 6, 9, 11])
    cb.set_label("target layer", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.4, length=0)
    cb.outline.set_visible(False)
    return fig


OUTCOMES = (("pass", "passes", "#0044ff"),
            ("mean", "fails only the mean ablations (A and/or C)", tint("#0044ff", 0.62)),
            ("zero", "fails zero ablation (Z)", "#9aa0a8"),
            ("nec", "fails necessity", INK_SECONDARY),
            ("undef", "a score is undefined", "#e3e5e9"))


def outcome(s):
    """One outcome per circuit, in order: pass; a score undefined (no defined Z / A / C or necessity, counted as a fail
    by the rule); fails necessity; fails zero ablation (necessity holding); fails only the mean ablations."""
    ok = {k: s[c].between(0.8, 1.5) for k, c in zip("ZAC", M.HEAD)}
    undef = s[M.HEAD + ["phi_sup_blind_tk"]].isna().any(axis=1)
    nec = s.phi_sup_blind_tk >= 0.9
    return np.select([s.passes, undef, ~nec, ~ok["Z"]], ["pass", "undef", "nec", "zero"], "mean")


def figure_v4(s):
    """MAIN-RESULT FIGURE, v4 draft (2026-10-05, from the Figure 2 survey; preview only): the anatomy of the joint test.
    One stacked bar per target layer, one panel per site kind; every circuit of the full run counted once."""
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.8, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    s = s.copy()
    s["outcome"] = outcome(s)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35), sharey=True, gridspec_kw={"wspace": 0.08})   # \linewidth
    for j, (ax, (k, name, _)) in enumerate(zip(axes, KINDS)):
        g = s[s.kind == k]
        share = (g.groupby("layer").outcome.value_counts(normalize=True).unstack()
                 .reindex(columns=[o for o, _, _ in OUTCOMES]).fillna(0) * 100)
        bottom = np.zeros(len(share))
        for o, lab, col in OUTCOMES:
            ax.bar(share.index, share[o], 0.78, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=0.4,
                   label=lab, zorder=2)
            bottom += share[o].values
        for l, v in share["pass"].items():
            inside = v >= 9
            ax.text(l, v - 1.5 if inside else v + 1.5, "%.0f" % v, ha="center", va="top" if inside else "bottom",
                    fontsize=5.7, color="white" if inside else INK, zorder=4)
        if 0 not in share.index:                                        # no site upstream of layer-0 attention
            ax.text(0, 2, "no\nsite", ha="center", va="bottom", fontsize=5.4, color=INK_SECONDARY)
        ax.set(xticks=range(12), ylim=(0, 100), xlim=(-0.6, 11.6))
        ax.set_xlabel("target layer")
        ax.grid(False)
        ax.set_title("(%s) %s, %.0f%% pass" % ("abc"[j], {"resid": "Residual", "mlp": "MLP",
                                                          "attn": "Attention"}[k], 100 * g.passes.mean()), fontsize=7.4)
    axes[0].set_ylabel("circuits (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handlelength=1.0,
               handleheight=0.8, columnspacing=1.2)
    print("== v4 outcome shares, by depth band:")
    print((s.groupby("depth").outcome.value_counts(normalize=True).unstack() * 100).round(1).to_string())
    print("all: " + ", ".join("%s %.1f%%" % (o, 100 * (s.outcome == o).mean()) for o, _, _ in OUTCOMES)
          + " | near-threshold among attention 'zero' fails: %.0f%%" % (
              100 * s[(s.kind == "attn") & (s.outcome == "zero")].near.mean()))
    return fig


def budget_cdf(times, observed, xs):
    """Share of ALL targets whose smallest passing circuit has <= N latents (targets that never pass count as not
    explained, so this is a lower bound; a Kaplan-Meier version was tried and rejected 2026-10-05: censoring is
    informative, the hardest targets stop first, and it inflated the curves). NaN beyond the largest circuit fitted."""
    t, o = np.asarray(times, float), np.asarray(observed, bool)
    F = np.searchsorted(np.sort(t[o]), xs, side="right") / len(t)
    return np.where(xs <= t.max(), F, np.nan)


def per_target(rows):
    """Per target: (smallest passing size, True) or (largest size fitted, False)."""
    g = rows.groupby("seed")
    best = rows[rows.passes].groupby("seed").n.min()
    big = g.n.max()
    return pd.DataFrame({"t": best.reindex(big.index).fillna(big), "o": big.index.isin(best.index)})


def figure_budget(s):
    """BUDGET CURVES, draft for Figure 3 (2026-10-05; preview only): per depth band, the share of targets with a
    passing circuit of at most N latents, from the 190-target sweep (each target's smallest passing circuit over the
    penalties it was fitted at: WCM 4e-3 .. 2.5e-5, eight penalties, the three weakest from out_bands / out_depth /
    out_lowlam; unweighted 1e-3 .. 1e-5, five; targets that never pass count as not explained; 90% bootstrap band
    over targets; each curve ends at the largest circuit fitted). WCM (blue) against unweighted circuit masking
    (grey); the full run at the production penalty (ink, dotted: one penalty only, so it flattens at that penalty's
    pass rate)."""
    import json
    import depth_test as D
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.8, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0})
    sw = pd.read_csv(OUT / "sweep_rows.csv")
    low = []
    for d in ("out_bands", "out_depth", "out_lowlam"):                 # WCM at 1e-4, 5e-5, 2.5e-5: every sweep target
        if not (HERE / d / "sweep" / "eval.shard0.jsonl").exists():
            continue
        for line in open(HERE / d / "sweep" / "eval.shard0.jsonl"):
            r = json.loads(line)
            if "error" not in r and "skip" not in r and "_s" not in r["arm"]:
                low.append(r)
    low = pd.DataFrame(low).drop_duplicates(["seed", "arm"], keep="last")
    low = low[~D.vacuous(low)].copy()
    low["passes"] = D.passes(low)
    low["depth"] = low.seed.map(lambda k: M.depth_band(int(k.split(".")[0])))
    xs = np.logspace(np.log10(30), np.log10(3e5), 600)
    rng = np.random.default_rng(0)
    grey = "#9aa0a8"
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})    # \linewidth
    for j, (ax, (band, lab)) in enumerate(zip(axes, BANDS)):
        ax.axvspan(100, 1000, color=tint(INK_SECONDARY, 0.92), zorder=0, linewidth=0)
        for weighted, col, name in ((False, grey, "unweighted (α = 1), 5 penalties"), (True, "#0044ff",
                                                                                     "WCM, 8 penalties")):
            rows = sw[(sw.weighted == weighted) & (sw.depth == band)][["seed", "n", "passes"]]
            if weighted:
                rows = pd.concat([rows, low[(low.depth == band) & low.seed.isin(set(rows.seed))][["seed", "n", "passes"]]])
            pt = per_target(rows)
            F = budget_cdf(pt.t, pt.o, xs)
            boots = []
            for _ in range(300):
                b = pt.sample(len(pt), replace=True, random_state=int(rng.integers(1 << 31)))
                boots.append(np.where(np.isnan(F), np.nan, budget_cdf(b.t, b.o, xs)))
            with np.errstate(all="ignore"):
                lo, hi = np.nanpercentile(np.array(boots), [5, 95], axis=0)
            ok = ~np.isnan(F)
            # end each curve exactly at the largest circuit fitted, at the share that ever passes (the grid alone
            # stops short of it and understates the last step)
            xe, ye = np.append(xs[ok], pt.t.max()), np.append(F[ok], pt.o.mean())
            ax.fill_between(xs[ok], 100 * lo[ok], 100 * hi[ok], color=tint(col, 0.75), linewidth=0, zorder=1)
            ax.plot(xe, 100 * ye, color=col, linewidth=1.5, zorder=3, drawstyle="steps-post", label=name)
            n_band = len(pt)
            ax.annotate("%.0f%%" % (100 * ye[-1]), (xe[-1], 100 * ye[-1]), xytext=(2, 2),
                        textcoords="offset points", fontsize=6.0, color=col if weighted else INK_SECONDARY)
        g = s[s.depth == band]
        y = 100 * np.searchsorted(np.sort(g.n.where(g.passes).dropna().values), xs, side="right") / len(g)
        ax.plot(xs, y, color=INK, linestyle=(0, (1, 1.2)), linewidth=1.1, zorder=4,
                label="WCM full run, production penalty only")
        ax.annotate("%.0f%%" % y[-1], (xs[-1], y[-1]), xytext=(-2, -7), textcoords="offset points", fontsize=6.0,
                    ha="right", color=INK)
        ax.set(xscale="log", xlim=(30, 3e5), ylim=(0, 100))              # unweighted circuits reach ~2e5 latents
        ax.set_xticks([1e2, 1e3, 1e4, 1e5])
        ax.set_xlabel("circuit size N (upstream latents)")
        ax.grid(axis="x", visible=False)
        ax.set_title("(%s) Layers %s (%d targets)" % ("abc"[j], lab, n_band))
    axes[0].set_ylabel("targets with a passing circuit\nof at most N latents (%)")
    axes[0].text(316, 97, "$10^{2}$–$10^{3}$", ha="center", va="top", fontsize=5.8, color=INK_SECONDARY)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handlelength=2.0,
               columnspacing=1.0)
    return fig


def figure_sizes(s):
    """APPENDIX (2026-10-06, from the old main-result figure's right panel): the distribution of circuit sizes in the
    full run, log scale, 10^2-10^3 shaded."""
    plt = configure_matplotlib()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.8, "axes.titlepad": 4.0, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
                         "axes.labelpad": 2.0})
    fig, bx = plt.subplots(figsize=(3.4, 2.2))
    n = s["n"].clip(lower=1)
    bins = np.logspace(0, np.log10(n.max()) + 0.05, 45)
    bx.axvspan(100, 1000, color=tint(INK_SECONDARY, 0.88), zorder=0)
    bx.hist(n, bins=bins, color=tint("#0044ff", 0.25), edgecolor=SURFACE, linewidth=0.5, zorder=2)
    bx.set(xscale="log", xlabel="circuit size (upstream latents)", ylabel="circuits", xlim=(max(1, n.min() * 0.8), None))
    bx.text(bx.get_xlim()[0] * 1.3, bx.get_ylim()[1] * 0.93, "shaded: $10^{2}$–$10^{3}$\n%.0f%% of circuits" % (
        100 * ((n >= 100) & (n <= 1000)).mean()), ha="left", va="top", fontsize=6.4, color=INK)
    bx.set_title("Circuit size, full run (median %d)" % s["n"].median())
    bx.grid(axis="x", visible=False)
    return fig


def bars(ax, s, value, fmt, ylim, title, ylabel):
    """Grouped bars: depth band (+ all layers) x site kind, value printed on each bar."""
    groups = [lab for _, lab in BANDS] + ["all"]
    width, offsets = grouped_bar_geometry(len(KINDS))
    x = np.arange(len(groups))
    for (k, name, color), off in zip(KINDS, offsets):
        vals = [value(s[(s.depth == b) & (s.kind == k)]) for b, _ in BANDS] + [value(s[s.kind == k])]
        bs = ax.bar(x + off, vals, width, color=color, label=name, zorder=2)
        for bar in bs:
            ax.annotate(fmt % bar.get_height(), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK,
                        zorder=4)
    ax.set(xticks=x, xticklabels=groups, xlabel="target layers", ylim=ylim, ylabel=ylabel)
    ax.set_title(title, fontsize=11.5)
    round_bars(ax)


def induce_figure(s):
    """Sufficiency to induce (phi_ind with fitted alpha, activation read) on held-out contrast contexts, where the
    target is silent: does setting the circuit's values switch it on? Reported, not part of the pass rule."""
    plt = configure_matplotlib()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.1), gridspec_kw={"wspace": 0.25})
    ax.axhline(1.0, color=INK_SECONDARY, lw=1.0, ls=(0, (3, 3)), zorder=1)
    bars(ax, s, lambda g: g.phi_cf_alpha_blind_tk.median(), "%.2f", (0, 1.6),
         "Sufficiency to induce (median)", "$\\phi_{\\mathrm{ind}}$ (1 = natural level)")
    bars(bx, s, lambda g: 100 * (g.phi_cf_alpha_blind_tk >= 0.5).mean(), "%.0f", (0, 115),
         "Switch the target on to half or more", "circuits (%)")
    styled_legend(ax, loc="upper left", fontsize=8.5)                  # left panel's top left is empty
    return fig


def main():
    s, mid = load()
    if os.environ.get("FIG") in ("v2", "v3", "v4", "budget"):        # preview only: never written to paper/figures
        v = os.environ["FIG"]
        fig = {"v2": figure_v2, "v3": figure_v3, "v4": figure_v4, "budget": figure_budget}[v](s)
        print("preview:", save_figure(fig, OUT / ("main-result-%s.png" % v)))
        return
    numbers(s, mid)
    FIGS.mkdir(parents=True, exist_ok=True)
    # 2026-10-06 (Daniel): Figure 2 = failure-anatomy (figure_v4), Figure 3 = budget-curves (figure_budget), the size
    # histogram moves to the appendix (circuit-sizes). The old bars + histogram figure (figure()) is no longer written.
    for fig, name in ((figure_v4(s), "failure-anatomy"), (figure_budget(s), "budget-curves"),
                      (figure_sizes(s), "circuit-sizes"), (induce_figure(s), "induce")):
        png = save_figure(fig, OUT / ("%s.png" % name))
        shutil.copy(png.with_suffix(".pdf"), FIGS / ("%s.pdf" % name))
        print("figure:", png, "->", FIGS / ("%s.pdf" % name))
    ind = s.phi_cf_alpha_blind_tk
    print("== induce by band x kind (median | share >= 0.5):")
    for b, lab in BANDS:
        print("   %-6s " % lab + " | ".join("%s %.2f / %.0f%%" % (k, s[(s.depth == b) & (s.kind == k)].phi_cf_alpha_blind_tk.median(),
                                                               100 * (s[(s.depth == b) & (s.kind == k)].phi_cf_alpha_blind_tk >= .5).mean())
                                           for k, _, _ in KINDS))
    print("   all: median %.2f, >= 0.5 %.0f%%, > 2 (overshoot) %.0f%%, < 0.1 %.0f%%; among PASSING circuits median %.2f, >= 0.5 %.0f%%" % (
        ind.median(), 100 * (ind >= .5).mean(), 100 * (ind > 2).mean(), 100 * (ind < .1).mean(),
        s[s.passes].phi_cf_alpha_blind_tk.median(), 100 * (s[s.passes].phi_cf_alpha_blind_tk >= .5).mean()))


if __name__ == "__main__":
    main()
