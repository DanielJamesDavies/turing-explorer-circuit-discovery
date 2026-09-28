"""PAPER MAIN RESULT (DAN-75): the full protocol-v1 run on TuringLLM, under the DAN-8 pass rule.

  figure  paper/figures/main-result.pdf: (left) pass rate by depth band x site kind, value on each bar;
          (right) the distribution of circuit sizes, log scale, with the 10^2-10^3 range shaded
  prints  every number the paper quotes from the run (headline, by kind, by layer and band, sizes, near-threshold,
          specificity, mid-band, sufficiency to induce), so the text can be checked against one output

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/paper_main.py
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
    numbers(s, mid)
    FIGS.mkdir(parents=True, exist_ok=True)
    for fig, name in ((figure(s), "main-result"), (induce_figure(s), "induce")):
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
