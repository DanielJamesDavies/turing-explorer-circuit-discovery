"""DAN-44: the distribution of the fitted coefficients alpha of the WCM v4 production circuits.

Population: main circuits whose held-out-strongest eval row is non-vacuous (merge.vacuous), every member (the seed
node excluded). alpha = the member node's `amplitude` (rounded to 4 dp by the engine).

Writes  out/run2026_coef_alpha_summary.csv / paper/tables/run2026_coef_alpha_summary.tex  (per model)
        out/run2026_coef_alpha_by_{distance,kind,freq}.csv, paper/tables/run2026_coef_alpha_breakdown.tex
        out/run2026_coef_alpha_per_circuit.csv (one row per circuit: quantiles, shares)
        out/run2026_coef_freq_check.csv (frequency source checked against the 084 null profiles)
        paper/figures/run2026_coef_alpha_hist.pdf/.png
  python -I -X utf8 experiments/113-coefficients-shape/c1_alpha.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lib113 as C  # noqa: E402

L = C.L
NC = __import__("nulls_core")
NEAR1 = 0.01          # |alpha - 1| < 0.01: left at its initial value 1 (to the engine's 4 dp)
DIST_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 11, "8-11")]


def shares(g, by):
    """Per group: members, circuits, median alpha, pooled share < 1, > 1, ~1, with target-cluster CIs."""
    out = []
    for key, x in g.groupby(by, sort=False):
        per = x.groupby("seed").agg(n=("alpha", "size"), lt=("lt1", "sum"), gt=("gt1", "sum"), eq=("eq1", "sum"))
        r = dict(group=key, members=len(x), circuits=len(per), alpha_median=float(x.alpha.median()),
                 alpha_p10=float(x.alpha.quantile(.1)), alpha_p90=float(x.alpha.quantile(.9)))
        for c in ("lt", "gt", "eq"):
            lo, hi = C.cluster_share_ci(per[c].values, per.n.values)
            r["share_%s1" % c], r["share_%s1_lo" % c], r["share_%s1_hi" % c] = float(per[c].sum() / per.n.sum()), lo, hi
        out.append(r)
    return out


def dist_bin(d):
    for lo, hi, lab in DIST_BINS:
        if lo <= d <= hi:
            return lab
    return "?"


def freq_check(model, df, fdec_of):
    """For each 084 null-panel target: the members' frequency-decile histogram from our store vs the profile's."""
    d = L.find(model, "work" if model == "turingllm" else "out", "harder_nulls")
    rows = L.read_rows(d, "profile")
    out = []
    for r in rows:
        s = r.get("seed")
        sw = r.get("set_wcm") or {}
        if s is None or "fdec_hist" not in sw:
            continue
        x = df[df.seed == s]
        if x.empty:
            continue
        mine = np.bincount(np.clip(x.fdec.values, 0, 9)[x.fdec.values >= 0], minlength=10).tolist()
        out.append(dict(model=model, seed=s, n=len(x), profile_n=sw.get("n"), same=mine == list(sw["fdec_hist"]),
                        mine=json.dumps(mine), profile=json.dumps(sw["fdec_hist"])))
    return out


def main():
    summ, per_rows, brk, fchk, hist = [], [], {"distance": [], "kind": [], "freq": []}, [], {}
    for model in C.MODELS:
        pop = C.strong_population(model)
        df = C.members(model, "main")
        df = df[df.seed.isin(pop.index)].copy()
        df["lt1"], df["gt1"] = df.alpha < 1, df.alpha > 1
        df["eq1"] = (df.alpha - 1).abs() < NEAR1
        df["dist"] = (df.t_layer - df.layer).astype(int)
        df["dist_bin"] = df.dist.map(dist_bin)
        cnt = C.freq_store(model)
        if cnt is not None:
            dec = {}
            for (l, k), x in df.groupby(["layer", "kind"]):
                fd = NC.freq_deciles(cnt[C.comp_idx(model, l, k)])
                dec[(l, k)] = fd
            df["fdec"] = [int(dec[(l, k)][i]) for l, k, i in zip(df.layer, df.kind, df["index"])]
            fchk += freq_check(model, df, dec)
        else:
            df["fdec"] = -2
        hist[model] = df.alpha.values
        # per circuit
        pc = df.groupby("seed").alpha.agg(n="size", median="median", p10=lambda v: v.quantile(.1),
                                         p90=lambda v: v.quantile(.9), max="max", min="min")
        pc["share_lt1"] = df.groupby("seed").lt1.mean()
        pc["share_gt1"] = df.groupby("seed").gt1.mean()
        pc["share_eq1"] = df.groupby("seed").eq1.mean()
        pc["model"] = model
        pc["band"] = [L.band_of(model, s.split(".")[0]) for s in pc.index]
        per_rows.append(pc.reset_index())
        s = shares(df.assign(all="all"), "all")[0]
        s.update(model=model, circuits_pop=len(pop), circ_median_median=pc["median"].median(),
                 circ_p10_median=pc.p10.median(), circ_p90_median=pc.p90.median(), circ_max_median=pc["max"].median(),
                 circ_max_p90=pc["max"].quantile(.9), alpha_max=float(df.alpha.max()), alpha_min=float(df.alpha.min()),
                 share_lt04=float((df.alpha < 0.4).mean()), share_gt2=float((df.alpha > 2).mean()),
                 share_gt4=float((df.alpha > 4).mean()),
                 circ_share_gt1_median=pc.share_gt1.median(), circ_majority_gt1=float((pc.share_gt1 > 0.5).mean()))
        summ.append(s)
        for name, col in (("distance", "dist_bin"), ("kind", "kind"), ("freq", "fdec")):
            if name == "freq" and cnt is None:
                continue
            order = {"distance": [b[2] for b in DIST_BINS], "kind": list(L.KIND_ORDER), "freq": list(range(-1, 10))}[name]
            g = df[df[col].isin(order)].copy()
            g["_o"] = g[col].map({v: i for i, v in enumerate(order)})
            g = g.sort_values("_o")
            for r in shares(g, col):
                r.update(model=model, by=name)
                brk[name].append(r)
        print("%-11s circuits %d members %d median %.3f  <1 %.1f%%  >1 %.1f%%  ~1 %.1f%%  max %.2f" % (
            model, s["circuits"], s["members"], s["alpha_median"], 100 * s["share_lt1"], 100 * s["share_gt1"],
            100 * s["share_eq1"], s["alpha_max"]))
    C.OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(per_rows).to_csv(C.OUT / (C.PREFIX + "alpha_per_circuit.csv"), index=False)
    for k, v in brk.items():
        pd.DataFrame(v).to_csv(C.OUT / (C.PREFIX + "alpha_by_%s.csv" % k), index=False)
    if fchk:
        fc = pd.DataFrame(fchk)
        fc.to_csv(C.OUT / (C.PREFIX + "freq_check.csv"), index=False)
        print("frequency source vs 084 profiles (identical decile histograms):",
              fc.groupby("model").same.agg(["sum", "size"]).to_dict("index"))
    S = pd.DataFrame(summ)

    pct = lambda r, c: "%.1f {[}%.1f, %.1f{]}" % tuple(100 * getattr(r, c + x) for x in ("", "_lo", "_hi"))  # noqa: E731
    rows = [[L.LABEL[r.model], L.tex_int(r.circuits), L.tex_int(r.members), "%.3f" % r.alpha_median,
             "%.2f / %.2f / %.2f / %.2f" % (r.circ_p10_median, r.circ_median_median, r.circ_p90_median, r.circ_max_median),
             pct(r, "share_lt1"), pct(r, "share_gt1"), pct(r, "share_eq1")] for r in S.itertuples()]
    C.write_table("alpha_summary", rows, ["Model", "Circuits", "Members", r"Median $\alpha$",
                                          r"Per circuit p10 / median / p90 / max", r"$\alpha<1$ (\%)",
                                          r"$\alpha>1$ (\%)", r"$|\alpha-1|<0.01$ (\%)"], "lrrrlll",
                  footer=[r"Held-out-strongest non-vacuous circuits. Per-circuit columns: median over circuits of each "
                          r"circuit's quantile. Brackets: 95\% bootstrap CI over targets (4,000 resamples)."],
                  csv_df=S)
    # compact breakdown table: median alpha and share > 1 per group, models side by side
    tex = []
    for name, lab in (("distance", "layer distance"), ("kind", "member site"), ("freq", "frequency decile")):
        B = pd.DataFrame(brk[name])
        if B.empty:
            continue
        tex.append([r"\midrule"]) if tex else None
        for gval in B.group.unique():
            cells = [lab if gval == B.group.unique()[0] else "",
                     L.KIND_LABEL.get(gval, gval) if name == "kind" else ("never active" if gval == -1 else str(gval))]
            for model in C.MODELS:
                x = B[(B.model == model) & (B.group == gval)]
                cells.append(L.DASH if x.empty else "%.2f / %.0f" % (x.alpha_median.iloc[0], 100 * x.share_gt1.iloc[0]))
            tex.append(cells)
    C.write_table("alpha_breakdown", tex, ["By", "Group"] + [L.LABEL[m] for m in C.MODELS], "llrrr",
                  footer=[r"Cells: median $\alpha$ / share of members with $\alpha > 1$ (\%). Layer distance = target "
                          r"layer $-$ member layer; frequency decile per site over latents active in the corpus store."])
    figure(hist, pd.DataFrame(brk["distance"]))


def figure(hist, bd):
    plt, st = C.figure_style()
    cols = {"turingllm": st.BLUE, "pythia-70m": st.CATEGORICAL[1], "gpt2-small": st.CATEGORICAL[2]}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    edges = np.logspace(np.log10(0.05), np.log10(15), 71)       # values outside are clipped into the end bins
    for m in C.MODELS:
        a = np.clip(hist[m], edges[0], edges[-1])
        h, _ = np.histogram(a, bins=edges)
        ax.stairs(100 * h / h.sum(), edges, color=cols[m], lw=1.3, label=L.LABEL[m])
    ax.axvline(1.0, color=st.INK_MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel(r"fitted coefficient $\alpha$ (log scale)")
    ax.set_ylabel("members (%)")
    ax.set_title(r"(a) $\alpha$ over all members")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    order = [b[2] for b in DIST_BINS]
    for m in C.MODELS:
        x = bd[bd.model == m].set_index("group").reindex(order).dropna(subset=["members"])
        xs = [order.index(g) for g in x.index]
        ax.plot(xs, 100 * x.share_gt1, marker="o", ms=3, color=cols[m])
        ax.fill_between(xs, 100 * x.share_gt1_lo, 100 * x.share_gt1_hi, color=st.tint(cols[m], 0.75), lw=0)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("layer distance to target")
    ax.set_ylabel(r"members with $\alpha>1$ (%)")
    ax.set_title("(b) amplification by distance")
    st.save_figure(fig, C.FIGURES / (C.PREFIX + "alpha_hist.png"))


if __name__ == "__main__":
    main()
