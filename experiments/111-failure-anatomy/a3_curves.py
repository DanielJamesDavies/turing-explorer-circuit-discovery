"""111 a3: training curves of the production main fits (main/train.shard*.jsonl, curve.n_members / loss / penalty per
step, 400 steps), per model, with 062 curves.py's statistics:
  - first step at which any gate falls below the keep threshold (n_members below its step-0 value);
  - median members at steps 0, 50, 100, 150, 200, 300, 399;
  - step at which a circuit is first within 2x of its final size (np.argmax(size < 2 * size[-1]));
  - shrink over the last LAST = 100 steps, 1 - size[399] / size[299]; share still shrinking > 5% and > 10%;
  - within each depth band, pass rate (held-out strongest, non-vacuous, merge rule) of the settled half (shrink <=
    the band median) against the still-shrinking half, with a bootstrap CI of the difference (paired by nothing:
    an unpaired bootstrap over targets in each half).
Bands are the production bands (lib109.bands_of), not curves.py's curve_bands.

Writes paper/figures/run2026_training_curves.pdf/.png (rows = models; size per step by band, data loss per pass by
band, shrink over the last 100 steps by target layer), out/run2026_training_curves.csv, out/a3.json

  python -X utf8 experiments/111-failure-anatomy/a3_curves.py
"""
import json

import numpy as np
import pandas as pd

import lib111 as X

L = X.L
PERIOD = 12


def load_curves(model):
    fits = {}
    for d in L.find(model, "out", "main"):
        for p in [d / "train.jsonl"] + sorted(d.glob("train.shard*.jsonl")):
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    c = r.get("curve") or {}
                    if "error" in r or not c.get("n_members"):
                        continue
                    fits[r["seed"]] = (c["n_members"], c.get("loss"), c.get("penalty"), r.get("t_fit"))
    keys = sorted(fits)
    lens = {len(fits[k][0]) for k in keys}
    if len(lens) != 1:
        raise SystemExit("%s: curves of different lengths %s" % (model, sorted(lens)))
    size = np.array([fits[k][0] for k in keys], float)
    loss = np.array([fits[k][1] for k in keys], float)
    data = loss - np.array([fits[k][2] for k in keys], float)
    return keys, size, data


def boot_diff(a, b, B=None, seed=L.BOOT_SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    B = B or L.BOOT_B
    s = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    rows, summary, figdata = [], {}, {}
    for model in L.MODELS:
        lab = L.LABEL[model]
        keys, size, data = load_curves(model)
        strong, _ = X.strong_mid(model)
        ev = strong.set_index("seed")
        layer = np.array([int(k.split(".")[0]) for k in keys])
        band = np.array([L.band_of(model, l) for l in layer])
        T = size.shape[1]
        shrink = 1.0 - size[:, -1] / np.maximum(size[:, -1 - X.LAST], 1.0)
        first_drop = np.array([int(np.argmax(x < x[0])) if (x < x[0]).any() else T for x in size])
        reach = np.array([int(np.argmax(x < 2 * x[-1])) for x in size])
        ok = np.array([k in ev.index and not ev.loc[k, "vac"] for k in keys])
        passed = np.array([bool(ev.loc[k, "pass"]) if o else False for k, o in zip(keys, ok)])
        final_match = np.mean([abs(size[i, -1] - ev.loc[k, "n"]) < 0.5 for i, k in enumerate(keys) if k in ev.index])
        sm = dict(model=lab, fits=len(keys), steps=T, final_size_matches_eval_n=float(final_match),
                  first_gate_below_keep_median=float(np.median(first_drop)),
                  reach_2x_median=float(np.median(reach)), reach_2x_p10=float(np.percentile(reach, 10)),
                  reach_2x_p90=float(np.percentile(reach, 90)),
                  **{"members_step%d_median" % s_: float(np.median(size[:, s_])) for s_ in (0, 50, 100, 150, 200, 300, T - 1)},
                  shrink_median_pct=100 * float(np.median(shrink)), shrink_p90_pct=100 * float(np.percentile(shrink, 90)),
                  shrinking_gt5_pct=100 * float((shrink > .05).mean()), shrinking_gt10_pct=100 * float((shrink > .10).mean()),
                  data_loss_last100_decrease_median_pct=100 * float(np.median(
                      1 - data[:, -PERIOD:].mean(1) / np.maximum(data[:, -X.LAST - PERIOD:-X.LAST].mean(1), 1e-12))))
        sm["by_band"] = {}
        for b, _, _ in L.bands_of(model):
            m = band == b
            mm = m & ok
            cut = np.median(shrink[mm])
            lo, hi = mm & (shrink <= cut), mm & (shrink > cut)
            dlo, dhi = boot_diff(passed[lo], passed[hi])
            sm["by_band"][b] = dict(n=int(m.sum()), shrink_median_pct=100 * float(np.median(shrink[m])),
                                    shrinking_gt5_pct=100 * float((shrink[m] > .05).mean()),
                                    settled_pass_pct=100 * float(passed[lo].mean()), settled_n=int(lo.sum()),
                                    shrinking_pass_pct=100 * float(passed[hi].mean()), shrinking_n=int(hi.sum()),
                                    settled_minus_shrinking_lo=100 * dlo, settled_minus_shrinking_hi=100 * dhi)
            rows.append(dict(model=lab, band=b, **sm["by_band"][b]))
        sm["shrink_by_layer_median_pct"] = {int(l): 100 * float(np.median(shrink[layer == l])) for l in np.unique(layer)}
        rows.append(dict(model=lab, band="all", n=len(keys), shrink_median_pct=sm["shrink_median_pct"],
                         shrinking_gt5_pct=sm["shrinking_gt5_pct"]))
        summary[model] = sm
        figdata[model] = (layer, band, size, data, shrink)
        print("%-11s fits %d (%d steps; final size = eval n for %.1f%%) | first gate below keep: median step %d | "
              "members median step 100 %.0f, final %.0f | within 2x of final by median step %d (p10 %d, p90 %d) | "
              "shrink last 100: median %.1f%% (p90 %.1f%%), >5%% %.0f%%, >10%% %.0f%% | data loss decrease over the last "
              "100 steps (per-pass means) median %.1f%%" % (
                  lab, len(keys), T, 100 * final_match, sm["first_gate_below_keep_median"], sm["members_step100_median"],
                  sm["members_step%d_median" % (T - 1)], sm["reach_2x_median"], sm["reach_2x_p10"], sm["reach_2x_p90"],
                  sm["shrink_median_pct"], sm["shrink_p90_pct"], sm["shrinking_gt5_pct"], sm["shrinking_gt10_pct"],
                  sm["data_loss_last100_decrease_median_pct"]))
        for b, v in sm["by_band"].items():
            print("    %-6s shrink median %.1f%%, >5%% %.0f%% | pass settled half %.1f%% (n %d) vs still-shrinking half "
                  "%.1f%% (n %d); diff CI [%.1f, %.1f]" % (b, v["shrink_median_pct"], v["shrinking_gt5_pct"],
                                                           v["settled_pass_pct"], v["settled_n"], v["shrinking_pass_pct"],
                                                           v["shrinking_n"], v["settled_minus_shrinking_lo"],
                                                           v["settled_minus_shrinking_hi"]))
        print("    shrink by layer (median %%): " + ", ".join("L%d %.1f" % kv for kv in sm["shrink_by_layer_median_pct"].items()))
    pd.DataFrame(rows).to_csv(X.OUT / "run2026_training_curves.csv", index=False)
    L.write_json("a3", summary)
    figure(figdata)


def figure(figdata):
    plt, style = L.figure_style()
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 6.8, "xtick.labelsize": 6.2,
                         "ytick.labelsize": 6.2, "legend.fontsize": 6.0})
    ramp = ("#8aa4ff", "#0044ff", "#002894")
    models = [m for m in L.MODELS if m in figdata]
    fig, axes = plt.subplots(len(models), 3, figsize=(5.5, 1.6 * len(models) + 0.2), squeeze=False)
    for i, model in enumerate(models):
        layer, band, size, data, shrink = figdata[model]
        steps = np.arange(size.shape[1])
        pp = np.array([np.convolve(r, np.ones(PERIOD) / PERIOD, mode="valid") for r in data])
        for (b, _, _), col in zip(L.bands_of(model), ramp):
            m = band == b
            for ax, x, y in ((axes[i, 0], steps, size[m]), (axes[i, 1], steps[PERIOD - 1:], np.clip(pp[m], 1e-5, None))):
                lo, md, hi = np.percentile(y, [25, 50, 75], axis=0)
                ax.fill_between(x, lo, hi, color=style.tint(col, 0.75), linewidth=0, zorder=1)
                ax.plot(x, md, color=col, label="%s (n %d)" % (b, m.sum()), zorder=2, linewidth=1.1)
        axes[i, 0].set(yscale="log", ylabel="%s\nlatents in circuit" % L.LABEL[model])
        axes[i, 1].set(yscale="log", ylabel="data loss (per pass)", ylim=(1e-3, 5.0))
        axes[i, 0].legend(loc="upper right", frameon=False)
        ls = np.unique(layer)
        q = np.array([np.percentile(shrink[layer == l], [25, 50, 75]) * 100 for l in ls])
        axes[i, 2].vlines(ls, q[:, 0], q[:, 2], color=style.tint("#0044ff", 0.55), linewidth=2.0, zorder=1)
        axes[i, 2].plot(ls, q[:, 1], "o", color="#0044ff", markersize=4, zorder=2)
        axes[i, 2].set(ylabel="removed, last %d steps (%%)" % X.LAST, ylim=(0, None),
                       xticks=range(0, int(ls.max()) + 1, 2 if ls.max() > 6 else 1))
        if i == 0:
            axes[i, 0].set_title("Circuit size during training")
            axes[i, 1].set_title("Data loss (Z + C + A + R)")
            axes[i, 2].set_title("Still shrinking at step %d" % size.shape[1])
        if i == len(models) - 1:
            axes[i, 0].set_xlabel("training step")
            axes[i, 1].set_xlabel("training step")
            axes[i, 2].set_xlabel("target layer")
    p = style.save_figure(fig, L.FIGURES / "run2026_training_curves.png")
    print("figure:", p)


if __name__ == "__main__":
    main()
