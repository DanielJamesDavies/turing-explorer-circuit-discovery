"""100 E4 report: which changes to v2 improve donor-background robustness without hurting production (no GPU).

Baselines (v2 = wcm_1e-3, v2CR = v2cr_1e-3) come from out/main (Run A / Run C); E4 arms from out/e4.
Applies the README's pre-stated selection rule. Writes out/e4/tables_e4.md, tables_e4.json, e4_recovery_vs_pass.png.

  python experiments/100-scrub-frontier/report_e4.py
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from report import fin, load_rows, med, miqr, q, rec, f3  # noqa: E402

MAIN, E4 = HERE / "out" / "main", HERE / "out" / "e4"
BASE = ("wcm_1e-3", "v2cr_1e-3")
ORDER = ["wcm_1e-3", "v2cr_1e-3", "v2cr-w0.1_1e-3", "v2cr-w0.5_1e-3", "v2cr-w1_1e-3", "v2cr-mixCR_1e-3",
         "v2cr-mixCAR_1e-3", "rsAC_1e-3", "v2cr-k4_1e-3", "v2cr_2.5e-4", "v2cr-cap2_1e-3", "v2cr-s800_1e-3"]


def score_of(r):
    a, c = rec(r, "fit", "A"), rec(r, "fit", "C")
    return None if a is None or c is None else min(a, c)


def main():
    rows = {k: r for k, r in load_rows(MAIN).items() if k[1] in BASE}
    if (E4 / "rows.jsonl").exists():
        rows.update(load_rows(E4))
    arms = [a for a in ORDER if any(k[1] == a for k in rows)] + sorted({a for _, a in rows} - set(ORDER))
    L, J = ["# 100 E4 tables", "",
            "Development evidence, 099's small availability-based panel. Score = median over targets of min(pooled donor "
            "A, pooled donor C). Fit-time ratio = arm t_fit / v2 t_fit per target (v2 = 099's fit times, same GPU, 2 days "
            "earlier; today's refits of two of them were 8-12% faster, so ratios are slightly conservative).", ""], {}

    def sel(arm, split):
        return [r for (k, a), r in sorted(rows.items()) if a == arm and r.get("split") == split]

    for split in ("dev", "check"):
        L += ["## %s" % split, "",
              "| arm | n | median size (x v2) | prod pass | pre pass | donor A | donor C | donor R | score min(A,C) | "
              "donor pass | random A / C | prod Z median | fit time x v2 (median; sums) | paired d score vs v2CR (wins) |",
              "|---|---:|---|---:|---:|---|---|---|---:|---:|---|---:|---|---|"]
        J[split] = {}
        for arm in arms:
            rs = sel(arm, split)
            if not rs:
                continue
            v2 = {r["seed"]: r for r in sel("wcm_1e-3", split)}
            sr = [r["n"] / v2[r["seed"]]["n"] for r in rs if r["seed"] in v2]
            tr = [r["fit"]["t_fit"] / v2[r["seed"]]["fit"]["t_fit"] for r in rs
                  if r["seed"] in v2 and r["fit"].get("t_fit") and v2[r["seed"]]["fit"].get("t_fit")]
            sa = sum(v2[r["seed"]]["fit"]["t_fit"] for r in rs if r["seed"] in v2 and r["fit"].get("t_fit"))
            sb = sum(r["fit"]["t_fit"] for r in rs if r["seed"] in v2 and r["fit"].get("t_fit"))
            sc = [score_of(r) for r in rs]
            npass = sum(r["pass_rule"] for r in rs)
            J[split][arm] = dict(n=len(rs), score=med(sc), score_q=(q(sc, .25), q(sc, .75)), pass_rule=npass,
                                 pass_pre=sum(r["pass_pre"] for r in rs),
                                 donor_pass=sum(bool(r["donor_pass"].get("fit")) for r in rs),
                                 fit_ratio=med(tr), fit_ratio_sums=sb / sa if sa else None, size_ratio=med(sr),
                                 rec={b: [rec(r, "fit", b) for r in rs] for b in "ACR"}, seeds=[r["seed"] for r in rs])
            cr = {r["seed"]: r for r in sel("v2cr_1e-3", split)}
            dsc = [score_of(r) - score_of(cr[r["seed"]]) for r in rs
                   if r["seed"] in cr and score_of(r) is not None and score_of(cr[r["seed"]]) is not None]
            J[split][arm]["d_score_vs_v2cr"] = dsc
            L.append("| %s | %d | %g (x%s) | %d/%d | %d/%d | %s | %s | %s | %s | %d/%d | %s / %s | %s | %s; %s | %s (%d/%d) |" % (
                arm, len(rs), med([r["n"] for r in rs]), f3(med(sr)), npass, len(rs),
                sum(r["pass_pre"] for r in rs), len(rs), miqr([rec(r, "fit", "A") for r in rs]),
                miqr([rec(r, "fit", "C") for r in rs]), miqr([rec(r, "fit", "R") for r in rs]), f3(med(sc)),
                sum(bool(r["donor_pass"].get("fit")) for r in rs), len(rs),
                f3(med([rec(r, "rand", "A") for r in rs])), f3(med([rec(r, "rand", "C") for r in rs])),
                f3(med([r["prod"].get("free0_tk") for r in rs])), f3(med(tr)), f3(sb / sa if sa else None),
                f3(med(dsc)), sum(1 for x in dsc if x > 0), len(dsc)))
        L.append("")
        if split == "dev":
            # ---- the pre-stated selection rule, applied
            d = J["dev"]
            elig = {a: v for a, v in d.items() if a != "wcm_1e-3" and v["n"] == 9 and v["pass_rule"] >= 4
                    and v["score"] is not None}
            L += ["### Selection (pre-stated rule, applied to dev)", ""]
            if elig:
                best_score = max(v["score"] for v in elig.values())
                tied = [a for a, v in elig.items() if v["score"] >= best_score - 0.02]
                best = min(tied, key=lambda a: (elig[a]["fit_ratio"] or 9e9))
                rest = sorted((a for a in elig if a != best), key=lambda a: -elig[a]["score"])
                second = rest[0] if rest and elig[rest[0]]["score"] >= elig[best]["score"] - 0.05 else None
                v2cr_score = d.get("v2cr_1e-3", {}).get("score")
                confirm = [a for a in (best, second) if a and a != "v2cr_1e-3" and v2cr_score is not None
                           and elig[a]["score"] > v2cr_score]
                inel = sorted(a for a, v in d.items() if a not in elig and a != "wcm_1e-3")
                L += ["- eligible (dev production pass >= 4/9, all 9 targets): " + ", ".join(
                          "%s (%.3f, %d/9)" % (a, v["score"], v["pass_rule"]) for a, v in sorted(elig.items(), key=lambda x: -x[1]["score"])),
                      "- not eligible: " + (", ".join("%s (%s, %d/%d)" % (a, f3(d[a]["score"]), d[a]["pass_rule"], d[a]["n"]) for a in inel) or "none"),
                      "- best: **%s** (score %.3f; ties within 0.02: %s)" % (best, elig[best]["score"], ", ".join(tied)),
                      "- second within 0.05: %s" % (second or "none"),
                      "- v2CR dev score: %s" % f3(v2cr_score),
                      "- to confirm on check (must beat v2CR's dev score): %s" % (", ".join(confirm) or "none: nothing beats v2CR"), ""]
                J["selection"] = dict(best=best, second=second, tied=tied, confirm=confirm, v2cr_score=v2cr_score)
    E4.mkdir(parents=True, exist_ok=True)
    (E4 / "tables_e4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (E4 / "tables_e4.json").write_text(json.dumps(J, indent=1, default=str) + "\n")
    print("\n".join(L))

    # ---- figure: score vs production pass, per arm (dev filled, check hollow)
    sys.path.insert(0, str(ROOT / "src"))
    from analysis import style as S
    plt = S.configure_matplotlib()
    picked = set((J.get("selection") or {}).get("confirm") or [])
    fig, axes = plt.subplots(1, 2, figsize=S.FIGSIZE_WIDE, sharey=True)

    def role(a):
        if a == "wcm_1e-3":
            return S.INK_MUTED, "v2 (no donor training)"
        if a == "v2cr_1e-3":
            return S.CATEGORICAL[0], "v2CR (baseline)"
        if a in picked:
            return S.CATEGORICAL[3], "selected for check"
        return S.CATEGORICAL[1], "other E4 arms"
    for ax, split in zip(axes, ("dev", "check")):
        pts = [(a, v) for a, v in J.get(split, {}).items() if v["score"] is not None]
        by_x = {}
        for a, v in pts:
            by_x.setdefault(v["pass_rule"], []).append((a, v))
        for x0, group in by_x.items():
            group.sort(key=lambda t: t[1]["score"])
            last_label_y = -9.0
            for j, (a, v) in enumerate(group):
                x = x0 + (j - (len(group) - 1) / 2) * 0.07
                lo, hi = v["score_q"]
                colr, _ = role(a)
                ax.errorbar([x], [v["score"]], yerr=[[v["score"] - lo], [hi - v["score"]]], color=colr, marker="o",
                            ms=6, capsize=2, lw=1.0, zorder=3)
                ly = max(v["score"], last_label_y + 0.035)
                last_label_y = ly
                ax.annotate(a.replace("_1e-3", "").replace("wcm", "v2").replace("v2cr", "v2CR"), (x, v["score"]),
                            xytext=(x + 0.12, ly), textcoords="data", fontsize=7, color=S.INK_SECONDARY, va="center")
        ax.axhline(0.8, color=S.BASELINE, lw=0.8, ls="--")
        ax.set_xlabel("production post-Top-K passes (of 9)")
        ax.set_title("%s targets" % ("development (selection)" if split == "dev" else "check (confirmation)"))
        ax.set_xlim(1.5, 9.5)
    axes[0].set_ylabel("median min(donor A, donor C) recovery")
    from matplotlib.lines import Line2D
    seen = {}
    for a in arms:
        colr, lab = role(a)
        seen.setdefault(lab, colr)
    axes[1].legend(handles=[Line2D([], [], color=c, marker="o", lw=0, label=l) for l, c in seen.items()],
                   loc="center left", fontsize=8, frameon=False)
    S.style_suptitle(fig, "E4: donor-background robustness vs production pass per arm (median, IQR across targets)")
    S.save_figure(fig, E4 / "e4_recovery_vs_pass.png")


if __name__ == "__main__":
    main()
