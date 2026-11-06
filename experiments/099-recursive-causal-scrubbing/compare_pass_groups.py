"""Post-hoc comparison of production-passing/failing WCM circuits.

Targets, rather than donor draws, are the units. Descriptive only: the small
availability panel is neither random nor balanced by pass status.
"""
import json
from pathlib import Path
import statistics

from final_report import production_pass, pooled_recovery

HERE = Path(__file__).parent


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def main():
    panel = HERE / "out/panel18"
    manifest = json.loads((panel / "manifest.json").read_text())
    rows = []
    for key in manifest["targets"]:
        raw = json.loads((panel / "targets" / (key + ".json")).read_text())
        p = raw["production"]["wcm"]
        layer, kind, _ = key.split(".")
        row = dict(target=key, production_pass=production_pass(p), kind=kind,
                   band="L0-4" if int(layer) <= 4 else "L5-7" if int(layer) <= 7 else "L8-11",
                   nodes=p["n"], live={d: pooled_recovery(raw, d, "wcm") for d in ("A", "C", "R")},
                   pinned={d: pooled_recovery(raw, d, "wcm_pinned") for d in ("A", "C", "R")},
                   production={k: p.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")})
        row["donor_pass"] = bool(p.get("free0_tk") is not None and .8 <= p["free0_tk"] <= 1.5
                                and p.get("phi_sup_blind_tk", 0) >= .9
                                and all(row["live"][d] is not None and .8 <= row["live"][d] <= 1.5 for d in ("A", "C")))
        rows.append(row)
    groups = {}
    for status, label in [(True, "passing"), (False, "failing")]:
        selected = [r for r in rows if r["production_pass"] == status]
        groups[label] = dict(n=len(selected), targets=[r["target"] for r in selected],
                             donor_passes=sum(r["donor_pass"] for r in selected),
                             median_nodes=median([r["nodes"] for r in selected]),
                             live={d: median([r["live"][d] for r in selected]) for d in ("A", "C", "R")},
                             pinned={d: median([r["pinned"][d] for r in selected]) for d in ("A", "C", "R")},
                             undefined_live={d: sum(r["live"][d] is None for r in selected) for d in ("A", "C", "R")})
    strata = []
    for field, values in [("kind", ["attn", "mlp", "resid"]), ("band", ["L0-4", "L5-7", "L8-11"])]:
        for value in values:
            for status in (True, False):
                rs = [r for r in rows if r[field] == value and r["production_pass"] == status]
                strata.append(dict(field=field, value=value, production_pass=status, n=len(rs),
                                   live_C=median([r["live"]["C"] for r in rs]),
                                   pinned_C=median([r["pinned"]["C"] for r in rs])))
    recursive = []
    for name in ("development", "check"):
        directory = HERE / "out" / ("recursive_" + name + "_diagnostics")
        m = json.loads((directory / "manifest.json").read_text())
        rr = json.loads((directory / "results.json").read_text())
        parent = next(r for r in rows if r["target"] == m["target"])
        recursive.append(dict(target=m["target"], production_pass=parent["production_pass"],
                              fitted_agreement=next(1 - r["balanced_disagreement"] for r in rr if r["arm"] == "wcm_fit")))
    result = dict(analysis="post-hoc descriptive production pass/fail comparison", groups=groups,
                  strata=strata, targets=rows, recursive=recursive)
    (panel / "pass_groups.json").write_text(json.dumps(result, indent=2) + "\n")
    p, f = groups["passing"], groups["failing"]
    lines = ["# Do production-passing WCM circuits generalise better?", "",
             "Post-hoc comparison on the frozen 18-target clean panel. Groups use the original Z/A/C plus necessity "
             "pass rule; no scrubbing outcome determines group membership. Recovery pools donor outputs and empty "
             "baselines over draws before division. Table entries are medians across targets.", "",
             f"| Measurement | Production passing (n={p['n']}) | Production failing (n={f['n']}) |",
             "|---|---:|---:|"]
    for ds, label in [("A", "Activating donors"), ("C", "Contrast donors"), ("R", "Random donors")]:
        lines.append(f"| Live recovery: {label} | {p['live'][ds]:.4f} | {f['live'][ds]:.4f} |")
    for ds, label in [("A", "Activating donors"), ("C", "Contrast donors"), ("R", "Random donors")]:
        lines.append(f"| Pinned recovery: {label} | {p['pinned'][ds]:.4f} | {f['pinned'][ds]:.4f} |")
    lines += [f"| Donor-background passes | {p['donor_passes']}/{p['n']} | {f['donor_passes']}/{f['n']} |",
              f"| Median nodes | {p['median_nodes']:g} | {f['median_nodes']:g} |", "",
              "## Stratified contrast-donor results", "",
              "| Stratum | Production pass | n | Live recovery | Pinned recovery |", "|---|---|---:|---:|---:|"]
    fmt = lambda v: "--" if v is None else f"{v:.4f}"
    for r in strata:
        lines.append(f"| {r['field']}={r['value']} | {r['production_pass']} | {r['n']} | {fmt(r['live_C'])} | {fmt(r['pinned_C'])} |")
    lines += ["", "## Interpretation limits", "",
              "- These are 13 passing and 5 failing selected targets, not randomized groups. Site kind, circuit size, "
              "target and optimisation difficulty can confound the comparison.",
              "- Pinning supplies the selected members' natural activations and is a diagnostic, not evidence that "
              "the selected nodes form a self-contained computation.",
              "- Both actual recursive examples are production-passing. Their fitted agreements are "
              + ", ".join(f"{r['fitted_agreement']:.1%} ({r['target']})" for r in recursive)
              + ". There is no production-failing recursive comparator, so this run cannot establish a recursive pass/fail association.",
              "- No new model fits or GPU experiments are used in this analysis.", ""]
    (panel / "pass_groups.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
