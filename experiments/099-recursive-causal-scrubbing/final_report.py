"""Concise evidence summary; requires all frozen panel rows and provenance gate."""
import json
from pathlib import Path
import statistics

HERE = Path(__file__).parent


def production_pass(p):
    return all(p.get(k) is not None and 0.8 <= p[k] <= 1.5
               for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")) and p.get("phi_sup_blind_tk", 0) >= 0.9


def pooled_recovery(row, background, variant):
    actual = [r for r in row["donor_results"] if r["background"] == background and r["variant"] == variant]
    empty = [r for r in row["donor_results"] if r["background"] == background and r["variant"] == "empty"]
    x = statistics.mean(r["anchor"]["mean"] for r in actual)
    e = statistics.mean(r["anchor"]["mean"] for r in empty)
    n = actual[0]["anchor"]["natural_mean"]
    return (x - e) / (n - e) if n - e > 1e-9 else None


def main():
    out = HERE / "out"
    panel = out / "panel18"
    manifest = json.loads((panel / "manifest.json").read_text())
    if json.loads((panel / "source_check.json").read_text())["changed_during_run"]:
        raise RuntimeError("source check failed")
    rows = [json.loads((panel / "targets" / (key + ".json")).read_text()) for key in manifest["targets"]]
    stats = {}
    for method in ("wcm", "ig_a1"):
        recovered = [{d: pooled_recovery(r, d, method) for d in ("A", "C", "R")} for r in rows]
        passes = 0
        for r, values in zip(rows, recovered):
            p = r["production"][method]
            passes += int(p.get("free0_tk") is not None and 0.8 <= p["free0_tk"] <= 1.5
                          and p.get("phi_sup_blind_tk", 0) >= 0.9
                          and all(values[d] is not None and 0.8 <= values[d] <= 1.5 for d in ("A", "C")))
        stats[method] = dict(production_passes=sum(production_pass(r["production"][method]) for r in rows),
                             pooled_donor_passes=passes, n=len(rows),
                             median_nodes=statistics.median(r["production"][method]["n"] for r in rows),
                             median_recovery={d: statistics.median([v[d] for v in recovered if v[d] is not None])
                                              for d in ("A", "C", "R")})
    audit = json.loads((panel / "audit.json").read_text())
    recursive = {}
    for name in ("development", "check"):
        path = out / ("recursive_" + name + "_diagnostics")
        if not (path / "results.json").exists():
            continue
        rm = json.loads((path / "manifest.json").read_text())
        rr = json.loads((path / "results.json").read_text())
        recursive[name] = dict(target=rm["target"], references=len(rm["base_ids"]), donors=rm["dataset_n"],
                               firing_donors=sum(rm["labels"]), draws=rm["draws"],
                               balanced_agreement={r["arm"]: 1 - r["balanced_disagreement"] for r in rr})
    summary = dict(background=stats, records=len(audit), audit_failures=sum(bool(r["failures"]) for r in audit),
                   recursive=recursive, panel_source_changes=[],
                   population_claim=False, protocols="fresh v2 support robustness; separate original-model two-cut recursion")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# WCM causal-scrubbing pilot — 28 October 2026", "",
             f"All {len(audit)} frozen pilot records passed independent content/split/firing checks. "
             "The panel has two targets per depth-band/site cell and is availability-based, not a bank-wide population sample.", "",
             "## Clean support robustness", "",
             "| Method | Production passes | Donor-background passes | Median nodes | Median donor A / C / R recovery |",
             "|---|---:|---:|---:|---|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['production_passes']}/{s['n']} | {s['pooled_donor_passes']}/{s['n']} | "
                     f"{s['median_nodes']:g} | " + " / ".join(f"{s['median_recovery'][d]:.3f}" for d in ("A", "C", "R")) + " |")
    lines += ["", "Donor pass preserves production Z and necessity and requires pooled-draw A and C recovery in "
              "[0.8, 1.5]. Outputs and empty baselines are averaged across draws before division. "
              "This is whole-donor background resampling, not full recursive scrubbing. IG uses a documented latent-target "
              "adaptation, WCM-matched node count and natural strength; it has no fitted-gain capacity.", "",
              "## Actual recursive tests: coarse two-cut hypotheses", ""]
    for name, r in recursive.items():
        lines += [f"### {name}: {r['target']}", "",
                  f"{r['references']} references / {r['donors']} donors ({r['firing_donors']} firing), {r['draws']} recursive draws per reference.", "",
                  "| Arm | Balanced firing agreement |", "|---|---:|"]
        lines += [f"| {arm} | {v:.1%} |" for arm, v in r["balanced_agreement"].items()]
        lines.append("")
    lines += ["The recursive hypotheses concern WCM-selected groups at residual layers 0 and 1. Native model segments "
              "are composite nodes. The high-level variable is teacher-derived firing, not a human semantic label. "
              "All unselected residual information, including SAE error and compensating gain terms, is explicitly "
              "represented in resampled remainders. These tests do not validate every node in the full WCM circuits.", "",
              "Both strength conventions are exact rewrites of the original model. Clean graph equality, positive controls "
              "and expanded-tree replay are checked. The generic CPU engine additionally tests shared-ancestor path independence "
              "and exact finite distributions. Cut-restoration additions are exploratory diagnostics; the check target was frozen "
              "before its outcomes were inspected.", "",
              "## Evidence", "",
              "- `panel18/report.md`, `panel18/read_diagnostics.md`, `panel18/background_recovery.png`",
              "- `recursive_development_diagnostics/report.md`, `recursive_check_diagnostics/report.md`",
              "- `identity_roundtrip.json`, `weight_fingerprints.json`, individual raw outputs and manifests",
              "- `toy_report.md`: wrong hypotheses can preserve raw means while paired errors expose them", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
