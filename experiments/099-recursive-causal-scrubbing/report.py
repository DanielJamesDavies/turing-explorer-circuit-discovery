"""Summarise completed pilot artifacts without loading the model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).parent


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value):
    return "undefined" if value is None else f"{value:.3f}"


def passes(prod):
    return all(prod.get(k) is not None and 0.8 <= prod[k] <= 1.5
               for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")) and prod.get("phi_sup_blind_tk", 0) >= 0.9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=HERE / "out/first3")
    args = p.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    rows = [json.loads(f.read_text()) for f in sorted((args.run / "targets").glob("*.json"))]
    by_key = {r["target"]: r for r in rows}
    rows = [by_key[k] for k in manifest["targets"] if k in by_key]
    lines = ["# Clean WCM support-robustness pilot", "",
             f"Completed {len(rows)} / {len(manifest['targets'])} frozen targets; "
             f"{manifest['steps']} fitting steps, {manifest['draws']} donor draws per background.", "",
             "These are whole-donor background tests, not the general recursive algorithm. "
             "A/C/R denote activating, held-out contrast, and random corpus donors. "
             "Ratios are medians over donor draws, normalized by each draw's empty-circuit baseline. "
             "Undefined ratios are excluded from a median and counted separately in the JSON artifacts.", "",
             "| Target | Nodes | Production Z / A / C | Necessary | Production pass | Donor A / C / R | Pinned A / C / R |",
             "|---|---:|---|---:|---|---|---|"]
    summary = []
    for row in rows:
        prod = row["production"]["wcm"]
        groups = defaultdict(list)
        for r in row["donor_results"]:
            groups[r["background"], r["variant"]].append(r)
        recoveries = {f"{ds}/{variant}": median([r["recovery"] for r in rs])
                      for (ds, variant), rs in groups.items()}
        donor_pass = []
        for d in range(manifest["draws"]):
            pair = [next(r["recovery"] for r in groups[ds, "wcm"] if r["draw"] == d) for ds in ("A", "C")]
            donor_pass.append(bool(prod.get("free0_tk") is not None and 0.8 <= prod["free0_tk"] <= 1.5
                                   and prod.get("phi_sup_blind_tk", 0) >= 0.9
                                   and all(v is not None and 0.8 <= v <= 1.5 for v in pair)))
        lines.append(f"| {row['target']} | {prod['n']} | "
                     + " / ".join(fmt(prod.get(k)) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk"))
                     + f" | {fmt(prod.get('phi_sup_blind_tk'))} | {passes(prod)} | "
                     + " / ".join(fmt(recoveries[f"{ds}/wcm"]) for ds in ("A", "C", "R")) + " | "
                     + " / ".join(fmt(recoveries[f"{ds}/wcm_pinned"]) for ds in ("A", "C", "R")) + " |")
        summary.append(dict(target=row["target"], production_pass=passes(prod), recoveries=recoveries,
                            per_draw_donor_passes=sum(donor_pass), draws=len(donor_pass)))
    lines += ["", "## Methods under the same intervention", "",
              "IG is the documented latent-target adaptation of the aggregate SFC-style ranking, "
              "truncated to WCM's node count, at natural strength. It has no fitted coefficients. "
              "A difference here does not isolate selection from gain fitting.", "",
              "| Target | Method | Nodes | Production pass | Donor A / C / R |",
              "|---|---|---:|---|---|"]
    for row in rows:
        for name, prod in row["production"].items():
            recoveries = [median([r["recovery"] for r in row["donor_results"]
                                  if r["background"] == ds and r["variant"] == name]) for ds in ("A", "C", "R")]
            lines.append(f"| {row['target']} | {name} | {prod.get('n')} | {passes(prod)} | "
                         + " / ".join(fmt(v) for v in recoveries) + " |")
    lines += ["", "## Identity and numerical sensitivity", "",
              "Self-donor identity is evaluated at natural strength. Fitted gains need not preserve identity. "
              "Maximum all-position errors can include Top-K membership discontinuities; anchor errors are also shown.", "",
              "| Target | Self empty: max absolute / anchor NRMSE | Self alpha=1: max absolute / anchor NRMSE | Fitted self mean / natural mean |",
              "|---|---|---|---|"]
    for row in rows:
        i = row["identity"]
        lines.append(f"| {row['target']} | {fmt(i['empty']['max_abs'])} / {fmt(i['empty']['anchor']['nrmse'])} | "
                     f"{fmt(i['wcm_a1']['max_abs'])} / {fmt(i['wcm_a1']['anchor']['nrmse'])} | "
                     f"{fmt(i['wcm']['anchor']['mean'])} / {fmt(i['wcm']['anchor']['natural_mean'])} |")
    lines += ["", "## Scope", "",
              "The panel is selected by artifact availability and frozen balancing rules; it is not a population estimate. "
              "The same WCM nodes at alpha=1 are a gain-dependence diagnostic, not a separately optimized unweighted baseline. "
              "Pinned member values are a diagnostic and do not establish a closed mechanism. "
              "Live SAE reconstruction errors remain in all donor-background arms. "
              "Per-context outputs, false firing, ReLU(pre-activation), donor indices, and undefined denominators are retained in target JSON files.", ""]
    (args.run / "report.md").write_text("\n".join(lines))
    (args.run / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
