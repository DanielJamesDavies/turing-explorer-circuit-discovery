"""Arena C (DAN-196): faithfulness-vs-size curves, MIB-style areas and the smallest passing sizes, SFC vs WCM, each
under our evaluation and under SFC's. CPU only (reads the score files).

Sources
  SFC, SFC's eval   out/sfc/faithfulness/summary_features_features_wo_errs_*.json (their get_fcs: "features" = errors
                    as nodes, "features_wo_errs" = errors removed), n_nodes = SFC's count (layers >= 2)
  SFC, our eval     out/cross/sfc_ours.jsonl (score_sfc_ours.py: converted circuits, alpha = 1)
  WCM, our eval     <wcm_out>/eval.jsonl (wcm_marks.run), donor rows <wcm_out>/donor/rows.jsonl
  WCM, SFC's eval   out/cross/wcm_sfc.jsonl (score_wcm_sfc.py: membership at alpha = 1, errors kept / removed)

Definitions [ours, stated]
  size (main unit)  distinct (site, latent) members over the 18 noembed sites, positions collapsed (107 rule 6; for
                    SFC the converted circuit's members at the same threshold)
  native unit       SFC's node count at layers >= 2: (position, latent) feature nodes + error nodes (WCM: members x T,
                    + 12 x T error nodes when errors are kept)
  our faithfulness  for the curve: min(Z, A, C) on the task metric (the pass rule's binding term); None counts as 0
  pass, ours        wcm_eval.passes_task (Z, A, C in [0.75, 1.5], necessity >= 0.9) and not vacuous; resample_pass adds
                    donor recovery A and C in [0.75, 1.5]
  pass, SFC's eval  faithfulness in [0.75, 1.5] [ours: SFC defines no pass rule]
  area              specs/evaluation.md 3.3-3.4: grid 10..1e5 in 1-2-5 steps (13 points), reading(g) = f of the largest
                    circuit of size <= g (0 below the smallest), trapezoid on log10 g, divided by the width (nAUC)
Writes out/figs/curves_<task>.{png,pdf}, out/cross/summary.json, and prints the markdown tables for the README.

  ./.venv/bin/python experiments/107-true-baselines/arena_c/curves.py [--wcm_out out/wcm_tasks_zacr]
"""
import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from analysis import style as S  # noqa: E402

BAND = (0.75, 1.5)
NEC_MIN = 0.9
GRID = [m * 10 ** e for e in range(1, 6) for m in (1, 2, 5)][:13]          # 10, 20, 50, ..., 1e5
TASKS = ("simple", "within_rc", "rc", "nounpp")


def jl(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def num(v):
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


def ours_f(r):
    vals = [num(r.get("task_" + k)) for k in ("free0", "freeM_topk", "freeN_topk")]
    return 0.0 if any(v is None for v in vals) else min(vals)


def ours_pass(r):
    if r.get("task_vacuous"):
        return False
    vals = [num(r.get("task_" + k)) for k in ("free0", "freeM_topk", "freeN_topk")]
    nec = num(r.get("task_nec"))
    return all(v is not None and BAND[0] <= v <= BAND[1] for v in vals) and nec is not None and nec >= NEC_MIN


def in_band(v):
    v = num(v)
    return v is not None and BAND[0] <= v <= BAND[1]


def nauc(points):
    """points [(size, f)] -> (nAUC on the log grid, reached range)."""
    pts = sorted((s, f) for s, f in points if s is not None and s > 0)
    if not pts:
        return None, None

    def reading(g):
        best = [f for s, f in pts if s <= g]
        return best[-1] if best else 0.0
    xs, rs = [math.log10(g) for g in GRID], [reading(g) for g in GRID]
    area = sum((xs[i + 1] - xs[i]) * (rs[i] + rs[i + 1]) / 2 for i in range(len(GRID) - 1))
    return area / (xs[-1] - xs[0]), (pts[0][0], pts[-1][0])


def smallest(points):
    ok = [s for s, p in points if p]
    return min(ok) if ok else None


def load(wcm_out: Path):
    sfc_summary = sorted(p for p in glob.glob(str(HERE / "out" / "sfc" / "faithfulness" /
                                                  "summary_features_features_wo_errs_*.json")) if "_F4" not in p)
    sfc_own = json.load(open(sfc_summary[-1]))["results"] if sfc_summary else {}
    sfc_ours = jl(HERE / "out" / "cross" / "sfc_ours.jsonl")
    wcm_ours = [r for r in jl(wcm_out / "eval.jsonl") if r.get("data_mode") == "sfc"]
    donor = {r["circuit"]: r for r in jl(wcm_out / "donor" / "rows.jsonl")}
    wcm_sfc = jl(HERE / "out" / "cross" / "wcm_sfc.jsonl")
    return sfc_own, sfc_ours, wcm_ours, donor, wcm_sfc


def series(task, sfc_own, sfc_ours, wcm_ours, donor, wcm_sfc, wcm_out):
    """{(evaluation, label): [dict(size, native, f, pass, resample_pass)]}."""
    S_ = defaultdict(list)
    # SFC under our evaluation; distinct latents per threshold (for SFC's own curve's x too)
    lat_of = {}
    for r in sfc_ours:
        if r["task"] != task:
            continue
        lat_of[round(r["threshold"], 9)] = r["n_latents"]
        S_[("ours", "SFC, means %s" % r["means"])].append(dict(
            size=r["n_latents"], native=r["native"]["sfc_nodes_ge2"], f=ours_f(r), passed=ours_pass(r),
            rpass=bool(r.get("resample_pass")), th=r["threshold"]))
    for setting, lab in (("features", "SFC (errors as nodes)"), ("features_wo_errs", "SFC (errors removed)")):
        for p in (sfc_own.get(setting, {}).get(task, {}).get("curve") or []):
            S_[("sfc", lab)].append(dict(size=lat_of.get(round(p["threshold"], 9)), native=p["n_nodes"],
                                         f=p["faithfulness"], passed=in_band(p["faithfulness"]), th=p["threshold"],
                                         completeness=p.get("completeness")))
    for r in wcm_ours:
        if r["sfc_task"] != task:
            continue
        d = donor.get(r["circuit"])
        S_[("ours", "WCM, means %s" % r.get("means", "pooled"))].append(dict(
            size=r["n_latents"], native=r["sfc_nodes_ge2"], f=ours_f(r), passed=ours_pass(r),
            rpass=bool(d and d.get("resample_pass")), lam=r["lam"],
            recovery=None if d is None else {k: v["fit"] for k, v in d["recovery"].items()}))
    for r in wcm_sfc:
        if r["task"] != task or "/%s/" % wcm_out.name not in r["circuit"].replace("\\", "/"):
            continue
        lab = "WCM %s, errors %s" % (r["means"], "kept" if r["errors"] == "errors_kept" else "removed")
        S_[("sfc", lab)].append(dict(size=r["n_latents"], native=r["sfc_nodes_with_errors"] if r["errors"] ==
                                     "errors_kept" else r["sfc_nodes_without_errors"], f=r["faithfulness"],
                                     passed=in_band(r["faithfulness"]), lam=r["lam"],
                                     completeness=r["completeness"]))
    return S_


def plot(task, S_, path):
    plt = S.configure_matplotlib()
    fig, axs = plt.subplots(2, 2, figsize=S.panel_figsize(2, 2))
    labels = sorted({lab for _, lab in S_})
    col = {lab: S.CATEGORICAL[i % len(S.CATEGORICAL)] for i, lab in enumerate(
        sorted(labels, key=lambda x: (not x.startswith("SFC"), x)))}
    for j, ev in enumerate(("ours", "sfc")):
        for i, unit in enumerate(("size", "native")):
            ax = axs[i][j]
            for (e, lab), pts in sorted(S_.items()):
                if e != ev:
                    continue
                pts = sorted((p for p in pts if p.get(unit)), key=lambda p: p[unit])
                if not pts:
                    continue
                ls = "--" if "removed" in lab or "per_position" in lab else "-"
                ax.plot([p[unit] for p in pts], [p["f"] for p in pts], ls, marker="o", ms=3.5, color=col[lab],
                        label=lab)
            ax.set_xscale("log")
            ax.axhspan(BAND[0], BAND[1], color=S.GRID, alpha=0.5, lw=0)
            ax.set_ylim(-0.2, 1.3)
            ax.set_xlabel("distinct latents (18 sites)" if unit == "size" else "SFC nodes (layers >= 2)")
            ax.set_ylabel("min(Z, A, C)" if ev == "ours" else "SFC faithfulness")
            ax.set_title("%s eval, %s" % ("our" if ev == "ours" else "SFC's", "latents" if unit == "size" else
                                          "SFC's unit"))
            S.styled_legend(ax, fontsize=7.5, loc="lower right")
    S.style_suptitle(fig, "Arena C, %s: SFC vs WCM" % task)
    S.save_figure(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wcm_out", default=str(HERE / "out" / "wcm_tasks_zacr"))
    a = ap.parse_args()
    wcm_out = Path(a.wcm_out)
    # the main run keeps the plain names; any other WCM run (wcm_tasks_s1200, warm_path, ...) gets its own suffix
    tag = "" if wcm_out.name == "wcm_tasks_zacr" else "_" + wcm_out.name
    data = load(wcm_out)
    summary = {}
    lines = ["| task | method | evaluation | points | range (latents) | nAUC | smallest pass | smallest resample pass |",
             "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        S_ = series(task, *data, wcm_out)
        if not S_:
            continue
        plot(task, S_, HERE / "out" / "figs" / ("curves_%s%s.png" % (task, tag)))
        summary[task] = {}
        for (ev, lab), pts in sorted(S_.items()):
            area, rng = nauc([(p["size"], p["f"]) for p in pts])
            sp = smallest([(p["size"], p["passed"]) for p in pts])
            rp = smallest([(p["size"], p.get("rpass")) for p in pts]) if ev == "ours" else None
            summary[task]["%s | %s" % (ev, lab)] = dict(n=len(pts), range=rng, nauc=area, smallest_pass=sp,
                                                        smallest_resample_pass=rp, points=pts)
            lines.append("| %s | %s | %s | %d | %s | %s | %s | %s |" % (
                task, lab, "ours" if ev == "ours" else "SFC's", len(pts),
                "-" if rng is None else "%d-%d" % rng, "-" if area is None else "%.3f" % area,
                "-" if sp is None else "{:,}".format(sp), "-" if ev != "ours" else ("-" if rp is None
                                                                                     else "{:,}".format(rp))))
    (HERE / "out" / "cross").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(HERE / "out" / "cross" / ("summary%s.json" % tag), "w"), indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
