"""103 pod cost projection for the donor-background stages (no GPU).

Local timing model (RTX 5070 Ti, one process): per target, a launch scoring n arms costs
t_shared(layer) + n x t_arm(layer), t_arm = the fit + a1 + pinned passes. Both are fitted linear in the layer (least
squares) on experiment 100's rows (the same code path, bit-identical results; 100 ran on an otherwise idle GPU and
recorded t_donor_shared_target and t_donor per variant). This experiment's own runs shared the GPU with another
agent's fits (arms-102, ~85% busy), so their times are reported as a contention factor (measured / model on the same
targets), not used for the model. Public models: GPT-2 small is taken as TuringLLM-sized (both 12-layer ~124M);
Pythia-70m from its smoke rows divided by the contention factor of the TuringLLM smoke rows of the same session.

Pod: 8 H100 x P=6 processes (48). A pod process is not the local process: 062's 15k run measured, per process under
MPS P=6, scoring (forward only, batch 16, the donor test's kind of work) ~1.6x FASTER than this GPU idle (062
out_full status t_eval / 2 vs 100 t_prod, same layers), and the fits ~1.6x SLOWER (vs 093 v1 t_fit). The projection
gives both: optimistic = local x 0.6, pessimistic = local x 1.6, then / 48 processes.

  python experiments/103-donor-stage/cost.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E100 = HERE.parent / "100-scrub-frontier" / "out" / "main" / "rows.jsonl"
D94 = HERE.parent / "094-tl-v2-records"
VARS = ("fit", "a1", "pinned")


def rows(p):
    out = []
    for f in sorted(p.parent.glob(p.name)):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r and not r.get("skip"):
                out.append(r)
    return out


def linfit(pts):
    n = len(pts)
    mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else 0.0
    return my - b * mx, b


def mean_layer(p, default=5.5):                                     # the model is linear: the mean layer suffices
    ks = p.read_text().split() if p.exists() else []
    return sum(int(k.split(".")[0]) for k in ks) / len(ks) if ks else default


def main():
    sh, arm, seen = [], [], set()
    for r in rows(E100):
        if r["arm"] in ("wcm_1e-3", "v2cr_1e-3"):
            if r["seed"] not in seen:
                seen.add(r["seed"])
                sh.append((r["layer"], r["t_donor_shared_target"]))
            arm.append((r["layer"], sum(r["t_donor"][v] for v in VARS)))
    a_s, b_s = linfit(sh)
    a_v, b_v = linfit(arm)

    def T(layer, n_arm):
        return (a_s + b_s * layer) + n_arm * (a_v + b_v * layer)

    def contention(rs):
        by = {}
        for r in rs:
            by.setdefault(r["seed"], []).append(r)
        m = [(rr[0]["t_target_total"], T(rr[0]["layer"], len(rr))) for rr in by.values()]
        return (sum(x for x, _ in m) / sum(y for _, y in m)) if m else None

    c_eq = contention(rows(HERE / "out" / "equiv" / "rows*.jsonl"))
    tl_smoke = rows(HERE / "smoke_chain" / "out_turingllm" / "out_v2" / "donor" / "main" / "rows*.jsonl")
    c_sm = contention(tl_smoke)
    py = rows(HERE / "smoke_chain" / "out_pythia-70m" / "donor" / "rows*.jsonl")
    py_t = (sum(r["t_target_total"] for r in py) / len(py)) if py else None
    py_idle = py_t / c_sm if (py_t and c_sm) else None
    l8k, l190 = mean_layer(D94 / "targets_tl8k.txt"), mean_layer(D94 / "targets_tl8k_sweep190.txt")
    print("local model (100, idle GPU; %d targets, %d target x arm): shared %.2f + %.2f L s, per arm (fit + a1 + pinned)"
          " %.2f + %.2f L s" % (len(sh), len(arm), a_s, b_s, a_v, b_v))
    print("mean layer: 8k %.2f, sweep190 %.2f; contention factor (measured / model): equivalence run %s, TL smoke %s"
          % (l8k, l190, "%.2f" % c_eq if c_eq else "-", "%.2f" % c_sm if c_sm else "-"))
    print("Pythia-70m: smoke %s s per target (contended), ~%s s idle" % (
        "%.1f" % py_t if py_t else "-", "%.1f" % py_idle if py_idle else "-"))
    jobs = [("T11 donor 8k main", 8000 * T(l8k, 1)),
            ("T12 donor sweep 190 x 13 arms", 190 * T(l190, 13)),
            ("T12 noR 190 x 1", 190 * T(l190, 1)),
            ("GPT-2 small main 512", 512 * T(5.5, 1))]
    if py_idle:
        jobs += [("Pythia-70m main 512", 512 * py_idle), ("Pythia-70m U 128", 128 * py_idle)]
    tot = sum(s for _, s in jobs)
    out = dict(model=dict(shared=[a_s, b_s], per_arm=[a_v, b_v], contention_equiv=c_eq, contention_smoke=c_sm,
                          pythia_per_target_idle=py_idle, mean_layer_8k=l8k, mean_layer_190=l190), jobs={})
    print("%-32s %12s %16s %16s" % ("job", "local GPU-h", "pod wall h x0.6", "pod wall h x1.6"))
    for n_, s in jobs + [("TOTAL", tot)]:
        out["jobs"][n_] = dict(local_gpu_h=s / 3600, pod_wall_h_opt=s * 0.6 / 48 / 3600,
                               pod_wall_h_pess=s * 1.6 / 48 / 3600)
        print("%-32s %12.1f %16.2f %16.2f" % (n_, s / 3600, s * 0.6 / 48 / 3600, s * 1.6 / 48 / 3600))
    print("per circuit at the 8k mean layer: %.1f s local idle (shared %.1f + arm %.1f); x13-arm sweep target %.1f s"
          % (T(l8k, 1), a_s + b_s * l8k, a_v + b_v * l8k, T(l190, 13)))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "cost.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
