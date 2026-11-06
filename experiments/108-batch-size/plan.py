"""Experiment 108 sizing (CPU, stdlib): after the timing stage, choose N (the first N targets of targets.txt) so the
remaining work fits before DEADLINE (env, epoch s), keeping every arm: fewer targets rather than fewer arms, N in
[N_MIN 8, 16]. Writes out/targets_run.txt and out/plan.json, prints the plan.

Cost model, from the timing fits (B4 and B16 on the deepest target 11.resid.10297 and the shallow 4.attn.35016, in
out/<arm>/b108_fits.jsonl): per step, t_fit / steps = o + accum * u (o = per-step overhead incl. the amortised fit
set-up, u = one micro-chunk of 4 contexts: its forwards + backwards). B4 (accum 1) and B16 (accum 4) on the same
target give u = (r16 - r4) / 3, o = r4 - u. o and u are interpolated linearly in the target's upstream site count
(floored at 25% of the shallow timing target's value). A fit = steps * (o + accum * u) x MARGIN (1.15); scoring =
B4's measured t_eval interpolated the same way; per process start-up = measured (the timing driver walls minus their
rows' ctx + fit + eval time, out/stages.tsv), default 90 s; donor test (DONOR auto | 1 | 0) = the timing donor rows'
t_target_total per target per arm + one start-up per arm. DONOR=auto runs it only when it still fits after N is set.
PRIOR=1 (DRY_RUN): no timing data; o / u from latent_pilot's production-equivalent dev48 fits
(dev48_geo_x3.15_warmup_censor B4 t_fit, assumed u = 0.85 x t_fit / steps), clearly labelled.
"""
import json
import os
import statistics as S
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms as A  # noqa: E402

OUT = Path(os.environ.get("OUT108", str(HERE / "out")))
N_MIN, N_MAX = 8, 16
MARGIN = 1.15
RESERVE_S = 600                       # analysis + slack at the end
PRIOR = os.environ.get("PRIOR") == "1"
PRIOR_RUN = HERE.parents[0] / "107-true-baselines" / "latent_pilot" / "out" / "dev48_geo_x3.15_warmup_censor" / "main"


def jl(p):
    rows = []
    if Path(p).exists():
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return rows


def done(arm):
    """Targets whose status row is final (ok, skipped or rejected) in OUT/<arm>."""
    out = set()
    for p in sorted((OUT / arm / "main").glob("status.shard*.jsonl")):
        for r in jl(p):
            if "error" not in r:
                out.add(r["seed"])
    return out


def interp(pts, n):
    """Linear in site count through the two (sites, value) points, floored at 25% of the shallower point's value."""
    (n1, v1), (n2, v2) = sorted(pts)
    v = v1 + (v2 - v1) * (n - n1) / (n2 - n1) if n2 != n1 else v1
    return max(v, 0.25 * v1)


def model_measured(timing):
    fits = {a: {r["seed"]: r for r in jl(OUT / a / "b108_fits.jsonl")} for a in A.TIMING_ARMS}
    o_pts, u_pts, e_pts = [], [], []
    for k in timing:
        r4, r16 = fits["B4"].get(k), fits["B16"].get(k)
        if not r4 or not r16:
            raise SystemExit("plan: timing fit missing for %s (B4 %s, B16 %s)" % (k, bool(r4), bool(r16)))
        s4, s16 = r4["t_fit"] / r4["steps"], r16["t_fit"] / r16["steps"]
        u = max((s16 - s4) / (r16["accum"] - r4["accum"]), 1e-3)
        o = max(s4 - r4["accum"] * u, 0.0)
        n = A.n_sites(k)
        o_pts.append((n, o)); u_pts.append((n, u))
        st = {r["seed"]: r for r in jl(OUT / "B4" / "main" / "status.shard0.jsonl") if "error" not in r}
        e_pts.append((n, float((st.get(k) or {}).get("t_eval") or 10.0)))
    # start-up per process: stage walls minus the work their rows account for
    starts = []
    for line in (OUT / "stages.tsv").read_text().splitlines() if (OUT / "stages.tsv").exists() else []:
        f = line.split("\t")
        if len(f) >= 5 and f[0] in ("timing_B4", "timing_B16") and f[3] == "0":
            arm = f[0].split("_")[1]
            st = [r for r in jl(OUT / arm / "main" / "status.shard0.jsonl") if r.get("seed") in timing and "error" not in r]
            work = sum(float(r.get("t_ctx") or 0) + float(r.get("t_fit") or 0) + float(r.get("t_eval") or 0) for r in st)
            starts.append(max(float(f[2]) - float(f[1]) - work, 0.0))
    startup = S.median(starts) if starts else 90.0
    don = [float(r["t_target_total"]) for r in jl(OUT / "B4" / "donor" / "rows.jsonl") if r.get("t_target_total")]
    donor_t = S.median(don) if don else None
    return dict(o=o_pts, u=u_pts, eval=e_pts, startup=startup, donor_t=donor_t, source="measured (timing stage)")


def model_prior():
    st = {r["seed"]: r for r in jl(PRIOR_RUN / "status.shard0.jsonl") if r.get("t_fit")}
    pts = [(A.n_sites(k), st[k]) for k in ("4.attn.35016", "11.resid.10297") if k in st]
    if len(pts) < 2:
        raise SystemExit("plan: PRIOR run %s lacks the timing targets" % PRIOR_RUN)
    u_pts = [(n, 0.85 * r["t_fit"] / 400) for n, r in pts]
    o_pts = [(n, 0.15 * r["t_fit"] / 400) for n, r in pts]
    e_pts = [(n, float(r.get("t_eval") or 10.0)) for n, r in pts]
    return dict(o=o_pts, u=u_pts, eval=e_pts, startup=90.0, donor_t=15.0,
                source="PRIOR (dry run: latent_pilot dev48 B4-equivalent fits; u = 0.85 t/step assumed)")


def main():
    panel = [k for k in (HERE / "targets.txt").read_text().split() if k]
    timing = panel[:2]
    m = model_prior() if PRIOR else model_measured(timing)
    now = time.time()
    deadline = float(os.environ["DEADLINE"])
    budget = deadline - now - RESERVE_S
    donor_mode = os.environ.get("DONOR", "auto")

    def fit_cost(arm, k):
        a = A.arm(arm)
        n = A.n_sites(k)
        return a["steps"] * (interp(m["o"], n) + a["accum"] * interp(m["u"], n)) * MARGIN + interp(m["eval"], n)

    def cost(n_t, with_donor):
        ks = panel[:n_t]
        fits = 0.0
        for arm in A.ORDER:
            todo = [k for k in ks if k not in done(arm)]
            if todo:
                fits += m["startup"] + sum(fit_cost(arm, k) for k in todo)
        don = 0.0
        if with_donor:
            don = len(A.ORDER) * (m["startup"] + n_t * (m["donor_t"] or 15.0))
        return fits, don

    n_fit = N_MIN
    for n_t in range(N_MIN, N_MAX + 1):
        if sum(cost(n_t, donor_mode == "1")) <= budget:
            n_fit = n_t
    fits, don = cost(n_fit, True)
    fits_ok = fits + (don if donor_mode == "1" else 0.0) <= budget
    donor = donor_mode == "1" or (donor_mode == "auto" and fits + don <= budget)
    per_arm = {arm: round(sum(fit_cost(arm, k) for k in panel[:n_fit] if k not in done(arm)) / 60, 1) for arm in A.ORDER}
    plan = dict(N=n_fit, targets=panel[:n_fit], donor=bool(donor), budget_min=round(budget / 60, 1),
                predicted_fit_min=round(fits / 60, 1), predicted_donor_min=round(don / 60, 1),
                per_arm_fit_min=per_arm, fits_within_budget=bool(fits_ok), margin=MARGIN, model=m,
                deadline=deadline, planned_at=round(now))
    suffix = ".dry" if PRIOR else ""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("targets_run%s.txt" % suffix)).write_text("\n".join(panel[:n_fit]) + "\n")
    (OUT / ("plan%s.json" % suffix)).write_text(json.dumps(plan, indent=1) + "\n")
    print("plan (%s): N = %d targets, donor %s; budget %.0f min, predicted fits %.0f min + donor %.0f min%s" % (
        m["source"], n_fit, "yes" if donor else "no", budget / 60, fits / 60, don / 60,
        "" if fits_ok else "  ** WARNING: even N = %d does not fit; the deadline will cut the last arms **" % N_MIN))
    print("  per-arm fit minutes: %s" % per_arm)
    print("  targets: %s" % " ".join(panel[:n_fit]))
    if not donor:
        print("DONOR=0")
    else:
        print("DONOR=1")


if __name__ == "__main__":
    main()
