"""Experiment 108 analysis (CPU; reads out/<arm>/, writes markdown to stdout).

Per target x arm: members n, held-out strongest Z / A / C / necessity, pass (production rule, eval.wcm_eval.passes;
vacuous rows, wcm_eval.is_vacuous, shown as "vac" and dropped from every denominator), held-out mid-band pass, induce
(phi_cf_alpha_blind_tk, strongest row), donor recovery A / C / R and resample pass when the donor test ran, fit wall time,
s / step, torch peak reserved MiB, and the zero term's late rebound (max after its minimum - minimum, training curve).
Per arm: the roll-up and the change against B4 on the targets both scored. Membership Jaccard for the seed pairs
(B4 / B4s2, B48 / B48s2) and B4 against each batch arm. Then the pre-stated reading rule (README), with the numbers it
names; the reading itself is left to the reader.

  PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/108-batch-size/analyze.py > experiments/108-batch-size/out/analysis.md
"""
import json
import os
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import arms as A  # noqa: E402

OUT = Path(os.environ.get("OUT108", str(HERE / "out")))
PARITY = REPO / "experiments" / "107-true-baselines" / "latent_pilot" / "out" / "dev48_geo_x3.15_warmup_censor" / "main" \
    / "circuits"
PAIRS = [("B4", "B4s2"), ("B48", "B48s2"), ("B4", "B48"), ("B4", "B16m"), ("B4", "B16")]
RULE = ("a larger batch helps if it raises the held-out pass count, or lowers median size at an equal pass count, or "
        "raises seed overlap, without lowering the others; differences within +-1 target of 16 are noise.")


def jl(p):
    rows = []
    for q in sorted(Path(p).parent.glob(Path(p).name)) if "*" in str(p) else [Path(p)]:
        if q.exists():
            for line in open(q):
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return rows


def last(rows, key):
    out = {}
    for r in rows:
        if "error" not in r:
            out[key(r)] = r
    return out


def f2(x):
    return "-" if not isinstance(x, (int, float)) else "%.2f" % x


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return S.median(xs) if xs else None


def members(path):
    import torch
    if not path.exists():
        return None
    c = torch.load(path, weights_only=False, map_location="cpu")
    if c is None:
        return None
    return {(n.feature_id.layer, n.feature_id.kind, n.feature_id.index) for n in c.nodes.values()
            if n.metadata.get("role") != "seed" and n.feature_id is not None}


def rebound(curve):
    z = ((curve or {}).get("terms") or {}).get("zero")
    if not z:
        return None
    i = min(range(len(z)), key=lambda j: z[j])
    return max(z[i:]) - z[i]


def load_arm(arm):
    d = OUT / arm / "main"
    ev = last(jl(d / "eval.shard*.jsonl"), lambda r: (r["seed"], r.get("held")))
    st = last(jl(d / "status.shard*.jsonl"), lambda r: r["seed"])
    tr = last(jl(d / "train.shard*.jsonl"), lambda r: r["seed"])
    fx = last(jl(OUT / arm / "b108_fits.jsonl"), lambda r: r["seed"])
    dn = last(jl(OUT / arm / "donor" / "rows*.jsonl"), lambda r: r["seed"])
    ab = [ln.split("\t")[0] for ln in (OUT / arm / "b108_aborted.txt").read_text().splitlines()] \
        if (OUT / arm / "b108_aborted.txt").exists() else []
    return dict(ev=ev, st=st, tr=tr, fx=fx, dn=dn, aborted=ab)


def main():
    from eval.wcm_eval import is_vacuous, passes
    tfile = OUT / "targets_run.txt"
    panel = [k for k in (tfile if tfile.exists() else HERE / "targets.txt").read_text().split() if k]
    D = {a: load_arm(a) for a in A.ORDER}

    def verdict(a, k, held="strong"):
        r = D[a]["ev"].get((k, held))
        if r is None:
            return None
        return "vac" if is_vacuous(r) else bool(passes(r))

    print("# Experiment 108 results\n")
    print("Targets: %d (%s). Arms (batch = micro 4 x accum, steps, peak lr):" % (len(panel), tfile.name if tfile.exists()
                                                                                 else "targets.txt"))
    for a in A.ORDER:
        x = A.arm(a)
        print("- %s: batch %d (accum %d), %d steps, peak lr %.6g, sum(lr) %.2f, R seed %s, %d context passes" % (
            a, x["batch"], x["accum"], x["steps"], x["peak_lr"], x["lr_sum"], x["r_seed"] or "production",
            x["context_passes"]))
    plan = OUT / "plan.json"
    if plan.exists():
        p = json.loads(plan.read_text())
        print("\nPlan: N = %d, donor %s, budget %.0f min, predicted fits %.0f min." % (
            p["N"], p["donor"], p["budget_min"], p["predicted_fit_min"]))

    # ------------------------------------------------------------------ per target x arm
    print("\n## Per target x arm (held-out strongest)\n")
    print("| target | arm | n | Z | A | C | nec | pass | mid pass | induce | donor A/C/R | resample | fit s | s/step "
          "| peak MiB | zero rebound |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in panel:
        for a in A.ORDER:
            r = D[a]["ev"].get((k, "strong"))
            st, fx, tr, dn = D[a]["st"].get(k, {}), D[a]["fx"].get(k, {}), D[a]["tr"].get(k, {}), D[a]["dn"].get(k)
            if r is None and not st:
                if k in D[a]["aborted"]:
                    print("| %s | %s | aborted (watchdog) |%s" % (k, a, " |" * 13))
                continue
            v, vm = verdict(a, k), verdict(a, k, "mid")
            rec = (dn or {}).get("recovery", {}).get("fit", {})
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                k, a, st.get("n", "-"), f2((r or {}).get("free0_tk")), f2((r or {}).get("freeM_topk_tk")),
                f2((r or {}).get("freeN_topk_tk")), f2((r or {}).get("phi_sup_blind_tk")),
                {True: "PASS", False: "fail", "vac": "vac", None: "-"}[v], {True: "PASS", False: "fail", "vac": "vac",
                                                                           None: "-"}[vm],
                f2((r or {}).get("phi_cf_alpha_blind_tk")),
                "/".join(f2(rec.get(p_)) for p_ in "ACR") if dn else "-",
                ("PASS" if dn.get("resample_pass") else "fail") if dn else "-",
                st.get("t_fit", "-"), fx.get("s_per_step", "-"), fx.get("peak_reserved_mib", "-"),
                f2(rebound(tr.get("curve")))))

    # ------------------------------------------------------------------ roll-up
    print("\n## Per arm\n")
    print("| arm | scored | vacuous | pass strong | pass mid | vs B4 (common): gained / lost | median n | size / B4 on "
          "pass-in-both (median) | median induce | resample pass | fit min total | median s/step | max peak MiB "
          "| median zero rebound |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    roll = {}
    for a in A.ORDER:
        ks = [k for k in panel if (k, "strong") in D[a]["ev"]]
        vs = {k: verdict(a, k) for k in ks}
        sc = [k for k in ks if vs[k] != "vac"]
        n_pass = sum(1 for k in sc if vs[k] is True)
        vm = [verdict(a, k, "mid") for k in ks]
        n_mid = sum(1 for x in vm if x is True)
        n_mid_sc = sum(1 for x in vm if x in (True, False))
        common = [k for k in sc if verdict("B4", k) in (True, False)]
        gained = sum(1 for k in common if vs[k] is True and verdict("B4", k) is False)
        lost = sum(1 for k in common if vs[k] is False and verdict("B4", k) is True)
        both = [k for k in common if vs[k] is True and verdict("B4", k) is True
                and D[a]["st"].get(k, {}).get("n") and D["B4"]["st"].get(k, {}).get("n")]
        ratio = med([D[a]["st"][k]["n"] / D["B4"]["st"][k]["n"] for k in both])
        dn = [D[a]["dn"][k] for k in sc if k in D[a]["dn"]]
        roll[a] = dict(scored=len(sc), passes=n_pass, gained=gained, lost=lost, common=len(common), ratio=ratio,
                       median_n=med([D[a]["st"].get(k, {}).get("n") for k in ks]))
        print("| %s | %d | %d | %d / %d | %d / %d | +%d / -%d (%d) | %s | %s (%d) | %s | %s | %.1f | %s | %s | %s |" % (
            a, len(sc), len(ks) - len(sc), n_pass, len(sc), n_mid, n_mid_sc, gained, lost, len(common),
            f2(roll[a]["median_n"]), f2(ratio), len(both),
            f2(med([D[a]["ev"][(k, "strong")].get("phi_cf_alpha_blind_tk") for k in ks])),
            ("%d / %d" % (sum(1 for r in dn if r.get("resample_pass")), len(dn))) if dn else "-",
            sum(float(D[a]["st"].get(k, {}).get("t_fit") or 0) for k in ks) / 60,
            f2(med([D[a]["fx"].get(k, {}).get("s_per_step") for k in ks])),
            max([D[a]["fx"].get(k, {}).get("peak_reserved_mib") or 0 for k in ks] or [0]) or "-",
            f2(med([rebound(D[a]["tr"].get(k, {}).get("curve")) for k in ks]))))

    # ------------------------------------------------------------------ membership overlap
    print("\n## Membership Jaccard\n")
    print("| pair | targets | median | min | max | per target |")
    print("|---|---|---|---|---|---|")
    mem = {}
    for a, b in PAIRS:
        js = []
        for k in panel:
            for x in (a, b):
                if (x, k) not in mem:
                    mem[(x, k)] = members(OUT / x / "main" / "circuits" / ("%s.pt" % k))
            ma, mb = mem[(a, k)], mem[(b, k)]
            if ma is None or mb is None or not (ma | mb):
                continue
            js.append((k, len(ma & mb) / len(ma | mb)))
        v = [j for _, j in js]
        print("| %s vs %s | %d | %s | %s | %s | %s |" % (a, b, len(js), f2(med(v)), f2(min(v) if v else None),
                                                       f2(max(v) if v else None),
                                                       ", ".join("%s %.2f" % (k, j) for k, j in js)))
        roll.setdefault("_jac", {})["%s/%s" % (a, b)] = med(v)
    # wrapper sanity: B4 should reproduce latent_pilot's v4-equivalent fits of the same targets (v3 + geometric x3.15
    # + warm-up 0.1 + censor 0.05 as overrides; latent_pilot's check_v4_parity stage compared the two paths), up to GPU
    # nondeterminism
    js = []
    for k in panel:
        ma = mem.get(("B4", k)) if ("B4", k) in mem else members(OUT / "B4" / "main" / "circuits" / ("%s.pt" % k))
        mb = members(PARITY / ("%s.pt" % k))
        if ma is not None and mb is not None and (ma | mb):
            js.append(len(ma & mb) / len(ma | mb))
    print("| B4 vs latent_pilot v4-equivalent (parity) | %d | %s | %s | %s | (wrapper sanity check) |" % (
        len(js), f2(med(js)), f2(min(js) if js else None), f2(max(js) if js else None)))

    # ------------------------------------------------------------------ the pre-stated reading
    print("\n## Pre-stated reading rule (README, written before any result)\n")
    print("> " + RULE + "\n")
    j = roll.get("_jac", {})
    print("Seed overlap: B4 / B4s2 median %s, B48 / B48s2 median %s." % (f2(j.get("B4/B4s2")), f2(j.get("B48/B48s2"))))
    for a in ("B48", "B16m", "B16"):
        r = roll[a]
        if not r["scored"]:
            print("- %s: not run (or nothing scored)." % a)
            continue
        d = r["passes"] - roll["B4"]["passes"]
        print("- %s vs B4: passes %d vs %d (difference %+d%s), gained / lost on %d common targets +%d / -%d; median "
              "size ratio on pass-in-both %s." % (a, r["passes"], roll["B4"]["passes"], d,
                                                   ", within noise" if abs(d) <= 1 else "", r["common"], r["gained"],
                                                   r["lost"], f2(r["ratio"])))


if __name__ == "__main__":
    main()
