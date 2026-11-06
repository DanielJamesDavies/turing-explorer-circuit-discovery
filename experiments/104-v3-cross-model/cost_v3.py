"""104 part D: the v3 pod budget from MEASURED per-target times, split into training (fit) and scoring (eval /
specificity / donor). CPU only; reads the 104 runs and the 062 15k pod run's status rows. -> out/cost_v3.json + a
markdown table on stdout (copied into the README / runbook).

Inputs (M = measured here, E = estimated):
  TuringLLM, local (M): out/timing_tl/{v1,v2,v3} (12 targets, one per layer, T3's job: fit + eval strong + mid +
      specificity; idle GPU) and out/timing_tl/{v2,v3}/donor (T11's job, variants fit / a1 / pinned).
  TuringLLM, pod (M): 062's 15k v1 run on 8 x H100, MPS P = 6 (48 processes): per-process t_fit / t_eval / t_spec /
      t_ctx per target (out_full/out/main/status). For the 12 panel targets this gives the local -> pod per-process
      ratio of the same work (v1), separately for training (fit) and scoring (eval + spec): r_fit, r_score.
  Public, local (M): out/<model>/{v2,v3} (24 targets each, smoke-store records; fit + eval strong only, no spec) and
      their donor rows; fit times taken on the full-size records (48 training contexts, as the pod's records), and the
      chain smoke (smoke_chain) for specificity seconds.
  Public, pod (E): no pod measurement exists. The local -> H100 per-GPU factor is r = 2 (1.5-2.7), the plan's value.

Model: pod GPU-s of a TuringLLM job = sum over its targets of (pod v1 per-process seconds at the target's layer x the
measured local v3 / v1 ratio of that part) / 6 (P = 6 processes share one H100). Fit and scoring parts scale
separately. Donor (forward-only, scoring-type work): local seconds / r_score_gpu, where r_score_gpu = 6 x (local eval
s / pod per-process eval s) on the panel. Public jobs: local seconds / r.

  PYTHONPATH=src python experiments/104-v3-cross-model/cost_v3.py
"""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
O = HERE / "out"
P_POD = 6
PRICE = (2.7, 3.1, 3.5)
CAP = 550.0
R_PUB = (1.5, 2.0, 2.7)                    # public local -> H100 per-GPU factor (E); low r = slow pod


def jl(d, pat):
    out = []
    for p in sorted(Path(d).glob(pat)):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def ok_status(d):
    return {r["seed"]: r for r in jl(d, "status.shard*.jsonl") if r.get("ok")}


def donor_rows(d):
    return {r["seed"]: r for r in jl(d, "rows*.jsonl") if "error" not in r and not r.get("skip")}


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def layers_of(path):
    return [int(k.split(".")[0]) for k in Path(path).read_text().split() if k]


def main():
    J = {}
    # ------------------------------------------------------------------ TuringLLM: local panel vs the 062 pod run
    keys = [k for k in (HERE / "targets_timing_tl.txt").read_text().split() if k]
    loc = {V: ok_status(O / "timing_tl" / V / "main") for V in ("v1", "v2", "v3")}
    pod = {}
    for r in jl(EXP / "062-h100-protocol-v1" / "out_full" / "out" / "main", "status.shard*.jsonl"):
        if r.get("ok"):
            pod[r["seed"]] = r
    pod_layer = defaultdict(lambda: defaultdict(list))
    for k, r in pod.items():
        for f in ("t_fit", "t_eval", "t_spec", "t_ctx"):
            pod_layer[int(k.split(".")[0])][f].append(r.get(f) or 0.0)
    pod_L = {L: {f: mean(v) for f, v in d.items()} for L, d in pod_layer.items()}
    J["pod_v1_per_process_by_layer"] = pod_L
    tl = dict(n=len(keys))
    for V in ("v1", "v2", "v3"):
        tl[V] = {f: [loc[V].get(k, {}).get(f) for k in keys] for f in ("t_fit", "t_eval", "t_spec", "n")}
    # ratios per target (same target, same records, same GPU, idle)
    def ratio(a, b, f):
        return [x / y for x, y in zip(tl[a][f], tl[b][f]) if x and y]
    tl["fit_ratio_v3_v1"] = med(ratio("v3", "v1", "t_fit"))
    tl["fit_ratio_v2_v1"] = med(ratio("v2", "v1", "t_fit"))
    tl["fit_ratio_v3_v2"] = med(ratio("v3", "v2", "t_fit"))
    tl["fit_ratio_v3_v1_total"] = sum(x for x in tl["v3"]["t_fit"] if x) / sum(x for x in tl["v1"]["t_fit"] if x)
    tl["fit_ratio_v3_v2_total"] = sum(x for x in tl["v3"]["t_fit"] if x) / sum(x for x in tl["v2"]["t_fit"] if x)
    tl["eval_ratio_v3_v1"] = med(ratio("v3", "v1", "t_eval"))
    tl["spec_ratio_v3_v1"] = med(ratio("v3", "v1", "t_spec"))
    # local -> pod per-process (v1, same targets): pod per-process seconds / local seconds
    tl["pod_over_local_fit"] = med([pod[k]["t_fit"] / loc["v1"][k]["t_fit"] for k in keys if k in pod and k in loc["v1"]])
    tl["pod_over_local_eval"] = med([pod[k]["t_eval"] / loc["v1"][k]["t_eval"] for k in keys
                                     if k in pod and k in loc["v1"] and loc["v1"][k].get("t_eval")])
    tl["r_fit_gpu"] = P_POD / tl["pod_over_local_fit"]            # local single-process s per H100-s
    tl["r_score_gpu"] = P_POD / tl["pod_over_local_eval"]
    # donor (T11 job), local, v3 and v2 circuits: total per target and its split
    for V in ("v2", "v3"):
        dr = donor_rows(O / "timing_tl" / V / "donor")
        xs = [int(k.split(".")[0]) for k in keys if k in dr]
        tot = [dr[k]["t_target_total"] for k in keys if k in dr]
        sh = [dr[k].get("t_shared_target") for k in keys if k in dr]
        fit = [dr[k].get("t_variant", {}).get("fit") for k in keys if k in dr]
        tl["donor_%s" % V] = dict(n=len(tot), total_lin=linfit(xs, tot) if len(tot) > 2 else None,
                                  shared_lin=linfit(xs, sh) if len(sh) > 2 else None,
                                  fit_lin=linfit(xs, fit) if len(fit) > 2 else None,
                                  total_mean=mean(tot), shared_mean=mean(sh), fit_variant_mean=mean(fit))
    J["turingllm_panel"] = tl

    def tl_pod_gpu_s(layers, fit_ratio, n_fits=1.0, eval_ratio=1.0, with_eval=True, with_spec=True, with_ctx=True):
        s = 0.0
        for L in layers:
            p = pod_L[L]
            s += n_fits * p["t_fit"] * fit_ratio
            if with_eval:
                s += p["t_eval"] * eval_ratio
            if with_spec:
                s += p["t_spec"]
            if with_ctx:
                s += p["t_ctx"]
        return s / P_POD

    def donor_local_s(layers, V, variants="all"):
        d = tl["donor_%s" % V]
        if variants == "all":
            a, b = d["total_lin"]
            return sum(a + b * L for L in layers)
        a1, b1 = d["shared_lin"]
        a2, b2 = d["fit_lin"]
        return sum(a1 + b1 * L + a2 + b2 * L for L in layers)

    D94 = EXP / "094-tl-v2-records"
    L8k = layers_of(D94 / "targets_tl8k.txt")
    L190 = layers_of(D94 / "targets_tl8k_sweep190.txt")
    L128 = layers_of(D94 / "targets_tl8k_stable128.txt")
    L126 = layers_of(D94 / "targets_tl8k_nulls126.txt")
    r3 = tl["fit_ratio_v3_v1"]
    r2 = tl["fit_ratio_v2_v1"]
    er = tl["eval_ratio_v3_v1"] or 1.0
    # the 062 v1 per-process eval covers strong + mid; a sweep / null / stable row scores strong only (~half)
    st = []                                                       # (stage, GPU-h, basis, M/E)
    h = lambda s: s / 3600.0  # noqa: E731
    st.append(("T1 scan", 3.5, "unchanged (plan 2.8-4.2)", "E (plan)"))
    st.append(("T2 gate + build + verify", 0.8 + 5.0 + 4.8, "unchanged (plan; build 6.1 s, verify 13 s per target M)", "E (plan)"))
    t3 = tl_pod_gpu_s(L8k, r3, eval_ratio=er)
    t3_fit = tl_pod_gpu_s(L8k, r3, with_eval=False, with_spec=False, with_ctx=False)
    st.append(("T3 main (8,000)", h(t3), "fit %.1f + score %.1f GPU-h; pod v1 per layer x v3/v1 fit %.2f, eval x %.2f"
               % (h(t3_fit), h(t3 - t3_fit), r3, er), "M"))
    st.append(("T3 alpha = 1", 1.3, "score only, unchanged (plan)", "E (plan)"))
    t4 = 12 * tl_pod_gpu_s(L190, r3, eval_ratio=er * 0.5, with_spec=False, with_ctx=False)
    st.append(("T4 sweep (190 x 12 new fits)", h(t4), "as T3 per fit, strong eval only; weak-lambda arms not scaled", "M"))
    t5 = tl_pod_gpu_s(L128, r3, n_fits=2, eval_ratio=er, with_spec=False)
    st.append(("T5 stable core (128 x 2 fits)", h(t5), "2 fits + 2 strong evals per target", "M"))
    t6_fit = tl_pod_gpu_s(L126, r3, n_fits=(316 + 126) / 126.0, with_eval=False, with_spec=False, with_ctx=False)
    t6_score = 0.0
    for L in L126:
        t6_score += pod_L[L]["t_eval"] * 0.5 * 1900 / 126.0 / P_POD
    st.append(("T6 nulls (316 + 126 fits, ~1,900 scores)", h(t6_fit + t6_score),
               "fits as T3 (null fits taken as full fits: upper end), scores = strong evals", "M (fits) / E (counts)"))
    # 083 (README runtime table, v1-era local estimate): ~960 local-s per target, of which ~120 s are the 3 support
    # fits; under v3 those take x fit_ratio_v3_v1. GPU-h at the measured TuringLLM r_fit (all of 083 is TL fit-type
    # or heavier work; its own P = 4); the plan's r = 2 is the upper end.
    t7_local = 190 * (960 + 120 * (r3 - 1))
    t7 = t7_local / tl["r_fit_gpu"]
    J["t7"] = dict(local_h=t7_local / 3600, gpu_h=h(t7), gpu_h_at_r2=t7_local / 2.0 / 3600)
    st.append(("T7 DAN-159 (083)", h(t7), "083's ~960 local-s per target (+%.0f s v3 support fits) x 190 = %.1f local-h / "
               "TL r_fit %.2f (at the plan's r = 2: %.1f)" % (120 * (r3 - 1), t7_local / 3600, tl["r_fit_gpu"],
                                                             t7_local / 2.0 / 3600), "E (local-s) / M (r)"))
    t10 = tl_pod_gpu_s(L190, r2, eval_ratio=0.5, with_spec=False, with_ctx=False)
    st.append(("T10 without-R arm (190 v2 fits)", h(t10), "pod v1 per layer x v2/v1 fit %.2f" % r2, "M"))
    rs = tl["r_score_gpu"]
    t11 = donor_local_s(L8k, "v3") / rs
    t11_fit = donor_local_s(L8k, "v3", "fit") / rs
    st.append(("T11 donor test, 8k main", h(t11), "local donor s (v3 circuits, by layer) / r_score %.1f" % rs, "M"))
    t12 = 0.0
    if tl["donor_v3"]["shared_lin"]:
        a1, b1 = tl["donor_v3"]["shared_lin"]
        per_arm = [tl["donor_v3"]["total_lin"][0] - a1, tl["donor_v3"]["total_lin"][1] - b1]
        t12 = sum(a1 + b1 * L + 14 * (per_arm[0] + per_arm[1] * L) for L in L190) / rs   # 13 arms + noR (own launch)
        t12 += sum(a1 + b1 * L for L in L190) / rs                                   # the noR launch's shared work
        t12_fit = sum(2 * (a1 + b1 * L) + 14 * (tl["donor_v3"]["fit_lin"][0] + tl["donor_v3"]["fit_lin"][1] * L)
                      for L in L190) / rs
    st.append(("T12 donor test, 190 x 13 arms + noR", h(t12), "shared once per launch + per arm", "M"))
    J["tl_stages"] = st
    J["tl_donor_fit_only_gpu_h"] = dict(T11=h(t11_fit), T12=h(t12_fit) if t12 else None)

    # ------------------------------------------------------------------ public models (local measured, r estimated)
    pub = {}
    smoke = HERE / "smoke_chain"
    for F in ("pythia-70m", "gpt2-small"):
        d = {}
        sel = {r["seed"]: r for r in jl(O / F, "select.jsonl")}
        tr = {V: {r["seed"]: r for r in jl(O / F / V / "main", "train.shard*.jsonl")} for V in ("v2", "v3")}
        for V in ("v2", "v3"):
            s = ok_status(O / F / V / "main")
            full = [k for k in s if (sel.get(k) or {}).get("n_train") == 48]
            d[V] = dict(n=len(s), n_full=len(full),
                        fit_full_median=med([s[k]["t_fit"] for k in full]),
                        fit_all_median=med([r["t_fit"] for r in s.values()]),
                        eval_strong_median=med([r["t_eval"] for r in s.values()]))
            dr = donor_rows(O / F / V / "donor")
            d[V]["donor_total_median"] = med([r["t_target_total"] for r in dr.values()])
            d[V]["donor_shared_median"] = med([r.get("t_shared_target") for r in dr.values()])
            d[V]["donor_fit_variant_median"] = med([r.get("t_variant", {}).get("fit") for r in dr.values()])
        both = [k for k in ok_status(O / F / "v2" / "main") if k in ok_status(O / F / "v3" / "main")
                and (sel.get(k) or {}).get("n_train") == 48]
        s2, s3 = ok_status(O / F / "v2" / "main"), ok_status(O / F / "v3" / "main")
        d["fit_ratio_v3_v2_full"] = med([s3[k]["t_fit"] / s2[k]["t_fit"] for k in both])
        sp = [r.get("t_spec") for r in jl(smoke / ("out_%s" % F) / "main", "status.shard*.jsonl") if r.get("ok")]
        d["spec_smoke_median"] = med(sp)
        pub[F] = d
    J["public_local"] = pub

    def pub_local_s(F, donor_variants="all"):
        d = pub[F]
        fit = d["v3"]["fit_full_median"]
        ev = d["v3"]["eval_strong_median"]
        sp = d["spec_smoke_median"] or (1.0 if F == "pythia-70m" else 3.0)
        dn = d["v3"]["donor_total_median"] if donor_variants == "all" else (
            d["v3"]["donor_shared_median"] + d["v3"]["donor_fit_variant_median"])
        # (name, fit-s, score-s): training and scoring kept apart (they scale differently on the pod)
        rows = [("pilot 90", 90 * fit, 90 * ev),
                ("main 512 (fit; strong + mid + spec)", 512 * fit, 512 * (2 * ev + sp)),
                ("alpha = 1, 512 (strong + mid)", 0.0, 512 * 2 * ev),
                ("nulls 320 fits, 1,344 scores + 128 profiles", 320 * fit, (1344 + 128) * ev),
                ("stable 32 x (2 fits + 2 scores)", 32 * 2 * fit, 32 * 2 * ev),
                ("073 strict 128 (~ the donor's shared part)", 0.0, 128 * d["v3"]["donor_shared_median"]),
                ("donor 512", 0.0, 512 * dn)]
        if F == "pythia-70m":
            rows.insert(5, ("unweighted 128", 128 * fit, 128 * ev))
            rows.append(("donor_unweighted 128", 0.0, 128 * dn))
        return rows

    J["public_rows"] = {F: pub_local_s(F) for F in pub}
    J["public_rows_fitonly_donor"] = {F: pub_local_s(F, "fit") for F in pub}
    (O / "cost_v3.json").write_text(json.dumps(J, indent=1, default=str))

    # ------------------------------------------------------------------ totals and variants
    tl_total = sum(g for _, g, _, _ in st)
    t7 = next(g for s, g, _, _ in st if s.startswith("T7"))
    t11g = next(g for s, g, _, _ in st if s.startswith("T11"))
    t12g = next(g for s, g, _, _ in st if s.startswith("T12"))
    donor_cut_tl = (t11g - J["tl_donor_fit_only_gpu_h"]["T11"]) + (t12g - (J["tl_donor_fit_only_gpu_h"]["T12"] or t12g))
    DATA_PASS = 2.3
    # public local -> H100 factors (fit, score): the plan's r for both (1.5 / 2.0 / 2.7), and "TL" = TuringLLM's
    # measured per-GPU ratios applied to the public models (an assumption: different models, fp32 + TF32)
    SC = {"r1.5": (1.5, 1.5), "r2.0": (2.0, 2.0), "r2.7": (2.7, 2.7),
          "TL": (tl["r_fit_gpu"], tl["r_score_gpu"])}

    def scen(rf, rsc):
        pub_h = {F: sum(a / rf + b / rsc for _, a, b in J["public_rows"][F]) / 3600 for F in pub}
        pub_h_fit = {F: sum(a / rf + b / rsc for _, a, b in J["public_rows_fitonly_donor"][F]) / 3600 for F in pub}
        return pub_h, pub_h_fit

    res = {}
    for name_r, (rf, rsc) in SC.items():
        r = name_r
        pub_h, pub_h_fit = scen(rf, rsc)
        base = tl_total + sum(pub_h.values()) + DATA_PASS
        var = {
            "full": base,
            "(a) TL_DAN159=0": base - t7,
            "(b) DONOR_VARIANTS=fit": base - donor_cut_tl - sum(pub_h[F] - pub_h_fit[F] for F in pub),
            "(c) public tracks local": tl_total,
        }
        var["(a)+(b)"] = var["(b) DONOR_VARIANTS=fit"] - t7
        var["(a)+(c)"] = tl_total - t7
        var["(b)+(c)"] = tl_total - donor_cut_tl
        var["(a)+(b)+(c)"] = tl_total - t7 - donor_cut_tl
        res[r] = dict(pub=pub_h, var=var)
    J["scenarios"] = {str(r): v for r, v in res.items()}
    J["tl_total"] = tl_total
    J["public_local_hours"] = {F: sum(a + b for _, a, b in J["public_rows"][F]) / 3600 for F in pub}
    J["public_local_hours_fitonly_donor"] = {F: sum(a + b for _, a, b in J["public_rows_fitonly_donor"][F]) / 3600
                                             for F in pub}
    (O / "cost_v3.json").write_text(json.dumps(J, indent=1, default=str))

    def wall(g):
        return g / 8 * 1.05 + 0.75 + 0.3

    print("| stage | GPU-h | basis | M/E |\n|---|---:|---|---|")
    for s, g, b, m in st:
        print("| %s | %.1f | %s | %s |" % (s, g, b, m))
    print("| **TuringLLM** | **%.1f** | | |" % tl_total)
    for F in pub:
        fs = sum(a for _, a, _ in J["public_rows"][F]) / 3600
        ss = sum(b for _, _, b in J["public_rows"][F]) / 3600
        print("| %s | %.1f at r 2.0 (r 1.5: %.1f; 2.7: %.1f; TL ratios: %.1f) | local %.1f h = fit %.1f + score %.1f "
              "(%s) | M local / E r |" % (
                  F, res["r2.0"]["pub"][F], res["r1.5"]["pub"][F], res["r2.7"]["pub"][F], res["TL"]["pub"][F],
                  J["public_local_hours"][F], fs, ss,
                  "; ".join("%s %.1f + %.1f h" % (n, a / 3600, b / 3600) for n, a, b in J["public_rows"][F])))
    print("| data passes | %.1f | unchanged (plan) | E |" % DATA_PASS)
    print()
    print("| variant | GPU-h (public r 2.0) | wall h | cost at $2.7 / 3.1 / 3.5 | GPU-h public r 2.7 .. 1.5 | "
          "cost range | GPU-h, TL ratios on public |")
    print("|---|---:|---:|---|---|---|---:|")
    for name in res["r2.0"]["var"]:
        g = res["r2.0"]["var"][name]
        lo, hi = res["r2.7"]["var"][name], res["r1.5"]["var"][name]
        w = wall(g)
        print("| %s | %.0f | %.1f | $%.0f / %.0f / %.0f | %.0f .. %.0f | $%.0f-%.0f | %.0f |" % (
            name, g, w, w * 8 * PRICE[0], w * 8 * PRICE[1], w * 8 * PRICE[2], lo, hi,
            wall(lo) * 8 * PRICE[0], wall(hi) * 8 * PRICE[2], res["TL"]["var"][name]))
    print("\npanel:", json.dumps({k: v for k, v in tl.items() if not isinstance(v, dict)}, default=str))
    print("public:", json.dumps(pub, default=str))


if __name__ == "__main__":
    main()
