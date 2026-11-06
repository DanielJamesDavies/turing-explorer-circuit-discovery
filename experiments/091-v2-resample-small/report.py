"""091 report (CPU): out/rows.jsonl -> out/tables.md."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ARMS = ["dense", "sp_ste", "v2R", "v2R_S"]


def f(x, p=2):
    return ("%%.%df" % p) % x if isinstance(x, (int, float)) and x is not None and not np.isnan(x) else "-"


def med(v):
    v = [x for x in v if isinstance(x, (int, float)) and x is not None]
    return float(np.median(v)) if v else None


def main():
    rows = {}
    for l in open(OUT / "rows.jsonl"):
        r = json.loads(l)
        if "error" not in r and "skip" not in r:
            rows[(r["seed"], r["arm"])] = r
    t = json.loads((OUT / "targets.json").read_text())
    keys = [k for k in t["keys"] if all((k, a) in rows for a in ARMS)]
    checks = [json.loads(l) for l in open(OUT / "check.jsonl")] if (OUT / "check.jsonl").exists() else []
    L = []
    w = L.append
    w("# 091: the donor-resample term R (+ silent term S) on top of v2 (090 sp_ste)\n")
    w("Targets: %d of 090's 20 (random.Random(%d)): fail89 %s; deep attention near %s, far %s; passing controls %s. "
      "Complete for all four versions: %d.\n" % (len(t["keys"]), t["seed"], ", ".join(t["fail89"]),
                                                 t["attn_near"][0], t["attn_far"][0], ", ".join(t["control"]), len(keys)))
    w("Versions: **dense** and **sp_ste** = 090's circuits (not refitted; fit time / peak memory from 090's rows, same "
      "RTX 5070 Ti). **v2R** = sp_ste + R (w_R 0.25, donors = the target's 48 training contrast contexts, one donor per "
      "training row, redrawn every forward). **v2R_S** = v2R + silent-position term (w_S 0.25). Production otherwise "
      "(400 steps, lambda 1e-3, gamma_Z 1, gamma_A = gamma_C 0.25, rank-keep 3e-3). One fit per target and arm.\n")
    w("Scoring, held-out strongest 16. Production scorer: Z, A, C under the rule's sparsity-preserving fill, A/C dense, "
      "necessity; pass = Z, A, C in [0.8, 1.5] and nec >= 0.9; dpass = same with dense A/C. Resample test (073 "
      "strict_eval.score_target, c_held): rs A (donors = strongest-train), rs C (donors = the 16 HELD-OUT contrast "
      "contexts, never a training donor), rs R (073's random-corpus evaluation pool); rs pass = Z, rs A, rs C in "
      "[0.8, 1.5] and nec >= 0.9 (079's definition). dense / sp_ste rescored here; they match 090's rows exactly "
      "(max |diff| on Z/A/C/nec = %s).\n" % f(max([rows[(k, a)].get("max_diff_vs_090_row") or 0 for k in keys
                                                  for a in ("dense", "sp_ste")] or [0]), 3))
    if checks:
        c = checks[-1]
        w("Patch check (`run.py CHECK=1`, %s, %d steps): combined patch with w_R = w_S = 0 under ste == 090 sp_engine "
          "ste (circuit %s, loss curve %s); combined with dense fill and w_R = w_S = 0.25 == 079 rs_engine "
          "RS0.25_S0.25 (circuit %s, loss curve %s); v2R_S differs from both (%s, %s). The donor patcher is excluded "
          "from the sparsity-preserving fill (R exactly as in 079).\n" % (
              c["key"], c["steps"], c["ste_w0_same_circuit"], c["ste_w0_same_curve"], c["rs_dense_same_circuit"],
              c["rs_dense_same_curve"], c["v2RS_differs_from_ste"], c["v2RS_differs_from_rs"]))

    w("\n## Per target\n")
    w("| target | group | arm | n | Z | A | C | nec | A dense | C dense | pass | dpass | rs A | rs C (held) | rs R |"
      " rs pass | fit s | peak MB | alpha med | alpha p90 | alpha max |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for a in ARMS:
            r = rows[(k, a)]
            w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
              " %s |" % (k, r.get("group"), a, r.get("n"), f(r.get("ho_free0_tk")), f(r.get("ho_freeM_topk_tk")),
                         f(r.get("ho_freeN_topk_tk")), f(r.get("ho_phi_sup_blind_tk")), f(r.get("ho_freeM_dense_tk")),
                         f(r.get("ho_freeN_dense_tk")), "Y" if r.get("pass") else "n", "Y" if r.get("dpass") else "n",
                         f(r.get("rs_A")), f(r.get("rs_C")), f(r.get("rs_R")), "Y" if r.get("rs_pass") else "n",
                         f(r.get("t_fit"), 0), r.get("fit_peak_mb"), f(r.get("alpha_med")), f(r.get("alpha_p90")),
                         f(r.get("alpha_max"))))

    w("\n## Medians per arm\n")
    w("| arm | targets | n | Z | A | C | nec | A dense | C dense | pass | dpass | rs A | rs C (held) | rs R | rs pass |"
      " fit s | peak MB | alpha med | alpha p90 | alpha max |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        R = [rows[(k, a)] for k in keys]
        m = lambda c: f(med([r.get(c) for r in R]))  # noqa: E731
        w("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %d/%d | %d/%d | %s | %s | %s | %d/%d | %s | %s | %s | %s | %s |"
          % (a, len(R), f(med([r.get("n") for r in R]), 0), m("ho_free0_tk"), m("ho_freeM_topk_tk"),
             m("ho_freeN_topk_tk"), m("ho_phi_sup_blind_tk"), m("ho_freeM_dense_tk"), m("ho_freeN_dense_tk"),
             sum(bool(r.get("pass")) for r in R), len(R), sum(bool(r.get("dpass")) for r in R), len(R), m("rs_A"),
             m("rs_C"), m("rs_R"), sum(bool(r.get("rs_pass")) for r in R), len(R), f(med([r.get("t_fit") for r in R]), 0),
             f(med([r.get("fit_peak_mb") for r in R]), 0), m("alpha_med"), m("alpha_p90"), m("alpha_max")))

    w("\n## Paired against sp_ste (same targets; median of per-target differences; pass changes as gained / lost)\n")
    w("| arm | size ratio | d Z | d A | d C | d nec | d rs A | d rs C | d rs R | rule pass +/- | rs pass +/- |"
      " fit time ratio | peak mem ratio |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        if a == "sp_ste":
            continue
        B = [rows[(k, "sp_ste")] for k in keys]
        X = [rows[(k, a)] for k in keys]

        def d(c):
            return f(med([x[c] - b[c] for x, b in zip(X, B) if x.get(c) is not None and b.get(c) is not None]), 2)

        def ratio(c):
            return f(med([x[c] / b[c] for x, b in zip(X, B) if x.get(c) and b.get(c)]), 2) + "x"

        def gl(c):
            g = sum(bool(x.get(c)) and not bool(b.get(c)) for x, b in zip(X, B))
            lo = sum(bool(b.get(c)) and not bool(x.get(c)) for x, b in zip(X, B))
            return "+%d / -%d" % (g, lo)
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, ratio("n"), d("ho_free0_tk"), d("ho_freeM_topk_tk"), d("ho_freeN_topk_tk"), d("ho_phi_sup_blind_tk"),
            d("rs_A"), d("rs_C"), d("rs_R"), gl("pass"), gl("rs_pass"), ratio("t_fit"), ratio("fit_peak_mb")))

    w("\n## Cost (per fit; median over targets of the per-target ratio)\n")
    w("| arm | fit s (median) | peak MB (median) | fit time vs dense | fit time vs sp_ste | peak mem vs dense |"
      " peak mem vs sp_ste | size vs dense | size vs sp_ste |")
    w("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        X = [rows[(k, a)] for k in keys]

        def rr(c, base):
            Bs = [rows[(k, base)] for k in keys]
            return f(med([x[c] / b[c] for x, b in zip(X, Bs) if x.get(c) and b.get(c)]), 2) + "x"
        w("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, f(med([x.get("t_fit") for x in X]), 0), f(med([x.get("fit_peak_mb") for x in X]), 0),
            rr("t_fit", "dense"), rr("t_fit", "sp_ste"), rr("fit_peak_mb", "dense"), rr("fit_peak_mb", "sp_ste"),
            rr("n", "dense"), rr("n", "sp_ste")))
    w("\nFit time = R.fit wall clock (the mask engine's discover; the donor-code capture for R, a few seconds, is "
      "outside it); peak = torch.cuda.max_memory_allocated during the fit, reset before it. dense / sp_ste figures are "
      "090's on the same GPU.\n")
    txt = "\n".join(L) + "\n"
    (OUT / "tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
