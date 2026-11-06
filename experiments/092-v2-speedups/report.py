"""092 report: out/tables.md from out/rows.jsonl, out/check.jsonl, 090 sp_ste rows and 091 rows (sp_ste, v2R).

  wsl: .venv/bin/python experiments/092-v2-speedups/report.py
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
E91 = HERE.parent / "091-v2-resample-small"


def load(p):
    if not p.exists():
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def f(x, d=2):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "Y" if x else "n"
    if isinstance(x, float):
        if x != 0 and abs(x) < 10 ** -d:
            return "%.1e" % x
        return "%.*f" % (d, x)
    return str(x)


def last_by(rows, key=("seed", "arm")):
    d = {}
    for r in rows:
        if "error" in r:
            continue
        d[tuple(r[k] for k in key)] = r
    return d


def main():
    rows = last_by(load(OUT / "rows.jsonl"))
    checks = load(OUT / "check.jsonl")
    r91 = last_by(load(E91 / "out" / "rows.jsonl"))
    keys = json.loads((E91 / "out" / "targets.json").read_text())["keys"]
    L = ["# 092: v2 speed-ups and Z + A + R_C", ""]
    if checks:
        c = checks[-1]
        L += ["## Step-0 equality (target %s, one training step, data terms only)" % c["key"], "",
              "| variant | max abs d(term) | abs d(sum of terms) | max abs d(grad theta) | max abs d(grad amp) |"
              " secs (1-step fit) | peak MB | token share |", "|---|---|---|---|---|---|---|---|",
              "| v2 (reference) | - | loss %s | scale %s | scale %s | %s | %s | 1 |" % (
                  f(c.get("loss_v2"), 6), f(c.get("grad_theta_scale"), 4), f(c.get("grad_amp_scale"), 4),
                  c.get("secs_v2"), c.get("peak_mb_v2"))]
        for t in ("trunc", "comb", "comb_trunc"):
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                t, "%.2e" % c["dterm_max_" + t], "%.2e" % c["dloss_" + t], "%.2e" % c["dgrad_theta_max_" + t],
                "%.2e" % c["dgrad_amp_max_" + t], c.get("secs_" + t), c.get("peak_mb_" + t),
                f(c.get("tok_frac_" + t), 3)))
        L.append("")
        L.append("Per-term values (zero, floor = C, pos = A): v2 %s; trunc %s; comb %s" % (
            [f(x, 6) for x in c.get("terms_v2", [])], [f(x, 6) for x in c.get("terms_trunc", [])],
            [f(x, 6) for x in c.get("terms_comb", [])]))
        L.append("")
    for arm, title in (("t1_trunc", "Test 1: truncate at the latest anchor per batch"),
                       ("t2_comb", "Test 2: Z / C / A in one combined forward"),
                       ("t12_both", "Tests 1 + 2 together (optional)")):
        rs = [rows[(k, arm)] for k in keys if (k, arm) in rows]
        if not rs:
            L += ["## %s" % title, "", "not run", ""]
            continue
        L += ["## %s (vs 090 sp_ste)" % title, "",
              "| target | group | n / n_ref | identical | Jaccard | max abs d alpha | dZ | dA | dC | dnec | pass (ref) |"
              " t_fit s (ref) | speed-up | peak MB (ref) | token share | rows/fwd |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        sp = []
        for r in rs:
            s = (r["ref_t_fit"] / r["t_fit"]) if r.get("t_fit") and r.get("ref_t_fit") else None
            if s:
                sp.append(s)
            L.append("| %s | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s (%s) | %s | %s (%s) | %s | %s |" % (
                r["seed"], r.get("group"), r.get("n"), r.get("n_ref"), f(r.get("identical")), f(r.get("jaccard"), 3),
                f(r.get("dalpha_max"), 3), f(r.get("d_free0_tk"), 3), f(r.get("d_freeM_topk_tk"), 3),
                f(r.get("d_freeN_topk_tk"), 3), f(r.get("d_phi_sup_blind_tk"), 3), f(r.get("pass")),
                f(r.get("ref_pass")), r.get("t_fit"), r.get("ref_t_fit"), f(s, 2) if s else "-", r.get("fit_peak_mb"),
                r.get("ref_peak_mb"), f(r.get("tok_frac"), 3), r.get("combine_rows") or "-"))
        tot = sum(r["t_fit"] for r in rs if r.get("t_fit")), sum(r["ref_t_fit"] for r in rs if r.get("ref_t_fit"))
        L += ["", "Summary: %d targets; identical circuits %d; median Jaccard %s; median speed-up %s; total fit time "
              "%.0f s vs %.0f s (x%.2f); max peak %s MB vs %s MB." % (
                  len(rs), sum(1 for r in rs if r.get("identical")), f(st.median([r["jaccard"] for r in rs]), 3),
                  f(st.median(sp), 2) if sp else "-", tot[0], tot[1], (tot[1] / tot[0]) if tot[0] else 0,
                  max(r.get("fit_peak_mb") or 0 for r in rs), max(r.get("ref_peak_mb") or 0 for r in rs)), ""]
    for arm3, title3 in (("t3_ZAR", "Test 3: Z + A + R_C (no C term; R under the sparsity-preserving donor fill)"),
                         ("t3b_ZAR_rawR", "Test 3b: Z + A + R_C (no C term; R exactly as 091 v2R, no donor fill)")):
        L += t3_table(rows, r91, keys, arm3, title3)
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def t3_table(rows, r91, keys, arm3, title3):
    L = []
    rs = [rows[(k, arm3)] for k in keys if (k, arm3) in rows]
    if rs:
        L += ["## %s, held-out strongest" % title3, "",
              "Rule A / C = production scorer (sparsity-preserving); rsA / rsC = 073 resample (C donors = 16 held-out "
              "contrast contexts). Each cell: this arm / 091 sp_ste / 091 v2R.", "",
              "| target | group | n | Z | A | C | nec | pass | dense A | dense C | rs A | rs C | rs pass | t_fit s | peak MB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        cnt = {a: dict(pass_=0, rs=0, n=0) for a in (arm3, "sp_ste", "v2R")}
        for r in rs:
            k = r["seed"]
            a = r91.get((k, "sp_ste"), {})
            b = r91.get((k, "v2R"), {})
            trip = lambda key, d=2: "%s / %s / %s" % (f(r.get(key), d), f(a.get(key), d), f(b.get(key), d))  # noqa: E731
            for nm, src in ((arm3, r), ("sp_ste", a), ("v2R", b)):
                if src:
                    cnt[nm]["n"] += 1
                    cnt[nm]["pass_"] += int(bool(src.get("pass")))
                    cnt[nm]["rs"] += int(bool(src.get("rs_pass")))
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                k, r.get("group"), trip("n", 0), trip("ho_free0_tk"), trip("ho_freeM_topk_tk"), trip("ho_freeN_topk_tk"),
                trip("ho_phi_sup_blind_tk"), trip("pass"), trip("ho_freeM_dense_tk"), trip("ho_freeN_dense_tk"),
                trip("rs_A"), trip("rs_C"), trip("rs_pass"), trip("t_fit", 0), trip("fit_peak_mb", 0)))
        L += [""] + ["- %s: rule pass %d / %d, resample pass %d / %d" % (nm, c["pass_"], c["n"], c["rs"], c["n"])
                     for nm, c in cnt.items()]
        for nm, src in ((arm3, {r["seed"]: r for r in rs}), ("sp_ste", {k: r91.get((k, "sp_ste"), {}) for k in keys}),
                        ("v2R", {k: r91.get((k, "v2R"), {}) for k in keys})):
            vals = [v.get("t_fit") for kk, v in src.items() if kk in {r["seed"] for r in rs} and v.get("t_fit")]
            if vals:
                L.append("- %s: total fit time %.0f s over %d targets" % (nm, sum(vals), len(vals)))
        L.append("")
    else:
        L += ["## %s" % title3, "", "not run", ""]
    return L


if __name__ == "__main__":
    main()
