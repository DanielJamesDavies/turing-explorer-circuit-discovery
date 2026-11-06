"""Arena A (2026-11-05 night): the panel pass table and the necessity-diagnosis tables, as markdown. CPU, reads files.

  pass table   per fit dir (pilot wcm_l*, panel wcm_l1e-3, panel arm_*): target, Z / A / C (free0_tk, freeM_topk_tk,
               freeN_topk_tk), necessity (phi_sup_blind_tk), n, fit s, steps, pass (compare_dev24.passes: the pass rule)
  diagnosis    per circuit in out/panel/diag_nec/results.jsonl: necessity under live / zero / mean error, Z under
               live / zero error, the frozen / fill variants as necessity-like fractions (clean - x) / (clean - e0),
               survival of the clean read, train necessity
  sites        median L0, members active per position and members' decoded-norm share at the anchor, by kind x band

  python experiments/107-true-baselines/arena_a/necdiag_summary.py   (writes out/panel/necdiag_summary.md)
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE / "out"
BAND, NEC_MIN = (0.75, 1.5), 0.9
FITS = [("pilot_l1e-3", O / "pilot" / "wcm_l1e-3"), ("pilot_l4e-3", O / "pilot" / "wcm_l4e-3"),
        ("panel_l1e-3", O / "panel" / "wcm_l1e-3"), ("panel_l1e-3_rerun", O / "panel" / "wcm_l1e-3_rerun"),
        ("arm_necw1", O / "panel" / "arm_necw1"),
        ("arm_l2.5e-4", O / "panel" / "arm_l2.5e-4"), ("arm_s1200", O / "panel" / "arm_s1200")]


def jl(paths):
    out = []
    for p in paths:
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def passes(e):
    if e is None or e.get("vacuous_tk"):
        return None
    v = [e.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = e.get("phi_sup_blind_tk")
    return bool(all(isinstance(x, (int, float)) and BAND[0] <= x <= BAND[1] for x in v)
                and isinstance(nec, (int, float)) and nec >= NEC_MIN)


def f(v, d=2):
    return "-" if v is None else ("%.*f" % (d, v))


def band(l):
    return "1-3" if l <= 3 else "4-7" if l <= 7 else "8-11"


def main():
    md = ["## Pass table", "", "| fit | target | Z | A | C | nec | n | fit s | steps | pass |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, d in FITS:
        ev = {r["seed"]: r for r in jl(sorted((d / "main").glob("eval.shard*.jsonl"))) if r.get("held") == "strong"}
        tr = {r["seed"]: r for r in jl(sorted((d / "main").glob("train.shard*.jsonl")))}
        if not ev:
            continue
        npass = 0
        for k in sorted(ev, key=lambda s: (int(s.split(".")[0]), s)):
            e, t = ev[k], tr.get(k, {})
            p = passes(e)
            npass += bool(p)
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                lab, k, f(e.get("free0_tk")), f(e.get("freeM_topk_tk")), f(e.get("freeN_topk_tk")),
                f(e.get("phi_sup_blind_tk")), e.get("n"), t.get("t_fit"), t.get("steps"),
                "vacuous" if p is None else "PASS" if p else "fail"))
        md.append("| %s | **%d / %d pass** | | | | | | | | |" % (lab, npass, len(ev)))
    rp = O / "panel" / "diag_nec" / "results_v2.jsonl"
    rows = jl([rp]) if rp.exists() else []
    cols = [("clean", lambda r: r["reads"]["clean"]), ("e0/clean", lambda r: r["survive"]["e0_live"]),
            ("full err0 /clean", lambda r: r["survive"]["full_zero"]),
            ("full errM /clean", lambda r: r["survive"]["full_mean"]),
            ("nec (b)", lambda r: r["nec"]["live"]), ("Z", lambda r: r["Z"]["live"]),
            ("(e) nm mean", lambda r: r["nec_like"]["rm_nmmean"]),
            ("g all frozen", lambda r: r["nec_like"]["g_frozen"]), ("g nm", lambda r: r["nec_like"].get("g_nm")),
            ("g err", lambda r: r["nec_like"].get("g_err")),
            ("(f) z frozen", lambda r: r["nec_like"]["f_zfrozen"]), ("tc frozen", lambda r: r["nec_like"]["f_tcfrozen"]),
            ("tc nm", lambda r: r["nec_like"].get("f_tc_nm")), ("tc err", lambda r: r["nec_like"].get("f_tc_err")),
            ("z err", lambda r: r["nec_like"].get("f_z_err")),
            ("surv rm 0-fill", lambda r: r["survive"]["rm_live"]),
            ("surv rm mean-fill", lambda r: r["survive"]["rm_memmean"]), ("nec train", lambda r: r["train"]["nec"])]
    md += ["", "## Necessity diagnosis (held-out strongest, post-activation read)", "",
           "e0 / full errE: reads relative to clean; nec (b) = the scorer's; the rest are necessity-like fractions "
           "(clean - x) / (clean - e0) with members zeroed: (e) non-members mean; g: every site's non-members (nm) "
           "and / or error (err) frozen at clean; tc / z: only those sites frozen. surv = read / clean.", "",
           "| circuit | n | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]

    def g(fn, r):
        try:
            return fn(r)
        except (KeyError, TypeError):
            return None
    for r in rows:
        md.append("| %s %s | %d | %s |" % (r["label"], r["key"], r["n"], " | ".join(f(g(fn, r)) for _, fn in cols)))
    groups = [("all", rows), ("base (pilot + panel)", [r for r in rows if not r["label"].startswith("arm")]),
              ("base, not vacuous", [r for r in rows if not r["label"].startswith("arm")
                                     and r["survive"]["e0_live"] is not None and r["survive"]["e0_live"] < 0.95])]
    for nm_, rs in groups:
        if not rs:
            continue
        vals = []
        for _, fn in cols:
            v = [g(fn, r) for r in rs]
            v = [x for x in v if x is not None]
            vals.append(f(st.median(v)) if v else "-")
        md.append("| **median, %s (%d)** | | %s |" % (nm_, len(rs), " | ".join(vals)))
    if rows:
        agg = {}
        for r in rows:
            if r["label"].startswith("arm"):
                continue
            for s, v in r["sites"].items():
                l, k = int(s.split(".")[0]), s.split(".")[1]
                a = agg.setdefault((k, band(l)), dict(l0=[], l0a=[], ma=[], maa=[], sh=[], nm=[], fvu=[]))
                a["fvu"].append(v.get("fvu", float("nan")))
                a["l0"].append(v["l0"]); a["l0a"].append(v["l0_anchor"]); a["ma"].append(v.get("mem_active", 0))
                a["maa"].append(v["mem_active_anchor"]); a["sh"].append(v["mem_norm_share"]); a["nm"].append(v["n_mem"])
        md += ["", "## Upstream sites (medians over circuit x site)", "",
               "| kind | band | L0 / position | L0 at anchor | members | members active / position | members active at "
               "anchor | members' decoded-norm share at anchor | FVU (centred) |", "|---|---|---|---|---|---|---|---|---|"]
        for (k, b), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (k, b, *(f(st.median(a[x]), 1) for x in (
                "l0", "l0a", "nm", "ma", "maa")), f(st.median(a["sh"])), f(st.median(a["fvu"]))))
    out = O / "panel" / "necdiag_summary.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
