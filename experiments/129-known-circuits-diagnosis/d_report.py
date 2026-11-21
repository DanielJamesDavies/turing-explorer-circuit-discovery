"""129 D (CPU, JSON only): tables from out/c_heads.jsonl (c_heads.py).

  python -I experiments/129-known-circuits-diagnosis/d_report.py  -> out/d_report.md, out/d_report.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a_mass import CLASSES, EXPECT, in_scope, base_rate, parse  # noqa: E402

P = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "c_heads.jsonl"


def med(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    rows = [json.loads(l) for l in open(P)]
    by = {(r["seed"], r["frame"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    rep = dict(checks=[], kinds=[], errzero=[])
    for seed in seeds:
        layer, kind, _ = parse(seed)
        full = by.get((seed, "full"))
        if not full:
            continue
        pw = full["pathway"]
        fpre = full["pre"]
        rep["errzero"].append(dict(seed=seed, pre=fpre, **{k: v / fpre for k, v in full["err_zero_pre"].items()}))
        for fname in ("full", "corpus", "task"):
            r = by.get((seed, fname))
            if not r:
                continue
            H = {(l, h, ro): v for l, h, ro, *v in r["heads"]}
            O = {(l, h, ro): v for l, h, ro, v in r["old"]}
            tot_T = sum(v[0] for v in H.values()) or 1.0
            tot_D = sum(v[2] for v in H.values()) or 1.0
            tot_O = sum(O.values()) or 1.0
            # rank of cells by T_abs
            order = sorted(H, key=lambda c: -H[c][0])
            for chk in EXPECT[pw]:
                hs = set(in_scope(CLASSES[chk]["heads"], layer, kind))
                if not hs:
                    continue
                role = CLASSES[chk]["role"]
                cells = [c for c in H if (c[0], c[1]) in hs and c[2] == role]
                T_abs = sum(H[c][0] for c in cells); T_sg = sum(H[c][1] for c in cells)
                D_abs = sum(H[c][2] for c in cells); D_sg = sum(H[c][3] for c in cells)
                E_abs = sum(H[c][4] for c in cells); E_sg = sum(H[c][5] for c in cells)
                O_m = sum(O.get(c, 0.0) for c in cells)
                rank = next((i + 1 for i, c in enumerate(order) if c in cells), None)
                rep["checks"].append(dict(seed=seed, pathway=pw, frame=fname, check=chk, base=base_rate(chk, layer, kind),
                                          share_T=T_abs / tot_T, share_D=D_abs / tot_D, share_old=O_m / tot_O,
                                          T_sg=T_sg / r["n_prompts"], D_sg=D_sg / r["n_prompts"],
                                          E_sg=E_sg / r["n_prompts"], err_frac_abs=E_abs / max(1e-12, D_abs + E_abs),
                                          rank_T=rank, n_cells=len(H), pre=r["pre"]))
            K = defaultdict(lambda: [0.0] * 4)
            for l, k, *v in r["kinds"]:
                for j in range(4):
                    K[k][j] += v[j]
            adj = [x for x in r["kinds"] if (x[0] == layer - 1 and x[1] == "resid") or (x[0] == layer and x[1] == "resid_mid")]
            rep["kinds"].append(dict(seed=seed, frame=fname,
                                     err_frac_abs={k: v[2] / max(1e-12, v[0] + v[2]) for k, v in K.items()},
                                     adj=[dict(site="%d.%s" % (x[0], x[1]), slot_sg=x[3] / r["n_prompts"],
                                               err_sg=x[5] / r["n_prompts"]) for x in adj]))
    md = ["# 129 D: exact head / error attribution (c_heads.jsonl, %d rows)" % len(rows), ""]
    md += ["## Full model: forward-only target pre-activation with the SAE error zeroed at some site kinds "
           "(ratio to the unedited model)", "",
           "| target | pre | attn | mlp | attn+mlp | resid_mid+resid | all (120) |", "|---|---|---|---|---|---|---|"]
    for e in rep["errzero"]:
        md.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (e["seed"], e["pre"], e["attn"], e["mlp"],
                                                                     e["attn+mlp"], e["resid"], e["all"]))
    md.append("| median | | %.2f | %.2f | %.2f | %.2f | %.2f |" % tuple(
        med([e[k] for e in rep["errzero"]]) for k in ("attn", "mlp", "attn+mlp", "resid", "all")))
    md += ["", "## Published head classes at their role position", "",
           "share_T = exact |G.oc_h| mass of the class heads at the role / all heads x positions in scope (every route); "
           "share_D = the same through dictionary slots only (members in a circuit frame); share_old = 117's |drive|-split "
           "member-slot mass (reproduces 117). err = share of the class heads' |effect| at their own site routed through the "
           "live SAE error. T/D/E = signed per-prompt effect on the target pre-activation (T = D + E). rank = best class "
           "head cell's rank by |T| among all (head, position) cells.", "",
           "| target | check | base | frame | share_old | share_D | share_T | T (= D + E) per prompt | err | rank_T |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep["checks"]:
        md.append("| %s | %s | %.3f | %s | %.3f | %.3f | %.3f | %.2f (= %.2f + %.2f) | %.2f | %s |" % (
            c["seed"], c["check"], c["base"], c["frame"], c["share_old"], c["share_D"], c["share_T"], c["T_sg"],
            c["D_sg"], c["E_sg"], c["err_frac_abs"], c["rank_T"]))
    # summaries
    md += ["", "## Summary by frame (over all target x check cells)", "",
           "| frame | n | share_T >= 2x base | share_T >= 1x base | share_D >= 2x base | median err | median rank_T | "
           "median T_circuit / T_full (signed) |", "|---|---|---|---|---|---|---|---|"]
    fullT = {(c["seed"], c["check"]): c["T_sg"] for c in rep["checks"] if c["frame"] == "full"}
    for fr in ("full", "corpus", "task"):
        cs = [c for c in rep["checks"] if c["frame"] == fr]
        if not cs:
            continue
        md.append("| %s | %d | %d | %d | %d | %.2f | %s | %.2f |" % (
            fr, len(cs), sum(c["share_T"] >= 2 * c["base"] for c in cs), sum(c["share_T"] >= c["base"] for c in cs),
            sum(c["share_D"] >= 2 * c["base"] for c in cs), med([c["err_frac_abs"] for c in cs]),
            med([c["rank_T"] for c in cs]),
            med([c["T_sg"] / fullT[(c["seed"], c["check"])] for c in cs if abs(fullT.get((c["seed"], c["check"]), 0)) > 1e-9])))
    md += ["", "By check (full model): median share_T / base, median err, median rank_T, median signed T per prompt", ""]
    bc = defaultdict(list)
    for c in rep["checks"]:
        bc[(c["check"], c["frame"])].append(c)
    md += ["| check | frame | n | share_T/base | share_D/base | err | rank_T | T per prompt | D per prompt |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (chk, fr), cs in sorted(bc.items()):
        md.append("| %s | %s | %d | %.2f | %.2f | %.2f | %s | %.3f | %.3f |" % (
            chk, fr, len(cs), med([c["share_T"] / c["base"] for c in cs]), med([c["share_D"] / c["base"] for c in cs]),
            med([c["err_frac_abs"] for c in cs]), med([c["rank_T"] for c in cs]), med([c["T_sg"] for c in cs]),
            med([c["D_sg"] for c in cs])))
    md += ["", "## SAE error share of attribution mass by site kind (|G.err| / (|slot attr| + |G.err|)), median over targets", "",
           "| frame | attn | resid_mid | mlp | resid |", "|---|---|---|---|---|"]
    for fr in ("full", "corpus", "task"):
        ks = [k for k in rep["kinds"] if k["frame"] == fr]
        md.append("| %s | %s |" % (fr, " | ".join("%.2f" % med([k["err_frac_abs"].get(x) for k in ks])
                                                for x in ("attn", "resid_mid", "mlp", "resid"))))
    md += ["", "Target-adjacent residual site (the target's direct input), signed per-prompt slot vs error attribution:", "",
           "| target | frame | site | slots | error |", "|---|---|---|---|---|"]
    for k in rep["kinds"]:
        for a in k["adj"]:
            md.append("| %s | %s | %s | %.2f | %.2f |" % (k["seed"], k["frame"], a["site"], a["slot_sg"], a["err_sg"]))
    (HERE / "out" / "d_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (HERE / "out" / "d_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
