"""Exp 135 analysis (CPU, saved tensors only): tables for the closure and candidate measurements -> OUT/report.md.

  .venv/bin/python -X utf8 experiments/135-ioi-parent-closure/analyse135.py   [OUT135=... for another out dir]
"""
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUT135", str(HERE / "out")))
E133 = HERE.parent / "133-ioi-wcm-sweep"
SETS = ("train", "dis", "sn")
CRIT = ("io", "s1", "s1_next", "s2", "end")


def f2(x):
    return "%.2f" % x if x == x else "-"


def load_circuit(name):
    d = json.loads((E133 / "out" / "circuits" / ("%s.json" % name)).read_text())
    al = {}
    for k, v in d["alphas"].items():
        l, kd = k.split("/")
        if v:
            al[(int(l), kd)] = {int(i) for i in v}
    return al, d.get("cfg", {}).get("sites", "all")


def closure_tables(lines):
    files = sorted(OUT.glob("closure_*_*.pt"))
    if not files:
        return
    rows_A, rows_C, rows_role, rows_ret = [], [], [], []
    for p in files:
        d = torch.load(p, weights_only=False)
        Fi = {k: i for i, k in enumerate(d["feats"])}
        roles = d["roles"]
        tag = "%s / %s" % (d["circuit"], d["set"])
        # retention (all members, by kind)
        rk = {}
        for s, r in d["ret"].items():
            rk.setdefault(s[1], []).append(r)
        cells = []
        for kd in ("attn", "resid_mid", "mlp", "resid"):
            if kd not in rk:
                cells.append("-")
                continue
            r = torch.cat(rk[kd])
            ok = r[:, 3] >= 5
            per = (r[ok, 1] / r[ok, 0].clamp_min(1e-9))
            aw = f2(float(r[:, 4].sum() / r[:, 0].sum())) if r.shape[1] > 4 else "-"
            cells.append("%s / %s / %s / %s" % (f2(float(r[:, 1].sum() / r[:, 0].sum())),
                                                f2(float(r[:, 2].sum() / r[:, 3].sum())),
                                                f2(float(per.median())) if ok.any() else "-", aw))
        rows_ret.append("| %s | %s | %.2f / %.2f |" % (tag, " | ".join(cells), d["m_full"], d["m_circuit_Z"]))
        if not d["acc_m"]:
            continue
        for kd in ("resid_mid", "resid"):
            ss = [s for s in d["acc_m"] if s[1] == kd]
            if not ss:
                continue
            M = torch.cat([d["acc_m"][s] for s in ss])                      # [members, F]
            Rr = sum(d["acc_r"][s] for s in ss)                               # [R, F]
            tot = M.sum(0)
            act = float(tot[Fi["act"]])
            n = float(tot[Fi["n"]])

            def sh(*names, v=tot):
                return float(sum(v[Fi[k]] for k in names)) / max(float(v[Fi["act"]]), 1e-9)
            cats = ("in", "out", "err", "bdec")
            comps = ["skip", "attn"] + (["mlp"] if kd == "resid" else [])
            A_in = sh(*["A_%s_in" % c for c in comps])
            A_out = sh(*["A_%s_out" % c for c in comps])
            A_err = sh(*["A_%s_err" % c for c in comps])
            A_emb = sh("emb")
            A_bias = sh("beff", *["A_%s_bdec" % c for c in comps])
            A_skip = sh("emb", *["A_skip_%s" % c for c in cats])
            # per member in-circuit share
            ok = M[:, Fi["n"]] >= 5
            mem_in = sum(M[:, Fi["A_%s_in" % c]] for c in comps) / M[:, Fi["act"]].clamp_min(1e-9)
            mem_in = mem_in[ok]
            drv = 1.0 - A_bias                                                 # the input-dependent part
            rows_A.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s (%s / %s / %s) | %s | %s | %s | %s | %s |" % (
                tag, kd, int(ok.sum()), int(n), f2(A_in), f2(A_out), f2(A_err), f2(A_emb), f2(A_bias), f2(A_skip),
                "%s / %s / %s / %s" % (f2(A_in / drv), f2(A_out / drv), f2(A_err / drv), f2(A_emb / drv)),
                f2(A_in), f2(sh("A_skip_in")), f2(sh("A_attn_in")), f2(sh("A_mlp_in")) if kd == "resid" else "-",
                f2(float(mem_in.median())) if ok.any() else "-", f2(float((mem_in >= 0.5).float().mean())) if ok.any() else "-",
                f2(float(tot[Fi["A_maxpar_ge25"]]) / n), f2(float(tot[Fi["A_maxpar_ge50"]]) / n),
                f2(float(tot[Fi["A_in_ge50"]]) / n)))
            # view B / C
            B_in = sh("B_rmid_in", "A_mlp_in") if kd == "resid" else float("nan")
            C_attn_in = {r: sh("C_attn_in_%s" % r) for r in roles}
            C_attn_tot = {r: sh(*["C_attn_%s_%s" % (c, r) for c in ("in", "out", "err", "bdec", "emb")]) for r in roles}
            C_in = (sh("A_skip_in") + (sh("A_mlp_in") if kd == "resid" else 0.0) + sum(C_attn_in.values()))
            rows_C.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                tag, kd, f2(B_in), f2(float(tot[Fi["B_in_ge50"]]) / n) if kd == "resid" else "-",
                f2(C_in), f2(float(tot[Fi["C_in_ge50"]]) / n),
                " ".join("%s %s" % (r, f2(C_attn_tot[r])) for r in roles),
                " ".join("%s %s" % (r, f2(C_attn_in[r])) for r in roles), f2(sh("C_attn_const"))))
            # by target role, view A
            cells = []
            for r in CRIT:
                v = Rr[roles.index(r)]
                if float(v[Fi["n"]]) < 1:
                    cells.append("-")
                    continue
                cells.append("%s / %s / %s" % (f2(sh(*["A_%s_in" % c for c in comps], v=v)),
                                               f2(sh(*["A_%s_out" % c for c in comps], v=v)),
                                               f2(sh(*["A_%s_err" % c for c in comps], v=v))))
            rows_role.append("| %s | %s | %s |" % (tag, kd, " | ".join(cells)))
        lines.append("")
    lines += ["## Measurement 1: closure of residual-stream members (natural pre-activation, exact decomposition)", "",
              "View A (the brief's parents: resid_{L-1} at the same position, attn_L, mlp_L). Shares of the pooled natural "
              "pre-activation (sum over all active member occurrences; parts sum to 1). in = in-circuit parent latents, "
              "out = out-of-circuit latents, err = SAE errors of the parent sites, emb = embedding (layer-0 targets only), "
              "bias = b_eff + the parent SAEs' b_dec; skip = the whole resid_{L-1} / embedding component. "
              "members = members with >= 5 occurrences; occ = occurrences. med in / frac >= .5 = per-member in-circuit "
              "share. maxpar >= .25 / .5 = share of occurrences whose single strongest in-circuit parent gives >= 25 / 50% "
              "of the pre-activation; in >= .5 = occurrences whose in-circuit parents together give >= 50%.", "",
              "drive = the same shares over the input-dependent part only (act minus bias; in / out / err / emb sum to 1).", "",
              "| circuit / set | kind | members | occ | in | out | err | emb | bias | skip | drive in / out / err / emb | in (skip / attn / mlp) | med in | frac in >= .5 | maxpar >= .25 | maxpar >= .5 | in >= .5 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows_A
    lines += ["", "View B (resid only: resid_mid_L + mlp_L) and view C (attn_L replaced by its source positions: resid_{L-1} "
              "latents at every source position s through the frozen attention pattern and LN1 scale). C in = skip in + "
              "mlp in + attention-carried in-circuit resid_{L-1} latents; attn total / attn in = the attention component's "
              "share by SOURCE role (all parts / in-circuit part only).", "",
              "| circuit / set | kind | B in | B occ in >= .5 | C in | C occ in >= .5 | C attn total by source role | C attn in-circuit by source role | C attn const |",
              "|---|---|---|---|---|---|---|---|---|"] + rows_C
    lines += ["", "View A by TARGET position role: in / out / err shares at the member's own position.", "",
              "| circuit / set | kind | %s |" % " | ".join(CRIT), "|---|---|%s" % ("---|" * len(CRIT))] + rows_role
    lines += ["", "Functional closure: the scorer's Z frame (members at alpha, non-members 0, errors live). Per kind: pooled "
              "retention sum(circuit code) / sum(natural code) at the members' natural occurrences / share of those "
              "occurrences still active / median per-member retention (members with >= 5 occurrences) / the same pooled "
              "retention with each circuit code times its alpha (what the circuit passes on). Last column: mean task metric D, full model / circuit in the Z frame.", "",
              "| circuit / set | attn | resid_mid | mlp | resid | D full / D circuit |", "|---|---|---|---|---|---|"] + rows_ret
    chk = []
    for p in files:
        d = torch.load(p, weights_only=False)
        chk.append("%s/%s A %.1e B %.1e C %.1e" % (d["circuit"], d["set"], d["chk"]["chkA"], d["chk"]["chkB"], d["chk"]["chkC"]))
    lines += ["", "Decomposition check (max relative |act - sum of parts| over all occurrences): " + "; ".join(chk), ""]


def candidate_tables(lines, circuits):
    att = {s: torch.load(OUT / ("attr_%s.pt" % s), weights_only=False) for s in SETS if (OUT / ("attr_%s.pt" % s)).exists()}
    if "train" not in att:
        return
    tr = att["train"]
    sites = tr["sites"]
    roles = list(tr["roles"])
    D = int(tr["z"][sites[0]].shape[-1])
    act_any = torch.stack([tr["act_n"][s].sum(0) > 0 for s in sites])                 # [S, D] train-active anywhere
    act_role = torch.stack([tr["act_n"][s] > 0 for s in sites])                       # [S, R, D]
    lines += ["## Measurement 2: high-attribution latents at the IOI positions vs the fits' candidates", "",
              "Candidate set (from the code): 133's fits pass no support and no theta_init_mode, so learned_mask starts "
              "EVERY latent of every in-scope site at theta_init 4 (gate 0.98 > keep threshold 0.5): all 1,572,864 "
              "(48-site) or 786,432 (attn + mlp) latents are offered. 'Never in the candidate set' is therefore empty by "
              "construction. The table separates what the fit could not have learned from (never active on any of the "
              "1,600 training prompts: under the Z / A / C frames such a latent only ever contributes its fill value) "
              "from what was offered, active in training, and dropped.", "",
              "Attribution = |mean over items of the role-summed gradient x activation| (z: g.f, the zero-ablation "
              "effect; atp: g.(f_ABC - f)). Top-N over the circuit's scope.", ""]
    hdr = ("| circuit | set | role | attr | N | in circuit | dropped, train-active (at this role) | dropped, never train-active | "
           "top-N kinds a/rm/m/r | attr mass in circuit |")
    lines += [hdr, "|---|---|---|---|---|---|---|---|---|---|"]
    kinds = ("attn", "resid_mid", "mlp", "resid")
    for cname in circuits:
        al, scope = load_circuit(cname)
        sc = [i for i, s in enumerate(sites) if scope == "all" or s[1] in ("attn", "mlp")]
        mem = torch.zeros(len(sites), D, dtype=torch.bool)
        for i, s in enumerate(sites):
            if s in al:
                mem[i, torch.tensor(sorted(al[s]))] = True
        for sname, a in att.items():
            n = float(a["n_items"])
            for st in ("z", "atp"):
                X = torch.stack([a[st][s] for s in sites]) / n                              # [S, R, D]
                for r in CRIT:
                    ri = roles.index(r)
                    V = X[sc, ri].abs()                                                     # [S', D]
                    flat = V.flatten()
                    msk = mem[sc].flatten()
                    tot = float(flat.sum())
                    mass = float(flat[msk].sum()) / max(tot, 1e-12)
                    for N in (20, 100, 500):
                        top = flat.topk(N).indices
                        si = top // D
                        li = top % D
                        sg = torch.tensor(sc)[si]
                        inc = mem[sg, li]
                        ta = act_role[sg, ri, li]
                        tany = act_any[sg, li]
                        kc = [sum(1 for x in sg.tolist() if sites[x][1] == k) for k in kinds]
                        lines.append("| %s | %s | %s | %s | %d | %.2f | %.2f (%.2f) | %.2f | %s | %s |" % (
                            cname, sname, r, st, N, float(inc.float().mean()), float((~inc & tany).float().mean()),
                            float((~inc & ta).float().mean()), float((~inc & ~tany).float().mean()),
                            "/".join(str(k) for k in kc), f2(mass) if N == 20 else ""))
    lines.append("")


def main():
    circuits = sorted({p.stem.split("_train")[0][len("closure_"):] for p in OUT.glob("closure_*_train.pt")})
    if not circuits:
        circuits = [c for c in os.environ.get("CIRCUITS", "").split(",") if c]
    lines = ["# 135 report (generated by analyse135.py from %s)" % OUT.name, ""]
    closure_tables(lines)
    candidate_tables(lines, circuits)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
