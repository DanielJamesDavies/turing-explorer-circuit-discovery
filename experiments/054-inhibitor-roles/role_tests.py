"""WHICH ROLE DEFINITION HOLDS AT SET LEVEL? (DAN-72) Excitatory vs inhibitory nodes of the production
weighted circuits, tested on the 60-circuit pilot with the v3 evaluation code (amp_eval_pass_v2.py).

WCM itself is unsigned: a node is kept because replacing it with its ablation value spoils the reproduction
of the target. A role says which way it spoils it. Definitions compared (sign >= 0 = excitatory):

  G     gradient x activation on the UNMODIFIED model, train slice (the v2/v3 pass's rule).
  F_pi  fitted-effect sign: d(target pre-activation) / d(log alpha_i) at the FITTED circuit, circuit-only
        execution under ablation method pi in {Z, A, C}, train slice.
  L_pi  leave-one-out against the ablation value: a(circuit) - a(circuit without node i), node i sent to
        its ablation value under pi, on the first 16 train contexts. Subset of circuits only (LOO=1,
        LOO_PER_LAYER circuits per layer).

Set-level test for every definition, on HELD-OUT activating contexts, circuit-only execution under pi:
  release     (a(C without its inhibitory nodes) - a(C)) / a_pos   -> should be > 0 (brakes released)
  remove_exc  (a(C without its excitatory nodes) - a(C)) / a_pos   -> should be < 0
reported on both reads (`_pre` pre-activation, `_tk` activation). Roles are signs of a continuous effect,
so definitions are built on the pre-activation (the activation is censored at 0 below the Top-K cutoff).

  PYTHONPATH=src python experiments/054-inhibitor-roles/role_tests.py            -> results/roles_pilot.jsonl
  SUMMARY=1 python experiments/054-inhibitor-roles/role_tests.py                 -> summary of existing rows
  env: LOO=0/1 (default 1), LOO_PER_LAYER (default 2), SEEDS (default: the 049 pilot seed list)
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
OUT = HERE / "results"
OUT_FILE = OUT / os.environ.get("ROLES_OUT", "roles_pilot.jsonl")
LOO = os.environ.get("LOO", "1") == "1"
LOO_PER_LAYER = int(os.environ.get("LOO_PER_LAYER", "2"))
LOO_N_CTX = 16
PIS = ("Z", "A", "C")


def score(c, do_loo):
    import torch
    import amp_eval_pass_v2 as V
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from pipeline.component_index import component_idx as comp_of

    G = V.G
    inference, bank, M0, KINDS, NK, D, K = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D", "K"))
    TapCO, _ = V._engine_classes()
    ts = time.time()
    seed = V.seed_of(c)
    sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    key = V.key_of(c)
    alphas = {}
    for n in c.nodes.values():
        if n is not seed:
            f = n.metadata["feature_id"]
            alphas.setdefault((f.layer, f.kind), {})[int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    SITES = sorted(alphas)
    pd_ = M0.build_probe_dataset(comp_of(layer, KINDS.index(kind), NK), sl)
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < V.MIN_POS or not SITES:
        return dict(seed=key, skip="thin_probes_or_empty")
    pt, pa, nt = pd_.pos_tokens[:V.N_SEQ], pd_.pos_argmax[:V.N_SEQ], pd_.neg_tokens[:V.N_SEQ]
    n_tr = V.split_n(int(pt.shape[0])); n_ntr = V.split_n(int(nt.shape[0]))
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    nt_tr = nt[:n_ntr]
    sae = bank.saes[kind][layer]; w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
    UPS = set(upstream_sites(bank, layer, kind)); site = (layer, kind); dev = pt.device
    idx_of = {s: torch.tensor(sorted(alphas[s]), device=dev, dtype=torch.long) for s in SITES}
    nodes = [(s, i) for s in SITES for i in sorted(alphas[s])]
    n_mem = len(nodes)

    means_A, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    means = {"Z": None, "A": means_A, "C": collect_site_means(inference, bank, nt_tr, UPS) if int(nt_tr.shape[0]) else None}
    pis = [p for p in PIS if p == "Z" or means[p] is not None]

    def scale_vectors(requires_grad=False):
        out = {}
        for s in SITES:
            sv = torch.ones(D, device=dev, dtype=torch.float32)
            sv[idx_of[s]] = torch.tensor([alphas[s][i] for i in sorted(alphas[s])], device=dev, dtype=torch.float32)
            out[s] = sv.requires_grad_(True) if requires_grad else sv
        return out
    SC = scale_vectors()

    def keep_all():
        return {s: set(alphas[s]) for s in SITES}

    def co(keep_, tokens, anchors, pi):
        """Engine circuit-only run: kept nodes at alpha x live value, every other upstream latent at its ablation value."""
        keep_ = {s: v for s, v in keep_.items() if v}
        kt = {s: torch.tensor(sorted(v), device=dev, dtype=torch.long) for s, v in keep_.items()}
        ks = {s: SC[s] for s in keep_} or None
        return V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                         seed_latent_idx=sl, pos_argmax=anchors[a:b], site_means=means[pi], topk=K,
                                         keep_tensors=kt, keep_scales=ks, w_seed=w_seed, b_seed=b_seed), tokens, anchors)

    def without(labels, want_exc):
        keep_ = keep_all()
        for (s, i), e in zip(nodes, labels):
            if e is not None and bool(e) == want_exc:
                keep_[s].discard(i)
        return keep_

    a_pos = V.read(lambda a, b: V.AmpInjectPatcher({}, site, w_seed, b_seed, sl), pt_ho, pa_ho)

    # ---- G: gradient x activation on the unmodified model (the eval pass's rule)
    attr = {s: torch.zeros(len(idx_of[s]), device=dev, dtype=torch.float32) for s in SITES}
    wdec = {s: bank.saes[s[1]][s[0]].decoder.weight.detach()[:, idx_of[s].to(bank.saes[s[1]][s[0]].decoder.weight.device)].to(device=dev, dtype=torch.float32)
            for s in SITES}
    inference.disable_compile()
    try:
        for s0 in range(0, n_tr, V.ROLE_BS):
            rp = V.RolePatcher(idx_of, site, w_seed, b_seed)
            inference.forward(pt_tr[s0:s0 + V.ROLE_BS], patcher=rp, grad_enabled=True, return_activations=False, tokenize_final=False)
            pre = rp.seed_pre; B = pre.shape[0]
            anc = pa_tr[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
            tgt = pre[torch.arange(B, device=pre.device), anc].float().sum()
            sites_ = list(rp.z)
            gs = torch.autograd.grad(tgt, [rp.z[s] for s in sites_], allow_unused=True)
            for s, g in zip(sites_, gs):
                if g is not None:
                    attr[s] += ((g.detach().float() @ wdec[s]) * rp.codes[s]).sum(dim=(0, 1))
            del rp, gs, tgt, pre
    finally:
        inference.enable_compile()
    del wdec
    vals = {"G": np.concatenate([attr[s].cpu().numpy() for s in SITES])}

    # ---- F_pi: d target / d log(alpha_i) at the fitted circuit, circuit-only execution under pi
    for pi in pis:
        sc = scale_vectors(requires_grad=True)
        acc = {s: torch.zeros(D, device=dev, dtype=torch.float32) for s in SITES}
        kt = dict(idx_of)
        inference.disable_compile()
        try:
            for s0 in range(0, n_tr, V.ROLE_BS):
                tk = pt_tr[s0:s0 + V.ROLE_BS]; B = int(tk.shape[0])
                p = TapCO(bank=bank, keep_indices=keep_all(), in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                          pos_argmax=pa_tr[s0:s0 + B], site_means=means[pi], topk=K, keep_tensors=kt, keep_scales=sc,
                          w_seed=w_seed, b_seed=b_seed)
                inference.forward(tk, patcher=p, grad_enabled=True, return_activations=False, tokenize_final=False)
                pre = p.seed_pre
                anc = pa_tr[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tgt = pre[torch.arange(B, device=pre.device), anc].float().sum()
                gs = torch.autograd.grad(tgt, [sc[s] for s in SITES], allow_unused=True)
                for s, g in zip(SITES, gs):
                    if g is not None:
                        acc[s] += g.detach().float()
                del p, gs, tgt, pre
        finally:
            inference.enable_compile()
        vals["F_" + pi] = np.concatenate([(acc[s][idx_of[s]] * SC[s][idx_of[s]]).cpu().numpy() for s in SITES])
        del sc, acc

    # ---- L_pi: leave-one-out against the ablation value (subset of circuits)
    t_loo = time.time()
    if do_loo:
        ctx, anc = pt_tr[:LOO_N_CTX], pa_tr[:LOO_N_CTX]
        for pi in pis:
            full = co(keep_all(), ctx, anc, pi)
            d = np.zeros(n_mem)
            for j, (s, i) in enumerate(nodes):
                keep_ = keep_all(); keep_[s].discard(i)
                d[j] = full["pre"] - co(keep_, ctx, anc, pi)["pre"]
            vals["L_" + pi] = d
    secs_loo = time.time() - t_loo

    # ---- labels: True = excitatory, False = inhibitory, None = exactly zero effect (LOO only)
    labels = {}
    for name, v in vals.items():
        labels[name] = [None if (name.startswith("L_") and x == 0) else bool(x >= 0) for x in v]

    row = dict(seed=key, layer=layer, kind=kind, n=n_mem, loo=bool(do_loo), a_pos_pre=a_pos["pre"], a_pos_tk=a_pos["tk"])
    for name, lab in labels.items():
        row["n_inh_" + name] = int(sum(1 for x in lab if x is False))
        row["n_zero_" + name] = int(sum(1 for x in lab if x is None))

    # ---- set-level test on held-out contexts, under each pi
    for pi in pis:
        full = co(keep_all(), pt_ho, pa_ho, pi)
        row["full_%s_pre" % pi], row["full_%s_tk" % pi] = full["pre"], full["tk"]
        for name in [n for n in labels if n == "G" or n.endswith("_" + pi)]:
            r_inh = co(without(labels[name], want_exc=False), pt_ho, pa_ho, pi)
            r_exc = co(without(labels[name], want_exc=True), pt_ho, pa_ho, pi)
            for rd in ("pre", "tk"):
                ap = a_pos[rd]
                ok = abs(ap) > 1e-9
                row["release_%s_%s_%s" % (name, pi, rd)] = (r_inh[rd] - full[rd]) / ap if ok else None
                row["remove_exc_%s_%s_%s" % (name, pi, rd)] = (r_exc[rd] - full[rd]) / ap if ok else None

    # ---- agreement between definitions (share of nodes given the same role)
    names = sorted(labels)
    agree = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            la, lb = labels[names[x]], labels[names[y]]
            both = [(p, q) for p, q in zip(la, lb) if p is not None and q is not None]
            agree["%s|%s" % (names[x], names[y])] = (sum(p == q for p, q in both) / len(both)) if both else None
    row["agree"] = agree
    row["vals"] = {k: [round(float(x), 5) for x in v] for k, v in vals.items()}
    row.update(secs=round(time.time() - ts, 1), secs_loo=round(secs_loo, 1))
    return row


def summarise():
    rows = [json.loads(l) for l in open(OUT_FILE)] if OUT_FILE.exists() else []
    rows = [r for r in rows if "skip" not in r and "error" not in r]
    print("circuits %d (LOO on %d)" % (len(rows), sum(r["loo"] for r in rows)))
    if not rows:
        return
    med = lambda xs: float(np.median(xs)) if xs else float("nan")
    defs = ["G"] + ["F_" + p for p in PIS] + ["L_" + p for p in PIS]
    print("\n=== inhibitory share of nodes (median over circuits) ===")
    for d in defs:
        xs = [r["n_inh_" + d] / max(r["n"], 1) for r in rows if "n_inh_" + d in r]
        if xs:
            print("  %-5s %.3f  (n=%d)" % (d, med(xs), len(xs)))
    print("\n=== SET-LEVEL TEST on held-out contexts (pre-activation read; activation read in brackets) ===")
    print("  release = removing all inhibitory nodes should RAISE the target (> 0); remove_exc should LOWER it (< 0)")
    print("  %-5s %-2s | %-26s | %-26s | %s" % ("def", "pi", "release: median, share > 0", "remove_exc: median, share < 0", "both hold"))
    for pi in PIS:
        for d in ["G", "F_" + pi, "L_" + pi]:
            rel = [r["release_%s_%s_pre" % (d, pi)] for r in rows if r.get("release_%s_%s_pre" % (d, pi)) is not None]
            rex = [r["remove_exc_%s_%s_pre" % (d, pi)] for r in rows if r.get("remove_exc_%s_%s_pre" % (d, pi)) is not None]
            relk = [r["release_%s_%s_tk" % (d, pi)] for r in rows if r.get("release_%s_%s_tk" % (d, pi)) is not None]
            if not rel:
                continue
            both = [r for r in rows if r.get("release_%s_%s_pre" % (d, pi)) is not None]
            ok = np.mean([(r["release_%s_%s_pre" % (d, pi)] > 0) and (r["remove_exc_%s_%s_pre" % (d, pi)] < 0) for r in both])
            print("  %-5s %-2s | %+.3f, %.2f  [%+.3f]      | %+.3f, %.2f                | %.2f  (n=%d)"
                  % (d, pi, med(rel), np.mean(np.array(rel) > 0), med(relk), med(rex), np.mean(np.array(rex) < 0), ok, len(rel)))
    print("\n=== agreement between definitions (share of nodes with the same role; median over circuits) ===")
    keys = sorted({k for r in rows for k in r["agree"]})
    for k in keys:
        xs = [r["agree"][k] for r in rows if r["agree"].get(k) is not None]
        print("  %-10s %.3f  (n=%d)" % (k, med(xs), len(xs)))
    print("\n=== by layer band: release > 0 share, pre read (G_Z | F_Z | L_Z) ===")
    for lo, hi in ((0, 3), (4, 7), (8, 11)):
        sub = [r for r in rows if lo <= r["layer"] <= hi]
        cells = []
        for d in ("G", "F_Z", "L_Z"):
            xs = [r["release_%s_Z_pre" % d] for r in sub if r.get("release_%s_Z_pre" % d) is not None]
            cells.append("%.2f (n=%d)" % (np.mean(np.array(xs) > 0), len(xs)) if xs else "-")
        print("  L%d-%d: %s" % (lo, hi, " | ".join(cells)))


def main():
    if os.environ.get("SUMMARY") == "1":
        summarise()
        return
    os.environ.setdefault("SEEDS", str(ROOT / "results_full" / "amp_eval_v2_pilot_seeds.txt"))
    os.environ.setdefault("TAG", "roles_unused")
    sys.path.insert(0, str(ROOT))
    import amp_eval_pass_v2 as V
    items, _, _, src = V.load_items()
    V.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_FILE.exists():
        for ln in open(OUT_FILE):
            try:
                r = json.loads(ln)
                if "error" not in r:
                    done.add(r["seed"])
            except Exception:
                pass
    per_layer = defaultdict(int); loo_keys = set()
    for c in items:                                     # LOO subset: the first LOO_PER_LAYER pilot circuits of each layer
        lay = V.seed_of(c).metadata["feature_id"].layer
        if per_layer[lay] < LOO_PER_LAYER:
            loo_keys.add(V.key_of(c)); per_layer[lay] += 1
    print("roles | %s | %d circuits (%d with LOO) | %d done -> %s" % (src, len(items), len(loo_keys) if LOO else 0, len(done), OUT_FILE), flush=True)
    t0 = time.time()
    with open(OUT_FILE, "a") as fh:
        for n_i, c in enumerate(items):
            key = V.key_of(c)
            if key in done:
                continue
            try:
                row = score(c, LOO and key in loo_keys)
            except Exception as e:  # noqa: BLE001
                row = dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:300]))
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("[%2d] %-16s %s | %.0fs total" % (n_i, key, row.get("error") or row.get("skip") or
                  "n=%d inh G %d F_Z %d | rel G_Z %+.3f F_Z %+.3f%s | %.1fs"
                  % (row["n"], row["n_inh_G"], row["n_inh_F_Z"], row["release_G_Z_pre"], row["release_F_Z_Z_pre"],
                     (" L_Z %+.3f" % row["release_L_Z_Z_pre"]) if row["loo"] else "", row["secs"]), time.time() - t0), flush=True)
    summarise()


if __name__ == "__main__":
    main()
