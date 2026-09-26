"""WIRED WEIGHTED CIRCUITS: direct-effect edges for WCM circuits, linearised around the circuit's own run.

A WCM circuit is a set of upstream latents with coefficients (a "star" onto the target). This attaches edges
between its nodes with the SFC direct-effect construction, computed in the circuit-only run (members at
alpha x live value, everything else at the ablation value, SAE error live) instead of the unmodified model; see
src/circuit/instrument/weighted_edges.py for the maths. One edge set per ablation method (Z, A, C), on the 16
held-out activating contexts, with the ablation values built exactly as the v3 eval / 056 specificity pass builds
them (A from the training activating contexts, C from the training close contrast contexts).

Checks, per (circuit, ablation method):
  replay        the instrument's target pre-activation vs CircuitOnlyPatcher's (same circuit, same fill)
  conservation  sum of each node's out-edges (incl. to the target) vs its own attribution A_u: exact chain rule at
                one linearisation point, so this tests the implementation (bf16 autocast sets the floor)
  causal nodes  top-N nodes by |A_u|: dropping u from the circuit (u -> its floor, everything else live) vs A_u
  causal edges  top-N member edges by |w|: every node pinned at its circuit value except d (live) and u (-> floor),
                the change in d weighted by g_d, vs w(u -> d). Tests the linearisation, not just the code.

  PYTHONPATH=src python experiments/057-wcm-edges/wcm_edges.py
      -> results/edges_<arm>.jsonl (one row per target x ablation method), data/<arm>/<target>.pt (matrices)
  env: ARM (rkeep3e3fix)  SEEDS (file, default 055 seeds.txt)  ONLY (one target key)  SMOKE=1 (first target, Z)
       CH (4 contexts per grad chunk)  VJP (16 cotangents per batched backward)  N_CHECK (10)  SUMMARY=1
"""
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
E055 = HERE.parent / "055-close-contrast"
E056 = HERE.parent / "056-specificity"
RES = HERE / "results"
ARM = os.environ.get("ARM", "rkeep3e3fix")
SEEDS_FILE = Path(os.environ.get("SEEDS", str(E055 / "seeds.txt")))
ONLY = os.environ.get("ONLY")
SMOKE = os.environ.get("SMOKE") == "1"
CH = int(os.environ.get("CH", 4))
VJP = int(os.environ.get("VJP", 16))
N_CHECK = int(os.environ.get("N_CHECK", 10))
THETAS = (1e-2, 1e-3)                     # edge thresholds, as a fraction of the circuit's total |A_u| mass
N_SEQ, EVAL_BS = 64, 16
PIS = ("Z",) if SMOKE else ("Z", "A", "C")
OUT = RES / ("edges_%s%s.jsonl" % (ARM, "_smoke" if SMOKE else ""))
DATA = HERE / "data" / ARM


def fwd(G, inst, tokens, grad):
    inference = G["inference"]
    with contextlib.redirect_stdout(io.StringIO()):
        inference.forward(tokens, patcher=inst, grad_enabled=grad, return_activations=False, tokenize_final=False)


def setup_target(G, V, c):
    """Contexts, ablation values and the circuit as tensors, built as in 056 specificity."""
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_anchors, collect_site_means
    from pipeline.component_index import component_idx as comp_of

    bank, M0, KINDS, NK = (G[k] for k in ("bank", "M0", "KINDS", "NK"))
    seed = V.seed_of(c); sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    members = defaultdict(list)
    for n in c.nodes.values():
        if n is not seed:
            f = n.metadata["feature_id"]
            members[(f.layer, f.kind)].append((int(f.index), float(n.metadata.get("amplitude", 1.0))))
    order = lambda s: s[0] * len(KINDS) + KINDS.index(s[1])
    sites = sorted(members, key=order)
    comp = comp_of(layer, KINDS.index(kind), NK)
    pd_ = M0.build_probe_dataset(comp, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    sel = M0._neg_context_selector().select(comp, sl, "close", max_sequences=N_SEQ, batch_size=EVAL_BS, exact=False,
                                            non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)
    nt = sel.tokens[:N_SEQ]
    n_tr = V.split_n(int(pt.shape[0])); n_ntr = V.split_n(int(nt.shape[0]))
    UPS = set(upstream_sites(bank, layer, kind)); dev = pt.device
    means = {"Z": None,
             "A": collect_site_anchors(G["inference"], bank, pt[:n_tr], UPS, pa[:n_tr], pin_position_specific=False)[0],
             "C": collect_site_means(G["inference"], bank, nt[:n_ntr], UPS)}
    keep, alphas, nodes = {}, {}, []
    for s in sites:
        ms = sorted(members[s])
        keep[s] = torch.tensor([i for i, _ in ms], device=dev, dtype=torch.long)
        alphas[s] = torch.tensor([a for _, a in ms], device=dev, dtype=torch.float32)
        nodes += [(s[0], s[1], i, a) for i, a in ms]
    sae = bank.saes[kind][layer]
    seed_vec = (sae.encoder.weight[sl].detach(), sae._get_bias_eff()[sl].detach())
    return dict(key="%d.%s.%d" % (layer, kind, sl), layer=layer, kind=kind, sl=sl, site=(layer, kind), UPS=UPS,
                pt=pt[n_tr:], pa=pa[n_tr:], means=means, keep=keep, alphas=alphas, sites=sites, nodes=nodes,
                seed_vec=seed_vec, off={s: sum(len(members[t]) for t in sites[:i]) for i, s in enumerate(sites)})


def make(G, T, pi, **kw):
    from circuit.instrument.weighted_edges import WeightedCircuitGraph
    return WeightedCircuitGraph(G["bank"], T["keep"], T["alphas"], T["UPS"], T["site"], T["seed_vec"],
                                kw.pop("anchors"), site_means=T["means"][pi], respect_topk=(pi != "Z"),
                                topk=G["K"], **kw)


def edges_for(G, T, pi):
    """Linear pass: A (node attribution), E[d, u] (member edges), Et[u] (edges to the target), per-chunk caches."""
    sites, off, keep = T["sites"], T["off"], T["keep"]
    N = len(T["nodes"]); Ntot = int(T["pt"].shape[0])
    A = torch.zeros(N, dtype=torch.float64); Et = torch.zeros(N, dtype=torch.float64)
    E = torch.zeros(N, N, dtype=torch.float64)
    cache, n_seq_vjp = [], 0
    inference = G["inference"]
    inference.disable_compile()
    try:
        for c0 in range(0, Ntot, CH):
            tok, anc = T["pt"][c0:c0 + CH], T["pa"][c0:c0 + CH]
            with torch.enable_grad():
                # total gradients g_u
                it = make(G, T, pi, anchors=anc, mode="total")
                fwd(G, it, tok, True)
                g = torch.autograd.grad(it.metric_vec.sum() / Ntot, [it.taps[s] for s in sites], allow_unused=True)
                g = {s: (gg.detach() if gg is not None else torch.zeros_like(it.values[s])) for s, gg in zip(sites, g)}
                delta = {s: (it.values[s] - it.floors[s]).float() for s in sites}
                for s in sites:
                    A[off[s]:off[s] + len(keep[s])] += (g[s].float() * delta[s]).sum((0, 1)).double().cpu()
                del it
                # direct Jacobians
                ist = make(G, T, pi, anchors=anc, mode="stop")
                fwd(G, ist, tok, True)
                leaves = [ist.leaves[s] for s in sites]
                h = torch.autograd.grad(ist.metric_vec.sum() / Ntot, leaves, retain_graph=True, allow_unused=True)
                for s, hh in zip(sites, h):
                    if hh is not None:
                        Et[off[s]:off[s] + len(keep[s])] += (hh.float() * delta[s]).sum((0, 1)).double().cpu()
                for di, d in enumerate(sites):
                    ups = sites[:di]
                    live_d = ist.live[d]
                    if not ups or not live_d.requires_grad:
                        continue
                    nd = len(keep[d])
                    for j0 in range(0, nd, VJP):
                        js = list(range(j0, min(nd, j0 + VJP)))
                        cot = torch.zeros((len(js),) + tuple(live_d.shape), device=live_d.device, dtype=live_d.dtype)
                        for k, j in enumerate(js):
                            cot[k, :, :, j] = g[d][:, :, j].to(live_d.dtype)
                        try:
                            grads = torch.autograd.grad(live_d, [ist.leaves[u] for u in ups], grad_outputs=cot,
                                                        retain_graph=True, allow_unused=True, is_grads_batched=True)
                        except RuntimeError:
                            n_seq_vjp += 1
                            per = [torch.autograd.grad(live_d, [ist.leaves[u] for u in ups], grad_outputs=cot[k],
                                                       retain_graph=True, allow_unused=True) for k in range(len(js))]
                            grads = tuple(None if per[0][i] is None else torch.stack([p[i] for p in per])
                                          for i in range(len(ups)))
                        for u, gr in zip(ups, grads):
                            if gr is None:
                                continue
                            w = (gr.float() * delta[u].unsqueeze(0)).sum((1, 2))                # [K, n_u]
                            E[off[d] + j0:off[d] + j0 + len(js), off[u]:off[u] + len(keep[u])] += w.double().cpu()
                cache.append(dict(c0=c0, g={s: g[s].float().cpu() for s in sites}))
                del ist, leaves, h, g
    finally:
        inference.enable_compile()
    return A, E, Et, cache, n_seq_vjp


@torch.no_grad()
def causal_checks(G, T, pi, A, E, cache):
    """Nonlinear versions of the top node attributions and the top member edges."""
    from eval.ablation_faithfulness import CircuitOnlyPatcher

    sites, off, keep, nodes = T["sites"], T["off"], T["keep"], T["nodes"]
    Ntot = int(T["pt"].shape[0])
    site_of = {}
    for s in sites:
        for j in range(len(keep[s])):
            site_of[off[s] + j] = (s, j)
    inference = G["inference"]
    inference.disable_compile()
    base, m_mine, m_eval = [], 0.0, 0.0
    scales = {}
    for s in sites:
        sv = torch.ones(G["D"], device=keep[s].device, dtype=torch.float32)
        sv[keep[s]] = T["alphas"][s]
        scales[s] = sv
    try:
        for ch in cache:
            tok, anc = T["pt"][ch["c0"]:ch["c0"] + CH], T["pa"][ch["c0"]:ch["c0"] + CH]
            it = make(G, T, pi, anchors=anc, mode="total")
            fwd(G, it, tok, False)
            base.append(dict(m=float(it.metric_vec.sum()), v=dict(it.values), fl=dict(it.floors)))
            m_mine += float(it.metric_vec.sum())
            ev = CircuitOnlyPatcher(bank=G["bank"], keep_indices={s: set(keep[s].tolist()) for s in sites},
                                    in_scope=T["UPS"], seed_layer=T["layer"], seed_kind=T["kind"], seed_latent_idx=T["sl"],
                                    pos_argmax=anc, site_means=T["means"][pi], respect_topk=(pi != "Z"), topk=G["K"],
                                    keep_tensors=dict(keep), keep_scales=scales, capture_preact=True,
                                    seed_vector=T["seed_vec"])
            fwd(G, ev, tok, False)
            m_eval += float(ev.captured_preactivation) * int(tok.shape[0])
        node_rows = []
        for u in torch.argsort(A.abs(), descending=True)[:N_CHECK].tolist():
            s, j = site_of[u]; dm = 0.0
            for ch, b0 in zip(cache, base):
                tok, anc = T["pt"][ch["c0"]:ch["c0"] + CH], T["pa"][ch["c0"]:ch["c0"] + CH]
                it = make(G, T, pi, anchors=anc, mode="total", override=(s, j, b0["fl"][s][:, :, j]))
                fwd(G, it, tok, False)
                dm += b0["m"] - float(it.metric_vec.sum())
            node_rows.append((float(A[u]), dm / Ntot))
        edge_rows = []
        flat = torch.argsort(E.abs().flatten(), descending=True)[:N_CHECK].tolist()
        for f in flat:
            di, ui = divmod(f, E.shape[1])
            if E[di, ui] == 0:
                continue
            (sd, jd), (su, ju) = site_of[di], site_of[ui]
            wc = 0.0
            for ch, b0 in zip(cache, base):
                tok, anc = T["pt"][ch["c0"]:ch["c0"] + CH], T["pa"][ch["c0"]:ch["c0"] + CH]
                ref = make(G, T, pi, anchors=anc, mode="pin", pins=b0["v"], live=(sd, jd))
                fwd(G, ref, tok, False)
                alt = make(G, T, pi, anchors=anc, mode="pin", pins=b0["v"], live=(sd, jd),
                           override=(su, ju, b0["fl"][su][:, :, ju]))
                fwd(G, alt, tok, False)
                dv = (ref.values[sd][:, :, jd] - alt.values[sd][:, :, jd]).float().cpu()
                wc += float((ch["g"][sd][:, :, jd] * dv).sum())
            edge_rows.append((float(E[di, ui]), wc))
    finally:
        inference.enable_compile()
    return node_rows, edge_rows, m_mine / Ntot, m_eval / Ntot


def graph_stats(A, E, Et, theta):
    """Edges above theta x sum|A|: count, mean in-degree, share of nodes with a path to the target, longest path."""
    tau = theta * float(A.abs().sum())
    M = (E.abs() >= tau)                                         # [d, u]; E is strictly lower-triangular by site order
    to_t = (Et.abs() >= tau)
    N = E.shape[0]
    reach = to_t.clone()
    depth = torch.where(to_t, torch.ones(N, dtype=torch.long), torch.zeros(N, dtype=torch.long))
    for u in range(N - 1, -1, -1):                               # later nodes first (downstream has larger index)
        outs = M[:, u].nonzero().flatten()
        if len(outs):
            r = reach[outs]
            if r.any():
                reach[u] = True
                depth[u] = max(int(depth[u]), 1 + int(depth[outs[r]].max()))
    return dict(n_edges=int(M.sum()), n_to_target=int(to_t.sum()), mean_in=float(M.sum(1).float().mean()),
                reach_target=float(reach.float().mean()), max_depth=int(depth.max()))


def corr(rows):
    if len(rows) < 3:
        return None, None
    a = np.array(rows)
    r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if a[:, 0].std() > 0 and a[:, 1].std() > 0 else None
    return r, float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])))


def summarise():
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(OUT)])
    df = df[df.get("error").isna()] if "error" in df.columns else df
    cols = ["n_nodes", "replay_rel", "conservation_rel", "node_r", "node_sign", "edge_r", "edge_sign",
            "n_edges_0.01", "mean_in_0.01", "reach_target_0.01", "max_depth_0.01", "n_edges_0.001", "max_depth_0.001"]
    lines = ["# 057 WCM direct-effect edges, arm %s (%d rows)\n" % (ARM, len(df))]
    lines.append(df.groupby("pi")[cols].median().round(3).to_string())
    if "consensus_0.01" in df.columns:
        lines.append("\n## consensus across Z/A/C (theta 1e-2), per target")
        lines.append(df[df.pi == "C"][["seed", "n_nodes", "consensus_0.01", "union_0.01"]].to_string(index=False))
    lines.append("\n## per target")
    lines.append(df[["seed", "pi", "n_nodes", "m_mine", "m_eval", "conservation_rel", "node_r", "edge_r", "edge_sign",
                     "n_edges_0.01", "max_depth_0.01", "reach_target_0.01"]].round(3).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    (RES / ("summary_%s.md" % ARM)).write_text(txt, encoding="utf-8")


def main():
    RES.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SUMMARY") == "1":
        summarise(); return
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(E056))
    os.environ.setdefault("TAG", "wcm_edges_unused")
    import amp_eval_pass_v2 as V
    import specificity as S
    G = V.setup()
    seeds = [s for s in SEEDS_FILE.read_text().split() if s]
    if ONLY:
        seeds = [ONLY]
    circuits = S.load_circuits(ARM, set(seeds), V)
    print("arm %s: %d / %d circuits" % (ARM, len(circuits), len(seeds)), flush=True)
    done = set()
    if OUT.exists():
        done = {r["seed"] for r in map(json.loads, open(OUT)) if "pi" in r and r["pi"] == PIS[-1]}
    with open(OUT, "a") as fh:
        for s in seeds:
            if s in done or s not in circuits:
                continue
            ts = time.time()
            try:
                T = setup_target(G, V, circuits[s])
                saved, rows = dict(nodes=T["nodes"]), []
                for pi in PIS:
                    t0 = time.time()
                    A, E, Et, cache, nseq = edges_for(G, T, pi)
                    node_rows, edge_rows, m_mine, m_eval = causal_checks(G, T, pi, A, E, cache)
                    out = E.sum(0) + Et
                    row = dict(arm=ARM, seed=s, pi=pi, n_nodes=len(T["nodes"]), n_sites=len(T["sites"]),
                               m_mine=m_mine, m_eval=m_eval, replay_rel=abs(m_mine - m_eval) / max(abs(m_eval), 1e-9),
                               conservation_rel=float((out - A).abs().sum() / A.abs().sum().clamp(min=1e-12)),
                               A_sum=float(A.sum()), A_abs=float(A.abs().sum()), Et_sum=float(Et.sum()),
                               node_checks=node_rows, edge_checks=edge_rows, n_sequential_vjp=nseq,
                               secs=round(time.time() - t0, 1))
                    row["node_r"], row["node_sign"] = corr(node_rows)
                    row["edge_r"], row["edge_sign"] = corr(edge_rows)
                    for th in THETAS:
                        for k, v in graph_stats(A, E, Et, th).items():
                            row["%s_%g" % (k, th)] = v
                    rows.append(row)
                    saved[pi] = dict(A=A.float(), E=E.float(), Et=Et.float())
                    print("  %-15s %s  nodes %d  replay %.3g  conserv %.3g  node_r %s  edge_r %s  edges@1e-2 %d  "
                          "depth %d  %.0fs" % (s, pi, row["n_nodes"], row["replay_rel"], row["conservation_rel"],
                                               row["node_r"], row["edge_r"], row["n_edges_0.01"],
                                               row["max_depth_0.01"], row["secs"]), flush=True)
                if len(PIS) == 3:
                    for th in THETAS:
                        tau = {pi: th * float(saved[pi]["A"].abs().sum()) for pi in PIS}
                        sets = [(saved[pi]["E"].abs() >= tau[pi]) for pi in PIS]
                        sign = (torch.sign(saved["Z"]["E"]) == torch.sign(saved["A"]["E"])) & \
                               (torch.sign(saved["A"]["E"]) == torch.sign(saved["C"]["E"]))
                        cons = int((sets[0] & sets[1] & sets[2] & sign).sum()); uni = int((sets[0] | sets[1] | sets[2]).sum())
                        for r in rows:
                            r["consensus_%g" % th], r["union_%g" % th] = cons, uni
                torch.save(saved, DATA / ("%s.pt" % s))
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                rows = [dict(arm=ARM, seed=s, error="%s: %s" % (type(e).__name__, str(e)[:300]))]
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("  %s done in %.0fs" % (s, time.time() - ts), flush=True)
            if SMOKE:
                return
    summarise()


if __name__ == "__main__":
    main()
