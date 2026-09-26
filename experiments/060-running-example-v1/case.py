"""THE RUNNING EXAMPLE UNDER PROTOCOL V1: the temperature latent (3.resid.35381, "temperature as a thermodynamic
state variable"; pre-v1 study in experiments/032-running-example).

The paper's case-study figure (Figure 5a) needs a circuit fitted under the final protocol and evidence that it is a
mechanism for this latent, not a concept amplifier. Steps, one process:

  1. contexts   059's protocol pools for this target: 64 strongest + mid-band reservoir, stratified split, close
                contrast contexts (pool_test.build, written to data/contexts.pt here, not 059's cache)
  2. fit        arms B (gamma 0.25, lambda 1e-3) and Bw (gamma 1, lambda 2e-3), train set 32 strongest + 16 mid
  3. eval       the eval pass on held-out strongest and held-out mid-band contexts
  4. specificity  056's diagnostic (siblings, lifts above the Top-K cut, rank)
  5. nodes      057's attribution A_u per node under Z / A / C (circuit's own run); nodes ranked by their weakest
                normalised attribution across the three (consensus importance)
  6. describe   token-driven read of the target and its top nodes: peak tokens + consistency, windows, activation on
                the target's contexts vs contrast contexts, first-order logit effect. No auto-interp labels; a node
                whose peak token is one string at ~100% consistency is flagged as a string detector.

  PYTHONPATH=src python experiments/060-running-example-v1/case.py      -> results/case_report.md (+ jsonl files)
  env: TARGET (3.resid.35381)  N_TOP (15)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "059-context-pool")); sys.path.insert(0, str(EXP / "049-circuit-graph"))
sys.path.insert(0, str(EXP / "056-specificity")); sys.path.insert(0, str(EXP / "057-wcm-edges"))
TARGET = os.environ.get("TARGET", "3.resid.35381")
N_TOP = int(os.environ.get("N_TOP", 15))
RES = HERE / "results"
ARMS = {"B": dict(gamma=0.25, lam=1e-3), "Bw": dict(gamma=1.0, lam=2e-3)}
PIS = ("Z", "A", "C")


def log(s):
    print(s, flush=True)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TAG", "case_unused")
    os.environ.setdefault("CTR_SOURCE", "close")
    import amp_eval_pass_v2 as V
    import protocol_harness as H
    P = H.P
    G = V.setup()

    # 1. contexts
    P.CTX = HERE / "data" / "contexts.pt"
    P.seeds = lambda: [TARGET]
    if not P.CTX.exists():
        P.build(G)
    ctx = torch.load(P.CTX, weights_only=False)
    rec = ctx[TARGET]
    log("contexts: top %d, mid %d" % (rec["n_top"], rec["n_mid"]))

    # 2. fit
    circuits = {}
    for arm, kw in ARMS.items():
        circuits[arm] = H.fit_arm(G, ctx, [TARGET], HERE / ("data_%s" % arm), train_arm="B", log=log, **kw).get(TARGET)

    # 3. eval
    pool = H.load_pool()
    for arm, c in circuits.items():
        if c is not None:
            H.eval_arm(G, V, ctx, [TARGET], {TARGET: c}, RES / "eval.jsonl", helds=("strong", "mid"), pool=pool,
                       tag=dict(arm=arm), log=log)

    # 4. specificity (held-out strongest)
    import specificity as S
    CleanTap, CircuitTap = S.make_patchers(G)
    spec_path = RES / "specificity.jsonl"
    done = {r["arm"] for r in map(json.loads, open(spec_path))} if spec_path.exists() else set()
    with open(spec_path, "a") as fh:
        for arm, c in circuits.items():
            if c is None or arm in done:
                continue
            H.patch_eval_contexts(G, rec, "strong")
            for r in S.score(G, c, V, CleanTap, CircuitTap, arm):
                fh.write(json.dumps(r) + "\n")
                log("  spec %s %s: target %s siblings %s control %s switched %s" % (
                    arm, r["pi"], r["target_faith_pre"], r["sibling_faith_median"], r["control_faith_median"],
                    r["switched_on_circuit"]))

    # 5. node attribution (057, circuit's own run) for the primary-candidate arm Bw (and B)
    import wcm_edges as W
    attr = {}
    for arm, c in circuits.items():
        if c is None:
            continue
        H.patch_eval_contexts(G, rec, "strong")
        T = W.setup_target(G, V, c)
        per = {}
        for pi in PIS:
            A, E, Et, _, _ = W.edges_for(G, T, pi)
            per[pi] = dict(A=A, Et=Et, E=E)
        An = torch.stack([per[p]["A"] / per[p]["A"].abs().sum() for p in PIS])
        same = (torch.sign(An) == torch.sign(An[0:1])).all(0)
        strength = torch.where(same, An.abs().min(0).values, torch.zeros_like(An[0]))
        attr[arm] = dict(nodes=T["nodes"], strength=strength, sign=torch.sign(An[0]),
                         direct_share={p: float(per[p]["Et"].sum() / per[p]["A"].sum()) for p in PIS})
        torch.save(dict(nodes=T["nodes"], per=per), HERE / "data" / ("attribution_%s.pt" % arm))
        log("  attribution %s: %d nodes, consensus-signed %d" % (arm, len(T["nodes"]), int(same.sum())))

    # 6. describe
    describe(G, V, rec, circuits, attr)


def describe(G, V, rec, circuits, attr):
    from model.tokenizer import Tokenizer
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import sparse_topk_to_dense
    import protocol_harness as H
    inference, bank, M0, KINDS, NK, D = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D"))
    tok = Tokenizer(); k2i = {k: i for i, k in enumerate(KINDS)}
    model = inference.model
    g_norm = model.transformer.norm_f.scale.detach().float(); W_U = model.lm_head.weight.detach().float()
    dec = lambda ids: tok.decode(ids).replace("\n", "\\n")

    def own(l, k, i, n_win=3):
        pd_ = M0.probe_builder.build_for_latent(comp_of(l, KINDS.index(k), NK), i, *_stores())
        pt, pa = pd_.pos_tokens.cpu(), pd_.pos_argmax.cpu()
        peak, wins = Counter(), []
        for b in range(pt.shape[0]):
            p = int(pa[b]); ids = pt[b].tolist(); peak[dec([ids[p]])] += 1
            if len(wins) < n_win:
                wins.append(dec(ids[max(0, p - 12):p]) + " [[" + dec([ids[p]]) + "]]" + dec(ids[p + 1:p + 4]))
        top = peak.most_common(4)
        return top, top[0][1] / max(1, pt.shape[0]), wins

    def logits(l, k, i, n=6):
        wd = bank.saes[k][l].decoder.weight.detach()[:, i].float().to(W_U.device)
        lg = (wd * g_norm.to(wd.device)) @ W_U.T
        return [dec([int(t)]) for t in lg.topk(n).indices.tolist()], [dec([int(t)]) for t in (-lg).topk(n).indices.tolist()]

    def acts(nodes, tokens, anchors):
        want = {}
        for j, (l, k, i) in enumerate(nodes):
            want.setdefault((l, k), []).append((j, i))
        cap = {s: [] for s in want}

        def hook(layer_idx, activations):
            for kd in KINDS:
                s = (layer_idx, kd)
                if s in want:
                    ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                    idx = torch.tensor([i for _, i in want[s]], device=ta.device)
                    cap[s].append(sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)[..., idx].cpu())
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(tokens.shape[0]), 16):
                    inference.forward(tokens[s0:s0 + 16], activations_callback=hook, return_activations=False,
                                      tokenize_final=False)
        finally:
            inference.enable_compile()
        out = np.full((len(nodes), 2), np.nan)
        for s, lst in want.items():
            A = torch.cat(cap[s], 0)
            for c_, (j, _) in enumerate(lst):
                if anchors is not None:
                    rr = torch.arange(A.shape[0]); anc = anchors.cpu().clamp(0, A.shape[1] - 1)
                    out[j, 0] = float(A[rr, anc, c_].mean())
                out[j, 1] = float(A[:, :, c_].max(dim=1).values.mean())
        return out

    l, k, i = H.parse(TARGET)
    lines = ["# Running example under protocol v1: %s\n" % TARGET]
    top, cons, wins = own(l, k, i, n_win=5)
    up, down = logits(l, k, i)
    lines.append("**Target.** Peak tokens %s (consistency %.0f%%). Logits + %s | − %s" % (
        ", ".join("%r×%d" % t for t in top), 100 * cons, up, down))
    lines += ["- …%s" % w for w in wins]
    ev = [json.loads(x) for x in open(RES / "eval.jsonl")] if (RES / "eval.jsonl").exists() else []
    lines.append("\n## Scores (held-out; activation read)\n")
    lines.append("| arm | held | nodes | free0 | freeM_topk | freeN_topk | necessity | sufficiency to induce |")
    lines.append("|---|---|---|---|---|---|---|---|")
    f = lambda x: "%.3f" % x if isinstance(x, float) else "-"
    for r in ev:
        if "error" in r:
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (r["arm"], r["held"], r["n"], f(r.get("free0_tk")),
                     f(r.get("freeM_topk_tk")), f(r.get("freeN_topk_tk")), f(r.get("phi_sup_blind_tk")),
                     f(r.get("phi_cf_alpha_blind_tk"))))
    sp = [json.loads(x) for x in open(RES / "specificity.jsonl")] if (RES / "specificity.jsonl").exists() else []
    lines.append("\n## Specificity (held-out strongest; pre-activation faithfulness)\n")
    lines.append("| arm | ablation | target | siblings (median) | control (median) | n siblings | lifted above cut (circuit / empty) | target in Top-K |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sp:
        lines.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s |" % (r["arm"], r["pi"], f(r.get("target_faith_pre")),
                     f(r.get("sibling_faith_median")), f(r.get("control_faith_median")), r.get("n_siblings"),
                     r.get("switched_on_circuit"), r.get("switched_on_empty"), f(r.get("in_topk_circuit"))))
    pt, pa, nt = M0.build_probe_dataset(0, 0).pos_tokens, M0.build_probe_dataset(0, 0).pos_argmax, rec["neg"].to(G["device"])
    for arm, a in attr.items():
        order = torch.argsort(a["strength"], descending=True)[:N_TOP].tolist()
        nodes = [tuple(a["nodes"][j][:3]) for j in order]
        alpha = [a["nodes"][j][3] for j in order]
        on_pos = acts([(int(x), y, int(z)) for x, y, z in nodes], pt, pa)
        on_ctr = acts([(int(x), y, int(z)) for x, y, z in nodes], nt, None)
        lines.append("\n## Top %d nodes, arm %s (by consensus attribution; direct-to-target share Z/A/C %s)\n" % (
            N_TOP, arm, " / ".join("%.2f" % a["direct_share"][p] for p in PIS)))
        lines.append("| node | α | attribution (min over Z/A/C, share) | sign | peak tokens (consistency) | e.g. | on target ctx (anchor / seq max) | on contrast (seq max) | logits + |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for n_, ((nl, nk, ni), al, j) in enumerate(zip(nodes, alpha, order)):
            top, cons, wins = own(int(nl), nk, int(ni), n_win=1)
            up, _ = logits(int(nl), nk, int(ni), n=4)
            flag = " **string detector?**" if cons >= 0.9 else ""
            lines.append("| %s%d/%d | %.2f | %.3f | %s | %s (%.0f%%)%s | …%s | %.2f / %.2f | %.2f | %s |" % (
                {"attn": "A", "mlp": "M", "resid": "R"}[nk], int(nl), int(ni), al, float(a["strength"][j]),
                "+" if a["sign"][j] >= 0 else "−", "; ".join("%r×%d" % t for t in top).replace("|", "/"), 100 * cons,
                flag, (wins[0] if wins else "").replace("|", "/"), on_pos[n_, 0], on_pos[n_, 1], on_ctr[n_, 1],
                ", ".join(repr(x) for x in up).replace("|", "/")))
    txt = "\n".join(lines)
    (RES / "case_report.md").write_text(txt, encoding="utf-8")
    log(txt)


def _stores():
    from store.context import mid_ctx, neg_ctx, top_ctx
    return top_ctx, mid_ctx, neg_ctx


if __name__ == "__main__":
    main()
