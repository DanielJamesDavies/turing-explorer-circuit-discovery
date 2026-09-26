"""LOOK AT THE NODES (DAN-72 follow-up): are some inhibitory nodes obviously inhibitors?

For N_CIRC of the circuits that have leave-one-out roles in results/roles_pilot.jsonl, report:
  target   its own top contexts (peak token histogram + windows) and a few of its contrast contexts;
  nodes    the K most inhibitory and K most excitatory by leave-one-out under C (L_C), plus up to K_DIS
           nodes where the gradient rule (G) and L_C disagree. For each node:
             - scores: alpha, L_Z / L_A / L_C (as a share of a_pos), F_C, G sign;
             - how it fires on the TARGET's contexts: mean activation at the target's anchor on activating
               contexts, and mean per-sequence max on activating vs contrast contexts;
             - its own top contexts: peak token histogram and consistency, two windows;
             - direct logit effect: decoder direction x final-norm gain x unembedding, top promoted and
               suppressed tokens (a first-order "what would it say" read, not a causal claim).

  PYTHONPATH=src python experiments/054-inhibitor-roles/inspect_nodes.py   -> results/node_report.md (+ .json)
  env: N_CIRC (8), K (6), K_DIS (4), SEEDS (explicit comma list of target keys, overrides N_CIRC)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
RES = HERE / "results"
N_CIRC = int(os.environ.get("N_CIRC", 8))
K = int(os.environ.get("K", 6))
K_DIS = int(os.environ.get("K_DIS", 4))


def pick_targets():
    rows = [json.loads(l) for l in open(RES / "roles_pilot.jsonl")]
    rows = [r for r in rows if r.get("loo") and "vals" in r and "L_C" in r["vals"]]
    if os.environ.get("SEEDS"):
        want = [s.strip() for s in os.environ["SEEDS"].split(",") if s.strip()]
        return [r for r in rows if r["seed"] in want]
    rows.sort(key=lambda r: (r["layer"], r["kind"]))
    step = max(1, len(rows) // N_CIRC)
    return rows[::step][:N_CIRC]


def main():
    picked = pick_targets()
    os.environ["SEEDS"] = ",".join(r["seed"] for r in picked)
    os.environ.setdefault("TAG", "inspect_unused")
    sys.path.insert(0, str(ROOT))
    import torch
    import amp_eval_pass_v2 as V
    from model.tokenizer import Tokenizer
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import sparse_topk_to_dense

    items, _, _, _ = V.load_items()
    G = V.setup()
    inference, bank, M0, KINDS, NK, D = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D"))
    tok = Tokenizer()
    k2i = {k: i for i, k in enumerate(KINDS)}
    model = inference.model
    g_norm = model.transformer.norm_f.scale.detach().float()         # TuringLLM's RMSNorm gain
    W_U = model.lm_head.weight.detach().float()
    by_key = {V.key_of(c): c for c in items}
    dec = lambda ids: tok.decode(ids).replace("\n", "\\n")

    def latent_label(l, k, i):
        try:
            pd_ = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
        except Exception as e:  # noqa: BLE001
            return dict(error=str(e)[:80])
        if pd_ is None or pd_.pos_tokens.shape[0] == 0:
            return dict(error="no contexts")
        pt, pa = pd_.pos_tokens.cpu(), pd_.pos_argmax.cpu()
        peak = Counter(); windows = []
        for b in range(pt.shape[0]):
            p = int(pa[b]); ids = pt[b].tolist()
            peak[dec([ids[p]])] += 1
            if len(windows) < 3:
                windows.append(dec(ids[max(0, p - 8):p]) + " [[" + dec([ids[p]]) + "]]")
        n = pt.shape[0]; top = peak.most_common(4)
        return dict(peak=[(t, c) for t, c in top], consistency=top[0][1] / n, windows=windows, pd=pd_)

    def logit_effect(l, k, i, n=6):
        wd = bank.saes[k][l].decoder.weight.detach()[:, i].float().to(W_U.device)
        lg = (wd * g_norm.to(wd.device)) @ W_U.T
        up = [dec([int(t)]) for t in lg.topk(n).indices.tolist()]
        down = [dec([int(t)]) for t in (-lg).topk(n).indices.tolist()]
        return up, down

    def node_acts(nodes, pt, pa, nt):
        """per node: mean activation at the target's anchors (activating ctx), mean per-seq max on activating and contrast ctx."""
        want = {}
        for j, (s, i) in enumerate(nodes):
            want.setdefault(s, []).append((j, i))
        out = np.zeros((len(nodes), 3))
        for col, (tokens, anchors) in enumerate(((pt, pa), (nt, None))):
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
                        inference.forward(tokens[s0:s0 + 16], activations_callback=hook, return_activations=False, tokenize_final=False)
            finally:
                inference.enable_compile()
            for s, lst in want.items():
                A = torch.cat(cap[s], 0)                                     # [B, T, n_site_nodes]
                for c_, (j, _) in enumerate(lst):
                    if col == 0:
                        rr = torch.arange(A.shape[0])
                        anc = anchors.cpu().clamp(0, A.shape[1] - 1)
                        out[j, 0] = float(A[rr, anc, c_].mean())
                        out[j, 1] = float(A[:, :, c_].max(dim=1).values.mean())
                    else:
                        out[j, 2] = float(A[:, :, c_].max(dim=1).values.mean())
        return out

    report, dump = [], []
    report.append("# Node inspection: are some inhibitory nodes obviously inhibitors?\n")
    report.append("Roles from `roles_pilot.jsonl` (leave-one-out, share of a_pos). **L>0 excitatory, L<0 inhibitory.** "
                  "`act@anchor` = mean activation at the target's anchor on its activating contexts; "
                  "`max act / max ctr` = mean per-sequence max on activating / contrast contexts. "
                  "Logit effect = decoder direction × final-norm gain × unembedding (first-order, not causal).\n")
    for r in picked:
        c = by_key[r["seed"]]
        seed = V.seed_of(c); sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
        alphas = {}
        for nd in c.nodes.values():
            if nd is not seed:
                f = nd.metadata["feature_id"]
                alphas.setdefault((f.layer, f.kind), {})[int(f.index)] = float(nd.metadata.get("amplitude", 1.0))
        nodes = [(s, i) for s in sorted(alphas) for i in sorted(alphas[s])]
        vals = {k_: np.array(v) / r["a_pos_pre"] for k_, v in r["vals"].items()}
        lc = vals["L_C"]
        order = np.argsort(lc)
        inh = [j for j in order[:K] if lc[j] < 0]
        exc = [j for j in order[::-1][:K] if lc[j] > 0]
        dis = [j for j in np.argsort(-np.abs(lc)) if (vals["G"][j] < 0) != (lc[j] < 0) and j not in inh and j not in exc][:K_DIS]
        tl = latent_label(layer, kind, sl)
        pd_t = tl.pop("pd", None)
        report.append("\n---\n\n## Target %s  (%d nodes; release L_C %+.3f)\n" % (r["seed"], r["n"], r.get("release_L_C_C_pre", float("nan"))))
        if "error" in tl:
            report.append("target contexts unavailable: %s\n" % tl["error"]); continue
        report.append("- **peak tokens:** %s (consistency %.0f%%)" % (", ".join("%r×%d" % pc for pc in tl["peak"]), 100 * tl["consistency"]))
        for w in tl["windows"]:
            report.append("- activating: …%s" % w)
        nt = pd_t.neg_tokens[:64]
        for b in range(min(3, int(nt.shape[0]))):
            report.append("- contrast: …%s" % dec(nt[b].tolist()[-24:]))
        up, down = logit_effect(layer, kind, sl)
        report.append("- target logit effect: + %s | − %s\n" % (up, down))
        pt, pa = pd_t.pos_tokens[:64], pd_t.pos_argmax[:64]
        sel = [("inhibitory (most negative L_C)", inh), ("excitatory (most positive L_C)", exc), ("G and L_C disagree", dis)]
        all_j = [j for _, js in sel for j in js]
        acts = node_acts([nodes[j] for j in all_j], pt, pa, nt)
        a_of = {j: acts[n_] for n_, j in enumerate(all_j)}
        for title, js in sel:
            if not js:
                continue
            report.append("\n### %s\n" % title)
            report.append("| node | α | L_Z | L_A | L_C | F_C | G | act@anchor | max act / max ctr | peak tokens (consistency) | e.g. | logits + / − |")
            report.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for j in js:
                (s, i) = nodes[j]
                nl = latent_label(s[0], s[1], i); nl.pop("pd", None)
                up, down = logit_effect(s[0], s[1], i, n=4)
                a = a_of[j]
                pk = "; ".join("%r×%d" % pc for pc in nl.get("peak", [])[:3]) + (" (%.0f%%)" % (100 * nl["consistency"]) if "consistency" in nl else nl.get("error", ""))
                eg = (nl.get("windows") or [""])[0].replace("|", "/")
                report.append("| %d.%s.%d | %.2f | %+.3f | %+.3f | %+.3f | %+.3f | %s | %.2f | %.2f / %.2f | %s | …%s | %s / %s |"
                              % (s[0], s[1], i, alphas[s][i], vals.get("L_Z", lc)[j], vals.get("L_A", lc)[j], lc[j], vals["F_C"][j],
                                 "exc" if vals["G"][j] >= 0 else "inh", a[0], a[1], a[2], pk.replace("|", "/"), eg,
                                 ", ".join(repr(x) for x in up).replace("|", "/"), ", ".join(repr(x) for x in down).replace("|", "/")))
                dump.append(dict(target=r["seed"], group=title, node="%d.%s.%d" % (s[0], s[1], i), alpha=alphas[s][i],
                                 L={k_: float(vals[k_][j]) for k_ in vals if k_.startswith("L_")}, F_C=float(vals["F_C"][j]),
                                 G=float(vals["G"][j]), act_anchor=a[0], act_max=a[1], ctr_max=a[2],
                                 peak=nl.get("peak"), windows=nl.get("windows"), logit_up=up, logit_down=down))
        print("done", r["seed"], flush=True)
    (RES / "node_report.md").write_text("\n".join(report), encoding="utf-8")
    json.dump(dump, open(RES / "node_report.json", "w"), indent=1)
    print("->", RES / "node_report.md")


if __name__ == "__main__":
    main()
