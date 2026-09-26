"""What are these latents? Token-driven description of a target and chosen circuit nodes (no auto-interp labels).

For each latent: its own top contexts (peak-token histogram, consistency, windows with the peak token marked), its
first-order logit effect (decoder direction x final-norm gain x unembedding; a "what would it say" read, not a
causal claim), and how it fires on the TARGET's contexts: mean activation at the target's anchor and mean
per-sequence max on the target's activating contexts vs its close (silence-checked) contrast contexts. Plus the
consensus edge weights between the listed nodes from the 057 matrices (per ablation method, / sum|A|).

  PYTHONPATH=src python experiments/057-wcm-edges/describe.py
      -> results/describe_<target>.md
  env: TARGET (11.mlp.30743)  NODES (comma list of layer.kind.index)  ARM (rkeep3e3fix)  N_WIN (4)
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent / "049-circuit-graph"
ARM = os.environ.get("ARM", "rkeep3e3fix")
TARGET = os.environ.get("TARGET", "11.mlp.30743")
NODES = [s for s in os.environ.get("NODES", "2.resid.1164,3.resid.21272,4.resid.22454,5.resid.38187,6.resid.1448,"
                                              "7.resid.14260,9.resid.37889,10.resid.36122").split(",") if s]
N_WIN = int(os.environ.get("N_WIN", 4))
PIS = ("Z", "A", "C")


def main():
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("TAG", "describe_unused")
    import amp_eval_pass_v2 as V
    from model.tokenizer import Tokenizer
    from pipeline.component_index import component_idx as comp_of
    from sae.dense import sparse_topk_to_dense

    G = V.setup()
    inference, bank, M0, KINDS, NK, D = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D"))
    tok = Tokenizer(); k2i = {k: i for i, k in enumerate(KINDS)}
    model = inference.model
    g_norm = model.transformer.norm_f.scale.detach().float()
    W_U = model.lm_head.weight.detach().float()
    dec = lambda ids: tok.decode(ids).replace("\n", "\\n")
    parse = lambda s: (int(s.split(".")[0]), s.split(".")[1], int(s.split(".")[2]))

    def own_contexts(l, k, i):
        pd_ = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
        pt, pa = pd_.pos_tokens.cpu(), pd_.pos_argmax.cpu()
        peak, wins = Counter(), []
        for b in range(pt.shape[0]):
            p = int(pa[b]); ids = pt[b].tolist()
            peak[dec([ids[p]])] += 1
            if len(wins) < N_WIN:
                wins.append(dec(ids[max(0, p - 14):p]) + " [[" + dec([ids[p]]) + "]]" + dec(ids[p + 1:p + 4]))
        top = peak.most_common(5)
        return top, top[0][1] / pt.shape[0], wins, pd_

    def logit_effect(l, k, i, n=8):
        wd = bank.saes[k][l].decoder.weight.detach()[:, i].float().to(W_U.device)
        lg = (wd * g_norm.to(wd.device)) @ W_U.T
        return [dec([int(t)]) for t in lg.topk(n).indices.tolist()], [dec([int(t)]) for t in (-lg).topk(n).indices.tolist()]

    def acts_on(nodes, tokens, anchors):
        """[n, 2]: mean act at anchors (if given) and mean per-sequence max, for each (l, k, i)."""
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

    tl, tk, ti = parse(TARGET)
    top, cons, wins, pd_t = own_contexts(tl, tk, ti)
    comp = comp_of(tl, KINDS.index(tk), NK)
    sel = M0._neg_context_selector().select(comp, ti, "close", max_sequences=64, batch_size=16, exact=False,
                                            non_activation_threshold=0.0, filter_batch_size=32, load_window_size=256)
    pt, pa, nt = pd_t.pos_tokens[:64], pd_t.pos_argmax[:64], sel.tokens[:64]
    all_nodes = [parse(s) for s in NODES] + [(tl, tk, ti)]
    on_pos = acts_on(all_nodes, pt, pa)
    on_ctr = acts_on(all_nodes, nt, None)

    d = torch.load(HERE / "data" / ARM / ("%s.pt" % TARGET), weights_only=False)
    idx_of = {(int(L), K, int(I)): j for j, (L, K, I, _) in enumerate(d["nodes"])}
    alpha_of = {(int(L), K, int(I)): float(a) for L, K, I, a in d["nodes"]}

    rep = ["# Target %s and chain nodes (%s)\n" % (TARGET, ARM),
           "Token-driven reads, no auto-interp. `on target ctx` = mean activation at the target's anchor / mean "
           "per-sequence max on the target's activating contexts; `on contrast` = mean per-sequence max on its close "
           "contrast contexts. Logit effect is first-order (decoder × final-norm gain × unembedding), not causal.\n"]
    for n_, key in enumerate(NODES + [TARGET]):
        l, k, i = parse(key)
        top, cons, wins, _ = own_contexts(l, k, i)
        up, down = logit_effect(l, k, i)
        role = "TARGET" if key == TARGET else "node, α %.2f" % alpha_of.get((l, k, i), float("nan"))
        rep.append("\n## %s (%s)\n" % (key, role))
        rep.append("- **peak tokens:** %s — consistency %.0f%%" % (", ".join("%r×%d" % pc for pc in top), 100 * cons))
        for w in wins:
            rep.append("- …%s" % w)
        rep.append("- **on target ctx:** at anchor %.2f, seq max %.2f; **on contrast:** seq max %.2f"
                   % (on_pos[n_, 0], on_pos[n_, 1], on_ctr[n_, 1]))
        rep.append("- **logits:** + %s | − %s" % (up, down))
    rep.append("\n## Consensus-relevant edge weights between listed nodes (w / Σ|A|, per Z / A / C)\n")
    rep.append("| upstream → downstream | Z | A | C |")
    rep.append("|---|---|---|---|")
    keys = [parse(s) for s in NODES]
    for a in keys:
        for b in keys:
            if a not in idx_of or b not in idx_of:
                continue
            ws = [float(d[p]["E"][idx_of[b], idx_of[a]] / d[p]["A"].abs().sum()) for p in PIS]
            if max(abs(x) for x in ws) >= 1e-3:
                rep.append("| %d.%s.%d → %d.%s.%d | %+.4f | %+.4f | %+.4f |" % (a + b + tuple(ws)))
        if a in idx_of:
            ws = [float(d[p]["Et"][idx_of[a]] / d[p]["A"].abs().sum()) for p in PIS]
            if max(abs(x) for x in ws) >= 1e-3:
                rep.append("| %d.%s.%d → TARGET | %+.4f | %+.4f | %+.4f |" % (a + tuple(ws)))
    txt = "\n".join(rep)
    (HERE / "results" / ("describe_%s.md" % TARGET)).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
