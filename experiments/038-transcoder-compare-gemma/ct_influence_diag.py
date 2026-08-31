"""VERIFICATION: why are seed-rooted circuit-tracer circuits 3-6 nodes
on Llama but 500-900 on Gemma?

Hypothesis: the row-normalised influence budget is spent on ERROR and
EMBEDDING nodes before it reaches many features. TopK-32 transcoders
leave more reconstruction in the error term than JumpReLU-~100, so on
Llama the seed's input is mostly "error", and their 0.8 cumulative node
threshold fills up on non-feature nodes.

For one window of the first seed in each arena: root the influence at
the seed's rows (their compute_node_influence, unchanged) and report
the influence mass by node class, plus how many FEATURE nodes the 0.8
cumulative threshold admits vs how many it would admit if features were
ranked among themselves.

  ARENA=gemma|llama ../../dev-notes/data/venv-ct/bin/python ct_influence_diag.py
"""
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ARENA = os.environ.get("ARENA", "gemma")


def main():
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "035-transcoder-compare"))
    import ct_prune
    from circuit_tracer import attribute
    from circuit_tracer.graph import compute_node_influence, find_threshold

    if ARENA == "gemma":
        import theirs_gtc as T
        scan = torch.load(HERE / "scan_gtc.pt", weights_only=False)
    else:
        import theirs_llama as T
        scan = torch.load(HERE.parent / "035-transcoder-compare"
                          / "scan_llama.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    S = seeds[sorted(seeds)[0]]
    L, sl = S["layer"], S["latent"]
    ids = toks[S["pos_windows"]][0].tolist()
    model = T.build_model()
    g = attribute(ids, model, max_feature_nodes=16384,
                  batch_size=int(os.environ.get("BATCH", 64)), verbose=False)

    adj = g.adjacency_matrix
    n_tokens = len(g.input_tokens)
    n_logits = len(g.logit_targets)
    n_feat = len(g.selected_features)
    n = adj.shape[0]
    n_err = n - n_feat - n_tokens - n_logits
    rows, af_sel = ct_prune.seed_rows(g, L, sl)
    w = torch.zeros(n, device=adj.device, dtype=adj.dtype)
    w[rows] = 1.0
    infl = compute_node_influence(adj, w)

    cls = torch.zeros(n, dtype=torch.long)
    cls[:n_feat] = 0                                   # features
    cls[n_feat:n_feat + n_err] = 1                     # error nodes
    cls[n_feat + n_err:n_feat + n_err + n_tokens] = 2  # embeddings
    cls[-n_logits:] = 3                                # logits
    tot = float(infl.sum())
    print("[%s] seed L%d/%d | nodes: %d feat, %d err, %d embed, %d logit"
          % (ARENA, L, sl, n_feat, n_err, n_tokens, n_logits))
    for c, name in [(0, "features"), (1, "error"), (2, "embed"), (3, "logit")]:
        m = float(infl[cls == c].sum())
        print("  influence share %-9s %6.2f%%" % (name, 100 * m / max(tot, 1e-12)))

    thr = find_threshold(infl, 0.8)
    keep = infl >= thr
    kf = int(keep[:n_feat].sum())
    ke = int(keep[n_feat:n_feat + n_err].sum())
    # features ranked among THEMSELVES to 80% of feature mass
    fi = infl[:n_feat]
    thr_f = find_threshold(fi, 0.8)
    kf_own = int((fi >= thr_f).sum())
    print("  0.8 cumulative over ALL nodes keeps: %d features + %d error nodes"
          % (kf, ke))
    print("  0.8 cumulative over FEATURES ONLY would keep: %d features" % kf_own)


if __name__ == "__main__":
    main()
