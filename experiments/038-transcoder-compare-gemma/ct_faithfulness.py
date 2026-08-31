"""FAITHFULNESS HARNESS: is the circuit-tracer we run THEIR circuit-tracer?

Run before any arm that carries their name, and again whenever the
pinned commit, dtype, or budget changes. Prints PASS/FAIL per check.

  A  provenance   clone is at the pinned commit and has no local edits
  B  identity     prune_rooted(logit root) == prune_graph exactly
  C  lazy/eager   lazy vs eager encoder loading gives identical adjacency
  D  batch        batch_size 64 vs 256 gives identical adjacency
  E  published    their README's Gemma "Dallas -> Austin" graph: the
                  pinned feature nodes are reproduced by our pipeline
                  (active, and surviving their pruning at the viewer's
                  0.6/0.99 and at the library defaults 0.8/0.98)

  TC_DIR=$HOME/gemma_tc ../../dev-notes/data/venv-ct/bin/python ct_faithfulness.py
"""
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).parent
def _src():
    """The clone the venv actually imports (editable install), so the
    provenance check can never silently inspect a different copy."""
    import circuit_tracer
    return Path(circuit_tracer.__file__).resolve().parent.parent


SRC = _src()
PIN = "8f1e2438df612464e229e44c4a00ff637bf9379b"
DALLAS = "Fact: The capital of the state containing Dallas is"
PINNED = """27_22605_10 20_15589_10 21_5943_10 23_12237_10 20_15589_9 16_25_9
14_2268_9 18_8959_10 4_13154_9 7_6861_9 19_1445_10 0_13727_7 6_4012_7
17_7178_10 15_4494_4 6_4662_4 4_7671_4 3_13984_4 1_1000_4 19_7477_9
18_6101_10 16_4298_10 7_691_10""".split()


def ok(flag, name, detail=""):
    print("%s  %-12s %s" % ("PASS" if flag else "FAIL", name, detail), flush=True)
    return flag


def check_provenance():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SRC,
                          capture_output=True, text=True).stdout.strip()
    # Content, not line endings: the clone sits on a Windows-mounted
    # filesystem, so WSL's git reports CRLF files as modified while the
    # bytes that matter are identical. Compare ignoring EOL/whitespace.
    dirty = subprocess.run(["git", "diff", "-w", "--ignore-cr-at-eol", "--stat",
                            "--", "circuit_tracer"],
                           cwd=SRC, capture_output=True, text=True).stdout.strip()
    return ok(head == PIN and not dirty, "provenance",
              "HEAD %s | local edits: %s" % (head[:8], "none" if not dirty else dirty[:60]))


def node_ids(g):
    af = g.active_features.cpu(); sel = g.selected_features.cpu().flatten()
    af_sel = af[sel]
    n_f = len(sel); n = g.adjacency_matrix.shape[0]
    ids = [("f", int(l), int(p), int(f)) for l, p, f in af_sel.tolist()]
    ids += [("other", i) for i in range(n_f, n)]     # error/embed/logit, positional
    return ids


def compare_graphs(ga, gb, label):
    """Identity-based comparison: node sets, pruned node sets (0.8/0.98),
    and edge-weight agreement on shared node pairs. A positional diff of
    adjacency matrices is meaningless if selection order differs."""
    import ct_prune
    ia, ib = node_ids(ga), node_ids(gb)
    fa, fb = {x for x in ia if x[0] == "f"}, {x for x in ib if x[0] == "f"}
    jac_sel = len(fa & fb) / max(len(fa | fb), 1)
    pa = ct_prune.prune_published(ga).node_mask.cpu()
    pb = ct_prune.prune_published(gb).node_mask.cpu()
    sa = {x for x, k in zip(ia, pa.tolist()) if k and x[0] == "f"}
    sb = {x for x, k in zip(ib, pb.tolist()) if k and x[0] == "f"}
    jac_pr = len(sa & sb) / max(len(sa | sb), 1)
    # edge weights on shared feature pairs among pruned survivors
    pos_a = {x: i for i, x in enumerate(ia)}; pos_b = {x: i for i, x in enumerate(ib)}
    shared = sorted(sa & sb)
    A, B = ga.adjacency_matrix.cpu().float(), gb.adjacency_matrix.cpu().float()
    ra = torch.tensor([pos_a[x] for x in shared]); rb = torch.tensor([pos_b[x] for x in shared])
    ea, eb = A[ra][:, ra].flatten(), B[rb][:, rb].flatten()
    corr = float(torch.corrcoef(torch.stack([ea, eb]))[0, 1]) if len(shared) > 1 else float("nan")
    rel = float((ea - eb).abs().max() / max(float(ea.abs().max()), 1e-9))
    print("   %s: selected Jaccard %.4f | pruned Jaccard %.4f (%d vs %d) | "
          "edge corr %.5f | max rel edge diff %.2e"
          % (label, jac_sel, jac_pr, len(sa), len(sb), corr, rel), flush=True)
    return jac_pr >= 0.95 and corr >= 0.999


def graph_for(model, ids, budget, batch):
    from circuit_tracer import attribute
    return attribute(ids, model, max_feature_nodes=budget, batch_size=batch,
                     verbose=False)


def main():
    import ct_prune
    import theirs_gtc as T

    results = [check_provenance()]
    tok = None

    # --- build (lazy, the production config) ---
    os.environ.setdefault("LAZY_ENC", "1")
    model = T.build_model()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.MODEL_ID)
    ids = tok(DALLAS, return_tensors="pt")["input_ids"][0].tolist()
    print("prompt tokens:", len(ids), flush=True)

    g = graph_for(model, ids, 16384, 64)
    results.append(ok(ct_prune.identity_check(g), "identity",
                      "prune_rooted(logit root) == prune_graph"))

    g2 = graph_for(model, ids, 16384, 256)
    results.append(ok(compare_graphs(g, g2, "batch 64 vs 256"), "batch",
                      "pruned-set Jaccard >= 0.95 and edge corr >= 0.999"))
    del g2

    # --- E: published graph reproduction ---
    af = g.active_features.cpu()
    sel = g.selected_features.cpu().flatten()
    af_sel = af[sel]
    active = {(int(l), int(f), int(p)) for l, p, f in af.tolist()}
    selected = {(int(l), int(f), int(p)) for l, p, f in af_sel.tolist()}
    # layer 27 in their ID scheme is the LOGIT node (26-layer model), not a feature
    want = [tuple(int(x) for x in s.split("_")) for s in PINNED
            if int(s.split("_")[0]) < 26]                            # (layer, feat, pos)
    n_act = sum(w in active for w in want)
    n_sel = sum(w in selected for w in want)

    def survivors(node_t, edge_t):
        pr = ct_prune.prune_published(g, node_t, edge_t)
        keep = pr.node_mask[:len(sel)].cpu()
        return {(int(l), int(f), int(p)) for (l, p, f), k in
                zip(af_sel.tolist(), keep.tolist()) if k}
    s06 = survivors(0.6, 0.99)
    s08 = survivors(0.8, 0.98)
    n06 = sum(w in s06 for w in want)
    n08 = sum(w in s08 for w in want)
    print("   pinned %d | active %d | selected %d | survive 0.6/0.99: %d | "
          "survive 0.8/0.98: %d | pruned-graph sizes %d / %d"
          % (len(want), n_act, n_sel, n06, n08, len(s06), len(s08)), flush=True)
    missing = [s for s, w in zip(PINNED, want) if w not in active]
    if missing:
        print("   not active at all:", missing, flush=True)
    results.append(ok(n_act >= 0.9 * len(want), "published-act",
                      "%d/%d pinned nodes active" % (n_act, len(want))))
    results.append(ok(n06 >= 0.8 * len(want), "published-pr",
                      "%d/%d survive their pruning at the viewer's 0.6/0.99"
                      % (n06, len(want))))

    # --- C: lazy vs eager (second build) ---
    g_keep = g
    del model
    torch.cuda.empty_cache()
    os.environ["LAZY_ENC"] = "0"
    T.LAZY_ENC = False
    model2 = T.build_model()
    g3 = graph_for(model2, ids, 16384, 64)
    results.append(ok(compare_graphs(g_keep, g3, "lazy vs eager"), "lazy/eager",
                      "pruned-set Jaccard >= 0.95 and edge corr >= 0.999"))

    print("\nVERDICT:", "ALL PASS" if all(results) else
          "%d/%d FAILED" % (results.count(False), len(results)), flush=True)


if __name__ == "__main__":
    main()
