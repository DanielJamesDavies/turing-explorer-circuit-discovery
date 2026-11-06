"""Arena A2 (A7, DAN-200): Dunefsky et al.'s circuits for the arena A transcoder targets, resumable per circuit. Run in
~/venvs/tc107:

  source ~/venvs/tc107/bin/tc_env.sh
  CUDA_VISIBLE_DEVICES= python -I -B experiments/107-true-baselines/arena_a/run_dunefsky.py \
      --targets out/targets_smoke --out out/dunefsky_smoke [--keys ...] [--k 2] [--shard 0/8] \
      [--branches 5,10,15,20,30,40,60] [--iters 5] [--no-gate]

Their code, unmodified (dunefsky.md section 6, README section 4): HookedTransformer.from_pretrained('gpt2') (TL 1.11.0
defaults) -> the 12 transcoders -> model.run_with_cache(toks) on the prompt truncated after the anchor (BOS prepended,
A15) -> make_sae_feature_vector(tc[L], i, token=-1) -> greedy_get_top_paths(model, tcs, cache, fv, num_iters,
num_branches, do_raw_attribution=True) -> paths_to_graph -> add_error_nodes_to_graph (own-eval column only).
Read-out glue [ours, no change to their code]: every graph node as a row (type, layer, token, latent / head, contrib);
the circuit = the distinct (layer, latent) transcoder nodes, root excluded (root = the '@-1' node; the root's own
(layer, latent) at an earlier token is kept in the rows but dropped from the distinct set, flagged); heads, embeddings
and error nodes in their own columns.
Only tc targets (the A-TC universe, dunefsky.md section 7.1); z targets are skipped. Gate at start-up (D-F1): their
walkthrough R1 (mlp8tc[89] on the rifle prompt, 3 x 15, raw) must print the notebook's saved text exactly.
Saved: OUT/circuits/<key>/p<j>.b<nb>.pt (one file per (prompt, num_branches)); OUT/status.jsonl. Reruns skip files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dunefsky_checks as DC  # noqa: E402  (puts the clone on sys.path; loader glue)
import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from transcoder_circuits.circuit_analysis import (  # noqa: E402
    ComponentType, add_error_nodes_to_graph, greedy_get_top_paths, make_sae_feature_vector, paths_to_graph)

BOS = 50256
ROOT_SUFFIX = "@-1"


def gate_df1(model):
    """D-F1 (R1) inside the runner: their walkthrough paths text, exact."""
    t0 = time.time()
    res = DC.walkthrough(model, {"R1"})["R1"]
    res["seconds"] = time.time() - t0
    return res


def node_row(name, fv):
    c = fv.component_path[-1]
    return dict(name=name, type=c.component_type.value, layer=c.layer, token=c.token, head=c.attn_head,
                latent=c.feature_idx, contrib=None if fv.contrib is None else float(fv.contrib),
                root=name.endswith(ROOT_SUFFIX) and len(fv.component_path) == 1)


def read_out(nodes, edges, nodes_e, root_layer, root_latent):
    rows = [node_row(k, v) for k, v in nodes.items()]
    tc = [r for r in rows if r["type"] == "mlp" and r["latent"] is not None and not r["root"]]
    distinct = sorted({(r["layer"], r["latent"]) for r in tc})
    root_again = (root_layer, root_latent) in distinct
    distinct = [d for d in distinct if d != (root_layer, root_latent)]
    err = [node_row(k, v) for k, v in nodes_e.items() if k not in nodes]
    return dict(rows=rows, n_edges=len(edges), tc_latents=distinct, n_tc_latents=len(distinct),
                root_latent_at_other_token=root_again,
                heads=sorted({(r["layer"], r["head"]) for r in rows if r["type"] == "attn"}),
                head_nodes=sorted({(r["layer"], r["head"], r["token"]) for r in rows if r["type"] == "attn"}),
                embeds=sorted({r["token"] for r in rows if r["type"] == "embed"}), error_nodes=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--branches", default="5,10,15,20,30,40,60")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    branches = [int(x) for x in a.branches.split(",")]
    rows = [json.loads(l) for l in (Path(a.targets) / "prompts.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["kind"] == "tc" and (not a.keys or r["key"] in set(a.keys.split(",")))]
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        rows = [r for j, r in enumerate(rows) if j % n == i]

    def path(r, j, nb):
        return out_dir / "circuits" / r["key"] / ("p%d.b%d.i%d.pt" % (j, nb, a.iters))
    todo = [(r, j, nb) for r in rows for j in range(len(r["prompts"]) if a.k is None else min(a.k, len(r["prompts"])))
            for nb in branches if not path(r, j, nb).exists()]
    print("dunefsky: %d tc targets, %d circuits to run" % (len(rows), len(todo)), flush=True)
    if not todo:
        return
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = open(out_dir / "status.jsonl", "a")
    if not a.no_gate:
        g = gate_df1(model)
        status.write(json.dumps(dict(gate="D-F1", **{k: g[k] for k in ("passed", "n_diff", "root", "seconds")})) + "\n")
        status.flush()
        print("gate D-F1 (R1): %s, root %.6f, %.0fs" % ("PASS" if g["passed"] else "FAIL", g["root"], g["seconds"]),
              flush=True)
        if not g["passed"]:
            raise SystemExit("D-F1 failed: their walkthrough paths do not reproduce; not running")
    tcs = [DC.load_transcoder(l) for l in range(12)]
    dev = next(tcs[0].parameters()).device
    cache_key, cache = None, None
    for r, j, nb in todo:
        pr = r["prompts"][j]
        t0 = time.time()
        res = dict(key=r["key"], j=j, num_branches=nb, num_iters=a.iters, seq_id=pr["seq_id"], anchor=pr["anchor"],
                   peak=pr["peak"])
        try:
            if cache_key != (r["key"], j):
                toks = torch.tensor([[BOS] + pr["tokens"]], dtype=torch.long, device=dev)
                _, cache = model.run_with_cache(toks)
                cache_key = (r["key"], j)
            L, i = r["layer"], r["latent"]
            fv = make_sae_feature_vector(tcs[L], i, token=-1)
            all_paths = greedy_get_top_paths(model, tcs, cache, fv, num_iters=a.iters, num_branches=nb,
                                             do_raw_attribution=True)
            edges, nodes = paths_to_graph(all_paths)
            _, nodes_e = add_error_nodes_to_graph(model, cache, tcs, edges, nodes)
            root = float(all_paths[0][0][0].contrib)
            res.update(status="ok", tokens=toks[0].tolist(), root=root,
                       peak_rel_diff=abs(root - pr["peak"]) / max(abs(pr["peak"]), 1e-9),
                       n_paths=sum(len(p) for p in all_paths), **read_out(nodes, edges, nodes_e, L, i))
        except Exception as e:  # noqa: BLE001
            res.update(status="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        res["seconds"] = time.time() - t0
        if res["status"] == "ok":
            p = path(r, j, nb)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            torch.save(res, tmp)
            tmp.replace(p)
        row = {k: res.get(k) for k in ("key", "j", "num_branches", "num_iters", "status", "error", "root", "peak",
                                       "peak_rel_diff", "n_tc_latents", "n_edges", "root_latent_at_other_token",
                                       "seconds")}
        status.write(json.dumps(row) + "\n")
        status.flush()
        print("%s p%d b%d %s root %s (peak %s) tc latents %s heads %s %.1fs" % (
            r["key"], j, nb, res["status"], res.get("root"), pr["peak"], res.get("n_tc_latents"),
            len(res.get("heads", [])), res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
