"""Arena A1 (A7, DAN-197): Ge et al.'s circuits for the arena A targets, resumable per circuit. Run in ~/venvs/ge107:

  source ~/venvs/ge107/bin/ge_env.sh
  CUDA_VISIBLE_DEVICES= python -I -B experiments/107-true-baselines/arena_a/run_ge.py \
      --targets out/targets_smoke --out out/ge_smoke [--keys 9.tc.1,8.z.2] [--k 2] [--shard 0/8] [--n-tau 30]

One circuit = one (target, prompt): root = the target latent's feature act at the prompt's last position (the anchor;
BOS prepended, so position anchor + 1), README section 3 / ge.md section 9 root rule. Per circuit (ge_core.py):
  ungated   one gate-off backward through their linear graph (LN frozen): the post-hoc attributions, and Theorem 3.1
            at the full graph (G1) checked on every circuit (g1_rel_gap, must be <= 1e-4).
  hier      THEIR hierarchical_attribute_transformer_with_saes(...) at each of n_tau thresholds, log-spaced in tau / a_t
            (A7; ge_core.tau_grid), unmodified; LN frozen around it [ours, A3].
  posthoc   S(tau) = {attr_ungated > tau} (their read-out rule, signed); A11 recovery = the fixed-mask re-run on S.
Saved (node lists + values, not graphs): OUT/circuits/<key>/p<j>.pt with
  sites            the 24 candidate hook points (index -> name); node rows are (site, pos, latent) with pos in the
                   BOS-prefixed prompt (pos 0 = BOS) and latent in the dictionary's own index (z L5 / L7: 0..49,151);
  hier[t]          nodes [n, 3] int32, attr [n] float32, leaf parts (embed, bias_model, bias_dict, err_tc, err_z);
  posthoc_pool     the ungated attributions > tau_min (nodes, attr): S(tau) = pool rows with attr > tau (nested sets);
  posthoc[t]       n and the A11 leaf parts;
plus a_t, the record's peak (the BOS convention check), tau grid, ungated parts. A status row per circuit goes to
OUT/status.jsonl. A rerun skips every circuit whose file exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ge_core as GC  # noqa: E402  (sets the clone paths)
import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from lm_saes.circuit.graph import Node  # noqa: E402

import ge_dicts as GD  # noqa: E402

BOS = 50256


def load_rows(targets_dir: Path, keys=None, shard=None):
    rows = [json.loads(l) for l in (targets_dir / "prompts.jsonl").read_text().splitlines() if l.strip()]
    if keys:
        rows = [r for r in rows if r["key"] in keys]
    if shard:
        i, n = map(int, shard.split("/"))
        rows = [r for j, r in enumerate(rows) if j % n == i]
    return rows


def hook_for(row):
    l = row["layer"]
    if row["dict_kind"] == "tc":
        return "blocks.%d.hook_mlp_out.sae.hook_feature_acts" % l
    return "blocks.%d.attn.hook_z.sae.hook_feature_acts" % l


def pack(nodes, site_idx):
    if not nodes:
        return torch.zeros(0, 3, dtype=torch.int32), torch.zeros(0)
    keys = sorted(nodes, key=lambda k: (site_idx[k[0]], k[1]))
    idx = torch.tensor([[site_idx[hp]] + [int(x) for x in red.split(".")[1:]] for hp, red in keys], dtype=torch.int32)
    return idx, torch.tensor([nodes[k] for k in keys], dtype=torch.float32)


def upstream(saes, row):
    """the dictionaries at or before the root's site (z_l for l <= L; tc_l for l < L, and tc_L for a tc root). The
    others sit downstream of the root, so they cannot change its value or receive gradient from it: leaving them out of
    the splice changes no value, node set or attribution [ours, exact]. It is needed for memory: a downstream candidate's
    gate closure keeps its autograd ancestors alive (ge_core.hier_pass), and the backward from the root never runs
    those nodes, so their saved activations would never be freed (~5-9 GB per circuit at T = 64, measured)."""
    L, tc_root = row["layer"], row["dict_kind"] == "tc"
    out = []
    for s in saes:
        kind, l = GC.kind_layer(s)
        if (kind == "z" and l <= L) or (kind == "tc" and (l < L or (l == L and tc_root))):
            out.append(s)
    return out


def run_circuit(model, all_saes, row, j, n_tau, site_idx):
    saes = upstream(all_saes, row)
    leaves = GC.leaf_params(model, saes)
    cands = GC.candidates(saes)
    pr = row["prompts"][j]
    toks = torch.tensor([[BOS] + pr["tokens"]], dtype=torch.long)
    pos = toks.shape[1] - 1
    target = Node(hook_for(row), "0.%d.%d" % (pos, row["dict_latent"]))
    st = GC.clean_state(model, saes, toks)
    a_t = GC.root_value(st, target)
    out = dict(key=row["key"], j=j, seq_id=pr["seq_id"], anchor=pr["anchor"], pos=pos, tokens=toks[0].tolist(),
               a_t=a_t, peak=pr["peak"], peak_rel_diff=abs(a_t - pr["peak"]) / max(abs(pr["peak"]), 1e-9),
               sites=list(site_idx), n_tau=n_tau)
    if a_t <= 0:
        out.update(status="inactive")
        return out
    attrs, parts_u = GC.ungated_pass(model, saes, toks, target, cands, leaves, st)
    taus = GC.tau_grid(a_t, n_tau)
    pool = GC.posthoc_nodes(attrs, taus[0], target)
    pool_idx, pool_attr = pack(pool, site_idx)
    hier, post = [], []
    for tau in taus:
        nodes, parts = GC.hier_pass(model, saes, toks, target, cands, tau, leaves, st)
        idx, val = pack(nodes, site_idx)
        hier.append(dict(tau=tau, nodes=idx, attr=val, parts=parts))
        S = {k: v for k, v in pool.items() if v > tau}
        _, parts_m = GC.ungated_pass(model, saes, toks, target, cands, leaves, st,
                                     masks=GC.masks_from_nodes(S, st, target))
        post.append(dict(tau=tau, n=len(S), parts=parts_m))
    g1 = abs(sum(parts_u.values()) - a_t) / abs(a_t)
    out.update(status="ok", taus=taus, ungated_parts=parts_u, g1_rel_gap=g1, g1_passed=g1 <= 1e-4,
               n_positive_ungated=sum(int((a > 0).sum()) for a in attrs.values() if a is not None),
               hier=hier, posthoc_pool=dict(nodes=pool_idx, attr=pool_attr), posthoc=post)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--k", type=int, default=None, help="prompts per target (default: all in prompts.jsonl)")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--n-tau", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rss-gb", type=float, default=9.0, help="re-exec past this host RSS (ge_core.maybe_reexec)")
    a = ap.parse_args()
    out_dir = Path(a.out)
    rows = load_rows(Path(a.targets), set(a.keys.split(",")) if a.keys else None, a.shard)
    todo = [(r, j) for r in rows for j in range(len(r["prompts"]) if a.k is None else min(a.k, len(r["prompts"])))
            if not (out_dir / "circuits" / r["key"] / ("p%d.pt" % j)).exists()]
    print("ge: %d targets, %d circuits to run (%d already written)" % (
        len(rows), len(todo), sum(len(r["prompts"]) if a.k is None else min(a.k, len(r["prompts"])) for r in rows)
        - len(todo)), flush=True)
    if not todo:
        return
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained("gpt2", device=a.device)       # TL defaults (fold_ln, centred unembed)
    saes = GD.all_saes(a.device)
    GC.prepare(model, saes)
    site_idx = {c.hook_point: i for i, c in enumerate(GC.candidates(saes))}       # all 24, fixed indexing
    out_dir.mkdir(parents=True, exist_ok=True)
    status = open(out_dir / "status.jsonl", "a")
    for r, j in todo:
        t0 = time.time()
        try:
            res = run_circuit(model, saes, r, j, a.n_tau, site_idx)
        except Exception as e:  # noqa: BLE001
            res = dict(key=r["key"], j=j, status="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        res["seconds"] = time.time() - t0
        if res["status"] != "error":
            p = out_dir / "circuits" / r["key"] / ("p%d.pt" % j)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            torch.save(res, tmp)
            tmp.replace(p)                                                    # atomic: a file means a finished circuit
        row = {k: res.get(k) for k in ("key", "j", "status", "error", "a_t", "peak", "peak_rel_diff", "g1_rel_gap",
                                       "g1_passed", "n_positive_ungated", "seconds")}
        if res.get("status") == "ok":
            row["n_hier"] = [int(h["nodes"].shape[0]) for h in res["hier"]]
            row["n_posthoc"] = [h["n"] for h in res["posthoc"]]
        status.write(json.dumps(row) + "\n")
        status.flush()
        print("%s p%d %s a_t %s (peak %s)  G1 gap %s  hier %s..%s  post %s..%s  %.0fs" % (
            r["key"], j, res["status"], res.get("a_t"), res.get("peak"), res.get("g1_rel_gap"),
            row.get("n_hier", [None])[-1], row.get("n_hier", [None])[0], row.get("n_posthoc", [None])[-1],
            row.get("n_posthoc", [None])[0], res["seconds"]), flush=True)
        GC.maybe_reexec(a.max_rss_gb)


if __name__ == "__main__":
    main()
