"""Arena A (A8 companion, ge107): Ge et al.'s leaf-sum recovery for circuits NOT found by Ge (WCM, Dunefsky), by their
own linear graph with a fixed keep mask: ge.md A11, the same mechanism run_ge.py uses for the post-hoc sets
(ge_core.ungated_pass with masks = gradient only through the kept nodes; LN frozen; their apply_sae splice and
detach_hook on the attention scores, unmodified). Recovery under the three readings (ge_core.recovery): interp =
embeddings / a_t (the paper's, primary), nonerr = embeddings + biases, total = the root of G'' / a_t (bounded).

Jobs come from `score_arena_a.py --emit-ge-jobs` (.venv, no model): one per (unit, prompt), with the prompt tokens
(no BOS: prepended here, root at the last position, as run_ge.py) and the nodes in Ge's space
[hook_point, pos (BOS-prefixed), dictionary latent]:
  WCM       position-collapsed members at every non-BOS position of the prompt, on each of the target's K prompts
            (score_arena_a averages the K readings per unit) [ours: WCM has no positions]
  Dunefsky  their graph's transcoder nodes at their own positions; no z nodes (their universe has none), so in Ge's
            graph attention reaches the root only through the z SAE error leaves, which are detached [ours, labelled]
The root is always kept (ge_core.masks_from_nodes, A8). Only the dictionaries at or before the root's site are spliced
(run_ge.upstream: exact). Resumable by job id (OUT append + flush; a torn last line is skipped).

  source ~/venvs/ge107/bin/ge_env.sh
  CUDA_VISIBLE_DEVICES= python -I -B experiments/107-true-baselines/arena_a/ge_recovery_tcz.py \
      --jobs <OUT>/ge_jobs.a1.jsonl [--jobs ...] --out <OUT>/ge_recovery.jsonl [--threads 2] [--limit N] [--check]
  --check: per job also the all-kept mask (every node of the upstream dictionaries + root), whose leaf sum must equal
           a_t (Theorem 3.1, G1 through the masked path; check_rel_gap <= 1e-4)
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

BOS = 50256


def read_jsonl(path):
    out = []
    p = Path(path)
    if p.exists():
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def node_set(job):
    """positional [hook_point, pos, latent] rows -> {(hook_point, '0.pos.latent')} (their reduction string, batch 0);
    rows with pos None (WCM, position-collapsed) are left to masks_for"""
    return {(hp, "0.%d.%d" % (int(p), int(f))) for hp, p, f in job["nodes"] if p is not None}


def masks_for(job, st, target):
    """ge_core.masks_from_nodes on the positional rows (root always kept, A8), plus every pos-None row at all non-BOS
    positions 1..T-1"""
    m = GC.masks_from_nodes(node_set(job), st, target)
    for hp, p, f in job["nodes"]:
        if p is None:
            m[hp][0, 1:, int(f)] = 1.0
    return m


def run_job(model, all_saes, job, check=False):
    import run_ge as RG
    from lm_saes.circuit.graph import Node
    row = dict(layer=job["layer"], dict_kind=job["dict_kind"])
    saes = RG.upstream(all_saes, row)
    leaves = GC.leaf_params(model, saes)
    cands = GC.candidates(saes)
    toks = torch.tensor([[BOS] + job["tokens"]], dtype=torch.long)
    pos = toks.shape[1] - 1
    target = Node(RG.hook_for(row), "0.%d.%d" % (pos, job["dict_latent"]))
    st = GC.clean_state(model, saes, toks)
    a_t = GC.root_value(st, target)
    out = dict(job=job["job"], uid=job["uid"], key=job["key"], j=job["j"], method=job["method"], a_t=a_t,
               n_nodes=len(job["nodes"]))
    if a_t <= 0:
        out.update(status="inactive")
        return out
    hps = set(st["acts"])
    bad = [n for n in job["nodes"] if n[0] not in hps]
    if bad:
        raise ValueError("%d nodes at dictionaries not upstream of the root, e.g. %s" % (len(bad), bad[0]))
    masks = masks_for(job, st, target)
    out["n_mask"] = int(sum(float(v.sum()) for v in masks.values()))
    _, parts = GC.ungated_pass(model, saes, toks, target, cands, leaves, st, masks=masks)
    rec = dict(GC.recovery(parts, a_t), bias=(parts["bias_model"] + parts["bias_dict"]) / a_t,
               err=(parts["err_tc"] + parts["err_z"]) / a_t)
    out.update(status="ok", parts=parts, rec=rec)
    if check:
        full = {hp: torch.ones_like(a) for hp, a in st["acts"].items()}
        _, pf = GC.ungated_pass(model, saes, toks, target, cands, leaves, st, masks=full)
        out["check_rel_gap"] = abs(sum(pf.values()) - a_t) / abs(a_t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rss-gb", type=float, default=9.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    jobs = [j for p in a.jobs for j in read_jsonl(p)]
    done = {r["job"] for r in read_jsonl(a.out) if r.get("status") in ("ok", "inactive")}
    todo = [j for j in jobs if j["job"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print("ge_recovery: %d jobs, %d done, %d to run" % (len(jobs), len(done), len(todo)), flush=True)
    if not todo:
        return
    from transformer_lens import HookedTransformer
    import ge_dicts as GD
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained("gpt2", device=a.device)       # TL defaults, as run_ge.py
    saes = GD.all_saes(a.device)
    GC.prepare(model, saes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "a")
    for job in todo:
        t0 = time.time()
        try:
            res = run_job(model, saes, job, a.check)
        except Exception as e:  # noqa: BLE001
            res = dict(job=job["job"], uid=job["uid"], status="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))
        res["seconds"] = round(time.time() - t0, 2)
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        rc = res.get("rec") or {}
        print("%-48s %s n %5s a_t %s interp %s nonerr %s total %s%s %.1fs" % (
            job["job"], res["status"], res.get("n_nodes"), res.get("a_t"), rc.get("interp"), rc.get("nonerr"),
            rc.get("total"), "" if "check_rel_gap" not in res else " check %.1e" % res["check_rel_gap"],
            res["seconds"]), flush=True)
        GC.maybe_reexec(a.max_rss_gb)


if __name__ == "__main__":
    main()
