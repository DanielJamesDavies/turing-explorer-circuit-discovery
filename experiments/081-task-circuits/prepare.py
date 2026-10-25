"""Experiment 081, stage 1 (GPU): generated contrast sets for the tasks whose datasets have NO patch / counterfactual
prompts (ROME known_1000 and every BLiMP subset; task["contrast"] == "generated").

Daniel's rule (2026-10-21), protocol v1's "close, verified silent" contrast contexts for latents, transferred to an
output endpoint:
  candidates  the OTHER training prompts of the same dataset / subset (never a held-out prompt; never a prompt of the
              item's own group, which would be the item itself or a duplicate prefix)
  ranking     cosine similarity to the item's prompt of the mean-pooled final-layer residual (the last layer's resid
              site activation, over real tokens; task_common.cosine_rank). This mirrors
              src/utils/neg_context_selector.py's "close" selector, which cannot be reused directly: it ranks corpus
              store sequences by their stored representation (seq_repr), not arbitrary prompts.
  silent      on the candidate prompt the model does NOT prefer the item's good answer over its bad one:
              metric(good_i, bad_i | candidate) <= 0 (task_common.SILENT_MAX; the summed teacher-forced log-prob
              difference for multi-token answers). Measured for each item's TOP_K_CANDIDATES (32) closest candidates.
  pool        round robin over the training items (seeded order), each contributing its closest silent candidate not
              yet pooled, until POOL_SIZE (64) distinct prompts (task_common.build_pool). The pool's mean is the C
              ablation value, in training and in scoring.
Cost and memory (2026-10-21, full training pools: ~700-900 prompts per generated-contrast task under GPT-2): the
embeddings are computed once per task; the cosine ranking is built in row chunks and keeps only each item's 32
closest; the candidate metrics read one chunk of candidates' [rows, vocab] log-probs at a time (engine.PROBE_CHUNK).
Writes tasks/<MODEL>/<task>.contrast.json: the pool (row ids), the rule's parameters, and diagnostics (silent share
of the measured candidates, the rank of each item's first silent candidate, the pool members' cosine to the item
that chose them). Resumable per task (an existing file is kept unless FORCE=1).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/081-task-circuits/prepare.py
  env: MODEL (gpt2-small)  TASKS (default: every generated-contrast task file)  FORCE=1
"""
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402


def generated_tasks():
    want = [t for t in os.environ.get("TASKS", "").split(",") if t]
    names = want or [p.stem for p in C.task_files()]
    return [n for n in names if C.task_spec(n)["contrast"] == "generated"]


def build_contrast(eng, task: dict) -> dict:
    """The contrast pool of one task (see the module docstring)."""
    rows = task["rows"]
    tr = list(task["split"]["train"])
    mode = task.get("answer_mode") or C.answer_mode(task)
    import torch
    t, _, v = C.pad_right([rows[i]["tokens"] for i in tr])
    reps = eng.reps(eng._t(t), eng._t(v, torch.bool))             # once per task: [n_train, d_model]
    groups = [rows[i]["group"] for i in tr]
    by_group = {}
    for j, g in enumerate(groups):
        by_group.setdefault(g, []).append(j)
    excl = [by_group[g] for g in groups]
    top = C.cosine_rank(reps.numpy(), reps.numpy(), exclude=excl, top_k=C.TOP_K_CANDIDATES)
    pairs = [(tr[i], tr[j]) for i, r in enumerate(top) for j in r]
    metric = eng.candidate_metric(task, pairs, mode)
    silent = {(i, j): C.is_silent(metric[(tr[i], tr[j])]) for i, r in enumerate(top) for j in r}
    order = list(range(len(tr)))
    random.Random("%s-pool-%d" % (task["name"], C.TASK_SEED)).shuffle(order)
    pool_local = C.build_pool(top, silent, C.POOL_SIZE, order)
    import numpy as np
    sims = reps.numpy() / np.maximum(np.linalg.norm(reps.numpy(), axis=1, keepdims=True), 1e-12)
    first_silent = []
    for i, r in enumerate(top):
        k = next((rank for rank, j in enumerate(r) if silent[(i, j)]), None)
        first_silent.append(k)
    chooser = {}
    pool_set = set(pool_local)
    for i in order:                                       # which item pooled each candidate (first in the round robin)
        for j in top[i]:
            if j in pool_set and j not in chooser and silent[(i, j)]:
                chooser[j] = i
    cos = [float(sims[chooser[j]] @ sims[j]) for j in pool_local if j in chooser]
    n_meas = len(silent)
    return dict(task=task["name"], model=C.MODEL, pool=[tr[j] for j in pool_local], pool_size=len(pool_local),
                rule=dict(candidates="other training prompts of the same dataset (own group excluded)",
                          ranking="cosine of mean-pooled final-layer residual (last-layer resid SAE input)",
                          silent="metric(good_i, bad_i | candidate) <= %g" % C.SILENT_MAX,
                          top_k=C.TOP_K_CANDIDATES, pool=C.POOL_SIZE, answer_mode=mode),
                diag=dict(n_items=len(tr), n_measured=n_meas, silent_share=sum(silent.values()) / max(1, n_meas),
                          items_without_silent=sum(1 for k in first_silent if k is None),
                          first_silent_rank_median=(float(np.median([k for k in first_silent if k is not None]))
                                                    if any(k is not None for k in first_silent) else None),
                          pool_cosine_median=float(np.median(cos)) if cos else None,
                          pool_metric_median=float(np.median([metric[(tr[chooser[j]], tr[j])] for j in pool_local
                                                              if j in chooser])) if pool_local else None),
                generated=time.strftime("%Y-%m-%d %H:%M"))


def main():
    import engine as E
    names = generated_tasks()
    todo = [n for n in names if os.environ.get("FORCE") == "1" or not C.contrast_path(n).exists()]
    print("MODEL %s | generated-contrast tasks: %s | to build: %s" % (C.MODEL, names, todo), flush=True)
    if not todo:
        return
    eng = E.Engine(C.MODEL)
    for n in todo:
        t0 = time.time()
        out = build_contrast(eng, C.load_task(n))
        C.contrast_path(n).write_text(json.dumps(out, indent=1) + "\n")
        d = out["diag"]
        print("%-44s pool %d | silent share %.2f | items without a silent candidate %d | first silent rank median %s "
              "| pool cosine median %s (%.0fs)" % (n, out["pool_size"], d["silent_share"], d["items_without_silent"],
                                                   d["first_silent_rank_median"], d["pool_cosine_median"],
                                                   time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
