"""TASK-METRIC SCORING of the fitted circuits under circuit-only execution
(zero-fill at all 36 sites, members at their fitted amplitudes) on the
HELD-OUT prompts (the runner's last 25%):

  greater-than: P(next digit > tens) mass under full / empty / circuit /
                circuit at alpha=1 / matched amp-null
  agreement   : logit(correct verb) - logit(wrong verb), same five frames
plus the target-token EF for reference. This is the "does the circuit
reproduce the task DECISION, not just the target's log-prob" check, and
the alpha=1 column is the amplitudes-matter check on these circuits.

  TASK=gt|agree PYTHONPATH=src python experiments/047-known-circuits/score_task_circuits.py
"""
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import CircuitOnlyPatcher
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

HERE = Path(__file__).parent
TASK = os.environ.get("TASK", "gt")
RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
EVAL_BS = 16

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
_apply_sweep_config(max_per_site=24)
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in bank.kinds)

data = torch.load(HERE / ("%s_clusters.pt" % TASK), weights_only=False)
n = len(data["windows"])
n_tr = max(8, int(n * 0.75))
idx_ho = list(range(n_tr, n))
pt = torch.tensor([[max(t, 0) for t in data["windows"][i][:63]] for i in idx_ho],
                  dtype=torch.long, device=device)
pa = torch.tensor([int(data["anchors"][i]) for i in idx_ho], dtype=torch.long)
tgt = torch.tensor([data["targets"][i] for i in idx_ho], dtype=torch.long)

keep, scales = {}, {}
for line in open(HERE / os.environ.get("MEMBERS", "%s_members.jsonl" % TASK)):
    r = json.loads(line)
    for site, d in r["alphas"].items():
        lyr, knd = site.split("/")
        st = (int(lyr), knd)
        keep[st] = set(int(i) for i in d)
        v = torch.ones(bank.d_sae)
        for i, a in d.items():
            v[int(i)] = float(a)
        scales[st] = v
n_mem = sum(len(v) for v in keep.values())

rng = random.Random(7)
keep_n, scales_n = {}, {}
for st, ids in keep.items():
    rid = rng.sample(range(bank.d_sae), len(ids))
    keep_n[st] = set(rid)
    v = torch.ones(bank.d_sae)
    amps = [float(scales[st][i]) for i in ids]
    rng.shuffle(amps)
    for i, a in zip(rid, amps):
        v[i] = a
    scales_n[st] = v


def logits_at_anchor(keep_, scales_):
    out_all = []
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s in range(0, int(pt.shape[0]), EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                p = (CircuitOnlyPatcher(bank=bank, keep_indices=keep_,
                                        in_scope=set(ALL_SITES), seed_layer=-1,
                                        seed_kind="", seed_latent_idx=0,
                                        site_means=None, keep_scales=scales_)
                     if keep_ is not None else None)
                out = inference.forward(tk, patcher=p, all_logits=True,
                                        grad_enabled=False,
                                        return_activations=False,
                                        tokenize_final=False)
                lg = out[1] if isinstance(out, (tuple, list)) else out
                b = torch.arange(tk.shape[0], device=device)
                out_all.append(lg[b, pa[s:s + EVAL_BS].to(device)].float().cpu())
    finally:
        inference.enable_compile()
    return torch.cat(out_all, 0)


def task_metric(lg):
    if TASK == "gt":
        p = torch.softmax(lg, -1)
        dig = torch.tensor(data["digit_ids"])
        pd = p[:, dig]
        vals = []
        for r, i in enumerate(idx_ho):
            tens = int(data["tens"][i])
            vals.append(float(pd[r, tens + 1:].sum() - pd[r, :tens].sum()))
        return sum(vals) / len(vals), "P(>tens)-P(<tens)"
    else:
        vals = []
        for r, i in enumerate(idx_ho):
            vals.append(float(lg[r, data["targets"][i]] - lg[r, data["wrong_tokens"][i]]))
        return sum(vals) / len(vals), "logit(correct)-logit(wrong)"


def target_logp(lg):
    lp = torch.log_softmax(lg, -1)
    return float(lp[torch.arange(lg.shape[0]), tgt].mean())


frames = [("full model", None, None), ("empty (all zero-filled)", {}, None),
          ("circuit + amplitudes", keep, scales), ("circuit at alpha=1", keep, None),
          ("amp-null (random ids, same amps)", keep_n, scales_n)]
res = {}
for name, k, s in frames:
    lg = logits_at_anchor(k, s)
    res[name] = (task_metric(lg), target_logp(lg))
full_lp, empty_lp = res["full model"][1], res["empty (all zero-filled)"][1]
mname = res["full model"][0][1]
print("task %s | %d held-out prompts | %d members" % (TASK, len(idx_ho), n_mem))
print("  %-34s %22s %12s %8s" % ("frame", mname, "logp(target)", "EF"))
for name, ((m, _), lp) in res.items():
    ef = (lp - empty_lp) / (full_lp - empty_lp)
    print("  %-34s %22.3f %12.3f %8.3f" % (name, m, lp, ef))
