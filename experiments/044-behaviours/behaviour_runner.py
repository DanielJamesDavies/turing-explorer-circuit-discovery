"""TRI-AMP CIRCUITS FOR UNSUPERVISED BEHAVIOURS (SFC-style data).

For each chosen behaviour cluster (from cluster_behaviours.py, quanta
clustering after Michaud et al. / Marks et al.):
  metric  m = log P(cluster's actual next token | context)
  fit     objective="logit" (the G11 endpoint machinery), sites = ALL
          36 (layer, kind) sites, mask_floor_source="zero",
          free_amplitude=True  -> a tri-amp BEHAVIOUR circuit
  score   EF = (m(circuit) - m(empty)) / (m(full) - m(empty)) on
          HELD-OUT cluster contexts, mean-fill scope = all sites
          (the G11 scope lesson: score what the mask controls)
  nulls   2 random same-shape member sets scored identically

  CLUSTERS=3,5,19 PYTHONPATH=src python .../behaviour_runner.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from circuit.instrument.learned_mask import run_learned_mask
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import CircuitOnlyPatcher
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
CLUSTERS = os.environ.get("CLUSTERS", "auto3")
LAM = float(os.environ.get("LAM", 3e-3))
N_NULL = int(os.environ.get("N_NULL", 2))
EVAL_BS = 16

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
n_kinds = len(bank.kinds)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_batch_size = 4
cfg = disc.learned_mask
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in bank.kinds)

DATA = os.environ.get("DATA", "behaviour_clusters.pt")
OUT = os.environ.get("OUT", "behaviour")
data = torch.load(HERE / DATA, weights_only=False)
ANCHOR = data["anchor"]
assign = data["assign"]
fh = (HERE / ("%s_rows.jsonl" % OUT)).open("a")
mh = (HERE / ("%s_members.jsonl" % OUT)).open("a")


def main():
    if isinstance(CLUSTERS, str) and CLUSTERS.startswith("auto"):
        k_pick = int(CLUSTERS[4:])
        coh, sizes = data["coherence"], data["sizes"]
        order = torch.argsort(coh, descending=True).tolist()
        picks = [k for k in order if int(sizes[k]) >= 80][:k_pick]
        print("auto-picked clusters:", picks, flush=True)
    else:
        picks = [int(x) for x in CLUSTERS.split(",")]
    for ck in picks:
        idx = (assign == ck).nonzero(as_tuple=True)[0].tolist()
        wins = [data["windows"][i] for i in idx]
        tgts = [data["targets"][i] for i in idx]
        n = len(wins)
        n_tr = max(8, int(n * 0.75))
        pt = torch.tensor([[max(t, 0) for t in w[:ANCHOR + 1]]
                           for w in wins], dtype=torch.long, device=device)
        pa = torch.full((n,), ANCHOR, dtype=torch.long)
        tgt = torch.tensor(tgts, dtype=torch.long)
        other = [data["windows"][i] for i in range(len(assign))
                 if assign[i] != ck][:n]
        nt = torch.tensor([[max(t, 0) for t in w[:ANCHOR + 1]]
                           for w in other], dtype=torch.long, device=device)
        pt_tr, pa_tr, tgt_tr = pt[:n_tr], pa[:n_tr], tgt[:n_tr]
        pt_ho, pa_ho, tgt_ho = pt[n_tr:], pa[n_tr:], tgt[n_tr:]
        print("\n== cluster %d: %d contexts (%d train / %d held-out)"
              % (ck, n, n_tr, n - n_tr), flush=True)
        # scoring frame = the fit's own: zero-fill, amplitudes applied
        def metric(keep, tokens, anchors, targets, scales=None):
            tot, m = 0.0, int(tokens.shape[0])
            inference.disable_compile()
            try:
                with torch.no_grad():
                    for s in range(0, m, EVAL_BS):
                        tk = tokens[s:s + EVAL_BS]
                        p = CircuitOnlyPatcher(
                            bank=bank, keep_indices=keep,
                            in_scope=set(ALL_SITES), seed_layer=-1,
                            seed_kind="", seed_latent_idx=0,
                            site_means=None,
                            keep_scales=scales) if keep is not None else None
                        out = inference.forward(
                            tk, patcher=p, all_logits=True,
                            grad_enabled=False, return_activations=False,
                            tokenize_final=False)
                        lg = out[1] if isinstance(out, (tuple, list)) else out
                        b = torch.arange(tk.shape[0], device=device)
                        lp = torch.log_softmax(
                            lg[b, anchors[s:s + EVAL_BS].to(device)].float(),
                            dim=-1)
                        tot += float(
                            lp[b, targets[s:s + EVAL_BS].to(device)].sum())
            finally:
                inference.enable_compile()
            return tot / max(m, 1)

        m_full = metric(None, pt_ho, pa_ho, tgt_ho)
        m_empty = metric({}, pt_ho, pa_ho, tgt_ho)
        den = m_full - m_empty
        print("  logp full %.3f | empty %.3f | den %.3f"
              % (m_full, m_empty, den), flush=True)

        t0 = time.time()
        scores, prov = run_learned_mask(
            inference, bank, objective="logit", sites=ALL_SITES,
            seed_layer=bank.n_layer - 1, seed_kind="resid",
            seed_latent_idx=0, pos_tokens=pt_tr, pos_argmax=pa_tr,
            neg_tokens=nt[:n_tr], target_tokens=tgt_tr,
            mask_floor_source="zero", free_amplitude=True,
            steps=cfg.steps, lr=cfg.lr, l1_lambda=LAM,
            keep_threshold=cfg.keep_threshold,
            batch_size=disc.probe_batch_size,
            holdout_frac=cfg.holdout_frac, log_every=0,
            deep_site_threshold=cfg.deep_site_threshold,
            deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
            weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
            binarize=cfg.binarize, theta_init=cfg.theta_init,
            lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
            warmup_frac=cfg.warmup_frac)
        ak = prov.get("amp_kept") or {}
        alphas = {}
        for k, d in ak.items():
            lyr, knd = k.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(v)
                                       for i, v in d.items()}
        keep = {s: set(d) for s, d in alphas.items() if d}
        scales = {}
        for st, d in alphas.items():
            if not d:
                continue
            v = torch.ones(bank.d_sae)
            for i, a in d.items():
                v[i] = a
            scales[st] = v
        nmem = sum(len(v) for v in keep.values())
        m_circ = metric(keep, pt_ho, pa_ho, tgt_ho, scales)
        ef = (m_circ - m_empty) / den if abs(den) > 1e-9 else None
        # train EF: the overfit meter (large train-vs-holdout gap = memorised)
        m_full_tr = metric(None, pt_tr, pa_tr, tgt_tr)
        m_empty_tr = metric({}, pt_tr, pa_tr, tgt_tr)
        m_circ_tr = metric(keep, pt_tr, pa_tr, tgt_tr, scales)
        ef_tr = ((m_circ_tr - m_empty_tr) / (m_full_tr - m_empty_tr)
                 if abs(m_full_tr - m_empty_tr) > 1e-9 else None)
        print("  behav%d n=%d EF_ho=%.3f EF_tr=%.3f (logp %.3f) %.0fs"
              % (ck, nmem, ef, ef_tr, m_circ, time.time() - t0), flush=True)

        rng = random.Random(9)
        nulls = []
        for j in range(N_NULL):
            # matched null: random ids, same counts, own amps permuted on
            na, ns = {}, {}
            for s, v in keep.items():
                ids = rng.sample(range(bank.d_sae), len(v))
                na[s] = set(ids)
                vv = torch.ones(bank.d_sae)
                amps = [alphas[s][i] for i in v]
                rng.shuffle(amps)
                for i, a in zip(ids, amps):
                    vv[i] = a
                ns[s] = vv
            mn = metric(na, pt_ho, pa_ho, tgt_ho, ns)
            nulls.append((mn - m_empty) / den)
            print("    null%d EF=%.3f" % (j, nulls[-1]), flush=True)

        fh.write(json.dumps({
            "cluster": ck, "n_ctx": n, "n_members": nmem,
            "EF": round(ef, 4), "EF_tr": round(ef_tr, 4), "m_full": round(m_full, 4),
            "m_empty": round(m_empty, 4), "m_circ": round(m_circ, 4),
            "nulls": [round(x, 4) for x in nulls]}) + "\n")
        fh.flush()
        mh.write(json.dumps({
            "cluster": ck,
            "alphas": {"%d/%s" % s: {str(i): round(a, 4)
                                     for i, a in d.items()}
                       for s, d in alphas.items()}}) + "\n")
        mh.flush()


if __name__ == "__main__":
    main()
