"""D2.5 (pin variant) + D4.6 (true generalisation) — one GPU pass.

D2.5  Which pin convention better PREDICTS held-out phi-cf: collapsed
      (position-independent posctx means) or position-specific pins?
      For each seed and K, score the SAME driver head under both pin
      conventions and correlate each against the head's cf.
D4.6  TRUE generalisation: drivers were discovered on the probe set and
      have so far been imposed on held-out probes from the SAME store
      slice. Here they are imposed on FRESH contexts drawn from the
      corpus that no stage of discovery or evaluation has seen: the
      seed's MID-band contexts (mid_ctx) and a random corpus draw.

  PYTHONPATH=src python experiments/021-d25-d46/runner.py
"""
import gzip
import json
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.context import mid_ctx

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
D22 = HERE.parent / "019-roles-drivers"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
KS = (16, 64, 256)
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053),
         (17, 38268), (20, 35678), (26, 17432), (35, 6599)]

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
config.discovery.eval_batch_size = EVAL_BS

ALPHA64 = {}
import glob
for p in glob.glob(str(D1 / "rows_s*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if r.get("arm") == "C" and r.get("K") == 64 and r.get("alpha_star"):
            ALPHA64[r["seed"]] = float(r["alpha_star"])


class InjectPatcher:
    def __init__(self, targets, seed_site, seed_idx):
        self.targets, self.seed_site, self.seed_idx = targets, seed_site, seed_idx
        self.argmax_chunk = None
        self.seed_capture = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            ta, ti = bank.encode(x, kind, layer_idx)
            s_dense = target_latent_activations(ta, ti, self.seed_idx)
            pa_c = self.argmax_chunk
            if pa_c is not None:
                B = min(s_dense.shape[0], pa_c.shape[0])
                pa_c = pa_c[:B].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                rows = torch.arange(B, device=s_dense.device)
                self.seed_capture = float(s_dense[rows, pa_c].mean())
            else:
                self.seed_capture = float(s_dense.mean())
            return x
        t = self.targets.get((layer_idx, kind))
        if not t:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        c_new = dense.clone()
        idxs = torch.tensor(sorted(t), device=dense.device, dtype=torch.long)
        vals = torch.tensor([t[int(i)] for i in idxs], device=dense.device,
                            dtype=dense.dtype)
        c_new[..., idxs] = vals
        out = bank.decode(c_new - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["task"], r.get("K"), r.get("variant")))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den_free = a_pos_ev - a_e0
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)
    _, pins_p = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=True)

    # rankings
    rank = []
    rp = D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
    if rp.exists():
        with gzip.open(rp, "rt", encoding="utf-8") as gz:
            for i, line in enumerate(gz):
                if i >= 4096:
                    break
                s_, l_, kd_, idx_, role_, rr_ = json.loads(line)
                rank.append(((l_, kd_), int(idx_)))
    dwp = D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl))
    dw = torch.load(dwp, map_location="cpu", weights_only=False)["direct"]
    tri = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(2048, w.numel()))
        tri += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
    tri.sort(key=lambda x: -x[0])
    rank_c = [(s, i) for _, s, i in tri]

    print("\n[%s] L%d %s | a_pos %.3f" % (seed_key, layer, kind, a_pos_ev),
          flush=True)

    def keep_of(mem):
        k = {}
        for s, i in mem:
            k.setdefault(s, set()).add(i)
        return k

    def pin0(mem, pins):
        if abs(den_free) < 1e-9:
            return None
        a = float(circuit_only_activation(
            inference, bank, keep_of(mem), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins))
        return round((a - a_e0) / den_free, 4)

    def act_under(patcher, tokens, argmax):
        if patcher is None:
            return float(measure_seed_activation(
                inference, bank, tokens, layer, kind, sl, argmax,
                batch_size=EVAL_BS))
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_capture = None
                patcher.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                tot += float(patcher.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    # ---------------- D2.5: pin conventions ----------------
    for K in KS:
        if (seed_key, "D2.5", K, None) in done:
            continue
        head = rank[:K]
        if not head:
            continue
        row = {"seed": seed_key, "layer": layer, "kind": kind, "task": "D2.5",
               "K": K, "variant": None, "n": len(head),
               "pin_collapsed": pin0(head, pins_c),
               "pin_position": pin0(head, pins_p),
               "free0": pin0(head, None)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  D2.5 K=%-4d pin_c=%-8s pin_p=%-8s free0=%s"
              % (K, row["pin_collapsed"], row["pin_position"], row["free0"]),
              flush=True)

    # ---------------- D4.6: fresh-context generalisation ----------------
    # AMPC K=64 driver at alpha*, imposed on (a) held-out store negatives
    # [reference], (b) MID-band contexts, (c) a random corpus draw.
    alpha = ALPHA64.get(seed_key, 2.0)
    targets, nw = {}, 0
    for site, idx in rank_c:
        if nw >= 64:
            break
        v = float(pins_c[site][idx]) if site in pins_c else 0.0
        if v > 0:
            targets.setdefault(site, {})[idx] = alpha * v
            nw += 1
    if not targets:
        continue
    fresh = {}
    try:
        mids = [int(x) for x in mid_ctx.ctx_seq_idx[sc_idx, sl].tolist() if int(x) > 0]
        if mids:
            fresh["mid"] = probe_builder._load_all_ids(mids[:EVAL_BS],
                                                      max_length=64)
    except Exception as exc:
        print("  mid_ctx unavailable: %s" % str(exc)[:60], flush=True)
    try:
        rng = random.Random(101)
        tot_seqs = len(loader)
        if tot_seqs > 100:
            ids = [rng.randrange(1, tot_seqs) for _ in range(EVAL_BS)]
            fresh["corpus"] = probe_builder._load_all_ids(ids, max_length=64)
    except Exception as exc:
        print("  corpus draw unavailable: %s" % str(exc)[:60], flush=True)

    for label, toks in [("store_heldout", nt_ev)] + list(fresh.items()):
        if (seed_key, "D4.6", 64, label) in done or toks is None:
            continue
        toks = toks.to(nt_ev.device)[:EVAL_BS]
        am = pa_ev[:toks.shape[0]]
        a_base = act_under(None, toks, am)
        a_int = act_under(InjectPatcher(targets, (layer, kind), sl), toks, am)
        den = a_pos_ev - a_base
        row = {"seed": seed_key, "layer": layer, "kind": kind, "task": "D4.6",
               "K": 64, "variant": label, "n_seq": int(toks.shape[0]),
               "alpha": alpha, "a_base": round(a_base, 4),
               "a_int": round(a_int, 4), "a_pos": round(a_pos_ev, 4),
               "cf": round((a_int - a_base) / den, 4) if abs(den) > 1e-9 else None}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  D4.6 %-14s base=%7.3f int=%7.3f cf=%s"
              % (label, a_base, a_int, row["cf"]), flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
