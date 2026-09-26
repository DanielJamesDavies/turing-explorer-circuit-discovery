"""AMPLITUDE-AWARE EVALUATION PASS over the production circuits.

The pipeline stored every circuit's members + fitted amplitudes but
evaluated only the BARE member set (legacy anchors). This pass scores
each stored circuit the way the tri-amp method is meant to be scored
(029-panel / 045 frames), on the HELD-OUT probe slice the fit never saw
(the engine's split is deterministic: first 48 of 64 train, last 16 held
out). No fitting — evaluation only, ~5-8 s per circuit on an RTX.

Per circuit:
  a_pos      seed activation, natural stream (held-out)
  e0         empty circuit, zero-fill of every upstream site
  ampF0      circuit-only zero-fill, members at their fitted gains   -> F0_amp = (ampF0-e0)/(a_pos-e0)
  F0_a1      same with gains = 1                                      -> F0_a1
  ampFMd     circuit-only with non-members at their posctx MEANS     -> FMd_amp
  cf_amp     members SET to gain x posctx pin on the seed's NEGATIVE contexts,
             read at the seed's own would-be-firing position         -> (cf_amp-a_base)/(a_pos-a_base)
  cf_bare / sup  the pipeline's bare evals recomputed with anchor_mode="negctx_preact"
             (the position-collapse fix), plus cf_bounded
  nulls      ampF0 with the gains PERMUTED among members; F0 of a random
             site-matched member set of the same size at gains = 1

  SEED_SHARD=0/16 PYTHONPATH=src python experiments/049-circuit-graph/amp_eval_pass.py
    -> results_full/amp_eval.shard<i>.jsonl (resumable). SMOKE=1 -> first 6 circuits.
  DATA=<dir of discovered_circuits.shard*.pt>  OUT=<results dir>
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import circuit_only_activation, measure_seed_activation, upstream_sites
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import component_idx as comp_of
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from store.circuits import Circuit, CircuitNode

HERE = Path(__file__).parent
DATA = Path(os.environ.get("DATA", str(HERE / "data_full")))
OUT = Path(os.environ.get("OUT", str(HERE / "results_full")))
SMOKE = os.environ.get("SMOKE") == "1"
SHARD_I, SHARD_K = (int(x) for x in os.environ.get("SEED_SHARD", "0/16").split("/"))
N_SEQ, N_TRAIN, EVAL_BS = 64, 48, 16
random.seed(SHARD_I); np.random.seed(SHARD_I)

torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
KINDS = list(bank.kinds); NK = len(KINDS); D = bank.d_sae
avg_acts = torch.zeros((bank.n_layer * NK, D), device=bank.device)
disc = config.discovery
disc.probe_sequence_count = N_SEQ; disc.eval_sequence_count = N_SEQ; disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4; disc.position_aware = False
M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)

ONLY = os.environ.get("ONLY")          # csv with an `skey` column: score just these seeds, across all shards
if ONLY:
    want = set(__import__("pandas").read_csv(ONLY)["skey"])
    items = []
    for sp in sorted(glob_stores := sorted(__import__("glob").glob(str(DATA / "discovered_circuits.shard*.pt")))):
        for c in torch.load(sp, weights_only=False, map_location="cpu").values():
            sd = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed").metadata["feature_id"]
            if "%d.%s.%d" % (sd.layer, sd.kind, sd.index) in want:
                items.append(c)
    store_path = Path("subset:%d-seeds" % len(items))
    out_path = OUT / "amp_eval_subset.jsonl"
else:
    store_path = DATA / ("discovered_circuits.shard%d.pt" % SHARD_I)
    cs = torch.load(store_path, weights_only=False, map_location="cpu")
    items = list(cs.values())
    out_path = OUT / ("amp_eval%s.shard%d.jsonl" % ("_smoke" if SMOKE else "", SHARD_I))
if SMOKE:
    items = items[:6]
done = set()
if out_path.exists():
    for ln in open(out_path):
        try:
            r = json.loads(ln); done.add(r["seed"])
        except Exception:
            pass
fh = open(out_path, "a")
print("shard %d/%d: %d circuits in %s | %d already done -> %s" % (SHARD_I, SHARD_K, len(items), store_path.name, len(done), out_path.name), flush=True)

# live pool per site for the random-set null (from this shard's own members)
live_pool = defaultdict(set)
for c in items:
    for n in c.nodes.values():
        if n.metadata.get("role") != "seed":
            f = n.metadata["feature_id"]; live_pool[(f.layer, f.kind)].add(f.index)
live_pool = {s: np.array(sorted(v)) for s, v in live_pool.items()}


class AmpCircuitPatcher:
    """Members at alpha x live value, non-members at floor (zero when floors is None); seed tapped."""

    def __init__(self, alphas, floors, seed_site, w_seed, b_seed):
        self.alphas, self.floors = alphas, floors or {}
        self.seed_site = seed_site; self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype); b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        al = self.alphas.get((layer_idx, kind))
        if al is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        fl = self.floors.get((layer_idx, kind))
        code = fl.to(device=dense.device, dtype=dense.dtype).expand_as(dense).clone() if fl is not None else torch.zeros_like(dense)
        if al:
            idx = torch.tensor(sorted(al), device=dense.device, dtype=torch.long)
            av = torch.tensor([al[int(i)] for i in idx], device=dense.device, dtype=dense.dtype)
            code[..., idx] = dense[..., idx] * av
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


class AmpInjectPatcher:
    """Members SET to alpha_i x pin_i in the otherwise-live stream; seed tapped."""

    def __init__(self, inject, seed_site, w_seed, b_seed):
        self.inject = inject; self.seed_site = seed_site; self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype); b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        inj = self.inject.get((layer_idx, kind))
        if not inj:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        code = dense.clone()
        idx = torch.tensor(sorted(inj), device=dense.device, dtype=torch.long)
        code[..., idx] = torch.tensor([inj[int(i)] for i in idx], device=dense.device, dtype=dense.dtype)
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def read(patcher, tokens, anchors):
    tot, n = 0.0, 0
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]; patcher.seed_pre = None
                inference.forward(tk, patcher=patcher, grad_enabled=False, return_activations=False, tokenize_final=False)
                pre = patcher.seed_pre; B = pre.shape[0]; rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum()); n += B
    finally:
        inference.enable_compile()
    return tot / max(n, 1)


def frac(v, lo, hi):
    return (v - lo) / (hi - lo) if abs(hi - lo) > 1e-9 else None


t0 = time.time(); n_done = 0; n_fail = 0
for c in items:
    md = c.metadata
    seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
    sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    key = "%d.%s.%d" % (layer, kind, sl)
    if key in done:
        continue
    ts = time.time()
    try:
        alphas = defaultdict(dict)
        for n in c.nodes.values():
            if n is seed:
                continue
            f = n.metadata["feature_id"]; alphas[(f.layer, f.kind)][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
        alphas = dict(alphas); n_mem = sum(len(d) for d in alphas.values())
        pd_ = M0.build_probe_dataset(comp_of(layer, KINDS.index(kind), NK), sl)
        if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
            fh.write(json.dumps(dict(seed=key, skip="thin_probes", n=n_mem)) + "\n"); fh.flush(); continue
        pt, pa, nt = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ], pd_.neg_tokens[:N_SEQ]
        pt_tr, pa_tr = pt[:N_TRAIN], pa[:N_TRAIN]; pt_ho, pa_ho = pt[N_TRAIN:], pa[N_TRAIN:]
        nt_ho = nt[N_TRAIN:] if int(nt.shape[0]) >= N_SEQ else nt[-16:]
        sae = bank.saes[kind][layer]; w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
        UP = sorted(upstream_sites(bank, layer, kind)); site = (layer, kind)
        a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer, kind, sl, pa_ho, batch_size=EVAL_BS))
        a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind, sl, pa_tr, batch_size=EVAL_BS))
        e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho, layer, kind, sl, pos_argmax=pa_ho, batch_size=EVAL_BS))
        means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr, set(UP), pa_tr, pin_position_specific=False)
        eMd_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho, layer, kind, sl, pos_argmax=pa_ho, site_means=means_tr, batch_size=EVAL_BS))
        ampF0 = read(AmpCircuitPatcher(alphas, None, site, w_seed, b_seed), pt_ho, pa_ho)
        ampF0_tr = read(AmpCircuitPatcher(alphas, None, site, w_seed, b_seed), pt_tr, pa_tr)
        a1 = {s: {i: 1.0 for i in d} for s, d in alphas.items()}
        F0_a1 = read(AmpCircuitPatcher(a1, None, site, w_seed, b_seed), pt_ho, pa_ho)
        ampFMd = read(AmpCircuitPatcher(alphas, means_tr, site, w_seed, b_seed), pt_ho, pa_ho)
        # negctx: baseline at the seed's own would-be-firing positions, then amp-aware injection
        p0 = AmpInjectPatcher({}, site, w_seed, b_seed); chunks = []
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(nt_ho.shape[0]), EVAL_BS):
                    p0.seed_pre = None
                    inference.forward(nt_ho[s0:s0 + EVAL_BS], patcher=p0, grad_enabled=False, return_activations=False, tokenize_final=False)
                    chunks.append(p0.seed_pre.detach())
        finally:
            inference.enable_compile()
        neg_pre = torch.cat(chunks, 0); na_ho = neg_pre.argmax(dim=1).cpu()
        a_base = float(torch.relu(neg_pre[torch.arange(neg_pre.shape[0], device=neg_pre.device), na_ho.to(neg_pre.device)]).mean())
        inject = {s: {i: float(a * float(pins_tr[s][i])) for i, a in d.items()} for s, d in alphas.items() if s in pins_tr}
        cf_amp = read(AmpInjectPatcher(inject, site, w_seed, b_seed), nt_ho, na_ho)
        # nulls
        perm = {}
        for s, d in alphas.items():
            vals = list(d.values()); random.shuffle(vals); perm[s] = dict(zip(sorted(d), vals))
        ampF0_perm = read(AmpCircuitPatcher(perm, None, site, w_seed, b_seed), pt_ho, pa_ho)
        rnd = {}
        for s, d in alphas.items():
            pool = live_pool.get(s)
            if pool is None or len(pool) == 0:
                continue
            n_ = min(len(d), len(pool)); rnd[s] = {int(i): 1.0 for i in np.random.choice(pool, n_, replace=False)}
        F0_rand = read(AmpCircuitPatcher(rnd, None, site, w_seed, b_seed), pt_ho, pa_ho)
        # bare evals with the anchor fix
        circ = Circuit(name=key)
        for s, d in alphas.items():
            for i in d:
                circ.add_node(CircuitNode(metadata={"layer_idx": s[0], "kind": s[1], "latent_idx": int(i), "role": "ablation_support"}))
        try:
            cf_b, sup_b, det = evaluate_counterfactual_faithfulness(inference, bank, avg_acts, circ, neg_tokens=nt_ho, pos_tokens=pt_ho,
                                                                    seed_layer=layer, seed_kind=kind, seed_latent_idx=sl, pos_argmax=pa_ho,
                                                                    circuit_layers={l for (l, _) in alphas}, anchor_mode="negctx_preact", return_details=True)
            cf_b, sup_b, cf_bnd = float(cf_b), float(sup_b), float(det.get("cf_bounded", float("nan")))
        except Exception as e:
            cf_b = sup_b = cf_bnd = None
        row = dict(seed=key, layer=layer, kind=kind, n=n_mem, a_pos_ho=a_pos_ho, a_pos_tr=a_pos_tr, e0_ho=e0_ho, eMd_ho=eMd_ho, a_base=a_base,
                   ampF0_ho=ampF0, ampF0_tr=ampF0_tr, F0a1_ho=F0_a1, ampFMd_ho=ampFMd, cf_amp_raw=cf_amp, ampF0_perm=ampF0_perm, F0_rand=F0_rand,
                   F0_amp=frac(ampF0, e0_ho, a_pos_ho), F0_amp_tr=frac(ampF0_tr, e0_ho, a_pos_tr), F0_a1=frac(F0_a1, e0_ho, a_pos_ho),
                   FMd_amp=frac(ampFMd, eMd_ho, a_pos_ho), cf_amp=frac(cf_amp, a_base, a_pos_ho),
                   F0_perm=frac(ampF0_perm, e0_ho, a_pos_ho), F0_rand_frac=frac(F0_rand, e0_ho, a_pos_ho),
                   cf_bare_anch=cf_b, sup_anch=sup_b, cf_bounded=cf_bnd, vacuous=bool(abs(a_pos_ho - e0_ho) < 0.05 * abs(e0_ho) if e0_ho else False),
                   amp_median=(md.get("amp_stats") or {}).get("median"), cf_bare_legacy=(md.get("evals") or {}).get("counterfactual_faithfulness"),
                   secs=round(time.time() - ts, 1))
        fh.write(json.dumps(row) + "\n"); fh.flush(); n_done += 1
        if n_done % 10 == 1 or SMOKE:
            print("[%4d] %-16s n=%4d | a_pos %.2f e0 %.2f | F0_amp %s | F0_a1 %s | FMd_amp %s | cf_amp %s | perm %s rand %s | cf_bare(anch) %s sup %s | %.1fs (%.0f s/seed avg)"
                  % (n_done, key, n_mem, a_pos_ho, e0_ho, *["%.2f" % row[k] if row[k] is not None else "-" for k in ("F0_amp", "F0_a1", "FMd_amp", "cf_amp", "F0_perm", "F0_rand_frac")],
                     "%.2f" % cf_b if cf_b is not None else "-", "%.2f" % sup_b if sup_b is not None else "-", row["secs"], (time.time() - t0) / max(n_done, 1)), flush=True)
    except Exception as e:
        n_fail += 1
        fh.write(json.dumps(dict(seed=key, error="%s: %s" % (type(e).__name__, str(e)[:200]))) + "\n"); fh.flush()
        print("[FAIL] %s: %s %s" % (key, type(e).__name__, str(e)[:160]), flush=True)
        if n_fail >= 5 and n_done == 0:
            sys.exit("aborting: 5 failures before any success")
print("DONE shard %d: %d scored, %d failed, %.0fs" % (SHARD_I, n_done, n_fail, time.time() - t0))
