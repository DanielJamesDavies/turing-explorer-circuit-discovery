"""8xH100 PRODUCTION DRIVER: tri-amp (and neg-amp variant) circuits for
EVERY candidate seed, sharded per GPU. The per-seed protocol is the
029-panel runner's, verbatim (triple floor, 48/16 train/held-out split,
ampF0/ampFMd/cf_amp/cf_bare/sup, amplitude-fitted nulls); this driver
adds only the scale machinery:

  SHARDING   SEED_SHARD=i/k -> this process takes candidates where
             enumeration_index % k == i; rows go to rows.shard<i>.jsonl
             (concatenate after the run; every row carries
             comp_idx/latent/arm). One shard per GPU
             (CUDA_VISIBLE_DEVICES pins it; see h100_launch.sh).
  ARMS       ARMS=triamp400[,gate400,sgnamp400,negsup400]
             triamp400  triple floor + free amplitudes (production)
             gate400    triple floor, gates only
             sgnamp400  neg-amp SIGNED variant (alpha may go negative)
             negsup400  neg-amp SUPPRESS variant (NEG_W penalty on the
                        seed's negctx read; know_runner default 0.5)
  NULLS      amplitude-fitted same-size random sets are a FIT each, so
             they are thinned: N_NULL draws only on seeds where
             (comp_idx*131071 + latent) % NULL_EVERY == 0
             (deterministic across shards/resumes).
  SKIP ROWS  every seed that is not fitted writes an arm="skip" row
             with the reason, so shard completeness is checkable:
             every shard seed ends with all arm rows or a skip row.
             The end-of-run report counts both (harness trap #1/#4).
  ABORT      3 consecutive per-seed failures abort the shard nonzero
             (harness trap #2: an empty rows file must never say done).

  SMOKE=1 SEED_SHARD=0/2048 ARMS=... -> 2 seeds, nulls forced, 1 draw.

  RUN_ROOT=<artifacts dir> SEED_SHARD=0/8 ARMS=triamp400 \
    PYTHONPATH=src python experiments/045-h100-production/driver.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path(os.environ.get(
    "RUN_ROOT",
    "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
    "20260531-152059-37117a33/20260531-152059-37117a33"))
HERE = Path(__file__).parent
SMOKE = os.environ.get("SMOKE") == "1"
SHARD_I, SHARD_K = (int(x) for x in
                    os.environ.get("SEED_SHARD", "0/1").split("/"))
ARMS = [a for a in os.environ.get("ARMS", "triamp400").split(",") if a]
LAM = float(os.environ.get("LAM", 1e-3))
NEG_W = float(os.environ.get("NEG_W", 0.5))
N_NULL = int(os.environ.get("N_NULL", 4))
NULL_EVERY = int(os.environ.get("NULL_EVERY", 20))
N_SEQ, N_TRAIN, EVAL_BS, D_SAE = 64, 48, 16, 40960

# (tag, free_amplitude, signed, negsup) — every arm is 400 production
# steps at LAM; new arms MUST be added here AND nowhere else (the
# completeness accounting below derives from this registry).
ARM_SPECS = {
    "triamp400": (True, False, False),
    "gate400": (False, False, False),
    "sgnamp400": (True, True, False),
    "negsup400": (True, False, True),
}
for a in ARMS:
    if a not in ARM_SPECS:
        sys.exit("unknown arm %r (registry: %s)" % (a, sorted(ARM_SPECS)))

torch.set_float32_matmul_precision("high")
load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
SHARD = [(int(c["comp_idx"]), int(c["latent_idx"]))
         for i, c in enumerate(_cand) if i % SHARD_K == SHARD_I]
if SMOKE:
    SHARD = SHARD[:2]
print("shard %d/%d: %d of %d candidates | arms %s | lam %g"
      % (SHARD_I, SHARD_K, len(SHARD), len(_cand), ARMS, LAM), flush=True)

ROWS_PATH = HERE / ("rows.shard%d.jsonl" % SHARD_I)
done = set()
if ROWS_PATH.exists():
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"], r["arm"]))
        except Exception:
            pass
fh = ROWS_PATH.open("a")


MEMBERS_PATH = HERE / ("members.shard%d.jsonl" % SHARD_I)
mh = MEMBERS_PATH.open("a")


def skip(comp_idx, sl, reason):
    fh.write(json.dumps({"comp_idx": comp_idx, "latent": sl,
                         "arm": "skip", "reason": reason}) + "\n")
    fh.flush()


def dump_members(comp_idx, sl, tag, alphas):
    """The circuit itself: membership + fitted amplitudes. Real arms
    only (null memberships are random draws, not results)."""
    mh.write(json.dumps({
        "comp_idx": comp_idx, "latent": sl, "arm": tag,
        "alphas": {"%d/%s" % s: {str(i): round(a, 4)
                                 for i, a in d.items()}
                   for s, d in alphas.items()}}) + "\n")
    mh.flush()


def wants_nulls(comp_idx, sl):
    if SMOKE:
        return True
    return (comp_idx * 131071 + sl) % NULL_EVERY == 0


class AmpCircuitPatcher:
    """Members at alpha * live value, non-members at floor (zero when
    floors is None). Verbatim from 029-panel/runner.py."""

    def __init__(self, alphas, floors, seed_site, w_seed, b_seed):
        self.alphas, self.floors = alphas, floors or {}
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        al = self.alphas.get((layer_idx, kind))
        if al is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        fl = self.floors.get((layer_idx, kind))
        code = (fl.to(device=dense.device, dtype=dense.dtype)
                .expand_as(dense).clone() if fl is not None
                else torch.zeros_like(dense))
        if al:
            idx = torch.tensor(sorted(al), device=dense.device,
                               dtype=torch.long)
            av = torch.tensor([al[int(i)] for i in idx], device=dense.device,
                              dtype=dense.dtype)
            code[..., idx] = dense[..., idx] * av
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class AmpInjectPatcher:
    """Members SET to alpha_i * pin_i in the otherwise-live stream.
    Verbatim from 029-panel/runner.py."""

    def __init__(self, inject, seed_site, w_seed, b_seed):
        self.inject = inject
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        inj = self.inject.get((layer_idx, kind))
        if not inj:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        idx = torch.tensor(sorted(inj), device=dense.device, dtype=torch.long)
        vals = torch.tensor([inj[int(i)] for i in idx], device=dense.device,
                            dtype=dense.dtype)
        code[..., idx] = vals
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class LiveCounter:
    def __init__(self, sites):
        self.sites = set(sites)
        self.live = {s: torch.zeros(D_SAE, dtype=torch.bool) for s in sites}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s in self.sites:
            ta, ti = bank.encode(x, kind, layer_idx)
            idx = ti.reshape(-1)[ta.reshape(-1) > 0].to(torch.long).cpu()
            self.live[s][idx] = True
        return x


def process_seed(comp_idx, sl):
    """One candidate through every requested arm. Returns True if the
    seed produced rows (fit or legitimate skip), False on failure."""
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    band = "L%d %s" % (layer, kind)
    triple_w = 0.10 if layer <= 5 else 0.05     # 029-panel band calibration
    null_tags = (["null%d" % j for j in range(1 if SMOKE else N_NULL)]
                 if wants_nulls(comp_idx, sl) else [])
    if all((comp_idx, sl, t) in done for t in ARMS + null_tags):
        return True

    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg_acts, pb)
    try:
        pd_ = m0.build_probe_dataset(comp_idx, sl)
    except Exception as e:
        print("[%s %d] probes FAILED %s: %s" % (band, sl, type(e).__name__,
                                                e), flush=True)
        skip(comp_idx, sl, "probes_failed:%s" % type(e).__name__)
        del m0
        return False
    del m0
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
        skip(comp_idx, sl, "thin_probes")
        return True
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TRAIN], pa[:N_TRAIN], nt[:N_TRAIN]
    pt_ho, pa_ho, nt_ho = pt[N_TRAIN:], pa[N_TRAIN:], nt[N_TRAIN:]

    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind))
    if not UP:
        skip(comp_idx, sl, "no_upstream_sites")
        return True
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer,
                                             kind, sl, pa_tr,
                                             batch_size=EVAL_BS))
    a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer,
                                             kind, sl, pa_ho,
                                             batch_size=EVAL_BS))
    if a_pos_ho < 0.05:
        skip(comp_idx, sl, "a_pos_ho_small")
        return True
    means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr,
                                             set(UP), pa_tr,
                                             pin_position_specific=False)
    e0_tr = float(circuit_only_activation(inference, bank, {}, UP, pt_tr,
                                          layer, kind, sl, pos_argmax=pa_tr,
                                          batch_size=EVAL_BS))
    e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                          layer, kind, sl, pos_argmax=pa_ho,
                                          batch_size=EVAL_BS))
    eMd_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                           layer, kind, sl, pos_argmax=pa_ho,
                                           site_means=means_tr,
                                           batch_size=EVAL_BS))
    p0 = AmpInjectPatcher({}, (layer, kind), w_seed, b_seed)
    chunks = []
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(nt_ho.shape[0]), EVAL_BS):
                p0.seed_pre = None
                inference.forward(nt_ho[s0:s0 + EVAL_BS], patcher=p0,
                                  grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
                chunks.append(p0.seed_pre.detach())
    finally:
        inference.enable_compile()
    neg_pre = torch.cat(chunks, 0)
    na_ho = neg_pre.argmax(dim=1).cpu()
    a_base = float(torch.relu(
        neg_pre[torch.arange(neg_pre.shape[0], device=neg_pre.device),
                na_ho.to(neg_pre.device)]).mean())
    vac_f0 = abs(a_pos_ho - e0_ho) < 0.05 * abs(e0_ho)
    print("[%s %d] a_pos ho %.3f | e0_ho %.3f | a_base %.3f%s"
          % (band, sl, a_pos_ho, e0_ho, a_base,
             " | F0 VACUOUS" if vac_f0 else ""), flush=True)

    def read(patcher, tokens, anchors):
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_pre = None
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
            # (read at the anchor position of each sequence)
                pre = patcher.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(
                    0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum())
                n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def fit(free_amp, signed, negsup, support_members=None):
        kw = dict(sites=UP, seed_layer=layer, seed_kind=kind,
                  seed_latent_idx=sl, pos_tokens=pt_tr, pos_argmax=pa_tr,
                  neg_tokens=nt_tr, mask_floor_source="triple",
                  dual_floor_weight=cfg.dual_floor_weight,
                  triple_floor_weight=triple_w, free_amplitude=free_amp,
                  steps=int(cfg.steps), lr=cfg.lr,
                  keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, log_every=0,
                  deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size,
                  optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                  code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                  lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
        if signed:
            kw.update(signed_amplitude=True)
        if negsup:
            kw.update(neg_suppress_weight=NEG_W)
        if support_members is None:
            kw.update(l1_lambda=LAM, binarize=cfg.binarize,
                      theta_init=cfg.theta_init)
        else:
            support = {}
            for s, i in support_members:
                support.setdefault(s, []).append(i)
            kw.update(l1_lambda=0.0, binarize="none", theta_init=40.0,
                      support={s: torch.tensor(v, dtype=torch.long)
                               for s, v in support.items()})
        scores, prov = run_learned_mask(inference, bank, objective="pos",
                                        **kw)
        if free_amp:
            ak = prov.get("amp_kept") or {}
            alphas = {}
            for k, d in ak.items():
                lyr, knd = k.split("/")
                alphas[(int(lyr), knd)] = {int(i): float(v)
                                           for i, v in d.items()}
        else:
            alphas = {}
            for fid in scores:
                alphas.setdefault((fid.layer, fid.kind), {})[
                    int(fid.index)] = 1.0
        return alphas, (prov.get("amp_stats") or {})

    def score(alphas, st, tag, secs):
        f0_tr = ((read(AmpCircuitPatcher(alphas, None, (layer, kind),
                                         w_seed, b_seed), pt_tr, pa_tr)
                  - e0_tr) / (a_pos_tr - e0_tr)
                 if abs(a_pos_tr - e0_tr) > 1e-9 else None)
        aw0 = read(AmpCircuitPatcher(alphas, None, (layer, kind),
                                     w_seed, b_seed), pt_ho, pa_ho)
        awM = read(AmpCircuitPatcher(alphas, means_tr, (layer, kind),
                                     w_seed, b_seed), pt_ho, pa_ho)
        inject = {s: {i: float(a * float(pins_tr[s][i]))
                      for i, a in d.items()}
                  for s, d in alphas.items() if s in pins_tr}
        cfa = read(AmpInjectPatcher(inject, (layer, kind), w_seed, b_seed),
                   nt_ho, na_ho)
        n_mem = sum(len(d) for d in alphas.values())
        circ = Circuit(name=tag)
        for (l, kd), dd in alphas.items():
            for i in dd:
                circ.add_node(CircuitNode(metadata={
                    "layer_idx": l, "kind": kd, "latent_idx": int(i),
                    "role": "ablation_support"}))
        try:
            cf_bare, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, circ, neg_tokens=nt_ho,
                pos_tokens=pt_ho, seed_layer=layer, seed_kind=kind,
                seed_latent_idx=sl, pos_argmax=pa_ho,
                circuit_layers={l for (l, _) in alphas})
            cf_bare, sup_v = round(float(cf_bare), 4), round(float(sup_v), 4)
        except Exception as e:
            print("  cf_bare/sup failed: %s" % e, flush=True)
            cf_bare = sup_v = None
        row = {"comp_idx": comp_idx, "band": band, "latent": sl,
               "arm": tag, "n": n_mem,
               "ampF0_tr": round(f0_tr, 4) if f0_tr is not None else None,
               "ampF0": (round((aw0 - e0_ho) / (a_pos_ho - e0_ho), 4)
                         if abs(a_pos_ho - e0_ho) > 1e-9 else None),
               "ampFMd": (round((awM - eMd_ho) / (a_pos_ho - eMd_ho), 4)
                          if abs(a_pos_ho - eMd_ho) > 1e-9 else None),
               "cf_amp": (round((cfa - a_base) / (a_pos_ho - a_base), 4)
                          if abs(a_pos_ho - a_base) > 1e-9 else None),
               "cf_bare": cf_bare, "sup": sup_v,
               "f0_vacuous": bool(vac_f0),
               "alpha_med": st.get("median"), "alpha_p90": st.get("p90"),
               "a_pos_ho": round(a_pos_ho, 3),
               "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print("  %-10s n=%-6d F0=%-8s FMd=%-8s cf_amp=%-8s sup=%-8s"
              % (tag, n_mem, row["ampF0"], row["ampFMd"], row["cf_amp"],
                 sup_v), flush=True)
        return row

    n_ref = None
    for tag in ARMS:
        fa, sg, ns = ARM_SPECS[tag]
        if (comp_idx, sl, tag) in done:
            continue
        t0 = time.time()
        alphas, st = fit(fa, sg, ns)
        r = score(alphas, st, tag, time.time() - t0)
        dump_members(comp_idx, sl, tag, alphas)
        if tag == "triamp400":
            n_ref = r["n"]

    if null_tags:
        if n_ref is None:
            for line in ROWS_PATH.open():
                r = json.loads(line)
                if (r["comp_idx"], r["latent"],
                        r["arm"]) == (comp_idx, sl, "triamp400"):
                    n_ref = r["n"]
        if n_ref:
            lc = LiveCounter(UP)
            inference.disable_compile()
            try:
                with torch.no_grad():
                    for s0 in range(0, int(pt_tr.shape[0]), EVAL_BS):
                        inference.forward(pt_tr[s0:s0 + EVAL_BS], patcher=lc,
                                          grad_enabled=False,
                                          return_activations=False,
                                          tokenize_final=False)
            finally:
                inference.enable_compile()
            live = [(s, i) for s in UP
                    for i in lc.live[s].nonzero(as_tuple=True)[0].tolist()]
            rng = random.Random(1000 + sl)
            for tag in null_tags:
                # sample BEFORE the skip so resumes draw identical sets
                members = rng.sample(live, min(n_ref, len(live)))
                if (comp_idx, sl, tag) in done:
                    continue
                t0 = time.time()
                alphas, st = fit(True, False, False, support_members=members)
                score(alphas, st, tag, time.time() - t0)
    torch.cuda.empty_cache()
    return True


def main():
    consec_fail = 0
    for si, (comp_idx, sl) in enumerate(SHARD):
        if process_seed(comp_idx, sl):
            consec_fail = 0
        else:
            consec_fail += 1
            if consec_fail >= 3:
                sys.exit("3 consecutive seed failures — aborting shard "
                         "(last: comp %d latent %d)" % (comp_idx, sl))
        if si % 25 == 24:
            print("--- shard %d/%d: %d/%d seeds processed"
                  % (SHARD_I, SHARD_K, si + 1, len(SHARD)), flush=True)

    # completeness accounting (never print done without counting)
    fh.close()
    rows_by_seed, arm_counts = {}, {}
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows_by_seed.setdefault((r["comp_idx"], r["latent"]),
                                set()).add(r["arm"])
        arm_counts[r["arm"]] = arm_counts.get(r["arm"], 0) + 1
    missing = []
    for comp_idx, sl in SHARD:
        arms_here = rows_by_seed.get((comp_idx, sl), set())
        need = set(ARMS) | set("null%d" % j
                               for j in range(1 if SMOKE else N_NULL)
                               if wants_nulls(comp_idx, sl))
        if "skip" not in arms_here and not need <= arms_here:
            missing.append((comp_idx, sl, sorted(need - arms_here)))
    print("\nSHARD %d/%d REPORT: %d seeds | rows per arm: %s | "
          "incomplete: %d" % (SHARD_I, SHARD_K, len(SHARD),
                              sorted(arm_counts.items()), len(missing)),
          flush=True)
    for m in missing[:10]:
        print("  MISSING %s" % (m,), flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
