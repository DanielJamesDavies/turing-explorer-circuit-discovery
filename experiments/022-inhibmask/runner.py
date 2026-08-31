"""Inhibitor-mask ("raise" objective) — 8-seed probe with early stopping.

Objective (Daniel's proposal): push the seed ABOVE its natural level on
POSCTX by silencing latents, paying l1 per silenced latent. The mirror of
abl-mask: pos = "keep the seed at natural, pay to KEEP" (support);
raise = "push above natural, pay to SILENCE" (brake set). Inert latents
are evicted because silencing them buys nothing; activators are never
recruited because silencing them LOWERS the seed.

Arms per seed: l1_lambda in {1e-5 (house), 1e-3, 1e-2} at gamma 1.5.
Early stopping is built in (the cf-mask dynamics lesson): every 25 steps
the live circuit is scored and the BEST snapshot is reported alongside
the final one.

Metrics per snapshot:
  raise_ho   (a_raised - a_pos)/a_pos on the HELD-OUT posctx split, with
             members silenced — the objective's own transfer test
  raise_rand same, for a size-matched RANDOM set — the control that says
             the raise is not generic
  sup        the INDEPENDENT test: members injected at negctx values on
             posctx (phi-sup) — do these latents SILENCE the seed when
             fired? Never trained on.
  al_R_inh / al_R_act  alignment with restoration's attribution-signed
             inhibitors / activators (D2.2 archive, 24 seeds)
  al_maskMF  alignment with the D3.6 abl-mask closure circuit

  PYTHONPATH=src python experiments/022-inhibmask/runner.py
"""
import gzip
import json
import os
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import measure_seed_activation, upstream_sites
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D22 = HERE.parent / "019-roles-drivers"
D36 = HERE.parent / "018-maskrefine"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
SNAP_EVERY = 25
GAMMA = float(os.environ.get("GAMMA", 1.5))
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "1e-5,1e-3,1e-2").split(",")]
TAG = os.environ.get("TAG", "v1")
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053),
         (20, 35678), (26, 17432), (27, 6859), (35, 6599)]

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
lm = config.discovery.learned_mask


class SilencePatcher:
    """Zero the given latents; everything else natural."""

    def __init__(self, zero_idx, seed_site, seed_idx):
        self.zero_idx = zero_idx
        self.seed_site, self.seed_idx = seed_site, seed_idx
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
        z = self.zero_idx.get((layer_idx, kind))
        if not z:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        c_new = dense.clone()
        zi = torch.tensor(sorted(z), device=dense.device, dtype=torch.long)
        c_new[..., zi] = 0.0
        out = bank.decode(c_new - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


OUT = HERE / ("rows_%s.jsonl" % TAG)
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    if all((seed_key, "lam%g" % l) in done for l in LAMBDAS):
        continue

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] < N_TR:
        print("[%s] too few positives — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))

    ref_r_act, ref_r_inh = set(), set()
    rp = D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
    if rp.exists():
        with gzip.open(rp, "rt", encoding="utf-8") as gz:
            for n, line in enumerate(gz):
                if n >= 20000:
                    break
                s_, l_, kd_, idx_, role_, rr_ = json.loads(line)
                (ref_r_inh if role_ == "counterfactual_inhibitor"
                 else ref_r_act).add(((l_, kd_), int(idx_)))
    ref_mask = set()
    mp = D36 / ("members_MF_%d_%d.jsonl.gz" % (sc_idx, sl))
    if mp.exists():
        with gzip.open(mp, "rt", encoding="utf-8") as gz:
            for line in gz:
                l_, kd_, idx_, m_ = json.loads(line)
                ref_mask.add(((l_, kd_), int(idx_)))

    print("\n[%s] L%d %s — %d sites | a_pos ev %.3f tr %.3f | target %.3f "
          "| refs R %d inh / %d act, maskMF %d"
          % (seed_key, layer, kind, len(up), a_pos_ev, a_pos_tr,
             GAMMA * a_pos_tr, len(ref_r_inh), len(ref_r_act), len(ref_mask)),
          flush=True)

    def act_silencing(members, tokens, argmax):
        """seed activation with `members` silenced (natural elsewhere)."""
        zero_idx = {}
        for site, idx in members:
            zero_idx.setdefault(site, []).append(idx)
        p = SilencePatcher(zero_idx, (layer, kind), sl)
        total, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                p.seed_capture = None
                p.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                total += float(p.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return total / max(n, 1)

    def sup_of(members):
        """phi-sup: members injected at negctx values on posctx (independent
        of the training objective)."""
        if not members:
            return None
        c = Circuit(name="inh")
        for (l, kd), idx in members:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx,
                "role": "counterfactual_inhibitor"}))
        try:
            _cf, sup = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev,
                pos_tokens=pt_ev, seed_layer=layer, seed_kind=kind,
                seed_latent_idx=sl, pos_argmax=pa_ev,
                circuit_layers={l for (l, _), _ in members})
            return round(float(sup), 4)
        except Exception as exc:
            print("    sup error: %s" % str(exc)[:70], flush=True)
            return None

    rng = random.Random(17)

    for lam in LAMBDAS:
        arm = "lam%g" % lam
        if (seed_key, arm) in done:
            continue
        best = {"score": -1e9}
        snaps = []

        def hook(step, ctx):
            if step % SNAP_EVERY and step != int(lm.steps) - 1:
                return
            kt = ctx["keep_threshold"]
            mem = []
            with torch.no_grad():
                for site, th in ctx["thetas"].items():
                    edit = 1.0 - torch.sigmoid(th)
                    idx = (edit > kt).nonzero(as_tuple=True)[0]
                    mem += [(site, int(i)) for i in idx.tolist()]
            if not mem:
                snaps.append({"step": step, "n": 0})
                return
            a_r = act_silencing(mem, pt_ev, pa_ev)
            raise_ho = round((a_r - a_pos_ev) / max(a_pos_ev, 1e-9), 4)
            rnd = [(up_sorted[rng.randrange(len(up_sorted))],
                    rng.randrange(D_SAE)) for _ in range(len(mem))]
            a_rand = act_silencing(rnd, pt_ev, pa_ev)
            raise_rand = round((a_rand - a_pos_ev) / max(a_pos_ev, 1e-9), 4)
            row = {"step": step, "n": len(mem), "raise_ho": raise_ho,
                   "raise_rand": raise_rand, "sup": sup_of(mem),
                   "al_R_inh": (round(len(set(mem) & ref_r_inh) / len(mem), 4)
                                if ref_r_inh else None),
                   "al_R_act": (round(len(set(mem) & ref_r_act) / len(mem), 4)
                                if ref_r_act else None),
                   "al_maskMF": (round(len(set(mem) & ref_mask) / len(mem), 4)
                                 if ref_mask else None)}
            snaps.append(row)
            # early-stopping score: transfer above the random control
            sc = raise_ho - max(raise_rand, 0.0)
            if sc > best["score"]:
                best.update({"score": sc, **row})

        t0 = time.time()
        try:
            scores, prov = run_learned_mask(
                inference, bank, objective="raise", sites=up_sorted,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
                target_act=a_pos_tr, raise_gamma=GAMMA, scale_normalize=True,
                mask_floor_source="zero",
                binarize=lm.binarize, steps=lm.steps, lr=lm.lr,
                l1_lambda=lam, keep_threshold=lm.keep_threshold,
                batch_size=4, holdout_frac=lm.holdout_frac,
                theta_init=lm.theta_init, log_every=0,
                deep_site_threshold=lm.deep_site_threshold,
                deep_batch_size=lm.deep_batch_size,
                optimizer=lm.optimizer, weight_decay=lm.weight_decay,
                code_dtype=lm.code_dtype, lr_schedule=lm.lr_schedule,
                lr_min_frac=lm.lr_min_frac, warmup_frac=lm.warmup_frac,
                step_hook=hook)
        except Exception as exc:
            print("  %-8s ERROR %s: %s" % (arm, type(exc).__name__,
                                           str(exc)[:110]), flush=True)
            continue
        secs = round(time.time() - t0, 1)
        fin = snaps[-1] if snaps else {"n": 0}
        row = {"seed": seed_key, "layer": layer, "kind": kind, "arm": arm,
               "gamma": GAMMA, "l1_lambda": lam,
               "n_final": len(scores), "secs": secs,
               "final": fin, "best": {k: v for k, v in best.items()
                                      if k != "score"},
               "holdout_loss": prov.get("holdout_data_loss"),
               "n_snaps": len(snaps)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        with gzip.open(HERE / ("members_%s_%s_%d_%d.jsonl.gz"
                               % (TAG, arm, sc_idx, sl)), "wt",
                       encoding="utf-8") as gz:
            for f, v in scores.items():
                gz.write(json.dumps([f.layer, f.kind, f.index,
                                     round(float(v), 4)]) + chr(10))
        b = best
        print("  %-8s n_fin=%-6d | BEST step %-3s n=%-6s raise=%-8s "
              "rand=%-8s sup=%-7s R_inh=%-6s (%ss)"
              % (arm, len(scores), b.get("step"), b.get("n"),
                 b.get("raise_ho"), b.get("raise_rand"), b.get("sup"),
                 b.get("al_R_inh"), secs), flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
