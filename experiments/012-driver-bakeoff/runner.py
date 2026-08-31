"""D1 — Driver-circuit method bake-off on the frozen exam.

12 seeds (the 8 layer-stratified panel seeds + 4 kind-balancing: documented
L8-mlp and L9-attn + sampled L1-attn and L4-mlp), 6 ranking arms + controls,
matched GLOBAL budgets K in {64, 256, 1024, 4096}.

Arms (cheap -> dear per seed):
  A    abl-ig_mean PA head        (also yields DEF-B K* via pinned bisection)
  D    cf-ig_mean PA head         (signed roles kept)
  C    direct-mass                (full-dict unmediated weights; standing arm)
  AMPC amplified direct-mass      (C at K, injected at alpha* fitted on the
                                   train split to hit a_pos — D1 arm 8)
  MI   cf mask_inject             (scouting: ranking by learned score)
  R    abl-restoration PA         (rounds=sites; ranking by (round, -|score|))
  H    hybrid restoration+ig_mean (the measured node-selection champion)
  RAND random control; AMPR amplified-random control at K=256 only.

Exam per (arm, K): free0 / pin0_collapsed / pin0_pos (eval split), phi-cf /
phi-sup (eval split), calibrated imposition error, layer profile, median
direct percentile, pct of upstream dictionary. D0.1 lessons applied: phi-cf
at small K is the headline; pinned and intervention families reported
separately; roles normalised so no member is silently dropped by the cf eval.

  SEED_IDX=0 PYTHONPATH=src python experiments/012-driver-bakeoff/runner.py
"""
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
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
N_SEQ, N_TR, EVAL_BS, GRAD_B, PA_PCTL = 64, 48, 16, 8, 90.0
KS = (64, 256, 1024, 4096)
PIN_TARGET = 0.8
D_SAE = 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
cf_cfg, ab_cfg = disc.counterfactual_gradient, disc.ablation_gradient

# ---- panel: 8 stratified (layers 0,2,3,5,6,8,9,11) + 4 kind-balancing -----
_all_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu", weights_only=False)
_by_layer = defaultdict(list)
for _i, _c in enumerate(_all_cand):
    _by_layer[int(_c["comp_idx"]) // n_kinds].append(_i)
_rng = random.Random(42)
for _L in sorted(_by_layer):
    _rng.shuffle(_by_layer[_L])
PANEL_LAYERS = [0, 2, 3, 5, 6, 8, 9, 11]
SEEDS = [(int(_all_cand[_by_layer[L][-1]]["comp_idx"]),
          int(_all_cand[_by_layer[L][-1]]["latent_idx"])) for L in PANEL_LAYERS]


def _kind_pick(layer_want, kind_substr):
    for i in _by_layer[layer_want]:
        ci = int(_all_cand[i]["comp_idx"])
        if kind_substr in bank.kinds[ci % n_kinds]:
            return (ci, int(_all_cand[i]["latent_idx"]))
    return None


SEEDS += [(25, 10628), (27, 6859)]          # documented L8-mlp, L9-attn
for pick in (_kind_pick(1, "attn"), _kind_pick(4, "mlp")):
    if pick is not None and pick not in SEEDS:
        SEEDS.append(pick)
print("panel: %d seeds" % len(SEEDS), flush=True)

if os.environ.get("SEED_IDX"):
    i = int(os.environ["SEED_IDX"])
    SEEDS, OUT = [SEEDS[i]], HERE / ("rows_s%d.jsonl" % i)
else:
    OUT = HERE / "rows.jsonl"


def base_state(n_probes, deep=False):
    """deep flag retained for provenance, but after the instrument-release
    fix (vram-ledger 2026-07-31 post-fix) the cross-pass leak is gone and a
    25-site seed at pbs=4 peaks at 10.2GB — inside dedicated VRAM. pbs=4 is
    also the wall-clock sweet spot (18.3s vs 22.9s @2, 49.8s @8 spilled),
    so batching is uniform again."""
    disc.probe_sequence_count = n_probes
    disc.eval_sequence_count = n_probes
    disc.eval_batch_size = EVAL_BS
    disc.probe_batch_size = 4
    disc.position_aware = True
    disc.position_aware_select = "abs_pctl"
    disc.position_aware_threshold = PA_PCTL
    disc.floor_source = "posctx"
    disc.magnitude_prune = False
    disc.recurrence_prune = False
    disc.min_faithfulness = -100.0
    cf_cfg.max_neg_sequences = n_probes
    cf_cfg.neg_batch_size = 4 if deep else 8
    cf_cfg.negative_roles = "include"
    ab_cfg.negative_roles = "include"
    cf_cfg.pruning_threshold = 0.0
    ab_cfg.pruning_threshold = 0.0


def restoration_rounds(n_sites):
    for c in (ab_cfg, cf_cfg):
        c.restoration.rounds = max(1, n_sites)
        c.restoration.round_select = "abs_pctl"
        c.restoration.round_abs_pctl = 95.0


KNOWN_ROLES = {"counterfactual_activator", "counterfactual_inhibitor",
               "ablation_support"}


class InjectPatcher:
    """Set member latents to alpha * target on every forward position; all
    other latents natural; reconstruction errors untouched (delta-decode).
    Captures the SEED's post-Top-K activation itself (CircuitOnlyPatcher
    style) — set seed_site/seed_idx/argmax_chunk before each forward."""

    def __init__(self, targets, seed_site, seed_idx):
        # targets: {(layer, kind): {idx: value}}
        self.targets = targets
        self.seed_site = seed_site
        self.seed_idx = seed_idx
        self.argmax_chunk = None
        self.seed_capture = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            ta, ti = bank.encode(x, kind, layer_idx)
            s_dense = target_latent_activations(ta, ti, self.seed_idx)  # [B,T]
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


fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"], r["K"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    dict_up = max(len(up), 1) * D_SAE
    DEEP = ("very" if len(up) >= 25 else True) if len(up) >= 18 else False
    grad_b = 4 if DEEP else GRAD_B

    base_state(N_SEQ)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]

    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_e0_ev = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                            kind, sl, pos_argmax=pa_ev,
                                            batch_size=EVAL_BS))
    den_ev = a_pos_ev - a_e0_ev
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)
    _, pins_p = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=True)
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_e0_tr = float(circuit_only_activation(inference, bank, {}, up, pt_tr, layer,
                                            kind, sl, pos_argmax=pa_tr,
                                            batch_size=EVAL_BS))
    den_tr = a_pos_tr - a_e0_tr
    _, pins_c_tr = collect_site_anchors(inference, bank, pt_tr, up, pa_tr,
                                        pin_position_specific=False)

    print("\n[%s] L%d %s — %d sites | eval a_pos %.3f | train a_pos %.3f"
          % (seed_key, layer, kind, len(up), a_pos_ev, a_pos_tr), flush=True)

    def signed_members(circ, by_round=False):
        out = []
        for node in circ.nodes.values():
            role = node.metadata.get("role")
            if role == "seed":
                continue
            f = node.feature_id
            if f is None or (f.layer, f.kind) not in up:
                continue
            sc = node.metadata.get("effect_score")
            if sc is None:
                sc = node.metadata.get("attribution_score")
            if sc is None:
                sc = node.metadata.get("weight") or 0.0
            rr = node.metadata.get("selected_round", 0) if by_round else 0
            role_n = role if role in KNOWN_ROLES else "ablation_support"
            out.append((rr, abs(float(sc)), (f.layer, f.kind), int(f.index), role_n))
        out.sort(key=lambda x: (x[0], -x[1]))
        return [(s, site, idx, role) for _, s, site, idx, role in out]

    def discover(name, mode, needs_rounds=False, npa=False):
        base_state(N_TR, deep=DEEP)
        if needs_rounds:
            restoration_rounds(len(up))
        if npa:
            disc.position_aware = False    # mask modes are NPA by design
            # dual floor is pos-objective-only; inject/negctx need a plain floor
            disc.learned_mask.mask_floor_source = "zero"
        meth = _build_mode_method(name, mode, inference, bank, avg_acts,
                                  probe_builder)
        if name == "hybrid_gradient":
            # HybridGradientDiscovery.discover reads these but __init__ only
            # sets them on its SUB-methods — plumbing gap, patched locally.
            for attr, val in (("recurrence_prune", False),
                              ("recurrence_prune_min_sequences", 2),
                              ("recurrence_prune_min_keep", 0),
                              ("magnitude_prune", False)):
                if not hasattr(meth, attr):
                    setattr(meth, attr, val)
        t0 = time.time()
        circ = meth.discover(sc_idx, sl)
        secs = time.time() - t0
        if circ is None:
            raise RuntimeError("discovery returned no circuit "
                               "(empty at these hyperparameters)")
        return circ, secs

    # ---- rankings (built lazily, cached in-process) ----------------------
    rankings, disc_secs = {}, {}

    def get_rank(name):
        if name in rankings:
            return rankings[name]
        if name == "A":
            circ, s = discover("ablation_gradient", "ig_mean")
            rankings[name] = signed_members(circ)
        elif name == "D":
            circ, s = discover("counterfactual_gradient", "ig_mean")
            rankings[name] = signed_members(circ)
        elif name == "MI":
            circ, s = discover("counterfactual_gradient", "mask_inject",
                               npa=True)
            rankings[name] = signed_members(circ)
        elif name == "R":
            circ, s = discover("ablation_gradient", "restoration",
                               needs_rounds=True)
            rankings[name] = signed_members(circ, by_round=True)
        elif name == "H":
            circ, s = discover("hybrid_gradient", "restoration+ig_mean",
                               needs_rounds=True)
            rankings[name] = signed_members(circ)
        elif name == "C":
            s = build_direct()
        else:
            raise ValueError(name)
        disc_secs[name] = round(s, 1)
        torch.cuda.empty_cache()
        return rankings[name]

    def build_direct():
        sae = bank.saes[kind][layer]
        w_seed = sae.encoder.weight[sl].detach()
        b_seed = sae._get_bias_eff()[sl].detach()
        instrument = SAEGraphInstrument(bank)
        seed_pre = []
        orig = instrument.transform

        def tap(layer_idx, kd, x):
            if layer_idx == layer and kd == kind:
                seed_pre.append(x @ w_seed.to(x.device, x.dtype)
                                + b_seed.to(x.device, x.dtype))
                return x
            return orig(layer_idx, kd, x)

        instrument.transform = tap
        t0 = time.time()
        inference.disable_compile()
        try:
            inference.forward(pt_tr[:grad_b], patcher=instrument,
                              grad_enabled=True, return_activations=False,
                              tokenize_final=False)
        finally:
            inference.enable_compile()
        pre = seed_pre[0]
        bi = torch.arange(min(grad_b, pa_tr.shape[0]), device=pre.device)
        metric = pre[bi, pa_tr[:len(bi)].to(pre.device).clamp(0, pre.shape[1] - 1)].mean()
        graph = instrument.graph
        dsites = [s for s in sorted(graph.activations) if s in up]
        anchors = [graph.get_latents(*s)[0].act for s in dsites]
        grads = torch.autograd.grad(metric, anchors, allow_unused=True)
        dw = {}
        for s, a, g in zip(dsites, anchors, grads):
            if g is not None:
                dw[s] = (g * a.detach()).sum(dim=1).mean(dim=0).abs().float().cpu()
        del instrument, graph, anchors, grads, seed_pre, pre
        torch.cuda.empty_cache()
        torch.save({"direct": dw, "seed": (sc_idx, sl)},
                   HERE / ("direct_full_%d_%d.pt" % (sc_idx, sl)))
        triples = []
        for s, w in dw.items():
            v, ix = torch.topk(w, k=min(8192, w.numel()))
            triples += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
        triples.sort(key=lambda x: -x[0])
        rankings["C"] = [(sc, site, idx, "ablation_support")
                         for sc, site, idx in triples]
        globals()["_dw"] = dw
        all_dw = torch.cat([w for w in dw.values()])
        globals()["_sorted_dw"], _ = torch.sort(all_dw)
        return time.time() - t0

    def direct_pctl(site, idx):
        dw = globals().get("_dw")
        if not dw or site not in dw:
            return None
        v = float(dw[site][idx])
        pos = int(torch.searchsorted(globals()["_sorted_dw"], torch.tensor(v)))
        return 100.0 * pos / max(len(globals()["_sorted_dw"]), 1)

    # ---- exam helpers (same as D0.1) --------------------------------------
    def keep_of(entries):
        keep = {}
        for _, site, idx, _ in entries:
            keep.setdefault(site, set()).add(idx)
        return keep

    def phi0(entries, pins=None, train=False):
        tokens, argmax = (pt_tr, pa_tr) if train else (pt_ev, pa_ev)
        a_e, den = (a_e0_tr, den_tr) if train else (a_e0_ev, den_ev)
        if abs(den) < 1e-9:
            return None
        a_c = float(circuit_only_activation(
            inference, bank, keep_of(entries), up, tokens, layer, kind, sl,
            pos_argmax=argmax, batch_size=EVAL_BS, pin_values=pins))
        return round((a_c - a_e) / den, 4)

    def cf_eval(entries):
        c = Circuit(name="d1")
        for _, (l, kd), idx, role in entries:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx, "role": role}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev,
                circuit_layers={l for _, (l, _), _, _ in entries})
            return round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception as exc:
            print("    cf_eval error: %s" % str(exc)[:80], flush=True)
            return None, None

    def seed_act_under(patcher, tokens, argmax):
        if patcher is None:
            return float(measure_seed_activation(
                inference, bank, tokens, layer, kind, sl, argmax,
                batch_size=EVAL_BS))
        total, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_capture = None
                patcher.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
                total += float(patcher.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return total / max(n, 1)

    def alpha_fit(entries):
        """Fit alpha on the TRAIN split so injecting members at
        alpha * clean value hits a_pos; returns (alpha*, cf_alpha on eval)."""
        targets = {}
        for _, site, idx, _ in entries:
            v = float(pins_c[site][idx]) if site in pins_c else 0.0
            if v > 0:
                targets.setdefault(site, {})[idx] = v
        if not targets:
            return None, None
        base = {s: dict(t) for s, t in targets.items()}

        def act_at(alpha, tokens, argmax):
            scaled = {s: {i: alpha * v for i, v in t.items()}
                      for s, t in base.items()}
            return seed_act_under(InjectPatcher(scaled, (layer, kind), sl),
                                  tokens, argmax)

        lo, hi = 0.25, 8.0
        a_lo = act_at(lo, nt_tr[:16], pa_tr[:16])
        a_hi = act_at(hi, nt_tr[:16], pa_tr[:16])
        if a_hi < a_pos_tr:            # even 8x cannot reach — report ceiling
            alpha = hi
        elif a_lo > a_pos_tr:
            alpha = lo
        else:
            for _ in range(6):
                mid = (lo + hi) / 2
                if act_at(mid, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                    lo = mid
                else:
                    hi = mid
            alpha = (lo + hi) / 2
        a_int = act_at(alpha, nt_ev, pa_ev)
        a_base = seed_act_under(None, nt_ev, pa_ev)
        den = a_pos_ev - a_base
        cf_a = round((a_int - a_base) / den, 4) if abs(den) > 1e-9 else None
        return round(alpha, 3), cf_a

    def profile(entries):
        if not entries:
            return None, None, None
        dists = [layer - l for _, (l, _), _, _ in entries]
        pctls = [p for p in (direct_pctl(site, idx)
                             for _, site, idx, _ in entries) if p is not None]
        near = sum(1 for d in dists if d <= 2) / len(dists)
        medp = sorted(pctls)[len(pctls) // 2] if pctls else None
        return (round(sum(dists) / len(dists), 2), round(near, 3),
                (round(medp, 1) if medp is not None else None))

    # ---- K* on the A ranking (train split) --------------------------------
    get_rank("C")                      # direct weights first (profiles need them)
    rank_a = get_rank("A")
    k_star = None
    if den_tr > 1e-9 and rank_a:
        def pin_tr(k):
            v = phi0(rank_a[:k], pins=pins_c_tr, train=True)
            return v if v is not None else -1
        lo, hi = 1, len(rank_a)
        if pin_tr(hi) < PIN_TARGET:
            k_star = -1
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if pin_tr(mid) >= PIN_TARGET:
                    hi = mid
                else:
                    lo = mid + 1
            k_star = lo
    print("  K*(pin0>=%.1f) = %s" % (PIN_TARGET, k_star), flush=True)

    rng = random.Random(42)
    ARM_ORDER = ["A", "D", "C", "MI", "R", "H"]
    for arm in ARM_ORDER:
        try:
            rank = get_rank(arm)
        except Exception as exc:
            print("  %-4s DISCOVERY ERROR %s: %s" % (arm, type(exc).__name__,
                                                     str(exc)[:90]), flush=True)
            continue
        for K in KS:
            if (seed_key, arm, K) in done:
                continue
            entries = rank[:K]
            if not entries:
                continue
            t0 = time.time()
            cf_v, sup_v = cf_eval(entries)
            alpha_s = cf_a = None
            if arm == "C":
                try:
                    alpha_s, cf_a = alpha_fit(entries)
                except Exception as exc:
                    print("    alpha_fit error: %s" % str(exc)[:80], flush=True)
            mean_d, near2, medp = profile(entries)
            row = {
                "seed": seed_key, "layer": layer, "kind": kind,
                "arm": arm, "K": K, "n": len(entries),
                "pct_dict": round(100.0 * len(entries) / dict_up, 4),
                "free0": phi0(entries),
                "pin0_c": phi0(entries, pins=pins_c),
                "pin0_p": phi0(entries, pins=pins_p),
                "cf": cf_v, "sup": sup_v,
                "imp_err": (round(abs(cf_v * den_ev + a_e0_ev - a_pos_ev)
                                  / max(a_pos_ev, 1e-9), 4)
                            if cf_v is not None else None),
                "alpha_star": alpha_s, "cf_alpha": cf_a,
                "mean_dist": mean_d, "near2": near2, "med_direct_pctl": medp,
                "k_star": k_star if (arm == "A" and K == KS[0]) else None,
                "secs_disc": disc_secs.get(arm), "secs_eval": round(time.time() - t0, 1),
            }
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-4s K=%4d cf=%-7s sup=%-7s pin0c=%-7s pin0p=%-7s "
                  "free0=%-7s a*=%-5s cf_a=%-7s (disc %ss eval %ss)"
                  % (arm, K, cf_v, sup_v, row["pin0_c"], row["pin0_p"],
                     row["free0"], alpha_s, cf_a, row["secs_disc"],
                     row["secs_eval"]), flush=True)

    # controls
    for K in KS:
        if (seed_key, "RAND", K) in done:
            continue
        entries = [(0.0, up_sorted[rng.randrange(len(up_sorted))],
                    rng.randrange(D_SAE), "ablation_support") for _ in range(K)]
        cf_v, sup_v = cf_eval(entries)
        alpha_s = cf_a = None
        if K == 256:
            try:
                alpha_s, cf_a = alpha_fit(entries)
            except Exception as exc:
                print("    alpha_fit error: %s" % str(exc)[:80], flush=True)
        row = {"seed": seed_key, "layer": layer, "kind": kind, "arm": "RAND",
               "K": K, "n": K, "pct_dict": round(100.0 * K / dict_up, 4),
               "free0": phi0(entries), "pin0_c": phi0(entries, pins=pins_c),
               "pin0_p": phi0(entries, pins=pins_p), "cf": cf_v, "sup": sup_v,
               "imp_err": None, "alpha_star": alpha_s, "cf_alpha": cf_a,
               "mean_dist": None, "near2": None, "med_direct_pctl": None,
               "k_star": None, "secs_disc": None, "secs_eval": None}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  RAND K=%4d cf=%-7s sup=%-7s cf_a=%s"
              % (K, cf_v, sup_v, cf_a), flush=True)
    rankings.clear()
    torch.cuda.empty_cache()

fh.close()
print("\nwrote %s" % OUT)
