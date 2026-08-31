"""D4.2 — Driver identifiability: split-half probe resampling + precision
perturbation, on the 11-seed frozen-exam panel.

Per seed:
  halves A/B     disjoint probe halves ([0:24] / [24:48] of the
                 deterministic store order) via a runner-level
                 ProbeDatasetBuilder proxy; probe_sequence_count=24.
  R_A, R_B       abl-restoration PA discovered on each half
  A_A, A_B       abl-ig_mean PA discovered on each half
  C_A, C_B       direct-mass from disjoint 8-probe subsets of each half
  R_BF16         one R discovery on the full 48 with autocast_bf16=True
                 (precision perturbation; also the D2.6 Jaccard +
                 wall-clock A/B against the archived fp32 R ranking)

Measures:
  jac@K          Jaccard of top-K driver heads between halves, K in
                 {16, 64, 256, 1024} — DRIVER stability
  jac_full       Jaccard of FULL memberships between halves — CLOSURE
                 (membership) stability, same discoveries
  alpha_A/B      AMPC alpha fitted per half on the C_half sets (K=16,64);
                 cf_alpha of each on the FROZEN eval split
  kstar_A/B      pinned-bisection K* per half ranking against the frozen
                 eval anchors (resampling error bar for D1's K*)
Pre-registered: drivers (top-K heads) far more stable than closure
membership; if the K=16 heads are NOT stable, the single-latent-driver
headline needs the "a driver, not the driver" framing.

  PYTHONPATH=src python experiments/017-identifiability/runner.py
"""
import gzip
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDataset, ProbeDatasetBuilder
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

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D22 = HERE.parent / "019-roles-drivers"
N_SEQ, N_TR, N_HALF, EVAL_BS, PA_PCTL = 64, 48, 24, 16, 90.0
KS = (16, 64, 256, 1024)
D_SAE = 40960
PIN_TARGET = 0.8
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053), (17, 38268),
         (20, 35678), (25, 10628), (26, 17432), (27, 6859), (29, 2753),
         (35, 6599)]

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

KNOWN_ROLES = {"counterfactual_activator", "counterfactual_inhibitor",
               "ablation_support"}


class HalfBuilder:
    """Proxy that slices the probe dataset's POSITIVES to one half.
    Negatives untouched (floors are posctx-sourced in this config)."""

    def __init__(self, base, lo, hi):
        self._base, self._lo, self._hi = base, lo, hi

    def build_for_latent(self, *a, **kw):
        pd = self._base.build_for_latent(*a, **kw)
        return ProbeDataset(
            pos_tokens=pd.pos_tokens[self._lo:self._hi],
            target_tokens=pd.target_tokens[self._lo:self._hi],
            neg_tokens=pd.neg_tokens,
            pos_argmax=pd.pos_argmax[self._lo:self._hi],
            metadata=dict(pd.metadata, half=(self._lo, self._hi)),
        )

    def __getattr__(self, name):
        return getattr(self._base, name)


def base_state(n_probes):
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
    cf_cfg.neg_batch_size = 8
    cf_cfg.negative_roles = "include"
    ab_cfg.negative_roles = "include"
    cf_cfg.pruning_threshold = 0.0
    ab_cfg.pruning_threshold = 0.0
    for c in (ab_cfg, cf_cfg):
        c.restoration.round_select = "abs_pctl"
        c.restoration.round_abs_pctl = 95.0


class InjectPatcher:
    def __init__(self, targets, seed_site, seed_idx):
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
            done.add(json.loads(line)["seed"])

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    if seed_key in done:
        continue
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    base_state(N_TR)
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] < N_TR:
        print("[%s] fewer than %d positives — skip" % (seed_key, N_TR),
              flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    halves = {"A": (0, N_HALF), "B": (N_HALF, N_TR)}

    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_e0_ev = float(circuit_only_activation(inference, bank, {}, up, pt_ev,
                                            layer, kind, sl, pos_argmax=pa_ev,
                                            batch_size=EVAL_BS))
    den_ev = a_pos_ev - a_e0_ev
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    print("\n[%s] L%d %s — %d sites | a_pos %.3f" % (seed_key, layer, kind,
                                                     len(up), a_pos_ev),
          flush=True)

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
            out.append((rr, abs(float(sc)), (f.layer, f.kind), int(f.index)))
        out.sort(key=lambda x: (x[0], -x[1]))
        return [(site, idx) for _, _, site, idx in out]

    def discover_half(method, mode, half, by_round, autocast=False, full=False):
        lo, hi = (0, N_TR) if full else halves[half]
        base_state(N_TR if full else N_HALF)
        if mode == "restoration":
            for c in (ab_cfg, cf_cfg):
                c.restoration.rounds = max(1, len(up))
        builder = probe_builder if full else HalfBuilder(probe_builder, lo, hi)
        disc.autocast_bf16 = bool(autocast)
        try:
            meth = _build_mode_method(method, mode, inference, bank, avg_acts,
                                      builder)
            t0 = time.time()
            circ = meth.discover(sc_idx, sl)
            secs = round(time.time() - t0, 1)
            del meth
        finally:
            disc.autocast_bf16 = False
        if circ is None:
            raise RuntimeError("no circuit")
        rank = signed_members(circ, by_round=by_round)
        del circ
        torch.cuda.empty_cache()
        return rank, secs

    def build_direct_half(half):
        lo, hi = halves[half]
        pth, pah = pt[lo:lo + 8], pa[lo:lo + 8]
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
        inference.disable_compile()
        try:
            inference.forward(pth, patcher=instrument, grad_enabled=True,
                              return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
        pre = seed_pre[0]
        bi = torch.arange(min(8, pah.shape[0]), device=pre.device)
        metric = pre[bi, pah[:len(bi)].to(pre.device).clamp(0, pre.shape[1] - 1)].mean()
        graph = instrument.graph
        dsites = [s for s in sorted(graph.activations) if s in up]
        anchors = [graph.get_latents(*s)[0].act for s in dsites]
        grads = torch.autograd.grad(metric, anchors, allow_unused=True)
        triples = []
        for s, a, g in zip(dsites, anchors, grads):
            if g is not None:
                w = (g * a.detach()).sum(dim=1).mean(dim=0).abs().float().cpu()
                v, ix = torch.topk(w, k=min(2048, w.numel()))
                triples += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
        instrument.release()
        del instrument, graph, anchors, grads, seed_pre, pre
        torch.cuda.empty_cache()
        triples.sort(key=lambda x: -x[0])
        return [(s, i) for _, s, i in triples]

    def jac(r1, r2, k=None):
        s1 = set(r1 if k is None else r1[:k])
        s2 = set(r2 if k is None else r2[:k])
        u = len(s1 | s2)
        return round(len(s1 & s2) / u, 4) if u else None

    def keep_of(entries):
        keep = {}
        for site, idx in entries:
            keep.setdefault(site, set()).add(idx)
        return keep

    def pin0(entries):
        if abs(den_ev) < 1e-9:
            return None
        a_c = float(circuit_only_activation(
            inference, bank, keep_of(entries), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins_c))
        return (a_c - a_e0_ev) / den_ev

    def kstar(rank):
        if not rank or den_ev <= 1e-9:
            return None
        lo, hi = 1, len(rank)
        v = pin0(rank[:hi])
        if v is None or v < PIN_TARGET:
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if (pin0(rank[:mid]) or -1) >= PIN_TARGET:
                hi = mid
            else:
                lo = mid + 1
        return lo

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

    a_base_ev = seed_act_under(None, nt_ev, pa_ev)
    den_cf = a_pos_ev - a_base_ev

    def alpha_fit(members, half):
        lo_h, _ = halves[half]
        nt_fit = nt[lo_h:lo_h + 16]
        targets = {}
        for site, idx in members:
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
        if act_at(hi, nt_fit, pa_tr[:16]) < a_pos_tr:
            alpha = hi
        elif act_at(lo, nt_fit, pa_tr[:16]) > a_pos_tr:
            alpha = lo
        else:
            for _ in range(6):
                mid = (lo + hi) / 2
                if act_at(mid, nt_fit, pa_tr[:16]) < a_pos_tr:
                    lo = mid
                else:
                    hi = mid
            alpha = (lo + hi) / 2
        a_int = act_at(alpha, nt_ev, pa_ev)
        cf_a = round((a_int - a_base_ev) / den_cf, 4) if abs(den_cf) > 1e-9 else None
        return round(alpha, 3), cf_a

    row = {"seed": seed_key, "layer": layer, "kind": kind, "n_sites": len(up)}
    try:
        ranks = {}
        for m, (method, mode, br) in (("R", ("ablation_gradient", "restoration", True)),
                                      ("A", ("ablation_gradient", "ig_mean", False))):
            for h in ("A", "B"):
                r, secs = discover_half(method, mode, h, br)
                ranks[(m, h)] = r
                row["n_%s_%s" % (m, h)] = len(r)
                row["secs_%s_%s" % (m, h)] = secs
            for K in KS:
                row["jac_%s_%d" % (m, K)] = jac(ranks[(m, "A")], ranks[(m, "B")], K)
            row["jacfull_%s" % m] = jac(ranks[(m, "A")], ranks[(m, "B")])
        for h in ("A", "B"):
            ranks[("C", h)] = build_direct_half(h)
        for K in KS:
            row["jac_C_%d" % K] = jac(ranks[("C", "A")], ranks[("C", "B")], K)

        # precision perturbation: R full-48 bf16 vs archived fp32
        rb, secs_b = discover_half("ablation_gradient", "restoration", "A",
                                   True, autocast=True, full=True)
        row["secs_R_bf16"] = secs_b
        apath = D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
        r_fp32 = []
        with gzip.open(apath, "rt", encoding="utf-8") as gz:
            for line in gz:
                s, l, kd, idx, role, rr = json.loads(line)
                r_fp32.append(((l, kd), int(idx)))
        for K in KS:
            row["jac_prec_%d" % K] = jac(r_fp32, rb, K)
        row["jacfull_prec"] = jac(r_fp32, rb)
        row["n_R_bf16"] = len(rb)

        # K* per half on the A ranking (eval-anchor bisection)
        for h in ("A", "B"):
            row["kstar_%s" % h] = kstar(ranks[("A", h)])
        # AMPC alpha stability + transfer at K=16/64
        for K in (16, 64):
            for h in ("A", "B"):
                al, cfa = alpha_fit(ranks[("C", h)][:K], h)
                row["alpha%d_%s" % (K, h)] = al
                row["cfa%d_%s" % (K, h)] = cfa
        # archive half rankings (heads only, 4096) for anatomy reuse
        for (m, h), r in ranks.items():
            with gzip.open(HERE / ("rank_%s_%s_%d_%d.jsonl.gz"
                                   % (m, h, sc_idx, sl)), "wt",
                           encoding="utf-8") as gz:
                for site, idx in r[:4096]:
                    gz.write(json.dumps([site[0], site[1], idx]) + "\n")
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        print("  ERROR %s" % row["error"], flush=True)
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("  jacR16=%s jacR64=%s jacA16=%s jacC16=%s | jacfull R=%s A=%s | "
          "prec16=%s | K* %s/%s | a16 %s/%s cf %s/%s"
          % (row.get("jac_R_16"), row.get("jac_R_64"), row.get("jac_A_16"),
             row.get("jac_C_16"), row.get("jacfull_R"), row.get("jacfull_A"),
             row.get("jac_prec_16"), row.get("kstar_A"), row.get("kstar_B"),
             row.get("alpha16_A"), row.get("alpha16_B"),
             row.get("cfa16_A"), row.get("cfa16_B")), flush=True)

print("ALL DONE", flush=True)
fh.close()
