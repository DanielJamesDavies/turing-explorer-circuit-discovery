"""VERIFICATION of amp_eval_pass_v2.py (rows v=3; checks B, C, E, F, G; D is in amp_eval_v2_summary.py;
A, the v1-vs-v2 regression in amp_eval_v2_regress.py, no longer applies: v3 fixes v1's scoring bugs).

  B  identity: a "circuit" holding EVERY latent at alpha = 1 must reproduce a_pos under zero fill
     (engine CircuitOnlyPatcher) and under phi_pin with all latents pinned.
  G  scope (DAN-67): every circuit-only run ablates EVERY upstream site, including sites holding no member;
     a circuit whose only members sit at ONE site must score differently from one that also keeps every
     latent at the other sites, and the row's all_sites_ablated flag must be set.
  C  Top-K fill contract of the engine's _respect_topk_fill: synthetic unit test + an audit of every
     site of a real freeM_topk run (fill count = k - active members; members never filled; filled =
     the highest-mean non-members at their means; everything else exactly zero).
  E  role attribution (grad x activation) against finite differences of a single member's scale, and
     against single-member zero ablation.
  F  LazyPins (member columns only) against the engine's own position-specific pins.

  SEEDS=5.resid.13677,2.mlp.11004 PYTHONPATH=src python experiments/049-circuit-graph/amp_eval_v2_checks.py
"""
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("SEEDS", "5.resid.13677,2.mlp.11004")
os.environ.setdefault("TAG", "checks_unused")
sys.path.insert(0, str(Path(__file__).parent))
import amp_eval_pass_v2 as V  # noqa: E402

items, pool, _, src = V.load_items()
G = V.setup()
from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites  # noqa: E402
from eval.floors import collect_site_anchors  # noqa: E402
from pipeline.component_index import component_idx as comp_of  # noqa: E402
from sae.dense import sparse_topk_to_dense  # noqa: E402

inference, bank, M0, KINDS, NK, D, K = (G[k] for k in ("inference", "bank", "M0", "KINDS", "NK", "D", "K"))
TapCO, TapCF = V._engine_classes()
print("checks on %s | d_sae %d k %d" % (src, D, K), flush=True)

# ------------------------------------------------------------------ C (unit)
print("\n=== C. _respect_topk_fill contract: synthetic unit test ===")
torch.manual_seed(0)
dev = G["device"]
p = CircuitOnlyPatcher(bank=bank, keep_indices={}, in_scope=set(), seed_layer=0, seed_kind="mlp", seed_latent_idx=0,
                       respect_topk=True, topk=K)


def audit_fill(patched, mean_vector, keep_tensor, kept_values, k):
    """returns dict of violation counts for one [B, T, D] fill"""
    Dn = patched.shape[-1]
    is_keep = torch.zeros(Dn, dtype=torch.bool, device=patched.device)
    if keep_tensor is not None:
        is_keep[keep_tensor] = True
    v = {}
    if keep_tensor is not None:
        v["member_value_changed"] = int((patched[..., keep_tensor] != kept_values.to(patched.dtype)).sum())
        n_act = (kept_values != 0).sum(-1)
    else:
        v["member_value_changed"] = 0
        n_act = torch.zeros(patched.shape[:2], dtype=torch.long, device=patched.device)
    budget = (k - n_act).clamp(min=0)
    rank = mean_vector.clone().float(); rank[is_keep] = float("-inf")
    order = torch.argsort(rank, descending=True)
    pos_in_order = torch.empty(Dn, dtype=torch.long, device=patched.device); pos_in_order[order] = torch.arange(Dn, device=patched.device)
    should = (pos_in_order.view(1, 1, -1) < budget.unsqueeze(-1)) & (~is_keep).view(1, 1, -1)       # [B, T, D] expected fill set
    expect = torch.where(should, mean_vector.to(patched.dtype).view(1, 1, -1).expand_as(patched), torch.zeros_like(patched))
    nonmem = patched.clone(); nonmem[..., is_keep] = 0
    v["nonmember_value_mismatch"] = int((nonmem != expect).sum())
    n_fill_nz = (nonmem != 0).sum(-1)
    n_should_nz = (expect != 0).sum(-1)
    v["fill_count_mismatch"] = int((n_fill_nz != n_should_nz).sum())
    v["budget_min"], v["budget_max"] = int(budget.min()), int(budget.max())
    v["fill_slots_with_zero_mean"] = int((should & (expect == 0)).sum())       # ranked-in latents whose mean is 0: filled "at their mean" = 0
    return v


for trial, (n_keep, p_active) in enumerate(((0, 0.0), (40, 0.5), (300, 0.9), (300, 0.05))):
    mean_vector = torch.rand(D, device=dev) * (torch.rand(D, device=dev) < 0.2)
    keep_tensor = torch.randperm(D, device=dev)[:n_keep].sort().values if n_keep else None
    kept_values = None
    if n_keep:
        kept_values = torch.rand(2, 5, n_keep, device=dev) * (torch.rand(2, 5, n_keep, device=dev) < p_active)
    allz = torch.zeros(2, 5, D, device=dev)
    out = p._respect_topk_fill(allz, mean_vector, keep_tensor, kept_values)
    print("  trial %d (members %d, active frac %.2f): %s" % (trial, n_keep, p_active, audit_fill(out, mean_vector, keep_tensor, kept_values, K)))


# ------------------------------------------------------------------ per-circuit checks
class AuditCO(TapCO):
    log = []

    def _respect_topk_fill(self, all_latents, mean_vector, keep_tensor, kept_values):
        out = super()._respect_topk_fill(all_latents, mean_vector, keep_tensor, kept_values)
        AuditCO.log.append(audit_fill(out, mean_vector, keep_tensor, kept_values, self.topk))
        return out


for c in items:
    seed = V.seed_of(c); sf = seed.metadata["feature_id"]; layer, kind, sl = sf.layer, sf.kind, sf.index
    key = V.key_of(c)
    print("\n" + "=" * 90 + "\nCIRCUIT %s" % key, flush=True)
    row = V.score_circuit(c, pool)
    last = G["last"]; attr, msets = last["attr"], last["msets"]
    pd_ = M0.build_probe_dataset(comp_of(layer, KINDS.index(kind), NK), sl)
    pt, pa = pd_.pos_tokens[:64], pd_.pos_argmax[:64]
    n_tr = V.split_n(int(pt.shape[0]))
    pt_tr, pa_tr, pt_ho, pa_ho = pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]
    sae = bank.saes[kind][layer]; w_seed = sae.encoder.weight[sl].detach(); b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind)); UPS = set(UP); site = (layer, kind)
    SK = dict(w_seed=w_seed, b_seed=b_seed)
    alphas = {}
    for n in c.nodes.values():
        if n is not seed:
            f = n.metadata["feature_id"]; alphas.setdefault((f.layer, f.kind), {})[int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    a_pos = V.read(lambda a, b: V.AmpInjectPatcher({}, site, w_seed, b_seed, sl), pt_ho, pa_ho)
    print("a_pos held-out: pre %.4f tk %.4f (row a_pos_tk %.4f)" % (a_pos["pre"], a_pos["tk"], row["a_pos_tk"]))

    # ---- G scope: sites with no member are ablated too (DAN-67)
    print("\n=== G. scope: every upstream site ablated, members or not ===")
    one = next(iter(sorted(alphas))) if alphas else None
    if one is None or len(UP) < 2:
        print("  skipped (needs members and >= 2 upstream sites)")
    else:
        keep_one = {one: set(alphas[one])}
        kt_one = {one: torch.tensor(sorted(alphas[one]), device=pt.device, dtype=torch.long)}
        r_one = V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_one, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                          seed_latent_idx=sl, pos_argmax=pa_ho[a:b], keep_tensors=kt_one, **SK), pt_ho, pa_ho)
        keep_rest = dict(keep_one); kt_rest = dict(kt_one)
        for s in UP:
            if s != one:
                keep_rest[s] = set(range(D)); kt_rest[s] = torch.arange(D, device=pt.device)
        r_rest = V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_rest, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                           seed_latent_idx=sl, pos_argmax=pa_ho[a:b], keep_tensors=kt_rest, **SK), pt_ho, pa_ho)
        print("  members only at %s: sites edited %d of %d upstream | seed pre %.4f tk %.4f" % (one, len(r_one["edited"]), len(UP), r_one["pre"], r_one["tk"]))
        print("  same, other sites kept whole (the old bug's behaviour): seed pre %.4f tk %.4f" % (r_rest["pre"], r_rest["tk"]))
        print("  %s | row all_sites_ablated = %s" % ("ok: other sites ablated" if r_one["edited"] == UPS and abs(r_one["pre"] - r_rest["pre"]) > 1e-6
                                                     else "CHECK: scope looks wrong", row.get("all_sites_ablated")))

    # ---- B identity
    print("\n=== B. identity: every latent a member at alpha = 1 ===")
    all_idx = torch.arange(D, device=pt.device)
    keep_all = {s: set(range(D)) for s in UP}
    kt_all = {s: all_idx for s in UP}
    r = V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_all, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                                  pos_argmax=pa_ho[a:b], keep_tensors=kt_all, **SK), pt_ho, pa_ho)
    print("  engine CircuitOnlyPatcher, zero fill : pre %.4f tk %.4f | F0 = %.5f (tk %.5f)" % (r["pre"], r["tk"], r["pre"] / a_pos["pre"], r["tk"] / a_pos["tk"]))
    r = V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_all, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                                  pos_argmax=pa_ho[a:b], keep_tensors=kt_all, site_means={s: torch.rand(D, device=pt.device) for s in UP},
                                  respect_topk=True, topk=K, **SK), pt_ho, pa_ho)
    print("  engine, topk fill (random means)     : pre %.4f tk %.4f | ratio %.5f  (all members -> nothing to fill)" % (r["pre"], r["tk"], r["pre"] / a_pos["pre"]))

    # phi_pin with all latents pinned to their clean values
    cap = {}
    k2i = {k: i for i, k in enumerate(KINDS)}

    def hook(layer_idx, activations):
        for kd in KINDS:
            s = (layer_idx, kd)
            if s in UPS:
                ta, ti = bank.encode(activations[k2i[kd]], kd, layer_idx)
                cap[s] = sparse_topk_to_dense(ta, ti, D, dtype=torch.float32)
    inference.disable_compile()
    with torch.no_grad():
        inference.forward(pt_ho, activations_callback=hook, return_activations=False, tokenize_final=False)
    inference.enable_compile()
    pins_all = V.LazyPins(cap, kt_all, tuple(pt_ho.shape[:2]))
    r = V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep_all, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                                  pos_argmax=pa_ho[a:b], keep_tensors=kt_all, pin_values=pins_all, **SK), pt_ho, pa_ho)
    print("  phi_pin, all latents pinned (zero)   : pre %.4f tk %.4f | ratio %.5f (tk %.5f)" % (r["pre"], r["tk"], r["pre"] / a_pos["pre"], r["tk"] / a_pos["tk"]))

    # ---- F LazyPins vs the engine's own position-specific pins (members only, alpha = 1 and fitted alpha)
    print("\n=== F. LazyPins (member columns) vs engine position-specific pins ===")
    keep = {s: set(d) for s, d in alphas.items()}
    kt = {s: torch.tensor(sorted(d), device=pt.device, dtype=torch.long) for s, d in alphas.items()}
    scales = {}
    for s, d in alphas.items():
        sv = torch.ones(D, device=pt.device); sv[kt[s]] = torch.tensor([d[int(i)] for i in kt[s].tolist()], device=pt.device); scales[s] = sv
    _, pins_eng = collect_site_anchors(inference, bank, pt_ho, UPS, pa_ho, pin_position_specific=True)
    pins_lazy = V.LazyPins({s: cap[s][..., kt[s]] for s in kt}, kt, tuple(pt_ho.shape[:2]))
    for label, sc_ in (("alpha = 1", None), ("fitted alpha", scales)):
        for fill, mm, tk_ in (("zero", None, False),):
            out = []
            for pins in (pins_eng, pins_lazy):
                out.append(V.read(lambda a, b: TapCO(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=layer, seed_kind=kind,
                                                     seed_latent_idx=sl, pos_argmax=pa_ho[a:b], keep_tensors=kt, pin_values=pins,
                                                     site_means=mm, respect_topk=tk_, topk=K, keep_scales=sc_, **SK), pt_ho, pa_ho))
            print("  %-12s %s fill: engine pins pre %.4f tk %.4f | LazyPins pre %.4f tk %.4f" % (label, fill, out[0]["pre"], out[0]["tk"], out[1]["pre"], out[1]["tk"]))
    print("  row: phi_pin_alpha_zero_raw pre %.4f tk %.4f | free0_raw pre %.4f tk %.4f" % (row["phi_pin_alpha_zero_raw_pre"], row["phi_pin_alpha_zero_raw_tk"], row["free0_raw_pre"], row["free0_raw_tk"]))
    del pins_eng, cap

    # ---- C audit on the real run
    print("\n=== C. Top-K fill audit on the real freeM_topk run (every upstream site) ===")
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    AuditCO.log = []
    r = V.read(lambda a, b: AuditCO(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                                    pos_argmax=pa_ho[a:b], keep_tensors=kt, site_means=means_tr, respect_topk=True, topk=K,
                                    keep_scales=scales, **SK), pt_ho, pa_ho)
    agg = {k_: sum(d[k_] for d in AuditCO.log) for k_ in ("member_value_changed", "nonmember_value_mismatch", "fill_count_mismatch", "fill_slots_with_zero_mean")}
    print("  sites audited %d | %s | budget range %d..%d | freeM_topk raw pre %.4f (row %.4f)"
          % (len(AuditCO.log), agg, min(d["budget_min"] for d in AuditCO.log), max(d["budget_max"] for d in AuditCO.log), r["pre"], row["freeM_topk_raw_pre"]))
    nz = {s: int((means_tr[s] > 0).sum()) for s in UP}
    print("  posctx means with > 0 mass per site: min %d median %d (k = %d)" % (min(nz.values()), sorted(nz.values())[len(nz) // 2], K))

    # ---- E gradient roles vs finite differences / single-member ablation (TRAIN slice, natural stream)
    print("\n=== E. role attribution vs finite difference and single-member ablation (train slice) ===")
    flat = [(float(a), s, int(i)) for s in msets for a, i in zip(attr[s].tolist(), msets[s].tolist())]
    flat.sort(key=lambda t: t[0])
    picks = flat[:4] + flat[-4:] + flat[len(flat) // 2 - 1:len(flat) // 2 + 1]
    nat_pre, _ = V.read(lambda a, b: V.AmpInjectPatcher({}, site, w_seed, b_seed, sl), pt_tr, pa_tr, per_seq=True)
    f1 = float(nat_pre.sum())
    print("  natural sum of seed pre over %d train contexts: %.3f" % (n_tr, f1))
    print("  %-18s %12s %12s %12s %14s" % ("member", "grad x act", "FD slope", "abl f(0)-f(1)", "sign agree FD/abl"))
    for a_, s, i in picks:
        idx = torch.tensor([i], device=pt.device)
        vals = {}
        for w in (0.0, 0.75, 1.25):
            sc = {s: (idx, torch.tensor([w], device=pt.device))}
            pre_w, _ = V.read(lambda a, b: V.LiveScalePatcher(sc, site, w_seed, b_seed, sl), pt_tr, pa_tr, per_seq=True)
            vals[w] = float(pre_w.sum())
        fd = (vals[1.25] - vals[0.75]) / 0.5
        ab = vals[0.0] - f1
        print("  %-18s %12.4f %12.4f %12.4f %8s / %s" % ("%d.%s.%d" % (s[0], s[1], i), a_, fd, ab,
                                                         "ok" if (a_ >= 0) == (fd >= 0) else "DIFF", "ok" if (a_ >= 0) == (ab <= 0) else "DIFF"))
print("\nchecks complete")
