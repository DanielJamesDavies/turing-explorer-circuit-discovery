"""090: does training WCM's A and C terms under the SCORER's sparsity-preserving fill lift the pass-rule A / C?

Production trains A / C with the dense mean fill (every non-member at its mean, no Top-K); the pass rule scores them
with the sparsity-preserving fill (CircuitOnlyPatcher respect_topk=True, topk=K: members at alpha x live, the top
(K - #active members) non-members by mean at their mean, the rest 0). sp_engine.py patches the engine in memory so
the A and C training passes can use that fill (src untouched). Arms, everything else production (gamma_Z 1,
gamma_A = gamma_C 0.25, rank-keep 3e-3, lambda 1e-3, 400 steps, protocol-v1 contexts / split / AdamW):

  dense       production fill (the in-run baseline, refit on this GPU)
  sp_hard     sparsity-preserving fill, hard keep mask (gradients only through kept entries)
  sp_ste      sparsity-preserving fill, straight-through backward
  curriculum  dense for steps 1-200, sp_hard for steps 201-400

Scoring as 089: driver.Runner.score (the production scorer) on held-out strongest, and the same scorer on the train
probe (089 score_train). Each circuit also gets an eviction diagnostic under the rule's A fill on the held-out anchors
(evict_stats). The production circuit itself is rescored as arm "prod".

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/090-sparse-fill-training/run.py
  env: ARMS (comma list, default dense,sp_hard,sp_ste,curriculum)  OUT90 (default experiments/090-.../out)
       STEPS (driver: smoke only)  LIMIT (first N targets)  TARGETS (key file, default targets.txt)
       CHECK=1 (real-model equality check of the training fill against the scorer on the first target, then exit)
       SKIP_PROD=1 (do not rescore the production circuits)
"""
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
os.environ.setdefault("OUT", str(EXP / "062-h100-protocol-v1" / "out_full" / "out"))
os.environ.setdefault("CURVE_STRIDE", "20")
sys.path.insert(0, str(EXP / "062-h100-protocol-v1"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sp_engine  # noqa: E402

OUT90 = Path(os.environ.get("OUT90", str(HERE / "out")))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets.txt")))
ARMS = {"dense": "dense", "sp_hard": "hard", "sp_ste": "ste", "curriculum": "curriculum"}
WANT = [a for a in os.environ.get("ARMS", ",".join(ARMS)).split(",") if a]
SWITCH = 200
GAMMA, LAM = 0.25, 1e-3


# ------------------------------------------------------------------------------------------------ helpers (089)
def alpha_summary(c):
    a = np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"])
    if a.size == 0:
        return dict(alpha_med=None, alpha_p90=None, alpha_max=None, alpha_min=None)
    return dict(alpha_med=float(np.median(a)), alpha_p90=float(np.percentile(a, 90)), alpha_max=float(a.max()),
                alpha_min=float(a.min()))


KEEP = ("n", "a_pos_tk", "a_pos_pre", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre",
        "freeM_topk_pre", "freeN_topk_pre", "phi_sup_blind_pre", "freeM_dense_tk", "freeN_dense_tk", "freeM_dense_pre",
        "freeN_dense_pre", "free0_tr_tk", "free0_a1_tk", "phi_cf_alpha_blind_tk", "vacuous_tk", "alpha_median",
        "alpha_p90", "n_alpha_off", "e0_tk", "eM_topk_tk", "eN_topk_tk", "eM_dense_tk", "eN_dense_tk")


def passes(r, dense=False):
    names = ("free0_tk", "freeM_dense_tk", "freeN_dense_tk") if dense else ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")
    v = [r.get(k) for k in names]
    nec = r.get("phi_sup_blind_tk")
    if None in v or nec is None:
        return False
    return all(0.8 <= x <= 1.5 for x in v) and nec >= 0.9


def score_train(R, c, rec):
    """089's train probe: the production scorer with reference = the 48 strongest-train rows and scored slice = the
    fit's strongest training rows (D_train; arm A: strongest-train)."""
    import circuit.protocol_contexts as PC
    V, G = R.V, R.G
    arm, _ = R.train_arm(rec)
    ref = PC.pick(rec, "strong", rec["strong"]["train"])
    tr_idx = rec["D_train"] if arm == "B" else rec["strong"]["train"]
    trn = PC.pick(rec, "strong", tr_idx)
    pos, tgt, arg = PC.cat([ref, trn])
    dev = R.dev
    pd_ = PC.ProbeDataset(pos_tokens=pos.to(dev), target_tokens=tgt.to(dev), neg_tokens=rec["neg"].to(dev),
                          pos_argmax=arg.to(dev), metadata=dict(n_train=int(ref[0].shape[0]), n=int(pos.shape[0])))
    G["M0"].build_probe_dataset = lambda comp, i, _p=pd_: _p
    G["M0"]._neg_context_selector = lambda _s=PC.FixedSelector(pd_.neg_tokens): _s
    old = (V.N_SEQ, V.EVAL_SPLIT)
    V.N_SEQ, V.EVAL_SPLIT = int(pos.shape[0]), "explicit"
    try:
        row = V.score_circuit(c, {}, skip_roles=True)
    finally:
        V.N_SEQ, V.EVAL_SPLIT = old
    row["n_train_scored"] = int(trn[0].shape[0])
    return row


def slim(r, prefix):
    return {prefix + k: r.get(k) for k in KEEP if k in r}


def circuit_alphas(c):
    out = defaultdict(dict)
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    return dict(out)


def eval_slices(R, rec):
    """The scorer's held-out slices exactly as amp_eval_pass_v2.score_circuit takes them (held 'strong')."""
    V = R.V
    pd_ = R.H.patch_eval_contexts(R.G, rec, "strong")
    pt, pa = pd_.pos_tokens[:V.N_SEQ], pd_.pos_argmax[:V.N_SEQ]
    n_pos = int(pt.shape[0])
    n_tr = V.split_n(n_pos)
    if V.EVAL_SPLIT == "explicit" and (pd_.metadata or {}).get("n_train"):
        n_tr = int(pd_.metadata["n_train"])
    return pt[:n_tr], pa[:n_tr], pt[n_tr:], pa[n_tr:]


# ------------------------------------------------------------------------------------------------ eviction diagnostic
@torch.no_grad()
def evict_stats(R, c, rec):
    """Under the rule's A fill (respect_topk, means over the 48 strongest-train reference rows) on the held-out
    anchors, summed over every upstream site and anchor:
      act        members active (nonzero alpha x live value)
      over_k     (site, anchor) pairs with more than K active members (no mean fill there at all)
      lit_evict  active members a LITERAL Top-K over {members' values} + {non-members' means} would drop (value below
                 the K-th largest); the scorer itself never evicts a member
      budget     mean fill budget K - act per (site, anchor)."""
    from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
    from eval.floors import collect_site_anchors
    G = R.G
    inference, bank, K, D = G["inference"], G["bank"], int(G["K"]), int(G["D"])
    seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
    sf = seed.metadata["feature_id"]
    UPS = set(upstream_sites(bank, sf.layer, sf.kind))
    al = {s: d for s, d in circuit_alphas(c).items() if s in UPS and d}
    pt_tr, pa_tr, pt_ho, pa_ho = eval_slices(R, rec)
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    dev = pt_ho.device
    keep = {s: set(d) for s, d in al.items()}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items()}
    scales = {}
    for s, v in kt.items():
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        sv[v] = torch.tensor([al[s][int(i)] for i in v.tolist()], device=dev, dtype=torch.float32)
        scales[s] = sv
    acc = dict(act=0, over_k=0, lit_evict=0, pairs=0, budget_sum=0.0)

    class Probe(CircuitOnlyPatcher):
        def _respect_topk_fill(self, all_latents, mean_vector, keep_tensor, kept_values):
            patched = super()._respect_topk_fill(all_latents, mean_vector, keep_tensor, kept_values)
            Bx, Tx = all_latents.shape[:2]
            nb = min(Bx, self.pos_argmax.shape[0])
            rr = torch.arange(nb, device=all_latents.device)
            anc = self.pos_argmax[:nb].to(all_latents.device).clamp(0, Tx - 1)
            acc["pairs"] += nb
            if keep_tensor is None:
                acc["budget_sum"] += float(K) * nb
                return patched
            kv = patched[rr, anc][:, keep_tensor].float()                    # [nb, n_members] at the anchors
            a = (kv != 0)
            n_act = a.sum(-1)
            acc["act"] += int(n_act.sum())
            acc["over_k"] += int((n_act > K).sum())
            acc["budget_sum"] += float((K - n_act).clamp(min=0).sum())
            nonm = mean_vector.float().clone()
            nonm[keep_tensor] = float("-inf")
            union = torch.cat([kv.masked_fill(~a, float("-inf")), nonm.view(1, -1).expand(nb, -1)], dim=1)
            kth = union.topk(K, dim=-1).values[:, -1:]
            acc["lit_evict"] += int((a & (kv < kth)).sum())
            return patched

    inference.disable_compile()
    try:
        for s0 in range(0, int(pt_ho.shape[0]), 16):
            tk = pt_ho[s0:s0 + 16]
            p = Probe(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
                      pos_argmax=pa_ho[s0:s0 + int(tk.shape[0])], site_means=means_tr, respect_topk=True, topk=K,
                      keep_tensors=kt, keep_scales=scales)
            inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
    pairs = max(acc["pairs"], 1)
    return dict(ev_act_per_anchor=acc["act"] / pairs, ev_over_k_share=acc["over_k"] / pairs,
                ev_lit_evict_share=(acc["lit_evict"] / acc["act"]) if acc["act"] else None,
                ev_budget_mean=acc["budget_sum"] / pairs, ev_pairs=acc["pairs"])


# ------------------------------------------------------------------------------------------------ real-model check
@torch.no_grad()
def real_check(R, mod, key):
    """On the real model and bank: the engine's training patcher (binary gates on the production circuit's members,
    its alphas, ablation value = the scorer's A means) vs the scorer's CircuitOnlyPatcher, every upstream site
    patched, read = relu(seed pre-activation) at the held-out anchors. hard must match respect_topk, dense must match
    the dense fill, and the two fills must differ."""
    from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
    from eval.floors import collect_site_anchors
    G = R.G
    inference, bank, K, D = G["inference"], G["bank"], int(G["K"]), int(G["D"])
    rec = R.contexts(key)
    c = torch.load(R_OUT_MAIN() / "circuits" / ("%s.pt" % key), weights_only=False)
    seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
    sf = seed.metadata["feature_id"]
    layer, kind, sl = sf.layer, sf.kind, sf.index
    UPS = set(upstream_sites(bank, layer, kind))
    al = {s: d for s, d in circuit_alphas(c).items() if s in UPS and d}
    pt_tr, pa_tr, pt_ho, pa_ho = eval_slices(R, rec)
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
    dev = pt_ho.device
    sae = bank.saes[kind][layer]
    w_seed, b_seed = sae.encoder.weight[sl].detach(), sae._get_bias_eff()[sl].detach()
    thetas, amps, scales = {}, {}, {}
    for s in UPS:
        th = torch.full((D,), -200.0, device=dev)
        ps = torch.full((D,), float(np.log(np.expm1(1.0))), device=dev)
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        for i, a in al.get(s, {}).items():
            th[i] = 200.0
            ps[i] = float(np.log(np.expm1(max(a, 1e-4))))
        thetas[s], amps[s] = th, ps
        sv_eff = mod._amp_of(ps)
        for i in al.get(s, {}):
            sv[i] = float(sv_eff[i])
        scales[s] = sv
    keep = {s: set(d) for s, d in al.items()}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items()}
    rr = torch.arange(int(pt_ho.shape[0]), device=dev)

    def scorer(topk):
        class Tap(CircuitOnlyPatcher):
            def transform(self, li, kd, x):
                if (li, kd) == (layer, kind):
                    self.pre = x @ w_seed.to(x.dtype) + b_seed.to(x.dtype)
                return super().transform(li, kd, x)
        out = []
        inference.disable_compile()
        try:
            for s0 in range(0, int(pt_ho.shape[0]), 16):
                tk = pt_ho[s0:s0 + 16]
                p = Tap(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
                        pos_argmax=pa_ho[s0:s0 + int(tk.shape[0])], site_means=means_tr, respect_topk=topk, topk=K,
                        keep_tensors=kt, keep_scales=scales)
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                out.append(p.pre)
        finally:
            inference.enable_compile()
        pre = torch.cat(out)
        return torch.relu(pre[rr, pa_ho.to(dev)]).float()

    def engine(mode):
        sp_engine.set_mode(mod, mode)
        p = mod.LearnedMaskPatcher(bank, thetas, layer, kind, w_seed, b_seed, amps=amps, code_dtype="stream",
                                   floors=means_tr, binarize="anneal", bin_threshold=0.5)
        p.temperature = 0.05
        out = []
        for s0 in range(0, int(pt_ho.shape[0]), 16):
            out.append(mod._forward_preact(inference, p, pt_ho[s0:s0 + 16], grad=False))
        sp_engine.set_mode(mod, "dense")
        pre = torch.cat(out)
        return torch.relu(pre[rr, pa_ho.to(dev)]).float()

    s_tk, s_dn, e_hd, e_st, e_dn = scorer(True), scorer(False), engine("hard"), engine("ste"), engine("dense")
    scale = float(s_tk.abs().mean().clamp(min=1e-6))
    res = dict(key=key, n_members=sum(len(d) for d in al.values()), mean_scorer_topk=float(s_tk.mean()),
               mean_scorer_dense=float(s_dn.mean()),
               maxdiff_hard_vs_topk=float((e_hd - s_tk).abs().max()), maxdiff_ste_vs_topk=float((e_st - s_tk).abs().max()),
               maxdiff_dense_vs_dense=float((e_dn - s_dn).abs().max()),
               maxdiff_topk_vs_dense=float((s_tk - s_dn).abs().max()), scale=scale)
    print("090 CHECK %s" % json.dumps(res), flush=True)
    return res


def R_OUT_MAIN():
    import driver
    return driver.OUT / "main"


# ------------------------------------------------------------------------------------------------ main
def main():
    import driver
    keys = [t for t in TARGETS.read_text().split() if t][:LIMIT]
    groups = {}
    tj = HERE / "targets.json"
    if tj.exists():
        g = json.loads(tj.read_text())
        for name in ("fail89", "attn", "control"):
            for k in g.get(name, []):
                groups[k] = name
        sub = g.get("attn_sub", {})
    else:
        sub = {}
    mod = sp_engine.install()
    R = driver.Runner()
    assert sys.modules[sp_engine.MODNAME] is mod
    if os.environ.get("CHECK") == "1":
        OUT90.mkdir(parents=True, exist_ok=True)
        with open(OUT90 / "check.jsonl", "a") as fh:
            for key in keys:
                fh.write(json.dumps(real_check(R, mod, key)) + "\n")
        return
    OUT90.mkdir(parents=True, exist_ok=True)
    rows_path = OUT90 / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in open(rows_path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["seed"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    fh, fr = open(rows_path, "a"), open(OUT90 / "train.jsonl", "a")
    t0 = time.time()
    print("090: %d targets, arms %s, steps %d, out %s" % (len(keys), WANT, driver.STEPS, OUT90), flush=True)

    def write(row):
        fh.write(json.dumps(row) + "\n"); fh.flush()
        f = lambda x: "%.2f" % x if isinstance(x, (int, float)) and x is not None else "-"
        print("[%5.0fs] %-10s %-16s n %s | ho Z %s A %s C %s nec %s pass %s | dense A %s C %s | tr Z %s A %s C %s |"
              " evict %s overK %s | %s" % (
                  time.time() - t0, row["arm"], row["seed"], row.get("ho_n"), f(row.get("ho_free0_tk")),
                  f(row.get("ho_freeM_topk_tk")), f(row.get("ho_freeN_topk_tk")), f(row.get("ho_phi_sup_blind_tk")),
                  row.get("ho_pass"), f(row.get("ho_freeM_dense_tk")), f(row.get("ho_freeN_dense_tk")),
                  f(row.get("tr_free0_tk")), f(row.get("tr_freeM_topk_tk")), f(row.get("tr_freeN_topk_tk")),
                  f(row.get("ev_lit_evict_share")), f(row.get("ev_over_k_share")), row.get("error", "")), flush=True)

    def score_all(c, rec, base):
        ho = R.score(c, rec, "strong")
        tr = score_train(R, c, rec)
        row = dict(base, **slim(ho, "ho_"), **slim(tr, "tr_"), ho_pass=passes(ho), tr_pass=passes(tr),
                   ho_dpass=passes(ho, True), tr_dpass=passes(tr, True), tr_n_scored=tr.get("n_train_scored"),
                   **alpha_summary(c))
        try:
            row.update(evict_stats(R, c, rec))
        except Exception as e:  # noqa: BLE001
            row["ev_error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        return row

    def base_of(key, arm):
        return dict(seed=key, arm=arm, group=groups.get(key), attn_sub=sub.get(key), layer=int(key.split(".")[0]),
                    kind=key.split(".")[1])

    if os.environ.get("SKIP_PROD") != "1":
        for key in keys:
            if (key, "prod") in done:
                continue
            try:
                rec = R.contexts(key)
                c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
                write(score_all(c, rec, base_of(key, "prod")))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                write(dict(base_of(key, "prod"), error="%s: %s" % (type(e).__name__, str(e)[:300])))

    M = R.method(GAMMA, LAM, True)
    for arm in WANT:
        mode = ARMS[arm]
        for key in keys:
            if (key, arm) in done:
                continue
            base = dict(base_of(key, arm), fill=mode, switch=SWITCH if mode == "curriculum" else None,
                        gamma_AC=GAMMA, lam=LAM, rank_weight=driver.RANK, steps=driver.STEPS)
            try:
                rec = R.contexts(key)
                tr_arm, _ = R.train_arm(rec)
                ref = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
                sp_engine.set_mode(mod, mode, switch=SWITCH)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                c, t_fit = R.fit(M, rec, key, tr_arm, OUT90 / "circuits" / arm / ("%s.pt" % key), train_fh=fr,
                                 label=dict(arm90=arm, fill=mode))
                peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
                calls = mod.SPFILL.get("stats", {}).get("calls", 0)
                sp_engine.set_mode(mod, "dense")
                if c is None:
                    write(dict(base, skip="rejected"))
                    continue
                row = score_all(c, rec, base)
                row.update(t_fit=round(t_fit, 1), n_ref=driver.n_nodes(ref), sp_calls=calls, fit_peak_mb=peak)
                write(row)
            except Exception as e:  # noqa: BLE001
                sp_engine.set_mode(mod, "dense")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write(dict(base, error="%s: %s" % (type(e).__name__, str(e)[:300])))
        print("090: arm %s finished %.0fs" % (arm, time.time() - t0), flush=True)
    print("090: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
