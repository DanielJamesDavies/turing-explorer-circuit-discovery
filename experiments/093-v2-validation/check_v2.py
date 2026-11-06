"""093 checks of the protocol-v2 engine option (src learned_mask train_fill / sparse_grad, train_means_source) on the
real model and bank, before the validation run.

  CHECK=repro  v2 with train_means_source="fit" (= 090's sp_ste arm: sparse fill, straight-through, the engine's own
               means) refitted at 090's settings (400 steps, gamma 0.25, lambda 1e-3, rank-keep 3e-3, the 062 15k
               as-run contexts, as_run eval split) and compared with 090's sp_ste circuit: members, every node's
               metadata (alpha), and the held-out-strongest scores.
  CHECK=means  train_means_source="eval_ref": the means the hop computes (AblationGradientDiscovery._eval_ref_means on
               protocol_contexts.eval_reference) against the means the production scorer computes inside
               Runner.score (amp_eval_pass_v2: means_tr, means_neg), bit for bit; then 090's site check with src: at
               every upstream site, the engine's training code (train_fill="sparse", binary gates on the production
               circuit's members and alphas, floor = those A means) against the scorer's respect_topk code.

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 CHECK=repro .venv/bin/python -u \
         experiments/093-v2-validation/check_v2.py 2.mlp.4180 6.mlp.23423
  Writes out_check/<check>.jsonl (one row per target) and out_check/repro/<key>.pt.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E62 = EXP / "062-h100-protocol-v1"
E90 = EXP / "090-sparse-fill-training"
os.environ.setdefault("OUT", str(E62 / "out_full" / "out"))      # the 15k run: contexts + production circuits (read)
os.environ.setdefault("CURVE_STRIDE", "20")
os.environ.pop("WCM_VERSION", None)                                 # v1 Runner; the check sets the engine fields itself
sys.path.insert(0, str(E62))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUTC = HERE / "out_check"
CHECK = os.environ.get("CHECK", "repro")
GAMMA, LAM = 0.25, 1e-3


def node_map(c):
    out = {}
    for n in c.nodes.values():
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind), int(f.index))] = {k: v for k, v in n.metadata.items() if k != "feature_id"}
    return out


def alphas_of(c):
    out = defaultdict(dict)
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out[(int(f.layer), str(f.kind))][int(f.index)] = float(n.metadata.get("amplitude", 1.0))
    return dict(out)


def set_engine(train_fill, sparse_grad, means):
    from config import config
    lm = config.discovery.learned_mask
    lm.train_fill, lm.sparse_grad, lm.train_means_source = train_fill, sparse_grad, means


SCORE_KEYS = ("n", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "freeM_dense_tk", "freeN_dense_tk",
              "free0_pre", "freeM_topk_pre", "freeN_topk_pre", "e0_tk", "eM_topk_tk", "eN_topk_tk", "a_pos_tk")


def repro(R, key):
    import driver
    rec = R.contexts(key)
    arm, _ = R.train_arm(rec)
    M = R.method(GAMMA, LAM, True)
    set_engine("sparse", "ste", "fit")
    path = OUTC / "repro" / ("%s.pt" % key)
    with open(OUTC / "repro_train.jsonl", "a") as fr:
        c, t_fit = R.fit(M, rec, key, arm, path, train_fh=fr, label=dict(check="repro", train_fill="sparse"))
    prov = M.last_mask_provenance or {}
    set_engine("dense", "ste", "fit")
    ref = torch.load(E90 / "out" / "circuits" / "sp_ste" / ("%s.pt" % key), weights_only=False)
    a, b = node_map(c), node_map(ref)
    ho = R.score(c, rec, "strong")
    r90 = [json.loads(l) for l in open(E90 / "out" / "rows.jsonl")]
    r90 = [r for r in r90 if r.get("seed") == key and r.get("arm") == "sp_ste" and "error" not in r][-1]
    scores = {k: (ho.get(k), r90.get("ho_" + k)) for k in SCORE_KEYS if ("ho_" + k) in r90}
    res = dict(key=key, n=driver.n_nodes(c), n_090=driver.n_nodes(ref), t_fit=round(t_fit, 1),
               sparse_fill_calls=prov.get("sparse_fill_calls"), train_fill=prov.get("train_fill"),
               members_identical=set(a) == set(b), nodes_metadata_identical=a == b,
               alpha_maxdiff=max((abs(float(a[k].get("amplitude", 1)) - float(b[k].get("amplitude", 1)))
                                  for k in set(a) & set(b)), default=None),
               scores_identical=all(x == y for x, y in scores.values()),
               score_maxdiff=max(abs(x - y) for x, y in scores.values() if x is not None and y is not None),
               scores=scores, ho_pass_090=r90.get("ho_pass"))
    print("093 REPRO %s" % json.dumps({k: v for k, v in res.items() if k != "scores"}), flush=True)
    return res


@torch.no_grad()
def means_and_sites(R, key):
    """eval_ref means vs the production scorer's means (bitwise), then the real-model site check of the fill."""
    import driver
    import eval.floors as F
    from circuit.instrument import learned_mask as LM
    from circuit.protocol_contexts import eval_reference
    from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
    G = R.G
    inference, bank, K, D = G["inference"], G["bank"], int(G["K"]), int(G["D"])
    rec = R.contexts(key)
    c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
    seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
    sf = seed.metadata["feature_id"]
    layer, kind, sl = sf.layer, sf.kind, sf.index
    kinds = list(bank.kinds)
    sites = sorted(upstream_sites(bank, layer, kind), key=lambda s: (s[0], kinds.index(s[1])))
    UPS = set(sites)

    # (1) the scorer's means, captured from inside Runner.score (amp_eval_pass_v2 imports them at call time)
    cap = {}
    o_anc, o_mean = F.collect_site_anchors, F.collect_site_means

    def anc(inf_, bank_, tokens, sites_, argmax=None, *a, **k):
        out = o_anc(inf_, bank_, tokens, sites_, argmax, *a, **k)
        if argmax is not None and "A" not in cap:
            cap["A"], cap["A_rows"] = out[0], tokens
        return out

    def mean(inf_, bank_, tokens, sites_, *a, **k):
        out = o_mean(inf_, bank_, tokens, sites_, *a, **k)
        if "C" not in cap:
            cap["C"], cap["C_rows"] = out, tokens
        return out
    F.collect_site_anchors, F.collect_site_means = anc, mean
    try:
        R.score(c, rec, "strong")
    finally:
        F.collect_site_anchors, F.collect_site_means = o_anc, o_mean
    # (2) the hop's eval_ref means
    M = R.method(GAMMA, LAM, True)
    M._train_means_ref = eval_reference(rec, R.dev, split=R.V.EVAL_SPLIT)
    from config import config
    got = M._eval_ref_means(sites, config.discovery.learned_mask)
    eqA = all(torch.equal(got["pos_floor_means"][s], cap["A"][s]) for s in sites)
    eqC = all(torch.equal(got["floor_means"][s], cap["C"][s]) for s in sites)
    rows_eq = (torch.equal(M._train_means_ref[0], cap["A_rows"]), torch.equal(M._train_means_ref[1], cap["C_rows"]))
    # the v1 fit means differ from the scorer's A (arm B trains on 32 strongest + 16 mid) and equal its C
    from circuit.protocol_contexts import training_probe
    pd_ = training_probe(rec, R.dev, R.train_arm(rec)[0])
    fitA = o_mean(inference, bank, pd_.pos_tokens[:48], UPS)
    fitC = o_mean(inference, bank, pd_.neg_tokens[:48], UPS)
    fitA_eq = all(torch.equal(fitA[s], cap["A"][s]) for s in sites)
    fitC_eq = all(torch.equal(fitC[s], cap["C"][s]) for s in sites)

    # (3) the site check: engine training code (sparse) vs the scorer's respect_topk code at every upstream site
    al = {s: d for s, d in alphas_of(c).items() if s in UPS and d}
    means_tr = got["pos_floor_means"]
    pd_ev = R.H.patch_eval_contexts(G, rec, "strong")
    n_tr = int(M._train_means_ref[0].shape[0])
    pt_ho, pa_ho = pd_ev.pos_tokens[n_tr:64], pd_ev.pos_argmax[n_tr:64]
    dev = pt_ho.device
    sae = bank.saes[kind][layer]
    w_seed, b_seed = sae.encoder.weight[sl].detach(), sae._get_bias_eff()[sl].detach()
    thetas, amps, scales = {}, {}, {}
    for s in UPS:
        th = torch.full((D,), -200.0, device=dev)
        ps = torch.full((D,), float(np.log(np.expm1(1.0))), device=dev)
        for i, a in al.get(s, {}).items():
            th[i] = 200.0
            ps[i] = float(np.log(np.expm1(max(a, 1e-4))))
        thetas[s], amps[s] = th, ps
        sv = torch.ones(D, device=dev, dtype=torch.float32)
        eff = LM._amp_of(ps)
        for i in al.get(s, {}):
            sv[i] = float(eff[i])
        scales[s] = sv
    keep = {s: set(d) for s, d in al.items()}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items()}
    out = {}
    for grad in ("ste", "hard"):
        eng = LM.LearnedMaskPatcher(bank, thetas, layer, kind, w_seed, b_seed, amps=amps, code_dtype="stream",
                                    floors=means_tr, binarize="anneal", bin_threshold=0.5, train_fill="sparse",
                                    sparse_grad=grad)
        eng.temperature = 0.05
        cc = {}
        o_sc = LM.sparse_code

        def sc_cap(code, H, g):
            y = o_sc(code, H, g)
            cc["eng"] = y.detach()
            return y
        LM.sparse_code = sc_cap
        stats = []

        class Cmp(CircuitOnlyPatcher):
            def _respect_topk_fill(self, all_latents, mean_vector, keep_tensor, kept_values):
                y = super()._respect_topk_fill(all_latents, mean_vector, keep_tensor, kept_values)
                cc["sc"] = y.detach()
                return y

            def transform(self, li, kd, x):
                if (li, kd) not in self.in_scope:
                    return super().transform(li, kd, x)
                cc.clear()
                ys = super().transform(li, kd, x)
                ye = eng.transform(li, kd, x)
                stats.append(dict(code_equal=bool(torch.equal(cc["eng"], cc["sc"])),
                                  out_rel=float((ye.float() - ys.float()).norm() / ys.float().norm().clamp(min=1e-9))))
                return ys
        inference.disable_compile()
        try:
            p = Cmp(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
                    pos_argmax=pa_ho[:4], site_means=means_tr, respect_topk=True, topk=K, keep_tensors=kt,
                    keep_scales=scales)
            inference.forward(pt_ho[:4], patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
        finally:
            inference.enable_compile()
            LM.sparse_code = o_sc
        out[grad] = dict(sites=len(stats), code_equal=sum(s["code_equal"] for s in stats),
                         out_rel_max=max(s["out_rel"] for s in stats))
    res = dict(key=key, sites=len(sites), evalref_A_equals_scorer=eqA, evalref_C_equals_scorer=eqC,
               reference_rows_equal=rows_eq, n_ref_rows=[int(M._train_means_ref[0].shape[0]),
                                                         int(M._train_means_ref[1].shape[0])],
               v1_fitA_equals_scorer=fitA_eq, v1_fitC_equals_scorer=fitC_eq, site_check=out)
    print("093 MEANS %s" % json.dumps(res), flush=True)
    return res


def main():
    import driver
    keys = sys.argv[1:]
    R = driver.Runner()
    OUTC.mkdir(parents=True, exist_ok=True)
    fn = {"repro": repro, "means": means_and_sites}[CHECK]
    with open(OUTC / ("%s.jsonl" % CHECK), "a") as fh:
        for key in keys:
            fh.write(json.dumps(fn(R, key)) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
