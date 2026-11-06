"""090: site-by-site check of the training fill against the scorer's fill on the REAL model and bank.

The scorer's CircuitOnlyPatcher (respect_topk=True, the rule's A fill) runs the forward; at every upstream site the
engine's training patcher (sp_engine, mode hard, binary gates on the production circuit's members and alphas, ablation
value = the scorer's A means) transforms the SAME input x, and the two codes and the two output streams are compared.
Done in the stream dtype (bf16) and again with x cast to float32. Codes are compared directly (engine: captured from
sp_code; scorer: captured from _respect_topk_fill), so rounding inside the decode does not hide a selection mismatch.

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/090-sparse-fill-training/check_sites.py <key>...
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  (sets OUT / sys.path for driver)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sp_engine  # noqa: E402


@torch.no_grad()
def check(R, mod, key):
    import driver
    from eval.ablation_faithfulness import CircuitOnlyPatcher, upstream_sites
    from eval.floors import collect_site_anchors
    G = R.G
    inference, bank, K, D = G["inference"], G["bank"], int(G["K"]), int(G["D"])
    rec = R.contexts(key)
    c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
    seed = next(n for n in c.nodes.values() if n.metadata.get("role") == "seed")
    sf = seed.metadata["feature_id"]
    layer, kind, sl = sf.layer, sf.kind, sf.index
    UPS = set(upstream_sites(bank, layer, kind))
    al = {s: d for s, d in run.circuit_alphas(c).items() if s in UPS and d}
    pt_tr, pa_tr, pt_ho, pa_ho = run.eval_slices(R, rec)
    means_tr, _ = collect_site_anchors(inference, bank, pt_tr, UPS, pa_tr, pin_position_specific=False)
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
        eff = mod._amp_of(ps)
        for i in al.get(s, {}):
            sv[i] = float(eff[i])
        scales[s] = sv
    keep = {s: set(d) for s, d in al.items()}
    kt = {s: torch.tensor(sorted(d), device=dev, dtype=torch.long) for s, d in al.items()}
    eng = mod.LearnedMaskPatcher(bank, thetas, layer, kind, w_seed, b_seed, amps=amps, code_dtype="stream",
                                 floors=means_tr, binarize="anneal", bin_threshold=0.5)
    eng.temperature = 0.05
    cap = {}
    orig_sp_code = mod.sp_code

    def sp_code_cap(code, H, mode):
        out = orig_sp_code(code, H, mode)
        cap["eng"] = out.detach()
        return out
    mod.sp_code = sp_code_cap
    stats = []

    class Cmp(CircuitOnlyPatcher):
        cast = None

        def _respect_topk_fill(self, all_latents, mean_vector, keep_tensor, kept_values):
            out = super()._respect_topk_fill(all_latents, mean_vector, keep_tensor, kept_values)
            cap["sc"] = out.detach()
            return out

        def transform(self, li, kd, x):
            if (li, kd) not in self.in_scope:
                return super().transform(li, kd, x)
            for nm, xx in (("stream", x), ("fp32", x.float())):
                cap.clear()
                ys = super().transform(li, kd, xx)
                sc = cap["sc"]
                ye = eng.transform(li, kd, xx)
                ec = cap["eng"]
                stats.append(dict(site="%d.%s" % (li, kd), dtype=nm,
                                  code_equal=bool(torch.equal(ec, sc)),
                                  code_support_equal=bool(torch.equal(ec != 0, sc != 0)),
                                  code_maxdiff=float((ec.float() - sc.float()).abs().max()),
                                  out_rel=float((ye.float() - ys.float()).norm() / ys.float().norm().clamp(min=1e-9))))
            return super().transform(li, kd, x)

    sp_engine.set_mode(mod, "hard")
    inference.disable_compile()
    try:
        tk = pt_ho[:4]
        p = Cmp(bank=bank, keep_indices=keep, in_scope=UPS, seed_layer=-1, seed_kind="", seed_latent_idx=0,
                pos_argmax=pa_ho[:4], site_means=means_tr, respect_topk=True, topk=K, keep_tensors=kt,
                keep_scales=scales)
        inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
    finally:
        inference.enable_compile()
        sp_engine.set_mode(mod, "dense")
        mod.sp_code = orig_sp_code
    out = {}
    for nm in ("stream", "fp32"):
        ss = [s for s in stats if s["dtype"] == nm]
        out[nm] = dict(sites=len(ss), code_equal=sum(s["code_equal"] for s in ss),
                       support_equal=sum(s["code_support_equal"] for s in ss),
                       code_maxdiff=max(s["code_maxdiff"] for s in ss), out_rel_max=max(s["out_rel"] for s in ss),
                       out_rel_med=float(np.median([s["out_rel"] for s in ss])))
    print("090 SITES %s %s" % (key, json.dumps(out)), flush=True)
    bad = [s for s in stats if not s["code_support_equal"]][:5]
    if bad:
        print("  support mismatches:", bad, flush=True)
    return out


def main():
    import driver
    mod = sp_engine.install()
    R = driver.Runner()
    for key in sys.argv[1:]:
        check(R, mod, key)


if __name__ == "__main__":
    main()
