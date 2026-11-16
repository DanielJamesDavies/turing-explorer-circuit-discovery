"""116 diagnostic: what removing the SAE errors does to the target in the UNABLATED model (no circuit at all).

For each target, on its held-out strongest contexts (the scorer's own rows and anchors, 062 Runner + 049 helpers):
the target's mean activation (post-Top-K read, as the scorer's _tk) when, at every upstream site of the given kinds,
the stream is replaced by the SAE reconstruction decode(encode(x)) (error zeroed), or by the reconstruction plus the
error's mean over the strongest-train rows (error mean-ablated). Every latent stays at its live value: this is the
"model run on SAE reconstructions" and isolates the error's own contribution from the circuit's. Also the mean norm
ratio |reconstruction| / |x| and |reconstruction + mean error| / |x| per kind at the upstream sites.

  wsl: KEYS=<a,b,...> PYTHONPATH=src .venv/bin/python -X utf8 experiments/116-error-ablated-and-case-checks/diag_recon.py
  -> out/diag_recon.jsonl
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
PROD_OUT = ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2"
CTX = ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"
RS = HERE / "out" / "rs_turingllm"
os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(RS), CTX_DIR=str(CTX), SKIP_SPEC="1")
os.environ.setdefault("TAG", "diag116_unused")
for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1")):
    if p not in sys.path:
        sys.path.insert(0, p)
import driver  # noqa: E402


def main():
    import torch
    from eval.ablation_faithfulness import upstream_sites
    from eval.floors import collect_site_error_means
    from pipeline.component_index import component_idx
    from sae.dense import sparse_topk_to_dense
    keys = [k.strip() for k in os.environ["KEYS"].split(",") if k.strip()]     # strip: Windows-written lists carry \r
    done = set()
    p_out = HERE / "out" / "diag_recon.jsonl"
    if p_out.exists():
        done = {json.loads(x)["seed"] for x in p_out.read_text().splitlines() if x.strip()}
    keys = [k for k in keys if k not in done]
    R = driver.Runner()
    G, V, H = R.G, R.V, R.H
    bank, D = G["bank"], G["D"]
    out = open(HERE / "out" / "diag_recon.jsonl", "a")
    for key in keys:
        rec = R.contexts(key)
        H.patch_eval_contexts(G, rec, "strong")
        l, kd, i = H.parse(key)
        comp = component_idx(l, G["KINDS"].index(kd), G["NK"])
        pd_ = G["M0"].probe_builder.build(comp, i) if False else G["M0"].build_probe_dataset(comp, i)
        n_tr = int(pd_.metadata["n_train"])
        pt, pa = pd_.pos_tokens, pd_.pos_argmax
        pt_tr, pt_ho, pa_ho = pt[:n_tr], pt[n_tr:], pa[n_tr:]
        sae = bank.saes[kd][l]
        w, b = sae.encoder.weight[i].detach(), sae._get_bias_eff()[i].detach()
        UPS = upstream_sites(bank, l, kd)
        em = collect_site_error_means(G["inference"], bank, pt_tr, set(UPS))
        norms = {}

        class P:
            def __init__(self, kinds, mean):
                self.kinds, self.mean = kinds, mean
                self.w_seed, self.b_seed, self.seed_idx = w, b, i
                self.seed_pre = None; self.tap_tk = None

            def __call__(self, model):
                from model.hooks import multi_patch
                return multi_patch(model, self.transform)

            def transform(self, layer_idx, kind, x):
                if (layer_idx, kind) == (l, kd):
                    self.seed_pre = V._tap(self, x, (l, kd))
                    return x
                if (layer_idx, kind) not in UPS or kind not in self.kinds:
                    return x
                ta, ti = bank.encode(x, kind, layer_idx)
                rec_ = bank.decode(sparse_topk_to_dense(ta, ti, D, dtype=x.dtype), kind, layer_idx).to(x.dtype)
                if self.mean:
                    rec_ = rec_ + em[(layer_idx, kind)].to(device=x.device, dtype=x.dtype)
                nm = norms.setdefault((self.mean, kind), [0.0, 0])
                nm[0] += float((rec_.float().norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-6)).mean()); nm[1] += 1
                return rec_

        row = dict(seed=key)
        row["clean"] = V.read(lambda a, b_: V.AmpInjectPatcher({}, (l, kd), w, b, i), pt_ho, pa_ho)["tk"]
        for kinds in (("attn", "mlp", "resid"), ("resid",), ("mlp",), ("attn",)):
            for mean in (False, True):
                name = "%s_%s" % ("mean" if mean else "zero", "+".join(kinds))
                row[name] = V.read(lambda a, b_, k_=kinds, m_=mean: P(k_, m_), pt_ho, pa_ho)["tk"]
                row[name + "_ratio"] = row[name] / row["clean"] if row["clean"] else None
        row["norm_ratio"] = {"%s_%s" % ("mean" if m else "zero", k): v[0] / v[1] for (m, k), v in norms.items()}
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n"); out.flush()


if __name__ == "__main__":
    main()
