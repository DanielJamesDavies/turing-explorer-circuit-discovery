"""DAN-36 GATE: native kernels on vs off give matching results (Pythia-70m, production engine). GPU ONLY in practice.

The native kernels on the fit / scoring path all live in the NO-GRAD SAE encode (src/sae/topk_sae.SAE.encode):
  cublasLt   sae/fused_linear_relu.linear_relu: relu(x W^T + b) in the GEMM epilogue. Loaded whenever
             src/native/linear_relu_ext*.so exists; used only when x, W, b are CUDA bfloat16.
  triton     sae/triton_topk.topk_nonneg_bf16 (TURINGLLM_TOPK_IMPL=triton; config default "pytorch"); CUDA bf16 only.
  exact      sae/fused_exact_topk.linear_relu_topk_exact (TURINGLLM_SAE_ENCODE_IMPL=fused_exact_topk) with the native
             .so (TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE=1); CUDA only.
The mask's training forward runs with grad enabled and always takes the plain PyTorch branch; the native kernels
reach a fit through its no-grad passes (floors, held-out loss) and reach every scorer pass. They are CUDA-only, and
on the public families they need model.dtype=bfloat16: at the default float32 (what 033 / 053 ran at, and the
public-family default) none of them is ever called. Each child therefore COUNTS the native invocations, so a "match"
that exercised nothing is visible as such.

Modes (one subprocess each; the backends are module globals read at import):
  off        TURINGLLM_TOPK_IMPL=pytorch, TURINGLLM_SAE_ENCODE_IMPL=standard, cublasLt disabled
  off2       a repeat of `off`: the run-to-run floor on this GPU (scatter_add / atomics nondeterminism)
  on         cublasLt + TURINGLLM_TOPK_IMPL=triton (standard encode)
  on_exact   cublasLt + fused_exact_topk with the native .so
Per (dtype, seed, mode): one protocol-v1 WCM fit (the 062 settings: triple floor, gamma 0.25 / 0.25, lambda 1e-3,
rank-keep 3e-3, 400 steps) on 033's contexts for the seed, then src/eval/wcm_eval.score_circuit (split explicit).
Compared with `off`: membership Jaccard, max |d alpha|, every numeric score field, the DAN-8 pass verdict; gate =
on / on_exact inside the off-vs-off2 floor, with 046's thresholds (Jaccard >= 0.98, |d alpha| <= 1e-2) as fallback.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv/bin/python experiments/068-validation-gates/native_onoff.py
  env: DTYPES (float32,bfloat16)  SEEDS (attn/1/31694,resid/2/15688: 033 panel keys)  MODES (off,off2,on,on_exact)
       STEPS (400)  DEVICE (cuda)  OUT (./out_native)  REPORT=1 (report from saved results only)
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUT", str(HERE / "out_native")))
DTYPES = [d for d in os.environ.get("DTYPES", "float32,bfloat16").split(",") if d]
SEEDS = [s for s in os.environ.get("SEEDS", "attn/1/31694,resid/2/15688").split(",") if s]
MODES = [m for m in os.environ.get("MODES", "off,off2,on,on_exact").split(",") if m]
STEPS = int(os.environ.get("STEPS", 400))
DEVICE = os.environ.get("DEVICE", "cuda")
MODE_ENV = {
    "off": dict(TURINGLLM_TOPK_IMPL="pytorch", TURINGLLM_SAE_ENCODE_IMPL="standard", _NO_CUBLASLT="1"),
    "off2": dict(TURINGLLM_TOPK_IMPL="pytorch", TURINGLLM_SAE_ENCODE_IMPL="standard", _NO_CUBLASLT="1"),
    "on": dict(TURINGLLM_TOPK_IMPL="triton", TURINGLLM_SAE_ENCODE_IMPL="standard"),
    "on_exact": dict(TURINGLLM_TOPK_IMPL="pytorch", TURINGLLM_SAE_ENCODE_IMPL="fused_exact_topk",
                     TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE="1"),
}
SCORE_KEYS = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "phi_cf_alpha_blind_tk")


def result_path(dtype, seed, mode):
    return OUT / ("%s__%s__%s.json" % (dtype, seed.replace("/", "_"), mode))


# ------------------------------------------------------------------------------------------------ child
def child(dtype, seed, mode):
    """One fit + score with the mode's backends; writes result_path(...)."""
    import torch
    if os.environ.get("_NO_CUBLASLT") == "1":
        import sae.fused_linear_relu as FLR
        FLR._available = False                               # linear_relu() falls back to F.relu(F.linear(...))
    import sae.fused_exact_topk as FX
    import sae.topk_sae as T
    counts = dict(encode_nograd=0, encode_grad=0, cublaslt=0, triton=0, exact_native=0, exact_python=0)
    _lr, _tk, _ex, _enc = T._linear_relu, T._topk_nonneg_bf16, T._linear_relu_topk_exact, T.SAE.encode

    def lr(x, w, b):
        import sae.fused_linear_relu as FLR_
        if FLR_._load() and x.is_cuda and x.dtype == w.dtype == b.dtype == torch.bfloat16:
            counts["cublaslt"] += 1
        return _lr(x, w, b)

    def tk(p, k):
        counts["triton"] += 1
        return _tk(p, k)

    def ex(*a, **kw):
        native = kw.get("use_native")
        counts["exact_native" if (native if native is not None else FX._native_requested()) else "exact_python"] += 1
        return _ex(*a, **kw)

    def enc(self, x):
        counts["encode_grad" if torch.is_grad_enabled() else "encode_nograd"] += 1
        return _enc(self, x)
    T._linear_relu, T._topk_nonneg_bf16, T._linear_relu_topk_exact, T.SAE.encode = lr, tk, ex, enc

    sys.path.insert(0, str(HERE))
    import ref033
    from pythia_033_panel import panel
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    from circuit.probe_dataset import ProbeDataset
    from config import config
    from eval import wcm_eval
    from model.inference import Inference
    from sae.bank import SAEBank

    dev = torch.device(DEVICE)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    config.model.family = "pythia-70m"
    config.model.dtype = None if dtype == "float32" else dtype
    disc, lm = config.discovery, config.discovery.learned_mask     # 062's protocol v1 (070 smoke.configure)
    lm.mask_floor_source = "triple"; lm.free_amplitude = True; lm.l1_lambda = 1e-3
    lm.deep_site_threshold = 99; lm.dual_floor_weight = 0.25; lm.triple_floor_weight = 0.25
    lm.offtarget_weight = 0.0; lm.rank_weight = 3e-3; lm.rank_mode = "keep"; lm.floors_train_only = True
    lm.steps = STEPS; lm.log_every = max(1, STEPS // 4)
    disc.floor_negctx_mode = "close"; disc.eval_batch_size = 64; disc.probe_sequence_count = 128
    disc.eval_sequence_count = 128; disc.probe_batch_size = 4; disc.context_protocol = "legacy"
    disc.neg_context_selection.preact_filter = False

    seeds, toks = panel()
    kind, layer, latent, pw, nw, _ = seeds[seed]
    pos, neg = toks[pw], toks[nw]
    R = ref033.Ref033("cpu")                                   # anchors only (033's rule), off the GPU under test
    anchors = R.seed(kind, layer, latent, pos, neg).anchors()
    del R
    inference = Inference(dev, family="pythia-70m")
    bank = SAEBank(device=dev, family="pythia-70m", preload=False)
    kinds = list(bank.kinds)
    probe = ProbeDataset(pos_tokens=pos.to(dev), target_tokens=torch.roll(pos, -1, dims=1).to(dev),
                         neg_tokens=neg.to(dev), pos_argmax=anchors, metadata={"n_train": 48})
    M = _build_mode_method("ablation_gradient", "mask", inference, bank,
                           torch.zeros((bank.n_layer * len(kinds), bank.d_sae)), None)
    M.build_probe_dataset = lambda comp, i, _p=probe: _p
    M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
    t0 = time.time()
    c = M.discover(layer * len(kinds) + kinds.index(kind), latent)
    t_fit = time.time() - t0
    fit_counts = dict(counts)
    members = {}
    for nd in c.nodes.values():
        if nd.metadata.get("role") != "seed":
            f = nd.metadata["feature_id"]
            members["%d.%s.%d" % (f.layer, f.kind, f.index)] = nd.metadata.get("amplitude")
    t1 = time.time()
    row = wcm_eval.score_circuit(inference, bank, c, probe, split="explicit")
    t_score = time.time() - t1
    score_counts = {k: counts[k] - fit_counts[k] for k in counts}
    out = dict(dtype=dtype, seed=seed, mode=mode, steps=STEPS, device=DEVICE, t_fit=round(t_fit, 1),
               t_score=round(t_score, 1), members=members, row=row, passes=bool(wcm_eval.passes(row)),
               vacuous=bool(wcm_eval.is_vacuous(row)), fit_counts=fit_counts, score_counts=score_counts,
               env={k: os.environ.get(k) for k in ("TURINGLLM_TOPK_IMPL", "TURINGLLM_SAE_ENCODE_IMPL",
                                                    "TURINGLLM_FUSED_EXACT_TOPK_USE_NATIVE", "_NO_CUBLASLT")})
    result_path(dtype, seed, mode).write_text(json.dumps(out, default=str))
    print("[native %s %s %s] n=%d fit %.0fs score %.0fs | fit kernels %s | score kernels %s"
          % (dtype, seed, mode, len(members), t_fit, t_score, fit_counts, score_counts), flush=True)


# ------------------------------------------------------------------------------------------------ compare
def numeric_fields(row):
    return {k: float(v) for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("secs", "secs_roles",
                                                                                     "peak_mem_mb")}


def compare(a, b):
    ma, mb = a["members"], b["members"]
    inter = set(ma) & set(mb)
    jac = len(inter) / max(len(set(ma) | set(mb)), 1)
    dalpha = max((abs((ma[i] or 0) - (mb[i] or 0)) for i in inter), default=0.0)
    fa, fb = numeric_fields(a["row"]), numeric_fields(b["row"])
    diffs = {k: abs(fa[k] - fb[k]) for k in set(fa) & set(fb) if not (math.isnan(fa[k]) and math.isnan(fb[k]))}
    exact = sum(1 for k in set(fa) & set(fb) if fa[k] == fb[k])
    worst = max(diffs.items(), key=lambda kv: kv[1], default=(None, 0.0))
    return dict(jaccard=jac, dalpha=dalpha, n_fields=len(diffs), exact_fields=exact, max_field=worst[0],
                max_diff=worst[1], score_diff={k: diffs.get(k) for k in SCORE_KEYS},
                verdict_same=a["passes"] == b["passes"])


def report():
    lines = ["# DAN-36: native kernels on vs off (%s)\n" % time.strftime("%Y-%m-%d %H:%M"),
             "Each mode vs `off`. `off2` is the same configuration again: its row is the run-to-run floor. Kernel "
             "counts are native calls during the fit / the scorer pass (0 = the mode exercised nothing).\n",
             "| dtype | seed | mode | n | Jaccard | max dα | fields exact / compared | max field diff (field) | "
             "Z / A / C / nec / induce diff | pass same | fit kernels (cublasLt / triton / exact) | "
             "score kernels | fit s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for dtype in DTYPES:
        for seed in SEEDS:
            res = {m: json.loads(result_path(dtype, seed, m).read_text())
                   for m in MODES if result_path(dtype, seed, m).exists()}
            if "off" not in res:
                continue
            floor = compare(res["off"], res["off2"]) if "off2" in res else None
            for m, r in res.items():
                cmp_ = compare(res["off"], r)
                fc, sc = r["fit_counts"], r["score_counts"]
                lines.append("| %s | %s | %s | %d | %.4f | %.2g | %d / %d | %.3g (%s) | %s | %s | %d / %d / %d | "
                             "%d / %d / %d | %s |" % (
                                 dtype, seed, m, len(r["members"]), cmp_["jaccard"], cmp_["dalpha"],
                                 cmp_["exact_fields"], cmp_["n_fields"], cmp_["max_diff"], cmp_["max_field"],
                                 " / ".join("%.2g" % v if v is not None else "-" for v in cmp_["score_diff"].values()),
                                 "yes" if cmp_["verdict_same"] else "NO", fc["cublaslt"], fc["triton"],
                                 fc["exact_native"], sc["cublaslt"], sc["triton"], sc["exact_native"], r["t_fit"]))
                if m in ("on", "on_exact"):
                    exercised = (fc["cublaslt"] + fc["triton"] + fc["exact_native"]
                                 + sc["cublaslt"] + sc["triton"] + sc["exact_native"]) > 0
                    j_floor = floor["jaccard"] if floor else 1.0
                    a_floor = floor["dalpha"] if floor else 0.0
                    ok = (cmp_["verdict_same"] and (cmp_["jaccard"] >= min(j_floor, 0.98))
                          and cmp_["dalpha"] <= max(a_floor, 1e-2))
                    verdicts.append((dtype, seed, m, ok, exercised))
    lines.append("\n## Verdict\n")
    for dtype, seed, m, ok, ex in verdicts:
        lines.append("- %s %s %s: %s%s" % (dtype, seed, m, "MATCH" if ok else "DIFFERENT",
                                            "" if ex else " (no native kernel was called: vacuous)"))
    txt = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


def main():
    if os.environ.get("_CHILD"):
        dtype, seed, mode = os.environ["_CHILD"].split("|")
        child(dtype, seed, mode)
        return
    if os.environ.get("REPORT") != "1":
        OUT.mkdir(parents=True, exist_ok=True)
        for dtype in DTYPES:
            for seed in SEEDS:
                for mode in MODES:
                    if result_path(dtype, seed, mode).exists():
                        continue
                    env = dict(os.environ, _CHILD="%s|%s|%s" % (dtype, seed, mode), **MODE_ENV[mode])
                    print("[native] %s %s %s ..." % (dtype, seed, mode), flush=True)
                    rc = subprocess.call([sys.executable, __file__], env=env)
                    if rc:
                        print("[native] %s %s %s FAILED (rc %d)" % (dtype, seed, mode, rc), flush=True)
    report()


if __name__ == "__main__":
    main()
