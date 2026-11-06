"""Arena D3: Caples et al.'s per-position means on the Gemma-2-2B substrate, saved for the WCM side (run in caples107).

What it does: re-states interp_bottlenecks.py:main() (@f54a4ef) up to its training loop, calling THEIR helpers/utils.py
functions unmodified, and saves what their 2B configuration (scripts/sae_gap.sh: --use_mask --mean_mask
--use_mean_error --per_token_mask) builds before fitting:
  theirs  per SAE: code = sae.mean_ablation [T, d_sae] (get_sae_means) and err = sae.mean_error [T, d] (get_sae_error_means),
          over the patch prompts of batches 0..39 (640), with their running-mean bug kept (n = batch index + 1 for every
          prompt of a batch). Both are computed in their error-dropped splice (run_sae_hook_fn without use_mean_error), as
          theirs are; codes and errors are 0 at BOS.
  exact   [ours] the true per-position means of the same prompts in the same substrate (batched run_sae_hook_fn,
          summed, divided by the prompt count).
  ld      their logit differences, per batch: on their main()'s first 10 batches and on do_training_run's eval batches
          [0, -1, ..., -9]: the model, model + SAEs (error dropped), model + SAEs (mean error) = the open-gate LD of
          Caples' substrate, which our caples-mode site model must reproduce (README §4.3 test 4).
  mean_tokens [640, T]  the prompts the means are over (wcm_gemma checks them against its own task build).
wcm_gemma.load_caples_means reads the file ("theirs" | "exact") as the C fill and the Caples-mode eps_bar.

Modes:
  real     their model load (HookedSAETransformer.from_pretrained(--model), fp32 = theirs; --hf-model / --dtype bfloat16
           = 038's local route, a labelled smoke), their SAE selection (L0 closest to 100, layers range(0, 26, gap)),
           their data filter and file-order batches of 16 (N 3000).
  fixture  [tests] a tiny random Gemma-2 (HF weights + config from --fixture, converted by TL's own from_pretrained with
           the config's sizes patched in) and tiny sae-lens JumpReLU SAEs from the fixture's parameters; their utils
           functions unchanged.
Nothing is written under third_party/ (bytecode writing is off for the import; their functions write no files).

  pod:   WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ~/venvs/caples107/bin/python -I caples_means.py \
             --device cuda --out out/caples_means/rc_gap6.pt
  local: ... --device cpu --hf-model unsloth/gemma-2-2b --dtype bfloat16 --out out/caples_means/rc_gap6_bf16_SMOKE.pt
  test:  ... --fixture FIXTURE.pt --mean-batches 4 --out MEANS.pt
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAPLES = ROOT / "third_party" / "ScalableSAECircuits"
FORMAT = "arena-d-caples-means-v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--caples-src", default=str(CAPLES), help="their clone (read only)")
    p.add_argument("--task", default="sva/rc_train")
    p.add_argument("--example-length", type=int, default=7)
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sae-gap", type=int, default=6)
    p.add_argument("--l0-target", type=int, default=100)
    p.add_argument("--mean-batches", type=int, default=40, help="theirs: 40 (interp_bottlenecks.py:274-275)")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--hf-model", default=None, help="[smoke] local HF weights passed as hf_model= (038's route)")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--hf-cache", default=None, help="their --hf_cache (TL cache_dir)")
    p.add_argument("--fixture", default=None, help="[tests] a tiny-model fixture .pt (see module docstring)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _git_head(path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------------- loading
def load_real(a, torch, dtype):
    import pandas as pd
    from sae_lens import SAE, HookedSAETransformer
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    if a.hf_model:                                             # smoke route [ours]: local weights, labelled
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=dtype)
        tok = AutoTokenizer.from_pretrained(a.hf_model)
        model = HookedSAETransformer.from_pretrained(a.model, hf_model=hf, tokenizer=tok, device=a.device,
                                                     dtype=dtype)
        del hf
    else:                                                      # theirs (interp_bottlenecks.py:122-126)
        model = HookedSAETransformer.from_pretrained(a.model, device=a.device, cache_dir=a.hf_cache, dtype=dtype)
    for prm in model.parameters():
        prm.requires_grad_(False)
    # their SAE selection (interp_bottlenecks.py:138-174)
    df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
    df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
    release = {"google/gemma-2-9b": "gemma-scope-9b-pt-res", "google/gemma-2-2b": "gemma-scope-2b-pt-res"}[a.model]
    neuronpedia_dict = df.loc[release]["saes_map"]
    pattern = re.compile(r"layer_(\d+)/width_16k/average_l0_(\d+)")
    layer_dict = defaultdict(list)
    for s in neuronpedia_dict.keys():
        m = pattern.search(s)
        if m and neuronpedia_dict[s] is not None:
            layer_dict[int(m.group(1))].append((s, int(m.group(2))))
    closest = {L: min(v, key=lambda x: abs(x[1] - a.l0_target))[0] for L, v in layer_dict.items()}
    saes, ids = [], []
    for L in range(0, model.cfg.n_layers, a.sae_gap):
        if L in closest:
            sae = SAE.from_pretrained(release=release, sae_id=closest[L], device=a.device)[0]
            saes.append(sae.to(dtype=dtype) if dtype != torch.float32 else sae)
            ids.append(closest[L])
    # their data filter and batching (interp_bottlenecks.py:180-228)
    with open(Path(a.caples_src) / "data" / ("%s.json" % a.task)) as f:
        data = [json.loads(line) for line in f]
    cd, pd_, cl, pl = [], [], [], []
    for e in data:
        if len(model.tokenizer(e["clean_prefix"]).input_ids) == len(model.tokenizer(e["patch_prefix"]).input_ids) \
                == a.example_length:
            cd.append(e["clean_prefix"]); pd_.append(e["patch_prefix"])
            cl.append(e["clean_answer"]); pl.append(e["patch_answer"])
    cd, pd_, cl, pl = cd[:a.N], pd_[:a.N], cl[:a.N], pl[:a.N]
    clean, corr = model.to_tokens(cd), model.to_tokens(pd_)
    clean_lab = model.to_tokens(cl, prepend_bos=False).squeeze(-1)
    corr_lab = model.to_tokens(pl, prepend_bos=False).squeeze(-1)
    nb, bs = len(clean) // a.batch_size, a.batch_size
    clean = clean[:nb * bs].reshape(-1, bs, clean.shape[-1])
    corr = corr[:nb * bs].reshape(-1, bs, corr.shape[-1])
    clean_lab = clean_lab[:nb * bs].reshape(-1, bs)
    corr_lab = corr_lab[:nb * bs].reshape(-1, bs)
    return model, saes, ids, (clean, corr, clean_lab, corr_lab)


def load_fixture(a, torch):
    """Tiny random Gemma-2 through TL's own from_pretrained (sizes patched into its built-in gemma-2-2b config) and
    tiny sae-lens JumpReLU SAEs with Gemma Scope's loader config."""
    import transformer_lens.loading_from_pretrained as LP
    from sae_lens import SAE, HookedSAETransformer
    from sae_lens.toolkit.pretrained_sae_loaders import SAEConfigLoadOptions, get_gemma_2_config
    from transformers import AutoTokenizer, Gemma2Config, Gemma2ForCausalLM
    fx = torch.load(a.fixture, map_location="cpu", weights_only=False)
    hc = dict(fx["hf_config"])
    hf = Gemma2ForCausalLM(Gemma2Config(**hc)).eval()
    missing, unexpected = hf.load_state_dict(fx["hf_state"], strict=False)
    if [k for k in missing if k != "lm_head.weight"] or unexpected:
        raise RuntimeError("fixture state dict: missing %s, unexpected %s" % (missing, unexpected))
    hf.tie_weights()
    orig = LP.convert_hf_model_config

    def tiny_cfg(model_name, **kw):
        d = orig(model_name, **kw)
        d.update(d_model=hc["hidden_size"], d_head=hc["head_dim"], n_heads=hc["num_attention_heads"],
                 d_mlp=hc["intermediate_size"], n_layers=hc["num_hidden_layers"],
                 n_ctx=hc["max_position_embeddings"], d_vocab=hc["vocab_size"],
                 n_key_value_heads=hc["num_key_value_heads"], window_size=hc["sliding_window"])
        return d

    LP.convert_hf_model_config = tiny_cfg
    try:
        tok = AutoTokenizer.from_pretrained(fx.get("tokenizer", "unsloth/gemma-2-2b"))
        model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", hf_model=hf, tokenizer=tok, device=a.device,
                                                     dtype=torch.float32)
    finally:
        LP.convert_hf_model_config = orig
    for prm in model.parameters():
        prm.requires_grad_(False)
    saes, ids = [], []
    for L in sorted(fx["saes"]):
        p = fx["saes"][L]
        cfg = get_gemma_2_config("google/gemma-scope-2b-pt-res", "layer_%d/width_16k/average_l0_0" % L,
                                 SAEConfigLoadOptions(device=a.device))
        cfg.update(d_in=int(p["W_enc"].shape[0]), d_sae=int(p["W_enc"].shape[1]))
        sae = SAE.from_dict(cfg)
        sae.load_state_dict({k: p[k].float() for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")})
        sae.to(a.device)
        saes.append(sae)
        ids.append("fixture/layer_%d" % L)
    return model, saes, ids, (fx["clean"], fx["corr"], fx["clean_lab"], fx["corr_lab"])


# ------------------------------------------------------------------------------------------------------- compute
def compute(a, torch, utils, model, saes, toks, dtype):
    clean, corr, clean_lab, corr_lab = toks
    nb, bs, T = int(clean.shape[0]), int(clean.shape[1]), int(clean.shape[2])
    if nb < 10 or a.mean_batches > nb:
        raise ValueError("%d batches: their evaluation reads 10 and the means %d" % (nb, a.mean_batches))
    dev = a.device
    sets = {"first10": list(range(min(10, nb))),                   # main()'s model / SAE LDs (interp_bottlenecks.py)
            "eval": [(-i) % nb for i in range(10)]}                # do_training_run's batch[-i], i = 0..9
    amp = torch.autocast(device_type="cuda" if str(dev).startswith("cuda") else "cpu", dtype=dtype) \
        if dtype != torch.float32 else contextlib.nullcontext()
    ld = {k: defaultdict(list) for k in sets}

    def lds(name, fn):
        with torch.no_grad():
            for k, idx in sets.items():
                for i in idx:
                    ld[k][name].append(float(utils.logit_diff_fn(fn(clean[i].to(dev)), clean_lab[i].to(dev),
                                                                corr_lab[i].to(dev))))

    with amp:
        lds("model", lambda t: model(t))
        lds("sae", lambda t: utils.run_sae_hook_fn(model, saes, t, use_mean_error=False)[0])
        for sae in saes:                                           # as main() does under --use_mask
            sae.mask = utils.SparseMask(sae.cfg.d_sae, 1.0, seq_len=a.example_length).to(dev)
        corr_d = corr.to(dev)
        t0 = time.time()
        saes = utils.get_sae_means(model, saes, corr_d, total_batches=a.mean_batches, batch_size=bs, device=dev)
        saes = utils.get_sae_error_means(model, saes, corr_d, total_batches=a.mean_batches, batch_size=bs, device=dev)
        t_theirs = time.time() - t0
        lds("sae_mean_error", lambda t: utils.run_sae_hook_fn(model, saes, t, use_mean_error=True)[0])
        theirs = {int(s.cfg.hook_layer): dict(code=s.mean_ablation.detach().float().reshape(T, -1).cpu().clone(),
                                              err=s.mean_error.detach().float().reshape(T, -1).cpu().clone())
                  for s in saes}
        # exact [ours]: the same prompts, the same error-dropped splice, the true mean
        code_sum = {int(s.cfg.hook_layer): torch.zeros(T, s.cfg.d_sae, dtype=torch.float64) for s in saes}
        err_sum = {int(s.cfg.hook_layer): torch.zeros(T, s.cfg.d_in, dtype=torch.float64) for s in saes}
        with torch.no_grad():
            for i in range(a.mean_batches):
                _, saes = utils.run_sae_hook_fn(model, saes, corr_d[i], cache_sae_activations=True)
                for s in saes:
                    code_sum[int(s.cfg.hook_layer)] += s.feature_acts.double().sum(0).cpu()
                _ = utils.run_sae_hook_fn(model, saes, corr_d[i], calc_error=True)
                for s in saes:
                    err_sum[int(s.cfg.hook_layer)] += s.error_term.double().sum(0).cpu()
        n = a.mean_batches * bs
        exact = {l: dict(code=(code_sum[l] / n).float(), err=(err_sum[l] / n).float()) for l in code_sum}
    return dict(theirs=theirs, exact=exact, ld={k: dict(v) for k, v in ld.items()}, eval_batches=sets,
                mean_tokens=corr[:a.mean_batches].reshape(-1, T).cpu().long(), n_batches=nb, seq_len=T,
                t_theirs_s=round(t_theirs, 1))


def main():
    a = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.dont_write_bytecode = True                      # importing their helpers must not write into third_party/
    sys.path.insert(0, a.caples_src)
    import torch
    import helpers.utils as utils                       # theirs, unmodified
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    t0 = time.time()
    if a.fixture:
        model, saes, ids, toks = load_fixture(a, torch)
    else:
        model, saes, ids, toks = load_real(a, torch, dtype)
    t_load = time.time() - t0
    r = compute(a, torch, utils, model, saes, toks, dtype)
    label = ("TEST FIXTURE [tiny random Gemma-2]" if a.fixture else
             "SMOKE [ours: %s, %s]" % (a.hf_model, a.dtype) if (a.hf_model or a.dtype != "float32") else
             "theirs: %s fp32" % a.model)
    meta = dict(label=label, args=vars(a), sae_ids=ids, layers=[int(s.cfg.hook_layer) for s in saes],
                hook_names=[s.cfg.hook_name for s in saes], caples_commit=_git_head(a.caples_src),
                n_batches=r["n_batches"], seq_len=r["seq_len"], mean_prompts=int(r["mean_tokens"].shape[0]),
                eval_batches=r["eval_batches"], t_load_s=round(t_load, 1), t_theirs_means_s=r["t_theirs_s"],
                t_total_s=round(time.time() - t0, 1), torch=torch.__version__)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=FORMAT, meta=meta, layers=meta["layers"], mean_tokens=r["mean_tokens"],
                    theirs=r["theirs"], exact=r["exact"], ld=r["ld"]), out)
    summ = {k: {n: (sum(v) / len(v)) for n, v in d.items()} for k, d in r["ld"].items()}
    diff = {l: dict(code=float((r["theirs"][l]["code"] - r["exact"][l]["code"]).abs().max()),
                    err=float((r["theirs"][l]["err"] - r["exact"][l]["err"]).abs().max())) for l in r["theirs"]}
    print(json.dumps(dict(out=str(out), label=label, mean_ld=summ, theirs_minus_exact_max=diff,
                          t_total_s=meta["t_total_s"]), indent=1))


if __name__ == "__main__":
    main()
