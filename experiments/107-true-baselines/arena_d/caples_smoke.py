"""Arena D2 SMOKE ONLY [ours, never a reported row]: Caples et al.'s Gemma-2-2B code path at small size, in bf16.

What it is: `third_party/ScalableSAECircuits/interp_bottlenecks.py:main()` (@f54a4ef) re-stated step by step, calling
THEIR `helpers/utils.py` functions unmodified (`run_sae_hook_fn`, `get_sae_means`, `get_sae_error_means`,
`do_training_run`). It exists to catch environment / API / data bugs before the pod, where the real row runs their
script itself, unmodified, in fp32 on `google/gemma-2-2b` (README §5.2).

Every difference from their script, all memory or size only:
  1. weights: `hf_model=` the cached `unsloth/gemma-2-2b` (bf16) and its tokenizer, passed to
     `HookedSAETransformer.from_pretrained("google/gemma-2-2b", ...)` (038's route); theirs downloads google's fp32;
  2. dtype: the model and the five SAEs in bf16 (theirs: TransformerLens' fp32 default), run under torch.autocast so
     their fp32 means / mask arithmetic meets bf16 weights (sae-lens decode is a plain matmul);
  3. size: --n prompts (theirs 3000), --mean-batches (theirs 40 for both mean passes), --lams (theirs: the 5-step grid
     of --start/--end_threshold); --portion as theirs (0.4 in sae_gap.sh);
  4. device: passed explicitly to get_sae_means / get_sae_error_means / do_training_run (their defaults are cuda:0),
     so the smoke can run on CPU;
  5. outputs: their do_training_run writes masks/<task>/... relative to the working directory, so this script chdirs
     into --out first. Nothing is written under third_party/ (bytecode writing is off for the import).
The selection of SAEs (L0 closest to --l0-target among sae-lens' gemma-scope-2b-pt-res width_16k entries, layers
range(0, n_layers, gap)), the data filter (both prefixes exactly --example-length tokens), batching (16, remainder
dropped) and the order of calls are theirs (interp_bottlenecks.py:138-337).

  WANDB_MODE=disabled HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" ~/venvs/caples107/bin/python -I caples_smoke.py \
      --device cpu --n 176 --mean-batches 4 --lams 0.1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAPLES = ROOT / "third_party" / "ScalableSAECircuits"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--caples-src", default=str(CAPLES), help="their clone (read only)")
    p.add_argument("--task", default="sva/rc_train")
    p.add_argument("--example-length", type=int, default=7)
    p.add_argument("--n", type=int, default=176, help="prompts kept (theirs 3000); >= 160 for their 10 eval batches")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sae-gap", type=int, default=6)
    p.add_argument("--l0-target", type=int, default=100)
    p.add_argument("--mean-batches", type=int, default=4, help="theirs 40")
    p.add_argument("--lams", default="0.1", help="comma-separated sparsity multipliers")
    p.add_argument("--portion", type=float, default=0.4)
    p.add_argument("--hf-model", default="unsloth/gemma-2-2b")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(HERE / "out" / "caples_smoke"))
    return p.parse_args()


def main():
    a = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    sys.dont_write_bytecode = True                      # importing their helpers must not write into third_party/
    sys.path.insert(0, a.caples_src)
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_lens import SAE, HookedSAETransformer
    from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
    import helpers.utils as utils                       # theirs, unmodified

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[a.dtype]
    dev = a.device
    t0 = time.time()
    log = {"args": vars(a), "label": "SMOKE ONLY [ours]: their utils, bf16 / small N / unsloth weights"}

    # ---- model (difference 1, 2)
    hf = AutoModelForCausalLM.from_pretrained(a.hf_model, torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(a.hf_model)
    model = HookedSAETransformer.from_pretrained("google/gemma-2-2b", hf_model=hf, tokenizer=tok, device=dev,
                                                 dtype=dtype)
    del hf
    for prm in model.parameters():
        prm.requires_grad_(False)
    log["model_load_s"] = round(time.time() - t0, 1)

    # ---- SAEs: their selection code (interp_bottlenecks.py:138-174)
    df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
    df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
    neuronpedia_dict = df.loc["gemma-scope-2b-pt-res"]["saes_map"]
    pattern = re.compile(r"layer_(\d+)/width_16k/average_l0_(\d+)")
    layer_dict = defaultdict(list)
    for s in neuronpedia_dict.keys():
        m = pattern.search(s)
        if m and neuronpedia_dict[s] is not None:
            layer_dict[int(m.group(1))].append((s, int(m.group(2))))
    closest = {L: min(v, key=lambda x: abs(x[1] - a.l0_target))[0] for L, v in layer_dict.items()}
    saes = []
    for L in range(0, model.cfg.n_layers, a.sae_gap):
        sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res", sae_id=closest[L], device=dev)[0]
        saes.append(sae.to(dtype=dtype))                # difference 2
    log["saes"] = [closest[L] for L in range(0, model.cfg.n_layers, a.sae_gap)]

    # ---- data: their filter and batching (interp_bottlenecks.py:180-230)
    with open(Path(a.caples_src) / "data" / ("%s.json" % a.task)) as f:
        data = [json.loads(line) for line in f]
    cd, pd_, cl, pl = [], [], [], []
    for e in data:
        if len(model.tokenizer(e["clean_prefix"]).input_ids) == len(model.tokenizer(e["patch_prefix"]).input_ids) \
                == a.example_length:
            cd.append(e["clean_prefix"]); pd_.append(e["patch_prefix"])
            cl.append(e["clean_answer"]); pl.append(e["patch_answer"])
    cd, pd_, cl, pl = cd[:a.n], pd_[:a.n], cl[:a.n], pl[:a.n]
    clean = model.to_tokens(cd)
    corr = model.to_tokens(pd_)
    clean_lab = model.to_tokens(cl, prepend_bos=False).squeeze(-1)
    corr_lab = model.to_tokens(pl, prepend_bos=False).squeeze(-1)
    nb = len(clean) // a.batch_size
    bs = a.batch_size
    clean = clean[:nb * bs].reshape(-1, bs, clean.shape[-1])
    corr = corr[:nb * bs].reshape(-1, bs, corr.shape[-1])
    clean_lab = clean_lab[:nb * bs].reshape(-1, bs)
    corr_lab = corr_lab[:nb * bs].reshape(-1, bs)
    assert clean.shape[-1] == a.example_length, clean.shape
    assert nb >= 10, "their eval reads 10 batches; use --n >= 160"
    assert a.mean_batches <= nb
    log.update(n_batches=nb, seq_len=int(clean.shape[-1]))

    amp = torch.autocast(device_type="cuda" if str(dev).startswith("cuda") else "cpu", dtype=dtype) \
        if dtype != torch.float32 else contextlib.nullcontext()
    ne = min(10, nb)
    with amp:
        with torch.no_grad():
            ld = sum(utils.logit_diff_fn(model(clean[i]), clean_lab[i], corr_lab[i]) for i in range(ne)) / ne
            log["avg_model_ld"] = float(ld)
            ld = 0.0
            for i in range(ne):
                logits, saes = utils.run_sae_hook_fn(model, saes, clean[i], use_mean_error=False)
                ld = ld + utils.logit_diff_fn(logits, clean_lab[i], corr_lab[i])
            log["avg_sae_ld"] = float(ld / ne)
        for sae in saes:
            sae.mask = utils.SparseMask(sae.cfg.d_sae, 1.0, seq_len=a.example_length).to(dev)
        saes = utils.get_sae_means(model, saes, corr, total_batches=a.mean_batches, batch_size=bs, device=dev)
        saes = utils.get_sae_error_means(model, saes, corr, total_batches=a.mean_batches, batch_size=bs, device=dev)
        with torch.no_grad():
            ld = 0.0
            for i in range(ne):
                logits, saes = utils.run_sae_hook_fn(model, saes, clean[i], use_mean_error=True)
                ld = ld + utils.logit_diff_fn(logits, clean_lab[i], corr_lab[i])
            log["avg_sae_mean_error_ld"] = float(ld / ne)
        out = Path(a.out).resolve()                     # absolute: a relative --out breaks after the chdir below
        out.mkdir(parents=True, exist_ok=True)
        os.chdir(out)                                   # difference 5: their masks/ go under --out
        save_task = a.task + "_saegap" + str(a.sae_gap) + "_mean_error"
        runs = []
        for lam in [float(x) for x in a.lams.split(",")]:
            t1 = time.time()
            utils.do_training_run(model=model, saes=saes, token_dataset=clean, labels_dataset=clean_lab,
                                  corr_labels_dataset=corr_lab, sparsity_multiplier=lam, task=save_task,
                                  example_length=a.example_length, loss_function="logit_diff", per_token_mask=True,
                                  use_mask=True, mean_mask=True, portion_of_data=a.portion, use_mean_error=True,
                                  device=dev)
            res = json.load(open(out / "masks" / save_task / ("logit_diff_%s_run" % lam) / ("%s_run.json" % lam)))
            runs.append(dict(lam=lam, total_density=res["total_density"], faithfulness=res["faithfulness"],
                             logit_diff=res["logit_diff"], fit_s=round(time.time() - t1, 1)))
            print(runs[-1], flush=True)
    log["runs"] = runs
    log["total_s"] = round(time.time() - t0, 1)
    with open(out / "smoke_summary.json", "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
