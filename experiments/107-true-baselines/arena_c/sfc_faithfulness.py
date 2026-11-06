"""Arena C (DAN-191): SFC's own faithfulness / completeness evaluation, Pythia-70m-deduped.

A transcription of third_party/feature-circuits/experiments/faithfulness.ipynb @7fbd82b, cells 0-5 (`get_fcs` and
the loop over datasets / thresholds). The official clone is left untouched; every difference from their code is in
this file and is labelled OURS below. None changes the algorithm or the metric:

  OURS-1 (import path): ablation.py and the notebook import `load_examples` from `loading_utils`, where it no longer
         exists at HEAD (it moved to `data_loading_utils` in cbb5692). We alias it before importing ablation. At the
         paper-era commit 64d1cb3 `loading_utils.load_examples` is the identical function (specs/sfc.md §9.8).
  OURS-2 (issue #15): `run_with_ablations` raises a TypeError under nnsight 0.3.x because `res.shape[-1]` is a proxy
         inside the trace. Our copy below reads the same value (d_model) from the concrete patch state. Applied only
         if their function raises (FIX15=auto), so a run records whether it was needed.
  OURS-3 (config): model_name = Pythia-70m-deduped (the notebook at HEAD is set to Gemma-2-2B).
  OURS-4 (paths / caching): circuits are read from --circuit_dir and results written to --out; their md5 cache of
         `get_fcs` results is kept, under --out.
  OURS-5 (environment): torch 2.8 (Blackwell GPU) with TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 instead of torch < 2.6.

Run in the SFC env (~/venvs/sfc107), cwd = a work dir holding `data/` (a link to the clone's data):
  python sfc_faithfulness.py --circuit_dir circuits --out faithfulness --datasets rc [--thresholds 0.1]
"""
import argparse
import hashlib
import json
import os
import sys
import time

FC = os.environ.get("SFC_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party/feature-circuits"))
sys.path.insert(0, os.path.abspath(FC))

import torch as t
import data_loading_utils
import loading_utils

loading_utils.load_examples = data_loading_utils.load_examples   # OURS-1
from nnsight import LanguageModel
from activation_utils import SparseAct
from dictionary_learning.dictionary import IdentityDict
from dictionary_loading_utils import load_saes_and_submodules
import ablation
from data_loading_utils import load_examples

FIX15 = os.environ.get("FIX15", "auto")   # auto | on | off
fix15_used = False


def run_with_ablations_fixed(clean, patch, model, submodules, dictionaries, nodes, metric_fn, metric_kwargs=dict(),
                             complement=False, ablation_fn=lambda x: x.mean(dim=0).expand_as(x),
                             handle_errors="default"):
    """ablation.run_with_ablations @7fbd82b with only the issue #15 shape read changed (OURS-2)."""
    if patch is None:
        patch = clean
    patch_states = {}
    with model.trace(patch), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            patch_states[submodule] = SparseAct(act=f, res=x - x_hat).save()
    patch_states = {k: ablation_fn(v.value) for k, v in patch_states.items()}

    with model.trace(clean), t.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            submod_nodes = nodes[submodule]
            x = submodule.get_activation()
            f = dictionary.encode(x)
            res = x - dictionary(x)

            # ablate features
            if complement:
                submod_nodes = ~submod_nodes
            submod_nodes.resc = submod_nodes.resc.expand(
                *submod_nodes.resc.shape[:-1], patch_states[submodule].res.shape[-1]   # OURS-2: was res.shape[-1]
            )
            if handle_errors == "remove":
                submod_nodes.resc = t.zeros_like(submod_nodes.resc).to(t.bool)
            if handle_errors == "keep":
                submod_nodes.resc = t.ones_like(submod_nodes.resc).to(t.bool)

            f[..., ~submod_nodes.act] = patch_states[submodule].act[..., ~submod_nodes.act]
            res[..., ~submod_nodes.resc] = patch_states[submodule].res[..., ~submod_nodes.resc]

            submodule.set_activation(dictionary.decode(f) + res)

        metric = metric_fn(model, **metric_kwargs).save()
    return metric.value


def run_with_ablations(*a, **kw):
    global fix15_used
    if FIX15 == "on" or (FIX15 == "auto" and fix15_used):
        return run_with_ablations_fixed(*a, **kw)
    if FIX15 == "off":
        return ablation.run_with_ablations(*a, **kw)
    try:
        return ablation.run_with_ablations(*a, **kw)
    except TypeError as e:
        print(f"[OURS-2] their run_with_ablations raised TypeError ({e}); using the issue #15 fix from now on", flush=True)
        fix15_used = True
        return run_with_ablations_fixed(*a, **kw)


def clone_nodes(nodes):
    # run_with_ablations reassigns .resc on the SparseAct it is given; give each call its own copy so a failed first
    # attempt cannot leave a half-modified mask behind
    return {k: SparseAct(act=v.act.clone(), resc=v.resc.clone()) for k, v in nodes.items()}


# notebook cell 4, unchanged except: device / out / circuit_dir as arguments (OURS-4), run_with_ablations = ours above
def get_fcs(dataset, model, submodules, dictionaries, ablation_fn, thresholds, handle_errors="default",
            use_neurons=False, random=False, n_examples=40, device="cuda:0", circuit_dir="circuits",
            out_dir="faithfulness"):
    if "gemma-2" in model.config._name_or_path:
        model_name = "gemma-2-2b"
    else:
        model_name = "pythia-70m-deduped"

    hash_str = dataset + model_name + str([s.name for s in submodules]) + str(thresholds) + handle_errors + str(use_neurons) + str(random) + str(n_examples)
    hash = hashlib.md5(hash_str.encode()).hexdigest()
    if os.path.exists(f"{out_dir}/{hash}.pt"):
        return t.load(f"{out_dir}/{hash}.pt")

    circuit_path = f"{circuit_dir}/{model_name}_{dataset}_train_n100_aggnone" + (
        "_neurons" if use_neurons else ""
    ) + "_nodeall.pt"
    circuit = t.load(circuit_path)
    circuit = circuit['nodes']

    examples = load_examples(
        f"data/{dataset}_test.json", n_examples, model, use_min_length_only=True,
    )

    clean_inputs = [e['clean_prefix'] for e in examples]
    clean_answer_idxs = t.tensor(
        [model.tokenizer(e['clean_answer']).input_ids[-1] for e in examples], dtype=t.long, device=device
    )
    patch_inputs = [e['patch_prefix'] for e in examples]
    patch_answer_idxs = t.tensor(
        [model.tokenizer(e['patch_answer']).input_ids[-1] for e in examples], dtype=t.long, device=device
    )

    def metric_fn(model):
        logits = model.output.logits[:, -1, :]
        return (
            - t.gather(logits, dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) +
            t.gather(logits, dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
        )

    out = {"n_test_examples": len(examples), "test_examples": examples}

    with t.no_grad():
        with model.trace(clean_inputs):
            metric = metric_fn(model).save()
        fm = metric.value.mean().item()

        out["fm"] = fm

        # get m(∅)
        fempty_nodes = {
            submod: SparseAct(
                act=t.zeros(dictionaries[submod].dict_size, dtype=t.bool),
                resc=t.zeros(1, dtype=t.bool)
            ).to(device)
            for submod in submodules
        }

        fempty = run_with_ablations(
            clean_inputs, patch_inputs, model, submodules, dictionaries,
            nodes=fempty_nodes, metric_fn=metric_fn, ablation_fn=ablation_fn,
        ).mean().item()

        out["fempty"] = fempty

        for threshold in thresholds:
            out[threshold] = {}
            nodes = {
                submod: circuit[submod.name].abs() > threshold
                for submod in submodules
            }
            if handle_errors == 'remove':
                for k in nodes:
                    nodes[k].resc = t.zeros_like(nodes[k].resc, dtype=t.bool)
            elif handle_errors == 'resid_only':
                for k in nodes:
                    if "resid" not in k.name:
                        nodes[k].resc = t.zeros_like(nodes[k].resc, dtype=t.bool)
            n_nodes = sum(
                [n.act.sum() + n.resc.sum() for n in nodes.values()]
            ).item()
            if random:
                total_nodes = sum([n.act.numel() + n.resc.numel() for n in nodes.values()])
                p = n_nodes / total_nodes
                for k in nodes:
                    nodes[k].act = t.bernoulli(
                        t.ones_like(nodes[k].act, dtype=t.float) * p
                    ).to(device).to(dtype=t.bool)
                    nodes[k].resc = t.ones_like(nodes[k].resc, dtype=t.bool).to(device)
                out[threshold]['n_nodes'] = sum([n.act.sum() + n.resc.sum() for n in nodes.values()]).item()
            else:
                out[threshold]['n_nodes'] = n_nodes

            out[threshold]["fc"] = run_with_ablations(
                clean_inputs, patch_inputs, model, submodules, dictionaries,
                nodes=clone_nodes(nodes), metric_fn=metric_fn, ablation_fn=ablation_fn,
            ).mean().item()
            out[threshold]["fccomp"] = run_with_ablations(
                clean_inputs, patch_inputs, model, submodules, dictionaries,
                nodes=clone_nodes(nodes), metric_fn=metric_fn, ablation_fn=ablation_fn, complement=True
            ).mean().item()
            out[threshold]["faithfulness"] = (out[threshold]["fc"] - out["fempty"]) / (out["fm"] - out["fempty"])
            out[threshold]["completeness"] = (out[threshold]["fccomp"] - out["fempty"]) / (out["fm"] - out["fempty"])
            print(f"  {dataset} T_N={threshold:.3g}: n_nodes {out[threshold]['n_nodes']}, "
                  f"faithfulness {out[threshold]['faithfulness']:.4f}, completeness {out[threshold]['completeness']:.4f}",
                  flush=True)

    t.save(out, f"{out_dir}/{hash}.pt")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit_dir", default="circuits")
    ap.add_argument("--out", default="faithfulness")
    ap.add_argument("--datasets", default="rc,nounpp,simple,within_rc")
    ap.add_argument("--thresholds", default="", help="comma list; default the notebook's logspace(-4, 0, 15)")
    ap.add_argument("--settings", default="features", help="comma list of features, features_wo_errs, "
                                                             "features_wo_some_errs, neurons (notebook cell 5)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # notebook cells 1-3 (OURS-3: Pythia)
    model_name = "EleutherAI/pythia-70m-deduped"
    device = args.device
    dtype = {"EleutherAI/pythia-70m-deduped": t.float32, "google/gemma-2-2b": t.bfloat16}[model_name]
    model = LanguageModel(model_name, attn_implementation="eager", torch_dtype=dtype, device_map=device, dispatch=True)
    submodules, dictionaries = load_saes_and_submodules(model, include_embed=False, dtype=dtype, device=device)
    neuron_dicts = {submod: IdentityDict(dictionaries[submod].activation_dim) for submod in submodules}
    start_layer = {"EleutherAI/pythia-70m-deduped": 2, "google/gemma-2-2b": 8}[model_name]
    submodules = [s for s in submodules if int(s.name.split('_')[-1]) >= start_layer]
    ablation_fn = lambda x: x.mean(dim=0).expand_as(x)   # noqa: E731  (mean ablation, as the notebook)

    thresholds = ([float(x) for x in args.thresholds.split(",")] if args.thresholds
                  else t.logspace(-4, 0, 15).tolist())
    datasets = args.datasets.split(",")
    kw = {"features": {}, "features_wo_errs": {"handle_errors": "remove"},
          "features_wo_some_errs": {"handle_errors": "resid_only"}, "neurons": {"use_neurons": True}}
    t.cuda.reset_peak_memory_stats()
    summary = {"model": model_name, "start_layer": start_layer, "thresholds": thresholds, "results": {}}
    for setting in args.settings.split(","):
        dicts = neuron_dicts if setting == "neurons" else dictionaries
        for dataset in datasets:
            t0 = time.time()
            out = get_fcs(dataset, model, submodules, dicts, ablation_fn, thresholds=thresholds, device=device,
                          circuit_dir=args.circuit_dir, out_dir=args.out, **kw[setting])
            summary["results"].setdefault(setting, {})[dataset] = {
                "fm": out["fm"], "fempty": out["fempty"], "n_test_examples": out.get("n_test_examples"),
                "curve": [{"threshold": th, **out[th]} for th in thresholds], "seconds": round(time.time() - t0, 1)}
    summary["fix15_used"] = fix15_used
    summary["peak_gpu_mb"] = round(t.cuda.max_memory_allocated() / 2**20, 1)
    name = "summary_" + "_".join(args.settings.split(",")) + "_" + "_".join(datasets) + (
        "_custom" if args.thresholds else "") + ".json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, default=float))


if __name__ == "__main__":
    main()
