"""Arena A2 (DAN-200): fidelity checks of Dunefsky et al.'s official circuit code (third_party/transcoder_circuits @8721ff0,
unmodified) against the authors' own saved notebook outputs. Run in ~/venvs/tc107 (TL 1.11.0, torch 2.2.0):

  source ~/venvs/tc107/bin/tc_env.sh
  CUDA_VISIBLE_DEVICES= python experiments/107-true-baselines/arena_a/dunefsky_checks.py [--checks R1,R2,R3,R4,R5]

  R1  walkthrough cells 57/60: mlp8tc[89], "Oh, that rifle model is a 6M", 3 x 15, raw -> print_all_paths text, exact
  R2  walkthrough cells 68/70: paths_to_graph edges and node contribs, <= 1e-3 relative
  R3  walkthrough cell 73: add_error_nodes_to_graph node contribs, <= 1e-3 relative
  R4  greaterthan cells 12/14/15: W_U['40'] - W_U['20'] at resid_post 11, "The war lasted from 1735 to 17", 3 x 15, raw
  R5  greaterthan cell 21: the same on "The war lasted from 1527 to 15"

The expected values are parsed from the notebooks' saved outputs (not retyped). Glue that is ours, labelled:
  * CPU loading: their SparseAutoencoder.load_from_pretrained calls torch.load(path) without map_location, and the
    pickles hold cuda:0 tensors, so on a CPU-only machine we run the same lines (cls(cfg=...), load_state_dict) with
    map_location="cpu" and cfg.device = "cpu" (what their own mps branch does). On a GPU their loader runs as shipped.
  * the walkthrough loads transcoders 0-10 (range(11)), greaterthan 0-11 (range(12)): followed exactly.
  * greaterthan's make_one_hot puts its vector on "cuda" (a notebook helper, not library code): built on the CPU here.
Writes out/dunefsky_checks/results.json.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TCC = ROOT / "third_party" / "transcoder_circuits"
if str(TCC) not in sys.path:
    sys.path.insert(0, str(TCC))
OUT = HERE / "out" / "dunefsky_checks"
TC_SNAP = Path("/home/daniel/.cache/huggingface/hub/models--pchlenski--gpt2-transcoders/snapshots/"
               "4f3ca345572ce2d75018a60d10b4dca8bd899e99")
TC_FILE = "final_sparse_autoencoder_gpt2-small_blocks.%d.ln2.hook_normalized_24576.pt"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from sae_training.sparse_autoencoder import SparseAutoencoder  # noqa: E402
from transcoder_circuits.circuit_analysis import (  # noqa: E402
    Component, ComponentType, FeatureVector, add_error_nodes_to_graph, greedy_get_top_paths,
    make_sae_feature_vector, paths_to_graph, print_all_paths)

REL_TOL = 1e-3


# ------------------------------------------------------------------------------------------- expected (from notebooks)
def _cell_text(nb_name: str, idx: int) -> str:
    nb = json.loads((TCC / nb_name).read_text(encoding="utf8"))
    out = ""
    for o in nb["cells"][idx].get("outputs", []):
        out += "".join(o["text"]) if "text" in o else "".join(o.get("data", {}).get("text/plain", []))
    return out


def _parse_pairs(text: str):
    """lines 'key value' (nodes) or "('a', 'b') value" (edges) -> {key: float}"""
    res = {}
    for line in text.strip().splitlines():
        m = re.match(r"^\('(.+)', '(.+)'\) (\S+)$", line.strip())
        if m:
            res[(m.group(1), m.group(2))] = float(m.group(3))
            continue
        k, v = line.strip().rsplit(" ", 1)
        res[k] = float(v)
    return res


# ------------------------------------------------------------------------------------------- loading
def load_transcoder(layer: int) -> SparseAutoencoder:
    path = str(TC_SNAP / (TC_FILE % layer))
    if torch.cuda.is_available():
        return SparseAutoencoder.load_from_pretrained(path).eval()      # theirs, as shipped
    state = torch.load(path, map_location="cpu")                        # [ours] CPU: same lines, map_location
    state["cfg"].device = "cpu"
    inst = SparseAutoencoder(cfg=state["cfg"])
    inst.load_state_dict(state["state_dict"])
    return inst.eval()


def compare(got: dict, exp: dict) -> dict:
    missing = sorted(map(str, set(exp) - set(got)))
    extra = sorted(map(str, set(got) - set(exp)))
    worst, worst_key = 0.0, None
    for k in set(exp) & set(got):
        d = abs(got[k] - exp[k]) / max(abs(exp[k]), 1e-6)
        if d > worst:
            worst, worst_key = d, str(k)
    return dict(n_expected=len(exp), n_got=len(got), missing=missing, extra=extra, max_rel=worst,
                max_rel_key=worst_key, passed=not missing and not extra and worst <= REL_TOL)


def paths_text(all_paths) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_all_paths(all_paths)
    return buf.getvalue()


def text_compare(got: str, exp: str) -> dict:
    g, e = got.strip().splitlines(), exp.strip().splitlines()
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(g, e)) if a.strip() != b.strip()]
    return dict(lines_expected=len(e), lines_got=len(g), n_diff=len(diff) + abs(len(g) - len(e)),
                first_diffs=diff[:5], passed=not diff and len(g) == len(e))


# ------------------------------------------------------------------------------------------- checks
def walkthrough(model, checks) -> dict:
    tcs = [load_transcoder(l) for l in range(11)]                      # walkthrough cell 14: range(11)
    fv = make_sae_feature_vector(tcs[8], 89)                            # cell 52: live_features[77] == 89
    prompt = "Oh, that rifle model is a 6M"                             # cell 57
    t0 = time.time()
    _, cache = model.run_with_cache(prompt)
    all_paths = greedy_get_top_paths(model, tcs, cache, fv, num_iters=3, num_branches=15, do_raw_attribution=True)
    res = {"R1_seconds": time.time() - t0}
    if "R1" in checks:
        res["R1"] = text_compare(paths_text(all_paths), _cell_text("walkthrough.ipynb", 60))
        res["R1"]["root"] = float(all_paths[0][0][0].contrib)
    edges, nodes = paths_to_graph(all_paths)
    if "R2" in checks:
        got_e = {k: float(v) for k, v in edges.items()}
        got_n = {k: float(v.contrib) for k, v in nodes.items()}
        res["R2_edges"] = compare(got_e, _parse_pairs(_cell_text("walkthrough.ipynb", 68)))
        res["R2_nodes"] = compare(got_n, _parse_pairs(_cell_text("walkthrough.ipynb", 70)))
        keep = sorted({(n.component_path[-1].layer, n.component_path[-1].feature_idx) for k, n in nodes.items()
                       if k != "mlp8tc[89]@-1"
                       and n.component_path[-1].component_type == ComponentType.MLP})
        res["R2_keep_latents"] = dict(n=len(keep), latents=keep)
    if "R3" in checks:
        _, nodes_e = add_error_nodes_to_graph(model, cache, tcs, edges, nodes)
        got = {k: float(v.contrib) for k, v in nodes_e.items()}
        res["R3"] = compare(got, _parse_pairs(_cell_text("walkthrough.ipynb", 73)))
    return res


def greaterthan(model, checks) -> dict:
    tcs = [load_transcoder(l) for l in range(12)]                      # greaterthan cell 7: range(12)
    with torch.no_grad():                                               # cell 11-12 (one-hot built on the CPU)
        x = torch.zeros(model.W_U.shape[1])
        x[model.to_single_token("40")] = 1.0
        x[model.to_single_token("20")] -= 1.0
        x = model.W_U @ x
    fv = FeatureVector(component_path=[Component(layer=len(model.blocks) - 1, component_type=ComponentType.MLP,
                                                 token=-1)],
                       vector=x, layer=len(model.blocks) - 1, sublayer="resid_post", token=-1)
    res = {}
    for name, prompt, cell in (("R4", "The war lasted from 1735 to 17", 15), ("R5", "The war lasted from 1527 to 15", 21)):
        if name not in checks:
            continue
        _, cache = model.run_with_cache(prompt)
        all_paths = greedy_get_top_paths(model, tcs, cache, fv, num_iters=3, num_branches=15, do_raw_attribution=True)
        res[name] = text_compare(paths_text(all_paths), _cell_text("greaterthan.ipynb", cell))
    return res


def greaterthan_metric(model) -> dict:
    """R6, greaterthan cells 28-63, ported line for line to the CPU (the notebook helpers put tensors on "cuda"; the
    library's TranscoderReplacementContext and their SparseAutoencoder forward are used as shipped):
      cell 40  original model: mean over YY = 01..98 of P(> YY) - P(<= YY) over the 100 year tokens      0.8829426
      cell 62  MLP10 replaced by TC10 (TranscoderReplacementContext, error dropped), one prompt at a time  0.7978021
      cell 63  TC10 with every feature except 5315 (eval_tc_on_num: full replacement, batched prompts)    0.8453090
      cell 56  topk(var over the 100 prompts of tc10 feature acts at the last position)[22] == 5315
      cell 48  argmin |tc_eval - mlp_eval|[1:] == 24 (feature-vs-neuron crossing; extra, not a gate)"""
    from transformer_lens import utils
    from transcoder_circuits.replacement_ctx import TranscoderReplacementContext
    tc10 = load_transcoder(10)
    prompts = [f"The war lasted from 17{i:02} to 17" for i in range(100)]                      # cell 28
    yy = [model.to_single_token(f"{i:02}") for i in range(100)]
    res = {}
    with torch.no_grad():
        all_activs, mlp_activs = [], []
        for prompt in prompts:                                                                  # cells 28, 32
            _, cache = model.run_with_cache(prompt)
            all_activs.append(tc10(cache[utils.get_act_name("normalized", 10, "ln2")][:, -1])[1])
            mlp_activs.append(cache[utils.get_act_name("post", 10, "mlp")][:, -1])
        activs_tensor = torch.stack(all_activs)[:, 0, :]                                        # cell 30
        mlp_activs = torch.stack(mlp_activs)[:, 0, :]
        logits = model(prompts, return_type="logits")[:, -1, yy]                                # cells 39-40
        probs = torch.softmax(logits, dim=1)
        orig = float(np.mean([(probs[i, i + 1:].sum() - probs[i, :i + 1].sum()).item() for i in range(1, 99)]))
        prob_diffs = []                                                                         # cell 62
        with TranscoderReplacementContext(model, [tc10]):
            for i in range(1, 99):
                lg = model(prompts[i], return_type="logits")
                p = torch.nn.functional.softmax(lg[0, -1, yy], dim=-1)
                prob_diffs.append((p[i + 1:].sum() - p[:i + 1].sum()).item())
        tc_full = float(np.mean(prob_diffs))

        def total_diff(probsA, num):                                                            # cells 42-43 tail
            num = torch.tensor(num)
            new = [probsA[:, n + 1:].sum(dim=-1) - probsA[:, :n + 1].sum(dim=-1) for n in num]
            return float(torch.diagonal(torch.stack(new)).mean())

        def eval_tc_on_num(num, features_to_use):                                               # cell 43
            class TCReplacementModule(torch.nn.Module):
                def forward(self, x):
                    fa = tc10(x)[1][:, :, features_to_use]
                    return torch.einsum("btf, fd -> btd", fa, tc10.W_dec[features_to_use, :]) + tc10.b_dec_out
            original_mlp = model.blocks[10].mlp
            try:
                model.blocks[10].mlp = TCReplacementModule()
                lA = model([prompts[i] for i in num], return_type="logits")
            finally:
                model.blocks[10].mlp = original_mlp
            return total_diff(torch.softmax(lA[:, -1, yy], dim=-1), num)

        def eval_mlp_on_num(num, features_to_use):                                              # cell 42
            def mlp_replacement(acts, hook):
                new = torch.zeros(acts.shape, dtype=acts.dtype)
                new[:, :, features_to_use] = acts[:, :, features_to_use]
                return new
            lA = model.run_with_hooks([prompts[i] for i in num], return_type="logits",
                                      fwd_hooks=[(utils.get_act_name("post", 10, "mlp"), mlp_replacement)])
            return total_diff(torch.softmax(lA[:, -1, yy], dim=-1), num)

        feats = torch.arange(tc10.W_enc.shape[1])
        no5315 = eval_tc_on_num(list(range(1, 99)), feats[feats != 5315])                        # cell 63
        rank22 = int(torch.topk(torch.var(activs_tensor, dim=0), k=23).indices[22])             # cell 56
        top_f = torch.topk(torch.var(activs_tensor, dim=0), k=100).indices                      # cells 44-46
        top_n = torch.topk(torch.var(mlp_activs, dim=0), k=100).indices
        tc_eval = [eval_tc_on_num(list(range(1, 99)), top_f[:i]) for i in range(100)]
        mlp_eval = [eval_mlp_on_num(list(range(1, 99)), top_n[:i]) for i in range(100)]
        crossing = int(np.argmin(np.abs(np.array(tc_eval) - np.array(mlp_eval))[1:]))          # cell 48
        drop = int(np.argmin(np.diff(tc_eval)))                                                 # cell 53
    exp = dict(original=0.8829426078163848, tc10=0.7978020669246206, tc10_without_5315=0.8453089594841003)
    got = dict(original=orig, tc10=tc_full, tc10_without_5315=no5315)
    res = dict(got=got, expected=exp, abs_diff={k: abs(got[k] - exp[k]) for k in exp},
               variance_rank_22=rank22, crossing_index=crossing, crossing_expected=24, drop_index=drop,
               drop_expected=22, tc_eval_20_25=tc_eval[20:25],
               tc_eval_20_25_expected=[0.8435162901878357, 0.8432683944702148, 0.8314465880393982,
                                       0.7851703763008118, 0.786202073097229])
    res["passed"] = all(v <= 1e-3 for v in res["abs_diff"].values()) and rank22 == 5315
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="R1,R2,R3,R4,R5")
    a = ap.parse_args()
    checks = set(a.checks.split(","))
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")    # TL 1.11.0 defaults, as their notebooks
    res = dict(env=dict(torch=torch.__version__, cuda=torch.cuda.is_available()))
    if checks & {"R1", "R2", "R3"}:
        res.update(walkthrough(model, checks))
    if checks & {"R4", "R5"}:
        res.update(greaterthan(model, checks))
    if "R6" in checks:
        t0 = time.time()
        res["R6"] = greaterthan_metric(model)
        res["R6"]["seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    old = OUT / "results.json"                       # keep the rows of checks not re-run this time
    merged = json.loads(old.read_text()) if old.exists() else {}
    merged.update(res)
    old.write_text(json.dumps(merged, indent=1, default=str))
    for k, v in res.items():
        if isinstance(v, dict) and "passed" in v:
            print("%-16s %s  %s" % (k, "PASS" if v["passed"] else "FAIL",
                                    {kk: v[kk] for kk in v if kk in ("max_rel", "max_rel_key", "n_diff", "root",
                                                                       "missing", "extra", "got", "abs_diff",
                                                                       "variance_rank_22", "crossing_index",
                                                                       "drop_index")}))
    print("keep_latents:", res.get("R2_keep_latents"))


if __name__ == "__main__":
    main()
