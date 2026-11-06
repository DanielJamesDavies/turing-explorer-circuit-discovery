"""Arena C fidelity checks on the official SFC outputs (specs/sfc.md §10). Run in the SFC env, cwd = out/sfc.

  count  F3 node count both ways: T_N on |IE| over all 19 sites vs layers >= 2 (what the evaluator counts)
  diff   F6 / F10: max |ΔIE| between two nodes-only circuit files (two runs, or batch 10 vs batch 1)
  overlap F8: how many of the 40 evaluation examples (load_examples on <task>_test, min length) equal a discovery
         example (the 100 used by circuit.py on <task>_train), by clean_prefix string
"""
import argparse
import json
import os
import sys

FC = os.environ.get("SFC_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party/feature-circuits"))
sys.path.insert(0, os.path.abspath(FC))
import torch as t


def count(path, threshold):
    nodes = t.load(path)["nodes"]
    out = {"all_sites": {"act": 0, "resc": 0}, "layers_ge_2": {"act": 0, "resc": 0}, "per_site": {}}
    for name, sa in nodes.items():
        a = int((sa.act.abs() > threshold).sum())
        r = int((sa.resc.abs() > threshold).sum())
        out["per_site"][name] = a + r
        out["all_sites"]["act"] += a
        out["all_sites"]["resc"] += r
        if name != "embed" and int(name.split("_")[-1]) >= 2:
            out["layers_ge_2"]["act"] += a
            out["layers_ge_2"]["resc"] += r
        # distinct latents, positions collapsed (our size axis, specs/evaluation.md)
        out.setdefault("distinct_latents_all_sites", 0)
        out["distinct_latents_all_sites"] += int(((sa.act.abs() > threshold).any(dim=0)).sum()) if sa.act.dim() > 1 else a
    for k in ("all_sites", "layers_ge_2"):
        out[k]["total"] = out[k]["act"] + out[k]["resc"]
    out["shape_example"] = {n: list(sa.act.shape) for n, sa in list(nodes.items())[:2]}
    return out


def diff(a, b, threshold=0.1):
    """max |ΔIE|, how widespread the differences are, and whether the T_N node set changes."""
    na, nb = t.load(a)["nodes"], t.load(b)["nodes"]
    m, n_tot, n_1e4, n_1e3, worst = 0.0, 0, 0, 0, None
    set_a = set_b = 0
    sym = 0
    for k in na:
        for part in ("act", "resc"):
            x, y = getattr(na[k], part), getattr(nb[k], part)
            d = (x - y).abs()
            n_tot += d.numel()
            n_1e4 += int((d > 1e-4).sum())
            n_1e3 += int((d > 1e-3).sum())
            if float(d.max()) > m:
                m = float(d.max())
                i = int(d.argmax())
                worst = {"site": k, "part": part, "a": float(x.flatten()[i]), "b": float(y.flatten()[i])}
            ka, kb = x.abs() > threshold, y.abs() > threshold
            set_a += int(ka.sum())
            set_b += int(kb.sum())
            sym += int((ka ^ kb).sum())
    return {"max_abs_dIE": m, "worst": worst, "n_entries": n_tot, "n_diff_gt_1e-4": n_1e4, "n_diff_gt_1e-3": n_1e3,
            f"nodes_at_T{threshold}": [set_a, set_b], "node_set_symmetric_difference": sym}


def overlap(task, n_disc=100, n_eval=40):
    from nnsight import LanguageModel
    from data_loading_utils import load_examples
    model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="cpu", dispatch=True)
    disc = load_examples(f"data/{task}_train.json", n_disc, model, use_min_length_only=True)
    ev = load_examples(f"data/{task}_test.json", n_eval, model, use_min_length_only=True)
    ds = {e["clean_prefix"] for e in disc}
    train_all = {json.loads(l)["clean_prefix"] for l in open(f"data/{task}_train.json")}
    return {"task": task, "n_discovery": len(disc), "n_eval": len(ev),
            "eval_in_discovery_set": sum(e["clean_prefix"] in ds for e in ev),
            "eval_in_train_file": sum(e["clean_prefix"] in train_all for e in ev)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["count", "diff", "overlap"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--threshold", type=float, default=0.1)
    a = ap.parse_args()
    if a.what == "count":
        r = count(a.args[0], a.threshold)
    elif a.what == "diff":
        r = diff(a.args[0], a.args[1])
    else:
        r = [overlap(task) for task in (a.args or ["rc", "nounpp", "simple", "within_rc"])]
    print(json.dumps(r, indent=1))
