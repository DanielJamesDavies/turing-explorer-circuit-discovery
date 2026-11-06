"""Arena C (DAN-196) cross-scoring (b): WCM's circuits under SFC's evaluation. Run in the SFC env (~/venvs/sfc107).

SFC's evaluation exactly as sfc_faithfulness.py transcribes it (their notebook's get_fcs, with run_with_ablations =
their function or the labelled issue #15 copy; imported, not re-implemented): Pythia-70m-deduped, the 18 non-embed SAEs
(load_saes_and_submodules(include_embed=False)), start_layer 2 (12 evaluated submodules), mean ablation over the 40
evaluation PATCH prompts per position, the 40 min-length test examples, metric logit[clean] - logit[patch];
faithfulness = (F(C) - F(empty)) / (F(M) - F(empty)), completeness = the same for the complement.

The only new part is the node mask [ours, labelled "WCM membership at alpha = 1 under SFC's evaluator"]:
  - act = True for every member latent at EVERY position, at the SFC submodule of its site (no embed members exist);
    SFC's evaluator has no coefficients, so members run at their live value (alpha = 1);
  - errors, two settings: errors_kept (resc all True: WCM's semantics; SFC counts these as nodes, so n_nodes is
    reported with and without them) and errors_removed (resc all False: SFC's features_wo_errs convention).

Inputs: circuit JSONs under each --wcm_out folder (circuits/**.json, alphas {"L/kind": {latent: alpha}}).
Output: out/cross/wcm_sfc.jsonl, one row per (circuit, errors); resumable; STOP_EPOCH respected.

  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ~/venvs/sfc107/bin/python experiments/107-true-baselines/arena_c/score_wcm_sfc.py \
      --wcm_out experiments/107-true-baselines/arena_c/out/wcm_tasks_zacr [--selftest]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FC = HERE.parents[2] / "third_party" / "feature-circuits"
sys.path.insert(0, str(HERE))
import sfc_faithfulness as SF  # noqa: E402  (sets sys.path to the clone, OURS-1 alias, the issue #15 wrapper)
import torch as t  # noqa: E402
from activation_utils import SparseAct  # noqa: E402
from data_loading_utils import load_examples  # noqa: E402
from dictionary_loading_utils import load_saes_and_submodules  # noqa: E402
from nnsight import LanguageModel  # noqa: E402

OUT = HERE / "out" / "cross"
START_LAYER = 2
ERRORS = ("errors_kept", "errors_removed")


def setup(device):
    model = LanguageModel("EleutherAI/pythia-70m-deduped", attn_implementation="eager", torch_dtype=t.float32,
                          device_map=device, dispatch=True)
    subs, dicts = load_saes_and_submodules(model, include_embed=False, dtype=t.float32, device=device)
    subs = [s for s in subs if int(s.name.split("_")[-1]) >= START_LAYER]
    return model, subs, dicts


class Task:
    """get_fcs's per-dataset part: examples, inputs, metric, F(M), F(empty) (computed once per task)."""

    def __init__(self, dataset, model, subs, dicts, device):
        self.model, self.subs, self.dicts, self.device = model, subs, dicts, device
        ex = load_examples(str(FC / "data" / ("%s_test.json" % dataset)), 40, model, use_min_length_only=True)
        self.n, self.T = len(ex), len(model.tokenizer(ex[0]["clean_prefix"]).input_ids)
        self.clean = [e["clean_prefix"] for e in ex]
        self.patch = [e["patch_prefix"] for e in ex]
        ci = t.tensor([model.tokenizer(e["clean_answer"]).input_ids[-1] for e in ex], dtype=t.long, device=device)
        pi = t.tensor([model.tokenizer(e["patch_answer"]).input_ids[-1] for e in ex], dtype=t.long, device=device)

        def metric_fn(model):
            logits = model.output.logits[:, -1, :]
            return (- t.gather(logits, dim=-1, index=pi.view(-1, 1)).squeeze(-1)
                    + t.gather(logits, dim=-1, index=ci.view(-1, 1)).squeeze(-1))

        self.metric_fn = metric_fn
        self.ablation_fn = lambda x: x.mean(dim=0).expand_as(x)       # noqa: E731  (mean ablation, the notebook's)
        with t.no_grad():
            with model.trace(self.clean):
                m = metric_fn(model).save()
            self.fm = m.value.mean().item()
            empty = {s: SparseAct(act=t.zeros(dicts[s].dict_size, dtype=t.bool),
                                  resc=t.zeros(1, dtype=t.bool)).to(device) for s in subs}
            self.fempty = self.run(empty).mean().item()

    def run(self, nodes, complement=False):
        return SF.run_with_ablations(self.clean, self.patch, self.model, self.subs, self.dicts,
                                     nodes=SF.clone_nodes(nodes), metric_fn=self.metric_fn,
                                     ablation_fn=self.ablation_fn, complement=complement)

    def evaluate(self, nodes):
        with t.no_grad():
            fc = self.run(nodes).mean().item()
            fcc = self.run(nodes, complement=True).mean().item()
        d = self.fm - self.fempty
        return dict(fm=self.fm, fempty=self.fempty, fc=fc, fccomp=fcc, faithfulness=(fc - self.fempty) / d,
                    completeness=(fcc - self.fempty) / d)


def wcm_nodes(alphas_json, task: Task, errors: str, everything: bool = False):
    """SparseAct masks per evaluated submodule: members at every position; resc all True (kept) or False (removed)."""
    nodes, n_feat, n_err = {}, 0, 0
    for s in task.subs:
        kind, layer = s.name.rsplit("_", 1)
        act = t.zeros(task.T, task.dicts[s].dict_size, dtype=t.bool)
        if everything:
            act[:] = True
        else:
            for i in (alphas_json.get("%s/%s" % (layer, kind)) or {}):
                act[:, int(i)] = True
        resc = t.full((task.T, 1), errors == "errors_kept", dtype=t.bool)
        nodes[s] = SparseAct(act=act, resc=resc).to(task.device)
        n_feat += int(act.sum())
        n_err += int(resc.sum())
    return nodes, n_feat, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wcm_out", nargs="*", default=[])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest", action="store_true", help="the WCM 'everything' circuit on every task (test 1)")
    ap.add_argument("--export", action="store_true",
                    help="write SFC's nodes-only circuits as plain tensors (out/cross/sfc_nodes_<task>.pt: {name: "
                         "{act, resc}}) for score_sfc_ours.py, whose env cannot unpickle their file (it imports nnsight)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.export:
        for name in ("simple", "rc", "nounpp", "within_rc"):
            src = HERE / "out" / "sfc" / "circuits" / ("pythia-70m-deduped_%s_train_n100_aggnone_nodeall.pt" % name)
            nodes = t.load(src, map_location="cpu")["nodes"]
            t.save(dict(source=str(src.relative_to(HERE)),
                        nodes={k: dict(act=v.act.float().cpu(), resc=v.resc.float().cpu()) for k, v in nodes.items()}),
                   out / ("sfc_nodes_%s.pt" % name))
            print("exported", name, {k: tuple(v.act.shape) for k, v in list(nodes.items())[:2]})
        return
    model, subs, dicts = setup(a.device)
    tasks = {}

    def task(name):
        if name not in tasks:
            tasks[name] = Task(name, model, subs, dicts, a.device)
        return tasks[name]

    if a.selftest:
        res = {}
        for name in ("simple", "rc", "nounpp", "within_rc"):
            tk = task(name)
            nodes, nf, ne = wcm_nodes({}, tk, "errors_kept", everything=True)
            res[name] = dict(tk.evaluate(nodes), n_feat=nf, n_err=ne)
        print("JSON" + json.dumps(res))
        return
    path = out / "wcm_sfc.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["circuit"], r["errors"]))
            except Exception:  # noqa: BLE001
                pass
    for od in map(Path, a.wcm_out):
        for cp in sorted((od / "circuits").rglob("*.json")):
            if os.environ.get("STOP_EPOCH") and time.time() >= float(os.environ["STOP_EPOCH"]):
                print("deadline passed: stopping", flush=True)
                return
            c = json.loads(cp.read_text())
            if c.get("data_mode", "sfc") != "sfc":
                continue
            tk = task(c["task"])
            n_lat = sum(len(d) for d in c["alphas"].values())
            n_lat_ge2 = sum(len(d) for k, d in c["alphas"].items() if int(k.split("/")[0]) >= START_LAYER)
            for err in ERRORS:
                key = (str(cp), err)
                if key in done:
                    continue
                ts = time.time()
                nodes, nf, ne = wcm_nodes(c["alphas"], tk, err)
                r = tk.evaluate(nodes)
                row = dict(method="WCM", label="WCM membership at alpha = 1 under SFC's evaluator", circuit=str(cp),
                           task=c["task"], arm=c["arm"], lam=c["lam"], steps=c["steps"], sites=c.get("sites"),
                           means=c.get("means", "pooled"), wcm=c.get("wcm"), errors=err, n_latents=n_lat,
                           n_latents_ge2=n_lat_ge2, T=tk.T, sfc_feature_nodes=nf, sfc_error_nodes=ne,
                           sfc_nodes_with_errors=nf + ne, sfc_nodes_without_errors=nf, start_layer=START_LAYER,
                           n_examples=tk.n, fix15_used=SF.fix15_used, secs=round(time.time() - ts, 2), **r)
                with open(path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print("%s %s %s: %d latents, SFC nodes %d (+%d err) | faith %.3f compl %.3f" % (
                    c["task"], cp.name, err, n_lat, nf, ne, r["faithfulness"], r["completeness"]), flush=True)


if __name__ == "__main__":
    main()
