"""Full recursion for a COARSE, TWO-RESIDUAL-CUT WCM-induced hypothesis.

This is not recursive validation of every individual WCM node. At residual
boundaries 0 and 1, the selected WCM groups are hypothesised to transmit the
binary target-firing state. Every other channel (including SAE error) is an
explicit, unimportant remainder. Composite native model segments are exact.

Both gain conventions are exact decompositions of the ORIGINAL model:
    r = phi_alpha(r) + (r - phi_alpha(r)).
The remainder includes the compensating gain term. No model weight is changed.
The interpretation's target-state categories are teacher-derived, not semantic
labels; this is a representational-invariance hypothesis, not a semantic proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

from scrubbing import Graph, Hypothesis, Node, Scrubber
from run_baseline import ROOT, HERE, file_hash, write_json


def build_graph(prefix, middle, suffix, member0, member1, labels):
    g = Graph([
        Node("x"), Node("raw0", ("x",), prefix),
        Node("members0", ("raw0",), member0),
        Node("remainder0", ("raw0",), lambda r: r.float() - member0(r)),
        Node("state0", ("members0", "remainder0"), lambda a, b: (a + b).to(dtype=prefix.dtype)),
        Node("raw1", ("state0",), middle), Node("members1", ("raw1",), member1),
        Node("remainder1", ("raw1",), lambda r: r.float() - member1(r)),
        Node("state1", ("members1", "remainder1"), lambda a, b: (a + b).to(dtype=prefix.dtype)),
        Node("prediction", ("state1",), suffix),
        Node("loss", ("x", "prediction"), lambda x, p: float(int(p > 0) != labels[x])),
    ], "x", "loss")
    i = Graph([
        Node("x"), Node("raw0", ("x",), lambda x: labels[x]),
        Node("members0", ("raw0",), lambda x: x), Node("state0", ("members0",), lambda x: x),
        Node("raw1", ("state0",), lambda x: x), Node("members1", ("raw1",), lambda x: x),
        Node("state1", ("members1",), lambda x: x), Node("prediction", ("state1",), lambda x: x),
        Node("loss", ("x", "prediction"), lambda x, p: float(p != labels[x])),
    ], "x", "loss")
    return Hypothesis(g, i, {n: n for n in i.nodes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE / "out/first3")
    parser.add_argument("--target", default="2.resid.7349")
    parser.add_argument("--contexts", type=Path, default=ROOT / "experiments/094-tl-v2-records/out/smoke10/ctx_v2")
    parser.add_argument("--out", type=Path, default=HERE / "out/recursive_two_cut")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--max-base-per-class", type=int, default=4)
    parser.add_argument("--diagnostics", action="store_true", help="restore each cut in turn, and test the full SAE reconstruction")
    args = parser.parse_args()
    if args.draws < 1 or args.max_base_per_class < 1:
        parser.error("draw and base counts must be positive")
    import os
    os.environ.update(OUT=str(args.out / "driver"), WCM_VERSION="v2", EVAL_SPLIT="explicit",
                      CTX_DIR=str(args.contexts))
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/073-strict-eval")]
    import numpy as np
    import torch
    import strict_eval as SE
    from circuit.protocol_contexts import eval_probe, training_set
    from eval.wcm_eval import circuit_alphas, site_encoder
    from sae.dense import target_latent_activations
    from torch.nn.attention import sdpa_kernel, SDPBackend

    layer, kind, target_idx = args.target.split(".")
    target_idx = int(target_idx)
    if layer != "2" or kind != "resid":
        raise ValueError("two-cut adapter requires a layer-2 residual target")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads((args.baseline / "manifest.json").read_text())
    context_file = Path(os.environ["CTX_DIR"]) / (args.target + ".pt")
    if file_hash(context_file) != baseline_manifest["contexts"][args.target]:
        raise RuntimeError("context differs from fitted circuit")
    circuit_file = args.baseline / "circuits" / (args.target + ".wcm.pt")
    R = SE.driver.Runner()
    inf, bank, dev = R.G["inference"], R.G["bank"], R.G["device"]
    inf.disable_compile()
    model = inf.model
    rec = R.contexts(args.target)
    arm, _ = R.train_arm(rec)
    train = training_set(rec, arm)[0].to(dev)
    probe = eval_probe(rec, "strong", dev)
    ntr = int(probe.metadata["n_train"])
    from eval.wcm_eval import contrast_split
    nneg = contrast_split(len(probe.neg_tokens), probe.metadata)
    tokens = torch.cat([probe.pos_tokens[ntr:], probe.neg_tokens[nneg:]])
    alphas = circuit_alphas(torch.load(circuit_file, weights_only=False))
    site_counts = {str(s): len(alphas.get(s, {})) for s in [(0, "resid"), (1, "resid")]}
    if any(n == 0 for n in site_counts.values()):
        raise RuntimeError("selected residual group missing at one boundary; this hypothesis is inapplicable")

    def embed(t):
        pos = torch.arange(t.shape[1], device=dev)
        return model.transformer.wte(t) + model.transformer.wpe(pos)

    def natural_dense(t):
        x = embed(t)
        for layer in range(3):
            _, _, x = model.transformer.h[layer](x, None)
        return x

    def read(x, pos):
        a, i = bank.encode(x, "resid", 2)
        return target_latent_activations(a, i, target_idx)[:, pos]

    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        # Freeze read position using TRAINING examples only, then evaluate on
        # all 32 held-out contexts at that one position. No anchor misalignment.
        counts = torch.zeros(64, device=dev)
        for start in range(0, len(train), 8):
            a, i = bank.encode(natural_dense(train[start:start + 8]), "resid", 2)
            counts += (target_latent_activations(a, i, target_idx) > 0).sum(0)
        position = int(counts.argmax())
        # Independent full-inference references, in the same one-example shape
        # used by G. These do not call our segmented transformer functions.
        from eval.wcm_eval import NaturalTap
        W, b = site_encoder(bank, 2, "resid")
        clean = []
        for t in tokens:
            output = SE.run_full(inf, lambda a, z: NaturalTap(bank, (2, "resid"), W[target_idx], b[target_idx], target_idx), t[None])
            clean.append(float(output[0, position]))
        inf.disable_compile()
        labels = [int(value > 0) for value in clean]
        if min(labels.count(0), labels.count(1)) < 2:
            write_json(args.out / "coverage_failure.json", dict(position=position, labels=labels))
            raise RuntimeError("fewer than two held-out donors in a class; no informative recursion")
        base_ids = sum(([j for j, label in enumerate(labels) if label == value][:args.max_base_per_class]
                        for value in (0, 1)), [])
        write_json(args.out / "setup.json", dict(position=position, labels=labels, clean=clean,
                                                  base_ids=base_ids, site_counts=site_counts))

        def prefix(index):
            _, _, x = model.transformer.h[0](embed(tokens[index:index + 1]), None)
            return x
        prefix.dtype = next(model.parameters()).dtype

        def middle(x):
            return model.transformer.h[1](x, None)[2]

        def suffix(x):
            return float(read(model.transformer.h[2](x, None)[2], position)[0])

        def selected(layer, gain="a1", random_support=False):
            a = alphas[(layer, "resid")]
            ids = torch.tensor(sorted(a), device=dev)
            if random_support:
                # Active-on-training control; equal cardinality, no fitted gains.
                active = set()
                for start in range(0, len(train), 8):
                    x = embed(train[start:start + 8])
                    for l in range(layer + 1):
                        _, _, x = model.transformer.h[l](x, None)
                    values, indices = bank.encode(x, "resid", layer)
                    active.update(indices[values > 0].cpu().tolist())
                pool = sorted(active - set(a))
                if len(pool) < len(ids):
                    raise RuntimeError("random support cardinality unavailable")
                ids = torch.tensor(random.Random(20261028 + layer).sample(pool, len(ids)), device=dev)
            scales = torch.tensor([a[int(j)] for j in ids] if gain == "fit" else [1.0] * len(ids), device=dev)
            weight = bank.saes["resid"][layer].decoder.weight
            decoder = weight[:, ids.to(weight.device)].to(device=dev, dtype=torch.float32)

            def contribution(x):
                values, indices = bank.encode(x, "resid", layer)
                picked = (values.float().unsqueeze(-1) * (indices.unsqueeze(-1) == ids)).sum(-2)
                return (picked * scales) @ decoder.T
            contribution.ids = ids.cpu().tolist()
            return contribution

        controls = {
            "full_state": (lambda x: x.float(), lambda x: x.float()),
            "empty": (lambda x: torch.zeros_like(x).float(), lambda x: torch.zeros_like(x).float()),
            "wcm_a1": (selected(0), selected(1)),
            "wcm_fit": (selected(0, "fit"), selected(1, "fit")),
            "active_random_a1": (selected(0, random_support=True), selected(1, random_support=True)),
        }
        if args.diagnostics:
            controls["first_cut_full"] = (lambda x: x.float(), selected(1))
            controls["second_cut_full"] = (selected(0), lambda x: x.float())
            def full_sae(layer):
                def reconstruct(x):
                    from sae.dense import sparse_topk_to_dense
                    values, indices = bank.encode(x, "resid", layer)
                    dense = sparse_topk_to_dense(values, indices, bank.d_sae)
                    return bank.decode(dense, "resid", layer).float()
                return reconstruct
            controls["full_sae_reconstruction"] = (full_sae(0), full_sae(1))
        manifest = dict(target=args.target, hypothesis="binary target-state transmission at residual cuts 0 and 1",
                        scope="two WCM-selected residual groups; composite native model segments; explicit resampled remainders including errors",
                        position=position, position_selection="maximum training firing count, lowest index breaks ties",
                        dataset="held-out strongest plus held-out contrast, 32 contexts", labels=labels, clean=clean,
                        base_ids=base_ids, draws=args.draws, dataset_n=len(tokens), site_counts=site_counts,
                        diagnostics=args.diagnostics,
                        circuit_sha256=file_hash(circuit_file), contexts_sha256=file_hash(context_file),
                        engine_sha256=file_hash(HERE / "scrubbing.py"), adapter_sha256=file_hash(Path(__file__)),
                        teacher_categories="binary clean-model firing at fixed position; no human semantic interpretation",
                        gain_convention="both arms are exact decompositions of the original model",
                        remainder_convention="raw residual minus selected decoded contribution, including compensation and SAE error",
                        membership={name: [getattr(m, "ids", None) for m in members] for name, members in controls.items()})
        manifest_path = args.out / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("recursive manifest changed; choose a new output directory")
        write_json(manifest_path, manifest)
        rows = []
        for name, (m0, m1) in controls.items():
            start = time.time()
            h = build_graph(prefix, middle, suffix, m0, m1, labels)
            scrubber = Scrubber(h, list(range(len(tokens))))
            # Independent original-model reference: full inference, same shape.
            maximum_delta, changed_labels = 0.0, 0
            for j in range(len(tokens)):
                graph_prediction = h.G.evaluate(j, "prediction")
                maximum_delta = max(maximum_delta, abs(graph_prediction - clean[j]))
                changed_labels += int((graph_prediction > 0) != bool(labels[j]))
            if changed_labels or maximum_delta > 0.03:
                raise RuntimeError(f"unintervened graph equality failed for {name}: {maximum_delta}, {changed_labels}")
            outcomes, traces = [], []
            for base in base_ids:
                for draw in range(args.draws):
                    value, trace = scrubber.run(base, seed=20261028 + base * 1000 + draw)
                    if draw == 0:
                        replay = scrubber.replay_tree(trace)
                        if replay != value:
                            raise RuntimeError("real-model recursive/treeified output disagreement")
                    outcomes.append(dict(reference=base, label=labels[base], draw=draw, loss=value))
                    trace_summary = trace.summary()
                    informative = [r for r in trace.matches if r["node"] != "x"]
                    trace_summary["noninput_support_min"] = min(r["support"] for r in informative)
                    trace_summary["noninput_self_fraction"] = sum(r["reference"] == r["donor"] for r in informative) / len(informative)
                    traces.append(trace_summary)
            by_class = {str(label): float(np.mean([r["loss"] for r in outcomes if r["label"] == label]))
                        for label in (0, 1)}
            row = dict(arm=name, class_disagreement=by_class,
                       balanced_disagreement=sum(by_class.values()) / 2,
                       natural_max_abs_delta=maximum_delta, natural_changed_labels=changed_labels,
                       cases=outcomes, coverage=traces, seconds=time.time() - start)
            rows.append(row)
            write_json(args.out / "results.json", rows)
            print("RECURSIVE", name, "balanced disagreement", row["balanced_disagreement"],
                  "class", by_class, "seconds", round(row["seconds"], 1), flush=True)
            del scrubber
        if rows[0]["balanced_disagreement"] != 0:
            raise RuntimeError("full-state control failed; interpret adapter before WCM scores")


if __name__ == "__main__":
    main()
