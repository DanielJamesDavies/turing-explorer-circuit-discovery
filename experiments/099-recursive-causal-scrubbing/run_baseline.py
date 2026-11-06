"""Clean, isolated WCM pilot. Run from repository root with WSL's .venv.

The background experiment is explicitly NOT full recursive causal scrubbing.
New fits use 094 fresh records; no old circuit is imported. Outputs are resumable
only when their exact source/context/config manifest agrees.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(actual, natural, scale):
    import torch
    a, n = actual.float().flatten(), natural.float().flatten()
    ac, nc = a - a.mean(), n - n.mean()
    denominator = float(ac.norm() * nc.norm())
    mse = float(((a - n) ** 2).mean())
    return dict(mean=float(a.mean()), natural_mean=float(n.mean()), mse=mse,
                nrmse=(mse / max(scale, 1e-8)) ** 0.5,
                correlation=float(ac @ nc) / denominator if denominator > 1e-12 else None,
                firing_recall=float((a[n > 0] > 0).float().mean()) if (n > 0).any() else None,
                false_firing=float((a[n <= 0] > 0).float().mean()) if (n <= 0).any() else None,
                raw=a.detach().cpu().tolist(), natural_raw=n.detach().cpu().tolist())


def finite_median(values):
    import math
    import statistics
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def main(runner=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE / "targets_first3.txt")
    parser.add_argument("--contexts", type=Path, default=ROOT / "experiments/094-tl-v2-records/out/smoke10/ctx_v2")
    parser.add_argument("--out", type=Path, default=HERE / "out/first3")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--unweighted", action="store_true", help="also fit a separate unweighted support")
    parser.add_argument("--ig", action="store_true", help="adapted IG, aggregate ranking, matched WCM node count")
    args = parser.parse_args()
    if args.draws < 1 or args.steps < 1:
        parser.error("steps/draws must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OUT=str(args.out / "driver"), CTX_DIR=str(args.contexts),
                      WCM_VERSION="v2", EVAL_SPLIT="explicit", CURVE_STRIDE="20")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/073-strict-eval")]
    import numpy as np
    import torch
    import strict_eval as SE
    from circuit.protocol_contexts import eval_probe, training_probe, training_set
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import NaturalTap, circuit_alphas, contrast_split, site_encoder

    keys = args.targets.read_text().split()
    source_paths = sorted(set(ROOT.joinpath("src").rglob("*.py")) | set(HERE.glob("*.py")) |
                          {ROOT / "experiments/062-h100-protocol-v1/driver.py",
                           ROOT / "experiments/073-strict-eval/strict_eval.py",
                           ROOT / "experiments/094-tl-v2-records/build_records.py",
                           ROOT / "config.yaml"})
    source_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    manifest = dict(protocol="099-clean-background/v1", source=source_hashes, targets=keys,
                    contexts={key: file_hash(args.contexts / (key + ".pt")) for key in keys},
                    steps=args.steps, draws=args.draws, unweighted=args.unweighted, ig=args.ig, random_seed=20261028,
                    python=platform.python_version(), torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    ablations="production v2; donor fill preserves Top-K; live SAE error",
                    clean_scope="094 smoke records; exact/apfx/near rule, no source-document rule")
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("manifest changed; use a new output directory rather than reusing fits")
    write_json(manifest_path, manifest)
    torch.manual_seed(20261028)
    print("099: loading model; targets", keys, flush=True)
    R = SE.driver.Runner() if runner is None else runner
    if Path(R.ctx_dir).resolve() != args.contexts.resolve() or R.wcm_version != "v2":
        raise RuntimeError("resident runner does not match context directory/protocol")
    G, bank, dev = R.G, R.G["bank"], R.G["device"]
    inf = G["inference"]
    builder = load_module("fresh099", ROOT / "experiments/094-tl-v2-records/build_records.py")
    import rebuild as RB
    from config import config
    corpus = RB.Corpus(ROOT / "experiments/085-heldout-audit/out")
    # configure() makes this identical to the engine's actual split convention.
    M = R.method(SE.driver.GAMMA, SE.driver.LAM, True, steps=args.steps)

    def taps(key, tokens):
        layer, kind, idx = key.split(".")
        layer, idx = int(layer), int(idx)
        W, b = site_encoder(bank, layer, kind)
        return SE.run_full(inf, lambda a, z: NaturalTap(bank, (layer, kind), W[idx], b[idx], idx),
                           tokens.to(dev), pre=True)

    class AuditForward:
        def acts(self, key, tokens):
            return taps(key, tokens)[0].cpu()

    audits = []
    records = {}
    for key in keys:
        rec = R.contexts(key)
        records[key] = rec
        warnings = []
        failures = builder.verify_one(key, rec, corpus, AuditForward(), config.discovery.context_v1, warnings)
        arm, thin = R.train_arm(rec)
        probe = training_probe(rec, "cpu", arm)
        expected = int(probe.metadata["n_train"])
        actual = len(probe.pos_tokens) - round(len(probe.pos_tokens) * config.discovery.learned_mask.holdout_frac)
        if expected != actual:
            failures.append(f"engine actually consumes {actual} rows, record declares {expected}")
        if not torch.equal(probe.pos_tokens[:expected], training_set(rec, arm)[0]):
            failures.append("engine training prefix differs from training_set")
        audits.append(dict(target=key, failures=failures, warnings=warnings, expected_train=expected,
                           actual_train=actual, arm=arm, thin=thin))
        write_json(args.out / "audit.json", audits)
        print("AUDIT", key, "PASS" if not failures else failures, flush=True)
    if any(row["failures"] for row in audits):
        raise RuntimeError("record audit failed; no fitting performed")
    if args.audit_only:
        return

    # Use an existing deterministic corpus pool, but remove exact/near copies of
    # any pilot record. All retained donor tensors are fingerprinted below.
    pool_path = ROOT / "experiments/073-strict-eval/out/random_pool.pt"
    pool = torch.load(pool_path, weights_only=False)["tokens"].cpu()
    from circuit import heldout_rule as HR
    all_rows = torch.cat([d["pos"] for rec in records.values() for d in [rec["strong"], rec.get("mid")] if d is not None]
                         + [rec["neg"] for rec in records.values()]).numpy()
    keep = []
    for j, row in enumerate(pool.numpy()):
        ex, _, lcr, c8 = HR.pair_stats(row[None], all_rows)
        _, _, _, c8r = HR.pair_stats(all_rows, row[None])
        if not (ex.any() or HR.near_of(lcr, c8).any() or (c8r >= 0.5).any()):
            keep.append(j)
    pool = pool[keep]
    if len(pool) < 16:
        raise RuntimeError("too few independent random donors")
    torch.save(dict(tokens=pool, source_sha256=file_hash(pool_path), retained_indices=keep), args.out / "random_pool.pt")
    print("random donors retained", len(pool), flush=True)
    done_path = args.out / "results.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(line)["target"] for line in done_path.read_text().splitlines() if line.strip()}

    for key in keys:
        if key in done:
            continue
        t0 = time.time()
        rec = records[key]
        arm, _ = R.train_arm(rec)
        artifacts = {}
        for name, weighted in [("wcm", True)] + ([("unweighted", False)] if args.unweighted else []):
            method = M if weighted else R.method(SE.driver.GAMMA, SE.driver.LAM, False, steps=args.steps)
            path = args.out / "circuits" / f"{key}.{name}.pt"
            with open(args.out / "train.jsonl", "a") as fh:
                circuit, secs = R.fit(method, rec, key, arm, path, train_fh=fh, label=dict(pilot="099", method=name))
            if circuit is None:
                raise RuntimeError(f"fit rejected {key}/{name}")
            artifacts[name] = circuit
            print("FIT", key, name, round(secs, 1), "seconds", flush=True)
        if args.ig:
            sys.path.insert(0, str(ROOT / "experiments/083-v1-comparison"))
            import attr_baselines as AB
            import cmp_common as C
            AB.NATURAL, AB.IG_STEPS = False, 10
            pos, _, arg = training_set(rec, arm)
            t_attr = time.time()
            payload = AB.attribute(G, key, "abl_ig", pos.to(dev), arg.to(dev))
            n = sum(len(v) for v in C.members_by_site(artifacts["wcm"]).values())
            members = C.truncate_global(payload["rankings"]["sfc"], payload["sites"], n)
            artifacts["ig_a1"] = C.make_circuit(key, members, name="099_ig_aggregate")
            rank_path = args.out / "rankings" / (key + ".pt")
            rank_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, rank_path)
            print("ATTR", key, "IG aggregate", C.n_members(members), "nodes", round(time.time() - t_attr, 1), "seconds", flush=True)
        # Other methods can reconfigure global config; restore WCM convention.
        M = R.method(SE.driver.GAMMA, SE.driver.LAM, True, steps=args.steps)
        prod = {name: R.score(c, rec, "strong") for name, c in artifacts.items()}
        layer, kind, idx = key.split(".")
        layer, idx = int(layer), int(idx)
        site = layer, kind
        W, b = site_encoder(bank, layer, kind)
        sites = sorted(upstream_sites(bank, layer, kind))
        probe = eval_probe(rec, "strong", dev)
        ntr = int(probe.metadata["n_train"])
        nneg = contrast_split(len(probe.neg_tokens), probe.metadata)
        base, anchors = probe.pos_tokens[ntr:], probe.pos_argmax[ntr:]
        silent = probe.neg_tokens[nneg:]
        bases = torch.cat([base, silent])
        anchors = torch.cat([anchors, torch.zeros(len(silent), dtype=anchors.dtype, device=dev)])
        natural, natural_pre = taps(key, bases)
        train_tokens = training_set(rec, arm)[0].to(dev)
        train_nat, _ = taps(key, train_tokens)
        train_anchors = training_set(rec, arm)[2].to(dev)
        scale = float((SE.at(train_nat, train_anchors) ** 2).mean())
        natural_anchors = SE.at(natural[:len(base)], anchors[:len(base)])
        base_codes = SE.capture_codes(inf, bank, bases, sites)
        variants = {"empty": ({}, {})}
        for name, c in artifacts.items():
            al = circuit_alphas(c)
            kt = {s: torch.tensor(sorted(v), device=dev) for s, v in al.items() if v}
            sc = {s: torch.tensor([al[s][int(i)] for i in kt[s].tolist()], device=dev) for s in kt}
            variants[name] = kt, sc
            if name == "wcm":
                variants["wcm_a1"] = kt, {}
        pins = {}
        for s, kt in variants["wcm"][0].items():
            acts, indices = base_codes[s]
            pins[s] = (acts.unsqueeze(-1) * (indices.unsqueeze(-1) == kt).to(acts.dtype)).sum(-2)

        def execute(donors, kt, sc, pin=False):
            return SE.run_full(inf, lambda a, z: SE.ResamplePatcher(
                bank, set(sites), site, W[idx], b[idx], idx, kt, sc,
                {s: (v[a:z], i[a:z]) for s, (v, i) in donors.items()}, bank.k,
                pins={s: v[a:z] for s, v in pins.items()} if pin else None), bases, pre=True)

        results = []
        donor_sets = {"A": probe.pos_tokens[:ntr], "C": silent, "R": pool.to(dev)}
        rng = torch.Generator().manual_seed(20261028 + layer * 1000 + idx)
        for ds, tokens in donor_sets.items():
            codes = SE.capture_codes(inf, bank, tokens, sites)
            for draw in range(args.draws):
                # Without replacement inside each activating/silent half when possible.
                selection = torch.cat([torch.randperm(len(tokens), generator=rng)[:len(base)],
                                       torch.randperm(len(tokens), generator=rng)[:len(silent)]]).to(dev)
                if len(selection) != len(bases):
                    raise RuntimeError("donor set too small")
                donor = {s: (v[selection], i[selection]) for s, (v, i) in codes.items()}
                empty_mean = None
                for name, (kt, sc) in list(variants.items()) + [("wcm_pinned", variants["wcm"])]:
                    out, pre = execute(donor, kt, sc, pin=name == "wcm_pinned")
                    at = SE.at(out[:len(base)], anchors[:len(base)])
                    if name == "empty":
                        empty_mean = float(at.mean())
                    row = dict(background=ds, draw=draw, variant=name,
                               donor_indices=selection.cpu().tolist(),
                               anchor=metrics(at, natural_anchors, scale),
                               all_positions=metrics(out, natural, scale),
                               pre_relu=metrics(pre, natural_pre, scale),
                               silent_firing=float((out[len(base):] > 0).float().mean()),
                               recovery=SE.ratio(float(at.mean()), empty_mean, float(natural_anchors.mean())))
                    results.append(row)
                del donor
            del codes
        identities = {}
        for name in ("empty", "wcm_a1", "wcm"):
            out, pre = execute(base_codes, *variants[name])
            identities[name] = dict(max_abs=float((out - natural).abs().max()),
                                     anchor=metrics(SE.at(out[:len(base)], anchors[:len(base)]), natural_anchors, scale))
        # bf16 SAE decode/add roundoff is measured, rather than hidden in a ratio.
        row = dict(target=key, production=prod, train_scale_squared=scale,
                   identity=identities, donor_results=results, seconds=time.time() - t0,
                   context_sha256=manifest["contexts"][key])
        write_json(args.out / "targets" / (key + ".json"), row)
        summary = dict(target=key, seconds=row["seconds"],
                       production={name: {k: v for k, v in p.items() if isinstance(v, (str, float, int, bool)) or v is None}
                                   for name, p in prod.items()},
                       identity={name: value["max_abs"] for name, value in identities.items()},
                       donor_median={f"{ds}/{name}": finite_median([r["recovery"] for r in results
                           if r["background"] == ds and r["variant"] == name])
                           for ds in donor_sets for name in variants})
        with open(done_path, "a") as fh:
            fh.write(json.dumps(summary, allow_nan=False) + "\n")
        print("DONE", key, summary["donor_median"], flush=True)
        del base_codes, pins
    changed = [name for name, digest in source_hashes.items() if file_hash(ROOT / name) != digest]
    write_json(args.out / "source_check.json", dict(changed_during_run=changed))
    if changed:
        raise RuntimeError("source changed during run: " + str(changed))


if __name__ == "__main__":
    main()
