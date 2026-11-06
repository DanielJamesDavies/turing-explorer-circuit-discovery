"""Post-panel diagnostics, sharing one resident model across recursive jobs.

These cut-restoration diagnostics were chosen after seeing the first development
case. The layer-2 check target was frozen in panel.json before its scores.
"""
import json
import os
from pathlib import Path
import sys
import traceback

from run_baseline import HERE, ROOT, write_json


def roundtrip_checks(runner):
    import torch
    import strict_eval as SE
    from circuit.protocol_contexts import eval_probe
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import NaturalTap, TapCircuitOnly, site_encoder
    inf, bank, dev = runner.G["inference"], runner.G["bank"], runner.dev
    rows = []
    for key in ("2.resid.7349", "6.mlp.209", "10.attn.39376"):
        layer, kind, idx = key.split(".")
        layer, idx = int(layer), int(idx)
        p = eval_probe(runner.contexts(key), "strong", dev)
        n = int(p.metadata["n_train"])
        tokens, anchors = p.pos_tokens[n:], p.pos_argmax[n:]
        W, b = site_encoder(bank, layer, kind)
        natural = SE.run_full(inf, lambda a, z: NaturalTap(bank, (layer, kind), W[idx], b[idx], idx), tokens)
        sites = set(upstream_sites(bank, layer, kind))
        all_kept = {s: torch.arange(bank.d_sae, device=dev) for s in sites}
        out = SE.run_full(inf, lambda a, z: TapCircuitOnly(
            bank=bank, keep_indices={s: set() for s in sites}, in_scope=sites,
            seed_layer=layer, seed_kind=kind, seed_latent_idx=idx, pos_argmax=anchors[a:z],
            keep_tensors=all_kept, w_seed=W[idx], b_seed=b[idx]), tokens)
        target, clean = SE.at(out, anchors), SE.at(natural, anchors)
        row = dict(target=key, control="all latents retained at natural strength, same decode/error round trip",
                   max_all_position_abs=float((out - natural).abs().max()),
                   anchor_relative_rmse=float(((target - clean) ** 2).mean().sqrt() / clean.square().mean().sqrt()),
                   anchor_mean=float(target.mean()), natural_anchor_mean=float(clean.mean()))
        rows.append(row)
        print("ROUNDTRIP", row, flush=True)
    write_json(HERE / "out/identity_roundtrip.json", rows)


def main():
    records = HERE / "out/records/ctx_v2"
    baseline = HERE / "out/panel18"
    os.environ.update(OUT=str(HERE / "out/finish_driver"), CTX_DIR=str(records), WCM_VERSION="v2", EVAL_SPLIT="explicit")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/073-strict-eval")]
    import strict_eval as SE
    import run_recursive_model
    runner = SE.driver.Runner()
    roundtrip_checks(runner)
    factory = SE.driver.Runner
    # Scoped factory injection avoids repeatedly loading the 15 GB checkpoint
    # files. All jobs use this same model, bank, fresh records and protocol.
    SE.driver.Runner = lambda: runner
    status = []
    try:
        for name, target in [("development", "2.resid.7349"), ("check", "2.resid.29259")]:
            out = HERE / "out" / ("recursive_" + name + "_diagnostics")
            sys.argv = ["run_recursive_model.py", "--baseline", str(baseline), "--contexts", str(records),
                        "--target", target, "--out", str(out), "--draws", "32", "--max-base-per-class", "32", "--diagnostics"]
            try:
                run_recursive_model.main()
                status.append(dict(stage=name, target=target, status="complete", out=str(out)))
            except Exception as exc:
                traceback.print_exc()
                status.append(dict(stage=name, target=target, status="failed", error=str(exc), out=str(out)))
            write_json(HERE / "out/finish_status.json", status)
    finally:
        SE.driver.Runner = factory
    if any(row["status"] != "complete" for row in status):
        raise RuntimeError("one diagnostic failed; see finish_status.json (no substitutions made)")


if __name__ == "__main__":
    main()
