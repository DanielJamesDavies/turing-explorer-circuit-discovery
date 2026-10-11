"""Pass 1 / neg_ctx / pass 2 / candidates for a public model family (DAN-37).

The same functions src/pipeline/run_pipeline.py calls (no discovery), with the family's corpus and a per-family
output root, so each public model gets TuringLLM's artefact set:

  pass1      latent_stats.pt, top_ctx.pt, mid_ctx.pt, seq_repr.pt, logit_ctx.pt, seq_latent_index/   (first_pass.py)
  negctx     neg_ctx.pt, neg_ctx_stats.json, global_negctx_ids.pt                                   (neg_ctx stage)
  pass2      top_coactivation.pt                                                                    (second_pass.py)
  candidates candidates.pt                                                                          (candidate_selection.py)

Stages after pass1 load the earlier artefacts from --out when run on their own (--stages negctx,pass2,...).
A run manifest (data_pass_manifest.json: family, grid, corpus manifest, config overrides, stage timings, artefact
sizes, missing SAE sites) is written to --out.

  PYTHONPATH=src .venv/bin/python experiments/069-public-data-pass/run_data_pass.py \
      --family pythia-70m --data data_public/pythia-70m --out outputs_public/pythia-70m
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (sets sys.path for src/)

STAGES = ("pass1", "negctx", "pass2", "candidates")


def _artefacts(out_root: str) -> dict:
    sizes = {}
    for name in common.ARTEFACTS:
        p = os.path.join(out_root, name)
        sizes[name] = os.path.getsize(p) if os.path.exists(p) else None
    idx_dir = os.path.join(out_root, "seq_latent_index")
    if os.path.isdir(idx_dir):
        files = os.listdir(idx_dir)
        sizes["seq_latent_index/"] = dict(files=len(files),
                                          bytes=sum(os.path.getsize(os.path.join(idx_dir, f)) for f in files))
    return sizes


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="DAN-37 data passes for a public model family")
    ap.add_argument("--family", required=True, choices=["pythia-70m", "gpt2-small"])
    ap.add_argument("--data", required=True, help="shard directory written by prepare_corpus.py")
    ap.add_argument("--out", required=True, help="artefact root, e.g. outputs_public/<family>")
    ap.add_argument("--stages", default=",".join(STAGES), help="comma list out of %s" % (STAGES,))
    ap.add_argument("--batch-size", type=int, default=None, help="sequences per batch (config default 512)")
    ap.add_argument("--mid-warmup", type=int, default=None, help="mid_ctx warm-up batches (config default 100)")
    ap.add_argument("--skip-missing-saes", action="store_true",
                    help="treat dictionaries absent from the HF cache as missing sites (GPT-2 layer 10)")
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"],
                    help="model/SAE precision (default float32, as 033/053/070)")
    ap.add_argument("--n-seeds", type=int, default=None, help="candidate count (config default)")
    args = ap.parse_args(argv)
    stages = [s for s in args.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error("unknown stages %s" % bad)

    cfg = common.configure(args.family, args.data, args.out, batch_size=args.batch_size, mid_warmup=args.mid_warmup,
                           skip_missing_saes=args.skip_missing_saes, dtype=args.dtype, n_seeds=args.n_seeds)
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    from observability.timing import phase_timer
    import pipeline.distributed  # noqa: F401  (import order run_pipeline uses; avoids a runtime <-> distributed cycle)
    from pipeline.persist import offload_model_and_sae, offload_to_cpu, reload_model_and_sae, save_results
    from pipeline.runtime import get_runtime, initialize_resources, initialize_runtime

    timings: dict = {}

    def timed(name):
        class _T:
            def __enter__(self):
                self.t0 = time.perf_counter()
                self.cm = phase_timer(name)
                self.cm.__enter__()

            def __exit__(self, *exc):
                self.cm.__exit__(*exc)
                timings[name] = round(time.perf_counter() - self.t0, 1)
        return _T()

    with timed("initialize"):
        initialize_runtime()
        initialize_resources()
    runtime = get_runtime()
    grid = common.check_store_grid(runtime.bank)
    print("store grid:", grid)

    missing: set = set()
    if "pass1" in stages:
        from pipeline.first_pass import run_first_pass
        with timed("pass1"):
            run_first_pass(seq_latent_index_output_dir=os.path.join(out_root, "seq_latent_index"))
        missing |= set(getattr(runtime.bank, "missing_sites", set()))    # every site was touched by pass 1
        with timed("pass1_save"):
            save_results(out_root)
    else:
        # Later stages need pass-1 state: latent_stats (pass 2 frequency factors / PMI, candidates), the context
        # stores and seq_repr (neg_ctx).
        from store.context import mid_ctx, top_ctx
        from store.latent_stats import latent_stats
        from store.logit_context import logit_ctx
        latent_stats.load(os.path.join(out_root, "latent_stats.pt"))
        top_ctx.load(os.path.join(out_root, "top_ctx.pt"))
        mid_ctx.load(os.path.join(out_root, "mid_ctx.pt"))
        logit_ctx.load(os.path.join(out_root, "logit_ctx.pt"))
        runtime.seq_repr.load(os.path.join(out_root, "seq_repr.pt"))
        if getattr(runtime.bank, "skip_missing", False) is True:      # record which sites are missing
            bank = runtime.bank
            missing |= {(k, l) for k in bank.kinds for l in range(bank.n_layer) if not bank.has_site(k, l)}

    if "negctx" in stages:
        from pipeline.negative_context import build_negative_contexts
        with timed("negctx"):
            offload_to_cpu()
            offload_model_and_sae()
            build_negative_contexts(out_root)
            reload_model_and_sae()
    elif "pass2" in stages or "candidates" in stages:
        from store.context import neg_ctx
        neg_ctx.load(os.path.join(out_root, "neg_ctx.pt"))

    if "pass2" in stages:
        from pipeline.second_pass import run_second_pass
        with timed("pass2"):
            run_second_pass(out_root)
    elif "candidates" in stages:
        from store.top_coactivation import top_coactivation
        top_coactivation.load(os.path.join(out_root, "top_coactivation.pt"))

    if "candidates" in stages:
        from pipeline.candidate_selection import run_candidate_selection
        with timed("candidates"):
            run_candidate_selection(out_root)

    corpus_manifest_path = os.path.join(cfg.data.dataset_path, "corpus_manifest.json")
    manifest_path = os.path.join(out_root, "data_pass_manifest.json")
    previous = common.load_json(manifest_path) if os.path.exists(manifest_path) else {}
    manifest = dict(
        schema="public_data_pass_manifest_v1",
        updated=_dt.datetime.now().isoformat(timespec="seconds"),
        family=args.family,
        grid=grid,
        data_root=cfg.data.dataset_path,
        n_shards=cfg.data.n_shards,
        n_sequences=int(sum(runtime.loader._shard_sequence_counts)),
        corpus=common.load_json(corpus_manifest_path) if os.path.exists(corpus_manifest_path) else None,
        overrides=dict(batch_size=cfg.data.batch_size, mid_warmup_batches=cfg.latents.mid_ctx.warmup_batches,
                       skip_missing_saes=cfg.model.skip_missing_saes, dtype=cfg.model.dtype,
                       device=str(runtime.device)),
        settings=dict(top_ctx=cfg.latents.top_ctx.model_dump(), mid_ctx=cfg.latents.mid_ctx.model_dump(),
                      neg_ctx=cfg.latents.neg_ctx.model_dump(),
                      top_coactivation=cfg.latents.top_coactivation.model_dump(),
                      seed_criteria=list(cfg.discovery.seed_criteria), n_seeds=cfg.discovery.n_seeds),
        missing_sae_sites=[list(s) for s in sorted(missing)] or previous.get("missing_sae_sites", []),
        stage_timings_s={**previous.get("stage_timings_s", {}), **timings},
        artefacts=_artefacts(out_root),
    )
    common.write_json(manifest_path, manifest)
    print("run manifest:", manifest_path)
    for name, size in manifest["artefacts"].items():
        print("  %-22s %s" % (name, size))
    return manifest


if __name__ == "__main__":
    main()
