"""CLI: python -m export.explorer <stage> [options]  (run from src/)."""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN_062 = os.path.join(REPO_ROOT, "experiments", "062-h100-protocol-v1", "out_stage1", "out")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m export.explorer")
    sub = parser.add_subparsers(dest="stage", required=True)

    pf = sub.add_parser("preflight", help="stage 0: read-only checks of every input")
    pf.add_argument("--outputs", default=os.path.join(REPO_ROOT, "outputs"), help="store directory")
    pf.add_argument("--data", default=os.path.join(REPO_ROOT, "data"), help="token shard directory")
    pf.add_argument("--run", default=RUN_062, help="062 run output dir (holding ctx/ and main/); '' to skip")
    pf.add_argument("--sample-shards", type=int, default=0, help="scan N random shards (0 = all)")
    pf.add_argument("--token-targets", type=int, default=100, help="targets whose tokens are round-tripped")
    pf.add_argument("--full-targets", type=int, default=15046, help="target count for the size estimate")
    pf.add_argument("--skip-stores", action="store_true", help="skip the outputs/ store checks")
    pf.add_argument("--report", default="", help="write the report as JSON here")

    b = sub.add_parser("build", help="build (or update) a bundle: stages 1-4, 6 and 7 (search index)")
    b.add_argument("--outputs", default=os.path.join(REPO_ROOT, "outputs"), help="store directory")
    b.add_argument("--data", default=os.path.join(REPO_ROOT, "data"), help="token shard directory")
    b.add_argument("--run", default=RUN_062, help="062 run output dir (holding ctx/ and main/)")
    b.add_argument("--run-name", default="062-main", help="run label used in circuit keys")
    b.add_argument("--bundle-id", default="", help="default: <date>_<run-name>")
    b.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "explorer_bundle"), help="parent directory")
    b.add_argument("--stages", default="1,2,3,4,6", help="comma list of stages to run")
    b.add_argument("--force", action="store_true", help="re-run stages already marked done")
    b.add_argument("--wiring", nargs="*", default=[], help="wired-edge jsonl files to overlay (stage 4)")
    b.add_argument("--limit-targets", type=int, default=0, help="only the first N targets (quick test builds)")
    b.add_argument("--no-checksums", action="store_true", help="skip sha256 of bundle files in the manifest")
    b.add_argument("--search-k", type=int, default=16, help="top contexts per latent in the search index (stage 7)")

    args = parser.parse_args()
    if args.stage == "preflight":
        from export.explorer import preflight
        return preflight.run(args)
    if args.stage == "build":
        return build(args)
    return 2


def build(args) -> int:
    import time

    from export.explorer.bundle import Bundle
    from export.explorer.run_source import RunSource

    bundle_id = args.bundle_id or f"{time.strftime('%Y-%m-%d')}_{args.run_name}"
    bundle = Bundle(os.path.join(args.out, bundle_id))
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    print(f"bundle {bundle.root}\nstages {stages}")
    run = None

    def get_run() -> RunSource:
        nonlocal run
        if run is None:
            run = RunSource(args.run, args.limit_targets)
        return run

    def stage(name: str, fn) -> None:
        if not args.force and bundle.stage_info(name):
            print(f"\n== stage {name}: already done, skipping (--force to redo)")
            return
        print(f"\n== stage {name}")
        bundle.clear(name)
        info = fn()
        bundle.mark(name, info)
        print(f"   done: {info}")

    if "1" in stages:
        from export.explorer import stage_latents
        stage("1", lambda: stage_latents.build(bundle, args.outputs))
    if "2" in stages:
        from export.explorer import stage_tokens
        stage("2", lambda: stage_tokens.build(bundle, args.data))
    if "3" in stages:
        from export.explorer import stage_targets
        stage("3", lambda: stage_targets.build(bundle, get_run(), args.run_name))
    if "4" in stages:
        from export.explorer import stage_circuits
        stage("4", lambda: stage_circuits.build(bundle, get_run(), args.run_name))
        if args.wiring:
            print("\n== stage 4 wiring overlay")
            info = stage_circuits.apply_wiring(bundle, args.wiring, args.run_name)
            bundle.mark("4-wiring", info)
            print(f"   done: {info}")
    if "6" in stages:
        from export.explorer import stage_finalize
        infos = {s: bundle.stage_info(s) for s in ("1", "2", "3", "4", "4-wiring", "5", "7")}
        missing = [s for s in ("1", "2", "3", "4") if not infos[s]]
        if missing:
            print(f"stage 6 needs stages {missing} first")
            return 1
        bundle.clear("6")
        print("\n== stage 6")
        info = stage_finalize.build(bundle, bundle_id, args.outputs, get_run(), args.run_name, args.data,
                                    REPO_ROOT, infos, checksums=not args.no_checksums)
        bundle.mark("6", info)
        print(f"   done: {info}")
    if "7" in stages:
        from export.explorer import stage_search
        missing = [s for s in ("1", "2") if not bundle.stage_info(s)]
        if missing:
            print(f"stage 7 needs stages {missing} first")
            return 1
        stage("7", lambda: stage_search.build(bundle, args.search_k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
