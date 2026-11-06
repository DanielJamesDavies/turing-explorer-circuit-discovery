"""Build missing frozen-panel records and evaluate, sharing one model load."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

from run_baseline import HERE, ROOT, load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out/panel18")
    parser.add_argument("--records", type=Path, default=HERE / "out/records/ctx_v2")
    args = parser.parse_args()
    out, records = args.out, args.records
    records.mkdir(parents=True, exist_ok=True)
    os.environ.update(OUT=str(out / "driver"), CTX_DIR=str(records), WCM_VERSION="v2", EVAL_SPLIT="explicit",
                      SCAN=str(ROOT / "experiments/086-heldout-rebuild/out/scan_smoke40/scan_top512.pt"))
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/073-strict-eval")]
    import torch
    import strict_eval as SE
    from config import config
    R = SE.driver.Runner()
    builder = load_module("fresh099panel", ROOT / "experiments/094-tl-v2-records/build_records.py")
    import rebuild as RB
    # Same Engine methods, attached to the already loaded model and bank.
    engine = object.__new__(RB.Engine)
    engine.dev, engine.loader = R.dev, R.G["M0"].probe_builder.loader
    engine.inference, engine.bank = R.G["inference"], R.G["bank"]
    engine.kinds, engine._sel = list(engine.bank.kinds), None
    source = None
    keys = (HERE / "targets_panel18.txt").read_text().split()
    build_dir = HERE / "out/records/build"
    build_dir.mkdir(parents=True, exist_ok=True)
    with open(build_dir / "build.jsonl", "a") as fh:
        for key in keys:
            path = records / (key + ".pt")
            if path.exists():
                continue
            if source is None:
                source = builder.Sources(engine)
            start = time.time()
            rec, row = builder.build_one(key, engine, source, config.discovery.context_v1)
            builder.save_atomic({key: rec}, path)
            row["seconds"] = time.time() - start
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print("BUILD", key, round(row["seconds"], 1), "seconds", flush=True)
    del engine, source
    gc.collect()
    torch.cuda.empty_cache()
    import run_baseline
    sys.argv = ["run_baseline.py", "--targets", str(HERE / "targets_panel18.txt"),
                "--contexts", str(records), "--out", str(out), "--draws", "8", "--ig"]
    run_baseline.main(runner=R)


if __name__ == "__main__":
    main()
