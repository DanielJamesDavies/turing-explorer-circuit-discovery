"""Cost of one attribution at the node budget that actually includes our
seed, as a function of batch size -- with a hard VRAM cap in force.

WHY THIS EXISTS. An earlier probe timed budget 16384 at 260s, but it ran
with BATCH=2048 and was spilling into Windows "shared GPU memory"
(system RAM over PCIe, ~5x slower). That number was therefore measuring
the spill, not the method, and the run-length estimate built on it was
wrong. theirs_llama now caps the allocator (MEM_FRAC) so an over-large
config raises OOM instead of quietly crawling -- which makes these
timings trustworthy.

Reports seconds and PEAK allocated/reserved VRAM per batch size, and
confirms the seed still has a row.

  TC_DIR=$HOME/tc_llama ../../dev-notes/data/venv-ct/bin/python probe_attr.py
"""
import os
import time
from pathlib import Path

import torch

import theirs_llama as T

HERE = Path(__file__).parent
BUDGET = int(os.environ.get("BUDGET", 16384))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "256,512").split(",")]
# circuit-tracer's own lever for exactly our problem: offload weights
# off-GPU during attribution. Cheaper to test than pre-folding weights
# so the encoders can be lazy (the RMS fold blocks in-memory laziness).
OFFLOADS = [None if x == "none" else x for x in
            os.environ.get("OFFLOADS", "cpu,none").split(",")]


def main():
    from circuit_tracer import attribute

    scan = torch.load(HERE / "scan_llama.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    S = seeds[sorted(seeds)[0]]
    L, sl = S["layer"], S["latent"]
    ids = toks[S["pos_windows"]][0].tolist()
    print("seed L%d feat %d | %d tokens | budget %d"
          % (L, sl, len(ids), BUDGET), flush=True)

    model = T.build_model()
    for off in OFFLOADS:
      for bs in BATCHES:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            g = attribute(ids, model, max_feature_nodes=BUDGET,
                          batch_size=bs, offload=off, verbose=False)
        except torch.cuda.OutOfMemoryError as e:
            print("offload %-4s batch %-5d | OOM (cap held): %s"
                  % (off, bs, str(e)[:52]), flush=True)
            torch.cuda.empty_cache()
            continue
        secs = time.time() - t0
        sel = g.selected_features.flatten()
        af_sel = g.active_features.to(sel.device)[sel]
        has = int(((af_sel[:, 0] == L) & (af_sel[:, 2] == sl)).sum())
        peak_a = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_r = torch.cuda.max_memory_reserved() / 1024 ** 3
        print("offload %-4s batch %-5d | %6.1fs | peak alloc %5.2f GB "
              "reserved %5.2f GB | seed rows %d"
              % (off, bs, secs, peak_a, peak_r, has), flush=True)
        del g
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
