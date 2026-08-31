"""THE GO/NO-GO NUMBER for their side on Gemma: cost and coverage of one
attribution as a function of node budget, under the VRAM cap.

On Llama the seed needed a 16k-node budget to get an adjacency ROW at
all (circuit-tracer selects nodes by influence on the top LOGITS; an
internal feature has to make that cut). Gemma-2 is a harsher regime:
JumpReLU L0 ~50-100 x 26 layers x 64 positions is ~100k+ active
features per prompt, several times Llama's ~32k. If the seed needs the
same FRACTION of the budget, 16k will not reach it and the adjacency
needed would not fit in 16 GB.

Reports per budget: seconds, peak VRAM, n_active, n_selected, and
whether the first seed has a row. OOM under the cap is reported, never
silently spilled.

  TC_DIR=$HOME/gemma_tc ../../dev-notes/data/venv-ct/bin/python probe_attr_gtc.py
"""
import os
import time
from pathlib import Path

import torch

import theirs_gtc as T

HERE = Path(__file__).parent
BUDGETS = [int(x) for x in os.environ.get("BUDGETS", "4096,16384,32768").split(",")]
BATCH = int(os.environ.get("BATCH", 256))


def main():
    from circuit_tracer import attribute

    scan = torch.load(HERE / "scan_gtc.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    model = T.build_model()
    for key in sorted(seeds)[:2]:
        S = seeds[key]
        L, sl = S["layer"], S["latent"]
        ids = toks[S["pos_windows"]][0].tolist()
        print("\nseed L%d feat %d | %d tokens" % (L, sl, len(ids)), flush=True)
        for budget in BUDGETS:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                g = attribute(ids, model, max_feature_nodes=budget,
                              batch_size=BATCH, verbose=False)
            except torch.cuda.OutOfMemoryError as e:
                print("  budget %-6d | OOM under cap: %s" % (budget, str(e)[:60]),
                      flush=True)
                torch.cuda.empty_cache()
                continue
            secs = time.time() - t0
            af = g.active_features
            sel = g.selected_features.flatten()
            af_sel = af.to(sel.device)[sel]
            has = int(((af_sel[:, 0] == L) & (af_sel[:, 2] == sl)).sum())
            act = int(((af[:, 0] == L) & (af[:, 2] == sl)).sum())
            print("  budget %-6d | %6.1fs | peak %5.2f GB | adj %-16s active %-7d "
                  "selected %-6d | seed rows %d (active occ %d)"
                  % (budget, secs, torch.cuda.max_memory_allocated() / 1024 ** 3,
                     tuple(g.adjacency_matrix.shape), af.shape[0], len(sel),
                     has, act), flush=True)
            del g
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
