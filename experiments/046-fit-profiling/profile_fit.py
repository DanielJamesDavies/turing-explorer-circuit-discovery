"""KERNEL-LEVEL PROFILE of one tri-amp fit (2026-09-05, H100 launch prep).

The phase timers say the fit is 99% of per-seed time at ~1% of peak
FLOPs. This runs ONE seed's discovery through the production pipeline
path (same window, same method object, same config) under
torch.profiler and prints:
  * top CUDA ops by total time (what the GPU actually executes)
  * top CPU ops (launch/Python overhead)
  * kernel launches per step
  * a Chrome trace (outputs/profile_<comp>_<latent>.json; open in
    chrome://tracing or perfetto) for the full timeline
Steps are cut to PROF_STEPS (default 40) — per-step cost is what we
want, not a converged mask.

  SEED=<index into candidates.pt> PROF_STEPS=40 PYTHONPATH=src \
    python experiments/046-fit-profiling/profile_fit.py
  (SEED_LAYER=<L> picks the first candidate at that layer instead)
"""
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "src")
from circuit.discovery_window import DiscoveryWindow
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

PROF_STEPS = int(os.environ.get("PROF_STEPS", 40))
SEED = int(os.environ.get("SEED", 0))
SEED_LAYER = os.environ.get("SEED_LAYER")

torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices()
device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
cands = torch.load("outputs/candidates.pt", weights_only=False)
n_kinds = len(bank.kinds)
if SEED_LAYER is not None:
    L = int(SEED_LAYER)
    cand = next(c for c in cands if int(c["comp_idx"]) // n_kinds == L)
else:
    cand = cands[SEED]
comp_idx, latent = int(cand["comp_idx"]), int(cand["latent_idx"])
print("profiling seed comp %d (L%d %s) latent %d | steps %d"
      % (comp_idx, comp_idx // n_kinds, bank.kinds[comp_idx % n_kinds],
         latent, PROF_STEPS), flush=True)

config.discovery.learned_mask.steps = PROF_STEPS
window = DiscoveryWindow(inference, bank, loader)
method = window.methods[0]
assert type(method).__name__ == "AblationGradientDiscovery", type(method)

# warm-up seed (compile/caches), unprofiled
t0 = time.perf_counter()
method.discover(comp_idx, latent)
torch.cuda.synchronize()
print("warm-up discover: %.1f s" % (time.perf_counter() - t0), flush=True)

if os.environ.get("BENCH") == "1":
    # NOISE-CONTROLLED FIT BENCHMARK: the fit phases only (no evals, no
    # assembly), REPEATS times on the same seed. Compare trees on the
    # per-repeat seed.fit numbers; the spread across repeats is the noise
    # floor. ~30 s per repeat at PROF_STEPS=100.
    from observability.phases import reset_phases, snapshot_phases
    reps = int(os.environ.get("REPEATS", 2))
    for r in range(reps):
        reset_phases()
        method.discover(comp_idx, latent)
        torch.cuda.synchronize()
        ph = snapshot_phases()
        g = lambda k: ph.get(k, {}).get("s", 0.0)
        print("BENCH L%d comp %d latent %d rep %d | seed.fit %.2f s | "
              "fwd %.2f | bwd %.2f | pen_bwd %.2f | opt %.2f | per step %.1f ms"
              % (comp_idx // n_kinds, comp_idx, latent, r, g("seed.fit"),
                 g("fit.fwd"), g("fit.bwd"), g("fit.penalty_bwd"),
                 g("fit.opt"), 1000 * g("seed.fit") / PROF_STEPS), flush=True)
    sys.exit(0)

if os.environ.get("CPU_PROFILE") == "1":
    # ptrace-free Python-level CPU profile: which of OUR functions (and
    # which torch entry points) hold the CPU. cProfile overhead inflates
    # absolute times ~2x; the ranking and proportions are what matter.
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    method.discover(comp_idx, latent)
    torch.cuda.synchronize()
    pr.disable()
    st = pstats.Stats(pr)
    print()
    print("=== TOP FUNCTIONS BY OWN TIME (tottime) ===")
    st.sort_stats("tottime").print_stats(30)
    print()
    print("=== TOP PROJECT FUNCTIONS BY INCLUSIVE TIME (cumtime) ===")
    st.sort_stats("cumtime").print_stats("src", 25)
    sys.exit(0)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=False) as prof:
    t0 = time.perf_counter()
    method.discover(comp_idx, latent)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

print("\nprofiled discover: %.1f s (%.0f ms/step incl. evals)"
      % (wall, 1000 * wall / PROF_STEPS))
ka = prof.key_averages()
n_kernels = sum(e.count for e in ka if e.device_type.name == "CUDA")
print("CUDA kernel launches: %d total, ~%d per step\n"
      % (n_kernels, n_kernels / PROF_STEPS))
print("=== TOP CUDA OPS (by total device time) ===")
print(ka.table(sort_by="cuda_time_total", row_limit=30,
               max_name_column_width=60))
print("\n=== TOP CPU OPS (launch / Python overhead) ===")
print(ka.table(sort_by="cpu_time_total", row_limit=20,
               max_name_column_width=60))
out = "outputs/profile_%d_%d.json" % (comp_idx, latent)
prof.export_chrome_trace(out)
print("\ntrace ->", out)
