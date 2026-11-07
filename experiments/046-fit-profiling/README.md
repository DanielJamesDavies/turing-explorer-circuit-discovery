# 046: profiling one learned-mask fit, and an equivalence check between circuit stores

Engineering tools from the preparation of the tri-amp H100 run (`045-h100-production`), before WCM. The phase
timers showed the fit taking almost all of each seed's time at a small fraction of the GPU's peak throughput. This
folder holds a profiler for one fit and a check that a performance change leaves the circuits unchanged.

## Question

- Where does one fit spend its time (CUDA kernels, CPU launch overhead, per-step phases)?
- After a change to the engine code, are the circuits the same?

## What the code does

**`profile_fit.py`** runs one seed's discovery through the production pipeline path: the data-pass artifacts in
`outputs/` (`outputs/candidates.pt`), the `config.yaml` settings and the first discovery method, which it asserts is
`AblationGradientDiscovery`. Steps are cut to `PROF_STEPS` (default 40). After an unprofiled warm-up it does one of:
- default: a `torch.profiler` run that prints the top CUDA ops, the top CPU ops and kernel launches per step, and
  writes a Chrome trace to `outputs/profile_<comp>_<latent>.json`;
- `BENCH=1`: the fit only, `REPEATS` times (default 2), printing the per-phase times (forward, backward, penalty
  backward, optimiser) and ms per step; the spread across repeats is the noise floor;
- `CPU_PROFILE=1`: a cProfile of the Python functions.

```
SEED=<index into candidates.pt> PROF_STEPS=40 PYTHONPATH=src python experiments/046-fit-profiling/profile_fit.py
# SEED_LAYER=<L> picks the first candidate at that layer instead of SEED
```

**`compare_stores.py A.pt B.pt`** loads two circuit stores and matches circuits by (seed comp, seed latent,
discovery method). Per seed it prints both sizes, the Jaccard of the member sets and the largest |Δα| over the
shared members. It prints EQUIVALENT and exits 0 if the worst Jaccard is ≥ 0.98 and the worst |Δα| ≤ 2e-2;
otherwise DIFFERENT, exit 1. A seed present in only one store counts as Jaccard 0. (The docstring says alphas within
~1e-2; the code's threshold is 2e-2.)

## Files

| File | What |
|---|---|
| `profile_fit.py` | Kernel-level, benchmark and CPU profiles of one fit |
| `compare_stores.py` | Membership and amplitude equivalence check between two circuit stores |

## Results

None in the repo. The profiler writes to `outputs/` (gitignored) and prints to stdout, and no log of a run is kept
here.

## Status

The profiler is historical: it profiles the pipeline path of the tri-amp era, not the WCM v4 fit run by
`062-h100-protocol-v1/driver.py`. The later profile of the WCM fit is `106-speed-profile`. `compare_stores.py` is still
in use: `068-validation-gates/golden_turingllm.py` writes dict stores for it (its README gives the optional
`compare_stores.py` commands), and the golden check itself uses the same Jaccard-and-|Δα| comparison, with
|Δα| ≤ 1e-2.
