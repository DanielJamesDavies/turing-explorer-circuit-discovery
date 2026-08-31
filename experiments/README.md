# experiments/

One directory per campaign (`YYYY-MM-DD-<name>`, so listings sort chronologically), each holding the
scripts that ran it, a README with the result and its caveats, and the
small result files (`*.jsonl` rows/members, convention `*.json`) that
the paper cites. This directory is the evidence behind every number in
`paper/main.tex`; the `% data:` comments there point here.

## What is tracked and what is not
Tracked: code, READMEs, `*.json`, `*.jsonl`, small figures.
Ignored (see `.gitignore`): `*.pt` (seed scans, regenerable from the
fixed RNG), `*.safetensors` / `dictionaries_*` / `transcoders_*`
(weights, tens of GB), `*.zip`, `*.pickle`, `*.log`, `__pycache__`.

Heavy artifacts therefore live only on the working machine. Weight
copies used by the cross-architecture work sit on WSL-native disk:
`$HOME/tc_llama`, `$HOME/tc_llama_folded`, `$HOME/gemma_tc`,
`$HOME/gemmascope` (see the campaign READMEs for why: `/mnt/x` reads at
~216 MB/s and barely caches).

## External code
`../external/circuit-tracer/` is a git submodule pinned to the commit every
circuit-tracer number was produced with (`8f1e2438`, 2026-07-17). The
faithfulness harness (`038-transcoder-compare-gemma/
ct_faithfulness.py`) verifies the imported clone matches that pin and
reproduces a graph the library's authors published.

The isolated environment for circuit-tracer (it pins an older
`transformers`) is `dev-notes/data/venv-ct` (untracked; absolute paths
baked in, so it stays where it was created). The circuit-tracer scripts
are run as `../../dev-notes/data/venv-ct/bin/python <script>` from
inside their campaign directory. Everything else runs from the repo
root with the main `.venv` and `PYTHONPATH=src`.

## Conventions every campaign follows
* Held-out evaluation (48 train / 16 held-out windows) unless the README
  says otherwise; seeds fixed before any scoring.
* Conventions of external dictionaries are PROBED, never assumed
  (`probe_*.py` / `*_loader.py probe`), and an identity gate certifies
  that an unmodified circuit is a no-op before any metric is quoted.
* Anything scored against another method goes through the same scoring
  closure as our own arms, re-certified by a `selfcheck` arm.
* Comparisons are made only INSIDE one arena (model + dictionary + seeds
  + windows + scorer); never across arenas.
