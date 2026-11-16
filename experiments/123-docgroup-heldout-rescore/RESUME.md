# 123 RESUME (DAN-217): what is done, and the exact commands for the GPU part

## State (2026-11-16 ~02:15)

Done (CPU, local):
- Code, plus 16/16 tests passing.
- Full plan for all 7,786 TuringLLM targets (`out/turingllm/plan_{doc,topic}.jsonl`).
- Public document-disjoint records built for 509 Pythia and 512 GPT-2 targets (`out/<model>/ctx_doc/`,
  `build_doc.jsonl`), with check 0 FAIL.
- `rescore.py STAGE=prep` for public orig + doc and TuringLLM orig (hard links into `out/<model>/rs_*`).
- Dry-run report: `out/report.md`.

Not done: everything that needs a GPU. That is the TuringLLM build (firing + duplicate rule + fill from the T1 scan)
and every re-score. **No GPU job was launched.** The coordinator decides when.

Every step resumes. After a kill, rerun the same command: done targets are skipped (keyed by seed) and a torn last
jsonl line is repaired.

## Option A: pod (recommended; all three models; ~1.5–2 h wall on 5× H100, ~8–10 GPU-h)

The pod must be set up as for the 082 chain: `scripts/pod_quickstart.sh`, then `scripts/pod_restore.sh`.
- TuringLLM tree: `/root/turing-tl` (T1 scan, T2 records, T3 out_v2).
- Public tree: `/root/turing` (082 out_<model>, plus the S10-stores data pass in data_public/ and outputs_public/).

1. **Upload the files that are not in git** into `/root/turing-tl`, same relative paths:
   - `experiments/123-docgroup-heldout-rescore/`: code only, not `out/`. On the pod, plan rebuilds the public records
     in about 10 s.
   - `experiments/115-docdisjoint-rescore-pilot/build_heldout.py`, md5 `eb17be7aa26dc2dd361c8c9700794805`.
   - Check that these exist and match the local md5s; upload them if not:
     - `data/text_paths.txt`, md5 `de442344a910090344017e4a22689001`;
     - `experiments/085-heldout-audit/out/corpus_src.npy`, md5 `3f2531d882d6cb6bf89bad089a1010ba`;
     - `experiments/085-heldout-audit/out/corpus_hash.npy`, md5 `00d5a8e4a0ced9fc78abd942b282516a`.

2. **Dry run** (prints every command and runs nothing):
   ```
   DRY_RUN=1 TL_ROOT=/root/turing-tl PUB_ROOT=/root/turing GPUS=0,1,2,3,4 P=6 \
     bash /root/turing-tl/experiments/123-docgroup-heldout-rescore/run_gpu.sh
   ```

3. **Launch** (in its own tmux / ssh tab):
   ```
   TL_ROOT=/root/turing-tl PUB_ROOT=/root/turing GPUS=0,1,2,3,4 P=6 BUILD_P=2 MPS=1 STAGGER=5 \
     STEPS_ONLY=plan,build,check,prep,score,verify \
     bash /root/turing-tl/experiments/123-docgroup-heldout-rescore/run_gpu.sh 2>&1 | tee /root/run123.log
   ```
   - Order: plan (CPU, ~2 min) → TuringLLM build doc + topic (GPU, ~10 min) → check → prep → score TuringLLM orig /
     doc / topic (~20 min each) → Pythia orig / doc → GPT-2 orig / doc (~25 min in all) → verify.
   - Progress: `tail -f /root/turing-tl/experiments/123-docgroup-heldout-rescore/out/logs/<model>/rs_<variant>/main.shard*.log`.
   - `verify` must print `REFIT GUARD ... OK` for every model and variant.

4. **Bring back** `/root/turing-tl/experiments/123-docgroup-heldout-rescore/out/` without `ctx_*` and `rs_*/main/circuits`:
   ```
   tar czf /workspace/results/pub-123-docgroup.tgz -C /root/turing-tl/experiments/123-docgroup-heldout-rescore \
     --exclude='*/ctx_*' --exclude='*/circuits' out
   ```
   Unpack it locally into this folder, then run the report (CPU; it needs the 109 archives for the production rows):
   ```
   python -X utf8 experiments/123-docgroup-heldout-rescore/report.py
   ```

## Option B: local WSL GPU (TuringLLM only; ~30–42 h)

The public models cannot be scored locally: this repo's `data_public/<model>` is the 069 smoke corpus, and
`run_gpu.sh` skips them. Use this only once the GPU is free.
```
wsl: cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && \
     MODELS=turingllm GPUS=0 P=1 BUILD_P=1 bash experiments/123-docgroup-heldout-rescore/run_gpu.sh 2>&1 | tee experiments/123-docgroup-heldout-rescore/out/run_local.log
```
- Build: ~1.7 h. Score: ~13 h per variant, so ~40 h for orig + doc + topic.
- Add `VARIANTS=orig,doc` for ~28 h.
- `TARGETS_TURINGLLM=<list>` limits the build to a subset, for example a stratified sample. Then run
  `rescore.py STAGE=prep` again, and set `LIMIT` if needed.

## Single steps (any machine; prefix TL_ROOT / PUB_ROOT on a pod)

```
python -X utf8 experiments/123-docgroup-heldout-rescore/rebuild.py                                        # STAGE=plan, CPU
STAGE=build MODELS=turingllm SHARD=0/1 PYTHONPATH=src python -X utf8 experiments/123-docgroup-heldout-rescore/rebuild.py   # GPU
STAGE=check PYTHONPATH=src python -X utf8 experiments/123-docgroup-heldout-rescore/rebuild.py              # CPU
STAGE=prep python -X utf8 experiments/123-docgroup-heldout-rescore/rescore.py                             # CPU
STAGE=cmd  python -X utf8 experiments/123-docgroup-heldout-rescore/rescore.py                             # prints launch.sh lines
STAGE=verify python -X utf8 experiments/123-docgroup-heldout-rescore/rescore.py                           # CPU
python -X utf8 experiments/123-docgroup-heldout-rescore/report.py                                         # CPU
python -X utf8 -m pytest -q -p no:cacheprovider experiments/123-docgroup-heldout-rescore/test_123.py      # CPU tests
```

## Things to know

- The local prep made hard links from `out/*/rs_*/main/circuits/` to the archived production circuits. The driver
  never writes an existing circuit path, but do not delete or rewrite those files in place. On the pod, prep runs
  again with pod paths.
- If the TuringLLM build runs again after a change in the rule, delete the out files first. Delete them only via
  agent-gate: `out/turingllm/ctx_<group>/`, `build_<group>*.jsonl`, `targets_<group>.txt` and `rs_<group>/`.
- Unscorable targets (0 held rows after the rebuild) are not listed for the driver. The report counts them in the
  conservative comparison (as fails) and lists them in `prep_<variant>.json`.
- Open decisions are listed in README.md, last section.
