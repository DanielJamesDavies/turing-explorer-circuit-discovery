# Plan: God-File Refactoring Manager

> **Goal:** Coordinate the refactor of the largest high-risk files into focused modules while preserving public imports, CLI behavior, artifact contracts, and test coverage.
>
> **Created:** 2026-05-23

---

## Central Context

This is the manager plan for the refactoring work in `agent-planning/refactoring/`. Each row below links to a detailed implementation plan for one file. The goal is not to redesign algorithms during the first pass; it is to make ownership boundaries clearer while keeping behavior stable.

Shared rules for every refactor:

- Keep the original large file as a compatibility facade until all callers are migrated or explicitly updated.
- Preserve existing public import paths, CLI commands, artifact schemas, report shapes, marker semantics, and tensor layouts.
- Prefer mechanical extraction first; defer algorithm cleanup, lifecycle abstractions, and performance changes until after tests pass.
- Extract low-risk contracts/helpers before high-coupling orchestration functions.
- Run the focused tests listed in each plan before and after each phase.
- After each detailed refactor plan is complete, run whole-system validation: the relevant full focused suite from that plan plus a local pipeline smoke (`bash scripts/run.sh`) and a local distributed controller dry-run.
- Do not touch generated artifacts, native binaries, output runs, or unrelated files as part of these refactors.

## Candidate Plans

Approximate LOC from the repository scan on 2026-05-23.

| Priority | File | LOC | Refactor plan file | Target shape |
| ---: | --- | ---: | --- | --- |
| 1 | `src/pipeline/distributed/pass2_reduce.py` | 1,874 | [`pass2-reduce-refactor.md`](pass2-reduce-refactor.md) | `src/pipeline/distributed/pass2/` package plus facade |
| 2 | `src/pipeline/distributed/pass1_merge.py` | 1,516 | [`pass1-merge-refactor.md`](pass1-merge-refactor.md) | `src/pipeline/distributed/pass1/` package plus facade |
| 3 | `src/circuit/instrument/attribution.py` | 1,493 | [`attribution-refactor.md`](attribution-refactor.md) | sibling attribution modules plus facade |
| 4 | `src/store/neg_context.py` | 1,291 | [`neg-context-store-refactor.md`](neg-context-store-refactor.md) | `src/store/neg_ctx/` package plus facade |
| 5 | `src/pipeline/distributed/worker.py` | 1,022 | [`distributed-worker-refactor.md`](distributed-worker-refactor.md) | phase-specific worker modules plus facade |
| 6 | `src/pipeline/negative_context.py` | 1,017 | [`negative-context-pipeline-refactor.md`](negative-context-pipeline-refactor.md) | `src/pipeline/negative_context_stage/` package plus facade |
| 7 | `src/pipeline/distributed/controller.py` | 973 | [`distributed-controller-refactor.md`](distributed-controller-refactor.md) | sibling controller modules plus facade |

## Suggested Execution Order

- [x] Start with [`pass2-reduce-refactor.md`](pass2-reduce-refactor.md), because it is the largest file and already has clear boundaries between contracts, simple exact reduce, MapReduce, reports, and CLI.
- [x] Then do [`pass1-merge-refactor.md`](pass1-merge-refactor.md), aligning it with the new `distributed/pass1/` package that later worker extraction can use.
- [x] Do [`neg-context-store-refactor.md`](neg-context-store-refactor.md) before [`negative-context-pipeline-refactor.md`](negative-context-pipeline-refactor.md), so the pipeline stage can depend on the cleaner store backend boundary.
- [x] Do [`distributed-worker-refactor.md`](distributed-worker-refactor.md) after the `pass1/` and `pass2/` packages exist, so phase-specific worker code has natural homes.
- [x] Do [`distributed-controller-refactor.md`](distributed-controller-refactor.md) after worker extraction, because controller imports and command generation depend on the worker facade remaining stable.
- [x] Do [`attribution-refactor.md`](attribution-refactor.md) independently when circuit/instrumentation tests can be run; it does not need to block distributed pipeline refactors.

## Dependency Notes

- `src/pipeline/distributed/pass2_reduce.py` and `src/pipeline/distributed/pass1_merge.py` should become package facades only after their extracted modules are stable.
- `src/pipeline/distributed/worker.py` should keep `python -m pipeline.distributed.worker` working throughout; controller dry-runs and external scheduler workflows rely on this command.
- `src/pipeline/distributed/controller.py` should use sibling modules first, not a `controller/` package, to avoid a Python import conflict with the existing `controller.py` module.
- `src/pipeline/negative_context.py` should use `negative_context_stage/` first, not `negative_context/`, to avoid a Python import conflict with the existing file.
- `src/store/neg_context.py` can safely target a true `src/store/neg_ctx/` package because the package name differs from the existing facade file.
- `src/circuit/instrument/attribution.py` should use sibling modules first, not an `attribution/` package, to avoid a Python import conflict.

## Progress Tracker

- [x] `pass2_reduce.py` refactor planned.
- [x] `pass2_reduce.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `pass2_reduce.py` refactor.
- [x] `pass1_merge.py` refactor planned.
- [x] `pass1_merge.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `pass1_merge.py` refactor.
- [x] `attribution.py` refactor planned.
- [x] `attribution.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `attribution.py` refactor.
- [x] `store/neg_context.py` refactor planned.
- [x] `store/neg_context.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `store/neg_context.py` refactor.
- [x] `distributed/worker.py` refactor planned.
- [x] `distributed/worker.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `distributed/worker.py` refactor.
- [x] `pipeline/negative_context.py` refactor planned.
- [x] `pipeline/negative_context.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `pipeline/negative_context.py` refactor.
- [x] `distributed/controller.py` refactor planned.
- [x] `distributed/controller.py` refactor implemented and tests passing.
- [x] Whole-system validation completed after `distributed/controller.py` refactor.

### Latest Validation Notes

- `distributed/controller.py` refactor completed on 2026-05-24: implementation moved into sibling `src/pipeline/distributed/controller_*.py` modules, with `src/pipeline/distributed/controller.py` reduced to a 147-line compatibility facade.
- Focused and integration verification passed: `tests/pipeline/test_distributed_controller.py`, `tests/pipeline/test_distributed_config.py tests/pipeline/test_distributed_devices.py tests/pipeline/test_distributed_layout.py tests/pipeline/test_distributed_manifest.py tests/pipeline/test_distributed_assignments.py`, and `tests/pipeline/test_distributed_worker.py tests/pipeline/test_negative_context_stage.py tests/pipeline/test_distributed_operating_modes.py`.
- Whole-system WSL validation passed with `.venv`: `bash scripts/run.sh` completed successfully, and the distributed controller CPU dry-run completed successfully with `PYTHONPATH=src python -m pipeline.distributed.controller --project-root . --config config.yaml --output-base outputs --use-cpu --worker-count 1 --dry-run`.
- All listed god-file refactors in this manager plan are now implemented and validated.

---

## Open Questions

- Should the manager list later include borderline files such as `src/config.py`, `src/circuit/discovery/layerwise_gradient_upstream.py`, and `src/circuit/discovery/cluster_contrast.py`?
- Should completed refactor plans remain in this folder permanently, or move into an archive once implemented and verified?
- Should facade modules define explicit `__all__` everywhere as part of the refactor standard?

## Risks / Assumptions

- The biggest risk is behavior drift hidden by mechanical movement of tensor-heavy code. Focused tests must run after every phase.
- Import cycles are likely when extracting facade modules; extract contracts and pure helpers first.
- CLI and public import compatibility matter as much as internal cleanliness because controller/worker commands are part of the distributed runtime contract.
- Refactors should remain independent unless a plan explicitly notes a dependency.
