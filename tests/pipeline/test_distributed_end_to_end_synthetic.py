import json
from pathlib import Path

import torch

from pipeline.distributed.equivalence import (
    compare_run_roots,
    save_equivalence_report,
)


RUN_ID = "20260523-141500-abcdef12"


def _write_synthetic_canonical_artifacts(
    run_root: Path,
    *,
    floating_offset: float = 0.0,
    candidate_offset: int = 0,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "circuits").mkdir(parents=True, exist_ok=True)
    shape = (2, 3)
    ctx_shape = (2, 3, 2)
    topk_shape = (2, 3, 4)

    torch.save(
        {
            "active_count": torch.arange(6, dtype=torch.int64).reshape(shape),
            "mean": torch.arange(6, dtype=torch.float32).reshape(shape) / 10.0
            + floating_offset,
            "seq_count": torch.ones(shape, dtype=torch.int64),
            "mean_seq": torch.ones(shape, dtype=torch.float32) + floating_offset,
        },
        run_root / "latent_stats.pt",
    )
    for name, ctx_type in (
        ("top_ctx.pt", "top"),
        ("mid_ctx.pt", "mid"),
        ("neg_ctx.pt", "neg"),
    ):
        torch.save(
            {
                "ctx_type": ctx_type,
                "ctx_seq_idx": torch.arange(12, dtype=torch.int32).reshape(ctx_shape),
                "ctx_seq_val": torch.arange(12, dtype=torch.float32).reshape(ctx_shape) / 100.0
                + floating_offset,
            },
            run_root / name,
        )
    torch.save(
        {
            "latent_counts": torch.ones(shape, dtype=torch.int64),
            "top_tokens": torch.arange(12, dtype=torch.int32).reshape(ctx_shape),
            "top_probs": torch.arange(12, dtype=torch.float32).reshape(ctx_shape) / 100.0
            + floating_offset,
        },
        run_root / "logit_ctx.pt",
    )
    torch.save(
        {
            "top_indices": torch.arange(24, dtype=torch.int32).reshape(topk_shape),
            "top_values": torch.arange(24, dtype=torch.float32).reshape(topk_shape) / 10.0
            + floating_offset,
            "mode": "pmi",
        },
        run_root / "top_coactivation.pt",
    )
    torch.save(
        [
            {"comp_idx": 0, "latent_idx": 1 + candidate_offset},
            {"comp_idx": 1, "latent_idx": 2 + candidate_offset},
        ],
        run_root / "candidates.pt",
    )
    (run_root / "circuits" / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "circuit_count": 2,
                "methods": {"counterfactual_gradient": 2},
                "workers": [0, 1],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_synthetic_one_worker_two_worker_and_mapreduce_roots_match_single_process(tmp_path):
    single_root = tmp_path / "outputs" / RUN_ID
    one_worker_root = tmp_path / "outputs" / f"{RUN_ID}-one-worker"
    two_worker_root = tmp_path / "outputs" / f"{RUN_ID}-two-worker"
    mapreduce_root = tmp_path / "outputs" / f"{RUN_ID}-mapreduce"
    for run_root in (single_root, one_worker_root, two_worker_root):
        _write_synthetic_canonical_artifacts(run_root)
    # Floating-point reduction order can differ between reducer implementations.
    _write_synthetic_canonical_artifacts(mapreduce_root, floating_offset=5e-7)

    one_worker_report = compare_run_roots(single_root, one_worker_root)
    two_worker_report = compare_run_roots(single_root, two_worker_root)
    mapreduce_report = compare_run_roots(single_root, mapreduce_root, atol=1e-6)

    assert one_worker_report["status"] == "equivalent"
    assert two_worker_report["equivalence"]["passed"] is True
    assert mapreduce_report["status"] == "equivalent"
    assert single_root.parent.name == "outputs"
    assert single_root.name == RUN_ID

    reports_root = two_worker_root / "distributed" / "reports"
    saved = save_equivalence_report(
        two_worker_report,
        reports_root / "equivalence_tiny_synthetic.json",
    )
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["ok"] is True


def test_synthetic_equivalence_report_identifies_canonical_artifact_drift(tmp_path):
    single_root = tmp_path / "outputs" / RUN_ID
    candidate_root = tmp_path / "outputs" / f"{RUN_ID}-candidate"
    _write_synthetic_canonical_artifacts(single_root)
    _write_synthetic_canonical_artifacts(candidate_root, candidate_offset=10)

    report = compare_run_roots(single_root, candidate_root)

    assert report["status"] == "different"
    assert report["ok"] is False
    differing = [
        artifact["artifact"]
        for artifact in report["artifacts"]
        if not artifact["equivalent"]
    ]
    assert differing == ["candidates.pt"]
