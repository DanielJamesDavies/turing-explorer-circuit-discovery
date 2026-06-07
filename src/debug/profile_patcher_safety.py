"""Profile safe SAE dense expansion and optional patcher forwards.

Examples:
    python -m debug.profile_patcher_safety --device cuda
    python -m debug.profile_patcher_safety --device cuda --model-forward
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from typing import Any

import torch

from circuit.types.feature_id import FeatureID
from hardware import detect_devices
from model.hooks import multi_patch
from model.inference import Inference
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--d-sae", type=int, default=40960)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--model-forward", action="store_true", help="Also benchmark a real model patcher forward.")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kind", choices=("attn", "mlp", "resid"), default="attn")
    parser.add_argument("--latent-idx", type=int, default=0)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return detect_devices()[0]
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**3)


@contextmanager
def profile_section(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    yield lambda: None
    _sync(device)
    elapsed = time.perf_counter() - start
    profile_section.elapsed = elapsed  # type: ignore[attr-defined]
    profile_section.peak_gb = _peak_gb(device)  # type: ignore[attr-defined]


def _old_scatter(top_acts: torch.Tensor, top_indices: torch.Tensor, d_sae: int) -> torch.Tensor:
    dense = torch.zeros(*top_acts.shape[:-1], int(d_sae), device=top_acts.device, dtype=top_acts.dtype)
    dense.scatter_(dim=-1, index=top_indices.long(), src=top_acts)
    return dense


def benchmark_dense(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    shape = (args.batch_size, args.seq_len, args.top_k)
    top_acts = torch.rand(shape, device=device, dtype=torch.float32)
    top_indices = torch.randint(0, args.d_sae, shape, device=device, dtype=torch.int64)
    top_indices[..., -4:] = 0
    top_acts[..., -4:] = 0.0
    top_indices[..., 0] = 0
    top_acts[..., 0] = torch.maximum(top_acts[..., 0], torch.tensor(0.5, device=device))

    rows: list[dict[str, object]] = []
    for label, fn in (("scatter", _old_scatter), ("scatter_reduce_amax", sparse_topk_to_dense)):
        for _ in range(args.warmup):
            fn(top_acts, top_indices, args.d_sae)
        with profile_section(device):
            for _ in range(args.iters):
                dense = fn(top_acts, top_indices, args.d_sae)
        rows.append(
            {
                "section": f"dense_{label}",
                "iters": args.iters,
                "seconds": profile_section.elapsed,  # type: ignore[attr-defined]
                "seconds_per_iter": profile_section.elapsed / args.iters,  # type: ignore[attr-defined]
                "peak_gb": profile_section.peak_gb,  # type: ignore[attr-defined]
                "latent_zero_mean": float(dense[..., 0].mean().item()),
            }
        )
    return rows


class SingleLatentZeroPatcher:
    def __init__(self, bank: SAEBank, fid: FeatureID):
        self.bank = bank
        self.fid = fid
        self.delta_norm = 0.0

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.fid.layer or kind != self.fid.kind:
            return x
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=x.dtype)
        recon = self.bank.decode(dense, kind, layer_idx)
        error = x - recon
        patched_dense = dense.clone()
        patched_dense[..., self.fid.index] = 0.0
        out = self.bank.decode(patched_dense, kind, layer_idx) + error
        self.delta_norm = float(torch.linalg.vector_norm((out - x).detach().float()).item())
        return out


def benchmark_model_forward(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    inference = Inference(device=device, compile=True)
    bank = SAEBank(devices=[device], load_decoders=True, compile=True)
    tokens = torch.randint(0, inference.model.config.vocab_size, (args.batch_size, args.seq_len), device=device)
    rows: list[dict[str, object]] = []

    for label, patcher in (
        ("model_no_patcher", None),
        ("model_single_latent_zero_patcher", SingleLatentZeroPatcher(bank, FeatureID(args.layer, args.kind, args.latent_idx))),
    ):
        for _ in range(args.warmup):
            inference.forward(tokens, patcher=patcher, return_activations=False, tokenize_final=False, all_logits=True)
        with profile_section(device):
            for _ in range(args.iters):
                inference.forward(tokens, patcher=patcher, return_activations=False, tokenize_final=False, all_logits=True)
        rows.append(
            {
                "section": label,
                "iters": args.iters,
                "seconds": profile_section.elapsed,  # type: ignore[attr-defined]
                "seconds_per_iter": profile_section.elapsed / args.iters,  # type: ignore[attr-defined]
                "peak_gb": profile_section.peak_gb,  # type: ignore[attr-defined]
                "patch_delta_norm": getattr(patcher, "delta_norm", None),
                "compiled_after": bool(inference._compiled),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    device = _device(args.device)
    rows = benchmark_dense(args, device)
    if args.model_forward:
        rows.extend(benchmark_model_forward(args, device))
    print(json.dumps({"device": str(device), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
