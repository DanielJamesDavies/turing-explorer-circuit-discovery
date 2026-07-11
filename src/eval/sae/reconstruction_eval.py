"""Fresh reconstruction quality (and optional CE recovered) for all 36 SAEs.

Runs TuringLLM over interpretability-corpus batches, captures the pre-SAE dense
activation at every (layer, kind) site, and measures each SAE's reconstruction:

- explained variance: 1 - Var(x - x_hat) / Var(x)
- NRMSE: ||x - x_hat|| / ||x||

With ``--ce`` it additionally splices each SAE's reconstruction into the model
one site at a time and reports the cross-entropy difference and the percentage
of CE recovered relative to zero-ablating that site (SFC-style; 2 extra forward
passes per site per batch, so 72 per batch).

Run from the repo root (GPU strongly recommended):
    PYTHONPATH=src python -m eval.sae.reconstruction_eval [--batches 4] [--seqs 128] [--ce]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from analysis.io import write_csv, write_json
from data.loader import DataLoader
from model.hooks import multi_patch
from model.inference import Inference
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

KINDS = ("attn", "mlp", "resid")
TABLE_FIELDS = [
    "kind",
    "layer",
    "tokens",
    "explained_variance",
    "nrmse",
    "ce_original",
    "ce_reconstructed",
    "ce_zero_ablated",
    "ce_difference",
    "ce_recovered_pct",
]


class _SiteSplice:
    """Patcher that replaces one site's activation with its SAE reconstruction
    (mode="recon") or zeros (mode="zero"), leaving every other site untouched."""

    def __init__(self, bank: SAEBank, layer: int, kind: str, mode: str):
        self.bank = bank
        self.layer = layer
        self.kind = kind
        self.mode = mode

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.layer or kind != self.kind:
            return x
        if self.mode == "zero":
            return torch.zeros_like(x)
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        f = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=x.dtype)
        return self.bank.decode(f, kind, layer_idx).to(x.dtype)


def _next_token_ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    """Mean next-token cross-entropy over the batch."""

    targets = tokens[:, 1:].reshape(-1).to(logits.device)
    predictions = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
    return float(F.cross_entropy(predictions, targets.long()))


def run_reconstruction_eval(
    *,
    batches: int,
    seqs_per_batch: int,
    with_ce: bool,
) -> list[dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(device, compile=False)
    bank = SAEBank(device=device)
    loader = DataLoader(device)

    n_layers = bank.n_layer
    residual_sq = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)
    input_sq = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)
    input_sum = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)
    element_count = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)
    token_count = 0
    ce_original: list[float] = []
    ce_recon = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)
    ce_zero = torch.zeros(n_layers, len(KINDS), dtype=torch.float64)

    batch_iter = loader.get_batches(device=device)
    for batch_index in range(batches):
        try:
            _, tokens = next(batch_iter)
        except StopIteration:
            break
        tokens = tokens[:seqs_per_batch]
        token_count += int(tokens.numel())

        _, logits, activations = inference.forward(
            tokens, num_gen=1, tokenize_final=False, return_activations=True, all_logits=with_ce
        )
        # activations: [B, L, K, G=1, T, d_model]
        activations = activations.detach()
        for layer in range(n_layers):
            for kind_idx, kind in enumerate(KINDS):
                x = activations[:, layer, kind_idx, 0].to(device)
                top_acts, top_indices = bank.encode(x, kind, layer)
                f = sparse_topk_to_dense(top_acts, top_indices, bank.d_sae, dtype=x.dtype)
                x_hat = bank.decode(f, kind, layer)
                residual = (x - x_hat).double()
                residual_sq[layer, kind_idx] += float(residual.pow(2).sum())
                input_sq[layer, kind_idx] += float(x.double().pow(2).sum())
                input_sum[layer, kind_idx] += float(x.double().sum())
                element_count[layer, kind_idx] += x.numel()
        del activations

        if with_ce:
            ce_original.append(_next_token_ce(logits, tokens))
            for layer in range(n_layers):
                for kind_idx, kind in enumerate(KINDS):
                    for mode, accumulator in (("recon", ce_recon), ("zero", ce_zero)):
                        patcher = _SiteSplice(bank, layer, kind, mode)
                        _, spliced_logits, _ = inference.forward(
                            tokens,
                            num_gen=1,
                            tokenize_final=False,
                            return_activations=False,
                            all_logits=True,
                            patcher=patcher,
                        )
                        accumulator[layer, kind_idx] += _next_token_ce(spliced_logits, tokens)
            print(f"[sae-eval] batch {batch_index + 1}/{batches} done (with CE)")
        else:
            print(f"[sae-eval] batch {batch_index + 1}/{batches} done")

    completed = len(ce_original) if with_ce else batches
    rows: list[dict[str, object]] = []
    mean_ce_original = sum(ce_original) / len(ce_original) if ce_original else None
    for layer in range(n_layers):
        for kind_idx, kind in enumerate(KINDS):
            count = float(element_count[layer, kind_idx])
            input_var = float(input_sq[layer, kind_idx]) / count - (float(input_sum[layer, kind_idx]) / count) ** 2
            residual_var = float(residual_sq[layer, kind_idx]) / count
            row: dict[str, object] = {
                "kind": kind,
                "layer": layer,
                "tokens": token_count,
                "explained_variance": 1.0 - residual_var / input_var if input_var > 0 else 0.0,
                "nrmse": (float(residual_sq[layer, kind_idx]) / float(input_sq[layer, kind_idx])) ** 0.5,
                "ce_original": "",
                "ce_reconstructed": "",
                "ce_zero_ablated": "",
                "ce_difference": "",
                "ce_recovered_pct": "",
            }
            if with_ce and mean_ce_original is not None and completed > 0:
                recon = float(ce_recon[layer, kind_idx]) / completed
                zero = float(ce_zero[layer, kind_idx]) / completed
                row["ce_original"] = mean_ce_original
                row["ce_reconstructed"] = recon
                row["ce_zero_ablated"] = zero
                row["ce_difference"] = recon - mean_ce_original
                denominator = zero - mean_ce_original
                row["ce_recovered_pct"] = 100.0 * (zero - recon) / denominator if denominator > 0 else ""
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SAE reconstruction quality.")
    parser.add_argument("--batches", type=int, default=4, help="Number of corpus batches to evaluate.")
    parser.add_argument("--seqs", type=int, default=128, help="Sequences per batch (memory bound).")
    parser.add_argument("--ce", action="store_true", help="Also compute CE recovered (72 extra passes/batch).")
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/sae-eval"), help="Output directory.")
    args = parser.parse_args(argv)

    rows = run_reconstruction_eval(batches=args.batches, seqs_per_batch=args.seqs, with_ce=args.ce)
    table_path = write_csv(args.out / "tables" / "sae-reconstruction-eval.csv", rows, TABLE_FIELDS)
    summary_path = write_json(
        args.out / "summaries" / "sae-reconstruction-eval.json",
        {"batches": args.batches, "seqs_per_batch": args.seqs, "with_ce": args.ce, "rows": rows},
    )
    print("wrote", table_path)
    print("wrote", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
