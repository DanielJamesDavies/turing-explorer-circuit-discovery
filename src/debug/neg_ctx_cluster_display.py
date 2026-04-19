"""
Cluster all neg_ctx sequences by embedding space and display representative
sequences from the largest clusters.

After Pass 1 and the neg_ctx build, every sequence that was ever assigned as a
hard negative for any latent gets a pooled residual-stream embedding stored in
seq_repr.  This script:

  1. Collects every unique sequence ID that appears in neg_ctx (across all
     components and latents).
  2. Retrieves their seq_repr embeddings.
  3. Runs cosine k-means (pure PyTorch, no external deps) to cluster them.
  4. Prints the top clusters by size, each with their most central sequences
     decoded to text.

Run from the repo root:
    python -m debug.neg_ctx_cluster_display
    python -m debug.neg_ctx_cluster_display --k 30 --top-clusters 10 --show 5
"""
import sys
import os
import argparse
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from store.context import neg_ctx
from store.seq_repr import SeqRepr
from data.loader import DataLoader
from model.tokenizer import Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# K-means
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_cosine(
    norm: torch.Tensor,     # [N, D] — already L2-normalised
    k: int,
    n_iters: int,
    chunk: int = 8192,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine k-means on pre-normalised embeddings.

    Returns:
        labels     [N]    — cluster index per sequence
        centroids  [k, D] — L2-normalised cluster centroids
    """
    N = norm.shape[0]
    k = min(k, N)

    torch.manual_seed(seed)
    centroids = F.normalize(norm[torch.randperm(N)[:k]].clone(), dim=1)  # [k, D]

    labels = torch.zeros(N, dtype=torch.long)

    for iteration in range(n_iters):
        # Assignment — chunked to avoid a single [N, k] allocation
        new_labels = torch.empty(N, dtype=torch.long)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            sims = norm[start:end] @ centroids.T          # [C, k]
            new_labels[start:end] = sims.argmax(dim=1)

        # Early stopping if assignments didn't change
        if iteration > 0 and (new_labels == labels).all():
            break
        labels = new_labels

        # Centroid update
        new_centroids = torch.zeros_like(centroids)
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                new_centroids[ci] = F.normalize(norm[mask].mean(dim=0), dim=0)
            else:
                new_centroids[ci] = centroids[ci]         # keep; avoid degenerate empty cluster
        centroids = new_centroids

    return labels, centroids


# ──────────────────────────────────────────────────────────────────────────────
# Sequence display helpers
# ──────────────────────────────────────────────────────────────────────────────

_WRAP_WIDTH = 100

def _fmt_sequence(text: str, max_chars: int = 300) -> str:
    """Wrap and truncate decoded sequence text for display."""
    text = text.replace("\n", "↵").replace("\r", "")
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    lines = textwrap.wrap(text, width=_WRAP_WIDTH) or [text]
    indent = "      "
    return ("\n" + indent).join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster neg_ctx sequences by embedding and display representatives."
    )
    parser.add_argument("--k",            type=int, default=20,
                        help="Number of k-means clusters (default: 20)")
    parser.add_argument("--top-clusters", type=int, default=8,
                        help="Largest clusters to display (default: 8)")
    parser.add_argument("--show",         type=int, default=4,
                        help="Sequences to show per cluster (default: 4)")
    parser.add_argument("--iters",        type=int, default=40,
                        help="K-means iterations (default: 40)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for centroid initialisation (default: 42)")
    parser.add_argument("--neg-ctx",      type=str, default="outputs/neg_ctx.pt",
                        help="Path to neg_ctx checkpoint")
    parser.add_argument("--seq-repr",     type=str, default="outputs/seq_repr.pt",
                        help="Path to seq_repr checkpoint")
    args = parser.parse_args()

    # ── Load stores ──────────────────────────────────────────────────────────
    print(f"Loading neg_ctx from  {args.neg_ctx!r} ...")
    neg_ctx.load(args.neg_ctx)

    print(f"Loading seq_repr from {args.seq_repr!r} ...")
    seq_repr_store = SeqRepr.__new__(SeqRepr)
    seq_repr_store.load(args.seq_repr)

    loader    = DataLoader(device=torch.device("cpu"), pin_memory=False)
    tokenizer = Tokenizer()

    # ── Collect all unique neg-ctx sequence IDs ───────────────────────────────
    print("\nCollecting unique neg_ctx sequence IDs ...")
    all_ids_flat = neg_ctx.ctx_seq_idx.reshape(-1)                         # [n_comp * d_sae * n_neg]
    seq_ids = torch.unique(all_ids_flat[all_ids_flat > 0]).long()          # deduplicated, 1-indexed
    print(f"  {seq_ids.shape[0]:,} unique sequence IDs found in neg_ctx")

    # ── Fetch embeddings and filter to those present in seq_repr ─────────────
    print("Fetching seq_repr embeddings ...")
    embeds = seq_repr_store.get_repr(seq_ids)                              # [N, 1024] float32
    valid  = embeds.abs().sum(dim=1) > 0
    seq_ids = seq_ids[valid]
    embeds  = embeds[valid]

    N = seq_ids.shape[0]
    print(f"  {N:,} sequences have non-zero embeddings (repr_dim={embeds.shape[1]})")

    if N == 0:
        print("No sequences with embeddings — nothing to cluster.")
        return

    # ── K-means ───────────────────────────────────────────────────────────────
    k = min(args.k, N)
    print(f"\nRunning cosine k-means  k={k}  iters={args.iters}  seed={args.seed} ...")
    norm = F.normalize(embeds, dim=1)
    labels, centroids = _kmeans_cosine(norm, k=k, n_iters=args.iters, seed=args.seed)

    cluster_sizes = torch.bincount(labels, minlength=k)
    print(f"  Cluster sizes — min={cluster_sizes.min()}  "
          f"max={cluster_sizes.max()}  "
          f"mean={cluster_sizes.float().mean():.1f}")

    # ── Display top clusters ──────────────────────────────────────────────────
    top_n     = min(args.top_clusters, k)
    top_ci    = cluster_sizes.argsort(descending=True)[:top_n].tolist()
    n_show    = args.show

    print(f"\nDisplaying top {top_n} clusters (by size), {n_show} representative sequences each")
    print("=" * 80)

    for rank, ci in enumerate(top_ci):
        size = int(cluster_sizes[ci].item())
        mask = labels == ci
        cluster_seq_ids = seq_ids[mask]
        cluster_norm    = norm[mask]                                       # [size, D]

        # Rank within cluster by cosine sim to centroid (descending = most central first)
        sims_to_centroid = cluster_norm @ centroids[ci]                    # [size]
        top_within       = sims_to_centroid.argsort(descending=True)[:n_show]

        print(f"\nCluster {ci}  (rank #{rank + 1} by size  |  {size} sequences)")
        print("-" * 60)

        for pos, idx in enumerate(top_within.tolist()):
            sid  = int(cluster_seq_ids[idx].item())
            sim  = float(sims_to_centroid[idx].item())

            try:
                tokens = loader.get_sequence(sid)
                text   = tokenizer.decode(tokens[:64].tolist())
                body   = _fmt_sequence(text)
            except Exception as exc:
                body = f"<decode error: {exc}>"

            print(f"  [{pos + 1}] seq_id={sid}  cosine_sim={sim:.4f}")
            print(f"      {body}")

    print("\n" + "=" * 80)
    print(f"Done.  {N:,} sequences  |  {k} clusters  |  top {top_n} displayed")


if __name__ == "__main__":
    main()
