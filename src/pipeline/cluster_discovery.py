"""
Cluster Contrast Discovery runner.

Loads seq_repr and neg_ctx artifacts, clusters all neg_ctx sequences once via
cosine k-means, then runs ClusterContrastDiscovery on each of the top clusters
(ranked by size).  Accepted circuits are saved individually under
outputs/cluster_circuits/ so the run can be inspected or resumed incrementally.

Usage (standalone, from repo root):
    python -m pipeline.cluster_discovery

Or called from run_pipeline.py when "cluster_contrast" is in config.discovery.methods.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from circuit.discovery.cluster_contrast import (
    ClusterContrastDiscovery,
    ClusterResult,
    collect_neg_seq_ids,
    kmeans_cosine,
)
from config import config
from store.circuits import Circuit
from store.context import neg_ctx
from store.latent_stats import latent_stats
from store.seq_repr import SeqRepr

_OUTPUT_DIR = "outputs/cluster_circuits"


# ──────────────────────────────────────────────────────────────────────────────
# Artifact loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_neg_ctx_if_needed() -> None:
    if not neg_ctx._allocated:
        path = "outputs/neg_ctx.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"neg_ctx not found at {path!r}. Run the pipeline's ANN step first."
            )
        print(f"  Loading neg_ctx from {path!r} ...")
        neg_ctx.load(path)


def _load_latent_stats_if_needed() -> None:
    if not latent_stats._allocated:
        path = "outputs/latent_stats.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"latent_stats not found at {path!r}. Run Pass 1 first."
            )
        print(f"  Loading latent_stats from {path!r} ...")
        latent_stats.load(path)


def _load_seq_repr() -> SeqRepr:
    path = "outputs/seq_repr.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"seq_repr not found at {path!r}. Run Pass 1 first."
        )
    print(f"  Loading seq_repr from {path!r} ...")
    store = SeqRepr.__new__(SeqRepr)
    store.load(path)
    return store


# ──────────────────────────────────────────────────────────────────────────────
# Cluster centre resolution
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_center_ids(
    cluster_result: ClusterResult,
    cluster_idx: int,
    n: int,
) -> List[int]:
    """
    Return the IDs of the N sequences closest to cluster centroid `cluster_idx`.
    """
    mask    = cluster_result.labels == cluster_idx
    c_norm  = cluster_result.norm[mask]       # [size, D]
    c_ids   = cluster_result.seq_ids[mask]    # [size]
    centroid = cluster_result.centroids[cluster_idx]   # [D]

    sims = c_norm @ centroid                           # [size]
    top  = sims.argsort(descending=True)[:n]
    return c_ids[top].tolist()


def _cf_seq_ids(
    cluster_result: ClusterResult,
    target_cluster_idx: int,
    n: int,
) -> List[int]:
    """
    Return the IDs of the N sequences closest to the target cluster centroid
    that are NOT members of that cluster (hard negatives).
    """
    centroid     = cluster_result.centroids[target_cluster_idx]          # [D]
    outside_mask = cluster_result.labels != target_cluster_idx
    outside_norm = cluster_result.norm[outside_mask]                     # [M, D]
    outside_ids  = cluster_result.seq_ids[outside_mask]                  # [M]

    sims = outside_norm @ centroid                                        # [M]
    top  = sims.argsort(descending=True)[:n]
    return outside_ids[top].tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(results: List[Tuple[int, int, Optional[Circuit]]]) -> None:
    """Print a compact table of accepted circuits."""
    accepted = [(cid, sz, c) for cid, sz, c in results if c is not None]
    total    = len(results)

    print(f"\n{'='*70}")
    print(f"Cluster Contrast Discovery — {len(accepted)}/{total} clusters accepted")
    print(f"{'='*70}")
    if not accepted:
        print("  (no circuits accepted)")
        return

    has_eval = any(
        "faithfulness" in c.metadata  # type: ignore[union-attr]
        for _, _, c in accepted
    )

    if has_eval:
        hdr = (
            f"  {'Cluster':>8}  {'Size':>7}  {'Nodes':>6}  {'Edges':>6}"
            f"  {'Acts':>6}  {'Inh':>6}  {'Faith':>8}  {'Spec':>8}"
        )
    else:
        hdr = (
            f"  {'Cluster':>8}  {'Size':>7}  {'Nodes':>6}  {'Edges':>6}"
            f"  {'Activators':>10}  {'Inhibitors':>10}"
        )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cid, sz, circuit in accepted:
        m = circuit.metadata  # type: ignore[union-attr]
        base = (
            f"  {cid:>8}  {sz:>7,}  "
            f"{len(circuit.nodes):>6}  {len(circuit.edges):>6}  "  # type: ignore[union-attr]
            f"{m.get('n_activators', '?'):>6}  {m.get('n_inhibitors', '?'):>6}"
        )
        if has_eval:
            faith = m.get("faithfulness")
            spec  = m.get("specificity")
            faith_s = f"{faith:.4f}" if faith is not None else "    -"
            spec_s  = f"{spec:.4f}"  if spec  is not None else "    -"
            print(f"{base}  {faith_s:>8}  {spec_s:>8}")
        else:
            print(base)
    print()


def _save_summary(results: List[Tuple[int, int, Optional[Circuit]]], output_dir: str) -> None:
    """Persist a JSON summary of accepted circuits."""
    accepted = [
        {
            "cluster_id":   cid,
            "cluster_size": sz,
            "circuit_name": c.name,
            "circuit_uuid": c.uuid,
            "n_nodes":      len(c.nodes),
            "n_edges":      len(c.edges),
            "metadata":     {k: v for k, v in c.metadata.items()
                             if isinstance(v, (int, float, str, bool))},
        }
        for cid, sz, c in results if c is not None
    ]
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(accepted, f, indent=2)
    print(f"  Summary saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_cluster_contrast_discovery(
    inference: Any,
    bank: Any,
    loader: Any,
) -> List[Circuit]:
    """
    Full cluster-contrast discovery run.

    1. Loads seq_repr, neg_ctx, and latent_stats if not already in memory.
    2. Collects all unique sequence IDs from neg_ctx and fetches their embeddings.
    3. Runs cosine k-means once to partition the embedding space.
    4. For each of the top `top_clusters` clusters (by size):
       a. Resolves positive (top-N in-cluster closest to centroid) and
          negative (top-N out-of-cluster closest to centroid) sequence IDs.
       b. Calls ClusterContrastDiscovery.discover_cluster().
       c. Saves accepted circuits to outputs/cluster_circuits/.
    5. Prints and saves a summary table.

    Returns the list of accepted Circuit objects.
    """
    cfg = config.discovery.cluster_contrast

    print("--- Cluster Contrast Discovery ---")

    # ── Load artifacts ────────────────────────────────────────────────────────
    _load_neg_ctx_if_needed()
    _load_latent_stats_if_needed()
    seq_repr_store = _load_seq_repr()

    # ── Collect and embed neg_ctx sequences ──────────────────────────────────
    print("\nCollecting neg_ctx sequence IDs ...")
    seq_ids = collect_neg_seq_ids(neg_ctx)
    print(f"  {seq_ids.shape[0]:,} unique sequence IDs in neg_ctx")

    print("Fetching seq_repr embeddings ...")
    embeds = seq_repr_store.get_repr(seq_ids)          # [N, D] float32
    valid  = embeds.abs().sum(dim=1) > 0
    seq_ids = seq_ids[valid]
    embeds  = embeds[valid]
    N = seq_ids.shape[0]
    print(f"  {N:,} sequences with non-zero embeddings (repr_dim={embeds.shape[1]})")

    if N == 0:
        print("No sequences with embeddings — cannot cluster.  Skipping.")
        return []

    # ── K-means ───────────────────────────────────────────────────────────────
    k = min(int(cfg.n_clusters), N)
    print(f"\nRunning cosine k-means  k={k}  iters={cfg.kmeans_iters}  seed={cfg.kmeans_seed} ...")
    kmeans_device = bank.device   # GPU if available, otherwise CPU
    t0   = time.perf_counter()
    norm = F.normalize(embeds, dim=1)
    labels, centroids = kmeans_cosine(
        norm, k=k, n_iters=int(cfg.kmeans_iters), seed=int(cfg.kmeans_seed),
        device=kmeans_device,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f} s  (device={kmeans_device})")

    cluster_sizes = torch.bincount(labels, minlength=k)
    print(
        f"  Cluster sizes — "
        f"min={cluster_sizes.min().item()}  "
        f"max={cluster_sizes.max().item()}  "
        f"mean={cluster_sizes.float().mean().item():.1f}"
    )

    cluster_result = ClusterResult(
        labels=labels, centroids=centroids, seq_ids=seq_ids, norm=norm,
    )

    # ── Select top clusters ───────────────────────────────────────────────────
    top_n         = min(int(cfg.top_clusters), k)
    top_cluster_indices = cluster_sizes.argsort(descending=True)[:top_n].tolist()
    print(f"\nTop {top_n} clusters by size: {[int(ci) for ci in top_cluster_indices]}")

    # ── Prepare output directory ──────────────────────────────────────────────
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ── Discovery loop ────────────────────────────────────────────────────────
    discoverer = ClusterContrastDiscovery(inference, bank, loader)
    results: List[Tuple[int, int, Optional[Circuit]]] = []
    run_eval = bool(cfg.run_eval)

    pbar = tqdm(top_cluster_indices, desc="Cluster circuits", unit="cluster")
    for ci in pbar:
        ci = int(ci)
        cluster_size = int(cluster_sizes[ci].item())
        save_path = os.path.join(_OUTPUT_DIR, f"cluster_{ci}_circuit.pt")

        # Resolve sequence IDs (needed for both discovery and eval)
        center_ids = _cluster_center_ids(
            cluster_result, ci, n=int(cfg.num_pos_seqs),
        )
        cf_ids = _cf_seq_ids(
            cluster_result, ci,
            n=int(cfg.num_neg_seqs),
        )

        # Skip discovery if already saved, but still run eval if missing
        if os.path.exists(save_path):
            try:
                circuit = torch.load(save_path, weights_only=False)
            except Exception:
                circuit = None

            needs_eval = (
                run_eval
                and circuit is not None
                and "kl_faithfulness" not in circuit.metadata
            )
            if not needs_eval:
                pbar.write(f"  [cluster {ci}] already saved — skipping")
                results.append((ci, cluster_size, circuit))
                continue

            pbar.write(f"  [cluster {ci}] already saved — running eval only")
        else:
            pbar.write(
                f"  [cluster {ci}]  size={cluster_size:,}  "
                f"center_seqs={len(center_ids)}  cf_seqs={len(cf_ids)}"
            )

            circuit = discoverer.discover_cluster(
                cluster_id=ci,
                center_seq_ids=center_ids,
                cf_seq_ids=cf_ids,
                cluster_size=cluster_size,
            )

            if circuit is not None:
                torch.save(circuit, save_path)
                pbar.write(
                    f"  [cluster {ci}] ACCEPTED — "
                    f"{len(circuit.nodes)} nodes  {len(circuit.edges)} edges  "
                    f"→ {save_path}"
                )
            else:
                pbar.write(f"  [cluster {ci}] rejected")

        # ── Faithfulness eval ─────────────────────────────────────────────────
        if run_eval and circuit is not None:
            from eval.cluster_faithfulness import evaluate_cluster_faithfulness

            pos_tokens = discoverer._load_tokens(center_ids)
            neg_tokens = discoverer._load_tokens(cf_ids)
            device = bank.device
            pos_tokens = pos_tokens.to(device)
            neg_tokens = neg_tokens.to(device)

            pbar.write(f"  [cluster {ci}] running faithfulness eval ...")
            try:
                scores = evaluate_cluster_faithfulness(
                    inference=inference,
                    bank=bank,
                    circuit=circuit,
                    pos_tokens=pos_tokens,
                    neg_tokens=neg_tokens,
                    eval_position=str(cfg.eval_position),
                    batch_size=int(cfg.batch_size),
                )
                circuit.metadata.update(scores)
                pbar.write(
                    f"  [cluster {ci}] "
                    f"faith={scores['faithfulness']:.4f}  "
                    f"spec={scores['specificity']:.4f}  "
                    f"(F_M={scores['f_M']:.3f}  "
                    f"F_C={scores['f_C']:.3f}  "
                    f"F_∅={scores['f_empty']:.3f})"
                )
                torch.save(circuit, save_path)
            except Exception as exc:  # noqa: BLE001
                pbar.write(f"  [cluster {ci}] eval failed: {exc}")

        results.append((ci, cluster_size, circuit))

    pbar.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results)
    _save_summary(results, _OUTPUT_DIR)

    accepted = [c for _, _, c in results if c is not None]
    return accepted


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.loader import DataLoader
    from hardware import detect_devices
    from model.inference import Inference
    from sae.bank import SAEBank

    devices = detect_devices()
    device  = devices[0]

    print("Initializing resources ...")
    loader    = DataLoader(device=device, pin_memory=False)
    inference = Inference(device=device, compile=False)
    bank      = SAEBank(devices=devices, load_decoders=True, compile=False)

    run_cluster_contrast_discovery(inference, bank, loader)


if __name__ == "__main__":
    main()
