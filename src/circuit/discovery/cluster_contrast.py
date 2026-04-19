"""
Cluster Contrast Discovery: seed-free circuit discovery via embedding-space clustering.

All neg_ctx sequences are clustered once by their seq_repr embeddings using cosine
k-means.  For each of the top N clusters (by size) a circuit is discovered that
explains the cluster's characteristic output distribution.

The gradient target is the KL divergence between:
  - the target cluster's mean logit distribution (computed on center sequences), and
  - the current output distribution of counterfactual sequences (centers of other
    clusters).

A single grad-enabled forward pass through SAEGraphInstrument + one backward pass via
compute_latent_counterfactual_scores discovers the upstream latents responsible for the
output gap.  No seed latent is required.

Two node types are discovered:
  - cluster_activator: latents absent on counterfactual sequences whose presence would
    push the output toward the target cluster's distribution.
  - cluster_inhibitor: latents active on counterfactual sequences that are causally
    pulling the output away from the target distribution.
"""
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F

from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.types.feature_id import FeatureID
from config import config
from observability.circuit_logger import CircuitLogger
from store.circuits import Circuit, CircuitNode
from store.latent_stats import latent_stats


# ──────────────────────────────────────────────────────────────────────────────
# Clustering utilities  (also imported by the runner in pipeline/cluster_discovery.py)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    """Output of a single k-means run over neg_ctx sequence embeddings."""
    labels: torch.Tensor     # [N] long  — cluster index per sequence
    centroids: torch.Tensor  # [k, D] float32 — L2-normalised centroids
    seq_ids: torch.Tensor    # [N] long  — global 1-indexed sequence IDs
    norm: torch.Tensor       # [N, D] float32 — L2-normalised embeddings


def collect_neg_seq_ids(neg_ctx: Any) -> torch.Tensor:
    """
    Flatten the entire neg_ctx store and return deduplicated, non-zero sequence IDs.

    Args:
        neg_ctx: Context instance with ctx_seq_idx allocated.

    Returns:
        [M] long tensor of unique 1-indexed sequence IDs.
    """
    all_ids_flat = neg_ctx.ctx_seq_idx.reshape(-1)
    return torch.unique(all_ids_flat[all_ids_flat > 0]).long()


def kmeans_cosine(
    norm: torch.Tensor,
    k: int,
    n_iters: int = 40,
    chunk: int = 8192,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cosine k-means on pre-normalised embeddings.

    When ``device`` is a CUDA device the entire run executes on GPU:
      - Assignment: single unchunked cuBLAS matmul  ([N, D] @ [D, k] → [N, k])
      - Centroid update: vectorised scatter_add_ — one kernel instead of k Python loops

    On CPU the assignment is chunked (avoids a large [N, k] peak tensor) and
    the centroid update uses the same scatter_add_ path.

    Converges early when labels stop changing between iterations.

    Args:
        norm:    [N, D] float32 — L2-normalised sequence embeddings (any device).
        k:       Number of clusters (clamped to N if k > N).
        n_iters: Maximum iterations.
        chunk:   CPU assignment batch size (ignored on CUDA).
        seed:    Manual seed for centroid initialisation.
        device:  Target device.  Defaults to the device of ``norm``.

    Returns:
        labels     [N] long  — cluster index per sequence  (CPU)
        centroids  [k, D]    — L2-normalised cluster centroids  (CPU)
    """
    N, D = norm.shape
    k = min(k, N)

    if device is None:
        device = norm.device

    norm_d = norm.to(device)

    torch.manual_seed(seed)
    init_idx  = torch.randperm(N, device=device)[:k]
    centroids = F.normalize(norm_d[init_idx].clone(), dim=1)  # [k, D]
    labels    = torch.zeros(N, dtype=torch.long, device=device)

    on_cuda = device.type == "cuda"

    for iteration in range(n_iters):
        # ── Assignment ────────────────────────────────────────────────────────
        if on_cuda:
            # Full matmul fits in VRAM: N=200k × k=20 × 4B ≈ 16 MB
            new_labels = (norm_d @ centroids.T).argmax(dim=1)        # [N]
        else:
            new_labels = torch.empty(N, dtype=torch.long)
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                new_labels[start:end] = (norm_d[start:end] @ centroids.T).argmax(dim=1)

        # ── Early stopping ────────────────────────────────────────────────────
        if iteration > 0 and (new_labels == labels).all():
            break
        labels = new_labels

        # ── Centroid update (vectorised scatter_add_) ─────────────────────────
        # Avoids a Python loop over k clusters; one scatter per iteration.
        sums   = torch.zeros(k, D, dtype=torch.float32, device=device)
        expand = labels.unsqueeze(1).expand(-1, D)          # [N, D]
        sums.scatter_add_(0, expand, norm_d)

        counts = torch.bincount(labels, minlength=k).float().unsqueeze(1)  # [k, 1]
        # Keep empty-cluster centroids unchanged to avoid degenerate zeros
        nonempty = counts.squeeze(1) > 0
        centroids[nonempty]  = F.normalize(sums[nonempty] / counts[nonempty], dim=1)
        # centroids[~nonempty] unchanged

    return labels.cpu(), centroids.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Discovery class
# ──────────────────────────────────────────────────────────────────────────────

class ClusterContrastDiscovery:
    """
    Seed-free circuit discovery using embedding-space cluster contrast.

    Unlike DiscoveryMethod subclasses (which take a seed latent), this class
    operates on pre-computed ClusterResult objects.  The runner
    (pipeline/cluster_discovery.py) clusters all neg_ctx sequences once and
    then calls discover_cluster() for each of the top clusters.
    """

    method_name = "cluster_contrast"

    def __init__(self, inference: Any, sae_bank: Any, loader: Any) -> None:
        self.inference = inference
        self.sae_bank = sae_bank
        self.loader = loader

        cfg = config.discovery.cluster_contrast
        self.top_k_activators    = cast(int,   cfg.top_k_activators)
        self.top_k_inhibitors    = cast(int,   cfg.top_k_inhibitors)
        self.top_k_scope         = cast(str,   cfg.top_k_scope)
        self.activator_threshold = cast(float, cfg.activator_threshold)
        self.inhibitor_threshold = cast(float, cfg.inhibitor_threshold)
        self.min_active_count    = cast(int,   cfg.min_active_count)
        self.eval_position       = cast(str,   cfg.eval_position)
        self.num_pos_seqs        = cast(int,   cfg.num_pos_seqs)
        self.batch_size          = cast(int,   cfg.batch_size)

    # ── Token loading ─────────────────────────────────────────────────────────

    def _load_tokens(self, seq_ids: List[int], max_length: int = 64) -> torch.Tensor:
        """Load and pad a list of sequence IDs into a [N, max_length] long tensor."""
        if not seq_ids:
            return torch.zeros((0, max_length), dtype=torch.long)
        batches = list(self.loader.get_batches_by_ids(seq_ids, max_length=max_length))
        if not batches:
            return torch.zeros((0, max_length), dtype=torch.long)
        return torch.cat([tokens for _, tokens in batches], dim=0)

    # ── Target logit computation ───────────────────────────────────────────────

    def _compute_target_logits(
        self,
        center_tokens: torch.Tensor,   # [N, T]
    ) -> Optional[torch.Tensor]:
        """
        Runs no-grad forward passes (in batches) on center sequences and returns
        the mean logit vector over all sequences.

        Returns [vocab] float32, or None if the forward produces no logits.
        """
        use_all_logits = (self.eval_position == "all")
        N = center_tokens.shape[0]
        accumulator: Optional[torch.Tensor] = None
        n_batches = 0

        self.inference.disable_compile()
        try:
            for start in range(0, N, self.batch_size):
                batch = center_tokens[start : start + self.batch_size]
                _, logits, _ = self.inference.forward(
                    batch,
                    all_logits=use_all_logits,
                    return_activations=False,
                    tokenize_final=False,
                )
                if logits is None:
                    continue
                logits = logits.float()
                if use_all_logits:
                    batch_mean = logits.mean(dim=(0, 1))   # [vocab]
                else:
                    batch_mean = logits.mean(dim=0)        # [vocab]
                accumulator = batch_mean if accumulator is None else accumulator + batch_mean
                n_batches += 1
        finally:
            self.inference.enable_compile()

        if accumulator is None or n_batches == 0:
            return None
        return accumulator / n_batches

    # ── Counterfactual gradient hop ────────────────────────────────────────────

    def _run_cf_hop(
        self,
        cf_tokens: torch.Tensor,        # [N, T]
        target_logits: torch.Tensor,    # [vocab]
        logger: CircuitLogger,
    ) -> Tuple[Dict[FeatureID, float], Dict[FeatureID, float], float]:
        """
        Grad-enabled forward on counterfactual sequences → KL loss → backward.

        Runs in mini-batches of self.batch_size and accumulates raw gradient tensors
        before applying top-K once at the end (gradient accumulation). Each mini-batch
        gets its own SAEGraphInstrument so the peak VRAM is bounded by batch_size,
        not the total number of CF sequences.

        Score accumulation:
          activator[j]  += grad_act[:,:,j].sum() * B_cur   (raw gradient)
          inhibitor[j]  += (acts * grad)[:,:,j].sum() * B_cur
        Divide by N_total → per-sequence average, then apply top-K.

        Returns (activator_scores, inhibitor_scores, mean_kl).
        """
        from pipeline.component_index import component_idx as _comp_idx

        n_kinds    = len(self.sae_bank.kinds)
        kinds      = self.sae_bank.kinds
        last_layer = self.sae_bank.n_layer - 1
        use_all    = (self.eval_position == "all")
        N_total    = cf_tokens.shape[0]
        batch_size = self.batch_size

        # Raw score accumulators: (layer, kind) → [d_sae] weighted sum
        raw_act: Dict[Tuple[int, str], torch.Tensor] = {}
        raw_inh: Dict[Tuple[int, str], torch.Tensor] = {}
        kl_weighted_sum = 0.0
        n_valid = 0

        target_probs_cpu = F.softmax(target_logits.detach().cpu(), dim=-1)

        for start in range(0, N_total, batch_size):
            batch = cf_tokens[start : start + batch_size]
            B_cur = batch.shape[0]

            instrument = SAEGraphInstrument(self.sae_bank)
            self.inference.disable_compile()
            try:
                _, cf_logits, _ = self.inference.forward(
                    batch,
                    patcher=instrument,
                    grad_enabled=True,
                    all_logits=use_all,
                    return_activations=False,
                    tokenize_final=False,
                )

                if cf_logits is None:
                    continue

                cf_logits = cf_logits.float()
                if use_all:
                    cf_log_probs = F.log_softmax(cf_logits, dim=-1).mean(dim=1)
                else:
                    cf_log_probs = F.log_softmax(cf_logits, dim=-1)

                target_probs = target_probs_cpu.to(cf_log_probs.device)
                kl = -(target_probs.unsqueeze(0) * cf_log_probs).sum(dim=-1).mean()

                if abs(kl.item()) < 1e-8:
                    continue

                kl_weighted_sum += kl.item() * B_cur
                n_valid += B_cur
                target_scalar = -kl

                # Collect upstream leaf anchors in stable order
                upstream_pairs = [
                    (l, k) for (l, k) in instrument.graph.activations
                    if l <= last_layer
                ]
                anchors: List[torch.Tensor] = []
                for l, k in upstream_pairs:
                    for acts_grad, _, _ in instrument.graph.activations[(l, k)]:
                        if acts_grad.act is not None:
                            anchors.append(acts_grad.act)
                        if acts_grad.res is not None:
                            anchors.append(acts_grad.res)

                if not anchors:
                    continue

                grads = torch.autograd.grad(
                    target_scalar, anchors, retain_graph=False, allow_unused=True
                )

                # Accumulate weighted raw scores
                grad_iter = iter(grads)
                for l, k in upstream_pairs:
                    for acts_grad, _, _ in instrument.graph.activations[(l, k)]:
                        grad_act = next(grad_iter) if acts_grad.act is not None else None
                        _grad_res = next(grad_iter) if acts_grad.res is not None else None

                        if grad_act is None or acts_grad.act is None:
                            continue

                        # Weight by B_cur so dividing by N_total gives per-sequence mean
                        act_s = grad_act.sum(dim=(0, 1)).detach() * B_cur          # [d_sae]
                        inh_s = (acts_grad.act.detach() * grad_act).sum(dim=(0, 1)).detach() * B_cur

                        key = (l, k)
                        if key not in raw_act:
                            raw_act[key] = act_s
                            raw_inh[key] = inh_s
                        else:
                            raw_act[key] = raw_act[key] + act_s
                            raw_inh[key] = raw_inh[key] + inh_s

            finally:
                del instrument
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self.inference.enable_compile()

        if not raw_act or n_valid == 0:
            logger.note("no gradient signal across all mini-batches")
            return {}, {}, 0.0

        mean_kl = kl_weighted_sum / n_valid

        logger.note(
            f"KL loss: {mean_kl:.4f} | "
            f"eval_position: {self.eval_position} | "
            f"cf_seqs: {n_valid}/{N_total} valid"
        )

        # Normalise: per-sequence mean
        for key in raw_act:
            raw_act[key] = raw_act[key] / n_valid
            raw_inh[key] = raw_inh[key] / n_valid

        # Apply top-K filtering (mirrors compute_latent_counterfactual_scores)
        all_activators: List[Tuple[FeatureID, float]] = []
        all_inhibitors: List[Tuple[FeatureID, float]] = []

        for (l, k), scores_act in raw_act.items():
            kind_idx = kinds.index(k)
            c_idx = _comp_idx(l, kind_idx, n_kinds)
            scores_inh = raw_inh[(l, k)]

            if latent_stats.active_count is not None:
                count_mask = (latent_stats.active_count[c_idx] >= self.min_active_count).to(scores_act.device)
                scores_act = scores_act * count_mask
                scores_inh = scores_inh * count_mask

            # Activators: positive gradient
            pos_nz = (scores_act > 0).nonzero(as_tuple=False).squeeze(1)
            if pos_nz.numel() > 0:
                pos_vals = scores_act[pos_nz]
                if self.top_k_scope == "layer_kind":
                    k_act = min(self.top_k_activators, pos_nz.numel())
                    topk_vals, topk_local = pos_vals.topk(k_act)
                    sel = pos_nz[topk_local]
                else:
                    sel, topk_vals = pos_nz, pos_vals
                for idx_int, score in zip(sel.cpu().tolist(), topk_vals.cpu().tolist()):
                    all_activators.append((FeatureID(layer=l, kind=k, index=idx_int), score))

            # Inhibitors: negative acts*grad
            neg_nz = (scores_inh < 0).nonzero(as_tuple=False).squeeze(1)
            if neg_nz.numel() > 0:
                neg_vals = scores_inh[neg_nz]
                if self.top_k_scope == "layer_kind":
                    k_inh = min(self.top_k_inhibitors, neg_nz.numel())
                    topk_vals, topk_local = (-neg_vals).topk(k_inh)
                    topk_vals = -topk_vals
                    sel = neg_nz[topk_local]
                else:
                    sel, topk_vals = neg_nz, neg_vals
                for idx_int, score in zip(sel.cpu().tolist(), topk_vals.cpu().tolist()):
                    all_inhibitors.append((FeatureID(layer=l, kind=k, index=idx_int), score))

        if self.top_k_scope == "global":
            all_activators.sort(key=lambda x: x[1], reverse=True)
            all_inhibitors.sort(key=lambda x: x[1])
            all_activators = all_activators[:self.top_k_activators]
            all_inhibitors = all_inhibitors[:self.top_k_inhibitors]

        activator_scores = {fid: s for fid, s in all_activators}
        inhibitor_scores = {fid: s for fid, s in all_inhibitors}

        return activator_scores, inhibitor_scores, mean_kl

    # ── Top-level cluster entry point ─────────────────────────────────────────

    def discover_cluster(
        self,
        cluster_id: int,
        center_seq_ids: List[int],
        cf_seq_ids: List[int],
        cluster_size: int,
    ) -> Optional[Circuit]:
        """
        Runs discovery for a single cluster.

        Args:
            cluster_id:     Index of the cluster being explained.
            center_seq_ids: Sequence IDs of the most central cluster members
                            (used to compute the target logit distribution).
            cf_seq_ids:     Sequence IDs drawn from the centers of other clusters
                            (counterfactual dataset for the gradient pass).
            cluster_size:   Total number of sequences in this cluster.

        Returns:
            A Circuit if at least one node passes thresholds, else None.
        """
        logger = CircuitLogger(cluster_id, 0, self.method_name)
        try:
            return self._discover_cluster(
                cluster_id, center_seq_ids, cf_seq_ids, cluster_size, logger,
            )
        finally:
            logger.save()

    def _discover_cluster(
        self,
        cluster_id: int,
        center_seq_ids: List[int],
        cf_seq_ids: List[int],
        cluster_size: int,
        logger: CircuitLogger,
    ) -> Optional[Circuit]:
        n_kinds = len(self.sae_bank.kinds)
        kinds   = self.sae_bank.kinds
        device  = self.sae_bank.device

        logger._w(f"Cluster  id={cluster_id}  size={cluster_size}")
        logger._w(f"Center seqs: {len(center_seq_ids)}  |  CF seqs: {len(cf_seq_ids)}")
        logger._w("")

        # 1. Load tokens
        center_tokens = self._load_tokens(center_seq_ids[:self.num_pos_seqs])
        cf_tokens     = self._load_tokens(cf_seq_ids)

        if center_tokens.shape[0] == 0:
            logger.reject("no center sequences could be loaded")
            return None
        if cf_tokens.shape[0] == 0:
            logger.reject("no counterfactual sequences could be loaded")
            return None

        center_tokens = center_tokens.to(device)
        cf_tokens     = cf_tokens.to(device)

        # 2. Compute target logits (no grad)
        target_logits = self._compute_target_logits(center_tokens)
        if target_logits is None:
            logger.reject("target logit computation returned None")
            return None
        logger.note(
            f"target logits: vocab={target_logits.shape[0]}  "
            f"max={target_logits.max().item():.3f}  "
            f"min={target_logits.min().item():.3f}"
        )

        # 3. Counterfactual gradient pass
        activator_scores, inhibitor_scores, kl_loss = self._run_cf_hop(
            cf_tokens, target_logits, logger,
        )
        logger.stage(
            "cf grad pass", 1, 0,
            note=(
                f"{len(activator_scores)} absent activators, "
                f"{len(inhibitor_scores)} present inhibitors before thresholding"
            ),
        )

        # 4. Build circuit
        circuit = Circuit(name=f"ClusterContrast_C{cluster_id}")

        anchor_node = CircuitNode(metadata={
            "role": "cluster_anchor",
            "cluster_id": cluster_id,
            "cluster_size": cluster_size,
        })
        circuit.add_node(anchor_node)
        fid_to_uuid: Dict[FeatureID, str] = {}

        # 5. Add absent activators
        n_activators = 0
        for upstream_fid, score in activator_scores.items():
            if score < self.activator_threshold:
                continue
            upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)
            if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                continue
            if upstream_fid not in fid_to_uuid:
                node = CircuitNode(metadata={
                    "feature_id": upstream_fid,
                    "role": "cluster_activator",
                    "attribution_score": score,
                })
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], anchor_node.uuid, weight=score)
            n_activators += 1

        # 6. Add present inhibitors
        n_inhibitors = 0
        for upstream_fid, score in inhibitor_scores.items():
            if abs(score) < self.inhibitor_threshold:
                continue
            upstream_comp, upstream_latent = upstream_fid.to_component_id(n_kinds, kinds)
            if latent_stats.active_count[upstream_comp, upstream_latent] < self.min_active_count:
                continue
            if upstream_fid not in fid_to_uuid:
                node = CircuitNode(metadata={
                    "feature_id": upstream_fid,
                    "role": "cluster_inhibitor",
                    "attribution_score": score,
                })
                circuit.add_node(node)
                fid_to_uuid[upstream_fid] = node.uuid
            circuit.add_edge(fid_to_uuid[upstream_fid], anchor_node.uuid, weight=score)
            n_inhibitors += 1

        logger.stage(
            "circuit assembly",
            len(circuit.nodes), len(circuit.edges),
            note=f"{n_activators} activators, {n_inhibitors} inhibitors after thresholding",
        )

        if len(circuit.nodes) <= 1:
            logger.reject("no activators or inhibitors passed threshold")
            return None

        circuit.metadata.update({
            "cluster_id":        cluster_id,
            "cluster_size":      cluster_size,
            "kl_loss":           kl_loss,
            "n_activators":      n_activators,
            "n_inhibitors":      n_inhibitors,
            "discovery_method":  self.method_name,
            "n_nodes":           len(circuit.nodes),
            "n_edges":           len(circuit.edges),
            "target_logits":     target_logits.cpu(),
        })
        logger.nodes(list(circuit.nodes.values()))
        logger.accept(len(circuit.nodes), len(circuit.edges))
        return circuit
