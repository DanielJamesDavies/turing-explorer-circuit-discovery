"""Direct-effects matrix construction for attribution graphs."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional, Tuple

import torch

from circuit.types.feature_id import FeatureID

from .attribution_active_nodes import collect_active_feature_nodes
from .attribution_autograd import (
    _all_feature_anchors_with_meta,
    _score_grads_into_adj,
    _score_token_grads_into_adj,
    _upstream_anchors_for_target,
)
from .ct_influence import _compute_partial_neumann_influence
from .sae_graph import SAEGraphInstrument, SAEGraphInstrumentWithEmbedding


def _find_logit_targets(
    inference: Any,
    tokens: torch.Tensor,
    logit_top_k: int,
    desired_logit_prob: float = 0.95,
) -> Tuple[List[int], torch.Tensor]:
    """
    Selects logit target tokens by cumulative softmax probability, matching
    Anthropic's two-stage logit-target selection from attribute_transformerlens.py.

    Two-stage selection:
        Stage 1 — Pool cap: restrict candidates to the top-`logit_top_k` tokens by
                  softmax probability using torch.topk (O(vocab) without a full sort).
                  This guarantees that no low-rank token is ever selected, even if the
                  cumulative threshold hasn't been reached within the pool.
        Stage 2 — Cumulative cutoff: within the pool, keep the shortest prefix whose
                  cumulative probability >= desired_logit_prob.

    The returned probabilities are renormalised to sum to 1.0 over the selected set
    so they can be used directly as logit root weights in the Neumann series.

    Returns:
        token_ids:  Vocabulary indices of the selected targets (descending prob order).
        probs:      Renormalised float32 tensor [K] used for influence root weighting.
    """
    _, logits, _ = inference.forward(
        tokens,
        patcher=None,
        grad_enabled=False,
        return_activations=False,
        tokenize_final=False,
        all_logits=True,
    )
    if logits is None:
        return [], torch.zeros(0, dtype=torch.float32)

    last_logits = logits[:, -1, :].float() if logits.dim() == 3 else logits.float()
    mean_logits = last_logits.mean(dim=0)

    probs = torch.softmax(mean_logits, dim=0)

    # Stage 1: restrict to the top-logit_top_k pool (matches topk(max_n_logits) in original)
    pool_k = min(logit_top_k, probs.shape[0])
    top_p, top_idx = probs.topk(pool_k)

    # Stage 2: cumulative-prob cutoff within the pool
    cumsum = top_p.cumsum(dim=0)
    n_needed = int((cumsum < desired_logit_prob).sum().item()) + 1
    n = max(1, min(n_needed, pool_k))

    selected_idx = top_idx[:n]
    selected_probs = top_p[:n]

    # Renormalise so weights sum to 1.0 for Neumann series seeding
    prob_sum = selected_probs.sum()
    if prob_sum > 0:
        selected_probs = selected_probs / prob_sum

    return selected_idx.cpu().tolist(), selected_probs.cpu()


def _compute_partial_one_hop_influence(
    partial_adj: Dict[Tuple[int, int], float],
    n_feature_nodes: int,
    n_error_nodes: int,
    n_logit_nodes: int,
    logit_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a sparse one-hop logit-influence score for each feature node.

    Formula (row-normalised):
        influence[i] = Σ_j  lw[j] * |adj[(i, j)]| / Σ_k |adj[(k, j)]|

    where j ranges over logit-target node indices and lw[j] is the softmax
    probability weight for logit j.

    Node index layout expected in partial_adj:
        0 .. n_feature_nodes-1             — feature nodes
        n_feature_nodes .. +n_error_nodes  — error nodes
        n_feature_nodes+n_error_nodes ..   — logit sentinel nodes (n_logit_nodes of them)

    Args:
        partial_adj:     Sparse dict (src_idx, tgt_idx) → score.  Can be the full
                         adj or a ranking-only subset (partial_adj_ranking from Pass A).
        n_feature_nodes: Number of feature nodes at the head of the index space.
        n_error_nodes:   Number of error sentinel nodes (follows features).
        n_logit_nodes:   Number of logit sentinel nodes (follows error nodes).
        logit_probs:     Float32 tensor [n_logit_nodes] of softmax probabilities.
                         Falls back to uniform 1/K if shape doesn't match.

    Returns:
        Float32 CPU tensor of shape [n_feature_nodes].  Higher = more directly
        influential on the logit targets.
    """
    logit_start = n_feature_nodes + n_error_nodes

    # Logit root weights (same logic as compute_ct_influence)
    lw_dict: Dict[int, float] = {}
    if n_logit_nodes > 0:
        if logit_probs.shape[0] == n_logit_nodes:
            for j in range(n_logit_nodes):
                lw_dict[logit_start + j] = float(logit_probs[j])
        else:
            for j in range(n_logit_nodes):
                lw_dict[logit_start + j] = 1.0 / n_logit_nodes

    # L1 norm of each logit target's incoming column (for row-normalisation)
    logit_row_abs_sum: Dict[int, float] = {}
    for (s, t), v in partial_adj.items():
        if t in lw_dict:
            logit_row_abs_sum[t] = logit_row_abs_sum.get(t, 0.0) + abs(v)

    # Accumulate one-hop influence per feature node
    feature_inf_dict: Dict[int, float] = {}
    for (s, t), v in partial_adj.items():
        lw_t = lw_dict.get(t, 0.0)
        if lw_t == 0.0:
            continue
        row_sum = logit_row_abs_sum.get(t, 0.0)
        if row_sum < 1e-12:
            continue
        if 0 <= s < n_feature_nodes:
            feature_inf_dict[s] = (
                feature_inf_dict.get(s, 0.0) + lw_t * abs(v) / row_sum
            )

    # Dense result tensor (at most n_feature_nodes ≤ max_feature_nodes entries)
    result = torch.zeros(n_feature_nodes, dtype=torch.float32)
    for feat_idx, score in feature_inf_dict.items():
        result[feat_idx] = score
    return result


def compute_direct_effects_matrix(
    tokens: torch.Tensor,
    inference: Any,
    bank: Any,
    logit_top_k: int,
    probe_batch_size: int,
    kinds: List[str],
    n_kinds: int,
    min_active_count: int = 1,
    active_count: Optional[torch.Tensor] = None,
    max_feature_nodes: int = 0,
    stop_error_grad: bool = False,
    target_chunk_size: int = 8,  # deprecated — kept for call-site compatibility only
    desired_logit_prob: float = 0.95,
    include_error_nodes: bool = True,
    online_ranking_interval: int = 4,
    feature_batch_size: int = 32,
    include_token_nodes: bool = False,
) -> Tuple[Dict[Tuple[int, int], float], List[FeatureID], torch.Tensor]:
    """
    Builds a sparse prompt-local direct-effects adjacency matrix over active SAE
    latents (attn/mlp/resid across all layers), plus logit sentinel nodes.

    Matches the spirit of Anthropic's Attribution Graphs method: one no-grad
    discovery pass to collect and rank active nodes, then ONE retained-graph
    grad-enabled forward pass per sequence.  All target backward passes run
    against that single graph — eliminating the O(N/chunk) forward-pass overhead
    of the old chunked implementation.

    adj[(src_idx, tgt_idx)] =
        mean over sequences of sum_{B,T}( f_grad_src[..., src.index]
                                          * grad_src[..., src.index] )

    Node layout in all_nodes:
        [0 : N_feat]                        — feature nodes (FeatureID with layer >= 0)
        [N_feat : N_feat+E]                 — error nodes (kind=k+"_err", one per active
                                              (layer,kind)); omitted when include_error_nodes=False.
        [N_feat+E : N_feat+E+T]             — token sentinel nodes (layer=-2, kind="token",
                                              index=position); omitted when include_token_nodes=False.
        [N_feat+E+T :]                      — logit sentinel nodes (layer=-1, kind="logit").
        Logit nodes are always last so logit_start = N - logit_top_k is stable.

    Memory strategy for 16 GB VRAM:
        • probe_batch_size=1 keeps each retained graph small (≈ 1× forward pass).
        • max_feature_nodes caps how many target backward passes are run.
        • torch.cuda.empty_cache() + gc.collect() once per sequence after freeing
          the retained graph.

    The caller is responsible for:
        • calling inference.disable_compile() before and enable_compile() after.
        • using bank.pin_decoders() context to keep decoder weights resident.

    Args:
        tokens:            [N, T] probe sequences.
        inference:         Inference instance (compile already disabled).
        bank:              SAEBank (decoders already pinned).
        logit_top_k:       Number of logit sentinel nodes (0 = no logit targets).
        probe_batch_size:  Sequences per forward/backward pass.
        kinds:             Ordered SAE kind names, e.g. ["attn", "mlp", "resid"].
        n_kinds:           len(kinds).
        min_active_count:  Exclude latents with global firing count below this.
        active_count:      Optional [n_components, d_sae] lifetime-count tensor.
        max_feature_nodes:    Cap on feature nodes after logit-influence ranking. 0 = unlimited.
                              When logit_top_k > 0, all active features are first collected
                              (no early cap), logit backward passes run against the full set,
                              then the top-N by one-hop logit influence are kept.
                              When logit_top_k == 0, falls back to peak-activation ranking.
        stop_error_grad:      If True, error-term gradients are zeroed in SAEGraphInstrument.
        target_chunk_size:    Deprecated. No longer used. Retained for call-site compat.
        desired_logit_prob:   Cumulative softmax probability used to select logit targets.
                              Tokens are added in descending probability order until the
                              running sum >= desired_logit_prob (capped at logit_top_k).
        include_error_nodes:  If True (default), add one error sentinel node per active
                              (layer, kind) pair representing the SAE reconstruction error.
                              Error nodes appear as sources in adj (they influence downstream
                              features/logits) but are never backward-pass targets.
                              Set False to omit them entirely (equivalent to stop_error_grad
                              but also removes sentinels from all_nodes).
        include_token_nodes:  If True, add T token sentinel nodes (one per input position)
                              to all_nodes.  Each backward pass also differentiates w.r.t.
                              the input embedding captured by SAEGraphInstrumentWithEmbedding,
                              giving per-position token→feature attribution.  Default False.
        online_ranking_interval: Re-rank remaining unvisited features every N cycles
                              of Step 7's backward-pass loop.  Each cycle processes
                              `feature_batch_size` features; re-ranking re-sorts the
                              queue by current partial influence so features that gain
                              importance via feature→feature→logit paths are promoted
                              earlier.  Default 4 matches the original circuit-tracer
                              `update_interval=4`.  Set to 0 for one-shot ordering
                              (original Pass-A ranking only, no online updates).
        feature_batch_size:   Number of features processed per online-ranking cycle.
                              Only active when online_ranking_interval > 0.  Default 32
                              matches the original circuit-tracer `batch_size=32`.
                              Effective features per ranking update = interval × batch
                              = 4 × 32 = 128 (matching original queue size).

    Returns:
        (adj, all_nodes, logit_probs) — sparse dict, ordered node list, and float32
        tensor [K] of softmax probabilities for the K selected logit target tokens.
        Returns ({}, [], zeros(0)) if no active latents are found on the probe sequences.
    """
    n_token_batches_total = (tokens.shape[0] + probe_batch_size - 1) // probe_batch_size
    adj: Dict[Tuple[int, int], float] = {}
    n_token_batches = 0
    all_nodes: List[FeatureID] = []
    logit_probs: torch.Tensor = torch.zeros(0, dtype=torch.float32)

    for token_batch in tokens.split(probe_batch_size):
        B = token_batch.shape[0]
        batch_num = n_token_batches + 1

        # ------------------------------------------------------------------
        # Step 1: Node discovery — no-grad forward for this batch
        # ------------------------------------------------------------------
        disc = SAEGraphInstrument(bank, stop_error_grad=stop_error_grad)
        with torch.no_grad():
            inference.forward(
                token_batch,
                patcher=disc,
                grad_enabled=False,
                return_activations=False,
                tokenize_final=False,
                all_logits=False,
            )
        # Collect ALL active features (no cap here when logit_top_k > 0 — selection
        # happens post-logit-ranking in Step 4).  Fall back to activation-magnitude
        # cap only when logit_top_k == 0 (no logit targets to rank by).
        discovery_cap = 0 if logit_top_k > 0 else max_feature_nodes
        feature_nodes_all = collect_active_feature_nodes(
            disc.graph, kinds, n_kinds, min_active_count, active_count,
            max_feature_nodes=discovery_cap,
        )
        # Collect error sentinels — one per (layer, kind) pair the SAE processed.
        # Must happen before del disc while graph.activations is still populated.
        error_nodes_all: List[FeatureID] = (
            [
                FeatureID(layer=l, kind=k + "_err", index=0)
                for (l, k) in sorted(disc.graph.activations.keys())
            ]
            if include_error_nodes else []
        )
        del disc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if not feature_nodes_all:
            print(
                f"[CTBaseline]   Batch {batch_num}/{n_token_batches_total}: "
                "no active features found, skipping.",
                flush=True,
            )
            n_token_batches += 1
            continue

        # ------------------------------------------------------------------
        # Step 2: Logit sentinel token IDs (cumulative probability selection)
        # ------------------------------------------------------------------
        logit_node_ids: List[int] = []
        batch_logit_probs: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        if logit_top_k > 0:
            logit_node_ids, batch_logit_probs = _find_logit_targets(
                inference, token_batch, logit_top_k, desired_logit_prob
            )
        # Keep probabilities from the first batch (representative for the circuit)
        if logit_probs.shape[0] == 0 and batch_logit_probs.shape[0] > 0:
            logit_probs = batch_logit_probs

        logit_nodes = [FeatureID(layer=-1, kind="logit", index=tid) for tid in logit_node_ids]
        n_logit_nodes_batch = len(logit_nodes)

        # ------------------------------------------------------------------
        # Step 3: Single retained-graph forward pass
        # ------------------------------------------------------------------
        # all_logits=True so the logits tensor is available for logit-target
        # backward passes without a second forward pass.
        if include_token_nodes:
            # SAEGraphInstrumentWithEmbedding captures the first-layer residual
            # stream as a detached leaf anchor for per-position attribution.
            instrument: SAEGraphInstrument = SAEGraphInstrumentWithEmbedding(
                bank,
                stop_error_grad=stop_error_grad,
                first_layer=0,
                first_kind=kinds[0],
            )
        else:
            instrument = SAEGraphInstrument(bank, stop_error_grad=stop_error_grad)
        try:
            _, logits, _ = inference.forward(
                token_batch,
                patcher=instrument,
                grad_enabled=True,
                return_activations=False,
                tokenize_final=False,
                all_logits=True,
            )

            # Build token-node list (T positions) when enabled and anchor was captured.
            T = token_batch.shape[1]
            emb_anchor: Optional[torch.Tensor] = None
            token_nodes: List[FeatureID] = []
            if (include_token_nodes
                    and isinstance(instrument, SAEGraphInstrumentWithEmbedding)
                    and instrument.emb_anchor is not None):
                emb_anchor = instrument.emb_anchor
                token_nodes = [
                    FeatureID(layer=-2, kind="token", index=p) for p in range(T)
                ]

            # Pre-cache upstream anchor groups: one lookup per (layer, kind) key.
            upstream_cache: Dict[Tuple[int, str], Tuple[List[torch.Tensor], List]] = {}
            for (l, k) in instrument.graph.activations.keys():
                ups_anchors, ups_meta = _upstream_anchors_for_target(
                    instrument.graph, l, k, kinds, n_kinds
                )
                if ups_anchors:
                    upstream_cache[(l, k)] = (ups_anchors, ups_meta)

            # All anchors for logit targets (every layer is upstream of logit output).
            all_anchors, all_anchor_meta = _all_feature_anchors_with_meta(instrument.graph)

            # ------------------------------------------------------------------
            # Step 4: Logit ranking passes (Pass A) — determine feature priority
            # ------------------------------------------------------------------
            # Run logit backward passes against the FULL feature set to score each
            # feature's direct influence on every logit target.  Results go into
            # partial_adj_ranking (used only for ranking; discarded afterwards).
            # All passes retain the graph since feature passes will follow.
            feature_nodes: List[FeatureID] = feature_nodes_all   # default: no cap
            n_feature_nodes_all = len(feature_nodes_all)

            if (logit_top_k > 0
                    and logit_node_ids
                    and logits is not None
                    and logits.grad_fn is not None):
                last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
                # Layout: features | error_nodes | logit_nodes
                # (mirrors final all_nodes so error sentinel indices are stable)
                all_nodes_full = feature_nodes_all + error_nodes_all + logit_nodes
                nti_full: Dict[FeatureID, int] = {n: i for i, n in enumerate(all_nodes_full)}
                partial_adj_ranking: Dict[Tuple[int, int], float] = {}

                for logit_node, token_id in zip(logit_nodes, logit_node_ids):
                    logit_tgt_full = nti_full.get(logit_node)
                    if logit_tgt_full is None or token_id >= last_logits.shape[-1]:
                        continue
                    scalar = (last_logits[:, token_id] - last_logits.mean(dim=-1)).sum()
                    if scalar.grad_fn is None:
                        continue
                    grads = torch.autograd.grad(
                        scalar, all_anchors, retain_graph=True, allow_unused=True
                    )
                    _score_grads_into_adj(
                        grads, all_anchor_meta, instrument.graph,
                        feature_nodes_all, nti_full, logit_tgt_full, partial_adj_ranking,
                    )

                # ------------------------------------------------------------------
                # Step 5: Rank features by one-hop logit influence; apply cap
                # ------------------------------------------------------------------
                # Computed SPARSELY to avoid a dense N_full×N_full matrix.
                # partial_adj_ranking has only K logit-target columns (K ≤ logit_top_k),
                # so we compute row-normalised one-hop influence directly from the dict.
                #
                # one_hop[i] = Σ_j  lw[j] * |adj[(i,j)]| / row_abs_sum[j]
                #
                # where j ranges over logit targets and row_abs_sum[j] is the L1 norm
                # of column j (= row j of A_norm since A is transposed before storage).
                if max_feature_nodes > 0 and max_feature_nodes < n_feature_nodes_all:
                    n_error_nodes_all = len(error_nodes_all)
                    feature_inf_tensor = _compute_partial_one_hop_influence(
                        partial_adj_ranking,
                        n_feature_nodes_all,
                        n_error_nodes_all,
                        n_logit_nodes_batch,
                        batch_logit_probs,
                    )
                    cap = min(max_feature_nodes, n_feature_nodes_all)
                    top_sorted = sorted(feature_inf_tensor.topk(cap).indices.tolist())
                    feature_nodes = [feature_nodes_all[i] for i in top_sorted]
                    print(
                        f"[CTBaseline]     Logit-influence ranking: "
                        f"{len(feature_nodes)}/{n_feature_nodes_all} features selected",
                        flush=True,
                    )

            # Canonical node list: features | error_nodes | token_nodes | logit_sentinels.
            # Established on the first batch; later batches score into the same indices.
            if not all_nodes:
                all_nodes = feature_nodes + error_nodes_all + token_nodes + logit_nodes

            node_to_idx: Dict[FeatureID, int] = {n: i for i, n in enumerate(all_nodes)}
            n_feature_nodes = len(feature_nodes)
            n_error_nodes = len(error_nodes_all)
            n_total_targets = n_feature_nodes + n_logit_nodes_batch

            n_token_nodes = len(token_nodes)
            print(
                f"[CTBaseline]   Batch {batch_num}/{n_token_batches_total} | "
                f"{n_feature_nodes} feature + {n_error_nodes} error"
                + (f" + {n_token_nodes} token" if n_token_nodes else "")
                + f" + {n_logit_nodes_batch} logit nodes"
                f" → {n_logit_nodes_batch + n_feature_nodes} backward passes",
                flush=True,
            )

            # ------------------------------------------------------------------
            # Step 6: Logit-target backward passes — Pass B (final, into adj)
            # ------------------------------------------------------------------
            # The retained graph is still alive from Pass A.  All logit passes
            # retain the graph since feature passes come after.
            # When include_token_nodes=True, emb_anchor is appended to the anchor
            # list so token attribution is computed alongside feature attribution.
            all_anchors_b = all_anchors + ([emb_anchor] if emb_anchor is not None else [])

            if logit_node_ids and logits is not None and logits.grad_fn is not None:
                last_logits = logits[:, -1, :] if logits.dim() == 3 else logits

                for logit_node, token_id in zip(logit_nodes, logit_node_ids):
                    logit_tgt_idx = node_to_idx.get(logit_node)
                    if logit_tgt_idx is None or token_id >= last_logits.shape[-1]:
                        continue
                    # Demean: attribute relative to mean logit, matching the
                    # original circuit tracer's use of W_U[i] - W_U.mean(0).
                    scalar = (last_logits[:, token_id] - last_logits.mean(dim=-1)).sum()
                    if scalar.grad_fn is None:
                        continue
                    grads = torch.autograd.grad(
                        scalar, all_anchors_b,
                        retain_graph=True,  # always retain: feature passes follow
                        allow_unused=True,
                    )
                    if emb_anchor is not None:
                        _score_grads_into_adj(
                            grads[:-1], all_anchor_meta, instrument.graph,
                            feature_nodes, node_to_idx, logit_tgt_idx, adj,
                        )
                        if grads[-1] is not None:
                            _score_token_grads_into_adj(
                                grads[-1], emb_anchor, token_nodes,
                                node_to_idx, logit_tgt_idx, adj,
                            )
                    else:
                        _score_grads_into_adj(
                            grads, all_anchor_meta, instrument.graph,
                            feature_nodes, node_to_idx, logit_tgt_idx, adj,
                        )

                print(
                    f"[CTBaseline]     {n_logit_nodes_batch} logit targets scored"
                    f" | edges so far: {len(adj)}",
                    flush=True,
                )

            # ------------------------------------------------------------------
            # Step 7: Feature-target backward passes (selected features only)
            # ------------------------------------------------------------------
            # Helper to run a single feature backward pass.  Returns True if a
            # torch.autograd.grad call was made, False if the feature was skipped.
            def _run_one_feature_pass(
                tgt_node: FeatureID, retain: bool
            ) -> bool:
                tgt_idx = node_to_idx.get(tgt_node)
                if tgt_idx is None:
                    return False
                try:
                    _, tgt_connected, _ = instrument.graph.get_latents(
                        tgt_node.layer, tgt_node.kind
                    )
                except (KeyError, IndexError):
                    return False
                if tgt_connected.act is None:
                    return False
                target_acts = tgt_connected.act[..., tgt_node.index]  # [B, T]
                pos_argmax = target_acts.argmax(dim=-1)                # [B]
                b_idx = torch.arange(B, device=target_acts.device)
                scalar = target_acts[b_idx, pos_argmax].sum()
                if scalar.grad_fn is None:
                    return False

                cached = upstream_cache.get((tgt_node.layer, tgt_node.kind))
                feat_anchors: List[torch.Tensor] = []
                feat_meta: List = []
                if cached is not None:
                    feat_anchors, feat_meta = cached

                # Append emb_anchor so token attribution is computed alongside
                # feature attribution in a single autograd.grad call.
                pass_anchors = feat_anchors + ([emb_anchor] if emb_anchor is not None else [])
                if not pass_anchors:
                    return False

                grads = torch.autograd.grad(
                    scalar, pass_anchors, retain_graph=retain, allow_unused=True
                )
                if feat_meta:
                    _score_grads_into_adj(
                        grads[:len(feat_anchors)], feat_meta, instrument.graph,
                        feature_nodes, node_to_idx, tgt_idx, adj,
                    )
                if emb_anchor is not None and grads[-1] is not None:
                    _score_token_grads_into_adj(
                        grads[-1], emb_anchor, token_nodes,
                        node_to_idx, tgt_idx, adj,
                    )
                return True

            use_online = (
                online_ranking_interval > 0
                and feature_batch_size > 0
                and n_feature_nodes > 1
            )

            if use_online:
                # Online path: re-rank unvisited features every
                # online_ranking_interval cycles so features promoted by
                # feature→feature→logit paths get processed earlier.
                unvisited: List[int] = list(range(n_feature_nodes))
                n_features_done = 0
                cycle = 0

                while unvisited:
                    if cycle % online_ranking_interval == 0:
                        # Full Neumann series on the growing partial adj, matching the
                        # original's compute_partial_influences (multi-hop ranking so
                        # feature A is promoted when it strongly drives already-processed
                        # feature B which in turn drives the logit).
                        n_nodes_total = len(all_nodes)
                        logit_start = n_feature_nodes + n_error_nodes + n_token_nodes
                        lw = torch.zeros(n_nodes_total, dtype=torch.float32)
                        if batch_logit_probs.shape[0] == n_logit_nodes_batch:
                            lw[logit_start:logit_start + n_logit_nodes_batch] = (
                                batch_logit_probs.float()
                            )
                        elif n_logit_nodes_batch > 0:
                            lw[logit_start:logit_start + n_logit_nodes_batch] = (
                                1.0 / n_logit_nodes_batch
                            )
                        full_inf = _compute_partial_neumann_influence(
                            adj, n_nodes_total, lw, max_iter=128
                        )
                        inf_vec = full_inf[:n_feature_nodes]
                        unvisited.sort(key=lambda fi: -float(inf_vec[fi]))

                    batch_fi = unvisited[:feature_batch_size]
                    unvisited = unvisited[feature_batch_size:]

                    for k, local_i in enumerate(batch_fi):
                        # retain=False only on the very last feature slot;
                        # use slot count (not backward count) — safe to
                        # over-retain; del instrument cleans up anyway.
                        slot = n_features_done + k
                        retain = (slot + 1) < n_feature_nodes
                        _run_one_feature_pass(feature_nodes[local_i], retain)

                    n_features_done += len(batch_fi)
                    cycle += 1
                    print(
                        f"[CTBaseline]     Online cycle {cycle}:"
                        f" {n_features_done}/{n_feature_nodes} features"
                        f" | edges: {len(adj)}",
                        flush=True,
                    )
            else:
                # One-shot sequential path (default)
                for i, tgt_node in enumerate(feature_nodes):
                    retain = (n_logit_nodes_batch + i) < n_total_targets - 1
                    _run_one_feature_pass(tgt_node, retain)

                    if (i + 1) % 256 == 0 or i == n_feature_nodes - 1:
                        print(
                            f"[CTBaseline]     {n_logit_nodes_batch + i + 1}"
                            f"/{n_total_targets} targets done | edges: {len(adj)}",
                            flush=True,
                        )

        finally:
            # ------------------------------------------------------------------
            # Step 6: Free the retained graph (runs even if a backward raises)
            # ------------------------------------------------------------------
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        n_token_batches += 1
        print(
            f"[CTBaseline]   Batch {n_token_batches}/{n_token_batches_total} done"
            f" | total edges: {len(adj)}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Step 7: Average scores across token batches
    # ------------------------------------------------------------------
    if n_token_batches > 1:
        adj = {k: v / n_token_batches for k, v in adj.items()}

    return adj, all_nodes, logit_probs


__all__ = ["compute_direct_effects_matrix"]
