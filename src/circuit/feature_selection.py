import os
import json
import torch
from typing import List, Dict, Any, Tuple, Optional, cast
from config import config
from pipeline.component_index import split_component_idx
from store.latent_stats import latent_stats
from store.logit_context import logit_ctx
from store.top_coactivation import top_coactivation
from store.context import top_ctx, mid_ctx, neg_ctx

_N_KINDS = 3  # attn / mlp / resid
_KINDS = ["attn", "mlp", "resid"]


def _seed_passes_filter(
    candidate: Dict[str, Any],
    filter_layers: "set[int]",
    filter_kinds: "set[str]",
) -> bool:
    """Return True if the candidate satisfies the active layer/kind allowlists.

    An empty set means "no constraint" for that dimension, so a candidate only
    needs to be rejected when it falls outside a *non-empty* allowlist.
    """
    layer_idx, kind_idx = split_component_idx(candidate["comp_idx"], _N_KINDS)
    if filter_layers and layer_idx not in filter_layers:
        return False
    if filter_kinds and _KINDS[kind_idx] not in filter_kinds:
        return False
    return True


class CandidateSelector:
    """
    Identifies 'Seed Latents' for global circuit discovery.
    Uses multi-dimensional scoring from Pass 1 & 2 stores to rank latents
    that are likely to be part of interesting mechanisms.

    Active criteria are controlled by config.discovery.seed_criteria, which
    allows full ablation of individual signals.
    """

    def __init__(self, n_seeds: Optional[int] = None, device: Optional[torch.device] = None):
        self.n_seeds = n_seeds or cast(int, config.discovery.n_seeds or 1000)
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def select_candidates(self) -> List[Dict[str, Any]]:
        """
        Runs all enabled scoring criteria and returns a merged, deduplicated
        list of candidate seed latents sorted by combined score.
        """
        active = set(config.discovery.seed_criteria)
        print(f"[CandidateSelector] Active seed criteria: {sorted(active)}")

        sf = config.discovery.seed_filter
        filter_layers: set = set(sf.layers)
        filter_kinds: set = set(sf.kinds)
        if filter_layers or filter_kinds:
            print(
                f"[CandidateSelector] Seed filter active — "
                f"layers: {sorted(filter_layers) if filter_layers else 'all'}, "
                f"kinds: {sorted(filter_kinds) if filter_kinds else 'all'}"
            )

        all_seeds: List[List[Dict[str, Any]]] = []

        # ------------------------------------------------------------------
        # 1. Logit Impact — sum of stored top-k token probabilities
        # ------------------------------------------------------------------
        if "logit_impact" in active:
            score = logit_ctx.top_probs.to(self.device).sum(dim=-1)
            all_seeds.append(self._top_k(score, "logit_impact"))

        # ------------------------------------------------------------------
        # 2. Connectivity — sum of co-activation weights (structural hubs)
        # ------------------------------------------------------------------
        if "connectivity" in active:
            score = top_coactivation.top_values.to(self.device)
            if top_coactivation.mode == "pmi":
                score = score.clamp(min=0)
            score = score.sum(dim=-1)
            all_seeds.append(self._top_k(score, "connectivity"))

        # ------------------------------------------------------------------
        # 3. Surprise — mean_seq * rarity boost (strong but rare latents)
        # ------------------------------------------------------------------
        if "surprise" in active:
            total_seqs = float(latent_stats.seq_count.max().item()) + 1.0
            p_fire = (latent_stats.seq_count.to(self.device).float() + 1e-6) / total_seqs
            rarity = torch.log10(1.0 / p_fire).clamp(1.0, 8.0)
            score = latent_stats.mean_seq.to(self.device) * rarity
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
            all_seeds.append(self._top_k(score, "surprise"))

        # ------------------------------------------------------------------
        # 4. Context Coherence — mean activation across top stored contexts
        # ------------------------------------------------------------------
        if "context_coherence" in active:
            score = top_ctx.ctx_seq_val.to(self.device).float().mean(dim=-1)
            all_seeds.append(self._top_k(score, "context_coherence"))

        # ------------------------------------------------------------------
        # 5. Activation Variance — coefficient of variation (std_seq / mean_seq)
        #    High CV = fires strongly in some contexts but barely in others.
        # ------------------------------------------------------------------
        if "activation_variance" in active:
            std_seq  = latent_stats.std_seq().to(self.device)
            mean_seq = latent_stats.mean_seq.to(self.device)
            score = std_seq / mean_seq.clamp(min=1e-6)
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 10, -1e9)
            all_seeds.append(self._top_k(score, "activation_variance"))

        # ------------------------------------------------------------------
        # 6. Logit Specificity — negative entropy of token prediction dist.
        #    More concentrated predictions (few specific tokens) → higher score.
        # ------------------------------------------------------------------
        if "logit_specificity" in active:
            p = logit_ctx.top_probs.to(self.device).clamp(min=1e-9)
            p = p / p.sum(dim=-1, keepdim=True)
            score = (p * p.log()).sum(dim=-1)   # negative entropy; higher = more specific
            score = score.masked_fill(logit_ctx.latent_counts.to(self.device) < 1, -1e9)
            all_seeds.append(self._top_k(score, "logit_specificity"))

        # ------------------------------------------------------------------
        # 7. Coactivation Diversity — count of distinct non-zero coact partners
        #    Complements connectivity (sum of weights) by rewarding wide neighbourhoods.
        # ------------------------------------------------------------------
        if "coactivation_diversity" in active:
            score = (top_coactivation.top_values.to(self.device) > 0).float().sum(dim=-1)
            all_seeds.append(self._top_k(score, "coactivation_diversity"))

        # ------------------------------------------------------------------
        # 8. Last-Token Activity — firing frequency at the prediction position
        #    logit_ctx.latent_counts tracks last-token aligned firing events.
        # ------------------------------------------------------------------
        if "last_token_activity" in active:
            score = logit_ctx.latent_counts.to(self.device).float()
            score = score.masked_fill(score < 5, -1e9)
            all_seeds.append(self._top_k(score, "last_token_activity"))

        # ------------------------------------------------------------------
        # 9. Top-Ctx Saturation — number of stored top-context sequences
        #    Near-full stores = richer positive evidence for discovery methods.
        # ------------------------------------------------------------------
        if "top_ctx_saturation" in active:
            score = (top_ctx.ctx_seq_idx.to(self.device) > 0).float().sum(dim=-1)
            all_seeds.append(self._top_k(score, "top_ctx_saturation"))

        # ------------------------------------------------------------------
        # 10. Pos/Neg Contrast — mean pos-ctx activation / mean neg-ctx activation
        #     Latents with clean on/off signal are the best seeds for discovery.
        # ------------------------------------------------------------------
        if "pos_neg_contrast" in active:
            pos_mean = top_ctx.ctx_seq_val.to(self.device).float().mean(dim=-1)
            neg_vals  = neg_ctx.ctx_seq_val.to(self.device).float()
            neg_count = (neg_ctx.ctx_seq_idx.to(self.device) > 0).float().sum(dim=-1)
            neg_mean  = neg_vals.sum(dim=-1) / neg_count.clamp(min=1)
            score = pos_mean / (neg_mean + 1e-6)
            score = score.masked_fill(neg_count < 4, -1e9)
            all_seeds.append(self._top_k(score, "pos_neg_contrast"))

        # ------------------------------------------------------------------
        # 11. Cross-Layer Reach  — number of distinct layers among coact partners
        # 12. Cross-Component Breadth — number of distinct kinds (attn/mlp/resid)
        #     Both require partner component decomposition; computed together.
        # ------------------------------------------------------------------
        if "cross_layer_reach" in active or "cross_component_breadth" in active:
            ci  = top_coactivation.top_indices.to(self.device)  # [C, D, K]
            cv  = top_coactivation.top_values.to(self.device)   # [C, D, K]
            C_x, D_x, K_x = ci.shape
            valid_x       = cv > 0
            partner_comp  = ci.long() // D_x            # component idx of each partner
            partner_layer = partner_comp // _N_KINDS
            partner_kind  = partner_comp % _N_KINDS
            n_layers_x    = C_x // _N_KINDS

            if "cross_layer_reach" in active:
                reach = torch.zeros(C_x, D_x, device=self.device)
                for l in range(n_layers_x):
                    reach += ((partner_layer == l) & valid_x).any(dim=-1).float()
                all_seeds.append(self._top_k(reach, "cross_layer_reach"))

            if "cross_component_breadth" in active:
                breadth = torch.zeros(C_x, D_x, device=self.device)
                for kind in range(_N_KINDS):
                    breadth += ((partner_kind == kind) & valid_x).any(dim=-1).float()
                all_seeds.append(self._top_k(breadth, "cross_component_breadth"))

        # ------------------------------------------------------------------
        # 13. Burstiness — active_count / seq_count
        #     Tokens-per-sequence when firing: ~1 = event-marker, high = pattern tracker.
        # ------------------------------------------------------------------
        if "burstiness" in active:
            score = (
                latent_stats.active_count.to(self.device).float()
                / latent_stats.seq_count.to(self.device).float().clamp(min=1)
            )
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
            all_seeds.append(self._top_k(score, "burstiness"))

        # ------------------------------------------------------------------
        # 14. Mid-Ctx Richness — saturation of the mid-band context reservoir
        #     Complements top_ctx_saturation: measures mid-activation coverage.
        # ------------------------------------------------------------------
        if "mid_ctx_richness" in active:
            score = (mid_ctx.ctx_seq_idx.to(self.device) > 0).float().sum(dim=-1)
            all_seeds.append(self._top_k(score, "mid_ctx_richness"))

        # ------------------------------------------------------------------
        # 15. Activation Skew — max(top_ctx) − mean(top_ctx)
        #     High skew = one dominant context stands far above the rest.
        # ------------------------------------------------------------------
        if "activation_skew" in active:
            ctx_vals = top_ctx.ctx_seq_val.to(self.device).float()  # [C, D, N]
            score = ctx_vals.max(dim=-1).values - ctx_vals.mean(dim=-1)
            all_seeds.append(self._top_k(score, "activation_skew"))

        # ------------------------------------------------------------------
        # 16. Logit Diversity — positive entropy of token prediction distribution
        #     High entropy = predicts many different tokens (polyfunctional / context-gating).
        #     Explicit inverse of logit_specificity for ablation.
        # ------------------------------------------------------------------
        if "logit_diversity" in active:
            p = logit_ctx.top_probs.to(self.device).clamp(min=1e-9)
            p = p / p.sum(dim=-1, keepdim=True)
            score = -(p * p.log()).sum(dim=-1)  # positive entropy; higher = more diverse
            score = score.masked_fill(logit_ctx.latent_counts.to(self.device) < 1, -1e9)
            all_seeds.append(self._top_k(score, "logit_diversity"))

        # ------------------------------------------------------------------
        # 17. PageRank Centrality — eigenvector centrality on the coact graph
        #     A latent is important if many important latents co-activate with it.
        #     Purely structural — the canonical unsupervised graph-importance signal.
        # ------------------------------------------------------------------
        if "pagerank_centrality" in active:
            ci_pr = top_coactivation.top_indices.to(self.device)   # [C, D, K]
            cv_pr = top_coactivation.top_values.to(self.device)    # [C, D, K]
            C_pr, D_pr, K_pr = ci_pr.shape
            N_pr  = C_pr * D_pr
            valid_pr = cv_pr > 0

            # Row-normalised edge weights: each source distributes its score
            # proportionally across its outgoing edges.
            row_sum  = cv_pr.reshape(N_pr, K_pr).sum(dim=-1, keepdim=True).clamp(min=1e-9)
            norm_w   = (cv_pr.reshape(N_pr, K_pr) / row_sum) * valid_pr.reshape(N_pr, K_pr).float()
            dst_flat = ci_pr.reshape(N_pr, K_pr).long().clamp(0, N_pr - 1)

            scores_pr = torch.full((N_pr,), 1.0 / N_pr, device=self.device)
            damping   = 0.85
            for _ in range(5):
                # Each source i contributes scores_pr[i] * norm_w[i,k] to destination dst[i,k]
                contrib = (scores_pr.unsqueeze(-1) * norm_w).reshape(-1)
                valid_edges = valid_pr.reshape(-1)
                new_scores  = torch.zeros(N_pr, device=self.device)
                new_scores.scatter_add_(0, dst_flat.reshape(-1)[valid_edges], contrib[valid_edges])
                scores_pr = damping * new_scores + (1.0 - damping) / N_pr
                scores_pr /= scores_pr.sum().clamp(min=1e-9)

            all_seeds.append(self._top_k(scores_pr.reshape(C_pr, D_pr), "pagerank_centrality"))

        # ------------------------------------------------------------------
        # 18. Activation Entropy — binary entropy of per-sequence firing probability
        #     Maximum at p=0.5 (fires in half of sequences) — maximally informative split.
        #     Purely information-theoretic; no task-specific inductive bias.
        # ------------------------------------------------------------------
        if "activation_entropy" in active:
            total_seqs = float(latent_stats.seq_count.max().item()) + 1.0
            p = (latent_stats.seq_count.to(self.device).float() + 1e-6) / total_seqs
            q = (1.0 - p).clamp(min=1e-9)
            p = p.clamp(min=1e-9)
            score = -(p * p.log() + q * q.log())  # binary entropy H(p)
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 1, -1e9)
            all_seeds.append(self._top_k(score, "activation_entropy"))

        # ------------------------------------------------------------------
        # 19. Coactivation Uniqueness — inverse of mean partner popularity
        #     Partners that few other latents share → unique neighbourhood.
        # ------------------------------------------------------------------
        if "coactivation_uniqueness" in active:
            ci_u  = top_coactivation.top_indices.to(self.device)  # [C, D, K]
            cv_u  = top_coactivation.top_values.to(self.device)   # [C, D, K]
            C_u, D_u, K_u = ci_u.shape
            N_u   = C_u * D_u
            valid_u = cv_u > 0

            # Count how many latents list each global ID as a partner
            popularity = torch.zeros(N_u, device=self.device)
            valid_partners = ci_u.reshape(-1)[valid_u.reshape(-1)].long().clamp(0, N_u - 1)
            popularity.scatter_add_(
                0, valid_partners, torch.ones(valid_partners.shape[0], device=self.device)
            )

            # For each latent, mean popularity of its partners
            flat_partners = ci_u.reshape(N_u, K_u).long().clamp(0, N_u - 1)
            flat_valid_u  = valid_u.reshape(N_u, K_u).float()
            mean_pop = (popularity[flat_partners] * flat_valid_u).sum(dim=-1) / flat_valid_u.sum(dim=-1).clamp(min=1)

            score = (1.0 / mean_pop.clamp(min=1.0)).reshape(C_u, D_u)
            all_seeds.append(self._top_k(score, "coactivation_uniqueness"))

        # ------------------------------------------------------------------
        # 20. Focal Monosemantic — high cohesion, low coupling
        #     context_coherence / (n_coact_partners + 1): latents that fire
        #     strongly and consistently on a focused set of contexts but are
        #     not broadly entangled with many other latents.
        # ------------------------------------------------------------------
        if "focal_monosemantic" in active:
            v = top_ctx.ctx_seq_val.to(self.device).float()              # [C, D, N]
            filled = (v > 0).float()
            cohesion = (v * filled).sum(dim=-1) / filled.sum(dim=-1).clamp(min=1.0)  # [C, D]
            n_partners = (top_coactivation.top_values.to(self.device) > 0).float().sum(dim=-1)  # [C, D]
            score = cohesion / (n_partners + 1.0)
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 5, -1e9)
            score = score.masked_fill(n_partners < config.discovery.focal_monosemantic_min_partners, -1e9)
            all_seeds.append(self._top_k(score, "focal_monosemantic"))

        # ------------------------------------------------------------------
        # 21. Rare Hub — rarity × co-activation connectivity
        #     Latents that fire rarely but are well-connected to other latents
        #     when they do. Targets low-frequency latents embedded in the
        #     co-activation graph rather than peripheral singletons.
        # ------------------------------------------------------------------
        if "rare_hub" in active:
            total_seqs_rh = float(latent_stats.seq_count.max().item()) + 1.0
            p_fire_rh = (latent_stats.seq_count.to(self.device).float() + 1e-6) / total_seqs_rh
            rarity_rh = torch.log10(1.0 / p_fire_rh).clamp(1.0, 8.0)
            conn_rh = top_coactivation.top_values.to(self.device)
            if top_coactivation.mode == "pmi":
                conn_rh = conn_rh.clamp(min=0)
            conn_sum_rh = conn_rh.sum(dim=-1)
            score = rarity_rh * conn_sum_rh
            score = score.masked_fill(latent_stats.seq_count.to(self.device) < 2, -1e9)
            score = score.masked_fill(conn_sum_rh == 0, -1e9)
            all_seeds.append(self._top_k(score, "rare_hub"))

        # ------------------------------------------------------------------
        # 22. Stratified Random — uniform sampling across all components
        #     True unsupervised baseline: no signal whatsoever.
        # ------------------------------------------------------------------
        if "stratified_random" in active:
            C_sr = latent_stats.active_count.shape[0]
            D_sr = latent_stats.active_count.shape[1]
            quota = max(1, self.n_seeds // C_sr)
            sr_seeds: List[Dict[str, Any]] = []
            for comp_idx in range(C_sr):
                perm = torch.randperm(D_sr, device=self.device)[:quota]
                for latent_idx in perm.tolist():
                    sr_seeds.append({
                        "comp_idx":   comp_idx,
                        "latent_idx": int(latent_idx),
                        "score":      1.0,
                        "reason":     "stratified_random",
                    })
            all_seeds.append(sr_seeds)

        # ------------------------------------------------------------------
        # 23. Circuit Yield — empirical score from previous discovery runs
        #     Latents that were productive seeds before (high faithfulness circuits)
        #     are prioritised. Gracefully skipped if no summary.json exists.
        # ------------------------------------------------------------------
        if "circuit_yield" in active:
            summary_path = os.path.join("outputs", "circuits", "summary.json")
            if not os.path.exists(summary_path):
                print("[CandidateSelector] circuit_yield: no summary.json found, skipping.")
            else:
                try:
                    with open(summary_path) as f:
                        circuits_data = json.load(f)
                    C_cy = latent_stats.active_count.shape[0]
                    D_cy = latent_stats.active_count.shape[1]
                    yield_scores = torch.zeros(C_cy, D_cy, device=self.device)
                    for circuit in circuits_data:
                        meta  = circuit.get("metadata", {})
                        sc    = meta.get("seed_comp")
                        sl    = meta.get("seed_latent")
                        if sc is None or sl is None:
                            continue
                        evals = meta.get("evals", {})
                        faith = (
                            evals.get("counterfactual_faithfulness")
                            or evals.get("faithfulness")
                            or evals.get("sufficiency")
                            or 0.0
                        )
                        if faith > 0 and 0 <= sc < C_cy and 0 <= sl < D_cy:
                            yield_scores[sc, sl] += float(faith)
                    all_seeds.append(self._top_k(yield_scores, "circuit_yield"))
                except Exception as e:
                    print(f"[CandidateSelector] circuit_yield: failed — {e}")

        # ------------------------------------------------------------------
        if not all_seeds:
            valid_opts = (
                "logit_impact, connectivity, surprise, context_coherence, "
                "activation_variance, logit_specificity, coactivation_diversity, "
                "last_token_activity, top_ctx_saturation, pos_neg_contrast, "
                "cross_layer_reach, cross_component_breadth, burstiness, "
                "mid_ctx_richness, activation_skew, logit_diversity, "
                "pagerank_centrality, activation_entropy, coactivation_uniqueness, "
                "focal_monosemantic, rare_hub, "
                "stratified_random, circuit_yield"
            )
            raise ValueError(
                f"[CandidateSelector] No valid seed criteria enabled. "
                f"config.discovery.seed_criteria={list(active)}. "
                f"Valid options: {valid_opts}."
            )

        # Merge and deduplicate — accumulate scores across criteria
        merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for seeds in all_seeds:
            for seed in seeds:
                key = (seed["comp_idx"], seed["latent_idx"])
                criterion = seed["reason"]
                raw_score = seed["score"]
                if key not in merged:
                    merged[key] = {
                        "comp_idx":       seed["comp_idx"],
                        "latent_idx":     seed["latent_idx"],
                        "score":          raw_score,
                        "reason":         criterion,
                        "criteria_scores": {criterion: raw_score},
                    }
                else:
                    merged[key]["score"] += raw_score
                    merged[key]["reason"] += f", {criterion}"
                    merged[key]["criteria_scores"][criterion] = raw_score

        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        if filter_layers or filter_kinds:
            before = len(results)
            results = [c for c in results if _seed_passes_filter(c, filter_layers, filter_kinds)]
            print(
                f"[CandidateSelector] After seed_filter: {len(results)} of {before} candidates remain."
            )

        print(f"[CandidateSelector] Selected {len(results)} unique seed latents. Returning top {self.n_seeds}.")
        return results[: self.n_seeds]

    def _top_k(self, score_tensor: torch.Tensor, reason: str) -> List[Dict[str, Any]]:
        """Return top-n_seeds (comp_idx, latent_idx) entries from a [C, D] score tensor.

        Raw scores are normalised to [0, 1] by dividing by the maximum value before
        being stored.  This ensures every criterion contributes equally to the combined
        ranking during the merge step, regardless of its raw scale.
        """
        C, D = score_tensor.shape
        flat  = score_tensor.view(-1)
        k     = min(self.n_seeds, flat.numel())
        top_vals, top_idx = torch.topk(flat, k)

        # Normalise to [0, 1] so criteria with large raw values (e.g. connectivity)
        # don't dominate the combined score
        max_val = top_vals[0].item() if top_vals.numel() > 0 else 1.0
        if max_val > 0:
            top_vals = top_vals / max_val

        results = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val <= 0:
                continue
            results.append({
                "comp_idx":   idx // D,
                "latent_idx": idx % D,
                "score":      float(val),
                "reason":     reason,
            })
        return results

    # Kept for backwards compatibility with external callers
    def _top_k_indices(self, score_tensor: torch.Tensor, k: int, reason: str) -> List[Dict[str, Any]]:
        return self._top_k(score_tensor, reason)

    def get_summary_stats(self, candidates: List[Dict[str, Any]]) -> None:
        """Prints a breakdown of selected candidates by layer and kind."""
        from model.turingllm import TuringLLMConfig
        n_layers = TuringLLMConfig().n_layer
        layer_counts = [0] * n_layers
        kind_counts  = {k: 0 for k in _KINDS}
        for c in candidates:
            layer_idx, kind_idx = split_component_idx(c["comp_idx"], _N_KINDS)
            layer_counts[layer_idx] += 1
            kind_counts[_KINDS[kind_idx]] += 1

        sf = config.discovery.seed_filter
        filter_layers: set = set(sf.layers)
        filter_kinds: set = set(sf.kinds)

        print("\nCandidate Summary:")
        if filter_layers or filter_kinds:
            print(
                f"  Filter:   layers={sorted(filter_layers) if filter_layers else 'all'}, "
                f"kinds={sorted(filter_kinds) if filter_kinds else 'all'}"
            )
        print(f"  By Kind:  {kind_counts}")
        print(f"  By Layer: {layer_counts}")
