"""Direct-effect edge attribution between selected circuit nodes.

Implements the edge-weight construction of Sparse Feature Circuits (Marks
et al., 2025): the weight of an edge u -> d is the indirect effect of u on
the metric via its DIRECT effect on d, excluding paths mediated by any other
feature node. (SFC's term is "edge weights"; we call these "direct-effect
edges" in prose. Cite the paper plainly, not by appendix letter.)

    w(u -> d) = grad_d(m) * grad_{u, stop(M)}(d) * (u_natural - u_baseline)

SAEGraphInstrument provides the stop(M) semantics natively: feature
contributions enter each site through detached leaf anchors (gradient
terminals) while raw model ops flow through identity passthroughs, so a
backward from a downstream site's CONNECTED code reaches upstream anchors
only along feature-free paths.

Cost per probe batch: one instrumented forward, one backward from the metric
(node gradient fields, graph retained), then per downstream site chunked
vector-Jacobian products - batched cotangents via is_grads_batched when
``batched`` is set, with a sequential fallback for kernels that reject vmap.
Edges never change circuit membership or eval scores; they add structure
(node_depth_*, n_internal_edges, DAG rendering, composition routing).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.types.feature_id import FeatureID

Site = Tuple[int, str]


def attach_direct_edges(
    circuit: Any,
    inference: Any,
    bank: Any,
    *,
    pos_tokens: torch.Tensor,
    pos_argmax: torch.Tensor,
    seed_layer: int,
    seed_kind: str,
    seed_latent_idx: int,
    site_baselines: Optional[Dict[Site, torch.Tensor]] = None,
    top_k_edges_per_node: int = 8,
    chunk_size: int = 32,
    batched: bool = True,
    min_abs_weight: float = 0.0,
) -> Dict[str, Any]:
    """Compute direct-effect edges among the circuit's nodes and attach them.

    site_baselines: per-site [d_sae] baseline (mean-ablation floor); zeros
    when omitted. Returns summary stats {n_edges_added, n_downstream_nodes}.
    """

    nodes_by_site: Dict[Site, List[Tuple[int, str]]] = {}
    uuid_by_fid: Dict[Tuple[int, str, int], str] = {}
    for uuid, node in circuit.nodes.items():
        fid = node.feature_id
        if fid is None or node.metadata.get("role") == "seed":
            continue
        nodes_by_site.setdefault((fid.layer, fid.kind), []).append((fid.index, uuid))
        uuid_by_fid[(fid.layer, fid.kind, fid.index)] = uuid
    if not nodes_by_site:
        return {"n_edges_added": 0, "n_downstream_nodes": 0}

    sae = bank.saes[seed_kind][seed_layer]
    w_seed = sae.encoder.weight[seed_latent_idx].detach()
    b_seed = sae._get_bias_eff()[seed_latent_idx].detach()

    instrument = SAEGraphInstrument(bank)
    seed_pre_act: List[torch.Tensor] = []
    original_transform = instrument.transform

    def transform_with_seed_tap(layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx == seed_layer and kind == seed_kind:
            w = w_seed.to(device=x.device, dtype=x.dtype)
            b = b_seed.to(device=x.device, dtype=x.dtype)
            seed_pre_act.append(x @ w + b)
            return x
        return original_transform(layer_idx, kind, x)

    instrument.transform = transform_with_seed_tap  # type: ignore[method-assign]

    inference.disable_compile()
    try:
        inference.forward(
            pos_tokens,
            patcher=instrument,
            grad_enabled=True,
            return_activations=False,
            tokenize_final=False,
        )
        if not seed_pre_act:
            raise RuntimeError("seed pre-activation was not captured")
        pre = seed_pre_act[0]
        B = min(pre.shape[0], pos_argmax.shape[0])
        idx = torch.arange(B, device=pre.device)
        pa = pos_argmax[:B].to(pre.device).clamp(0, pre.shape[1] - 1)
        metric = pre[:B][idx, pa].mean()

        graph = instrument.graph
        anchor_sites = sorted(
            site for site in graph.activations if site in nodes_by_site or True
        )
        anchors = {site: graph.get_latents(*site)[0].act for site in anchor_sites}
        anchor_list = [anchors[site] for site in anchor_sites]

        # Backward 0: node gradient fields (the grad_d(m) factor), graph kept.
        node_grads = torch.autograd.grad(
            metric, anchor_list, retain_graph=True, allow_unused=True
        )
        grad_by_site = {
            site: grad for site, grad in zip(anchor_sites, node_grads) if grad is not None
        }
        print(
            f"  [DirectEdges] anchor sites: {len(anchor_sites)} | with metric-grad: "
            f"{len(grad_by_site)} | member sites: {len(nodes_by_site)}"
        )
        sys.stdout.flush()

        kinds = list(bank.kinds)

        def site_order(site: Site) -> int:
            return site[0] * len(kinds) + kinds.index(site[1])

        n_edges = 0
        n_downstream = 0
        for down_site, members in sorted(nodes_by_site.items(), key=lambda kv: site_order(kv[0])):
            if down_site not in grad_by_site:
                continue
            upstream_sites = [
                site for site in anchor_sites if site_order(site) < site_order(down_site)
            ]
            if not upstream_sites:
                continue
            connected = graph.get_latents(*down_site)[1].act  # [B, T, d_sae]
            grad_field = grad_by_site[down_site]

            for chunk_start in range(0, len(members), chunk_size):
                chunk = members[chunk_start : chunk_start + chunk_size]
                cotangents = torch.zeros(
                    (len(chunk),) + tuple(connected.shape),
                    device=connected.device,
                    dtype=connected.dtype,
                )
                for k, (latent_idx, _uuid) in enumerate(chunk):
                    cotangents[k, :, :, latent_idx] = grad_field[:, :, latent_idx]

                upstream_anchors = [anchors[site] for site in upstream_sites]
                if batched:
                    grads = torch.autograd.grad(
                        connected,
                        upstream_anchors,
                        grad_outputs=cotangents,
                        retain_graph=True,
                        allow_unused=True,
                        is_grads_batched=True,
                    )
                else:
                    per_k = [
                        torch.autograd.grad(
                            connected,
                            upstream_anchors,
                            grad_outputs=cotangents[k],
                            retain_graph=True,
                            allow_unused=True,
                        )
                        for k in range(len(chunk))
                    ]
                    grads = tuple(
                        torch.stack([g[i] for g in per_k])
                        if per_k and per_k[0][i] is not None
                        else None
                        for i in range(len(upstream_anchors))
                    )

                n_downstream += len(chunk)
                for site, grad in zip(upstream_sites, grads):
                    if grad is None:
                        continue
                    natural = graph.get_latents(*site)[0].act.detach()
                    baseline = (
                        site_baselines[site].to(natural.device, natural.dtype)
                        if site_baselines is not None and site in site_baselines
                        else torch.zeros(natural.shape[-1], device=natural.device, dtype=natural.dtype)
                    )
                    delta = natural - baseline  # [B, T, d_sae]
                    weights = (grad.to(torch.float32) * delta.to(torch.float32)).sum(dim=(1, 2))
                    # weights: [K, d_sae] -> per downstream node, per upstream latent
                    for k, (_latent_idx, down_uuid) in enumerate(chunk):
                        row = weights[k]
                        k_top = min(top_k_edges_per_node, row.shape[0])
                        top_vals, top_idx = row.abs().topk(k_top)
                        for value, u_idx in zip(top_vals.tolist(), top_idx.tolist()):
                            if value <= min_abs_weight:
                                continue
                            key = (site[0], site[1], int(u_idx))
                            up_uuid = uuid_by_fid.get(key)
                            if up_uuid is None or up_uuid == down_uuid:
                                continue
                            circuit.add_edge(
                                up_uuid,
                                down_uuid,
                                weight=float(row[u_idx].item()),
                                kind="direct_effect",
                            )
                            n_edges += 1
    finally:
        inference.enable_compile()

    circuit.metadata["edge_attribution"] = "direct_effect"
    print(f"  [DirectEdges] downstream nodes: {n_downstream} | edges added: {n_edges}")
    sys.stdout.flush()
    return {"n_edges_added": n_edges, "n_downstream_nodes": n_downstream}


__all__ = ["attach_direct_edges"]
