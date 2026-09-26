"""Render the ablation-consensus wiring of WCM circuits (057 edge matrices) as layered DAGs.

Only CONSENSUS structure is drawn: an edge (or an edge into the target) must carry at least THETA of the circuit's
attribution mass sum|A_u| under every ablation method (Z, A, C) with one sign, and it is drawn at its WEAKEST
normalised weight across the three (what every method agrees on). Nodes are ranked by their weakest normalised
attribution min_pi |A_u| / sum|A|. Readability selection is a BACKBONE grown from the target: repeatedly add the strongest
consensus edge into a node already drawn (at most MAX_IN per node, MAX_NODES nodes, TOP_EDGES edges), so every
drawn node has a consensus path to the target.

Layout and edge convention follow src/analysis/circuits/circuit_dag.py (SFC-style: one row per site, target on
top, blue = positive, red = negative, width ~ |weight|). Colours per the figure-style memory: target honey,
circuit latents blue. Each panel title states how much of the edge mass the consensus set carries, so the drawing
is never read as the whole mechanism.

  PYTHONPATH=src python experiments/057-wcm-edges/render.py
      -> figures/wiring_<target>.png/.pdf and figures/wiring_consensus.png/.pdf (2 x 2)
  env: ARM (rkeep3e3fix)  TARGETS (comma list)  THETA (1e-3)  TOP_EDGES (28)  MAX_IN (4)
"""
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.circuits.circuit_dag import KIND_ABBR, layout_rows
from analysis.style import (BLUE, CATEGORICAL, INK, INK_MUTED, configure_matplotlib, save_figure, style_suptitle,
                            tint)

HERE = Path(__file__).parent
ARM = os.environ.get("ARM", "rkeep3e3fix")
TARGETS = os.environ.get("TARGETS", "1.resid.12137,6.resid.18234,2.attn.33479,11.mlp.30743").split(",")
THETA = float(os.environ.get("THETA", 1e-3))
TOP_EDGES = int(os.environ.get("TOP_EDGES", 28))
MAX_IN = int(os.environ.get("MAX_IN", 4))
MAX_NODES = int(os.environ.get("MAX_NODES", 22))
GAP = 1.25                                                                              # horizontal node spacing
KEY = ("blue edge = positive, red = negative; width = the weakest normalised weight across Z/A/C; "
       "node shade = the weakest attribution across Z/A/C; α = scaling coefficient")
PIS = ("Z", "A", "C")
RED = CATEGORICAL[1]
HONEY = CATEGORICAL[6]
TGT = "target"


def consensus(d):
    """Consensus member edges, edges into the target and node strengths, all normalised by sum|A| per method."""
    En = torch.stack([d[p]["E"].double() / d[p]["A"].double().abs().sum() for p in PIS])        # [3, N, N]
    Etn = torch.stack([d[p]["Et"].double() / d[p]["A"].double().abs().sum() for p in PIS])     # [3, N]
    An = torch.stack([d[p]["A"].double() / d[p]["A"].double().abs().sum() for p in PIS])       # [3, N]

    def agree(X):
        same = (torch.sign(X) == torch.sign(X[0:1])).all(0) & (X[0] != 0)
        return same & (X.abs() >= THETA).all(0), torch.sign(X[0]) * X.abs().min(0).values

    mask, w = agree(En)
    mask_t, wt = agree(Etn)
    # share of each method's member-edge mass carried by the consensus set, averaged over methods
    share = float(torch.stack([(En[i].abs() * mask).sum() / En[i].abs().sum() for i in range(3)]).mean())
    return mask, w, mask_t, wt, An.abs().min(0).values, share


def build(d, key):
    nodes = d["nodes"]
    mask, w, mask_t, wt, strength, share = consensus(d)
    # incoming consensus edges per downstream node
    into = {TGT: [(float(wt[j]), "n%d" % j) for j in mask_t.nonzero().flatten().tolist()]}
    for i, j in mask.nonzero().tolist():
        into.setdefault("n%d" % i, []).append((float(w[i, j]), "n%d" % j))
    # backbone: grow from the target, always adding the strongest consensus edge INTO a node already drawn, so
    # every drawn node has a consensus path to the target
    import heapq
    heap, in_graph, per_in, edges = [], {TGT}, {}, []

    def push(v):
        for s, u in into.get(v, []):
            heapq.heappush(heap, (-abs(s), s, u, v))

    push(TGT)
    while heap and len(edges) < TOP_EDGES:
        _, s, u, v = heapq.heappop(heap)
        if per_in.get(v, 0) >= MAX_IN or (len(in_graph) >= MAX_NODES and u not in in_graph):
            continue
        per_in[v] = per_in.get(v, 0) + 1
        edges.append(SimpleNamespace(source_uuid=u, target_uuid=v, weight=s))
        if u not in in_graph:
            in_graph.add(u); push(u)
    layer, kind, idx = key.split(".")
    kept = {TGT: SimpleNamespace(feature_id=SimpleNamespace(layer=int(layer), kind=kind, index=int(idx)),
                                 alpha=None, strength=None)}
    for e in edges:
        for u in (e.source_uuid, e.target_uuid):
            if u != TGT and u not in kept:
                j = int(u[1:]); L, K, I, al = nodes[j]
                kept[u] = SimpleNamespace(feature_id=SimpleNamespace(layer=int(L), kind=K, index=int(I)),
                                          alpha=float(al), strength=float(strength[j]))
    stats = dict(n_nodes=len(nodes), n_cons=int(mask.sum()), n_cons_t=int(mask_t.sum()), share=share,
                 n_drawn=len(edges), n_drawn_nodes=len(kept) - 1)
    return kept, edges, stats


def draw(ax, kept, edges, stats, key):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    positions, legend = layout_rows(kept, edges)
    positions = {u: (x * GAP, y) for u, (x, y) in positions.items()}                  # room for 5-digit labels
    n_rows = max(r for _, r in positions.values()) + 1
    width = max(x for x, _ in positions.values()) + 1
    ax.set_xlim(-2.2, width + 0.6); ax.set_ylim(-0.7, n_rows - 0.3); ax.axis("off")
    w_max = max((abs(e.weight) for e in edges), default=1.0)
    for e in sorted(edges, key=lambda e: abs(e.weight)):
        (x0, y0), (x1, y1) = positions[e.source_uuid], positions[e.target_uuid]
        st = abs(e.weight) / w_max
        col = BLUE if e.weight >= 0 else RED
        ax.add_patch(FancyArrowPatch((x0, y0 + 0.17), (x1, y1 - 0.21), arrowstyle="-|>", mutation_scale=7,
                                     connectionstyle="arc3,rad=0.12", linewidth=0.6 + 2.6 * st,
                                     color=tint(col, 0.55 * (1 - st)), alpha=0.45 + 0.55 * st, zorder=1))
    s_max = max((n.strength for n in kept.values() if n.strength is not None), default=1.0)
    for u, n in kept.items():
        x, y = positions[u]; fid = n.feature_id
        if u == TGT:
            face, edge_c, txt, lw = HONEY, HONEY, INK, 1.4
        else:
            st = n.strength / s_max
            face, edge_c, txt, lw = tint(BLUE, 0.9 - 0.55 * st), tint(BLUE, 0.25), INK, 0.9
        ax.add_patch(FancyBboxPatch((x - 0.41, y - 0.16), 0.82, 0.32, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=face, edgecolor=edge_c, linewidth=lw, zorder=3))
        ax.text(x, y, "%s%d/%d" % (KIND_ABBR[fid.kind], fid.layer, fid.index), ha="center", va="center",
                fontsize=8.3, color=txt, zorder=4, fontweight="bold" if u == TGT else "normal")
        if n.alpha is not None:
            ax.text(x, y - 0.21, "α %.2g" % n.alpha, ha="center", va="top", fontsize=6.3, color=INK_MUTED,
                    zorder=4)
    for row, layer, kind in legend:
        ax.text(-2.1, row, "L%d %s" % (layer, kind), ha="left", va="center", fontsize=9.0, color=INK_MUTED)
    ax.set_title("%s   (%d-node circuit)\nconsensus edges: %d between members + %d into the target, %.0f%% of edge "
                 "mass; %d drawn" % (key, stats["n_nodes"], stats["n_cons"], stats["n_cons_t"], 100 * stats["share"],
                                     stats["n_drawn"]), loc="left", fontsize=10.5)


def main():
    plt = configure_matplotlib()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    out = HERE / "figures"
    built = []
    for key in TARGETS:
        d = torch.load(HERE / "data" / ARM / ("%s.pt" % key), weights_only=False)
        kept, edges, stats = build(d, key)
        built.append((key, kept, edges, stats))
        fig, ax = plt.subplots(figsize=(11.0, 7.5))
        draw(ax, kept, edges, stats, key)
        fig.text(0.01, 0.005, KEY, fontsize=8.5, color=INK_MUTED)
        save_figure(fig, out / ("wiring_%s.png" % key))
        print("%-15s nodes %d  consensus %d + %d to target  share %.3f  drawn %d edges / %d nodes"
              % (key, stats["n_nodes"], stats["n_cons"], stats["n_cons_t"], stats["share"], stats["n_drawn"],
                 stats["n_drawn_nodes"]))
    fig, axes = plt.subplots(2, 2, figsize=(20.0, 15.0))
    for ax, (key, kept, edges, stats) in zip(axes.flat, built):
        draw(ax, kept, edges, stats, key)
    style_suptitle(fig, "Weighted circuits, wired: edges shared by all three ablation methods (%s)" % ARM)
    fig.text(0.01, 0.955, KEY, fontsize=11, color=INK_MUTED)
    save_figure(fig, out / "wiring_consensus.png")


if __name__ == "__main__":
    main()
