"""Case-study circuit DAG figure: render a wired Circuit as a layered
node-link diagram in the house style.

Layout follows the Sparse Feature Circuits convention (Marks et al. 2025,
circuit_plotting.py): one row per (layer, kind) site ascending so the seed
sits at the top, signed edges coloured by direction of effect with width
proportional to |direct-effect weight|. Only ``kind="direct_effect"`` edges
(SFC App. B) are drawn — discovery's member->seed star edges are attribution
bookkeeping, not structure.

Input: a dict of wired Circuits serialized by the case-study wiring script
(discovery with edges attached in-memory via attach_direct_edges), e.g.
analysis-restyled/case-study/wired-circuits.pt.

Node selection keeps the figure readable: the top-N members by |polished
attribution score| plus the seed, then the strongest incoming edges per node,
then isolated members are dropped.

Run: PYTHONPATH=src python -m analysis.circuits.circuit_dag \
    --input analysis-restyled/case-study/wired-circuits.pt \
    --key ablation_gradient__ig_restoration__comp11_lat35381 \
    --out analysis-restyled/case-study/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from analysis.style import BLUE, INK, INK_MUTED, configure_matplotlib, save_figure, tint

RED = "#fa1e4e"
KIND_ORDER = {"attn": 0, "mlp": 1, "resid": 2}
KIND_ABBR = {"attn": "A", "mlp": "M", "resid": "R"}


def select_subgraph(
    circuit: Any,
    *,
    top_n: int = 24,
    max_in_edges: int = 4,
    edge_weight_floor: float = 0.03,
    drop_isolated: bool = True,
) -> Tuple[Dict[str, Any], List[Any]]:
    """Pick the nodes and direct-effect edges to draw.

    Members are ranked by |attribution_score| (the polished, mutually
    comparable scores when final_ig_polish ran). Edges keep at most
    ``max_in_edges`` strongest per target and must carry at least
    ``edge_weight_floor`` of the subgraph's maximum |weight|. Isolated
    members (no drawn edge in either direction) are dropped by default.
    """

    seed = next(
        (n for n in circuit.nodes.values() if n.metadata.get("role") == "seed"), None
    )
    if seed is None:
        raise ValueError("circuit has no seed node")

    def score(node: Any) -> float:
        value = node.metadata.get("attribution_score")
        return abs(float(value)) if value is not None else 0.0

    members = [
        n for n in circuit.nodes.values()
        if n.metadata.get("role") != "seed" and n.feature_id is not None
    ]
    members.sort(key=score, reverse=True)
    kept: Dict[str, Any] = {seed.uuid: seed}
    for node in members[:top_n]:
        kept[node.uuid] = node

    direct = [
        e for e in circuit.edges
        if e.metadata.get("kind") == "direct_effect"
        and e.source_uuid in kept and e.target_uuid in kept
        and e.weight is not None
    ]
    by_target: Dict[str, List[Any]] = {}
    for edge in direct:
        by_target.setdefault(edge.target_uuid, []).append(edge)
    edges: List[Any] = []
    for target_edges in by_target.values():
        target_edges.sort(key=lambda e: abs(float(e.weight)), reverse=True)
        edges.extend(target_edges[:max_in_edges])
    if edges:
        w_max = max(abs(float(e.weight)) for e in edges)
        edges = [e for e in edges if abs(float(e.weight)) >= edge_weight_floor * w_max]

    # attach_direct_edges never targets the seed (direct effects are computed
    # between member sites only), so the seed's incoming edges are discovery's
    # total-effect attribution (star) edges — a different quantity, kept under
    # the same top-k rule and drawn dashed to mark the distinction.
    star_into_seed = [
        e for e in circuit.edges
        if e.target_uuid == seed.uuid and e.source_uuid in kept
        and e.metadata.get("kind") != "direct_effect" and e.weight is not None
    ]
    star_into_seed.sort(key=lambda e: abs(float(e.weight)), reverse=True)
    edges.extend(star_into_seed[:max_in_edges])

    if drop_isolated:
        connected = {seed.uuid}
        for edge in edges:
            connected.add(edge.source_uuid)
            connected.add(edge.target_uuid)
        kept = {uuid: node for uuid, node in kept.items() if uuid in connected}
        edges = [e for e in edges if e.source_uuid in kept and e.target_uuid in kept]
    return kept, edges


def _site_row(node: Any) -> Tuple[int, int]:
    fid = node.feature_id
    return (fid.layer, KIND_ORDER[fid.kind])


def layout_rows(
    kept: Mapping[str, Any], edges: Sequence[Any], *, sweeps: int = 4
) -> Tuple[Dict[str, Tuple[float, int]], List[Tuple[int, int, str]]]:
    """Assign each node (x, row) with rows = occupied (layer, kind) sites
    ascending (seed lands on the top row) and within-row order refined by
    barycenter sweeps to reduce crossings. Returns positions keyed by uuid
    and the row legend [(row_index, layer, kind)]."""

    sites = sorted({_site_row(n) for n in kept.values()})
    row_of_site = {site: i for i, site in enumerate(sites)}
    rows: Dict[int, List[str]] = {}
    for uuid, node in kept.items():
        rows.setdefault(row_of_site[_site_row(node)], []).append(uuid)
    for members in rows.values():
        members.sort(key=lambda u: kept[u].feature_id.index)

    neighbours: Dict[str, List[str]] = {}
    for edge in edges:
        neighbours.setdefault(edge.source_uuid, []).append(edge.target_uuid)
        neighbours.setdefault(edge.target_uuid, []).append(edge.source_uuid)

    x_of: Dict[str, float] = {}
    for members in rows.values():
        for i, uuid in enumerate(members):
            x_of[uuid] = float(i)
    for _ in range(sweeps):
        for members in rows.values():
            keyed = []
            for uuid in members:
                near = [x_of[v] for v in neighbours.get(uuid, []) if v in x_of]
                keyed.append((sum(near) / len(near) if near else x_of[uuid], uuid))
            keyed.sort()
            for i, (_, uuid) in enumerate(keyed):
                x_of[uuid] = float(i)
            rows_sorted = [uuid for _, uuid in keyed]
            members[:] = rows_sorted

    positions: Dict[str, Tuple[float, int]] = {}
    max_width = max(len(m) for m in rows.values())
    for row_index, members in rows.items():
        # Centre each row within the widest row's span.
        offset = (max_width - len(members)) / 2.0
        for i, uuid in enumerate(members):
            positions[uuid] = (offset + i, row_index)
    legend = [(i, site[0], [k for k, v in KIND_ORDER.items() if v == site[1]][0])
              for site, i in row_of_site.items()]
    return positions, sorted(legend)


_TEXT_WIDTH_CACHE: Dict[Tuple[str, float, bool], float] = {}


def _text_width(text: str, fontsize: float, bold: bool) -> float:
    """Rendered width of `text` in points (TextPath metrics, cached)."""

    key = (text, fontsize, bold)
    cached = _TEXT_WIDTH_CACHE.get(key)
    if cached is not None:
        return cached
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    if not text.strip():
        # Path extents ignore whitespace; approximate a space width.
        width = 0.28 * fontsize * len(text)
    else:
        prop = FontProperties(weight="bold" if bold else "normal")
        width = float(TextPath((0, 0), text, size=fontsize, prop=prop).get_extents().width)
    _TEXT_WIDTH_CACHE[key] = width
    return width


Word = List[Tuple[str, bool]]  # sub-fragments of one unbreakable word


def _fragments_to_words(fragments: Sequence[Tuple[str, bool]]) -> List[Word]:
    """Regroup token fragments into unbreakable words (split on spaces,
    preserving bold flags — a word may mix weights, e.g. temper**ature**)."""

    words: List[Word] = []
    current: Word = []
    for text, bold in fragments:
        parts = text.split(" ")
        for i, part in enumerate(parts):
            if i > 0:  # a space preceded this part
                if current:
                    words.append(current)
                current = []
            if part:
                current.append((part, bool(bold)))
    if current:
        words.append(current)
    return words


def _word_width(word: Word, fontsize: float) -> float:
    return sum(_text_width(text, fontsize, bold) for text, bold in word)


def _wrap_words(words: Sequence[Word], col_width: float, fontsize: float) -> List[List[Word]]:
    """Greedy wrap of words into lines no wider than col_width points."""

    space = _text_width(" ", fontsize, False)
    lines: List[List[Word]] = [[]]
    used = 0.0
    for word in words:
        width = _word_width(word, fontsize)
        needed = width if not lines[-1] else space + width
        if used + needed > col_width and lines[-1]:
            lines.append([word])
            used = width
        else:
            lines[-1].append(word)
            used += needed
    return [line for line in lines if line]


def _context_panel_box(panel: Mapping[str, Any], *, heading: str, col_width: float,
                       fontsize: float, max_neg: int = 3):
    """Build an offsetbox VPacker for one context panel (top + neg entries).

    Lines are JUSTIFIED to col_width points: inter-word spacing is computed
    per line from measured text widths (last line of each entry stays
    left-aligned, standard justification behaviour)."""

    from matplotlib.offsetbox import HPacker, TextArea, VPacker

    def word_box(word: Word, colour: str):
        areas = [
            TextArea(text, textprops=dict(
                fontsize=fontsize, color=colour,
                fontweight="bold" if bold else "normal",
            ))
            for text, bold in word if text
        ]
        return HPacker(children=areas, align="baseline", pad=0, sep=0)

    def entry_rows(words: List[Word], colour: str, marker: str):
        out = []
        marker_word: Word = [(marker, False)]
        marker_width = _word_width(marker_word, fontsize)
        space = _text_width(" ", fontsize, False)
        body_width = col_width - marker_width - space
        lines = _wrap_words(words, body_width, fontsize)
        for i, line in enumerate(lines):
            children = [word_box(marker_word if i == 0 else [("  ", False)], colour)]
            children += [word_box(w, colour) for w in line]
            is_last = i == len(lines) - 1
            if is_last or len(line) < 2:
                sep = space
            else:
                text_w = sum(_word_width(w, fontsize) for w in line)
                sep = max(space * 0.6, (body_width - text_w) / (len(line) - 1))
                sep = min(sep, space * 2.6)  # cap: sparse last-ish lines stay sane
            out.append(HPacker(children=children, align="baseline", pad=0, sep=sep))
        return out

    def spacer():
        return TextArea(" ", textprops=dict(fontsize=fontsize * 0.45))

    heading_areas = [TextArea(heading, textprops=dict(
        fontsize=fontsize + 2.4, color=INK, fontweight="bold"))]
    role = panel.get("role")
    if role:
        colour = RED if role == "inhibitor" else BLUE
        heading_areas.append(TextArea(f"  {role}", textprops=dict(
            fontsize=fontsize + 1.2, color=colour, style="italic")))
    rows = [HPacker(children=heading_areas, align="baseline", pad=0, sep=0)]
    for entry in panel.get("top", []):
        words = _fragments_to_words([tuple(f) for f in entry])
        rows.extend(entry_rows(words, INK, "▸"))
        rows.append(spacer())
    for text in list(panel.get("neg", []))[:max_neg]:
        words = _fragments_to_words([(text, False)])
        rows.extend(entry_rows(words, INK_MUTED, "✕"))
        rows.append(spacer())
    if len(rows) > 1:
        rows.pop()  # no trailing spacer
    return VPacker(children=rows, align="left", pad=0, sep=2.4)


def _draw_context_column(ax, panels: Sequence[Tuple[str, Mapping[str, Any]]], *,
                         col_width: float, fontsize: float, max_neg: int = 3) -> None:
    """Stack context panels down a text column axis (col_width in points)."""

    from matplotlib.offsetbox import AnchoredOffsetbox, VPacker

    boxes = [
        _context_panel_box(panel, heading=heading, col_width=col_width,
                           fontsize=fontsize, max_neg=max_neg)
        for heading, panel in panels
    ]
    stack = VPacker(children=boxes, align="left", pad=0, sep=11)
    anchored = AnchoredOffsetbox(loc="upper left", child=stack, frameon=False,
                                 pad=0, borderpad=0.2)
    ax.add_artist(anchored)
    ax.axis("off")


def _draw_dag(ax, circuit: Any, *, top_n: int, max_in_edges: int,
              edge_weight_floor: float, annotations: Optional[Mapping[str, str]],
              plt) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    kept, edges = select_subgraph(
        circuit, top_n=top_n, max_in_edges=max_in_edges, edge_weight_floor=edge_weight_floor
    )
    positions, legend = layout_rows(kept, edges)

    n_rows = max(r for _, r in positions.values()) + 1
    width = max(x for x, _ in positions.values()) + 1
    ax.set_xlim(-1.7, width + 0.7)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.axis("off")

    direct_edges = [e for e in edges if e.metadata.get("kind") == "direct_effect"]
    star_edges = [e for e in edges if e.metadata.get("kind") != "direct_effect"]
    w_max = max((abs(float(e.weight)) for e in direct_edges), default=1.0)
    star_max = max((abs(float(e.weight)) for e in star_edges), default=1.0)
    for edge in edges:
        x0, y0 = positions[edge.source_uuid]
        x1, y1 = positions[edge.target_uuid]
        weight = float(edge.weight)
        colour = BLUE if weight >= 0 else RED
        is_star = edge.metadata.get("kind") != "direct_effect"
        # Star (total-effect) weights live on a different scale — normalise
        # within their own family so widths stay comparable per family. They
        # are context rather than structure, so they arc wide, stay thin and
        # muted, and are dashed.
        strength = abs(weight) / (star_max if is_star else w_max)
        if is_star:
            style = dict(
                connectionstyle="arc3,rad=0.35",
                linewidth=0.5 + 0.9 * strength,
                color=tint(colour, 0.45 + 0.3 * (1 - strength)),
                alpha=0.65,
                linestyle=(0, (4, 2)),
            )
        else:
            style = dict(
                connectionstyle="arc3,rad=0.12",
                linewidth=0.6 + 2.4 * strength,
                color=tint(colour, 0.55 * (1 - strength)),
                alpha=0.5 + 0.5 * strength,
                linestyle="solid",
            )
        ax.add_patch(FancyArrowPatch(
            (x0, y0 + 0.16), (x1, y1 - 0.2),
            arrowstyle="-|>", mutation_scale=7, zorder=1, **style,
        ))

    scores = [abs(float(n.metadata.get("attribution_score") or 0.0)) for n in kept.values()
              if n.metadata.get("role") != "seed"]
    s_max = max(scores) if scores else 1.0
    for uuid, node in kept.items():
        x, y = positions[uuid]
        fid = node.feature_id
        is_seed = node.metadata.get("role") == "seed"
        is_inhibitor = "inhibitor" in str(node.metadata.get("role") or "")
        base = RED if is_inhibitor else BLUE
        if is_seed:
            face, edge_colour, text_colour, lw = INK, INK, "white", 1.4
        else:
            strength = abs(float(node.metadata.get("attribution_score") or 0.0)) / s_max
            face = tint(base, 0.9 - 0.55 * strength)
            edge_colour = tint(base, 0.25)
            text_colour = INK
            lw = 0.9
        label = f"{KIND_ABBR[fid.kind]}{fid.layer}/{fid.index}"
        gloss = (annotations or {}).get(label)
        box_w = 0.8
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, y - 0.16), box_w, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face, edgecolor=edge_colour, linewidth=lw, zorder=3,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.0,
                color=text_colour, zorder=4,
                fontweight="bold" if is_seed else "normal")
        if gloss:
            ax.text(x, y - 0.3, gloss, ha="center", va="top", fontsize=6.0,
                    color=INK_MUTED, zorder=4, style="italic")

    for row_index, layer, kind in legend:
        ax.text(-1.55, row_index, f"L{layer} {kind}", ha="left", va="center",
                fontsize=10.0, color=INK_MUTED)


def render_circuit_dag(
    circuit: Any,
    out_path: Path,
    *,
    title: Optional[str] = None,
    top_n: int = 24,
    max_in_edges: int = 4,
    edge_weight_floor: float = 0.03,
    annotations: Optional[Mapping[str, str]] = None,
    context_panels: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Render one wired circuit to out_path (PNG + vector PDF sibling).

    With context_panels (the JSON produced by the case-study context dump),
    the figure becomes a three-column composite: seed contexts on the left,
    the DAG in the middle, top-node contexts on the right. Bold spans mark
    tokens where the latent fires; ✕ lines are non-activating hard negatives.
    """

    plt = configure_matplotlib()

    if context_panels is None:
        fig, ax = plt.subplots(figsize=(12.0, 8.5))
        _draw_dag(ax, circuit, top_n=top_n, max_in_edges=max_in_edges,
                  edge_weight_floor=edge_weight_floor, annotations=annotations, plt=plt)
        if title:
            ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        return save_figure(fig, out_path)

    # Sized so the text columns wrap tightly around their measured widths —
    # widening the canvas only adds whitespace, the text does not reflow.
    fig_w, fig_h = 28.0, 14.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    left_margin, right_margin = 0.004, 0.998
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.45, 1.35, 1.45], wspace=0.03,
        left=left_margin, right=right_margin, bottom=0.005,
        top=0.955 if title else 0.99,
    )
    # Column text widths must match the ACTUAL axis widths (figure width
    # minus margins/wspace), else justified lines overflow their cells.
    ratios_total = 1.45 + 1.35 + 1.45
    usable_pts = fig_w * 72 * (right_margin - left_margin) * 0.98  # wspace share
    zone_pts = usable_pts * (1.45 / ratios_total)
    inner_wspace = 0.05  # gap between the two sub-columns in each zone
    seed_col_pts = zone_pts / 2 - 20
    node_col_pts = zone_pts / 2 - 20

    # Left: two seed-context columns — positives | negatives.
    seed_panel = context_panels.get("seed", {})
    left_grid = grid[0, 0].subgridspec(1, 2, wspace=inner_wspace)
    ax_pos = fig.add_subplot(left_grid[0, 0])
    ax_neg = fig.add_subplot(left_grid[0, 1])
    _draw_context_column(
        ax_pos,
        [("Positive Sequences", {"top": seed_panel.get("top", [])})],
        col_width=seed_col_pts, fontsize=11.0,
    )
    _draw_context_column(
        ax_neg,
        [("Negative Sequences", {"neg": seed_panel.get("neg", [])})],
        col_width=seed_col_pts, fontsize=11.0, max_neg=6,
    )

    ax_dag = fig.add_subplot(grid[0, 1])
    _draw_dag(ax_dag, circuit, top_n=top_n, max_in_edges=max_in_edges,
              edge_weight_floor=edge_weight_floor, annotations=annotations, plt=plt)

    # Right: 3 rows x 2 columns of node context panels (full sequences).
    nodes = list(context_panels.get("nodes", []))[:6]
    node_grid = grid[0, 2].subgridspec(3, 2, wspace=inner_wspace, hspace=0.04)
    for i, node in enumerate(nodes):
        ax_cell = fig.add_subplot(node_grid[i // 2, i % 2])
        _draw_context_column(
            ax_cell, [(f"{node.get('label', '')}", node)],
            col_width=node_col_pts, fontsize=10.0, max_neg=1,
        )

    if title:
        fig.text(0.005, 0.99, title, ha="left", va="top",
                 fontweight="bold", fontsize=16.0, color=INK)
    return save_figure(fig, out_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a wired circuit as a DAG figure.")
    parser.add_argument("--input", type=Path, required=True, help="wired-circuits.pt")
    parser.add_argument("--key", default=None, help="circuit key (default: render all)")
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/case-study/figures"))
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--max-in-edges", type=int, default=4)
    parser.add_argument("--edge-weight-floor", type=float, default=0.03)
    parser.add_argument("--annotations", type=Path, default=None,
                        help="JSON mapping node labels (e.g. 'M2/123') to gloss text")
    parser.add_argument("--context-panels", type=Path, default=None,
                        help="context-panels JSON (seed + node contexts with bold spans)")
    parser.add_argument("--title", default=None,
                        help="header line (seed identity + eval values)")
    args = parser.parse_args(argv)

    wired = torch.load(args.input, map_location="cpu", weights_only=False)
    annotations = (
        json.loads(args.annotations.read_text(encoding="utf-8")) if args.annotations else None
    )
    context_panels = (
        json.loads(args.context_panels.read_text(encoding="utf-8"))
        if args.context_panels else None
    )
    keys = [args.key] if args.key else list(wired)
    args.out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        circuit = wired[key]
        suffix = "__panels" if context_panels else ""
        path = render_circuit_dag(
            circuit,
            args.out / f"{key}{suffix}.png",
            title=args.title,
            top_n=args.top_n,
            max_in_edges=args.max_in_edges,
            edge_weight_floor=args.edge_weight_floor,
            annotations=annotations,
            context_panels=context_panels,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
