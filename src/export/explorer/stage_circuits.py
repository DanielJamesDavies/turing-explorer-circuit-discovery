"""Stage 4: circuits, members, metrics; optional wiring overlay.

Every circuit becomes one row with a compressed graph:
    nodes: [{gid, role: seed|member, attribution, alpha}]
    edges: [{src, dst, type: attribution|wired, weight, methods?, consensus?}]
Star circuits (062) store member -> seed attribution edges. A wiring file adds
member -> member edges to a subset of circuits and flips their topology to
'wired'; the rest stay stars.

Wiring input (jsonl, one edge per line; written by a future wiring run):
    {"target": "5.mlp.2277", "src": "3.attn.100", "dst": "4.mlp.55",
     "weight": 0.12, "methods": {"Z": .., "A": .., "C": ..}, "consensus": true}
Both endpoints must be nodes of that target's circuit.
"""
from __future__ import annotations

import collections
import json
import time
from typing import Dict, List, Optional

import numpy as np

from export.explorer import ids
from export.explorer.bundle import Bundle, flag, num, pack, unpack
from export.explorer.run_source import RunSource

HEADLINE = ["n", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "phi_cf_alpha_blind_tk",
            "amp_any", "sib_C", "rank_clean", "lifted_C", "near_threshold"]
EVAL_COLS = ["n", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "phi_cf_alpha_blind_tk",
             "phi_sup_alpha_tk", "phi_pin_alpha_topk_tk", "release_tk", "vacuous_tk",
             "free0_pre", "freeM_topk_pre", "freeN_topk_pre"]
SPEC_COLS = ["rank_clean", "rank_circuit", "rank_empty", "in_topk_clean", "in_topk_circuit", "jaccard_circuit",
             "n_siblings_scored", "sibling_faith_median", "n_control_scored", "control_faith_median",
             "specificity_gap", "target_faith_pre"]
CURVE_POINTS = 100

# Pass rule adopted for protocol v1 (DAN-8, 2026-09-27; experiments/062-h100-protocol-v1/pass_rule.py):
# held-out strongest contexts, Z, A and C faithfulness (activation read) all in [0.8, 1.5] AND necessity >= 0.9.
PASS_RULE = "Z,A,C faithfulness (free0/freeM_topk/freeN_topk _tk, held=strong) in [0.8, 1.5] and phi_sup_blind_tk >= 0.9"


def passes(ev: Optional[dict]) -> Optional[int]:
    if not ev:
        return None
    faith = [num(ev.get(c)) for c in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = num(ev.get("phi_sup_blind_tk"))
    ok = all(f is not None and 0.8 <= f <= 1.5 for f in faith) and nec is not None and nec >= 0.9
    return int(ok)


def _downsample(curve: dict) -> dict:
    out = {}
    for k, v in curve.items():
        if isinstance(v, dict):
            out[k] = _downsample(v)
        elif isinstance(v, list) and v:
            idx = np.unique(np.linspace(0, len(v) - 1, min(CURVE_POINTS, len(v))).round().astype(int))
            out[k] = [float(v[i]) for i in idx]
            out.setdefault("_step", [int(i) for i in idx])
    return out


def _graph(circ) -> tuple[dict, List[tuple]]:
    """(graph json, member rows without cid) from a research Circuit."""
    by_uuid = {}
    nodes = []
    seed_gid = None
    for n in circ.nodes.values():
        f = n.feature_id
        gid = ids.gid_of(f.layer, f.kind, f.index)
        role = "seed" if n.metadata.get("role") == "seed" else "member"
        if role == "seed":
            seed_gid = gid
        by_uuid[n.uuid] = gid
        nodes.append(dict(gid=gid, role=role, attribution=num(n.metadata.get("attribution_score")),
                          alpha=num(n.metadata.get("amplitude"))))
    nodes.sort(key=lambda d: (d["role"] != "seed", -(abs(d["attribution"] or 0))))
    edges = [dict(src=by_uuid[e.source_uuid], dst=by_uuid[e.target_uuid], type="attribution",
                  weight=num(e.metadata.get("weight"))) for e in circ.edges]
    members = [(d["gid"], "member", d["attribution"], d["alpha"]) for d in nodes if d["role"] == "member"]
    return dict(seed=seed_gid, nodes=nodes, edges=edges), members


def build(bundle: Bundle, run: RunSource, run_name: str) -> dict:
    t0 = time.time()
    conn = bundle.connect()
    old = [r[0] for r in conn.execute("SELECT cid FROM circuit WHERE run = ?", (run_name,))]
    for table in ("member", "circuit_eval", "circuit_spec", "train_curve"):
        conn.executemany(f"DELETE FROM {table} WHERE cid = ?", [(c,) for c in old])
    conn.execute("DELETE FROM circuit WHERE run = ?", (run_name,))
    next_cid = (conn.execute("SELECT COALESCE(MAX(cid), 0) FROM circuit").fetchone()[0] or 0) + 1

    n_circ, n_members = 0, 0
    for key in run.targets:
        circ = run.circuit(key)
        if circ is None:
            continue
        layer, kind, latent = ids.parse_key(key)
        seed_gid = ids.gid_of(layer, kind, latent)
        graph, members = _graph(circ)
        if graph["seed"] != seed_gid:
            raise ValueError(f"{key}: seed node {graph['seed']} != key gid {seed_gid}")
        crow = run.contexts.get(key, {})
        arm = crow.get("arm") or run.status.get(key, {}).get("arm") or "-"
        head = run.headline.get(key, {})
        ev = run.eval.get(key, {})
        tr = run.train.get(key, {})
        cid = next_cid
        next_cid += 1
        meta = dict(name=circ.name, uuid=circ.uuid, metadata=circ.metadata, target=key,
                    train={k: v for k, v in tr.items() if k not in ("curve", "curve_steps")},
                    arm_description={"B": "32 strongest + 16 mid-band training contexts",
                                     "A": "48 strongest (thin target fallback)"}.get(arm))
        conn.execute(
            "INSERT INTO circuit(cid, key, run, arm, seed_gid, layer, kind, latent, topology, n_nodes, n_edges, "
            + ", ".join(HEADLINE) + ", vacuous, pass, graph, meta) VALUES ("
            + ",".join("?" * (11 + len(HEADLINE) + 4)) + ")",
            (cid, ids.circuit_key(run_name, arm, key), run_name, arm, seed_gid, layer, kind, latent, "star",
             len(graph["nodes"]), len(graph["edges"]),
             *[flag(head.get(c)) if c in ("amp_any", "near_threshold") else num(head.get(c)) for c in HEADLINE],
             flag(ev.get("strong", {}).get("vacuous_tk")), passes(ev.get("strong")), pack(graph), pack(meta)))
        conn.executemany("INSERT INTO member VALUES (?,?,?,?,?)", [(g, cid, r, a, al) for g, r, a, al in members])
        for held, row in ev.items():
            conn.execute("INSERT INTO circuit_eval VALUES (" + ",".join("?" * (2 + len(EVAL_COLS) + 1)) + ")",
                         (cid, held, *[flag(row.get(c)) if c == "vacuous_tk" else num(row.get(c)) for c in EVAL_COLS],
                          pack(row)))
        for pi, row in run.spec.get(key, {}).items():
            conn.execute("INSERT INTO circuit_spec VALUES (" + ",".join("?" * (2 + len(SPEC_COLS) + 1)) + ")",
                         (cid, pi, *[num(row.get(c)) for c in SPEC_COLS], pack(row)))
        if tr.get("curve"):
            conn.execute("INSERT INTO train_curve VALUES (?,?)", (cid, pack(_downsample(tr["curve"]))))
        n_circ += 1
        n_members += len(members)
        if n_circ % 250 == 0:
            print(f"    circuits: {n_circ} ({time.time() - t0:.0f}s)", flush=True)
    conn.commit()
    conn.close()
    return dict(run=run.dir, run_name=run_name, n_circuits=n_circ, n_members=n_members, pass_rule=PASS_RULE,
                secs=round(time.time() - t0, 1))


def apply_wiring(bundle: Bundle, paths: List[str], run_name: str) -> dict:
    """Overlay wired edges from jsonl files onto this run's circuits (topology -> 'wired')."""
    edges: Dict[str, List[dict]] = collections.defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    edges[e["target"]].append(e)
    conn = bundle.connect()
    wired, rejected = 0, {}
    for target, elist in edges.items():
        row = conn.execute("SELECT cid, graph FROM circuit WHERE run = ? AND layer = ? AND kind = ? AND latent = ?",
                           (run_name, *ids.parse_key(target))).fetchone()
        if row is None:
            rejected[target] = "no circuit for target"
            continue
        cid, blob = row
        graph = unpack(blob)
        node_gids = {n["gid"] for n in graph["nodes"]}
        new = []
        for e in elist:
            s, d = ids.gid_of(*ids.parse_key(e["src"])), ids.gid_of(*ids.parse_key(e["dst"]))
            if s not in node_gids or d not in node_gids:
                rejected[target] = f"edge endpoint not in circuit: {e['src']} -> {e['dst']}"
                new = None
                break
            new.append(dict(src=s, dst=d, type="wired", weight=num(e.get("weight")),
                            methods=e.get("methods"), consensus=e.get("consensus")))
        if new is None:
            continue
        graph["edges"] = [x for x in graph["edges"] if x["type"] != "wired"] + new
        conn.execute("UPDATE circuit SET topology = 'wired', n_edges = ?, graph = ? WHERE cid = ?",
                     (len(graph["edges"]), pack(graph), cid))
        wired += 1
    conn.commit()
    conn.close()
    return dict(files=paths, n_wired=wired, n_rejected=len(rejected), rejected=dict(list(rejected.items())[:20]))
