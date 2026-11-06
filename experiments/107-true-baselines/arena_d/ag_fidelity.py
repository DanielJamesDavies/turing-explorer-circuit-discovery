"""Arena D1 fidelity: is the circuit-tracer we run THEIR circuit-tracer, reproducing their published Gemma graph?
(spec attribution-graphs.md §10; run in ct107; 038's ct_faithfulness.py checks, re-pinned to 255d989.)

  F1  provenance   circuit-tracer installed from git at 255d989 (direct_url.json), not editable, and every installed
                   file's sha256 equals its RECORD entry (no local edits); library versions recorded
  F3  Dallas       Neuronpedia's gemma-fact-dallas-austin graph (circuit-tracer 1.0.0; max_n_logits 10, desired 0.95,
                   max_feature_nodes 5000; prune 0.7 / 0.9; then their frontend filter create_used_nodes_and_edges):
                   reference out/fidelity/gemma-fact-dallas-austin.json = 697 feature nodes, 137 error nodes, 11
                   embeddings, 10 logits, 34,801 edges, " Austin" p 0.450. Pass: every reference logit prob within
                   +-0.01, feature-node Jaccard (layer_feature_pos ids) >= 0.9, feature-node count within +-10%
  F4  pinned       the reference's 22 pinned feature nodes are active and survive prune 0.6 / 0.99 and 0.8 / 0.98
                   (on the F3 graph)
  F5  identity     038's prune_rooted (logit root, logit probs) == their prune_graph, node and edge masks bit-exact,
                   at 0.8 / 0.98 and 0.7 / 0.9 (on the F3 graph)
  F6  invariance   batch 64 vs 256, and lazy vs eager encoders (a second build): pruned (0.8 / 0.98) feature-node
                   Jaccard >= 0.95 and edge corr on shared survivors >= 0.999
Tolerances are ours (spec §10: neither paper gives any for Gemma). A FAIL is a result, not a crash: the script exits 0
and writes every check to --out (exit 1 only with --strict).

  GPU night (bf16, unsloth mirror, offline):
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ~/venvs/ct107/bin/python -B -I ag_fidelity.py --checks F1,F3,F4,F5,F6 \
        --dtype bfloat16 --hf-model unsloth/gemma-2-2b --device cuda
  pod: --dtype float32 (and bfloat16), no --hf-model (google/gemma-2-2b by name; HF_TOKEN from the environment)
  CPU dry run (no weights): ... ag_fidelity.py --dry-run   (F1 for real; reference parse + tokenisation; F3's
    comparison on the reference against itself; F5 / F6 on their tiny random test model)
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D038 = ROOT / "experiments" / "038-transcoder-compare-gemma"
REF = HERE / "out" / "fidelity" / "gemma-fact-dallas-austin.json"
OUT_DEFAULT = HERE / "out" / "fidelity"
DALLAS = "Fact: The capital of the state containing Dallas is"
sys.path.insert(0, str(HERE))
import ag_tasks as AT  # noqa: E402


def _ct_prune():
    sys.dont_write_bytecode = True                       # nothing written into 038's folder
    if str(D038) not in sys.path:
        sys.path.insert(0, str(D038))
    import ct_prune  # noqa: E402  (038's, read only)
    return ct_prune


# ------------------------------------------------------------------------------------------------------- F1
def check_f1() -> dict:
    from importlib import metadata
    prov = AT.ct_provenance()
    dist = metadata.distribution("circuit-tracer")
    bad, n_checked = [], 0
    for f in dist.files or []:
        if f.hash is None or f.hash.mode != "sha256":
            continue
        p = Path(dist.locate_file(f))
        if not p.exists():
            bad.append(str(f) + " (missing)")
            continue
        h = base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).rstrip(b"=").decode()
        n_checked += 1
        if h != f.hash.value:
            bad.append(str(f))
    ok = prov.get("commit") == AT.CT_COMMIT and not prov.get("editable") and not bad and n_checked > 0
    return dict(passed=ok, provenance=prov, record_files_checked=n_checked, record_mismatches=bad[:20],
                versions=AT.lib_versions())


# ------------------------------------------------------------------------------------------------------- reference
def load_reference(path=REF) -> dict:
    d = json.loads(Path(path).read_text())
    nodes = d["nodes"]
    feats = {n["node_id"] for n in nodes if n["feature_type"] == "cross layer transcoder"}
    logits = {int(n["node_id"].split("_")[1]): float(n["token_prob"]) for n in nodes if n["feature_type"] == "logit"}
    pinned = [s for s in d["qParams"]["pinnedIds"] if re.fullmatch(r"\d+_\d+_\d+", s) and int(s.split("_")[0]) < 26]
    counts = {}
    for n in nodes:
        counts[n["feature_type"]] = counts.get(n["feature_type"], 0) + 1
    return dict(features=feats, logit_probs=logits, pinned=pinned, counts=counts, n_edges=len(d["links"]),
                n_distinct_latents=len({tuple(s.split("_")[:2]) for s in feats}),
                prompt_tokens=d["metadata"]["prompt_tokens"], settings=d["metadata"].get("generation_settings"),
                pruning=d["metadata"].get("pruning_settings"), generator=d["metadata"]["info"]["generator"],
                labels={int(n["node_id"].split("_")[1]): n.get("clerp") for n in nodes if n["feature_type"] == "logit"})


def frontend_view(torch, graph, model_tokenizer, node_t: float, edge_t: float) -> dict:
    """Their prune_graph, then their create_nodes / create_used_nodes_and_edges (create_graph_files.py), the frontend
    graph Neuronpedia shows; reduced to the same summary as load_reference."""
    from circuit_tracer.graph import prune_graph
    from circuit_tracer.utils.create_graph_files import create_nodes, create_used_nodes_and_edges
    nm, em, cum = (x.cpu() for x in prune_graph(graph, node_t, edge_t))
    graph.to("cpu")
    nodes = create_nodes(graph, nm, model_tokenizer, cum)
    used, edges = create_used_nodes_and_edges(graph, nodes, em)
    feats = {n.node_id for n in used if n.feature_type == "cross layer transcoder"}
    counts = {}
    for n in used:
        counts[n.feature_type] = counts.get(n.feature_type, 0) + 1
    probs = {int(t[1]): float(p) for t, p in zip(graph.logit_targets, graph.logit_probabilities.tolist())}
    return dict(features=feats, logit_probs=probs, counts=counts, n_edges=len(edges),
                n_distinct_latents=len({tuple(s.split("_")[:2]) for s in feats}))


def compare_f3(ours: dict, ref: dict) -> dict:
    jac = len(ours["features"] & ref["features"]) / max(len(ours["features"] | ref["features"]), 1)
    dp = {v: (ours["logit_probs"].get(v), p, None if v not in ours["logit_probs"] else abs(ours["logit_probs"][v] - p))
          for v, p in ref["logit_probs"].items()}
    probs_ok = all(x[2] is not None and x[2] <= 0.01 for x in dp.values())
    nf, nr = len(ours["features"]), len(ref["features"])
    count_ok = abs(nf - nr) <= 0.10 * nr
    return dict(passed=bool(probs_ok and jac >= 0.9 and count_ok), feature_jaccard=jac, n_feature_nodes=nf,
                ref_feature_nodes=nr, n_distinct_latents=ours["n_distinct_latents"],
                ref_distinct_latents=ref["n_distinct_latents"], counts=ours["counts"], ref_counts=ref["counts"],
                n_edges=ours["n_edges"], ref_edges=ref["n_edges"], logit_probs_ok=probs_ok,
                logit_probs={str(v): dict(ours=x[0], ref=x[1], absdiff=x[2]) for v, x in dp.items()},
                count_ok=count_ok)


# ------------------------------------------------------------------------------------------------------- F4-F6
def survivors(torch, graph, node_t, edge_t) -> set:
    from circuit_tracer.graph import prune_graph
    pr = prune_graph(graph, node_t, edge_t)
    n_f = len(graph.selected_features)
    af = graph.active_features[graph.selected_features.to(graph.active_features.device)].cpu()
    keep = pr.node_mask[:n_f].cpu()
    return {"%d_%d_%d" % (l, f, p) for (l, p, f), k in zip(af.tolist(), keep.tolist()) if k}


def check_f4(torch, graph, pinned) -> dict:
    active = {"%d_%d_%d" % (l, f, p) for l, p, f in graph.active_features.cpu().tolist()}
    s06, s08 = survivors(torch, graph, 0.6, 0.99), survivors(torch, graph, 0.8, 0.98)
    n_act, n06, n08 = (sum(x in s for x in pinned) for s in (active, s06, s08))
    return dict(passed=n_act == n06 == n08 == len(pinned), n_pinned=len(pinned), active=n_act, survive_06_099=n06,
                survive_08_098=n08, missing_active=[x for x in pinned if x not in active])


def check_f5(torch, graph) -> dict:
    cp = _ct_prune()
    res = {}
    for nt, et in ((0.8, 0.98), (0.7, 0.9)):
        res["%g/%g" % (nt, et)] = bool(cp.identity_check(graph, nt, et))
    return dict(passed=all(res.values()), identity=res)


def compare_graphs(torch, ga, gb) -> dict:
    """038's compare_graphs: identity-based (selection order may differ), pruned at 0.8 / 0.98."""
    def ids(g):
        sel = g.selected_features.cpu().flatten()
        af = g.active_features.cpu()[sel]
        out = [("f", int(l), int(p), int(f)) for l, p, f in af.tolist()]
        return out + [("o", i) for i in range(len(sel), g.adjacency_matrix.shape[0])]
    from circuit_tracer.graph import prune_graph
    ia, ib = ids(ga), ids(gb)
    fa, fb = {x for x in ia if x[0] == "f"}, {x for x in ib if x[0] == "f"}
    pa, pb = prune_graph(ga, 0.8, 0.98).node_mask.cpu(), prune_graph(gb, 0.8, 0.98).node_mask.cpu()
    sa = {x for x, k in zip(ia, pa.tolist()) if k and x[0] == "f"}
    sb = {x for x, k in zip(ib, pb.tolist()) if k and x[0] == "f"}
    pos_a, pos_b = {x: i for i, x in enumerate(ia)}, {x: i for i, x in enumerate(ib)}
    shared = sorted(sa & sb)
    A, B = ga.adjacency_matrix.cpu().float(), gb.adjacency_matrix.cpu().float()
    ra, rb = torch.tensor([pos_a[x] for x in shared]), torch.tensor([pos_b[x] for x in shared])
    corr = float("nan")
    if len(shared) > 1:
        ea, eb = A[ra][:, ra].flatten(), B[rb][:, rb].flatten()
        corr = float(torch.corrcoef(torch.stack([ea, eb]))[0, 1])
    jac_sel = len(fa & fb) / max(len(fa | fb), 1)
    jac_pr = len(sa & sb) / max(len(sa | sb), 1)
    return dict(passed=bool(jac_pr >= 0.95 and corr >= 0.999), selected_jaccard=jac_sel, pruned_jaccard=jac_pr,
                n_pruned=(len(sa), len(sb)), edge_corr=corr)


def attr(torch, model, prompt, batch, cap):
    from circuit_tracer import attribute
    t0 = time.time()
    g = attribute(prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=batch, max_feature_nodes=cap,
                  verbose=False)
    return g, round(time.time() - t0, 2)


# ------------------------------------------------------------------------------------------------------- main
def run_real(a, torch, checks, rep):
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = load_reference(a.reference)
    model, info = AT.build_model(torch, a.model, a.dtype, dev, a.hf_model, a.transcoders, lazy_encoder=True)
    rep["model"] = info
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    g, t = attr(torch, model, DALLAS, a.batch_size, a.max_feature_nodes)
    toks = [model.tokenizer.decode([int(x)]) for x in g.input_tokens.tolist()]
    rep["dallas"] = dict(t_attribute_s=t, n_pos=int(g.n_pos), n_active=int(g.active_features.shape[0]),
                         n_selected=int(len(g.selected_features)), tokens=toks, tokens_match_ref=toks ==
                         ref["prompt_tokens"],
                         peak_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if str(dev).startswith("cuda")
                         else None)
    g.to(dev)
    if "F4" in checks:
        rep["checks"]["F4"] = check_f4(torch, g, ref["pinned"])
    if "F5" in checks:
        rep["checks"]["F5"] = check_f5(torch, g)
    if "F6" in checks:
        g2, t2 = attr(torch, model, DALLAS, 256, a.max_feature_nodes)
        g2.to(dev)
        rep["checks"]["F6"] = dict(batch_64_vs_256=compare_graphs(torch, g, g2), t_attribute_b256_s=t2)
        del g2
    if "F3" in checks:
        rep["checks"]["F3"] = compare_f3(frontend_view(torch, g, model.tokenizer, 0.7, 0.9), ref)
    if "F6" in checks:
        gk = g
        g.to("cpu")
        # free the lazy model BEFORE the eager build. `del` alone most likely did not (module/hook reference cycles
        # keep the weights allocated until a gc pass): the 2026-11-05 pilot had both models resident at once
        # (16 GB local card: 15,951 MiB, WDDM spill -> "CUDA driver error: device not ready" in the eager build's
        # .to()). Memory only; nothing measured changes.
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 2 ** 30
            rep["checks"]["F6"].update(resident_gb_before_eager=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                                       free_gb_before_eager=round(free_gb, 2))
            # local-card guard (AG_F6_MIN_FREE_GB, default 0 = off = pod behaviour; run_pilot_d.sh sets it): if the
            # free memory is still too small, record F6's eager half as skipped instead of spilling / crashing
            min_free = float(os.environ.get("AG_F6_MIN_FREE_GB", "0"))
            if free_gb < min_free:
                f6 = rep["checks"]["F6"]
                f6.update(lazy_vs_eager="SKIPPED: %.2f GB free < AG_F6_MIN_FREE_GB %.2f" % (free_gb, min_free),
                          passed=False)
                print("NOTE F6 eager build skipped: %.2f GB free < %.2f" % (free_gb, min_free), flush=True)
                return
        model2, info2 = AT.build_model(torch, a.model, a.dtype, dev, a.hf_model, a.transcoders, lazy_encoder=False)
        g3, t3 = attr(torch, model2, DALLAS, a.batch_size, a.max_feature_nodes)
        gk.to(dev)
        g3.to(dev)
        f6 = rep["checks"]["F6"]
        f6.update(lazy_vs_eager=compare_graphs(torch, gk, g3), eager_model=info2, t_attribute_eager_s=t3)
        f6["passed"] = bool(f6["batch_64_vs_256"]["passed"] and f6["lazy_vs_eager"]["passed"])


def run_dry(a, torch, checks, rep):
    ref = load_reference(a.reference)
    rep["reference"] = {k: (sorted(v)[:5] if isinstance(v, set) else v) for k, v in ref.items() if k != "features"}
    rep["reference"]["n_features"] = len(ref["features"])
    # the F3 comparison on the reference against itself (exercises the code; must pass)
    rep["checks"]["F3_selfcompare"] = compare_f3(dict(ref), ref)
    # tokenisation of the Dallas prompt with the local mirror's tokenizer vs the reference's prompt_tokens
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(AT.local_or_id(a.hf_model or "unsloth/gemma-2-2b"))
        ids = tok(DALLAS).input_ids
        rep["dallas_tokens_match_ref"] = [tok.decode([i]) for i in ids] == ref["prompt_tokens"]
    except Exception as e:  # noqa: BLE001
        rep["dallas_tokens_match_ref"] = "tokenizer not available offline: %r" % (e,)
    # F5 / F6 machinery on their tiny random test model (CPU)
    model, info = AT.tiny_model(torch)
    g_ = torch.Generator().manual_seed(4)
    s = torch.tensor([0] + torch.randint(1, 16, (7,), generator=g_).tolist())
    with AT.special_ids(model, [0]):
        g, _ = attr(torch, model, s, 512, None)
        g2, _ = attr(torch, model, s, 4, None)
    rep["checks"]["F5_tiny"] = check_f5(torch, g)
    rep["checks"]["F6_tiny_batch"] = compare_graphs(torch, g, g2)
    rep["checks"]["scores_tiny"] = dict(
        ours=AT.graph_scores(torch, g),
        theirs=__import__("circuit_tracer.graph", fromlist=["compute_graph_scores"]).compute_graph_scores(g),
        keep_all=AT.graph_scores(torch, g, torch.ones(len(g.selected_features), dtype=torch.bool)))
    sc = rep["checks"]["scores_tiny"]
    sc["passed"] = max(abs(x - y) for x, y in zip(sc["ours"], sc["theirs"])) < 1e-6 and \
        max(abs(x - y) for x, y in zip(sc["keep_all"], sc["theirs"])) < 1e-6
    rep["plan"] = dict(checks=sorted(checks), dtype=a.dtype, hf_model=a.hf_model, model=a.model,
                       transcoders=a.transcoders, batch_size=a.batch_size, max_feature_nodes=a.max_feature_nodes)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checks", default="F1,F3,F4,F5,F6")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--hf-model", default=None)
    p.add_argument("--transcoders", default=AT.TC_REF)
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-feature-nodes", type=int, default=5000, help="Neuronpedia Dallas: 5000")
    p.add_argument("--reference", default=str(REF))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 if any check fails")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    import torch
    AT.set_threads(torch)
    checks = {c.strip() for c in a.checks.split(",") if c.strip()}
    rep = dict(label="DRY RUN [CPU, tiny model]" if a.dry_run else "fidelity %s %s" % (a.dtype,
                                                                                         a.hf_model or a.model),
               started=time.strftime("%Y-%m-%d %H:%M:%S"), checks={})
    if "F1" in checks:
        rep["checks"]["F1"] = check_f1()
    (run_dry if a.dry_run else run_real)(a, torch, checks, rep)
    out = Path(a.out) if a.out else OUT_DEFAULT / ("ag_fidelity_%s%s.json" % (
        "dryrun" if a.dry_run else a.dtype, "" if a.dry_run or not a.hf_model else "_" + a.hf_model.split("/")[0]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k, v in rep["checks"].items():
        brief = {x: v[x] for x in v if x not in ("passed", "logit_probs", "versions", "provenance", "record_mismatches",
                                                 "eager_model")}
        print("%s  %-14s %s" % ("PASS" if v.get("passed") else "FAIL", k, json.dumps(brief, default=str)[:300]))
    failed = [k for k, v in rep["checks"].items() if not v.get("passed")]
    print("VERDICT: %s -> %s" % ("ALL PASS" if not failed else "FAILED " + ",".join(failed), out))
    return 1 if (failed and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
