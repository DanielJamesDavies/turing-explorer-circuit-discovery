"""Arena D1: attribution graphs (circuit-tracer, as published) on the SVA task prompts, one graph per prompt.

Two subcommands, two environments:
  prompts  (repo .venv)  writes out/ag_tasks/prompts/<task>.json from wcm_gemma.build_task(task, "sfc"): SFC's 100
           discovery + 40 evaluation prompts (seed 12, min length, Gemma tokenizer), the exact rows D1-WCM fits and
           scores, so the AG and WCM rows see the same prompts. Gemma token ids with BOS.
  run      (ct107)       for each prompt of --split (default held = the 40 evaluation prompts, IMPLEMENTATION_PLAN §1.1):
             attribute(ids, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=B, max_feature_nodes=None,
                       offload=None)                            # their API, as published (logit root; spec §3.2)
             prune_graph(g, node_threshold=t, edge_threshold=0.98) for t in {0.1, ..., 0.9, 0.95, 0.98, 0.99}
           and saves, per prompt, ONE json (not the graph): logit targets + probs, the answer tokens' rank among
           them, their compute_graph_scores (unpruned), and per threshold the kept feature nodes (layer, pos, feature),
           kept error nodes (layer, pos), kept edge count, distinct (layer, feature) count, and the pruned-graph
           replacement / completeness with pruned feature nodes counted as error nodes (OUR implementation of their
           metric, spec §9.2: a dropped feature node becomes an error node, i.e. its incoming edges are removed and it
           joins the error set; their formulas are then applied unchanged; with nothing dropped this equals
           compute_graph_scores, checked by --fixture). Resumable per prompt (atomic write; an existing file is
           skipped). STOP_AT=HH:MM (env) stops before a new prompt.

Transcoders are pinned by revision, never the "gemma" alias (it resolves main):
  mwhanna/gemma-scope-transcoders@3412e19be8f72d47f8683cb92028a46513222b2b
Offline (HF_HUB_OFFLINE=1) with only the local snapshot bd577315 cached, the pinned revision cannot resolve; the run then
loads @bd577315 (every layer_*.safetensors LFS sha256 equals 3412e19b's, README §1 #3) and records it in each output
(`transcoders_used`). With the hub reachable (pod) the pin is used and a failure is an error.
Weights: --hf-model unsloth/gemma-2-2b (bf16 mirror via hf_model=, 038's route, labelled) or none = TL loads
google/gemma-2-2b by name (gated: HF_TOKEN from the environment only, read by huggingface_hub itself; never printed).

  .venv: CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 ./.venv/bin/python -B experiments/107-true-baselines/arena_d/ag_tasks.py prompts
  ct107: CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ~/venvs/ct107/bin/python -B -I experiments/107-true-baselines/arena_d/ag_tasks.py \
             run --tasks rc --split held [--limit 4] --dtype bfloat16 --hf-model unsloth/gemma-2-2b --device cuda
  CPU dry run (no weights, a tiny random Gemma-2 replacement model as in their own tests):
         ... ag_tasks.py run --fixture tiny --device cpu --out <tmp>
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "out" / "ag_tasks"
TC_REPO = "mwhanna/gemma-scope-transcoders"
TC_REV = "3412e19be8f72d47f8683cb92028a46513222b2b"
TC_REV_CACHED = "bd5773156dea09893636c801df1237d0410307d2"
TC_REF = "%s@%s" % (TC_REPO, TC_REV)
CT_COMMIT = "255d989e93be1299d53d67b59e9fd76a0e274b14"
NODE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
EDGE_THRESHOLD = 0.98
SVA_TASKS = ("simple", "within_rc", "rc", "nounpp")
FORMAT = "arena-d-ag-task-v1"


# ------------------------------------------------------------------------------------------------------- shared
def set_threads(torch):
    n = int(os.environ.get("TORCH_THREADS", "4"))
    torch.set_num_threads(max(1, min(n, 4)))


def ct_provenance() -> dict:
    """circuit-tracer install facts: version, direct_url commit, editable or not."""
    from importlib import metadata
    out = dict(expected_commit=CT_COMMIT)
    try:
        dist = metadata.distribution("circuit-tracer")
        out["version"] = dist.version
        du = dist.read_text("direct_url.json")
        d = json.loads(du) if du else {}
        out["commit"] = d.get("vcs_info", {}).get("commit_id")
        out["url"] = d.get("url")
        out["editable"] = bool(d.get("dir_info", {}).get("editable", False))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def lib_versions() -> dict:
    from importlib import metadata
    v = {}
    for p in ("torch", "transformer-lens", "transformers", "huggingface-hub", "circuit-tracer", "numpy"):
        try:
            v[p] = metadata.version(p)
        except Exception:  # noqa: BLE001
            v[p] = None
    return v


def local_or_id(repo: str) -> str:
    """Offline, a repo id -> its cached snapshot folder: transformers 4.57.3's AutoTokenizer.from_pretrained(<repo id>)
    calls the hub API (model_info) even with HF_HUB_OFFLINE=1 and fails; a local folder does not (checked
    2026-11-05 in ct107). Online (pod) the id is returned unchanged."""
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def build_model(torch, model_name: str, dtype: str, device: str, hf_model=None, transcoders: str = TC_REF,
                lazy_encoder: bool = True, lazy_decoder: bool = True):
    """(ReplacementModel, info). Their from_pretrained, TransformerLens backend; hf_model= the local mirror (labelled)."""
    from circuit_tracer import ReplacementModel
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    kw, info = {}, dict(model_name=model_name, hf_model=hf_model, dtype=dtype, lazy_encoder=lazy_encoder,
                        lazy_decoder=lazy_decoder, transcoders_requested=transcoders)
    if hf_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        src = local_or_id(hf_model)
        info["hf_model_path"] = src
        kw = dict(hf_model=AutoModelForCausalLM.from_pretrained(src, dtype=dt),
                  tokenizer=AutoTokenizer.from_pretrained(src))
    t0 = time.time()

    def load(ref):
        return ReplacementModel.from_pretrained(model_name, ref, device=torch.device(device), dtype=dt,
                                                lazy_encoder=lazy_encoder, lazy_decoder=lazy_decoder, **kw)
    try:
        model, used = load(transcoders), transcoders
    except (FileNotFoundError, OSError) as e:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or transcoders != TC_REF:
            raise
        used = "%s@%s" % (TC_REPO, TC_REV_CACHED)
        info["transcoders_fallback"] = ("offline: %s not resolvable (%s); loaded the cached snapshot %s, whose "
                                        "layer files are byte-identical to the pin (README §1 #3)"
                                        % (TC_REV[:8], type(e).__name__, TC_REV_CACHED[:8]))
        print("NOTE " + info["transcoders_fallback"], flush=True)
        model = load(used)
    kw.pop("hf_model", None)
    info.update(transcoders_used=used, load_s=round(time.time() - t0, 1))
    if str(device).startswith("cuda"):
        info["resident_gb"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    return model, info


# tiny random Gemma-2 replacement model: their own test construction (tests/test_attributions_gemma.py,
# load_dummy_gemma_model + test_small_gemma_model's config), for CPU dry runs only.
def tiny_model(torch, n_layers: int = 2):
    import numpy as np
    from torch import nn
    from transformer_lens import HookedTransformerConfig
    from circuit_tracer import ReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    torch.manual_seed(0)
    cfg = HookedTransformerConfig.from_dict(dict(
        n_layers=n_layers, d_model=8, n_ctx=8192, d_head=4, model_name="gemma-2-2b", n_heads=2, d_mlp=16,
        act_fn="gelu_pytorch_tanh", d_vocab=16, eps=1e-06, use_attn_result=False, use_attn_scale=True,
        attn_scale=np.float64(16.0), use_split_qkv_input=False, use_hook_mlp_in=False, use_attn_in=False,
        use_local_attn=True, ungroup_grouped_query_attention=False, original_architecture="Gemma2ForCausalLM",
        from_checkpoint=False, checkpoint_index=None, checkpoint_label_type=None, checkpoint_value=None,
        tokenizer_name="gpt2", window_size=4096, attn_types=["global", "local"], init_mode="gpt2",
        normalization_type="RMSPre", device="cpu", n_devices=1, attention_dir="causal", attn_only=False, seed=None,
        initializer_range=0.02, init_weights=False, scale_attn_by_inverse_layer_idx=False,
        positional_embedding_type="rotary", final_rms=True, d_vocab_out=16, parallel_attn_mlp=False, rotary_dim=4,
        n_params=2146959360, use_hook_tokens=False, gated_mlp=True, default_prepend_bos=True, dtype=torch.float32,
        tokenizer_prepends_bos=True, n_key_value_heads=2, post_embedding_ln=False, rotary_base=10000.0,
        trust_remote_code=False, rotary_adjacent_pairs=False, load_in_4bit=False, num_experts=None,
        experts_per_token=None, relative_attention_max_distance=None, relative_attention_num_buckets=None,
        decoder_start_token_id=None, tie_word_embeddings=False, use_normalization_before_and_after=True,
        attn_scores_soft_cap=50.0, output_logits_soft_cap=0.0, use_NTK_by_parts_rope=False,
        NTK_by_parts_low_freq_factor=1.0, NTK_by_parts_high_freq_factor=4.0, NTK_by_parts_factor=8.0))
    tcs = {l: SingleLayerTranscoder(cfg.d_model, cfg.d_model * 4, JumpReLU(torch.tensor(0.0), 0.1), l)
           for l in range(cfg.n_layers)}
    for t in tcs.values():
        for _, p in t.named_parameters():
            nn.init.uniform_(p, a=-1, b=1)
    model = ReplacementModel.from_config(cfg, TranscoderSet(tcs, feature_input_hook="mlp.hook_in",
                                                            feature_output_hook="mlp.hook_out"))
    for _, p in model.named_parameters():
        nn.init.uniform_(p, a=-1, b=1)
    for t in model.transcoders:
        nn.init.uniform_(t.activation_function.threshold, a=0, b=1)
    return model, dict(model_name="TINY FIXTURE (their test construction)", dtype="float32",
                       transcoders_used="random")


@contextlib.contextmanager
def special_ids(model, ids):
    """Their patch_tokenizer_special_ids (tests): token 0 counts as the special first token of the tiny model."""
    cls = type(model.tokenizer)
    orig = cls.all_special_ids
    try:
        cls.all_special_ids = property(lambda self: ids)
        yield
    finally:
        cls.all_special_ids = orig


def graph_scores(torch, graph, keep_features=None):
    """compute_graph_scores' formulas (graph.py:316-363) with a general error set. keep_features None: their score,
    unpruned. A bool [n_features] mask: [ours, spec §9.2] every dropped feature node becomes an error node (its row
    zeroed, since an error node has no inputs, and it joins the error set); everything else unchanged."""
    from circuit_tracer.graph import compute_influence, normalize_matrix
    A = graph.adjacency_matrix
    n_log, n_tok, n_f = len(graph.logit_targets), len(graph.input_tokens), len(graph.selected_features)
    e0, e1 = n_f, n_f + n_tok * graph.cfg.n_layers
    err = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
    err[e0:e1] = True
    if keep_features is not None:
        drop = ~keep_features.to(A.device)[:n_f]
        err[:n_f] = drop
        A = A.clone()
        A[:n_f][drop] = 0
    w = torch.zeros(A.shape[0], device=A.device)
    w[-n_log:] = graph.logit_probabilities.to(A.device)
    An = normalize_matrix(A)
    I = compute_influence(An, w)
    tok = I[e1:e1 + n_tok].sum()
    rep = tok / (tok + I[err].sum())
    nef = 1 - An[:, err].sum(dim=-1)
    out = I + w
    comp = (nef * out).sum() / out.sum()
    return float(rep), float(comp)


def prune_summary(torch, graph, node_t: float, edge_t: float = EDGE_THRESHOLD) -> dict:
    from circuit_tracer.graph import prune_graph
    pr = prune_graph(graph, node_threshold=node_t, edge_threshold=edge_t)
    nm, em = pr.node_mask, pr.edge_mask
    n_f, n_tok = len(graph.selected_features), len(graph.input_tokens)
    af = graph.active_features[graph.selected_features.to(graph.active_features.device)].cpu()
    keep_f = nm[:n_f].cpu()
    feats = af[keep_f].tolist()                                       # (layer, pos, feature)
    ke = nm[n_f:n_f + n_tok * graph.cfg.n_layers].cpu().nonzero().flatten().tolist()
    errs = [list(divmod(i, n_tok)) for i in ke]                       # (layer, pos), layer-major
    rep, comp = graph_scores(torch, graph, nm[:n_f])
    return dict(node_threshold=node_t, edge_threshold=edge_t, n_feature_nodes=len(feats),
                n_latents=len({(l, f) for l, _, f in feats}), n_error_nodes=len(errs), n_edges=int(em.sum()),
                replacement_pruned=rep, completeness_pruned=comp, features=feats, errors=errs)


def stop_passed(stop_at, started) -> bool:
    if not stop_at:
        return False
    hh, mm = (int(x) for x in stop_at.split(":"))
    stop = started.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if stop <= started:
        stop += _dt.timedelta(days=1)
    return _dt.datetime.now() >= stop


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------- prompts
def cmd_prompts(a) -> int:
    sys.path.insert(0, str(HERE))
    import wcm_gemma as W                                # repo .venv
    out = Path(a.out) / "prompts"
    out.mkdir(parents=True, exist_ok=True)
    tok = W.tokenizer()
    for task in [t for t in a.tasks.split(",") if t]:
        T = W.build_task(task, "sfc", tok=tok, family=W.FAM_PLT)
        held = set(T["split"]["held"])
        rows = [dict(id=r["id"], split="held" if r["id"] in held else "train", prompt=r["prompt"], tokens=r["tokens"],
                     good=r["good"], bad=r["bad"], answers=[r["meta"]["clean_answer"], r["meta"]["patch_answer"]])
                for r in T["rows"]]
        meta = dict(task=task, source="wcm_gemma.build_task(%r, 'sfc') = arena C's SFC loader, Gemma tokenizer %s"
                    % (task, W.HF_ID), split_meta=T["split_meta"], overlap=T["overlap"])
        write_atomic(out / ("%s.json" % task), dict(meta=meta, rows=rows))
        print("%s: %d train + %d held prompts, lengths %s / %s -> %s" % (
            task, T["split_meta"]["n_train"], T["split_meta"]["n_held"], T["prompt_len_train"],
            T["prompt_len_held"], out / ("%s.json" % task)))
    return 0


# ------------------------------------------------------------------------------------------------------- run
def one_prompt(torch, model, ids, row, a, dev) -> dict:
    from circuit_tracer import attribute
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    g = attribute(ids, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=a.batch_size,
                  max_feature_nodes=a.max_feature_nodes, offload=a.offload, verbose=False)
    t_attr = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if str(dev).startswith("cuda") else None
    if not torch.equal(g.input_tokens.cpu().long(), ids.cpu().long()):
        raise RuntimeError("their ensure_tokenized changed the prompt (%s -> %s)" % (ids.tolist(),
                                                                                    g.input_tokens.tolist()))
    g.to(dev)
    targets = [dict(token=str(t[0]), vocab_idx=int(t[1]), prob=float(p))       # LogitTarget = (token, vocab_idx)
               for t, p in zip(g.logit_targets, g.logit_probabilities.tolist())]
    tids = [t["vocab_idx"] for t in targets]
    ans = {}
    for k in ("good", "bad"):
        v = int(row[k][0]) if row.get(k) else None
        ans[k] = dict(id=v, rank=(tids.index(v) if v in tids else None),
                      prob=(targets[tids.index(v)]["prob"] if v in tids else None))
    t1 = time.time()
    rep, comp = graph_scores(torch, g)
    from circuit_tracer.graph import compute_graph_scores
    rep_c, comp_c = compute_graph_scores(g)                       # theirs, to pin our formula to theirs
    prunes = [prune_summary(torch, g, t, a.edge_threshold) for t in a.node_thresholds]
    rec = dict(format=FORMAT, id=row["id"], split=row.get("split"), prompt=row.get("prompt"), tokens=ids.tolist(),
               n_pos=int(g.n_pos), n_active=int(g.active_features.shape[0]), n_selected=int(len(g.selected_features)),
               logit_targets=targets, answers=ans, replacement=rep_c, completeness=comp_c,
               scores_formula_check=dict(replacement=rep, completeness=comp,
                                         max_abs_diff=max(abs(rep - rep_c), abs(comp - comp_c))),
               prunes=prunes, t_attribute_s=round(t_attr, 2), t_prune_s=round(time.time() - t1, 2), peak_mb=peak)
    del g
    return rec


def cmd_run(a) -> int:
    import torch
    set_threads(torch)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = _dt.datetime.now()
    stop_at = os.environ.get("STOP_AT")
    out = Path(a.out)
    if a.fixture:
        model, info = tiny_model(torch)
        g = torch.Generator().manual_seed(3)
        jobs = {"tiny": [dict(id=i, split="held", prompt=None,
                              tokens=[0] + torch.randint(1, 16, (6,), generator=g).tolist(), good=[3], bad=[5])
                         for i in range(a.limit or 3)]}
        ctx = special_ids(model, [0])
    else:
        jobs = {}
        for task in [t for t in a.tasks.split(",") if t]:
            pf = out / "prompts" / ("%s.json" % task)
            if not pf.exists():
                raise SystemExit("%s missing: run `ag_tasks.py prompts` in the repo .venv first" % pf)
            rows = [r for r in json.loads(pf.read_text())["rows"] if a.split == "all" or r["split"] == a.split]
            jobs[task] = rows[:a.limit] if a.limit else rows
        if a.dry_run:
            for task, rows in jobs.items():
                print("DRY %s: %d prompts of split %s -> %s" % (task, len(rows), a.split, out / task / a.split))
            print("DRY model: %s, transcoders %s, dtype %s, hf_model %s, batch %d, max_feature_nodes %s, offload %s, "
                  "node thresholds %s, edge %g" % (a.model, a.transcoders, a.dtype, a.hf_model, a.batch_size,
                                                   a.max_feature_nodes, a.offload, list(a.node_thresholds),
                                                   a.edge_threshold))
            print("DRY provenance: %s" % ct_provenance())
            return 0
        model, info = build_model(torch, a.model, a.dtype, dev, a.hf_model, a.transcoders)
        ctx = contextlib.nullcontext()
    info.update(provenance=ct_provenance(), versions=lib_versions(), device=str(dev),
                settings=dict(max_n_logits=10, desired_logit_prob=0.95, batch_size=a.batch_size,
                              max_feature_nodes=a.max_feature_nodes, offload=a.offload,
                              node_thresholds=list(a.node_thresholds), edge_threshold=a.edge_threshold))
    n_done = n_skip = 0
    with ctx:
        for task, rows in jobs.items():
            d = out / task / a.split
            write_atomic(d / "run_info.json", info)
            for row in rows:
                f = d / ("p%04d.json" % int(row["id"]))
                if f.exists():
                    n_skip += 1
                    continue
                if stop_passed(stop_at, started):
                    print("STOP_AT %s passed: no new prompt" % stop_at, flush=True)
                    return 0
                ids = torch.tensor(row["tokens"], dtype=torch.long)
                rec = one_prompt(torch, model, ids, row, a, dev)
                rec.update(task=task, transcoders_used=info.get("transcoders_used"), label=(
                    "TEST FIXTURE" if a.fixture else "AS PUBLISHED (circuit-tracer %s, %s, %s)"
                    % (info["provenance"].get("commit", "?")[:8], a.dtype, a.hf_model or a.model)))
                write_atomic(f, rec)
                n_done += 1
                p8 = next(p for p in rec["prunes"] if p["node_threshold"] == 0.8) if 0.8 in a.node_thresholds \
                    else rec["prunes"][-1]
                print("%s p%d: %d tok, %d active, attr %.1fs prune %.1fs peak %s MB | top %r p=%.3f | good rank %s | "
                      "rep %.3f comp %.3f | @%.2f: %d feat nodes, %d latents, %d err | formula diff %.1e" % (
                          task, row["id"], rec["n_pos"], rec["n_active"], rec["t_attribute_s"], rec["t_prune_s"],
                          rec["peak_mb"], rec["logit_targets"][0]["token"], rec["logit_targets"][0]["prob"],
                          rec["answers"]["good"]["rank"], rec["replacement"], rec["completeness"],
                          p8["node_threshold"], p8["n_feature_nodes"], p8["n_latents"], p8["n_error_nodes"],
                          rec["scores_formula_check"]["max_abs_diff"]), flush=True)
                if rec["scores_formula_check"]["max_abs_diff"] > 1e-4:
                    raise RuntimeError("our unpruned score formula disagrees with compute_graph_scores")
    print("done: %d written, %d skipped (resumed)" % (n_done, n_skip))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("prompts", "run"))
    p.add_argument("--tasks", default=",".join(SVA_TASKS))
    p.add_argument("--split", default="held", choices=("held", "train", "all"))
    p.add_argument("--limit", type=int, default=0, help="first N prompts of the split (0 = all)")
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--hf-model", default=None, help="local HF weights via hf_model= (e.g. unsloth/gemma-2-2b)")
    p.add_argument("--transcoders", default=TC_REF)
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-feature-nodes", type=int, default=None, help="None = every active feature (as in §5.1)")
    p.add_argument("--offload", default=None, choices=(None, "cpu", "disk"))
    p.add_argument("--node-thresholds", type=lambda s: tuple(float(x) for x in s.split(",")), default=NODE_THRESHOLDS)
    p.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    p.add_argument("--fixture", default=None, choices=(None, "tiny"), help="[CPU dry run] their tiny test model")
    p.add_argument("--dry-run", action="store_true", help="print the plan; no model load")
    p.add_argument("--out", default=str(OUT_DEFAULT))
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    return cmd_prompts(a) if a.cmd == "prompts" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
