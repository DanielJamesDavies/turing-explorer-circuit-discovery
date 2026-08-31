"""THEIRS side: circuit-tracer attribution graphs on Llama-3.2-1B with
the same converted TopK skip-transcoders, exporting the upstream
subgraph that feeds a chosen SEED FEATURE.

Runs in the isolated venv (../../dev-notes/data/venv-ct) because circuit-tracer pins
transformers<=4.57.3. It exchanges data with our side only through
JSON, never a shared process.

Why an internal root is well-posed here: circuit-tracer's Graph carries
a full adjacency matrix over [active_features, error, embed, logit]
nodes, where active_features are (layer, pos, feature_idx). So "what
feeds seed feature X" is simply X's ROW — no logit rooting needed, and
the question is the same one our mask answers.

Protocol (mirrors the train/test split on our side):
  * build graphs on the seed's 48 TRAIN windows
  * for each graph, find rows whose (layer, feature) match the seed,
    sum |edge weight| over source nodes that are upstream FEATURES
    (layer < seed layer), accumulating across positions and prompts
  * rank, and export the top-n where n is the seed's tri-amp size, so
    the two node sets are size-matched by construction
  * also export the full ranking, so compare.py can sweep size

  ../../dev-notes/data/venv-ct/bin/python theirs_llama.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
# Measured: /mnt/x (a Windows drive through WSL) reads at 216 MB/s cold
# and 279 MB/s warm -- it barely benefits from page cache. WSL-native
# ext4 reads cached at 28.5 GB/s. That gap matters because lazy decoders
# re-read W_dec slices from disk on EVERY access, so with the weights on
# /mnt/x every access pays full price. Point TC_DIR at a native-disk copy
# to make those reads essentially free.
TC_DIR = Path(os.environ.get("TC_DIR", str(HERE / "transcoders_llama_ct")))
MODEL_ID = "unsloth/Llama-3.2-1B"      # ungated mirror
TL_NAME = "meta-llama/Llama-3.2-1B"    # the name TransformerLens knows
N_LAYERS, K_SAE, D_TC = 16, 32, 131072
import torch as _t
DTYPE = _t.bfloat16   # circuit-tracer's own default; fp32 would need 34.5 GB
N_TRAIN = 48
MAX_PROMPTS = int(os.environ.get("MAX_PROMPTS", 48))
MAX_FEATURE_NODES = int(os.environ.get("MAX_FEATURE_NODES", 4096))
# Speed knobs. There is ~12 GB of headroom on the card, so both of these
# trade VRAM for wall-clock:
#   BATCH   how many backward passes attribute() runs at once (their
#           default is 512).
#   LAZY_DEC  with lazy decoders, W_dec slices are re-read FROM DISK on
#           every access -- and these files sit on /mnt/x, a Windows
#           drive through WSL, where I/O is slow. Eager decoders cost
#           ~537 MB/layer in bf16 (~8.6 GB for 16) but remove that.
BATCH = int(os.environ.get("BATCH", 512))
LAZY_DEC = os.environ.get("LAZY_DEC", "1") == "1"
# Lazy ENCODERS free the single largest resident block (~8.6 GB for 16
# layers in bf16), which is what a 16k-node graph needs in order to fit
# at all. This only became affordable once TC_DIR moved to ext4: it
# trades VRAM for reads that are now served from page cache.
LAZY_ENC = os.environ.get("LAZY_ENC", "0") == "1"
# HARD VRAM CAP. On Windows/WDDM the driver silently oversubscribes into
# "shared GPU memory" -- system RAM over PCIe -- instead of raising OOM.
# That is not extra capacity: it is ~5x slower, and it has bitten this
# project twice (once at 14.6 GB shared, once at 41 GB). Capping the
# allocator makes an over-large config fail FAST and loudly instead of
# quietly crawling, so timings stay trustworthy.
MEM_FRAC = float(os.environ.get("MEM_FRAC", 0.90))


def _prefolded(d):
    """True if TC_DIR holds weights with the RMS fold already baked in."""
    m = Path(d) / "meta.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text()).get("rms_fold") == "baked in"
    except Exception:
        return False


PREFOLDED = _prefolded(TC_DIR)
# Overridable so a smoke run writes elsewhere: main() treats seeds already
# present in OUT as done, so a 1-prompt smoke row written to the real file
# would silently make the real run skip that seed.
OUT = Path(os.environ.get("OUT", str(HERE / "theirs_llama_pruned.jsonl")))
MAX_SEEDS = int(os.environ.get("MAX_SEEDS", 0))   # 0 = all


def build_model():
    """Build circuit-tracer's ReplacementModel on the ungated Llama
    mirror, applying the RMSNorm-weight fold that this side needs.

    TWO corrections, both measured (check_hooks.py), not assumed:
      1. TransformerLens only knows the GATED meta-llama name, so the
         weights come from the ungated mirror via `hf_model=`.
      2. TransformerLens places `ln2.hook_normalized` BEFORE the RMSNorm
         weight, while these transcoders were trained on the HF MLP
         input, which includes it. Feeding the raw hook is wrong by a
         factor of 2.3-6.3x. Since x_full = x_hook * w, and
         x_full @ W_enc.T == x_hook @ (W_enc * w).T, we fold w into
         W_enc IN MEMORY for this side only. Our HF-based harness is fed
         the full input directly and needs no such fold, so the on-disk
         weights stay single-copy and canonical.
    """
    import torch as t
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.transcoder.activation_functions import TopK
    from circuit_tracer.transcoder.single_layer_transcoder import (
        SingleLayerTranscoder, TranscoderSet, load_transcoder)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # UPSTREAM GAP, patched here rather than worked around.
    #
    # TransformerLens's RMSNorm upcasts to float32 internally for
    # numerical stability, and `ln2.hook_normalized` -- our
    # feature_input_hook -- fires INSIDE that upcast. So the activation
    # circuit-tracer caches for the skip path is fp32 regardless of the
    # model's dtype (measured: model params and W_skip both bf16, yet
    # mat1 arrives as float).
    #
    # `encode()` already guards against this with
    # `input_acts.to(W_enc.dtype)`. `compute_skip()`, one method below,
    # does a bare `input_acts @ W_skip.T` -- so SKIP-transcoders die on
    # any non-fp32 model. We cannot run fp32 (16 fp32 encoders alone are
    # 17.2 GB, over this card), so we mirror encode()'s cast.
    #
    # Casting the input DOWN to W_skip's dtype (rather than W_skip up)
    # keeps the skip output in the model's dtype, matching the bf16
    # stream it is added into; casting up would leak fp32 into the
    # residual stream through `skip + (acts - skip).detach()`.
    def _compute_skip(self, input_acts):
        if self.W_skip is None:
            raise ValueError("Transcoder has no skip connection")
        return input_acts.to(self.W_skip.dtype) @ self.W_skip.T

    SingleLayerTranscoder.compute_skip = _compute_skip

    # VRAM: 16 transcoders at fp32 is 34.5 GB of encoder+decoder, far over
    # a 16 GB card. bfloat16 (circuit-tracer's own default for
    # SingleLayerTranscoder) plus lazy decoders puts the resident set at
    # ~8.6 GB of encoders + ~2.5 GB of model. The RMS fold below is applied
    # in fp32 before the cast so it does not inherit bf16 rounding.
    if t.cuda.is_available() and MEM_FRAC < 1.0:
        t.cuda.set_per_process_memory_fraction(MEM_FRAC)
        tot = t.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("VRAM cap: %.1f GB of %.1f GB (spill into shared memory "
              "raises OOM instead)" % (MEM_FRAC * tot, tot), flush=True)

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=t.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    transcoders = {}
    for L in range(N_LAYERS):
        tc = load_transcoder(
            str(TC_DIR / ("layer_%d.safetensors" % L)), layer=L,
            activation_fn=TopK(K_SAE), device=t.device("cuda"),
            dtype=t.float32, lazy_encoder=LAZY_ENC, lazy_decoder=LAZY_DEC)
        # Whether to fold is a property of the WEIGHTS, not of a flag:
        # prefold_weights.py bakes the fold into W_enc on disk (which is
        # what makes lazy encoders possible, since the lazy path reads
        # W_enc straight from the file and cannot be folded in memory).
        # Reading it from meta.json means pre-folded weights can never be
        # double-folded, nor unfolded weights silently left unfolded.
        if not PREFOLDED:
            assert not LAZY_ENC, (
                "lazy encoders need pre-folded weights (run "
                "prefold_weights.py); the RMS fold cannot be applied to a "
                "lazily-read W_enc")
            w = hf.model.layers[L].post_attention_layernorm.weight.data
            with t.no_grad():
                tc.W_enc.mul_(w.to(tc.W_enc.device, t.float32))
        transcoders[L] = tc.to(DTYPE)
        t.cuda.empty_cache()
    # The HF model is loaded fp32 ON PURPOSE so the RMS fold above does not
    # inherit bf16 rounding, but it is also the weight source for the bf16
    # ReplacementModel. Left fp32 it feeds fp32 activations into bf16 W_enc
    # and every attribution dies with "float != c10::BFloat16". Cast now
    # that the fold has been taken in full precision.
    hf = hf.to(DTYPE)
    tset = TranscoderSet(
        transcoders,
        feature_input_hook="ln2.hook_normalized",
        feature_output_hook="hook_mlp_out",
        scan_name="eleuther-llama32-1b-131k-topk")
    model = ReplacementModel.from_pretrained_and_transcoders(
        TL_NAME, tset, device=t.device("cuda"), dtype=DTYPE,
        hf_model=hf, tokenizer=tokenizer)

    # Passing dtype= through to TransformerLens is NOT sufficient here: it
    # arrives as **kwargs on a from_pretrained whose default is float32,
    # and the model still came out fp32. That matters because
    # circuit-tracer's SKIP path does `input_acts @ W_skip.T` with NO cast
    # (unlike encode(), which does `input_acts.to(W_enc.dtype)`), so an
    # fp32 model against bf16 skip-transcoders kills every attribution
    # with "float != c10::BFloat16". Cast the built model and assert the
    # two sides agree, rather than trusting the constructor argument.
    model = model.to(DTYPE)
    mp = {p.dtype for p in model.parameters()}
    tp = {transcoders[L].W_skip.dtype for L in transcoders}
    print("dtype check | model %s | transcoder W_skip %s"
          % (sorted(str(d) for d in mp), sorted(str(d) for d in tp)),
          flush=True)
    assert mp == {DTYPE}, "model dtypes %s, expected %s" % (mp, DTYPE)
    assert tp == {DTYPE}, "W_skip dtypes %s, expected %s" % (tp, DTYPE)
    return model


def main():
    # NOT `from circuit_tracer.attribution import attribute`: the package
    # uses lazy TYPE_CHECKING exports, so that path silently binds the
    # SUBMODULE circuit_tracer.attribution, and every call then fails with
    # "'module' object is not callable". The top-level name is the function.
    from circuit_tracer import attribute

    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / '038-transcoder-compare-gemma'))
    import ct_prune   # their pruning, called as their code (see ct_prune.py)

    scan = torch.load(HERE / "scan_llama.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    sizes = {}
    rows_path = HERE / "ours_llama_rows.jsonl"
    if rows_path.exists():
        for line in rows_path.open():
            r = json.loads(line)
            if r.get("arm") == "triamp400":
                sizes[(r["layer"], r["latent"])] = r["n"]

    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["layer"], r["latent"]))

    model = build_model()
    fh = OUT.open("a")
    keys = sorted(seeds)
    if MAX_SEEDS:
        keys = keys[:MAX_SEEDS]
    for key in keys:
        S = seeds[key]
        L, sl = S["layer"], S["latent"]
        if (L, sl) in done:
            print("[L%d %d] already exported" % (L, sl), flush=True)
            continue
        prompts = toks[S["pos_windows"]][:N_TRAIN][:MAX_PROMPTS]
        weight = defaultdict(float)
        n_graphs = 0
        pub_count, root_count, pin_count = Counter(), Counter(), Counter()
        pub_seed_alive, root_seed_alive, pub_sizes, root_sizes = 0, 0, [], []
        pin_sizes, pin_empty = [], 0
        # A per-prompt except is right for a genuinely flaky prompt, but a
        # SYSTEMATIC fault (bad import, OOM, wrong dtype) would otherwise
        # fail all 48 identically and still write a row and print ALL DONE
        # -- an empty result wearing the costume of a finished run. That
        # happened once here. Consecutive failures now abort loudly.
        consec = 0
        for i in range(prompts.shape[0]):
            ids = prompts[i].tolist()
            try:
                g = attribute(ids, model, max_feature_nodes=MAX_FEATURE_NODES,
                              batch_size=BATCH, verbose=False)
                consec = 0
            except Exception as e:
                consec += 1
                print("  prompt %d attribution failed: %s: %s"
                      % (i, type(e).__name__, e), flush=True)
                if consec >= 3:
                    raise RuntimeError(
                        "3 consecutive attribution failures -- systematic, "
                        "not flaky. Aborting rather than writing empty "
                        "rows.") from e
                continue
            adj = g.adjacency_matrix          # rows target, cols source
            # The Graph can hold its adjacency matrix and its feature
            # metadata on DIFFERENT devices (the adjacency matrix is the
            # big one and gets offloaded to CPU, while active_features
            # stays on the GPU), so indices derived from one cannot index
            # the other. Pin both to the adjacency matrix's device.
            af = g.active_features.to(adj.device)   # [n,3] (layer,pos,feat)
            n_feat = af.shape[0]
            # ROWS AND COLUMNS DO NOT SHARE AN INDEX SPACE.
            #
            # Columns span every node: [active_features, error, embed,
            # logit], so a column index < n_feat indexes active_features
            # directly. But when max_feature_nodes is set, only the
            # SELECTED features get rows -- graph.selected_features holds
            # "indices into active_features for selected nodes", and the
            # row block is those, in that order.
            #
            # Verified arithmetically on the first graph here: 4096
            # selected + 16*64 error + 64 embed + 10 logit = 5194 rows,
            # against 8092 active features. Indexing rows with
            # active_features indices (the obvious reading) silently
            # points at the wrong feature whenever it does not simply go
            # out of bounds, so this distinction is load-bearing.
            sel = g.selected_features.to(adj.device).flatten()
            af_sel = af[sel]
            n_sel = len(sel)
            tgt_rows = ((af_sel[:, 0] == L) & (af_sel[:, 2] == sl)).nonzero(
                as_tuple=True)[0]
            if not len(tgt_rows):
                continue
            # Source columns are the SELECTED features too, not all active
            # ones: measured adj is square at (5194, 5194) = 4096 selected
            # + 16*64 error + 64 embed + 10 logit, against 32,256 ACTIVE
            # features. So both axes are indexed through selected_features,
            # and af_sel -- never af -- is the right table for both.
            src_ok = (af_sel[:, 0] < L).nonzero(as_tuple=True)[0]
            if not len(src_ok):
                continue
            if n_graphs == 0:
                print("  layout: adj %s | active %d | selected %d"
                      % (tuple(adj.shape), n_feat, n_sel), flush=True)
            assert adj.shape[1] >= n_sel, (
                "unexpected adjacency layout: adj %s, selected %d"
                % (tuple(adj.shape), n_sel))
            pr = ct_prune.prune_published(g)
            alive, mem, _ = ct_prune.seed_circuit(g, pr, L, sl)
            if alive:
                pub_seed_alive += 1; pub_sizes.append(len(mem)); pub_count.update(mem)
            rows_s, _ = ct_prune.seed_rows(g, L, sl)
            # Daniel's blocker: as-published pruning, seed pinned only
            pr3 = ct_prune.prune_pinned(g, rows_s)
            _, mem3, _ = ct_prune.seed_circuit(g, pr3, L, sl)
            pin_sizes.append(len(mem3))
            if mem3: pin_count.update(mem3)
            else: pin_empty += 1
            del pr3
            pr2 = ct_prune.prune_rooted(g, rows_s)
            alive2, mem2, _ = ct_prune.seed_circuit(g, pr2, L, sl)
            if alive2:
                root_seed_alive += 1; root_sizes.append(len(mem2)); root_count.update(mem2)
            del pr, pr2
            contrib = adj[tgt_rows][:, :n_sel][:, src_ok].abs().sum(0)
            for j, c in zip(src_ok.tolist(), contrib.tolist()):
                if c:
                    weight[(int(af_sel[j, 0]), int(af_sel[j, 2]))] += c
            n_graphs += 1
        ranked = sorted(weight.items(), key=lambda kv: -kv[1])
        n_ref = sizes.get((L, sl))
        rec = {"layer": L, "latent": sl, "n_graphs": n_graphs,
               "n_ranked": len(ranked), "n_ref": n_ref,
               "top_matched": [[l, f] for (l, f), _ in ranked[:n_ref]]
               if n_ref else None,
               "ranking": [[l, f, round(w, 6)] for (l, f), w in
                           ranked[:20000]],
               "ct_published": {"seed_alive_windows": pub_seed_alive,
                                "size_per_window": pub_sizes,
                                "union": [[l, f] for (l, f) in sorted(pub_count)],
                                "freq": [[l, f, c] for (l, f), c in pub_count.most_common()]},
               # as published + the seed PINNED (one-line blocker; node/edge
               # retention still logit-weighted)
               "ct_seed_pinned": {
                   "empty_windows": pin_empty,
                   "size_per_window": pin_sizes,
                   "union": [[l, f] for (l, f) in sorted(pin_count)],
                   "freq": [[l, f, c] for (l, f), c in pin_count.most_common()]},
               "ct_seed_rooted": {"seed_alive_windows": root_seed_alive,
                                  "size_per_window": root_sizes,
                                  "union": [[l, f] for (l, f) in sorted(root_count)],
                                  "freq": [[l, f, c] for (l, f), c in root_count.most_common()]}}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("[L%d %d] %d graphs | %d distinct upstream features | "
              "exported top-%s" % (L, sl, n_graphs, len(ranked),
                                   n_ref if n_ref else "?"), flush=True)
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
