"""THEIRS side on THEIR turf: circuit-tracer attribution graphs on
Gemma-2-2B with its own default "gemma" transcoder scan, exporting the
upstream subgraph feeding a chosen seed feature.

Unlike the Llama comparison there is NO weight conversion and NO fold:
the shipped config reads TL's ln2.hook_normalized (unweighted), which
is exactly what the transcoders were trained on. Our side divides the
HF tensor by (1+w) to meet it (probe_gemma_tc.py); this side reads the
hook raw. check_agreement_gtc.py verifies both sides see the same
features before anything is compared.

Protocol and adjacency post-processing are identical to theirs_llama.py:
  * graphs on the seed's 48 TRAIN windows (MAX_PROMPTS to trim)
  * rows AND columns index graph.selected_features (measured on Llama:
    adjacency is square over selected nodes, not active ones)
  * sum |edge| from upstream-feature columns into the seed's rows,
    accumulate across positions/prompts, rank, export size-matched top-n
    plus the full ranking

Runs in ../../dev-notes/data/venv-ct. google/gemma-2-2b is gated, so weights come from
the unsloth mirror via hf_model=, as for Llama.

  TC_DIR=$HOME/gemma_tc MAX_FEATURE_NODES=16384 ../../dev-notes/data/venv-ct/bin/python theirs_gtc.py
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
MODEL_ID = "unsloth/gemma-2-2b"
TL_NAME = "google/gemma-2-2b"
DTYPE = torch.bfloat16
N_TRAIN = 48
MAX_PROMPTS = int(os.environ.get("MAX_PROMPTS", 48))
MAX_FEATURE_NODES = int(os.environ.get("MAX_FEATURE_NODES", 16384))
BATCH = int(os.environ.get("BATCH", 64))
LAZY_ENC = os.environ.get("LAZY_ENC", "1") == "1"
LAZY_DEC = os.environ.get("LAZY_DEC", "1") == "1"
MEM_FRAC = float(os.environ.get("MEM_FRAC", 0.90))
OUT = Path(os.environ.get("OUT", str(HERE / "theirs_gtc_pruned.jsonl")))
MAX_SEEDS = int(os.environ.get("MAX_SEEDS", 0))
# "gemma" = circuit-tracer's own alias for mwhanna/gemma-scope-transcoders.
# A local copy on native disk (TC_DIR) is used if present: lazy decoders
# re-read W_dec from disk on every access, and /mnt/x barely caches.
TC_REF = os.environ.get("TC_REF", "gemma")


def build_model():
    import torch as t
    from circuit_tracer.replacement_model import ReplacementModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if t.cuda.is_available() and MEM_FRAC < 1.0:
        t.cuda.set_per_process_memory_fraction(MEM_FRAC)
        tot = t.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print("VRAM cap: %.1f GB of %.1f GB" % (MEM_FRAC * tot, tot), flush=True)

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Lazy encoders are FREE on this side (no fold to apply in memory --
    # the shipped config reads TL's hook directly), and they remove the
    # largest resident block. Without them attribution OOMs under the cap
    # at every budget on Gemma-2 (resident set already at the cap).
    model = ReplacementModel.from_pretrained(
        TL_NAME, TC_REF, device=t.device("cuda"), dtype=DTYPE,
        hf_model=hf, tokenizer=tokenizer,
        lazy_encoder=LAZY_ENC, lazy_decoder=LAZY_DEC)
    print("resident after build: %.2f GB" % (t.cuda.memory_allocated() / 1024 ** 3),
          flush=True)
    model = model.to(DTYPE)
    mp = {p.dtype for p in model.parameters()}
    print("model dtypes %s | transcoders %s | hooks %s -> %s"
          % (sorted(str(d) for d in mp),
             type(model.transcoders).__name__,
             getattr(model, "feature_input_hook", "?"),
             getattr(model, "feature_output_hook", "?")), flush=True)
    return model


def main():
    from circuit_tracer import attribute

    import ct_prune

    scan = torch.load(HERE / "scan_gtc.pt", weights_only=False)
    toks, seeds = scan["tokens"], scan["seeds"]
    sizes = {}
    rows_path = HERE / "ours_gtc_rows.jsonl"
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
        n_graphs, consec, no_row = 0, 0, 0
        # circuit-tracer pruning, per window: as published (logit root)
        # and seed-rooted. Survivors are the seed's ANCESTORS in the
        # pruned edge graph. Counted across windows for the survival-
        # frequency ranking; union is the most generous single set.
        pub_count, root_count, pin_count = Counter(), Counter(), Counter()
        pub_seed_alive, root_seed_alive, pub_sizes, root_sizes = 0, 0, [], []
        pin_sizes, pin_empty = [], 0
        for i in range(prompts.shape[0]):
            ids = prompts[i].tolist()
            try:
                g = attribute(ids, model, max_feature_nodes=MAX_FEATURE_NODES,
                              batch_size=BATCH, verbose=False)
                consec = 0
            except Exception as e:
                consec += 1
                print("  prompt %d attribution failed: %s: %s"
                      % (i, type(e).__name__, str(e)[:120]), flush=True)
                if consec >= 3:
                    raise RuntimeError("3 consecutive failures: systematic") from e
                continue
            adj = g.adjacency_matrix
            af = g.active_features.to(adj.device)
            sel = g.selected_features.to(adj.device).flatten()
            af_sel = af[sel]
            n_sel = len(sel)
            tgt_rows = ((af_sel[:, 0] == L) & (af_sel[:, 2] == sl)).nonzero(
                as_tuple=True)[0]
            if not len(tgt_rows):
                no_row += 1
                continue
            src_ok = (af_sel[:, 0] < L).nonzero(as_tuple=True)[0]
            if not len(src_ok):
                continue
            if n_graphs == 0:
                print("  layout: adj %s | active %d | selected %d"
                      % (tuple(adj.shape), af.shape[0], n_sel), flush=True)
            # --- THEIR pruning (nothing of ours until seed_circuit) ---
            pr = ct_prune.prune_published(g)
            alive, mem, _ = ct_prune.seed_circuit(g, pr, L, sl)
            if alive:
                pub_seed_alive += 1
                pub_sizes.append(len(mem))
                pub_count.update(mem)
            # --- their body, root = the seed's rows (labelled adaptation) ---
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
                root_seed_alive += 1
                root_sizes.append(len(mem2))
                root_count.update(mem2)
            del pr, pr2
            contrib = adj[tgt_rows][:, :n_sel][:, src_ok].abs().sum(0)
            for j, c in zip(src_ok.tolist(), contrib.tolist()):
                if c:
                    weight[(int(af_sel[j, 0]), int(af_sel[j, 2]))] += c
            n_graphs += 1
            del g
            torch.cuda.empty_cache()
        ranked = sorted(weight.items(), key=lambda kv: -kv[1])
        n_ref = sizes.get((L, sl))
        rec = {"layer": L, "latent": sl, "n_graphs": n_graphs,
               "n_prompts_no_seed_row": no_row,
               "n_ranked": len(ranked), "n_ref": n_ref,
               "top_matched": [[l, f] for (l, f), _ in ranked[:n_ref]]
               if n_ref else None,
               "ranking": [[l, f, round(w, 6)] for (l, f), w in ranked[:20000]],
               # circuit-tracer AS PUBLISHED (prune_graph 0.8/0.98, logit root)
               "ct_published": {
                   "seed_alive_windows": pub_seed_alive,
                   "size_per_window": pub_sizes,
                   "union": [[l, f] for (l, f) in sorted(pub_count)],
                   "freq": [[l, f, c] for (l, f), c in pub_count.most_common()]},
               # their pruning body, rooted at the seed (adaptation)
               # as published + the seed PINNED (one-line blocker; node/edge
               # retention still logit-weighted)
               "ct_seed_pinned": {
                   "empty_windows": pin_empty,
                   "size_per_window": pin_sizes,
                   "union": [[l, f] for (l, f) in sorted(pin_count)],
                   "freq": [[l, f, c] for (l, f), c in pin_count.most_common()]},
               "ct_seed_rooted": {
                   "seed_alive_windows": root_seed_alive,
                   "size_per_window": root_sizes,
                   "union": [[l, f] for (l, f) in sorted(root_count)],
                   "freq": [[l, f, c] for (l, f), c in root_count.most_common()]}}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("[L%d %d] %d graphs (%d prompts had no seed row) | %d distinct "
              "upstream features | exported top-%s"
              % (L, sl, n_graphs, no_row, len(ranked), n_ref or "?"), flush=True)
        print("   ct_published: seed alive in %d/%d windows | circuit size/window "
              "med %s | union %d\n   ct_seed_rooted: alive %d/%d | size med %s | union %d"
              % (pub_seed_alive, n_graphs,
                 sorted(pub_sizes)[len(pub_sizes) // 2] if pub_sizes else None,
                 len(pub_count), root_seed_alive, n_graphs,
                 sorted(root_sizes)[len(root_sizes) // 2] if root_sizes else None,
                 len(root_count)), flush=True)
    fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
