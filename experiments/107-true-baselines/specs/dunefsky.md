# Spec: Dunefsky, Chlenski & Nanda 2024, transcoder circuit finding (arena A)

Status: specification for running the official code **unmodified** as a baseline. Written 2026-11-01 (agent
`spec-dunefsky`). Nothing here is built or run.

**Paper.** Jacob Dunefsky, Philippe Chlenski, Neel Nanda, "Transcoders Find Interpretable LLM Feature Circuits",
arXiv 2406.11944, NeurIPS 2024.
- v1: 17 Jun 2024. v2: 6 Nov 2024, current. Read: https://arxiv.org/html/2406.11944v2 (all section, equation and
  appendix numbers below are v2).

**Code.** https://github.com/jacobdunefsky/transcoder_circuits, branch `master`, 15 commits.
- HEAD = `8721ff06b64889e8cc50bb1c5e869c7f72b8303d` (2024-11-17, README edit only).
- The last code change is `cb47944e6b3fd1f0471fb93f3d7316a1deed3ff7` / `6353aed474b9a8b8aa5ffd4ced1af80dbfbd7a5f`
  (2024-06-17, "Add files from more recent experiments"). The circuit code at HEAD is that code.
- Files read at HEAD:
  - `transcoder_circuits/circuit_analysis.py` (927 lines): the whole circuit finder;
  - `transcoder_circuits/replacement_ctx.py`;
  - `transcoder_circuits/feature_dashboards.py` (de-embeddings and pullbacks only);
  - `sae_training/sparse_autoencoder.py` (forward and loader);
  - `train_transcoder.py`, `setup.sh`, `requirements.txt`, `README.md`;
  - notebooks: `walkthrough.ipynb` (all cells and outputs), `greaterthan.ipynb` (all), `case_study_caught.ipynb`
    (parts), and the `greedy_get_top_paths` calls in every notebook.
- Read through the GitHub API to stdout. Nothing was cloned or saved.

**Weights.** HF `pchlenski/gpt2-transcoders`, revision `4f3ca345572ce2d75018a60d10b4dca8bd899e99` (2024-04-25).

**Tag key.**
- [paper §x / Eq. y / App. z] is the paper.
- [code file:line] is `circuit_analysis.py` at `8721ff0` unless another file is named. Line numbers are exact (from
  `cat -n` of the raw file).
- [nb name cell N] is a notebook cell at `8721ff0`. Its printed output is the authors' saved output.
- [inferred] is my reading.
- [ours] is a protocol choice for 107, not theirs.

**Caveat on the paper reading.** The paper was read through WebFetch, which returns paraphrase. Equations below are
written out from that paraphrase plus the code. Where code and paper differ, the code is authoritative for "run
unmodified".

---

## 1. Summary and claims (circuit finding only)

**The idea** [paper §3.2]:
- Transcoders replace each MLP with a sparse, wide ReLU layer that reads the MLP input and predicts its output
  (Eq. 3–5).
- The attribution of an earlier transcoder feature to a later one then factorises (Eq. 7) into:
  - an **input-dependent** term: the earlier feature's activation on this token;
  - an **input-invariant** term: the earlier decoder vector dotted with the later encoder vector, from weights only.
- With the MLP replaced, the path from one feature to another is linear except for the attention pattern and the
  LayerNorm scale. Both are frozen at their values on this prompt [paper §3.2.2, App. D.3–D.4].

**The circuit finder** [paper §3.2.3, Fig. 2, App. D.5 Alg. 1–2]:
- Start from one transcoder feature at one token on one prompt.
- Repeatedly expand each kept path by its top contributors: lower-layer transcoder features at the same token,
  attention heads from earlier tokens, and the token embedding.
- Keep the top N paths per step, for L steps.
- Merge all kept paths into a graph, with node, edge and path attributions. Optionally add error nodes, following
  Marks et al.

**Claims about circuits** [paper §1, §3.2, §5]:
- Input-invariant connections plus sparse input-dependent ones give interpretable circuits.
- This is shown only through case studies:
  - blind reverse-engineering of random GPT-2 layer-8 features (§5.1, App. H);
  - the GPT-2 "greater-than" circuit (§5.2).
- **No quantitative faithfulness, size or held-out evaluation of the circuits the path finder returns.** The one
  intervention is a top-k-features vs top-k-neurons ablation curve for a single layer (MLP10) on greater-than
  (§5.2.2). It evaluates the transcoder basis, not the path finder.

**Out of scope here:** transcoder training (§3.1, App. E) except what the substrate needs; the SAE comparisons (§4);
the Pythia transcoders (§4.2, App. F).

---

## 2. Substrate

### 2.1 Model

- GPT-2 small loaded as `HookedTransformer.from_pretrained('gpt2')`, with TransformerLens defaults [nb walkthrough
  cell 7; nb greaterthan cell 3].
- The defaults are `fold_ln=True`, `center_writing_weights=True`, `center_unembed=True` [inferred from TL
  defaults].
- Pinned environment [requirements.txt]: `transformer_lens==1.11.0`, `torch==2.2.0`, `huggingface-hub==0.17.3`,
  `datasets==2.12.0`, `einops==0.7.0`.

### 2.2 Transcoders

| item | value | source |
|---|---|---|
| HF repo | `pchlenski/gpt2-transcoders` @ `4f3ca34` | README; setup.sh:18 |
| files | `final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt`, L = 0..11, each with `_log_feature_sparsity.pt`. Also `lower_l0/…blocks.11…` (an alternative layer-11 transcoder) | HF file listing |
| download | `snapshot_download(allow_patterns=["*.pt"])` into `./gpt-2-small-transcoders`. The pattern also pulls `lower_l0/` [inferred: fnmatch `*` matches `/`] | setup.sh:16-21 |
| sites | input `blocks.L.ln2.hook_normalized` (post-LN MLP input, pure normalisation under fold_ln); output target `blocks.L.hook_mlp_out` | file names; train_transcoder.py:37,55; 035 README (fold_ln convention measured); [paper App. E] "post-LayerNorm" input |
| width | 24,576 = 32 × 768 | [paper App. E]; file names |
| activation | ReLU: `relu((x − b_dec) @ W_enc + b_enc)`, out = `acts @ W_dec + b_dec_out` | sparse_autoencoder.py:110-134 |
| return value | a 6-tuple: `(sae_out, feature_acts, loss, mse, l1, ghost)`. The circuit code uses `[0]` (reconstruction) and `[1]` (activations) | sparse_autoencoder.py:181 |
| L1 | 8e-5 per the saved cfg (our reading in 035 / ge.md). The example script uses 1.4e-4 and lr 4e-4. Paper App. E gives lr 2e-5, L1 "by trial and error" | train_transcoder.py:25-26; [paper App. E]; 035 |
| training data | OpenWebText, 60M tokens, context 128, batch 4,096, Adam | [paper App. E]; train_transcoder.py:40,67-73 |
| loader | `SparseAutoencoder.load_from_pretrained(path)`. It is a pickled `{cfg, state_dict}` that needs the repo's `sae_training` package and torch's old `weights_only=False` default (torch 2.2 OK; torch ≥ 2.6 changes the default) | sparse_autoencoder.py:607-645; [inferred re torch versions] |
| layers used | walkthrough loads **layers 0–10 only** (`range(11)`). Case studies and greater-than load all 12 | nb walkthrough cell 14; nb greaterthan cell 7; nb case_study_* cell 11 |

**LayerNorm handling.**
- The transcoder reads `ln2.hook_normalized`. The circuit code treats that site as `resid_mid` and converts with a
  per-token scalar "LN constant" [code 240-245, 109-122].
- The constant is `dot(v, y) / dot(v, x)`, with x = `resid_mid` (or `resid_pre`) and y = `ln2` (or `ln1`)
  `hook_normalized` at that token [code 113-122].
- It is the **directional** ratio along the vector being pulled back. It returns 0 if `dot(v, x) = 0`.
- The paper describes this as an empirically estimated LN scaling constant, and says its pseudocode omits it
  [paper App. D.4, D.5]. Whether the paper calls it a ratio of norms or a directional ratio is unclear from my read.
- The final LayerNorm is not handled. The greater-than root is a raw `W_U` direction at `resid_post` 11 [nb
  greaterthan cell 12].

### 2.3 Attention: raw heads through OV, not attention SAEs

This is the most important substrate fact for arena A.

- **No attention SAEs.** Attention is attributed through the **raw per-head OV circuit with the clean attention
  pattern frozen** [paper §3.2.2 Eq. 8, App. D.3 Eq. 12, 22–24, 27; code 12-34, 330-364].
- **Node.** An attention node is `(layer L, head h, source token s)`, printed `attn{L}[{h}]@{s}` [code 166-182,
  350-355].
- **Contribution of head h from source s to a vector v at destination t** [code 13-34]:

      contrib(L,h,s → v@t) = A^{L,h}[t, s] · ( v_cache^{L}[s, h, :] @ W_O^{L,h} ) · v

  - `v_cache` is `hook_v`, so it includes b_V. The pattern is the clean `hook_pattern`.
  - The paper writes this as score × ((W_OV)ᵀ f_enc) · x_pre [paper Eq. 8].
- **Vector passed upstream** [code 343-347; paper Eq. 27]:

      v' = A^{L,h}[t, s] · (W_V W_O)^{L,h} v · LNconst_ln1(L, s, ·)

  The new vector lives in `resid_pre` of layer L at token s.
- **QK is not analysed.** Patterns are constants [paper §3.2.2, §7 Limitations].
- **b_O is never attributed.** Its share goes into the pruning error [inferred from code 13-34; no b_O term
  anywhere].
- **BOS is excluded as a source** when `ignore_bos=True` (the default) [code 335-342]. Attention-to-BOS mass is never a
  candidate and ends up in the pruning error [inferred].
- **Per-layer top-k for heads is capped at 12.** The head candidates per layer are `topk(k = min(k,
  len(attn_contribs)))`, where `attn_contribs` has shape [n_heads, n_src], so `len` = 12 = n_heads [code 337].
  - With `num_branches` > 12, at most 12 head candidates per layer enter the global top-k.
  - Transcoder candidates have no such cap [inferred; looks unintended].

### 2.4 Embedding

- One `EMBED` node per token position, `embed0@t`.
- Its contribution is `resid_pre_0[t] · v`, i.e. **token embedding + positional embedding together** [code 366-379].
- It is a leaf. Paths reaching `layer 0, resid_pre` are never expanded [code 418].

### 2.5 Error terms (added after the search, optional) [code 684-792; nb walkthrough cells 71-73; paper App. D.5]

For every node that has children in the graph:

| node | definition | units |
|---|---|---|
| `tc_error` | Σ_{layers l < max_layer} (mlp_out_l − TC_l(x_l))[token] · node.vector [code 707-713] | root-attribution units (the vector carries the path gradient) |
| `bias_error` | transcoder parents only: `b_enc[i] − W_enc[:, i] · b_dec` [code 742-744] | **the parent feature's own pre-activation units, not scaled by the path gradient**. Only for the root are these the same. Walkthrough: non-root bias errors are −7 to −33 against node contributions of ≈ 0.1–1.4 [nb walkthrough cell 73] [inferred: unit inconsistency] |
| `prune_error` | node.contrib − Σ incoming edge contribs [code 769-774] | root-attribution units |

- Error nodes are bookkeeping. They are never searched and never expanded.
- For a WCM comparison they are **not nodes of the circuit** (§7.2).

### 2.6 Node types in their graphs

| type | key string | identity | source |
|---|---|---|---|
| root | `mlp{L}tc[{i}]@{t}` (t may be −1) | transcoder feature at the root token | code 231-270 |
| transcoder feature | `mlp{l}tc[{j}]@{t}` | (layer, feature, token). Always the same token as the node it feeds, because MLPs are position-local | code 298-313 |
| attention head | `attn{l}[{h}]@{s}` | (layer, head, **source** token). The destination token is **not** part of the identity, so paths through the same head and source into different destinations merge into one node | code 350-364, 667-680 |
| embedding | `embed0@{t}` | token + position embedding at t | code 366-379 |
| error | `…tc_error{l}@t`, `…bias_error{l}@t`, `…prune_error{l}@t` | one of each per parent | code 714-790 |

`sum_over_tokens(edges, nodes)` collapses token positions in the strings, so `mlp3tc[7]@4` and `mlp3tc[7]@9` become
one node [code 794-809]. It is a provided utility, not used in the walkthrough.

---

## 3. The algorithm, exactly

### 3.1 Objects

- **FeatureVector** = (component_path, vector v ∈ R^768, layer, sublayer ∈ {resid_pre, resid_mid, resid_post,
  mlp_out}, token, contrib) [code 189-229].
- `contrib` is the attribution of **the whole path ending here to the root activation**. It is not an edge-local
  weight [inferred from code 285-313 and walkthrough cells 60, 68: e.g. the edge `(attn0[5]@9, mlp0tc[8829]@9)` =
  1.097 = the sum of path contribs 0.66 + 0.15 + 0.10 + 0.093 + 0.085].
- `vector` is the gradient of the root w.r.t. the residual site, along this path: the product of all
  input-invariant factors, patterns and LN constants so far.

### 3.2 Root [code 231-270, 403-413]

```
fv = make_sae_feature_vector(tc[L], i, use_encoder=True, token=t)
     # v = W_enc[:, i] (encoder direction, in ln2-normalised space); sublayer = 'resid_mid'; layer = L
root.v      ← v · LNconst_ln2(L, t, v)                     # convert to resid_mid space   [code 409-411]
root.contrib ← tc[L](cache[ln2.hook_normalized L])[1][0, t, i]   # the post-ReLU activation    [code 412]
```

- The root's ReLU and bias are ignored. The method assumes the root is active [paper App. D.2].
- The bias reappears only as `bias_error`.
- The root can be any resid-space vector. Greater-than uses `W_U[:, '40'] − W_U[:, '20']` at `resid_post` 11 [nb
  greaterthan cell 12]. Native latent roots are **transcoder features only**.

### 3.3 Input-dependent attribution: candidate contributions to a FeatureVector (v, layer L, sublayer, token t) [code 316-396]

Every candidate's contribution is (its output on this prompt) · v. That is, input × gradient along the path [paper
App. D.4 fn. 3].

```
(a) transcoder features, for every layer l < L  (l ≤ L if sublayer == resid_post), at the SAME token t:
      z_l = tc[l](cache[ln2.hook_normalized l])[1][0, t, :]                        # activations, real-model cache
      c_j = z_l[j] · (W_dec^l[j] · v)                                              # Eq. 7 / Eq. 18
      keep top-k_branches over j                                                    [code 286-288]
      child vector:  v_j = W_enc^l[:, j] · (W_dec^l[j] · v) · LNconst_ln2(l, t, ·) # Eq. 28        [code 292-296]
(b) attention heads, for every layer l < L  (l ≤ L if sublayer ∈ {resid_mid, resid_post}), all heads h,
    all sources s ≥ 1 (BOS dropped):
      c_{h,s} = A^{l,h}[t,s] · (v_cache^l[s,h] W_O^{l,h}) · v                       # Eq. 8
      keep top-min(k, 12) over (h, s) per layer                                    [code 337]
      child vector:  A^{l,h}[t,s] · (W_V W_O)^{l,h} v · LNconst_ln1(l, s, ·), at token s, sublayer resid_pre(l)
(c) embedding at t:  c = resid_pre_0[t] · v                                         [code 377]
pool (a) + (b) + (c); optional `filter`; optional `cap` (§3.5); return the global top-k by SIGNED c   [code 392-396]
```

- **Signed ranking.** `torch.topk` on raw signed contributions [code 288, 337, 395, 431]. Negative (inhibitory)
  contributors are never followed. Their circuits are **excitatory-only** [inferred].
- **The input-dependent term** is `z_l[j]` (and `A·v_cache` for heads).
- **The input-invariant term** is `W_dec^l[j] · v` (for v = the root encoder, this is f_dec · f_enc of Eq. 7). For
  heads, it is `(W_OV)ᵀ v`.
- **Same-token rule for MLPs.** Transcoder features are only searched at the token of the vector being explained.
  Cross-token influence enters only through head nodes [code 327, 286-287].

### 3.4 Input-invariant attribution (not used to build circuits)

Used for interpretation and as a comparison, never in the search:

| quantity | formula | code |
|---|---|---|
| pullback (input-invariant connection to an earlier transcoder) | W_dec^l · v (one score per feature j) [paper Eq. 7 second factor; walkthrough cell 99] | code 38, 62; feature_dashboards.py:305 |
| connection through a head | (f_dec^{l,j})ᵀ (W_OV^{L',h})ᵀ f_enc^{L,i} [paper §5.2, App. G] | nb greaterthan cell 35: `tc0.W_dec @ model.OV[9,1] @ vec` |
| de-embedding | W_E f_enc^{l,i} over the vocabulary, optionally through a head's OV [paper §3.2.4] | feature_dashboards.py:427-432, 449-451 |
| mean input-dependent connection over a dataset | mean of z_l ⊙ (W_dec^l v) over activating tokens (the 20th percentile and above) | code 55-107; nb walkthrough cells 103-114 |

- The walkthrough's overlap of top-k pullback vs top-k mean input-dependent features is 30 / 25 / 14% at k = 10 / 20
  / 50 for mlp8tc[89] ← MLP2.
- For mlp5tc[12450] ← MLP0 it is 60 / 50 / 44% [nb walkthrough cells 108, 114].
- These are the only cross-prompt numbers in the repo.

### 3.5 Greedy path search: `greedy_get_top_paths` [code 398-434; paper App. D.5 Alg. 1]

```
greedy_get_top_paths(model, transcoders, cache, feature_vector,
                     num_iters=2, num_branches=5, ignore_bos=True, do_raw_attribution=False, filter=None)

root ← §3.2
cur_paths ← [[root]] ; all_paths ← []
for it in 1..num_iters:                                   # L in the paper
    new_paths ← []
    for path in cur_paths:
        end ← path[-1]
        if end.layer == 0 and end.sublayer == 'resid_pre': continue      # embed or attn0 leaf: not expanded
        cap ← None
        if not do_raw_attribution:                                       # "cap" mode, the DEFAULT
            for f in path:
                if last component of f is TRANSCODER  or  (SAE and (cap is None or f.contrib < cap)):
                    cap ← f.contrib                                      # see A1: precedence
        children ← get_top_contribs(end, k=num_branches, cap=cap, filter=filter)   # §3.3, top-num_branches
        new_paths += [path + [c] for c in children]
    cur_paths ← top-num_branches of new_paths by new_path[-1].contrib     # beam, signed
    all_paths.append(cur_paths)
return all_paths            # list of num_iters beams; beam i holds ≤ num_branches paths of length i+2
```

- **Beam.** Width = `num_branches` (N in the paper), depth = `num_iters` (L in the paper).
  - Each step proposes up to N children per path, i.e. up to N² candidates.
  - It keeps the N best path-ends overall [code 429-432].
- **Paper wording.** "Keep the top N by attribution, then filter by the earliest-layer node" [paper App. D.5 Alg. 1].
  In the code these are the same operation: rank paths by their newest node's contrib [inferred].
- **Finished paths.** Paths that end at a layer-0 leaf drop out of the beam, but stay in `all_paths` at their
  length [code 418, 433].
- **No thresholds.** There are no attribution thresholds anywhere. Size is controlled only by (num_iters,
  num_branches).
- **Size bound.** Distinct graph nodes ≤ 1 + num_iters · num_branches. That is ≤ 76 for 5 × 15 and ≤ 241 for 6 × 40
  [inferred from the beam].
- **Cap mode** (`do_raw_attribution=False`, the default) [code 420-428, 387-391]:
  - each candidate contrib above `cap` is clipped to `cap` and re-typed `ZERO_ABLATION`;
  - the comment says this emulates zero-ablating the upstream feature;
  - the walkthrough recommends `do_raw_attribution=True` for "better mathematical properties" [nb walkthrough
    cell 58];
  - `paths_to_graph` and `get_raw_top_features_among_paths` **assert** every contrib is `RAW` [code 466, 658], so
    graph building fails if any contrib was clipped.

### 3.6 Merging paths into a graph: `paths_to_graph` [code 627-682; paper App. D.5 Alg. 2]

```
for path in flatten(all_paths):
    for j, prefix in enumerate(prefixes(path)):
        if str(prefix) seen: continue                      # de-dup identical prefixes (shared beam history)
        if j == 0: nodes[root].contrib += root.contrib (first time: copy)
        else:
            parent, child = path[j-1], path[j]
            edges[(str(child), str(parent))] += child.contrib     # child.contrib = attribution of the PATH to root
            nodes[str(child)].contrib += child.contrib ; nodes[str(child)].vector += child.vector
attention nodes: component_path trimmed to the head itself (merges destinations)   [code 670-680]
return edges, nodes
```

- **Node keys.** These are `__str__(show_full=False)`: the last component plus token. So the same feature at the
  same token, reached by different paths, is **one** node.
- **Its attribution** is the sum of the path attributions of all distinct path prefixes ending at it [code 655-665].
- **Edge key** = (child, parent), with value = the sum over paths [code 659].
- **Identity.** With zero transcoder error, a node's attribution would equal the sum of its incoming edges [paper
  App. D.5]. In code this is not enforced; `prune_error` records the gap.

### 3.7 Root, prompts, and aggregation

- **One circuit = one (prompt, token, root feature).** The cache is from one forward pass on one prompt [code 398;
  nb walkthrough cell 57].
  - In practice the prompt is **truncated to end at the target token**, with `token=-1` (e.g.
    `owt_tokens[8531, :111+1]`) [nb case_study_caught cells 24-26].
  - The examples are activating OpenWebText windows (context 128) above a threshold:
    - 10–16 in the paper [paper App. H];
    - 17.0 in `case_study_caught` cell 23.
- **No aggregation across prompts** in the circuit finder.
  - The "input-invariant" part is prompt-independent by construction, but it is used only for interpretation
    (§3.4).
  - The only dataset-level averages are `get_mean_ixg` (§3.4) and the greater-than variance ranking (§5).
- **Across positions:** `sum_over_tokens` (optional) [code 794-809].

---

## 4. Hyperparameters, paper vs code

| knob | paper | code default | values actually used in the repo |
|---|---|---|---|
| num_iters (path length L) | symbol L, value not given [paper App. D.5; §3.2.3] | 2 [code 399] | walkthrough 3; greater-than 3; case studies (citations, caught, local context) 5; restricted blind 5–6 (mostly 6); one-step subcircuits 1 |
| num_branches (beam N) | symbol N (Fig. 2 "k"), value not given | 5 [code 399] | walkthrough 15; greater-than 15; case studies 15 (20, 30, 60 in places); restricted blind 15–40 (mostly 20) [grep of all notebooks] |
| per-layer candidates | not stated | k = num_branches for transcoders; min(k, 12) for heads [code 288, 337] | same |
| do_raw_attribution | cap not described in the text I read | **False** (cap on) [code 399] | walkthrough and greater-than: True; most case-study calls: default False; caught cells 59-89: True |
| ignore_bos | not stated | True [code 317, 399] | default everywhere |
| filter | path filtering in tc8[355] [paper App. H] | None | `FeatureFilter(token=-1)`, `feature_type=TRANSCODER`, etc. |
| LN constants | "empirically estimated" [paper App. D.4] | directional ratio [code 113-122] | always on |
| error nodes | "can be added", following Marks et al. [paper App. D.5] | separate call, `do_bias=True` [code 685] | walkthrough |
| root token | one token | `token=-1` [code 232] | prompt truncated at the target |
| BOS | not stated | TL `run_with_cache(str)` prepends BOS [inferred, TL default] | OWT windows carry BOS [nb walkthrough cell 9-10] |
| transcoders loaded | all MLPs | n/a | 11 (layers 0–10) in the walkthrough; 12 elsewhere |

---

## 5. Their evaluation and the published numbers usable as fidelity checks

What they validate (all qualitative, except one curve):

1. **Blind case studies** [paper §5.1, App. H, Table 4].
   - Random layer-8 transcoder features: tc8[355], [1447], [6569], [4911], [6414], [235], [2725], [479], [9030].
   - Interpreted from circuits, without looking at activating examples first. Afterwards, checked against top
     activating examples.
   - Outcome: 5 successes, 2 mostly successful (tc8[235], [2725]), 2 failures (tc8[479], [9030]).
   - "Restricted" studies may not look at input-dependent layer-0 feature information.
   - The paper text says nine studies in one place and "six of the eight" restricted in another (as I read it). The
     count is inconsistent.
2. **Transcoder-replacement checks of a root's activation.** `TranscoderReplacementContext` runs the model with
   MLPs replaced, and they compare the root activation [nb walkthrough cells 28, 45-48; caught cells 52-58].
3. **Greater-than** [paper §5.2, App. G; nb greaterthan].
   - Prompts: "The war lasted from 17YY to 17", YY = 00..99 (100 prompts). Metric: probability of years > YY minus
     ≤ YY, averaged over YY = 01..98.
   - Paths show MLP10 features fed by MLP0 "year" features through head 9.1 (cells 14-22).
   - Intervention: keep the top-k variance-ranked MLP10 transcoder features vs top-k MLP10 neurons, zero the rest, for
     k = 0..99.
     - The transcoder variant is a **full replacement without the error term** (cell 43).
     - Transcoder features beat neurons up to ≈ 24 features. tc10[5315] causes a drop.
4. **No circuit-level faithfulness, completeness, size, held-out or random-null evaluation** for path-finder output
   [paper §5, §7; inferred from absence in code and notebooks].

**Numbers to reproduce** (labelled by source; tolerance in §8):

| check | setting | published value | source |
|---|---|---|---|
| R1 path search | mlp8tc[89] on `"Oh, that rifle model is a 6M"` (BOS + 9 tokens), root @-1 (token 9), num_iters=3, num_branches=15, do_raw_attribution=True, 12 transcoders | root 7.5700; [0][0] mlp0tc[8829]@9 0.8468; [0][1] mlp3tc[14171]@9 0.5593; [0][3] attn2[2]@8 0.3468; [1][0] attn0[5]@9 0.66; [2][0] 0.15 | nb walkthrough cells 57, 60, 68 |
| R2 graph | same | 24 non-error nodes: root + 11 transcoder nodes (11 distinct latents) + 10 head nodes + 2 embed; 33 edges; node mlp0tc[8829]@9 = 1.40737; attn0[5]@9 = 1.09723; embed0@9 = 0.86215 | nb walkthrough cells 68, 70 |
| R3 error nodes | same | root tc_error 1.49699, bias_error −7.15799, prune_error 2.54064; mlp0tc[10490]@8 bias −33.3509 | nb walkthrough cell 73 |
| R4 greater-than paths | root W_U['40'] − W_U['20'] at resid_post 11, `"The war lasted from 1735 to 17"`, 3 × 15, raw | [0][0] mlp10tc[21238]@8 ≈ 16; [0][1] attn9[1]@6 6.9; [0][2] mlp10tc[9647]@8 5.9 | nb greaterthan cell 15 |
| R5 | same, `"…1527 to 15"` | mlp10tc[21238]@8 ≈ 11; mlp10tc[10437]@8 ≈ 11; mlp9tc[11883]@8 5.3 | nb greaterthan cell 21 |
| R6 greater-than metric | 98 prompts YY = 01..98 | original 0.88294; TC10 replacement 0.79780; TC10 without tc10[5315] 0.84531; feature-vs-neuron curve crossing at index 24; tc10[5315] is variance-rank index 22 | nb greaterthan cells 40, 48, 56, 62, 63 |
| R7 transcoder fidelity | 100 batches × 128 OWT windows of 128 tokens, **11 transcoders (L0–10)** | CE: original 3.5928, all replaced 4.2337, MLPs zero-ablated 10.9406; L1–L9 replaced (MLP0 and the last loaded MLP kept) 4.0624; KL 0.6684 (400 × 32) | nb walkthrough cells 33-38 |
| R8 replacement activation | mlp8tc[89], rifle prompt via `tokenizer(prompt)` (no BOS) | 7.516 original; 2.361 with tc[0:8]; 5.805 with tc[1:8] | nb walkthrough cells 45-48 |

- R7 depends on the OWT sample: streaming shuffle with seed 42, buffer 10k, `take(25,600)`. Reproduce in
  distribution, not exactly.
- R1–R5 are deterministic given the weights and TL version.

---

## 6. Exact API calls: one circuit, one feature, one prompt

Run from the repo root in an isolated env with `requirements.txt` (TL 1.11.0, torch 2.2.0). Weights come from
`bash setup.sh`, or point the path at an existing HF cache of the same revision. Their code is untouched. The last
block is our read-out glue.

```python
import torch
from transformer_lens import HookedTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from transcoder_circuits.circuit_analysis import (
    make_sae_feature_vector, greedy_get_top_paths, paths_to_graph,
    add_error_nodes_to_graph, print_all_paths, FeatureType, ComponentType)

model = HookedTransformer.from_pretrained('gpt2')                       # TL defaults (fold_ln=True)
tmpl = "./gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576.pt"
transcoders = [SparseAutoencoder.load_from_pretrained(tmpl.format(l)).eval() for l in range(12)]

L, feat = 8, 89                                   # root: transcoder layer L, feature index
toks = model.to_tokens("Oh, that rifle model is a 6M")   # BOS prepended
pos = toks.shape[1] - 1                           # target position; for a corpus window, use toks[:, :pos+1]
_, cache = model.run_with_cache(toks[:, :pos+1])

fv = make_sae_feature_vector(transcoders[L], feat, token=-1)           # encoder direction at the last token
all_paths = greedy_get_top_paths(model, transcoders, cache, fv,
                                 num_iters=3, num_branches=15, do_raw_attribution=True)
edges, nodes = paths_to_graph(all_paths)
edges_e, nodes_e = add_error_nodes_to_graph(model, cache, transcoders, edges, nodes)   # optional

# --- read-out (ours, no change to their code) ---
root_key = all_paths[0][0][0].__str__(show_contrib=False, show_last_token=True)   # e.g. 'mlp8tc[89]@-1'
tc_nodes = [(n.component_path[-1].layer, n.component_path[-1].feature_idx, n.token)
            for k, n in nodes.items()
            if k != root_key and n.component_path[-1].feature_type == FeatureType.TRANSCODER]
keep_latents = sorted({(l, j) for l, j, _ in tc_nodes})               # distinct (layer, latent), position-collapsed
head_nodes   = [(n.component_path[-1].layer, n.component_path[-1].attn_head, n.token)
                for n in nodes.values() if n.component_path[-1].component_type == ComponentType.ATTN]
```

**Guards** [ours]:
- Skip the (prompt, root) if the root activation is 0. Their method assumes an active root.
- Assert that the root key equals `mlp{L}tc[{feat}]@-1`.

**Runtime and memory on a 16 GB GPU** (estimates, not measured; to be replaced by measurement per evaluation.md §5):
- **Resident memory:**
  - GPT-2 small fp32 ≈ 0.5 GB;
  - 12 transcoders × (W_enc + W_dec = 2 × 768 × 24,576 × 4 B ≈ 151 MB) ≈ 1.8 GB;
  - the cache is small.
- **Transient peak.** `get_attn_head_contribs` materialises `[1, 12, T, T, 768]` fp32 for every layer scanned
  [code 23-26]. That is 12 · T² · 3,072 B:
  - 0.60 GB at T = 128;
  - 2.4 GB at T = 256;
  - 9.7 GB at T = 512;
  - **38.7 GB at T = 1,024, which OOMs**.
- **Safe on 16 GB:** T ≤ ~512. With the authors' 128-token windows, peak is ≈ 3 GB [inferred arithmetic].
- **Compute per `get_top_contribs` call** (root at layer L, T = 128):
  - ≤ L transcoder forwards (each ≈ 2.4 GMAC over all T tokens; the code encodes every token, then indexes one);
  - ≤ L head einsums (each ≈ 12 · T² · 64 · 768 ≈ 9.7 GMAC).
  - Total ≈ 100 GMAC: tens of ms of GPU time plus Python overhead. **Guess: 50–200 ms per call.**
- **Calls per circuit** ≈ Σ beam sizes ≤ num_iters · num_branches. That is 45 for 3 × 15 and 75 for 5 × 15.
- **Guess:** ≈ 3–15 s per circuit at T = 128. A num_branches sweep of ~10 values at num_iters = 5 costs ~2–10 min
  per (target, prompt). Cost is linear in num_branches.
- No timing is reported by the authors for the path finder.

---

## 7. Mapping to arena A

### 7.1 Node universe

| | Dunefsky (native) | WCM arena A as planned (README / ge.md §9) |
|---|---|---|
| MLP | Dunefsky transcoder latents, per (layer, latent, token) | the same transcoders, latents position-free |
| attention | **raw heads**: (layer, head, source token), via the frozen-pattern OV | Kissane `hook_z` attention-SAE latents (24,576; 49,152 at L5 and L7) |
| embedding | one token+position embedding node per position | live (not a node) |
| errors | tc / bias / prune bookkeeping nodes | transcoder error passes through (035 semantics) |
| roots | transcoder latents (or any resid-space vector) | any latent |

**Plainly:**
- Dunefsky's method has no attention dictionary. Its circuits contain **zero attention-SAE latents**.
- Under the arena-A WCM universe, every attention-SAE latent outside the circuit gets ablated. A Dunefsky circuit
  would therefore be scored with all attention latents ablated.
  - It would fail for a node-universe reason, not a method reason.
  - Its head nodes, which it does select, would be ignored.
- That is **not a same-substrate comparison.**

**The most faithful pairing [ours]: arena A-TC.** The node universe is transcoder latents only, for both methods.
- **Attention and embeddings stay live and undecomposed** in every test. This was 035's original design ("attention
  and embeddings stay live and undecomposed"), and it is also what circuit-tracer's PLT graphs do.
- **WCM** must be run once more, on this universe: transcoder latents at layers below the target, no attention
  latents. Rule 2 of the README ("WCM runs on every dictionary a baseline runs on") requires it.
- **Dunefsky** is run unmodified. Its transcoder nodes form the keep set. Its head nodes are reported as a separate
  column, never on the latent axis (like SFC error nodes in evaluation.md §6.1).
- This lets Dunefsky share arena A's dictionaries, transcoder semantics and evaluation code with Ge et al. and WCM.
  The arena-A attention-SAE universe is still used for Ge vs WCM.

**Optional sensitivity, labelled ours, not a Dunefsky row:**
- Score a Dunefsky circuit in the full arena-A universe by mapping each kept head node (l, h) to the Kissane layer-l
  latents whose decoder mass lies mostly (≥ 50%) in head h's 64-dim block of `hook_z`.
- This is our adaptation. Kissane latents can span heads, so the mapping is lossy.

**Targets.**
- Only **transcoder latents** can be Dunefsky roots natively.
- Attention-SAE latents are not native roots. Their encoder reads `hook_z`, which is not a linear read-out of one
  residual position, because it depends on the pattern over sources [inferred].
- So the Dunefsky row covers the transcoder-latent targets of arena A only. Stratify by layer, as for Ge.

### 7.2 Circuit size (evaluation.md §6.1)

**Axis:** distinct `(layer, latent)` transcoder latents in `nodes` (after `paths_to_graph`):
- excluding the root;
- collapsed over positions;
- excluding head, embedding and error nodes.

**Second columns (native):**
- transcoder nodes with position;
- head nodes `(l, h, s)` and distinct heads `(l, h)`;
- embedding nodes;
- edges;
- paths (≤ num_iters · num_branches);
- error nodes.

**Reached range.** It is small:
- 11 latents at 3 × 15 (R2);
- at most ~1 + num_iters · num_branches nodes in general.
- So their native curve covers about 10¹–10².⁵ on the 10¹–10⁵ grid. The ≤ k rule (evaluation.md §3.3) carries the
  largest circuit forward above that.

**The knob for the curve.** `num_branches` (beam N) is the size knob [ours to sweep it; the authors only picked
values].
- Primary: num_iters = 5 (their modal case-study value), num_branches ∈ {5, 10, 15, 20, 30, 40, 60}. These are the
  values they used.
- Extended (labelled "extended knob"): {100, 150, 200, 300}, to reach ~10³ latents. It is still their algorithm.
- Secondary curves: num_iters = 3 (walkthrough, greater-than) and 6 (restricted blind).
- No threshold exists to sweep. Do not invent one.

### 7.3 Scoring their circuits under our tests (Z, A, C, necessity, R)

- **Keep set** = `keep_latents` (§6), kept **at every position** (position-collapsed, matching WCM membership).
  Every other transcoder latent at layers < target layer takes the test's fill. All attention, embeddings, and
  layers ≥ target stay live.
- **Intervention semantics** (035):
  - `mlp_out ← mlp_out + (ĉ − c) W_dec`, so an untouched circuit reproduces the model exactly and transcoder error
    passes through;
  - identical for WCM's circuits.
- **Target read:** the root latent's activation at the root position, with the same read as WCM (one read
  throughout; evaluation.md P6).
- **Prompts (per-prompt method, as Ge):**
  - for each target, a subset of K activating prompts;
  - each truncated at its max-activating position, with root `token=-1`, as in their case studies;
  - one Dunefsky circuit per prompt;
  - each scored on its own prompt (home turf) and on the held-out prompts (labelled ours);
  - readings averaged across prompts per grid point (evaluation.md §6.4 item 10).
- **Position-specific sensitivity [ours]:** keep each latent only at the token(s) where it appears in the graph. This
  is defined on the circuit's own prompt only.
- **Random null:** the size-matched random-latent row (evaluation.md 5f). Mandatory on these ReLU transcoders (README
  arena A caveat).

**"Own evaluation" column.** The paper defines no circuit metric, so this column has two parts.
- (a) **Attribution coverage** [ours, derived from their own error nodes] = Σ of edges into the root / root
  activation = 1 − root `prune_error` / activation.
  - R1–R3 give (7.570 − 2.541) / 7.570 = 0.664.
  - It is bookkeeping in their linearisation, not an intervention.
- (b) **Their greater-than intervention style** applied to the root's upstream: keep the circuit's transcoder latents,
  zero all others, with **no error term** (full replacement, as in nb greaterthan cell 43); read the root.
  - This is their ablation convention transferred from "top-k of one layer" to "circuit latents of all upstream
    layers". So it is ours, and labelled as such.

### 7.4 What is "our adaptation"

1. The arena A-TC universe, and the extra WCM run on it.
2. The choice of K prompts and of the root position (max-activating), with prompt truncation as in their case
   studies.
3. Position-collapsing for the size axis and for keeping.
4. Held-out scoring of per-prompt circuits.
5. The num_branches sweep as a curve (values above 60 are "extended").
6. `do_raw_attribution=True`. It is the walkthrough's recommended mode and the only one that builds graphs (§3.5).
   The default cap mode can be shown as a path-level sensitivity only.
7. The own-eval columns 7.3(a) and 7.3(b).
8. The read-out glue in §6.

Their code is not changed.

---

## 8. Ambiguities, paper-vs-code differences, fidelity checks

### 8.1 Ambiguities and differences

| # | issue | reading for 107 |
|---|---|---|
| A1 | **Cap precedence.** `A or B and C` parses as `A or (B and C)`. For transcoder features the cap is overwritten unconditionally, so it is the contrib of the **last** transcoder feature on the path, not the "smallest" the comment says [code 426]. The paper does not describe the cap | use `do_raw_attribution=True` (A6 of §7.4); note the bug |
| A2 | **Default mode.** `do_raw_attribution=False` is the default and is used in most case-study calls. The walkthrough and greater-than use True. Graphs require True (assert) [code 399, 658] | True |
| A3 | **Stale notebook outputs.** Case-study outputs show the root without a contrib, and path contribs above the root (e.g. 21 vs ≈ 19). That is inconsistent with HEAD code, which sets the root contrib [code 412] and therefore caps in default mode. Those outputs were likely produced by earlier code [inferred] | fidelity checks use only walkthrough and greater-than outputs (R1–R6) |
| A4 | **Head top-k capped at n_heads = 12** per layer when num_branches > 12 [code 337] | keep, since we run unmodified; note in the row caption |
| A5 | **Head node identity drops the destination token**, so different destinations merge [code 670-680] | count `(l, h, s)` natively; distinct `(l, h)` as a second column |
| A6 | **`bias_error` units** are not scaled by the path gradient for non-root nodes [code 742-744] | do not use bias errors in any metric |
| A7 | **LN constant** is the directional ratio dot(v, y) / dot(v, x). The paper's description reads as "empirically estimated" and is omitted from its pseudocode [paper App. D.4–D.5] | code |
| A8 | **Signed top-k**: only positive contributors are followed | inherent; report that circuits are excitatory-only |
| A9 | **Root ReLU and bias ignored**: the root is assumed active [paper App. D.2] | skip inactive roots |
| A10 | **b_O and attention-to-BOS** are never candidates [code 13-34, 335-342] | inherent; they enter the pruning error |
| A11 | **Embedding node = token + position** [code 377] | inherent |
| A12 | **L and N are not given in the paper.** The repo uses 3/15, 5/15, 6/20, up to 6/40 and 3/60 | §7.2 sweep |
| A13 | **Paper Alg. 1's second filter** ("earliest-layer node among the top N") vs the code's single top-N by newest contrib | equivalent [inferred] |
| A14 | **Walkthrough loads 11 transcoders** (no layer 11); others load 12. HF has a `lower_l0` layer-11 alternative | load 12 top-level files; R7 uses 11 |
| A15 | **BOS.** `run_with_cache(str)` prepends BOS; `tokenizer(str).input_ids` (used in `get_feature_scores` examples) does not. That is why R8's 7.516 differs from R1's 7.570 for the same prompt [inferred] | always `to_tokens` with BOS; never root at position 0 |
| A16 | **Training hyperparameters**: paper App. E lr 2e-5, script lr 4e-4 / L1 1.4e-4, saved cfg L1 8e-5 (035). Which lr produced the HF weights is unknown | irrelevant to circuit finding; record the HF revision |
| A17 | **Environment drift**: TL 1.11.0 / torch 2.2.0 pins. The pickled cfg needs `sae_training` and `weights_only=False`. `model.OV` and `get_act_name` behaviour in newer TL is unverified | isolated venv with the pins, like 035's `venv-ct` |
| A18 | **Paper case-study count**: "nine" vs "six of the eight" restricted [paper §5.1, as read] | descriptive only |
| A19 | **Pre-activation vs activation**: root contrib is post-ReLU; children's contribs are input × gradient through the root's pre-activation | inherent |

### 8.2 Fidelity checks (all before any comparison row)

- **F1. Path search reproduces the walkthrough (R1).**
  - Pass: root 7.5700, and the first-beam ordering and values of all 15 level-1 paths, within 1e-3 relative.
  - Also: the beam-2 and beam-3 path strings match cell 60 exactly.
  - Failure points to TL version, fold_ln, BOS or the transcoder files.
- **F2. Graph reproduces (R2, R3).**
  - Pass: 24 non-error nodes, 33 edges, the listed node and edge values, and the root's three error nodes, within
    1e-3 relative.
  - Pass: `keep_latents` has 11 entries: {mlp0[8829], mlp0[10490], mlp0[7829], mlp2[15119], mlp3[14171],
    mlp4[17353], mlp4[17443], mlp5[6434], mlp5[2296], mlp6[17269], mlp7[22013]}.
- **F3. Greater-than paths (R4, R5).** Pass: top-3 level-1 nodes and values match to 2 significant figures as
  printed.
- **F4. Greater-than metric (R6).** Pass: original 0.8829, TC10 replacement 0.7978, without 5315 0.8453 (± 1e-3);
  tc10[5315] at variance rank 22.
- **F5. Transcoder fidelity on our conversion (R7 + 035).**
  - CE with 11 transcoders replaced within ±0.05 nats of 4.234, on an OWT sample of the same construction (the sample
    differs).
  - Our 035 FVU per layer 0.003–0.27.
  - The same files feed WCM's arena A-TC adapter. Assert identical feature activations, ≤ 1e-5, between their
    `SparseAutoencoder` and our adapter on the same tokens.
- **F6. Path attribution is input × gradient (unit test, ours).**
  - Build a linearised model: frozen patterns; LN replaced by the code's directional constants along each path;
    MLPs replaced by transcoders with real activations; errors detached.
  - Pass: for a single path, the code's contrib equals a_leaf · ∂root/∂a_leaf restricted to that path, to fp32
    tolerance.
  - This checks our understanding of §3.3, not their code.
- **F7. Read-out glue.**
  - Pass: `sum_over_tokens` node set ⊇ `keep_latents`.
  - Pass: no error, head or embed node leaks into `keep_latents`.
  - Pass: the root is excluded.
- **F8. Scoring identity.** Pass: under our harness, keeping every transcoder latent (nothing filled) reproduces the
  clean root activation exactly (035 semantics), before any circuit is scored.

---

## Sources (read 2026-11-01; nothing downloaded or cloned)

- Paper: https://arxiv.org/abs/2406.11944 (versions), https://arxiv.org/html/2406.11944v2 (full text, §3.2, §4.1,
  §5, App. D.2–D.5, E, F, G, H).
- Code: https://github.com/jacobdunefsky/transcoder_circuits @ `8721ff06b64889e8cc50bb1c5e869c7f72b8303d`.
  - Commit list and tree via the GitHub API.
  - Files listed in the header, read raw to stdout.
- Weights: https://huggingface.co/pchlenski/gpt2-transcoders @ `4f3ca345572ce2d75018a60d10b4dca8bd899e99` (file
  listing via the HF API).
- Local:
  - `experiments/107-true-baselines/README.md`, `specs/ge.md`, `specs/evaluation.md`;
  - `experiments/035-transcoder-compare/README.md`, `convert_transcoders.py`, `transcoders_ct/meta.json`.
