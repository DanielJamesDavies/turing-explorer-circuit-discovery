# Spec: attribution graphs (Ameisen et al. 2025) and `circuit-tracer`, as published and as coded

Written 2026-11-01 by `spec-ag` for 107 **arena D** (Gemma-2-2B). This file describes what the paper and the official
library do. It does not describe what we think they should do. Anything we would change is labelled **our adaptation**.

## 0. Sources read, and how to read the tags

- **Methods paper:** Ameisen, Lindsey, Pearce, Gurnee, Turner, Chen et al., "Circuit Tracing: Revealing Computational
  Graphs in Language Models", transformer-circuits.pub/2025/attribution-graphs/methods.html. Read through WebFetch on
  2026-11-01: §2 (replacement model), §3 (attribution graphs), §5 (evaluations), §7 (limitations), Appendices
  "Attribution Graph Computation", "Graph Pruning", "Iterative Patching", "Details of Interventions". The page has no
  numbered headings; the numbers below are page order (§2 replacement model, §3 graphs, §5.1 CLT evaluation, §5.2 graph
  evaluation, §5.3 mechanistic faithfulness). WebFetch returns a model's paraphrase of the page, so **every paper number
  below was read through a summariser.** The pruning table and the score definitions were re-fetched verbatim (§5.2).
- **Biology paper:** Lindsey et al., "On the Biology of a Large Language Model", .../attribution-graphs/biology.html, read
  for pruning/presentation, interventions, aggregation across prompts and success rate only.
- **Library paper:** Hanna, Piotrowski, Lindsey & Ameisen, "circuit-tracer: A New Library for Finding Feature Circuits",
  BlackboxNLP 2025, pp. 239-249 (aclanthology 2025.blackboxnlp-1.14), pages 1-8 read. The WebFetch tool cached the PDF in
  the session's tool-results folder on its own; nothing was written to the repo.
- **Code:** github.com/**decoderesearch**/circuit-tracer. `safety-research/circuit-tracer` now redirects there (GitHub
  API). Pinned at main HEAD **`255d989e93be1299d53d67b59e9fd76a0e274b14`** (2026-10-14, "fix: preserve inactive token
  positions in CLT reconstruction (#118)"). Files read in full as raw text through `gh api`, never cloned:
  - `circuit_tracer/attribution/{attribute.py, attribute_transformerlens.py, context_transformerlens.py, targets.py}`;
  - `circuit_tracer/graph.py`;
  - `circuit_tracer/replacement_model/{replacement_model.py, replacement_model_transformerlens.py}`;
  - `circuit_tracer/transcoder/{single_layer_transcoder.py, cross_layer_transcoder.py}`;
  - `circuit_tracer/utils/{hf_utils.py, create_graph_files.py, disk_offload.py}`;
  - `circuit_tracer/{__init__.py, __main__.py}`, `README.md`, `pyproject.toml`;
  - `tests/test_attributions_gemma.py` (lines 1-200);
  - the code and markdown cells of `demos/attribute_demo.ipynb`, `demos/attribution_targets_demo.ipynb` and
    `demos/circuit_tracing_tutorial.ipynb`. Their saved outputs are stripped, so no timings are in the repo.
  - **Not read:** the nnsight backend (`attribute_nnsight.py`, `context_nnsight.py`, `replacement_model_nnsight.py`).
    Arena D uses the TransformerLens backend, which is the default [code replacement_model.py:29].
- **Diff since 038's pin** (`8f1e2438`, 2026-07-17; see `038-transcoder-compare-gemma/CIRCUIT_TRACER_VERSION`): the
  TransformerLens attribution path is unchanged. The changes are:
  - `graph.py` +4/−2: `find_threshold` no longer keeps zero-weight edges at threshold 1.0, PR #116;
  - `cross_layer_transcoder.py` +55/−5: affine CLT skip weights, PR #117, and CLT reconstruction with inactive trailing
    positions, PR #118;
  - the nnsight 0.8 port.

  So 038's PLT results are representative of HEAD. **Its CLT results would not be:** before PR #118, a CLT
  reconstruction could be cut short when the last positions had no active features [inferred from the PR title and the
  `n_pos` handling at code cross_layer_transcoder.py:291-325].
- **Hugging Face** (API, read 2026-11-01):
  - `mwhanna/gemma-scope-transcoders` @ `3412e19b`;
  - `mntss/gemma-scope-transcoders` @ `9250a2d4`;
  - `mntss/clt-gemma-2-2b-426k` @ `b1e9ab37`;
  - `mntss/clt-gemma-2-2b-2.5M` @ `637a2f8b`;
  - `mntss/transcoder-Llama-3.2-1B` @ `c37a82c1`;
  - `mntss/clt-llama-3.2-1b-524k` @ `d6ab3ffa`.

  For each repo, `config.yaml` and the file sizes were read, but no weights.
- **Neuronpedia:** graph metadata `api/graph/gemma-2-2b/gemma-fact-dallas-austin`, and that graph's JSON (streamed and
  summarised, not saved).
- **Local:**
  - 107 `README.md`, `specs/evaluation.md`, `specs/sfc.md`, `specs/ge.md` §9;
  - `dev-notes/2026-09-28-baseline-audit.md` §5;
  - `experiments/038-transcoder-compare-gemma/` (`README.md`, `theirs_gtc.py`, `ct_prune.py`, `ct_faithfulness.py`
    header, `ours_gtc.py` header and Runner, `theirs_run.log`, `theirs_pruned_run.log`, `faithfulness.log`,
    `CIRCUIT_TRACER_VERSION`);
  - `experiments/050-gemma-circuits/README.md`;
  - `src/circuit/instrument/learned_mask.py:315-334`.

**Tags.**
- [paper §x] = methods paper; [bio §x] = biology paper; [lib §x] = library paper.
- [code file:line] is at `255d989` unless another SHA is given. Paths are relative to `circuit_tracer/` unless they start
  with `tests/`, `demos/` or `README`.
- [HF repo] = Hugging Face API or config; [038 file] = our earlier run; [inferred] = my reading, not stated anywhere;
  [ours] = a recommendation of ours.
- Line numbers count from 1 in the raw file.

---

## 1. Summary

- **Replacement model.** The MLPs are replaced by transcoders: a sparse code read from the MLP's input that predicts
  the MLP's output.
  - A **per-layer transcoder (PLT)** reads one layer's MLP input and writes that layer's MLP output.
  - A **cross-layer transcoder (CLT)** feature reads the residual stream at its layer ℓ and has a separate decoder into
    every MLP output at layers ℓ' ≥ ℓ [paper §2 "Architecture"; code cross_layer_transcoder.py:15-48].
- **Local replacement model, for one prompt** [paper §2 "The Local Replacement Model"]:
  - attention patterns and normalisation denominators are frozen at their values on that prompt;
  - one **error node** per (layer, position) is added, equal to the true MLP output minus the transcoder reconstruction,
    so that activations and logits match the real model exactly;
  - the only nonlinearities left are the transcoder activation functions.
- **Attribution graph.** Nodes are input-token embeddings, active transcoder features at each (layer, position), error
  nodes and output logits [paper §3.1].
  - The edge s→t is A_{s→t} = a_s · w_{s→t}. Here w is the Jacobian of t's **pre-activation** with respect to a_s in the
    frozen model: a path through the residual stream and the attention OV circuits only [paper §3.1].
  - In the library the graph is a dense adjacency matrix, rows = targets and columns = sources
    [code graph.py:46-74]. Every row comes from one injected backward pass [code context_transformerlens.py:168-232].
- **Pruning** keeps the smallest node set carrying 80% of logit-weighted multi-hop influence, then the smallest edge set
  carrying 98% of edge influence, then removes dangling nodes [paper App. "Graph Pruning"; code graph.py:239-313].
- **Scope.** A graph is **one prompt, rooted at that prompt's last-position logits.** Rooting at an arbitrary feature
  node is **not native** (§4).

---

## 2. Substrate

### 2.1 Models and transcoder sets shipped for circuit-tracer

Gemma-2-2B shape [inferred, standard config]: 26 layers, d_model 2,304, d_mlp 9,216, 8 query / 4 KV heads of dim 256,
vocabulary 256,000, final-logit soft-cap 30, about 2.6 B parameters.

| model | set (HF id @ sha) | kind | features | activation | input → output hooks (TL) | weights on disk |
|---|---|---|---|---|---|---|
| Gemma-2-2B | `mwhanna/gemma-scope-transcoders` @ 3412e19b, alias **`gemma`** [code hf_utils.py:88-89] | PLT, GemmaScope re-upload as safetensors | 16,384 per layer × 26 = 425,984 | JumpReLU, per-feature threshold [code single_layer_transcoder.py:471-476] | `ln2.hook_normalized` → `hook_mlp_out` [HF config.yaml] | 26 × 302 MB fp32 = 7.86 GB [HF API] |
| Gemma-2-2B | `mntss/gemma-scope-transcoders` @ 9250a2d4 | the same PLTs, pointing at the original `google/gemma-scope-2b-pt-transcoders` .npz files | 16,384 per layer | JumpReLU [code single_layer_transcoder.py:408-444] | same | from Google's repo. No lazy loading of the .npz [inferred from loader] |
| Gemma-2-2B | **`mntss/clt-gemma-2-2b-426k`** @ b1e9ab37 | CLT | 16,384 per layer × 26 = 425,984 | **ReLU** [inferred: no `threshold_i` in `W_enc_i` files (sizes match W_enc + b_enc + b_dec in bf16 only); code cross_layer_transcoder.py:464 then picks relu]; the library paper says "ReLU CLTs" [lib §3.2.2] | `hook_resid_mid` → `hook_mlp_out` [HF config.yaml] | W_enc 1.96 GB + W_dec **26.5 GB** (bf16) [HF API] |
| Gemma-2-2B | `mntss/clt-gemma-2-2b-2.5M` @ 637a2f8b | CLT | 98,304 per layer × 26 = 2,555,904 | ReLU [inferred, same way] | same | W_enc 11.8 GB + W_dec **159 GB** [HF API] |
| Llama-3.2-1B | `mntss/transcoder-Llama-3.2-1B` @ c37a82c1, alias `llama` | PLT with skip (W_skip) [lib §3.2.2 "ReLU PLTs"; code hf_utils.py:90-91] | 131,072 per layer × 16 [inferred from 1.08 GB per layer in bf16] | ReLU | `hook_resid_mid` → `hook_mlp_out` | 17.3 GB |
| Llama-3.2-1B | `mntss/clt-llama-3.2-1b-524k` @ d6ab3ffa | CLT | 32,768 per layer × 16 | ReLU [lib §3.2.2] | same | 20.4 GB |
| Qwen3 0.6B-14B | `mwhanna/qwen3-*-transcoders(-lowl0)` | PLT | not read | ReLU [lib §3.2.2] | not read | not read |
| GPT-OSS-20B, Gemma-3, Llama-3.1-8B-Instruct | listed in README [code README.md:49-51] | CLT / PLT / Top-K PLT | not read | — | — | — |
| **GPT-2** | **no shipped set** [code README.md:43-51]. Local transcoders load from a path [README.md:53-57]. `mntss/sparsecoder-sweep-gpt2` exists on HF but is not listed and was not read | — | — | — | — | — |

- **Which Gemma PLT picks.** One width-16k L0 pick per layer. The config calls them "lowest L0", but the list has L0 76,
  65, 49, 54, 88, 87, 95, 70, 52, 72, 88 at layers 0-10, then 5-41 at layers 11-25 [HF `mntss/gemma-scope-transcoders`
  config]. That is about 1,020 active features per position summed over layers.
  - 038 measured about 1,000 active per position (58-67 k active features on 64-token windows), and L0 83 vs the
    advertised 88 at one layer [038 theirs_run.log; 038 README].
- **Which set the original papers used.** Neither paper used Gemma. The methods paper used an internal 18-layer model
  and Claude 3.5 Haiku, with JumpReLU CLTs of 300 K-10 M and 300 K-30 M features [paper §2 "Architecture", §5.1]. The
  Gemma CLTs were trained by the library authors [lib §3.2.2].
  - **So "the substrate it was built for" means: the library on Gemma-2-2B with the shipped sets.** The paper's own
    graphs cannot be rerun.
- **CLT vs PLT in the paper:**
  - CLTs beat PLTs on all of its interpretability and reconstruction metrics.
  - On one 10 M-feature prompt, replacement score 0.61 vs 0.37, completeness 0.80 vs 0.78, mean path length 2.3 vs 3.7
    [paper §5.2].
  - The paper warns that CLTs can merge repeated amplification chains, which the underlying model might not do
    [paper §5.2 "path length"].

### 2.2 What is frozen, and how the code implements it

The library does **not** build a separate linear model. It runs the **real model forward** and makes the backward pass
see only the linear paths [code replacement_model_transformerlens.py:189-250]:

| component | forward | backward (gradient) | code |
|---|---|---|---|
| attention pattern | real | `detach()`, so QK is frozen and only the OV path carries gradient | :197 |
| LayerNorm / RMSNorm scales (ln1, ln2, ln1_post, ln2_post, ln_final) | real | `hook_scale` detached, so the norm is linear in x | :198-204 |
| MLP output at `feature_output_hook` | **real MLP output** | replaced by `skip + (acts − skip).detach()`, so no gradient goes through the MLP (only through W_skip, if the set has one) | :223-232 |
| parameters | — | `requires_grad = False` | :206-207 |
| token embedding | — | `hook_embed` set to requires_grad, so token nodes get gradient | :209-213 |
| final logit soft-cap | real | logit rows are read at `unembed.hook_pre`, i.e. before W_U and the soft-cap | context_transformerlens.py:90; attribute_transformerlens.py:152-153 |

**Model loading:** `fold_ln=False`, `center_writing_weights=False`, `center_unembed=False`
[code replacement_model_transformerlens.py:111-117]. So TL's Gemma `ln2.hook_normalized` is the unweighted normalised
input (038 probed: HF MLP input / (1 + w_pre) [038 ours_gtc.py:7-11]).

**Position 0 (BOS) is removed from the graph:**
- `ensure_tokenized` prepends a BOS / special token unless the first token is already special
  [code replacement_model_transformerlens.py:352-422].
- Feature activations at `zero_positions = slice(0, 1)` are zeroed [code :169-171; single_layer_transcoder.py:166], and
  so are the error vectors at position 0 [code replacement_model_transformerlens.py:461].
- The BOS-position MLP output is therefore neither a feature nor an error node. It is a constant of the graph, like the
  biases [inferred].

### 2.3 Node types and the adjacency layout

The order is the same in rows and columns [code graph.py:46-74; create_graph_files.py:33-77]:

```
[ selected feature nodes (F) | error nodes (n_layers × n_pos), layer-major | token nodes (n_pos) | logit nodes (K) ]
```

- **Feature node:** (layer, pos, feature) with activation a > 0 at that position.
  - `graph.active_features` lists all active ones as (layer, pos, idx). `graph.selected_features` indexes the subset that
    got rows [code graph.py:57-66; attribute_transformerlens.py:252-256].
  - Its source vector is `a · W_dec` (scaled decoder) and its target vector is the raw encoder row `W_enc[f]`
    [code single_layer_transcoder.py:170-172, 186; attribute_transformerlens.py:234-237].
  - For a **CLT**, one node is one (encoder layer, pos, feature). Its decoders into each output layer ℓ' ≥ ℓ are
    separate entries, all summed into that node's column [code cross_layer_transcoder.py:235-289;
    context_transformerlens.py:107-112, 122-132 (`+=` into `encoder_to_decoder_map` rows)].
- **Error node:** (layer, pos), with vector = MLP output − reconstruction. It has **no incoming edges**: its row is all
  zero [code replacement_model_transformerlens.py:459-461; paper §3.1].
- **Token node:** pos, with vector `W_E[token]` injected at `hook_embed`. No incoming edges
  [code replacement_model_transformerlens.py:462; context_transformerlens.py:147-155].
- **Logit node:** the k-th target. Its target vector is the **demeaned** unembedding column, `W_U[:, v] − mean_v W_U`,
  read at the last position of `unembed.hook_pre` [code targets.py:376-411; attribute_transformerlens.py:196-207].
  - The paper also describes logit targets as the logit minus the mean logit [paper App. "Attribution Graph Computation"].

---

## 3. The algorithm (pseudocode from the code; formulas match the paper unless §9 says otherwise)

### 3.1 Edge weight

```
A[t, s] = (a_s · W_dec_s)  ·  J_{out(s) → in(t)}  ·  W_enc_t           feature s → feature t
A[t, e] = e_{ℓ,p}          ·  J_{out(ℓ) → in(t)}  ·  W_enc_t           error node e = (ℓ, p)
A[t, x] = W_E[tok_p]       ·  J_{embed → in(t)}   ·  W_enc_t           token node
A[k, ·] = same with  W_enc_t  replaced by  u_k = W_U[:, v_k] − mean(W_U)  at (unembed.hook_pre, last pos)
J = Jacobian of the frozen model (patterns, norm scales, MLP outputs detached; skip path kept if W_skip exists)
```

- **The target quantity is the pre-activation of t** [paper §3.1].
- **The direct effect is exact** in the frozen model: the repo's test doubles one feature's activation in a run where
  every layer is constrained, then asserts that the pre-activations and demeaned logits change by exactly that feature's
  adjacency column (atol 5e-4 / 1e-5) [code tests/test_attributions_gemma.py:123-188]. Error and token columns are
  tested the same way [:29-120].
- **For a CLT feature,** J is summed over all of its output layers ℓ' ≥ ℓ, each with its own decoder [paper §3.1; code
  as in §2.3].
- **The incoming edges do not sum exactly to t's pre-activation.** The remainder is b_enc, the decoder biases, the
  BOS-position MLP outputs and any skip-path constants. The paper says the pre-activation is the sum of the incoming
  edges [paper §3.1]; that holds only up to these constants [inferred from §2.2].

### 3.2 Graph computation (`attribute`)

```
attribute(prompt, model, attribution_targets=None, max_n_logits=10, desired_logit_prob=0.95,
          batch_size=512, max_feature_nodes=None, offload=None, verbose=False, update_interval=4)
                                                             # attribute.py:20-32; TL impl :43-55
0  ids = ensure_tokenized(prompt)                            # BOS prepended, :137
   ctx = setup_attribution(ids):  one no-grad forward; cache feature_input_hook and feature_output_hook per layer;
         encode all layers sparsely (pos 0 zeroed); recon; error = mlp_out − recon (pos 0 zeroed);
         token_vecs = W_E[ids]; encoder_vecs = W_enc rows of active feats; decoder_vecs = a·W_dec (CLT: one entry per
         output layer)                                       # replacement_model_transformerlens.py:425-473
   if offload: move transcoders to cpu/disk                  # :145-146
1  forward with grad on ids.expand(batch_size, -1), stop_at_layer = n_layers; cache resid at every
   feature_input_hook and ln_final(resid) at unembed.hook_pre      # :151-153; context_transformerlens.py:77-91
   if offload: move all block.mlp to cpu/disk                # :156-157 (MLPs are not on the gradient path)
2  targets = AttributionTargets(attribution_targets, logits[0, -1], W_U, ...)   # :166-173
       None  -> top-k by prob, k = min(max_n_logits, smallest k with cum prob >= desired_logit_prob)  # targets.py:184-217
       Tensor / [str] -> those tokens, prob = their softmax prob   ;   [CustomTarget(str, prob, vec)] -> vec, weight=prob
   if offload: move unembed, embed                            # :177-178
   N = #active features ; layout offsets ; F = min(max_feature_nodes or N, N)   # :180-185
   E = zeros(F + K, N + (n_layers+1)·n_pos)  on CPU, fp32    # :187
3  logit rows: for each batch of ≤ batch_size targets: compute_batch(layer = n_layers, pos = n_pos−1, inject = u_k)
                                                             # :196-207
4  feature rows (adaptive order):                            # :212-246
   while visited < F:
       if F == N: pending = all features (index order)       # no ranking at all
       else: infl = compute_partial_influences(E[:rows so far], logit_probs)  # graph.py:366-409, on GPU
             pending = top (update_interval · batch_size) unvisited features by infl
       for each batch of ≤ batch_size in pending:
             rows = compute_batch(layers, positions, inject = W_enc rows)   # one backward per batch
5  keep only selected feature columns (if F < N); reorder rows; build square (F+L·n+n+K)² adjacency  # :252-263
compute_batch(layers, positions, inject):                    # context_transformerlens.py:168-232
   register a gradient hook at resid[layer] that writes inject[i] into batch row i at position pos_i;
   backward from the deepest layer with zero upstream gradient;
   backward hooks at each blocks.ℓ.hook_mlp_out.hook_out_grad compute  grad · (a·W_dec)  per feature entry,
   grad · error_vec per (ℓ, p), and at hook_embed  grad · W_E[tok]  -> one row per batch element
```

- **Batching:** one backward pass computes `batch_size` rows at once. The forward is run on `batch_size` identical copies
  of the prompt, so each copy carries one source's injected gradient [code attribute_transformerlens.py:152;
  context_transformerlens.py:190-219].
- **Cost:** one backward per `batch_size` rows. `ceil(K / b) + ceil(F / b)` backwards in all [inferred]. The paper says
  the cost is linear in the number of active features and dominated by backward passes [paper App. "Attribution Graph
  Computation"].
- **`max_feature_nodes` is the paper's "adaptive" mode** [paper App. "Graph Pruning"]: features are expanded in order of
  their current estimated influence on the logits, and the unvisited ones are left out of the graph.
  - **The code drops the unvisited features' columns entirely** (they do not become error mass) [code
    attribute_transformerlens.py:252-256].
  - The paper says unexplored nodes are treated as errors [paper App. "Graph Pruning"]. See §9.3.

### 3.3 Influence and pruning (`prune_graph`)

```
normalize(A)  = |A| / rowsum(|A|)   (rowsum clamped ≥ 1e-10)        # graph.py:184-186
influence(Â, w) = w Â + w Â² + ...  (power series until zero; ≤ 1000 iters; DAG so finite)   # :189-206
w = zeros; w[logit rows] = logit_probabilities                      # :266-269

prune_graph(G, node_threshold=0.8, edge_threshold=0.98):            # :239-313
  I_node = influence(normalize(A), w)                                # over ALL columns: features, errors, tokens, logits
  keep_node = I_node >= find_threshold(I_node, node_threshold)       # smallest top set with cum share ≥ threshold
  keep_node[tokens, logits] = True
  A' = A with rows and cols of dropped nodes zeroed
  Â' = normalize(A');  I' = influence(Â', w) + w
  edge_score[t, s] = Â'[t, s] · I'[t]
  keep_edge = edge_score >= find_threshold(edge_score.flatten(), edge_threshold)
  repeat until stable:
     features and errors must have ≥ 1 kept OUTGOING edge (to a non-token, non-logit target column set)  # :290, :303
     features must have ≥ 1 kept INCOMING edge                                                             # :292, :305
     drop edges of dropped nodes
  cumulative_scores[i] = cumulative share of I_node at node i's rank   # :307-311
find_threshold(scores, θ): sort desc; cum = cumsum/sum; idx = searchsorted(cum, θ), capped at last non-zero; return
  scores[idx]                                                          # :221-230
```

- **The paper's description agrees** [paper App. "Graph Pruning"; §5.2 "Computing Node Influence"]:
  - normalised absolute adjacency, B = (I − A)⁻¹ − I, logit rows weighted by probability;
  - nodes kept to 0.8 cumulative influence, edges to 0.98 by "normalised weight × target's logit influence";
  - embedding and error nodes are "never pruned".
- **Errors in the code.** Error nodes **are** subject to the 0.8 node cut and to the outgoing-edge rule. Only tokens and
  logits are force-kept [code graph.py:275]. See §9.4.
- **Pruning is a separate step and does not touch the graph.** It returns masks. The CLI and `create_graph_files` apply
  it for the viewer [code create_graph_files.py:194-196; __main__.py:262-272].

### 3.4 Graph scores (`compute_graph_scores`), on the unpruned graph

```
Â = normalize(A);  I = influence(Â, w)
replacement  = Σ_{token nodes} I / ( Σ_{token nodes} I + Σ_{error nodes} I )                  # graph.py:340-357
completeness = Σ_i (1 − Σ_{e∈errors} Â[i, e]) · (I_i + w_i)  /  Σ_i (I_i + w_i)               # :359-361
```

Paper definitions [paper §5.2, verbatim phrases]:
- **completeness** = "the fraction of input edges (weighted by the target node's logit influence score)" that come from
  feature or embedding nodes rather than errors;
- **replacement** = "the fraction of end-to-end graph paths (weighted by strength) that proceed from embedding nodes to
  logit nodes" through features only;
- **for pruned graphs,** "pruned nodes now counting towards the error terms". The library has **no pruned-graph score
  function** [code graph.py:316-363; inferred from absence].

---

## 4. Rooting: logits by default. Can a graph be rooted at a feature node?

**Native, in the released code:**
1. **The default root is the salient logits at the last position.** The alternatives are explicit tokens, or
   `CustomTarget(token_str, prob, vec)` with an arbitrary d_model direction [code targets.py:29-43, 95-117;
   demos/attribution_targets_demo.ipynb cells 16-24].
   - **A `CustomTarget` is not a feature root.** Its vector is injected at `unembed.hook_pre`, i.e. after `ln_final`,
     at the **last position** only [code attribute_transformerlens.py:196-202; context_transformerlens.py:90].
   - It therefore attributes a direction of the final residual stream. A transcoder latent at layer L reads the stream
     at layer L, and the gradient path from the final stream runs through layers L…25 as well. So a custom target
     cannot express "latent (L, t) at position p" [inferred].
   - The demo's "abstract concept directions" are final-stream directions (logit differences, capital − state)
     [demos/attribution_targets_demo.ipynb cells 16-18].
2. **Every selected feature already has its own row** of incoming direct effects. One-hop parents of feature (L, p, t)
   are therefore native output, **if (L, p, t) is among the selected features** [code attribute_transformerlens.py:
   231-245].
   - With `max_feature_nodes = None` (the API default) every active feature is selected [code :184, :219-220].
   - With a cap (CLI 7,500; demos 8,192; Neuronpedia 5,000), selection follows **logit** influence, and the target row
     can be missing. 038 measured this: 1 of 47 and 4 of 48 windows had no seed row at L6 with a 16,384 cap on 64-token
     windows [038 theirs_run.log].
3. **The paper** iterates over "target nodes in the graph (output or feature nodes)" but defines pruning, influence and
   the scores relative to the logits only [paper App. "Attribution Graph Computation", "Graph Pruning"]. Neither paper
   describes pruning rooted at an intermediate feature [paper; bio, as read]. The biology paper builds graphs from output
   tokens or specific positions [bio, as read].

**Verdict: a latent-rooted attribution graph is NOT native. That row is "our adaptation".**

**The minimal adaptation**, which keeps their attribution code untouched [ours]:
- **A1. Truncate the prompt at the target position p** (`ids[:p+1]`).
  - In a causal model, activations at positions ≤ p are unchanged up to numerics [inferred]. So the target is at the
    last position and nothing after it is computed.
  - Input choice only; no code change.
- **A2. `max_feature_nodes = None`,** so every active feature gets a row and the target's row is guaranteed.
  - No code change; it is the API default.
  - Cost: F = all active features ≈ 1,000 × (p + 1) for the Gemma PLTs (§2.1). See §7.3 for memory.
- **A3. Root the pruning at the target row,** with 038's `prune_rooted` [038 ct_prune.py:48-94]:
  - it is their `prune_graph` body with one substitution, w = one-hot on the target row instead of the logit
    probabilities;
  - the root is kept and exempted from the outgoing-edge rule, as their code does for logits;
  - `identity_check` asserts that with the logit root it equals `prune_graph` exactly. It passed in 038 [038
    faithfulness.log, "PASS identity"].
  - **This is the only algorithmic change.**
- **A3 alternative, closer to their adaptive mode (more code changed):** rank features for selection by influence on
  the target. Seed `compute_partial_influences` with the target row instead of `logit_p` [code graph.py:398-399]. Needed
  only if A2's memory is too large (§7.3). Label it as a further adaptation.
- **Not acceptable as "theirs":**
  - one-hop direct edges into the seed summed over positions and prompts (038's ranking arm, `theirs_gtc.py:176`). The
    audit already flagged it [baseline audit §5];
  - pinning the seed in a logit-rooted pruning (038 `prune_pinned`). Both are ours.

**What the "true algorithm" row is** [ours]: AG exactly as published, logit-rooted on the target's prompt (truncated at
p or not). Report whether the target latent survives pruning, and its surviving ancestors (038's `ct_published` arm).
- 038 found the seed survived in only 18 of 44 windows for L6/6649 [038 theirs_pruned_run.log].
- This row answers "does AG as published find this latent's circuit". The latent-rooted row (A1-A3) answers "AG's
  attribution rooted where WCM is rooted".

---

## 5. Hyperparameters: paper vs code

| quantity | paper | `attribute()` API | CLI | demos / Neuronpedia / 038 | note |
|---|---|---|---|---|---|
| logit targets | top tokens covering 0.95 probability, at most 10 [paper §3.1; App. "Graph Pruning"] | `max_n_logits=10`, `desired_logit_prob=0.95` [attribute.py:25-26] | 10 / 0.95 [__main__.py:56-64] | same | consistent |
| logit target vector | logit minus mean logit [paper App.] | demeaned W_U column [targets.py:376-411] | — | — | consistent |
| feature cap | adaptive expansion, values not given [paper App. "Graph Pruning"] | `max_feature_nodes=None` (all) [attribute.py:28] | **7,500** [__main__.py:74-79] | demo 8,192 [attribute_demo cell 6]; Neuronpedia Dallas 5,000 [Neuronpedia JSON]; 038 16,384 | three defaults; always pass it explicitly |
| re-rank interval | not given | `update_interval=4` batches [attribute.py:31] | not exposed | — | — |
| batch size (rows per backward) | not given | **512** [attribute.py:27] | **256** [__main__.py:65-67] | demo 256; 038 64 | memory only. 038: batch 64 vs 256 gives selected Jaccard 1.0, pruned Jaccard 0.994, edge corr 0.99998 [038 faithfulness.log] |
| dtype | — | `ReplacementModel.from_pretrained` default **float32** [replacement_model.py:31] | float32 [__main__.py:49-55] | demos bf16 [attribute_demo cell 4]; 038 bf16 | fp32 does not fit 16 GB (§7.3) |
| lazy encoder / decoder | — | False / True [replacement_model.py:32-33] | `--lazy-encoder` off; `--lazy-decoder` store_true with default True, so it cannot be turned off [__main__.py:81-91] | 038 True / True | 038 lazy vs eager: identical adjacency [038 faithfulness.log] |
| offload | — | None [attribute.py:29] | None | demo `"cpu"` locally, `"disk"` on Colab | memory only [inferred: moves modules that are off the gradient path] |
| node threshold | **0.8** [paper App. "Graph Pruning"] | `prune_graph` 0.8 [graph.py:240] | 0.8 [__main__.py:117-122] | demo text 0.8, cell 21 0.7; Neuronpedia Dallas **0.7**; viewer slider 0.6 | knob for the curve |
| edge threshold | **0.98** [paper App.] | 0.98 [graph.py:240] | 0.98 [__main__.py:123-128] | demo 0.98 / 0.95; Neuronpedia Dallas **0.9**; library paper listing 0.95 [lib §3.1.3] | — |
| BOS handling | — | prepend BOS; zero features and errors at pos 0 [replacement_model_transformerlens.py:169-171, 402-421, 461] | same | — | — |
| backend | — | TransformerLens [replacement_model.py:29] | `transformerlens` | — | nnsight "slower and less memory-efficient" [README.md:59-62] |
| intervention default | constrained patching for validation [paper §5.3; bio] | `feature_intervention(freeze_attention=True, constrained_layers=None)` = iterative patching with frozen attention patterns [replacement_model_transformerlens.py:737-789] | — | library paper: "iterative patching by default" [lib §3.1.3] | §6.3 |

**Dependencies** [code pyproject.toml:7-23]: `transformer-lens>=2.16.0`, `transformers>=4.56.0,<=4.57.3`,
`nnsight>=0.8.0rc1,<0.9`, `huggingface_hub<1.0.0`, `torch>=2.0.0`, Python ≥ 3.10. The version is set by hatch-vcs from
git tags, with fallback 0.0.0. The Neuronpedia graph was made with `circuit-tracer` "1.0.0".

---

## 6. Their evaluation, exactly

### 6.1 Graph-level, no intervention

- **Replacement and completeness scores,** formulas in §3.4.
  - The paper reports them on the unpruned graph, and on pruned graphs with pruned nodes counted as error.
  - The paper's numbers are for the 18L model: CLT 10 M completeness 0.80, replacement 0.61; PLT 10 M 0.78 / 0.37 on one
    prompt [paper §5.2].
  - The text says averages over a pretraining prompt set appear only in plots [paper §5.2, as read].
  - **No published Gemma-2-2B value** in either paper or in the repo [lib, pp. 1-8; repo; inferred from absence].
- **Faithfulness-vs-size curve, theirs** [paper §5.2 "Graph Pruning", acronym prompt]:

  | node threshold | completeness (pruned → error) | nodes |
  |---|---|---|
  | 0.95 | 0.87 | 236 |
  | 0.90 | 0.83 | 137 |
  | 0.80 (default) | 0.70 | 55 |
  | 0.70 | 0.58 | 27 |

  - Summary: "reduce the number of nodes by an order of magnitude while reducing completeness by only 20%".
  - Pruning cuts nodes about 10× and edges about 500× [paper §5.2; App. "Graph Pruning"].
  - **This is the paper's only size curve, and it is a graph-internal score, not an intervention.**
- **Path length** (embedding → logit, mean): CLT 2.3 vs PLT 3.7 [paper §5.2].

### 6.2 Mechanistic faithfulness (interventions)

[paper §5.3; App. "Iterative Patching"]
- **Node influence vs ablation** (20 prompts): KL after ablating a node correlates best with its multi-hop logit
  influence, then its direct edge weight, then its activation.
- **Feature → feature** (20 prompts): influence vs the ablation effect on downstream features, **Spearman ρ = 0.72**.
  Ablations use constrained patching over layers [i, i+2].
- **Whole-model perturbation:** the same perturbation is applied in the local replacement model and in the real model,
  and the effects are compared by cosine similarity and normalised MSE.
  - One layer after the perturbation: cosine about 0.8, normalised MSE about 0.4.
  - Agreement degrades with depth. Magnitudes diverge more for larger dictionaries; the authors suspect the frozen
    normalisation.
- **Replacement-model quality** [paper §5.1]:
  - normalised MSE about 11.5% (18L, 10 M, L0 88) and 21.7% (Haiku, 30 M, L0 235);
  - the replacement model's top token matches the real model's on about 50% of prompts (prompts the model gets right
    with < 80% confidence).
- **Patching modes** [paper App. "Iterative Patching"]:
  - **constrained:** perturbed MLP outputs are computed once and clamped in the layers up to the intervention layer;
  - **iterative:** features downstream are recomputed from the perturbed stream.
- **Steering in the case studies** [bio; lib §4; paper App. "Details of Interventions"; demos]:
  - supernodes are clamped to negative multiples of their activation (−2×; often −4× to −6× before effects are reliable;
    −5× in the acronym case);
  - foreign features are patched in at high multiples (10×);
  - constrained to a layer range (layers 16-21 in the library's Dallas → Sacramento demo, lib §4.1);
  - read off as next-token distributions and downstream supernode activation fractions.
  - These are **manual, hypothesis-level** tests on hand-built supernodes, not a metric over a size grid.

### 6.3 What `feature_intervention` does in code

[code replacement_model_transformerlens.py:568-789]
- Interventions are tuples `(layer, pos, feature_idx, value)`. At `blocks.ℓ.hook_mlp_out` it adds
  `(value − current activation) · W_dec`.
  - For a CLT it adds into every output layer ℓ' ≥ ℓ [:648-696].
  - **The real MLP output stays live, so the transcoder error stays live.**
- `freeze_attention=True` (default) **and no `constrained_layers`:** only `hook_pattern` is frozen [:492-498]. LN scales
  stay live, as do the MLPs, so downstream features are recomputed (iterative). The docstring says LayerNorms are frozen
  too [:747-753]. The code freezes them only when `constrained_layers` covers every layer [:494-495]. See §9.6.
- `constrained_layers=range(a, b)`: the MLP outputs in that range are frozen to clean values plus the intervention
  deltas (direct effects only). With all layers in range, the LN scales are frozen too, which is the test's exactness
  setting [:494-498; tests/test_attributions_gemma.py:150-155].
- `freeze_attention=False` and no constraint: a plain live forward with deltas, i.e. **our semantics**
  (`mlp_out += (ĉ − c) W_dec`, error live, patterns live) [inferred from :614-625].

---

## 7. Per-prompt nature, aggregation, commands, memory

### 7.1 Per prompt, and how (not) aggregated

- **One graph = one prompt.** The paper calls it a "locally valid" graph [paper §7 / §8, as read]. Edges depend on that
  prompt's attention patterns and norm scales.
- **No aggregation across prompts is defined:**
  - supernodes are manual groupings within a graph;
  - cross-prompt comparisons in the case studies are informal;
  - global weights are prompt-independent transcoder virtual weights, a different object [paper §4; bio, as read].
- **The single exception:** the refusal analysis built graphs on 200 SORRY-Bench prompts and ranked features by average
  node influence [bio, as read]. That is a ranking, not a circuit.
- **Success rate:** satisfying insight on "about a quarter" of the prompts tried, stated as imprecise [bio, as read].
- **What 038 did was ours:** union over windows, survival-frequency cuts, |edge| summed over positions and prompts
  [038 README; theirs_gtc.py:115-209].

**Faithful multi-prompt protocol for arena D** (107 rule 4) [ours]:
1. For a target latent (L, t), take its subset of K training prompts. On each, choose the target position p_k (the
   max-activating position; our choice, labelled as in ge.md).
2. **AG side.** One graph per prompt.
   - Latent-rooted (A1-A3, our adaptation), plus the as-published logit-rooted row (§4).
   - Each graph is pruned over a **node-threshold sweep** at edge threshold 0.98, its native knob. Suggested grid:
     {0.1, 0.2, …, 0.9, 0.95, 0.98, 0.99}. Thresholds are an input, never fitted to a size.
   - Each threshold gives one per-prompt circuit.
3. **Evaluation:**
   - each per-prompt circuit on its own prompt and position (home turf);
   - and on the held-out prompts. Held-out is **our** reading (evaluation.md §4). The per-prompt circuit transfers as
     its position-collapsed latent set {(ℓ, f)} kept at every position, labelled ours.
4. **WCM side.** One circuit fitted on the same K prompts, swept over λ, scored on the same K and the same held-out.
5. **Curves** are read per prompt on the latent-count grid by the ≤ k rule, then averaged at each grid point
   (evaluation.md §6.4 item 10).

### 7.2 Exact commands for one prompt

**Environment** [ours]:
- a fresh venv;
- `pip install "git+https://github.com/decoderesearch/circuit-tracer@255d989e93be1299d53d67b59e9fd76a0e274b14"`;
- check `pip show -f` / `direct_url.json` for the commit. No editable install, so that 038's provenance FAIL (local edits
  to `__init__.py` and `__main__.py`, [038 faithfulness.log]) cannot recur;
- record `transformer_lens`, `transformers` and `torch` versions.

`google/gemma-2-2b` is gated. 038 loaded the weights from the `unsloth/gemma-2-2b` mirror via `hf_model=` / `tokenizer=`
kwargs to the TL loader [038 theirs_gtc.py:61-70]. This is an environment choice; record it.

**CLI** (logit-rooted, as published; Gemma PLT):

```
circuit-tracer attribute --prompt "Fact: The capital of the state containing Dallas is" \
  --transcoder_set gemma --dtype bf16 --batch_size 64 --max_feature_nodes 5000 \
  --offload cpu --lazy-encoder --graph_output_path dallas.pt \
  --slug dallas --graph_file_dir ./graph_files --node_threshold 0.7 --edge_threshold 0.9
```

- These are the Neuronpedia Dallas generation and pruning settings (§10 check F3) [__main__.py:24-142].
- `--model` is inferred from the config (`google/gemma-2-2b`) [__main__.py:236].
- For the CLT: `--transcoder_set mntss/clt-gemma-2-2b-426k`, with a smaller `--batch_size` (§7.3).

**Python** (as published, then the latent-rooted adaptation):

```python
import torch
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import prune_graph, compute_graph_scores

model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma",          # or "mntss/clt-gemma-2-2b-426k"
                                         dtype=torch.bfloat16, lazy_encoder=True, lazy_decoder=True)
ids = model.ensure_tokenized(prompt)                                            # BOS prepended
g = attribute(ids, model, max_n_logits=10, desired_logit_prob=0.95,
              batch_size=64, max_feature_nodes=8192, offload="cpu")             # AS PUBLISHED (logit root)
pr = prune_graph(g, node_threshold=0.8, edge_threshold=0.98)
rep, comp = compute_graph_scores(g)

# OUR ADAPTATION: latent root (L, p, t)
g = attribute(ids[: p + 1], model, batch_size=64, max_feature_nodes=None, offload="cpu")   # A1 + A2
rows = ct_prune.seed_rows(g, L, t)[0]          # rows of the selected features with (layer, feat) = (L, t)
rows = rows[g.active_features[g.selected_features[rows], 1] == p]                          # the target position only
pr = ct_prune.prune_rooted(g, rows, node_threshold=θ, edge_threshold=0.98)                 # A3 (038 ct_prune.py)
```

- `seed_rows` matches (layer, feature) at **all** positions. Rooting must keep only position p, so filter on the
  position column.
- In 038 the seed was rooted at every position where it was active [038 ct_prune.py:160-164]. That is a different root;
  do not reuse it unchanged [inferred].

**Interventions for validation:**
- `model.feature_intervention(ids, [(layer, pos, feat, value), ...], constrained_layers=range(a, b))` for constrained
  patching;
- `freeze_attention=False` for live iterative patching [replacement_model_transformerlens.py:737-789].

### 7.3 Memory and runtime on a 16 GB GPU (Gemma-2-2B)

**Measured** [038 logs, 16 GB card capped at 14.3 GB; 8f1e2438; PLT `gemma`; bf16; batch 64; 64-token windows;
`max_feature_nodes = 16384`; no offload]:
- resident after build: **6.10 GB** with lazy encoder and decoder, **7.99 GB** with eager encoders;
- graphs completed for every window. Adjacency 18,122² (16,384 + 27 × 64 + 10); 58-67 k active features per window;
- no failures logged;
- **no timings recorded.**

**Published:**
- "at least 15 GB VRAM" for Gemma-2-2B [lib §3.3];
- Colab T4 (15 GB VRAM, 12.7 GB RAM) works with the demo settings: bf16, batch 256, `max_feature_nodes` 8,192,
  `offload="disk"` [lib §3.3; README.md:18; attribute_demo cell 6];
- a saved graph is about 167 MB for the Dallas prompt [attribute_demo md 9].

**Estimates from tensor shapes** [inferred; bf16 = 2 B, fp32 = 4 B; n = positions, b = batch_size, N = active
features, F = selected]:

| item | formula | PLT example | CLT-426k example |
|---|---|---|---|
| model, TL bf16 | measured | 6.1 GB (incl. PLT biases/thresholds) | ≈ 6.1 GB |
| encoders, eager | 26 × 16,384 × 2,304 × 2 B | 1.96 GB (lazy → 0; 038 found no difference) | 1.96 GB |
| decoders | lazy, read per access from disk | per-layer slices | **must be lazy**: 26.5 GB on disk; put it on a local SSD (038: slow on the `/mnt/x` mount) |
| active-feature vectors | encoder_vecs N·d·2 + decoder_vecs (PLT N·d·2; CLT Σ (26−ℓ)·d·2 ≈ N × 62 KB) | N = 11 k (11 tokens): 0.1 GB; N = 64 k: 0.6 GB | N = 10 k: ≈ 0.6 GB; 60 k: ≈ 3.7 GB |
| cached stream for backward | 26 × b × n × 2,304 × 2 B (+ attention patterns 8·n²·2 B per layer per copy) | b = 64, n = 64: 0.5 GB; b = 256, n = 64: 2.0 GB | same |
| **score-hook gather (transient, per layer)** | b × N_ℓ × 2,304 × 2 B, N_ℓ = decoder entries writing into layer ℓ [context_transformerlens.py:107-112, 128] | PLT N_ℓ ≈ N / 26: b = 64, N = 64 k → 0.7 GB | **CLT: at the last layer N_ℓ = all active features** → b = 64, N = 10 k → 3.0 GB; b = 256 → 11.8 GB (**OOM**) |
| CPU edge buffer | (F + K) × (N + 27n + K) × 4 B [attribute_transformerlens.py:187] | F = 16 k, n = 64: 4.35 GB RAM | same shape |
| GPU influence re-rank (only if F < N) | copy of `E[:rows]` in fp32 [graph.py:393] | up to 4.35 GB GPU at the end of phase 4 | same |
| final adjacency | (F + 27n + K)² × 4 B | 18 k² → 1.3 GB; all features on a 32-token prompt (≈ 33 k) → 4.4 GB | — |

**Memory-safe setups for 16 GB** [ours; PLT from 038 measurements, CLT estimated]:

| run | settings | why |
|---|---|---|
| PLT, logit-rooted, ≤ 64 tokens | bf16, lazy_encoder + lazy_decoder, batch 64, `max_feature_nodes` 8192 (demo) or 16384, offload `"cpu"` optional | measured in 038 without offload |
| PLT, latent-rooted (A2: all features) | bf16, lazy both, batch 64, **target position ≤ 31** (≤ 32 tokens), offload `"cpu"`, ≥ 16 GB host RAM | N ≈ 32 k: CPU edge buffer 4.4 GB + final 4.4 GB; about 500 backwards. Longer prefixes need the A3 alternative (target-ranked cap) |
| CLT-426k, either root | bf16, lazy decoder (mandatory), lazy encoder, **batch 16-32**, offload `"cpu"`, prompts ≤ 32 tokens to start | the CLT gather at late layers scales with all active features. Measure the CLT's N per position first: its L0 is undocumented on HF |
| CLT-2.5M | not on this machine | 159 GB of decoders on disk; 11.8 GB of encoders |
| fp32 | not on 16 GB | model alone about 10.5 GB [inferred] |

**Runtime.** Not measured anywhere I could read. A backward pass skips the MLPs, so its cost is mainly attention-OV and
norm gradients plus the per-layer gathers [inferred]. Expect seconds to minutes per graph. **Measure** wall-clock and
`torch.cuda.max_memory_allocated` per graph in the smoke test (evaluation.md §5 fields), at n ∈ {16, 32, 64} and
b ∈ {16, 64, 256}, before fixing the protocol.

---

## 8. WCM on the same substrate (Gemma-2-2B + the same transcoders)

1. **Nodes.**
   - **PLT:** a node is the latent (ℓ, f), ℓ ∈ 0..25, f < 16,384.
   - **CLT:** a node is the latent (ℓ_enc, f). One mask gate multiplies the feature's code, and its effect is applied
     through **all** of its decoders (ℓ' ≥ ℓ_enc). This is the same node definition as circuit-tracer's, where one node
     is one (encoder layer, pos, feature) with all decoders summed [§2.3].
   - Masking decoders per output layer separately would give different nodes from AG's. That would be ours, so do not
     do it.
2. **Universe for target (L, t):**
   - PLT: features at layers < L. The target reads `ln2` of layer L, so its own layer's MLP is after the read.
   - CLT: features with encoder layer < L (their decoders write into MLP outputs < L, which resid_mid of L contains).
   - Attention, embeddings, BOS and errors are not nodes. This matches the AG edges into the target exactly: rows only
     have non-zero entries from features below L, tokens and errors [inferred from §3.1].
   - 038 showed that ablating every upstream PLT feature leaves the target at 0-75% of natural [038 README "e0 is not
     zero"]. Report universe-relative necessity.
3. **Intervention semantics.** An untouched circuit must reproduce the model exactly.
   - PLT (038's Runner, certified by its `check_gtc.py`) [038 ours_gtc.py:16-20, 251-311]:
     - READ c = JumpReLU(W_enc · x + b_enc) at the transcoder input. In HF this is
       `pre_feedforward_layernorm` output ÷ (1 + w_pre), i.e. TL `ln2.hook_normalized`;
     - WRITE `y ← y + (ĉ − c) · W_dec` at the `post_feedforward_layernorm` output, i.e. TL `hook_mlp_out`;
     - the real MLP runs, so the error stays live.
   - CLT [ours, same principle]:
     - READ at `hook_resid_mid` of ℓ: HF = the residual after the attention block, i.e. the input to
       `pre_feedforward_layernorm`;
     - c = ReLU(W_enc,ℓ x + b_enc,ℓ);
     - WRITE at every ℓ' ≥ ℓ: `y_ℓ' ← y_ℓ' + (ĉ − c) · W_dec[ℓ → ℓ']`;
     - c is read from the live (already perturbed) stream, as in `feature_intervention` [code
       replacement_model_transformerlens.py:648-696].
   - **Identity test before anything else:** all masks open gives max |Δ logits| = 0 (or below bf16 noise), as 038
     certified for the PLT.
4. **Target read.** The target is the **pre-activation** W_enc[t] · x + b_enc[t] at its read hook.
   - This is the same quantity an AG feature row explains [§3.1].
   - It is also what our engine's seed tap captures (`w.x + b`) [src/circuit/instrument/learned_mask.py:315-319].
5. **What the engine needs.** The current engine masks SAE sites, which re-encode the stream and write
   `decode(masked code) + error` back at the same site [learned_mask.py:315-319]. Arena A's GPT-2 transcoder adapter adds
   a read/write-split site ("tc" kind) [107 README arena A; ge.md §9]. For Gemma-2-2B:
   - **Hooks:** four norms per block. Read and write go at the norm outputs above, with Gemma's (1 + w) RMSNorm weights
     handled as 038 probed (dividing w out at the input is mandatory; folding it is "silently wrong") [038 ours_gtc.py:
     7-14, 40-46].
   - **Codes:**
     - PLT: JumpReLU with per-feature thresholds, 16,384 per layer, `activation_function.threshold` in the safetensors.
       W_enc and W_dec are [d_sae, d_model] in circuit-tracer's layout [038 ours_gtc.py:219-232; code
       single_layer_transcoder.py:69-76];
     - CLT: ReLU, `W_enc_ℓ` [16,384, 2,304], `W_dec_ℓ` [16,384, 26−ℓ, 2,304] [code cross_layer_transcoder.py:93-117].
   - **Floors (Z / A / C / R):** the same machinery over transcoder codes. A sparsity-preserving mean over active entries
     per feature, contrast fills from contrast contexts, resample from donors.
   - **Position 0:** exclude BOS from reads, means and FVU (038: BOS dominates a naive FVU) [038 ours_gtc.py:12-14].
   - **CLT memory:**
     - masking all upstream features needs the decoders from layers < L into layers < L: 75.5 MB × L(L+1)/2 in bf16.
       That is 1.6 GB at L = 6, 5.9 GB at L = 12 and 15.9 GB at L = 20 [inferred];
     - on 16 GB, CLT targets are therefore limited to roughly L ≤ 12, unless decoder rows are gathered per batch for
       the features with ĉ ≠ c only (as circuit-tracer does), or the run goes to a pod;
     - PLT decoders total 1.96 GB in bf16, which fits at any L.
   - **Implementation drift:** our engine is HF, circuit-tracer is TL. 038 measured TL vs HF stream differences of about
     4e-4 after layer-0 attention, and 0.1-0.9% JumpReLU support flips [038 README "Agreement gate"].
     - Score both methods' circuits through **one** implementation for "our tests" (ours, HF).
     - Score both through **TL graphs** for "their evaluation".
     - State the drift budget, not bit-exactness.
6. **Targets.** WCM target = transcoder latent (L, t), fitted across contexts. AG root = node (L, p_k, t) on each prompt
   k, which **exists only where the latent is active** (a > 0; JumpReLU: pre-activation > threshold).
   - Inactive prompts have no AG row and cannot be rooted. Count them.
   - Contrast contexts (WCM's C fill) have no AG counterpart.
   - **Layer selection:** stratify targets by layer. For the CLT, prefer layers where the WCM decoder budget fits
     (item 5).

---

## 9. Ambiguities and paper-vs-code differences

1. **"Embedding and error nodes are never pruned"** [paper App. "Graph Pruning"] vs code. Only tokens and logits are
   force-kept, and error nodes can be cut by the node threshold or the outgoing-edge rule [code graph.py:275, 290, 303].
   Run the code as is and count kept error nodes. The Neuronpedia Dallas graph kept 137 of 286 error nodes, which shows
   the code's behaviour.
2. **No pruned-graph scores in code.** The paper scores pruned graphs with pruned nodes counted as error [paper §5.2];
   `compute_graph_scores` only scores the unpruned adjacency [code graph.py:316-363]. Implementing the pruned version is
   **our implementation of their metric**. Formula: Â on the full graph, dropped feature nodes' columns added to the
   error mass of each row.
3. **Unexplored features.** Paper: treated as errors in adaptive mode [paper App.]. Code: their columns are deleted when
   `max_feature_nodes < N` [code attribute_transformerlens.py:252-256]. Row normalisation then spreads that missing mass
   over the remaining sources, which can **inflate replacement and completeness** [inferred]. Report scores only with
   `max_feature_nodes = None`, or flag the cap.
4. **Completeness counts sourceless rows as complete.**
   - Error and token rows have no inputs, so their normalised row is all zero and their "non-error fraction" is 1
     [code graph.py:359-361].
   - Error nodes carry logit influence, so they add to the numerator. The paper's wording ("fraction of input edges …")
     is silent on nodes with no inputs [inferred].
   - Use the code's definition for "their score", and note it.
5. **Pre-activation identity.** "pre-activation = sum of incoming edges" [paper §3.1] holds up to biases and the
   BOS-position MLP constants (§3.1).
6. **`feature_intervention` docstring vs code.** The docstring says that with `freeze_attention` the "MLPs and
   LayerNorms" are frozen; the code freezes only attention patterns unless `constrained_layers` is set
   [replacement_model_transformerlens.py:492-498 vs 747-753]. State the exact flags with every intervention result.
7. **Default patching.** The paper validates with constrained patching [paper §5.3; bio]. The library's default is
   iterative with frozen patterns [lib §3.1.3; code]. Their own "steering" evaluation therefore depends on the flags.
   Use constrained for their-evaluation rows, as the papers do.
8. **Three feature caps** (None / 7,500 / 8,192, plus Neuronpedia 5,000) and **three batch defaults** (512 / 256 / 64).
   Always pass them explicitly.
9. **Pruning thresholds in use differ from the default:**
   - Neuronpedia Dallas: 0.7 / 0.9;
   - the library paper's listing: 0.8 / 0.95;
   - demo prose: 0.8 / 0.98;
   - demo cell: 0.7 / 0.95.

   The paper's default is 0.8 / 0.98.
10. **CLT activation.** The paper's CLTs are JumpReLU; the shipped Gemma CLTs are ReLU [inferred from file sizes; lib
    §3.2.2]. ReLU codes are less sparse at low activations, so expect larger N per position [inferred]. Measure L0.
11. **"Lowest L0" label.** The Gemma PLT config comment says "lowest L0", but layers 0-10 use L0 49-95 [HF config].
    Irrelevant to fidelity, since we use the set as shipped. Do not describe it as lowest-L0.
12. **Seed rooted at all positions vs one.** 038's `seed_rows` matched the latent at every active position. A per-prompt
    root at p is one row. §7.2 code filters on the position.
13. **Paper numbers came through a summariser.** The pruning table and score phrases were re-fetched verbatim; the
    remaining §5 numbers (0.72, 0.8 cosine / 0.4 NMSE, 11.5% / 21.7%, about 50% top-token match) should be checked by
    eye before they are quoted in the paper.

---

## 10. Fidelity checks (does our run of their code match theirs?)

Tolerances are **our choice**. Neither paper gives any for Gemma.

| # | check | reference | pass if |
|---|---|---|---|
| F1 | Provenance: installed at `255d989` from git, no local edits | §7.2; 038's FAIL [038 faithfulness.log] | `direct_url.json` commit matches; `git diff` empty if a clone is ever used |
| F2 | **Exactness of edges** (their own test, on the real Gemma graph): for 100 random selected features, `feature_intervention(... value = 2a, constrained_layers = range(26), apply_activation_function = False)` changes every selected pre-activation and demeaned logit by the feature's adjacency column. Repeat for error and token columns | code tests/test_attributions_gemma.py:29-188 | allclose at their tolerances (act atol 5e-4, logit atol 1e-5) in fp32. In bf16, report the max relative error instead of pass/fail |
| F3 | **Neuronpedia Dallas graph reproduced** (PLT "gemmascope-transcoder-16k"; prompt "Fact: The capital of the state containing Dallas is"; settings max_n_logits 10, desired 0.95, `max_feature_nodes` 5000; prune 0.7 / 0.9) | Neuronpedia JSON (circuit-tracer 1.0.0): **697 feature nodes (656 distinct (layer, feature)), 137 error nodes, 11 embeddings, 10 logits, 34,801 edges; top logit " Austin" p = 0.450**, then " not" 0.073, " the" 0.055, " Texas" 0.040 | logit probs within ±0.01; feature-node Jaccard ≥ 0.9 against the JSON node ids (`layer_feature_pos`); node count within ±10%. Then the frontend filter (nodes with no kept edges dropped) [code create_graph_files.py:85-116] |
| F4 | The README's 22 pinned Dallas nodes are active and survive pruning at 0.6 / 0.99 and 0.8 / 0.98 | README.md:14 link; passed in 038 at 8f1e2438 (22/22) [038 faithfulness.log] | 22 / 22 |
| F5 | `prune_rooted(logit root) == prune_graph` bit-exactly | 038 ct_prune.py:148-157; passed in 038 | equal masks |
| F6 | Batch and lazy invariance | 038: batch 64 vs 256 pruned Jaccard 0.994, edge corr 0.99998; lazy vs eager identical | pruned Jaccard ≥ 0.95, edge corr ≥ 0.999 |
| F7 | Truncation invariance (adaptation A1): the target's row from the truncated prompt equals its row in the full-prompt graph with `max_feature_nodes = None` | [inferred: causal model] | edge corr ≥ 0.999 on the shared columns |
| F8 | Their pruning curve shape on a known prompt: completeness of the pruned graph (our implementation, §9.2) falls by about 20% from unpruned to the 0.8 cut, with about 10× fewer nodes | paper §5.2 table (18L, acronym prompt; a different model) | qualitative only: monotone, same order of magnitude. Not a numeric pass; no Gemma reference exists |
| F9 | Replacement and completeness of unpruned Gemma graphs, PLT vs CLT-426k, on the same prompts | paper §5.2: CLT > PLT in replacement (0.61 vs 0.37 on its model) | CLT replacement > PLT replacement on most prompts (qualitative; no Gemma numbers published) |
| F10 | Our transcoder adapter vs theirs: per-layer FVU / L0 at pos ≥ 1 through our HF hooks match circuit-tracer's `get_activations` | 038 probe: FVU 0.157, L0 83 vs 88 | support Jaccard ≥ 0.99 per layer (038 drift budget: flips < 5%) |

**Top check: F3.** It is the only published Gemma artefact with exact settings and counts. F2 is the strongest internal
check, and it uses their own test.

---

## 11. Size units: AG vs distinct latents

- **AG native count:**
  - feature nodes (layer, pos, feature);
  - error nodes (layer, pos), at most 26 × n;
  - token nodes n, always kept;
  - logit nodes K, always kept. In the latent-rooted adaptation the root replaces the logits.
  - The paper's node counts (55 at the 0.8 cut) appear to include all kept node types [paper §5.2; inferred]. Report
    every type separately.
- **Mapping onto evaluation.md §6.1** (unit = distinct `(site, latent)`, position-collapsed):
  - `n_latent = |{(layer, feature) : (layer, pos, feature) kept}|`, with the root excluded;
  - for a CLT, layer = encoder layer;
  - Neuronpedia Dallas example: 697 feature nodes → 656 distinct latents.
- **Extra columns, never on the latent axis:**
  - kept error nodes;
  - kept token nodes;
  - the native node count;
  - the kept edge count.
- **For the latent-rooted graph,** count only ancestors of the root in the kept edge graph (038 `seed_circuit`
  [ct_prune.py:167-188]). Prune's cleanup should already guarantee this; check it.
- **WCM:**
  - its count is already distinct (ℓ, f), position-free;
  - when a WCM circuit is scored by their completeness (§12 item 3), the kept AG nodes are WCM's latents at **every
    position where they are active** on that prompt. Report that position-expanded count too.

---

## 12. Mapping onto arena D: decisions to take

1. **Rows:**
   - D1: AG as published (logit root);
   - D2: AG latent-rooted (A1-A3, **our adaptation**);
   - D3: WCM on the same transcoders.
   - D1 is reported but has no size knob relative to the latent. It reports seed survival and ancestors only.
2. **Dictionaries:** PLT (`gemma`) first, which is measured to fit. Then CLT-426k, with the batch and memory limits of
   §7.3 and the WCM decoder budget of §8.5. Never compare across the two sets (107 rule 2).
3. **Their evaluation on every row** (rule 5):
   - completeness and replacement of the AG graph pruned to the row's node set, pruned → error (§9.2). For WCM rows the
     node set is WCM's latents at their active positions; **ours** to apply to WCM;
   - the influence-vs-ablation Spearman on the root (their §5.3 check, with the latent as root; **ours** for the root).
4. **Our evaluation:** Z / A / C / necessity / R in the live HF model, error kept live, on the K prompts and held-out.
   Ablation: AG members kept at clean, all other universe latents filled. Per-prompt circuits are applied
   position-collapsed on held-out (ours).
5. **Open:** K per target; the position rule p_k; the maximum prompt length for A2 (32 suggested); whether to run the A3
   alternative (target-ranked cap) for longer contexts.

---

## Sources

- Methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Biology: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Library paper: https://aclanthology.org/2025.blackboxnlp-1.14/
- Code: https://github.com/decoderesearch/circuit-tracer @ 255d989e93be1299d53d67b59e9fd76a0e274b14 (formerly
  safety-research/circuit-tracer)
- Transcoders: https://huggingface.co/mwhanna/gemma-scope-transcoders, https://huggingface.co/mntss/gemma-scope-transcoders,
  https://huggingface.co/mntss/clt-gemma-2-2b-426k, https://huggingface.co/mntss/clt-gemma-2-2b-2.5M,
  https://huggingface.co/mntss/transcoder-Llama-3.2-1B, https://huggingface.co/mntss/clt-llama-3.2-1b-524k
- Neuronpedia graph: https://www.neuronpedia.org/api/graph/gemma-2-2b/gemma-fact-dallas-austin
- Local: `experiments/038-transcoder-compare-gemma/`, `dev-notes/2026-09-28-baseline-audit.md` §5, 107 `README.md`,
  `specs/evaluation.md`, `specs/sfc.md`, `specs/ge.md`
