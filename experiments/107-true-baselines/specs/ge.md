# Spec: Ge et al. 2024, hierarchical attribution on a linear computation graph (arena A)

Status: specification for a faithful re-implementation. Written 2026-11-01 (agent `spec-ge`). Nothing here is built.

**Paper.** Xuyang Ge, Fukang Zhu, Wentao Shu, Junxuan Wang, Zhengfu He, Xipeng Qiu, "Automatically Identifying Local
and Global Circuits with Linear Computation Graphs", arXiv 2405.13868, ICML 2024 Mechanistic Interpretability
Workshop.
- v1: 22 May 2024. https://arxiv.org/html/2405.13868v1
- v2: 21 Jul 2024, current. https://arxiv.org/html/2405.13868v2

**Tag key.**
- [v1] / [v2] mark the paper version. "§" is a paper section, "Alg. 1" is Appendix C.
- [code ...] cites the OpenMOSS library port described in §0.
  - Line numbers are approximate. They were counted by the fetch tool from raw files and may be off by ±2.
  - SHAs are full commit SHAs.
- [inferred] is my reading, not stated anywhere.

---

## 0. Sources and code status (read this first)

### 0.1 Paper versions: v1 vs v2

The method, Alg. 1, Theorem 3.1, the appendices and all tables are the same in both versions. The differences are
naming and figures:

| item | v1 | v2 |
|---|---|---|
| MLP replacement name | "Skip SAE" (§2.2 title: tackling MLP non-linearity with Skip SAEs) | "Transcoder" (§2.2) |
| Skip-SAE / transcoder figure | Fig. 2: Skip SAE schematic + (b) Skip SAEs match or beat MLP-output SAEs in 8 of 12 layers | figure dropped |
| hierarchical vs standard plot | **Fig. 3b** | **Fig. 2b** |
| case-study figures | Figs. 4–6, App. E Fig. 8 | Figs. 3–5, App. E Fig. 7 |
| Appendix A wording | "Skip SAE" | still says "Skip SAE" in App. A, a leftover of the v1 naming |

So "Fig. 2b" in our README is v2 numbering, and it is v1's Fig. 3b. In this spec, figure numbers are v2 unless
marked.

### 0.2 Code: a library port exists. No paper-experiment code, and no weights.

The paper links no code [v1, v2; arXiv abs page; paperswithcode.co/paper/2405.13868].

The Language-Model-SAEs README (repo now also called OpenMOSS/Llamascopium) has a news entry dated 2024.5.22
pointing to this paper as "hierarchical tracing". The code that implements Alg. 1 was added to that repo
**after** v1:

| commit | date | what |
|---|---|---|
| `3ff7275f4d033da85867054fdd9b11b455731b82` | 2024-06-19 | "feat(circuit): add basic attributors". `src/lm_saes/circuit/attributor.py`: `TransformerHierachicalAttributor`, `TransformerDirectAttributor`, `apply_saes` |
| `f9cc2d656d0b82d2275ead248e1f708d7a7ec363` | 2024-06-20 | "refactor(circuit): decouple attributor with saes". Becomes `attributors.py` (`HierachicalAttributor`), `context.py` (`apply_sae`, `detach_at`), `graph.py` (`Node`) |
| `67bfb0b9f835f772f3af2aa176745c6b6135373a` | tag v0.1.0, committed 2024-09-12 | same module, plus `transformer.py` wrappers, `tests/intergration/test_attributor.py`, and `server/app.py` `/model/trace` |
| `e20105818d97169090856ce0485b9767e95b7bb3` | tag v1.1.0 | `circuit/` holds only `context.py`. **The attributors were removed** |
| `dev` (2026) | current | `src/llamascopium/circuits/attribution.py` is a circuit-tracer-style attribution graph: `torch.autograd.grad` over all sources, then cumulative-influence node/edge pruning (defaults 0.6 / 0.8). **No in-pass τ gating. This is not Alg. 1** |

All commits are by `dest1n1s` (Dest1n1). That this is first author Xuyang Ge is [inferred, unverified].

URLs, read via WebFetch only:
- https://github.com/OpenMOSS/Language-Model-SAEs/blob/3ff7275f4d033da85867054fdd9b11b455731b82/src/lm_saes/circuit/attributor.py
- https://github.com/OpenMOSS/Language-Model-SAEs/blob/67bfb0b9f835f772f3af2aa176745c6b6135373a/src/lm_saes/circuit/attributors.py
- …/67bfb0b…/src/lm_saes/circuit/context.py, graph.py, transformer.py; src/lm_saes/sae.py; src/lm_saes/utils/hooks.py;
  tests/intergration/test_attributor.py; server/app.py

**Does it implement Alg. 1 as published?** Yes for the gating rule, with four deviations:

1. **Gating rule, same as the paper.** It is a backward hook on every candidate tensor (the SAE / transcoder
   `hook_feature_acts`): `torch.where(v * grad > threshold, grad, 0)`. Here `v` is that tensor's forward value
   [code 67bfb0b attributors.py:157-159; 3ff7275 attributor.py:215-225].
   - The comparison is signed, with no `abs`.
   - It is elementwise over every (batch, position, feature) of the tensor.
   - Autograd calls the hook only after the tensor's incoming gradient is fully accumulated. That is exactly
     "accumulate from all successors, then gate" in reverse topological order [inferred from autograd semantics].
2. **Inequality.** The code keeps a node when `>` τ. The paper detaches when `< τ`, so it keeps `≥` τ. This only
   matters at ties.
3. **Default τ.** `threshold` defaults to `0.1`, annotated `int` [code 67bfb0b attributors.py:149; 3ff7275
   attributor.py:212; server/app.py `tracing_threshold: float = 0.1`].
4. **Output.** The output is a star: the target plus the nodes with attr > τ, and edges node → target in the
   refactored version. It is not a wired graph. The 3ff7275 version adds no edges at all [code 3ff7275
   attributor.py:236-243].

**What the code does to make the graph linear:**
- **Errors.** Every dictionary is spliced in as `recon + (tensor - recon).detach()`. The forward value equals the real
  activation, and the error is a gradient-dead constant [code 67bfb0b context.py:21, 34].
- **Transcoders.** The input hook stores x and the output hook computes `sae.forward(x, label=mlp_out)` [code 67bfb0b
  context.py:23-35].
- **Attention.** `blocks.{i}.attn.hook_attn_scores` gets `detach_hook` (`tensor.detach().requires_grad_(True)`). The
  pattern is constant, and the scores become gradient sinks that can be read as QK nodes [code 67bfb0b
  transformer.py:70; utils/hooks.py:21-25].
  - In the v0.1.0 wrappers the default candidates are all `hook_feature_acts` **plus** all `attn.hook_attn_scores`
    [code 67bfb0b transformer.py:72-73].
  - In 3ff7275 the candidates are SAE feature acts only [code 3ff7275 attributor.py:226-238].
- **LayerNorm: NOT frozen in the code.** No `ln*.hook_scale` is detached anywhere in the circuit module or the server
  [code 67bfb0b context.py, transformer.py, server/app.py]. See ambiguity A3.
- **SAE input normalisation: not detached in the code.**
  - `compute_norm_factor` computes `sqrt(d_model)/‖x‖` per token for `"token-wise"`. It has no detach [code
    67bfb0b sae.py:153-174].
  - `encode` multiplies x by the factor of x, then divides the pre-activation by the factor of `label`. For a
    transcoder, `label` is the true MLP output [code 67bfb0b sae.py:~262-287].
  - If `token-wise` was used, the graph is not strictly linear, and gradient leaks into the real MLP through the label
    norm. With `dataset-wise` (a constant) it is linear. Which mode their GPT-2 dictionaries used is unknown. See A4.

**Weights.**
- No GPT-2 dictionaries from this paper are public. I checked the Hugging Face org `OpenMOSS-Team`: 193 models, none
  GPT-2 / SAE / transcoder for GPT-2. The OpenMOSS SAE releases are Llama Scope, Lorsa and others.
- The paper links none.

**Third-party re-implementations.** None found (web search; paperswithcode lists no repo). OpenReview forum
`b8sq8Y5VFo` exists but was behind a browser challenge, so its reviews were **not read**.

### 0.3 What 034 / 083 / 098 did. This was withdrawn, so do not repeat it.

034 `runner.py`, 083 `ge_hier.py` and the 098 `ge` stage are an **adaptation**, not this algorithm:
- Top-K residual / attention / MLP-output SAEs, not transcoders.
- Gradients through the real MLPs, LN and attention.
- Site-level gating: one backward per site, gating applied between sites.
- `|attr| ≥ τ`, not signed.
- Root = encoder pre-activation.
- Attribution summed over 48 contexts and all positions.
- τ bisected to WCM's size.

Every one of these differs from the paper (see §3). Their paper notes (034 README "Their reported results") are
consistent with what I read, except for one point. The claim "no node counts reported anywhere" is true in the text,
but Fig. 2b's x-axis *is* the number of remaining nodes [v2 Fig. 2 caption].

---

## 1. Summary of the method and its claims

**The linear computation graph** [v2 §1, §2]:
- Dictionaries replace or decompose every module that writes to the residual stream of GPT-2 Small:
  - an SAE on the word embedding;
  - SAEs on the attention outputs;
  - transcoders (v1: "Skip SAEs") that read the MLP input and reconstruct the MLP output.
- With attention patterns held fixed, every edge between active features is linear:
  - OV edges: per head, with weight pattern × W_O^h W_V^h;
  - transcoder edges: weight (W_E^T W_D^S), input-invariant.
- So for a fixed input, the graph over **active features** is linear up to each node's own ReLU [v2 §2.2–2.3, §3
  Formulation].
- The paper claims this needs no linear approximation, and that one forward plus one backward pass suffices
  [v2 §1, Fig. 1].

**Attribution** [v2 Def. 3.2]: attr(v, t) = a_v · ∂a_t/∂a_v.

**Hierarchical attribution** [v2 §3, Alg. 1]:
- This is the main algorithmic contribution.
- During the single backward pass from root t, any node whose attribution falls below τ has its gradient zeroed at
  once.
- So no predecessor is credited through it.
- Standard attribution, by contrast, prunes only after the full backward pass [v2 Fig. 2 caption].

**Theorem 3.1** [v2 §3, App. D]:
- Claim: for any subgraph G' obtained by detaching nodes, the root activation equals the sum of the leaf
  attributions. a_t = Σ_{v: deg_in(v)=0} attr(v, t).
- Proof: an active node is linear in its inputs (the ReLU is the identity when a_u > 0). Unrolling the recursion gives
  the sum.
- Use: the authors use it as the faithfulness measure. It needs no intervention, which they argue avoids
  backup / hydra effects [v2 §3 Evaluation].

**Local vs global circuits.**
- The paper never defines "global circuit". The word appears only in the title [v1, v2].
- The abstract and text say end-to-end circuits (root = a logit) vs **local circuits** (root = an intermediate SAE
  feature, the "subgraph activating a given SAE feature") [v2 §4 intro].
- Both use the same graph and the same Alg. 1. Only the root differs [inferred, no separate global procedure is
  described].

**QK** is handled as a separate, later step:
- after the OV / transcoder attribution, "important" attention scores are attributed through the bilinear QK form
  (Eq. 4) to pairs of upstream features;
- those features can in turn be attributed back through OV / transcoders, recursively [v2 §2.3, Fig. 1 step 3].

**Claims tested.** There is one quantitative experiment: Fig. 2b. Hierarchical attribution beats standard attribution
on logit recovery at every sparsity, on 20 IOI samples × 30 thresholds. The rest is qualitative case studies:
bracket, induction, IOI. There are no interventions, no held-out data, and no comparison with other methods
[v2 §3–5, §7].

---

## 2. Substrate

### 2.1 Model and dictionaries [v2 §2.1, App. A]

- **Model:** GPT-2 Small (12 layers, d_model 768, 12 heads). It is the only model in the paper.

**Dictionaries trained:**
1. **Attention-output SAEs.** One per layer, 12 in all, on the attention output. Whether this is before or after W_O
   is not stated; "attention output" in TransformerLens is `hook_attn_out`, which is after W_O, summed over heads,
   with b_O [inferred]. They are not per-head.
2. **Transcoders (v1: Skip SAEs).** One per MLP, 12 in all.
   - Input: "residual stream before the MLP" (the pre-MLP activation). Whether this is before or after ln_2 is not
     stated.
   - Target: the MLP output (Eq. 3).
3. **Residual-stream SAEs.** 24 of them, labelled RPr and RM in Table 3 (resid_pre and resid_mid, layers 0–11)
   [inferred labels].
   - They are trained and reported, but **they are not nodes in the linear graph.**
   - Eq. 2 decomposes the residual as the sum, over upstream SAEs on the modules that *write* to the stream
     (embedding, attention outputs, MLP outputs), of their features plus their errors, plus the positional
     embedding p.
   - Adding residual SAEs would decompose the same vector twice. No case study names an RPr / RM feature
     [inferred from Eq. 2 and §4–5].
4. **A word-embedding SAE.** Eq. 2's "upstream SAEs" includes one on the word embeddings, and its features are the
   "interpretable leaves" [v2 §2.1, §3 Two Types of Leaf Nodes].
   - No table reports it.
   - It is unclear whether it is a separate SAE or L0RPr. L0RPr is resid_pre at layer 0 = token + position, so it
     would mix in p, which the paper keeps as a separate leaf.
   - See A6.

**Architecture and training** [App. A, identical in v1 and v2]:

| setting | value |
|---|---|
| form | f = ReLU(W_E x + b_E), x̂ = W_D f (Eq. 1). Transcoder: same but reconstruct y = MLP out (Eq. 3) |
| size | 24,576 features (32 × 768) for every dictionary |
| loss | MSE (normalised by output variance) + L1 on f (8e-5; 1.2e-4 for attention-output SAEs) + ghost gradients |
| optimiser | Adam, lr 4e-4, betas (0, 0.9999), batch 4,096 |
| data | 1B tokens of OpenWebText, first 256 tokens per sequence (shorter ones dropped), shuffled activation buffer |
| input normalisation | inputs rescaled to norm √768. Token-wise vs dataset-wise is not stated (A4) |
| decoder | untied; **decoder / pre-encoder bias removed** "to simplify circuit analysis"; decoder column norms clipped to ≤ 1 each step |
| pruning (A.1) | drop features with decoder norm < 0.99, max activation < 1, or frequency < 1e-6 |
| finetune (A.2) | after pruning, finetune decoder + a per-feature activation scaler on MSE only; encoder frozen (counters L1 shrinkage) |
| compute | ~20 h per SAE on one A100-80GB |
| quality | L0: attention 22.7–127.2; transcoders 19.6–94.9; residual 6.9–126.2. Variance explained: attention 69.2–92.3%; transcoders 72.8–94.2%; residual 85.4–99.0% (Tables 1–3) |

### 2.2 What is frozen or detached

| component | paper | code (library port) | spec decision |
|---|---|---|---|
| attention pattern | held fixed: "fixed attention scores" makes every edge linear [v2 §3]; Eq. 5 takes the pattern as given | `hook_attn_scores` detached, made a leaf that still takes grad [code 67bfb0b transformer.py:70] | **freeze** (detach scores or pattern) |
| SAE / transcoder errors | error nodes are zero-in-degree, uninterpretable leaves; "all SAE error nodes are detached first" [v2 §3, fn. 1] | `recon + (x − recon).detach()` [code context.py:21, 34] | **detach** (constant leaf, no upstream gradient) |
| LayerNorm scale (ln_1, ln_2, ln_final) | not mentioned. App. B's general rule: treat any input-dependent matrix W(x) as constant, then attribute linearly. LayerNorm is named there as a non-linearity [v2 App. B] | **not detached** | **freeze** (A3). Required for Theorem 3.1 to hold exactly |
| dictionary input-norm factor | unstated (A4) | not detached | **freeze** (or it is a constant anyway) |
| ReLU gates of the dictionaries | each node is a ReLU of its weighted inputs; only active nodes (a > 0) are in the graph [v2 §3, App. D] | autograd through `clamp(min=0)` | natural: the gradient is 0 at inactive features, so no special handling |
| MLP | replaced by the transcoder on the gradient path; the real MLP never enters the backward pass | same, except the label-norm leak (A4) | the real MLP output only enters through the detached error |

### 2.3 OV vs QK

- **OV** (Eq. 5): Attn(x)_i = Σ_h Σ_j A^h_ij W_O^h W_V^h x_j, with A fixed.
  - Heads are **not nodes**. Nodes are attention-output SAE features.
  - A head appears as one term of an edge's weight. The edge from upstream feature (pos j, S, p) to attention feature
    (pos i, L-attn, q) decomposes as Σ_h A^h_ij (W_E^{L-attn} W_O^h W_V^h W_D^S)_{q,p}, with LN scale frozen and LN
    folded [inferred from Eq. 2, 5].
  - Per-head shares such as "83.8% through L1A.H1" are this decomposition [v2 §4.1].
- **QK** (Eq. 4): the attention score is bilinear in the features at query position i and key position j:
  - score^h_ij = Σ_{S,T} Σ_{p,q} f^S_{i,p} f^T_{j,q} (W_D^S_p W_Q^{hT} W_K^h W_D^T_q^T);
  - plus bias terms; the bracket case credits the **query bias** [v2 §4.1];
  - with LN scale frozen [inferred].
- **Order** [v2 §2.3, last paragraph]:
  1. attribute through OV + transcoders (Alg. 1);
  2. pick the important attention scores;
  3. attribute each score through QK to feature pairs;
  4. optionally continue those features' attribution through OV / transcoders, and repeat.
- No threshold or selection rule for "important attention" is given (A9).

**Arena A scope:** OV / transcoder hierarchical attribution only. The QK step is out of scope (107 README). Freeze the
pattern and do not attribute to scores.

### 2.4 Exactly which nodes the graph contains

For one prompt (tokens 0..T-1), with all attention patterns frozen [v2 §2.1, §3; inferred where marked]:

- **Leaves, interpretable:**
  - the word-embedding SAE features at each position;
  - the positional embedding p_i at each position, as one leaf per position [v2 §2.1 Eq. 2].
- **Leaves, uninterpretable:** the error of every dictionary at every position. Detached, so it is excluded from
  circuits and from the recovery sum.
- **Leaves, constant (not discussed by the paper):** every bias.
  - Encoder biases b_E of active features, b_O, b_V (through the pattern), LN biases folded into downstream biases,
    and b_U for logit roots.
  - Theorem 3.1 as stated has no bias term. Each active node's value includes its b_E. So the leaf sum equals a_t
    only if biases count as leaves (A5) [inferred].
- **Internal nodes:** every **active** feature (a > 0) of:
  - every attention-output SAE, at every position and layer;
  - every transcoder, at every position and layer.
- **Root t:**
  - local: one SAE / transcoder feature at one position on one prompt;
  - end-to-end: one logit at one position.
  - Code: `Node(hook_point, "0.{pos}.{feat}")`, i.e. batch 0, one position, one feature [code graph.py:17-32;
    server/app.py `/model/trace`].
- **Edges:** only from a layer-earlier writer to a later reader: the residual is causal in layer, and attention moves
  information from position j ≤ i to i. There are no edges within a sublayer [inferred].
- **Reverse topological order:** sublayers from last to first: mlp_L, attn_L, …, mlp_0, attn_0, embedding. All
  positions of a sublayer form one tensor, gated together [inferred].

---

## 3. The algorithm, exactly

### 3.1 Linearisation (forward pass)

Run the real GPT-2 forward once, with these modifications:

```
for each layer L:
    # attention block
    x_ln1  = (x - mean(x)) * STOPGRAD(1/σ(x)) * w1 + b1        # LN scale frozen [A3]
    A      = STOPGRAD(softmax(scores(x_ln1)))                   # pattern frozen
    attn   = Σ_h A^h (x_ln1 W_V^h + b_V^h) W_O^h + b_O
    f_A    = ReLU(W_E^{A,L} attn_in + b_E)                      # attention-output SAE features (nodes)
    attn'  = W_D^{A,L} f_A + STOPGRAD(attn - W_D^{A,L} f_A)     # error detached; forward value == attn
    x      = x + attn'
    # MLP block
    x_ln2  = (x - mean(x)) * STOPGRAD(1/σ(x)) * w2 + b2
    f_M    = ReLU(W_E^{M,L} x_ln2 + b_E)                        # transcoder features (nodes)
    mlp'   = W_D^{M,L} f_M + STOPGRAD(MLP(x_ln2) - W_D^{M,L} f_M)
    x      = x + mlp'
root = f[L*, pos*, i*]                    (local)   or   logits[pos*, tok*] with ln_final scale frozen (end-to-end)
```

- The forward values equal the clean model exactly.
- The gradient from the root reaches only dictionary features, embeddings and bias leaves.
- Along every path, ∂/∂ is the product of frozen-pattern, frozen-LN-scale linear maps and 0/1 ReLU masks. So the graph
  is linear [v2 §2–3; code context.py:21, 34; transformer.py:70; LN freezing per A3].

### 3.2 Attribution

attr(v, t) = a_v · g_v, where g_v = ∂a_t/∂a_v is evaluated in the linearised graph [v2 Def. 3.2].
- It is **signed**.
- For the dictionaries above (Eq. 1–3), a_v is the post-ReLU feature activation.
  - The code's root is the `hook_feature_acts` value, which is post-ReLU and times the finetuned scaler [code sae.py
    encode; server/app.py].
- Inside Alg. 1, g_v is the gradient through **unpruned successors only**.

### 3.3 Algorithm 1: hierarchical attribution (paper form) [v1, v2 App. C]

```
input: τ > 0, linear graph G, root t
N' ← ∅
for v in reverse_topological_order(G):
    if v == t:
        v.grad ← 1                     # root is never gated, never added to N'  (A8)
    else:
        v.grad ← Σ_{u ∈ succ(v)} (∂a_u/∂a_v) · u.grad      # ordinary backprop; pruned u have u.grad = 0
        if v.grad · a_v < τ:           # SIGNED comparison
            v.grad ← 0 ; attr_v ← 0    # detach v: nothing upstream is credited through it
        else:
            attr_v ← v.grad · a_v ; N' ← N' ∪ {v}
return G' = G[N']                      # induced subgraph
```

Points a re-implementation must keep:
- **Gating is in the pass.** A node's gradient is complete before it is gated, because all successors are earlier in
  reverse topological order. Gating zeroes the node's contribution to every predecessor.
- **Leaves are gated too.** Embedding features and positional leaves are ordinary nodes v ≠ t.
- **Signed gating.** For τ > 0, every node with negative attribution is pruned. Hierarchical circuits therefore hold
  only positive contributors [inferred consequence of Alg. 1].
  - The bracket case's negative closing-bracket contributions come from Direct Contribution analysis, not Alg. 1
    [v1 §4.1].
- **Nodes downstream of t, or off its cone,** get grad 0 and are pruned.
- **Single root.** There is one root: one position, one prompt.
  - No multi-position or multi-prompt aggregation exists in the paper or the code.
  - The code's `cache[target].backward()` requires a scalar target [code attributors.py:168].

### 3.4 Algorithm 1 as the library implements it (autograd form) [code 67bfb0b attributors.py:128-187]

```
with dictionaries spliced (recon + detached error), attn scores detached:
    for each candidate tensor F (hook_feature_acts of every dictionary [+ attn_scores in v0.1.0 wrappers]):
        forward hook: remember V_F ← F ; F.retain_grad()
        backward hook on F's HookPoint: grad_in ← where(V_F * grad_out > τ, grad_out, 0)
    run forward on ONE prompt; target = F_t[0, pos, i] (or logits[0, pos, tok]); target.backward()
    for each candidate F: A_F ← F.grad * F            # F.grad is the UNMASKED incoming gradient [inferred]
    nodes ← {(F, idx) : A_F[idx] > τ}                  # same set the mask kept
    graph ← target + nodes (+ edge node→target, attribution=A)   # a star, not a wired graph
```

This is equivalent to §3.3, apart from `>` vs `≥` [inferred]. Recommended implementation for arena A: use this
autograd form on the §3.1 forward. It is one backward per (prompt, root, τ).

### 3.5 Reading out a circuit

- **Nodes.** N' = all nodes with attr ≥ τ after the pass. Include leaves (embedding features, positional) in N'; they
  are gated like any node.
- **Edges.** Neither the paper nor the code materialises edges for G'.
  - G' is defined as the induced subgraph G[N'] [Alg. 1].
  - The code returns a star [code attributors.py:185].
  - For a wired readout, the faithful edge weight for u → v with both in N' is the direct contribution:
    attr_{u→v} = a_u · (∂a_v/∂a_u)_direct · v.grad. This is the "Direct Contribution" of App. B, scaled by v's
    in-pass gradient.
  - Summing it over v's successors reproduces attr_u [inferred].
  - Edge weights are only needed for figures. Fig. 2b and Theorem 3.1 need nodes only.
- **Recovery.** Sum attr over the leaves of G' (§6.1).

### 3.6 Standard (post-hoc) attribution: the Fig. 2b comparator [v2 §3, Fig. 2a caption]

```
one backward with NO gating → attr_v = a_v · ∂a_t/∂a_v for all v
S ← {v : attr_v ≥ τ}             # pruned after the pass
G'' = G[S]
```

Same τ grid, same signed rule. Recovery is then measured in G'' (§6.2, A11).

### 3.7 Global / end-to-end circuits

Same as §3.3, with t = logit of the predicted token at the last position [v2 §5 intro: hierarchical attribution finds
the end-to-end flow].
- The final LN scale is frozen.
- Logit vs logit difference is unstated (A10).
- There is no separate global procedure [inferred].

---

## 4. Hyperparameters, paper vs code

| knob | paper | code | spec |
|---|---|---|---|
| τ | input of Alg. 1, τ > 0. **No value given for any case study** | default 0.1, absolute, in root units [attributors.py:149; app.py] | sweep (Fig. 2b protocol) |
| τ grid | "30 thresholds" (Fig. 2b); range and spacing not given | none | 30 values, log-spaced in τ / a_t (A7) |
| comparison | signed `< τ` → detach | signed `> τ` → keep | paper (`≥` keeps) |
| root | feature (local) or logit (end-to-end); one position, one prompt | scalar target, batch 0 | as paper |
| QK selection | "important" attention; no rule | `tracing_top_k` optional in the server's direct trace | out of scope |
| dictionary size | 24,576 everywhere | n/a | arena A: 24,576, except Kissane L5 / L7 at 49,152 (§9) |
| errors | detached | detached | detached |
| prompts, Fig. 2b | 20 IOI samples | n/a | 20 (A12) |

---

## 5. Data and case studies

| study | section (v2) | method actually used | root | prompts | published numbers |
|---|---|---|---|---|---|
| In-bracket features | §4.1, Fig. 3 | **Direct Contribution** (one-hop, manual), not Alg. 1 [v1 §4.1 states this explicitly] | L1A In-Square-Bracket feature (also an In-Round-Bracket one) at tokens inside brackets | one template "0 0 [1 1 1 [2] 3] 4" | L0M " [" feature contributes 104.1% (token "1"), 102.6% ("2"), 314.2% (" 3") of the feature's activation; 83.8% of it on average through head **L1.H1**. "]" features suppress, also via L1.H1. QK: L1.H1 attends to brackets, mainly via its query bias |
| Induction | §4.2, Fig. 4 | OV contributions + QK feature-pair attribution; method not named; consistent with direct contribution [inferred] | a "Capital Induction" L5A-level feature at the 2nd " Web", boosting "M" | one prompt (a WebM / HTML5 video sentence) | L0M.88 ("M") gives 35.0%; the top 7 auxiliary features (L0A / L1M / L3M) give 33.0%; flow mostly via **L5.H1** (induction head). QK top pairs: " Web" feature (L0M.1270, L1M.23399) × " Web"-preceding feature (L2A.14876, L2A.17608). Preceding features come via **L2.H2** (previous-token head), from L0A positional features |
| IOI | §5, Fig. 5, App. E Fig. 7 | **Hierarchical Attribution** (end-to-end) + QK of key heads | logit of " Mary" at the last position [inferred, A10] | 2 prompts: s_Mary "When Mary and John went to the store, John gave the bag to"; s_John swaps the first two names. Both predict " Mary" | head agreement: name movers L9.H6, L9.H9; S-inhibition L7.H3, L7.H6, L8.H10; induction L5.H5, L6.H9; previous-token L2.H2, L3.H2, L4.H1. Same heads, **disjoint features** across the two prompts |
| Hierarchical vs standard | §3, Fig. 2b | Alg. 1 vs post-hoc | logit [inferred] | **20 IOI samples**, source unstated | curves only: hierarchical higher at every sparsity |

All analysis is per prompt. Nothing is aggregated across prompts except the averaging implied in Fig. 2b [v2 §7: the
results are tied to specific inputs].

---

## 6. Their evaluation, exactly

### 6.1 Leaf-sum recovery (Theorem 3.1)

- **Exact statement.** In G' (any node-detached subgraph), a_t(G') = Σ_{deg_in=0 in G'} attr(v, t) [v2 §3, App. D].
- **Measure.** "Logit recovery", the faithfulness of the identified subgraph [v2 Fig. 2 caption].
  - The normalisation is not stated. The most faithful reading (A11):

    recovery(G') = Σ_{v ∈ interpretable leaves ∩ N'} attr_{G'}(v, t)  /  a_t(clean)

  - Here attr_{G'} is the in-pass attribution, which flows only through N'.
  - Errors are excluded, because they were detached first.
  - Bias leaves are excluded from the numerator [inferred, A5]. Report the bias share separately so the reader can
    add it back.
- **No intervention.** Nothing is ablated or patched in the real model.

### 6.2 Hierarchical vs post-hoc (Fig. 2b)

- **Axes.** x = "number of remaining nodes" (sparsity); y = logit recovery [v2 Fig. 2 caption].
- **Setup.** 20 IOI samples × 30 thresholds, both methods. The result is a curve per method.
- **Unstated.**
  - The τ range.
  - Whether points are averaged per τ across prompts or plotted per prompt.
  - What counts as a node: are leaves counted?
  - Logit vs logit difference.
- **Claim.** Hierarchical gives higher recovery at every sparsity.
  - Mechanism: post-hoc keeps nodes whose attribution flowed through nodes that were later pruned. Those nodes cost
    size but add no recovery inside G''.

### 6.3 Interventional checks

**None.** The paper argues against ablation-based evaluation (hydra effects) [v2 §3 Evaluation; §7].

### 6.4 Numbers we can reproduce

- No numeric Fig. 2b values are given. The reproducible claim is the qualitative ordering.
- Case-study percentages are specific to their dictionaries (feature IDs, L0). They do not transfer as numbers.
- What transfers is head-level structure (§8 F5–F7).

---

## 7. Ambiguities and the most faithful reading

| # | ambiguity | most faithful reading | why |
|---|---|---|---|
| A1 | Attention SAE site: before or after W_O | after W_O (`hook_attn_out`) | "attention output"; App. A lists 12 attention-output SAEs, no per-head ones. Arena A cannot match it anyway (§9) |
| A2 | Transcoder input: resid_mid or ln_2 output | ln_2 output (the true MLP input) | Eq. 3 "pre-MLP activation", and v1 §2.2 says Skip SAEs take the same input as the MLP. Their √768 renormalisation makes the two nearly equivalent up to the LN affine |
| A3 | LayerNorm scale | **frozen** (STOPGRAD on 1/σ) | the paper's claim of a strictly linear graph and its App. B rule (hold W(x) constant) both require it; Theorem 3.1 is false otherwise. The code does not freeze it, which is a code deviation from the paper's own premise |
| A4 | Input-norm factor of the dictionaries (√768 / ‖x‖) | treat it as frozen per token (or constant) | same reason as A3. Moot for arena A: Dunefsky / Kissane dictionaries have no runtime norm factor |
| A5 | Biases in Theorem 3.1 | constant leaves; they count toward exactness, not toward recovery | each active node's value includes b_E. The theorem omits biases; App. B says "the bias contributes itself". Their SAEs drop the decoder bias, which shrinks but does not remove this |
| A6 | Word-embedding SAE | a dictionary on the token embedding only, with p_i a separate leaf | Eq. 2 keeps p separate. L0RPr = token + position, so it would conflict with that |
| A7 | τ grid (30 values) | log-spaced τ / a_t, covering from "all positive nodes kept" to "≤ 1 node kept", same grid for both methods | Fig. 2b sweeps sparsity over its full range. Dividing by a_t is a per-prompt monotone reparametrisation, so no curve changes |
| A8 | Root gating | root never gated; not counted in N' | Alg. 1 has the root branch separate |
| A9 | "Important attention" selection for QK | out of scope | no rule given |
| A10 | IOI root: logit vs logit difference | logit of the IO token at the last position | Alg. 1 has a single node root; §2.3 / §3 speak of "logits". Report logit diff as a labelled sensitivity (ours) |
| A11 | Recovery of the post-hoc set S | re-run the backward with fixed mask S (gradient only through S), then sum interpretable-leaf attributions | Theorem 3.1 is stated in G[S], and the paper presents recovery as computed after pruning. Using the ungated attributions of the leaves in S would not be a property of G[S] |
| A12 | The 20 IOI samples | 20 prompts from Wang et al. IOI templates, GPT-2 single-token names, kept only if GPT-2's top-1 is the IO name; balanced ABBA / BABA | the paper's two named prompts are ABBA-type (s_Mary) and BABA-type (s_John). The source of the 20 is unstated, so this is ours and is labelled |
| A13 | Which nodes count for "number of remaining nodes" | feature nodes in N' (attention SAE + transcoder + embedding-SAE features), excluding root, positional leaves, errors and biases. Also report with positional leaves included | nodes are "activated features" [v2 §3 Formulation] |
| A14 | BOS token | follow the dictionaries' training convention (Dunefsky / Kissane: GPT-2 with BOS). Never root at position 0 | the paper is silent; the code server uses `prepend_bos=False` |
| A15 | Orphan nodes in G' (kept but every predecessor pruned) | they count toward size but contribute nothing to the interpretable-leaf recovery | literal "deg_in = 0" would make them leaves. But their value is not explained by interpretable inputs, and counting them would make recovery trivially 1 whenever the root's parents are all kept |

---

## 8. Fidelity checks (all must pass before any comparison row)

### F1: Exactness (Theorem 3.1 at full graph)

Setup: gating off (τ = −∞, not τ = 0, because signed τ = 0 prunes negative nodes).
- Pass: Σ attr over all leaves = a_t, to fp32 tolerance (|Δ| / |a_t| ≤ 1e-4) on every test prompt and root.
  - "All leaves" means token-embedding features, embedding-SAE error, positional, all dictionary errors, and all
    biases.
- A failure means the graph is not linear: an unfrozen LN, pattern or norm factor.

### F2: Exactness on a pruned graph (Theorem 3.1 for G')

Setup: any τ.
- Recompute the root by a forward pass of the *linearised* model that:
  - keeps only the N' nodes (pruned nodes' decoder contributions removed);
  - keeps ReLU gates at the clean pattern;
  - keeps patterns and LN scales frozen;
  - keeps errors and biases.
- Pass: the recomputed root equals the in-pass sum of the leaf attributions (including errors and biases) of G'.
- This checks that the gating is a clean subgraph detach.

### F3: In-pass vs post-hoc, toy graph (unit test)

**(a) The OpenMOSS unit test**, reproduced exactly [code 67bfb0b tests/intergration/test_attributor.py:35-55]:
- Model:
  - x = 1;
  - h1 = x·W1 with W1 = [1, 2]; x ← x + h1·W2 with W2 = [1, 1]ᵀ;
  - h2 = x·W3 with W3 = [1, 1]; out = x + h2·W4 with W4 = [2, 1]ᵀ;
  - root = out[0] = 16.
- Expected:
  - ungated: attr(h1) = [4, 8], attr(h2) = [8, 4];
  - τ = 5, hierarchical: h2[1] (4) is gated, then g(h1) = 3, so attr(h1) = [3, 6]. Kept: {h1[1]: 6, h2[0]: 8}, three
    nodes including the target.
  - Post-hoc at τ = 5 keeps the same set, with h1[1] = 8.
  - So this test checks values, not sets.
- Direct attribution (other nodes detached) gives h1 = [1, 2] and h2 = [8, 4].

**(b) A set-separating toy**, ours:
- Model: x = 1; A = x; B_k = A for k = 1..4; D = 3x; t = Σ_k B_k + D = 7. Each B_k has attr 1, A has 4, D has 3.
- At τ = 2:
  - post-hoc keeps {A, D}, with A disconnected in G[S];
  - hierarchical keeps {D}.
- Both have recovery 3/7 when computed per A11.
- So hierarchical reaches the same recovery with fewer nodes. That is the Fig. 2b mechanism in miniature.
- Add a node E = −x into t to check signed gating: at τ = 0.5, E (attr −1) is pruned by both.

### F4: Fig. 2b qualitative

- Setup: 20 IOI prompts (A12), root per A10, 30-τ grid (A7), nodes per A13, recovery per A11.
- Pass:
  - mean hierarchical recovery ≥ mean post-hoc recovery at matched node counts, interpolated on each curve, across
    the whole range;
  - and on most individual prompts.
- If this fails, suspect the implementation before the substrate.
  - Also report it against the substrate gap (§9), because the result depends on the dictionaries.

### F5: Bracket head check (Direct Contribution, dictionary-agnostic)

- Find an L1 attention-SAE feature that fires inside square brackets and grows with nesting.
- Pass:
  - a layer-0 transcoder " [" feature contributes positively;
  - "]" contributes negatively;
  - the majority of the " [" contribution passes through head L1.H1 (paper: 83.8%).

### F6: Induction head check

- Find a capital-letter induction feature at about layer 5 on the WebM prompt.
- Pass:
  - most OV flow is through L5.H1;
  - the preceding-token information comes through L2.H2.

### F7: IOI head agreement

- On s_Mary and s_John, decompose the edges of the hierarchical circuits per head.
- Pass: the heads named in §5 (L9.H6 / H9, L7.H3 / H6, L8.H10, L5.H5, L6.H9, L2.H2, L3.H2, L4.H1) carry the bulk of
  attention-edge attribution.
- Also check that the two prompts' feature sets are largely disjoint.

F5–F7 are qualitative. Treat them as sanity checks, not gates.

---

## 9. Mapping onto arena A (public GPT-2 Small substrate)

**Substrate:**
- **Model:** GPT-2 Small in TransformerLens, `fold_ln=True` (the Dunefsky transcoders were trained on
  `ln2.hook_normalized` under that convention; see 035 README, the LayerNorm-convention section).
- **MLPs:** Dunefsky et al. transcoders, `pchlenski/gpt2-transcoders`.
  - ReLU, 24,576 features, L1 8e-5, 12 layers.
  - Input `blocks.L.ln2.hook_normalized`, output `hook_mlp_out`.
  - They have a pre-encoder centring b_dec and an output bias.
  - Already converted and validated in 035 (FVU 0.003–0.27, LN fold checked).
  - Use them in the `fold_ln=True` / hook_normalized convention, or the 035 circuit-tracer conversion. Either way,
    freeze the ln_2 scale.
- **Attention:** Kissane et al. SAEs. HF repo id **verified: `ckkissane/attn-saes-gpt2-small-all-layers`** @
  `97e0da0e9182695e9813d15ea32184da83f5c874`.
  - Site `blocks.L.attn.hook_z`, heads concatenated (act_size 768).
  - dict_size 24,576 at all layers **except L5 and L7, which are 49,152** (dict_mult 64).
  - L1 coefficients vary per layer, 0.8–3.0.
  - 2B tokens, seq_len 128, b_dec initialised to zero (cfg JSON for L5; the other layers are from filenames).
  - A ReLU encoder with b_dec centring is [inferred; verify from the loader].
- **Residual SAEs (Bloom, `jbloom/GPT2-Small-SAEs-Reformatted`):** **not needed as graph nodes** (§2.1 item 3).
  - Answer to 107 open decision 2: no residual SAEs in either node universe.
  - The only candidate use is the embedding leaf (gap G4).

**The arena-A graph:**
- leaves: token-embedding node(s), positional leaf p_i, all errors, all biases;
- internal: Kissane z-SAE features and Dunefsky transcoder features at every position;
- frozen: patterns, LN scales;
- errors: detached;
- root and τ: per §3–4.

**Gaps vs Ge's substrate, stated plainly.** Each is an unavoidable gap or our adaptation, not a change to Alg. 1.

| # | gap | consequence |
|---|---|---|
| G1 | Dictionaries are not theirs (no public weights). Different training (OpenWebText vs ?; 2B tokens / seq 128 for Kissane), no pruning / finetune scaler, decoder biases present (they removed theirs) | Feature IDs and percentages do not transfer. More constant bias mass in the leaf sum (A5) |
| G2 | Attention SAEs sit on `hook_z` (before W_O, per-head blocks of the concatenation), not the attention output | Linear-equivalent placement: the feature writes W_D,i W_O to the residual, and the error passes through W_O. Per-head OV is exact per head block. Different dictionary geometry and errors |
| G3 | Kissane L5 / L7 are 49,152 wide, not 24,576 | Larger node universe at two layers |
| G4 | No public GPT-2 **word-embedding SAE** | **Our adaptation:** one leaf per position for the token embedding vector W_E[token_i] (one token, so trivially interpretable) and one positional leaf p_i. Their recovery would additionally lose the embedding-SAE error; ours does not. Alternative (also ours): Bloom L0 resid_pre SAE as the embedding dictionary, but it mixes in p_i (A6) |
| G5 | Dunefsky transcoders read the ln_2 output; theirs read "pre-MLP" (A2) | Same graph shape. ln_2 scale frozen either way |
| G6 | Dictionary quality differs (Kissane / Dunefsky L0 and FVU vs their Tables 1–3) | Report per-layer FVU / L0 of the arena dictionaries next to their tables. Recovery is bounded by error mass |
| G7 | QK step not run | Arena A compares OV / transcoder circuits only, which is what Fig. 2b compares too [inferred: Fig. 2b's graph is the Alg. 1 graph] |
| G8 | Fig. 2b protocol details (A7, A10–A13) are our readings | Each is labelled in the row's caption |

**Things that must NOT be done** (they made the 034 / 083 / 098 row ours):
- `|attr|` gating;
- site-level gating;
- gradients through real MLPs, LN or patterns;
- encoder pre-activation roots;
- summing attributions over contexts or positions;
- τ bisected to a target size.

Root rule: one (prompt, position) per circuit. The 107 plan's K circuits = K prompts, each at the target's position
on that prompt; the max-activating position is our choice, labelled.

---

## Sources (all read on 2026-11-01 via WebFetch / WebSearch; nothing downloaded)

- Paper: https://arxiv.org/abs/2405.13868, https://arxiv.org/html/2405.13868v1, https://arxiv.org/html/2405.13868v2
- Code:
  - https://github.com/OpenMOSS/Language-Model-SAEs (README news 2024.5.22);
  - commits 3ff7275 / f9cc2d6 / 67bfb0b (v0.1.0) / e201058 (v1.1.0);
  - `dev` `src/llamascopium/circuits/attribution.py`
- HF: https://huggingface.co/OpenMOSS-Team (no GPT-2 dictionaries);
  https://huggingface.co/ckkissane/attn-saes-gpt2-small-all-layers (cfg L5)
- Unread: https://openreview.net/forum?id=b8sq8Y5VFo (browser challenge)
- Local:
  - `experiments/107-true-baselines/README.md`;
  - `experiments/034-ge-replication/README.md`;
  - `experiments/083-v1-comparison/ge_hier.py`;
  - `experiments/098-circuitlasso-baseline/README.md`;
  - `experiments/035-transcoder-compare/README.md`;
  - `paper/resources/related-reading/A-closest.md` §4
