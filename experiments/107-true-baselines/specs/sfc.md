# Spec: Sparse Feature Circuits (Marks et al.), as published and as coded

Written 2026-11-01 for 107 arena C. This file describes what the paper and the official code do. It does not describe
what we think they should do. Anything we would change is labelled **our adaptation**.

## 0. Sources read, and how to read the tags

- **Paper:** Marks, Rager, Michaud, Belinkov, Bau & Mueller, "Sparse Feature Circuits: Discovering and Editing
  Interpretable Causal Graphs in Language Models", ICLR 2025. arXiv 2403.19647, **v3 (2025-03-27)**, HTML, read in full:
  main text, footnotes and Appendices A to H. I also read v2 (2024-03-31) for the passages that changed; they are
  marked [paper v2 §x].
- **Code:** github.com/saprmarks/feature-circuits at **`7fbd82b895ae16294f4e6fc7bfc675d1d680d659`** (main HEAD,
  2025-07-14, "add handling of device and dtype to IdentityDict"). This is the same SHA 081 used for the data. Files
  read line by line:
  - `circuit.py`, `attribution.py`, `ablation.py`, `activation_utils.py`, `loading_utils.py`, `data_loading_utils.py`
    and `dictionary_loading_utils.py`;
  - `scripts/get_circuit.sh`, `scripts/evaluate_circuit.sh` and `scripts/get_circuit_nopair.sh`;
  - `README.md` and `requirements.txt`;
  - every code cell of `experiments/faithfulness.ipynb` and `experiments/bib_shift.ipynb`;
  - the threshold logic of `circuit_plotting.py`.
- **Paper-era code:** `64d1cb368926b96767f4afe14884f3517d43788d` (2024-10-02, "finalizing gemma stuff"). This is the
  last commit to touch `experiments/faithfulness.ipynb`, so it is the closest commit to the v3 figures. I read its
  `attribution.py`, `circuit.py` and `loading_utils.py` for differences from HEAD (§9).
- **Branches, issues and PRs (follow-up):** all 14 branches listed; compare views for `jvp-fix`, `iclr-updates`,
  `eric`, `cluster_rerun`, `sam`, `triangles` and `nnsight_upgrade`. Full diffs read: `bce4ecd` (jvp-fix),
  `bce4ecd...8eaa619` (iclr-updates), `2b3ffae` (the memory fix), `3b940e1` (the nnsight pin), `7fbd82b` (HEAD), and
  `main...cluster_rerun`. Issues #10, #13, #14, #15, #16 and #17 and PR #12 read in full, with replies. See §12.
- **Dictionary library:** `saprmarks/dictionary_learning` at `61ac634845bd76c839482f3b725ab3d898c8b277` (2025-07-14).
  This is the companion commit to the HEAD above, with the same commit message. I read `dictionary_learning/dictionary.py`.
- **Weights:** HF `saprmarks/pythia-70m-deduped-saes`. Repo sha `50a434461d36ed78d1b0b901944e6edc829f1dce`,
  lastModified 2024-07-29. It holds one file, `dictionaries_pythia-70m-deduped_10.zip`, of 2,369,891,306 B.
- **Local:** the SFC data files are in `data/external/sfc_feature_circuits/data/` (same SHA, see its `SOURCE.md`). The
  unzipped dictionaries are already in `experiments/030-cross-sae/dictionaries/pythia-70m-deduped/` (19 site folders ×
  `10_32768/{ae.pt, config.json}`, 2.55 GB). I checked one `config.json` against App. B.1.2 (§2.2).

Nothing was cloned or downloaded. Code was read as raw text.

**Tags.** [paper §x / Eq. y / App. z] refers to v3 unless marked. [code file:line] is at `7fbd82b` unless another SHA is
given. [inferred] means my reading, not stated anywhere. Line numbers count from 1 in the raw file.

---

## 1. Summary

- **Nodes:** SFC builds a causal graph whose nodes are SAE feature activations and SAE error terms at specific token
  positions [paper §3.1].
- **Node scores:** each node gets an approximate indirect effect (IE) on a scalar metric m. This is computed by
  integrated gradients (IG) in the code, with attribution patching available [paper Eq. 3–4; code circuit.py:40-49].
  - The IE is averaged over examples.
  - With templatic data, the same feature at different positions counts as different nodes. With non-templatic data,
    effects are summed over positions before averaging [paper §3.1, App. A.2].
- **Node selection:** nodes with |mean IE| > T_N are kept [paper §3.1].
- **Edges:** each edge u→d is scored by a linear approximation to "the effect on m via d of patching u, holding the
  intermediate nodes fixed" [paper App. A.1 Eq. 5]. Edges with |IE| > T_E are kept.
- **What "the circuit" is:**
  - **For evaluation:** a set of nodes. Faithfulness and completeness ablate nodes and never look at edges
    [code ablation.py:9-65; faithfulness.ipynb cell 4].
  - **For display:** nodes plus edges. The edge threshold is applied only when plotting
    [code circuit_plotting.py:243-636].

---

## 2. Substrate (arena C: Pythia-70m-deduped)

### 2.1 Model

- **Model:** `EleutherAI/pythia-70m-deduped`, 6 layers, d_model 512, loaded through `nnsight.LanguageModel` in float32
  [code circuit.py:449-467].
- **Pythia flags:** `parallel_attn = True`, so no attn→mlp edges are computed (Pythia uses a parallel residual). The
  embedding is included as a node site (`include_embed = True`) [code circuit.py:453-460].
- **No BOS:** none is prepended. The Pythia tokenizer adds none [inferred]. The code strips a BOS from answer
  tokenisations only "if necessary" [code data_loading_utils.py:32-36].

### 2.2 Dictionaries (the "10_32768" suite)

**Sites:** 19 SAEs [code dictionary_loading_utils.py:35-86]:

| node name | nnsight submodule | output used | SAE path under `dictionaries/pythia-70m-deduped/` |
|---|---|---|---|
| `embed` | `gpt_neox.embed_in` | output | `embed/10_32768/ae.pt` |
| `attn_i` | `gpt_neox.layers[i].attention` | `output[0]` (`is_tuple`) | `attn_out_layer{i}/10_32768/ae.pt` |
| `mlp_i` | `gpt_neox.layers[i].mlp` | output | `mlp_out_layer{i}/10_32768/ae.pt` |
| `resid_i` | `gpt_neox.layers[i]` | `output[0]` (`is_tuple`) | `resid_out_layer{i}/10_32768/ae.pt` |

- **Where the dictionaries must sit:** `DICT_DIR` is the folder holding the code plus `/dictionaries`
  [code dictionary_loading_utils.py:11]. They must therefore be at `<repo>/dictionaries/pythia-70m-deduped/...`, which
  is what the README's unzip produces [code README.md:30-35].
- **Size:** d_SAE = 32,768 = 64 × 512 [paper §2; code circuit.py:197].
- **Architecture:** ReLU SAE with a tied pre-encoder bias. The encoder is f = ReLU(W_E(x − b_D) + b_E), the decoder is
  x̂ = W_D f + b_D, and the columns of W_D have unit norm [paper App. B.1.1]. In code:
  - `encode = ReLU(encoder(x − bias))` and `decode = decoder(f) + bias`;
  - `from_pretrained(..., normalize_decoder=True)` rescales the decoder to unit columns if needed, and compensates in the
    encoder [code dictionary_learning/dictionary.py:64-68, 103-140 @61ac634].
- **Training** [paper App. B.1.2]:
  - Pile, first 128 tokens of random documents;
  - loss ‖x̂ − x‖² + λ‖f‖₁ with λ = 0.1, lr 1e-4, batch 2^14;
  - 120k steps (about 2B tokens), resampling every 25k steps, 1k-step warmup.
  - The local `resid_out_layer3/10_32768/config.json` matches: `sparsity_penalty 0.1, lr 1e-4, steps 120000,
    resample_steps 25000, warmup_steps 1000, n_ctxs 10000, ctx_len 128, out_batch_size 16384`.
- **Quality, as published** [paper App. B.1.3, Tables 3-6; 128 contexts × 128 tokens]:
  - embed: 96% variance explained, L0 3, 98% CE recovered;
  - attention: VE 83-92%, L0 127-215;
  - MLP: VE 85-99%, L0 5-160;
  - resid: VE 85-96%, L0 54-108, CE recovered 92-97%.
  - "Resid i" is the output of layer i, so resid_5 is the final residual before the unembedding.

### 2.3 Error nodes

- **Definition:** ε(x) = x − x̂ [paper Eq. 1; code attribution.py:31, 92].
- **One node per site and position:** each error is a single node per (site, position). Its IE is the
  d_model-contracted product Σ_k ∇_ε m_k · Δε_k. In `SparseAct.__matmul__` the feature part is elementwise and the error
  part is summed to one scalar, `resc` [code activation_utils.py:43-46]. In the saved tensors the error node is index
  `n_feats` (the last of `f+1`) [code activation_utils.py:143-150; attribution.py:285, 337].
- **They count in circuit size.** The size counter is `n.act.sum() + n.resc.sum()` over the evaluated submodules
  [code faithfulness.ipynb cell 4].
- **They are ablatable:** an error node that is not in the circuit is mean-ablated like a feature (`handle_errors =
  'default'`) [code ablation.py:47-60].
- **Variants in Fig. 3** [paper §3.2; code faithfulness.ipynb cells 4-5]:
  - errors handled like any node (`features`);
  - all errors ablated (`features_wo_errs`, `'remove'`);
  - only resid errors kept as possible members (`features_wo_some_errs`, `'resid_only'`);
  - neurons, through `IdentityDict`, where the error is identically 0.

### 2.4 How the dictionaries attach

- **Splice:** `Submodule.get_activation()` and `set_activation()` read and overwrite the module output in an nnsight
  trace. `stop_grad()` sets the output's gradient to zeros [code loading_utils.py:14-56].
- **Attribution patching:** every site is spliced at once, as x ← x̂(f) + ε [paper App. A.3 Eq. 6]. Two gradient hooks
  are set:
  - `residual.grad = 0` stops the gradient on the error, so f gets a non-zero gradient;
  - `x.grad = x_recon.grad` is the pass-through ∇_{x_u} m ← ∇_{x_d} m [code attribution.py:29-37; paper App. A.3].
- **IG:** only one site at a time is spliced, as `decode(f.act) + f.res`. Every other site runs the unmodified model
  [code attribution.py:119-146].
- **Edges (JVP):** the upstream and downstream sites are both spliced through `hacked_forward` (the standard SAE
  forward for ReLU AEs). The upstream error has its gradient zeroed, and the listed intermediate sites have
  `stop_grad()` [code attribution.py:256-317].
- **Ablation:** each evaluated site is re-encoded in the clean run, its features and error are overwritten where
  needed, and `decode(f) + res` is written back [code ablation.py:36-62].

---

## 3. The algorithm (pseudocode written from the code; the formulas match the paper unless §9 says otherwise)

### 3.1 Data loading (paired SVA)

```
load_examples(path, n, model, use_min_length_only=True, seed=12):        # data_loading_utils.py:8-67
  lines <- read JSONL; random.Random(12).shuffle(lines)
  examples <- []; Lmin <- inf
  for each record (clean_prefix, patch_prefix, clean_answer, patch_answer):
     skip unless both answers tokenise to exactly 1 token
     skip unless len(tok(clean_prefix)) == len(tok(patch_prefix))
     skip unless tok(prefix)+tok(answer) == tok(prefix+answer)   (both sides)
     if len < Lmin: examples <- []; Lmin <- len        # restart on a new shortest length
     elif len > Lmin: skip
     append; if |examples| >= n: return examples
  (falls off the end -> returns None if fewer than n qualify)
```

- **Single length:** every example has the same token length, so there is no padding, the metric reads position −1,
  and per-position nodes line up across examples [code data_loading_utils.py:55-62; circuit.py:569] [inferred].
- **The subset is the shortest length only:** it is the first n records in shuffled order among the records of minimum
  token length. Records with longer tokenisations (for example multi-token nouns) are never used.
- **Ignored seed flag:** circuit.py's `--seed` (default 12) is parsed but never used. The data seed is
  `load_examples`'s own default of 12 [code circuit.py:443; data_loading_utils.py:14].

### 3.2 Metric

- **Discovery, paired** [code circuit.py:561-574]:
  `m(x) = logit[−1, patch_answer] − logit[−1, clean_answer]`, one value per example.
  - Paper: m = log P(patch verb) − log P(clean verb) [paper §2, §3.1]. The logit difference equals the log-probability
    difference exactly, because the softmax normaliser cancels [inferred].
  - Table 1 writes it as p(·) − p(·) [paper Table 1]; see §9.
- **Discovery, unpaired** (`--nopair`, clusters): `m = −logit[−1, answer]` [code circuit.py:551-559].
  - Paper: m = −log P(y|x) [paper §5]. These differ by the log-normaliser (§9).
- **Evaluation** [code ablation.py:176-180; faithfulness.ipynb cell 4]:
  `m_eval = logit[−1, clean_answer] − logit[−1, patch_answer]`. This is the opposite sign to discovery. Faithfulness is
  a ratio, so the sign cancels.

### 3.3 Node effects (`patching_effect`, method fixed to `"ig"` in `get_circuit`)

```
submods = [embed] + [attn_0, mlp_0, resid_0, ..., attn_5, mlp_5, resid_5]                 # circuit.py:35-37
# clean states (no grad): for s in submods: f_s = SAE_s.encode(x_s); eps_s = x_s - decode(f_s)   # attribution.py:85-95
# patch states: same on x_patch (no grad);  total_effect = m(x_patch) - m(x_clean)        # :103-114
# no patch (nopair / zero-ablation): patch state := zeros for f AND eps; total_effect = None  # :97-101
for s in submods:                               # one site at a time, all examples in the batch at once
   for k in 0..N-1  (N = steps = 10):           # :126-134
       alpha = k / N                            # alpha in {0, .1, ..., .9}
       z_k = (1 - alpha) * clean_s + alpha * patch_s       # both f and eps interpolated jointly
       run clean input with x_s := decode(z_k.f) + z_k.eps; record m
   backward of sum_k sum_batch m                # :135-136
   g_s = (1/N) sum_k grad wrt z_k               # :138-140 (features and the full d_model error vector)
   delta_s = patch_s - clean_s                  # :141
   IE_s[b, pos, i]   = g_s.f[b,pos,i] * delta_s.f[b,pos,i]                 # features
   IE_s[b, pos, err] = sum_d g_s.eps[b,pos,d] * delta_s.eps[b,pos,d]       # error node, contracted   :142
```

- **The IG path is per site, not per node.** All features and the error of a site move together from clean to patch,
  and the gradient is read along that joint path. Downstream sites are the real model [code attribution.py:119-146]
  [inferred]. This matches the paper's remark that IG "cannot be done in parallel" for a node and its descendant
  [paper §2].
- **Baseline:** the counterfactual (patch) activation for paired data. For unpaired data it is zero, for features and
  error alike [paper §2 last paragraph; code attribution.py:97-101].
- **AtP variant** (`method='attrib'`, used by default when `patching_effect` is called directly, but not by
  `get_circuit`): splices every site at once, then IE = ∇m|clean · Δ [paper Eq. 3; code attribution.py:15-72, 231].
- **Exact variant** (`method='exact'`): one forward per (position, feature) that differs, and per error position
  [code attribution.py:151-222]. App. H uses it for the comparison on 30 across-RC inputs [paper App. H, Fig. 25].
- **Positions where clean and patch are identical** (the token before the subject; nothing earlier can differ in a
  causal model) have Δ = 0 and therefore IE = 0 [inferred].

### 3.4 Node aggregation and threshold

```
per batch:   nodes[s] = IE_s                                   # (b, pos, F+1)
aggregation == "none" (templatic, the SVA setting): nodes[s] = mean_b IE_s  -> (pos, F+1)      # circuit.py:183-184
aggregation == "sum":  nodes[s] = mean_b ( sum_pos IE_s )   -> (F+1)                           # circuit.py:148-166
across batches: running sum of len(batch) * batch-mean, divided by num_examples  (= mean over all examples)  # :592-617
circuit node set(T_N) = { (s, pos, i) : |nodes[s][pos, i]| > T_N }                             # ablation.py:138-140; notebook cell 4
```

- **The saved file holds every node effect, unthresholded:** `nodes` is the whole tensor [code circuit.py:629-631].
  T_N is applied afterwards, by the evaluator or the plotter. The discovery-time `--node_threshold` only affects which
  edges are computed (§3.5).
- **Sign:** selection is on |IE| [paper §3.1; code ablation.py:139]. One exception: the BiB notebook selects
  signed `> 6` (Gemma) [code bib_shift.ipynb cell 9]; see §9.

### 3.5 Edges (`jvp`), Pythia ordering

```
features_by_submod[s] = |IE_s[b, pos, i]| > T_N   # PER EXAMPLE, PER POSITION, before aggregation   circuit.py:51-53
edges[resid_5]["y"] = IE_{resid_5}                # the root edge is the resid_5 node effect           :73-76
for layer L = 5..0:                                                                                    :93-124
   MR = N(mlp_L  -> resid_L)
   AR = N(attn_L -> resid_L, stop=[mlp_L])
   (AM = N(attn_L -> mlp_L) only if not parallel_attn; skipped for Pythia)
   prev = resid_{L-1}  (embed when L = 0)
   RM = N(prev -> mlp_L,   stop=[attn_L])
   RA = N(prev -> attn_L)
   RR = N(prev -> resid_L, stop=[mlp_L, attn_L])

N(u -> d, stop):                                   # attribution.py:245-338
   on the CLEAN input only, splice SAEs at u and d; zero the gradient into u's error (x_res.grad = 0)
   left  = g_d  (the IG mean gradient of m at d, from §3.3)      right = delta_u  (patch - clean at u)
   for each (b, pos, j) with features_by_submod[d] True:
       stop_grad on every site in `stop`
       scalar = left[b,pos,j] * act_d[b,pos,j]   (for the error index: sum_d left.eps * eps_d)
       backward(scalar);  w[(b,pos,j),(b',pos',i)] = grad_u[b',pos',i] * right[b',pos',i]  (error: contracted)
   -> sparse tensor of shape (b, pos, F+1, b, pos, F+1), downstream first
```

**Formula:** IE(u→d) ≈ ∇_d m · (∂d/∂u with the intermediates M held constant) · (u_patch − u_clean)
[paper App. A.1 Eq. 5]. The code evaluates:
- the first factor with the **IG-averaged** gradient (see §9);
- the Jacobian at the clean input;
- the JVP trick of A.3: backpropagate the scalar ⟨∇_d m, d⟩ with ∇_d m held as a constant [code attribution.py:305, 314].

**Edge aggregation** [code circuit.py:138-185]:
- `"none"`: w summed over both batch axes and divided by b, giving shape (pos_d, F+1, pos_u, F+1). Cross-position
  edges exist (attention).
- `"sum"`: w summed over both position axes, then over both batch axes, divided by b, giving (F+1, F+1).

**Edge threshold.** It is never applied in discovery. The plotter draws an edge only if both endpoints pass
|node| > T_N (on the aggregated means) and |w| > T_E [code circuit_plotting.py:611-636]. Edges into "y" are compared
against T_E only [code circuit_plotting.py:366].

**The residual stream:** there are resid_{L−1}→resid_L skip edges (RR, with mlp_L and attn_L stopped), and the
resid_5→y root edge. Attention and MLP nodes connect upward only through resid_L, the only downstream site of attn_L and
mlp_L in Pythia. There are no direct attn/mlp → y edges [code circuit.py:93-124] [inferred from the edge list].

### 3.6 Driver (paired SVA)

```
examples = load_examples(data/<task>_train.json, 100, use_min_length_only=True)   # circuit.py:483-486
batches of batch_size; per batch: nodes, edges = get_circuit(...); running weighted sum   # :543-615
save {"examples", "nodes", "edges"} to circuits/<model>_<dataset>_n<N>_agg<agg>_node<T_N>.pt
   (or _nodeall.pt with --nodes_only, edges = None)                                # :502-507, 629-631
if a file for the same base with a LOWER T_N exists, it is loaded instead of recomputing   # :514-527
plot (always; there is no check on --plot_circuit)                                    # :647-674
```

---

## 4. Hyperparameters: paper vs code

| quantity | paper (v3) | code default | scripts / notebook | flag |
|---|---|---|---|---|
| node estimator | IG used "in our experiments" [§3.1] | `get_circuit` hard-codes `method="ig"` [circuit.py:48]; `patching_effect` default `'attrib'` [attribution.py:231] | — | consistent for discovery |
| IG steps N | 10 [Eq. 4] | 10 [attribution.py:82, 232] | — | — |
| IG α grid | gradient at α·a_clean + (1−α)·a_patch, α ∈ {0, 1/N, …, (N−1)/N} [Eq. 4]: includes the patch point, excludes clean | (1−α)·clean + α·patch, α = k/10, k = 0..9 [attribution.py:127-128]: includes clean, excludes patch | — | **opposite Riemann endpoint** (§9.1) |
| patch baseline | counterfactual; zero if unpaired [§2] | same [attribution.py:97-101] | — | — |
| edge first factor | ∇_d m at d_clean [Eq. 5] | IG-mean gradient `grads[downstream]` [circuit.py:86] | — | **differs** (§9.2) |
| aggregation (SVA) | per-position mean over examples [§3.1, App. A.2] | `--aggregation sum` [circuit.py:383-388] | README example `none` [README:50]; notebook needs `aggnone` [cell 4] | default ≠ SVA setting; pass `none` |
| discovery examples | "typically 100-example subsamples" [README:21] | `--num_examples 100` [circuit.py:350-355] | get_circuit.sh 100 [:11] | — |
| batch size | — | 32 [circuit.py:377-382] | get_circuit.sh 10 [:12]; nopair 6 | no effect on values [inferred] |
| T_N | RC 0.1, PP 0.1, simple 0.2, within-RC 0.2 [App. C.1 Figs 7, 9, 11, 13]; BiB 0.1 [App. C.2] | `--node_threshold 0.2` [circuit.py:389-394]; `get_circuit()` 0.1 [:33]; ablation.py 0.1 [:77] | README 0.1 [:50] | three different defaults |
| T_E | RC/PP 0.01, simple/within-RC 0.02 [App. C.1]; BiB 0.01 | `--edge_threshold 0.02` [circuit.py:395-400] | README 0.01 | plotting only |
| T_N sweep (Fig. 3) | "sweeping over node thresholds" [§3.2] | — | `t.logspace(-4, 0, 15)` [notebook cell 5] | grid only in the notebook |
| eval examples | test split [§3.2] | ablation.py `--examples 100` [:94-99] | notebook `n_examples=40` [cell 4]; evaluate_circuit.sh 40 (broken) | 40 is what Fig. 3 uses [inferred] |
| eval start layer | ignore the first 1/3 [§3.2]; v2: "starting at layer 2" | ablation.py −1 [:106-111] | notebook 2 for Pythia [cell 2]; README 2 | consistent at 2 |
| embed in eval | not stated | `include_embed=False` [ablation.py:129; notebook cell 1] | — | embed never evaluated |
| ablation | mean, "position-specific" [§3.2 + fn. 2] | mean over the batch axis of the **patch** prompts of the eval batch [ablation.py:19-21, 26-34] | notebook same [cell 3] | which data the mean is over (§9.4) |
| error handling | errors are nodes [§2, §3.2] | `'default'` [ablation.py:22] | notebook 4 settings [cell 5] | — |
| data seed | — | 12 [data_loading_utils.py:14] | — | `--seed` unused |
| dtype | — | float32 for Pythia [circuit.py:461-464] | — | — |
| unpaired metric | −log P(y\|x) [§5] | −logit [circuit.py:554-559] | — | (§9.6) |

---

## 5. Data

### 5.1 Subject-verb agreement (the arena C tasks)

- **Origin:** Finlayson et al. (2021) nouns and structures, with verb sets restricted to verbs whose singular and plural
  forms are each one Pythia token [code README.md:19].
- **Format:** JSON lines with `clean_prefix, patch_prefix, clean_answer, patch_answer, case`.
- **Contrast:** the patch prefix flips the number of one noun, the subject in simple / RC / PP. In within-RC it flips
  the RC subject, which is the noun the verb agrees with (example below). The answers are the matching verb forms
  (checked in the local copies).
- **Files unchanged since 2024-03-29** (the commit history of `data/*.json`).

Record counts (local copies at the same SHA; prefix word lengths are constant within a file):

| task | file stem | example (clean → answer; patch → answer) | train / test records | words |
|---|---|---|---|---|
| Simple | `simple` | "The friends" → " have"; "The friend" → " has" | 480 / 120 | 2 |
| Within RC | `within_rc` | "The fathers that the chefs" → " do"; "...the chef" → " does" | 9,600 / 2,400 | 5 |
| Across RC | `rc` | "The friends that the dancer visits" → " go"; "The friend ..." → " goes" | 105,600 / 26,400 | 6 |
| Across PP | `nounpp` | "The teachers near the desk" → " are"; "The teacher ..." → " is" | 1,920 / 480 | 5 |

**What is actually used:**
- **Discovery:** the first 100 records of the seed-12 shuffle of `<task>_train.json` that have the minimum Pythia token
  length [code circuit.py:483-486].
- **Evaluation:** the first 40 such records of `<task>_test.json` [code notebook cell 4].
- **Lengths:** the minimum lengths under Pythia's tokenizer were not computed here (the tokenizer is not cached
  locally). Measure them when the environment is set up. The train and test minimum lengths must match for `aggnone`
  masks to apply; a mismatch fails loudly at the mask indexing in ablation.py:55-60 [inferred].
- **The split is not disjoint.** The paper says the test split consists of pairs "not used to discover the circuit"
  [paper §3.2]. 081 measured string-level overlap between the full files [081 README "Splits"]:
  - every test prefix of simple, within_rc and rc occurs in its train file;
  - identical (prefix, answer) items: simple 80 of 85 unique test items, within_rc 1,667 of 1,728, rc 18,202 of 18,933,
    nounpp 82 of 469.
  - How many of the 40 evaluation examples are among the 100 discovery examples is not known yet. Measure it under the
    Pythia tokenizer.

### 5.2 Bias in Bios (SHIFT, §4)

- **Data:** HF `LabHC/bias_in_bios`; professor (21) vs nurse (13) [code bib_shift.ipynb cell 1].
  - The ambiguous set is male professors (label 0) and female nurses (label 1), balanced to the smaller group and
    shuffled with seed 42.
  - The balanced set has 4 groups of equal size.
- **Classifier:** a linear probe on mean-pooled (non-pad) layer-ℓ residual, AdamW, lr 0.01, 1 epoch [paper App. E.1;
  code cell 3]. ℓ = 4 for Pythia and 22 for Gemma.
- **Discovery:** zero-ablation IG with `metric_fn = probe(acts)` if label 0, else −probe(acts) [code cell 6]. This is
  the signed probe logit towards the wrong class, which is the paper's m = −log C(y|x) up to a monotone transform
  [paper §4 step 1; §9.6].
  - 100 single-example batches; effects summed over positions excluding BOS, then averaged [code cells 7-8].
  - The paper's Pythia circuit: T_N 0.1, T_E 0.01, 67 features, 55 judged irrelevant [paper §4, App. C.2].
- **At HEAD the notebook is the Gemma version** (layer 22, threshold `> 6`, 43 features hand-listed) [code cells 0, 9,
  11]. The Pythia feature list is not in the repo, so Table 2's Pythia row cannot be re-run from the repo alone.

### 5.3 Clusters (§5)

- **Data:** downloaded from feature-circuits.xyz [code README.md:69-73].
- **Command:** `scripts/get_circuit_nopair.sh` runs with `--nopair`, sum aggregation, the zero patch and
  m = −logit(answer) [code get_circuit_nopair.sh:8-16; circuit.py:477-481, 551-559].
- **`get_circuit_cluster()` is broken:** it passes `edge_threshold=` to `get_circuit()`, which has no such parameter
  [code circuit.py:274-288 vs 19-34], so it would raise a TypeError [inferred]. The script path uses `__main__`, which
  does not call it.

---

## 6. Their evaluation, exactly

### 6.1 Formulas

Implemented as `get_fcs` in faithfulness.ipynb cell 4, with `run_with_ablations` from ablation.py:9-65:

```
E   = first 40 min-length examples of <task>_test.json (seed-12 shuffle)
S   = {attn_i, mlp_i, resid_i : i >= start_layer}      start_layer = 2 (Pythia); embed excluded; layers 0-1 untouched
mu_s = mean over the 40 examples (axis 0) of [f_s, eps_s] computed on the PATCH prompts of E, per position   (ablation.py:26-34)
F(M) = mean_E m_eval(clean, no SAEs spliced at all)
F(C) = mean_E m_eval(clean, for s in S in forward order:
            f_s, eps_s <- encode(x_s), x_s - x_hat_s;
            features not in C := mu_s.f ; error not in C := mu_s.eps (whole d_model vector);
            x_s <- decode(f_s) + eps_s)
F(∅) = same with C empty (every feature and every error at every evaluated site mean-ablated)
faithfulness(C)  = (F(C)       - F(∅)) / (F(M) - F(∅))
completeness(C)  = (F(M \ C)   - F(∅)) / (F(M) - F(∅))      # complement = ablate C, keep everything else
n_nodes(C)       = sum_{s in S} ( #features in C at s + #error nodes in C at s )   # per (position, unit) under aggnone
```

- **Ideal values:** faithfulness 1 and completeness 0 [paper Fig. 3 caption].
- **Paper wording:** m(C) is the mean of m with every node outside C mean-ablated, "set to their average value over
  data from D" [paper §3.2]. Footnote 2 says the means are position-specific (following Wang et al. 2023).
- **Which data the means come from:** the patch prompts of the 40-example evaluation batch itself, per position
  [code ablation.py:24-34]. Not the train split, not the clean prompts.
- **Denominators:** F(M) is the unspliced model, and F(∅) has all of S ablated. Because the sites run in order, the
  ablated sites' downstream recomputation is real; for example resid_L is re-encoded from the attn_L and mlp_L that were
  already modified [inferred from the forward-order splice].
- **Variants:** `handle_errors='remove'` forces errors out of C (so the complement keeps every error).
  `'resid_only'` forces the attn and mlp errors out of C [code notebook cell 4].
- **Random baseline** (`random=True`): Bernoulli membership with p = n_nodes / total, with **all errors kept**
  [code notebook cell 4]. It is not plotted in Fig. 3.

### 6.2 How the curves are produced

- **Input circuit:** `../circuits/pythia-70m-deduped_<task>_train_n100_aggnone_nodeall.pt`, a nodes-only, per-position
  run of circuit.py on 100 training examples [code notebook cell 4].
- **The knob:** T_N, swept over `t.logspace(-4, 0, 15)`, i.e. 1e-4 … 1 in 15 log-steps. Each threshold gives one
  (n_nodes, faithfulness, completeness) point [code notebook cell 5].
  - 0.1 is not on that grid: 10^−1 falls between grid points 10 and 11.
- **Tasks:** rc, nounpp, simple, within_rc.
- **Settings:** `features`, `features_wo_errs`, `features_wo_some_errs`, `neurons` [code notebook cell 5].
- **Bold mean curve:** each task's (n_nodes → faithfulness) is linearly interpolated (`scipy.interp1d`). The four
  interpolants are averaged on 100 log-spaced x values. The x range runs from (max over tasks of the minimum n_nodes) + 1
  to (min over tasks of the maximum n_nodes) − 1 [code notebook cell 6].
- **Plot limits:** faithfulness x ∈ [0, 30000], y ∈ [0, 1.1]; completeness x ∈ [0, 3000], y ∈ [−0.15, 1]
  [code cells 6-7].
- **Caching:** results are cached at `experiments/faithfulness/<md5>.pt`. The folder must exist [code cell 4].

### 6.3 Published numbers we could reproduce (Pythia only)

| claim | where | how to read it |
|---|---|---|
| Across-RC circuit, T_N = 0.1 (T_E = 0.01): **86 nodes, faithfulness 0.21** | §3.3; App. C.1 Fig. 7; v2 §3.3 "computed as in §3.2" | the one hard number on SVA |
| Feature circuits: the majority of performance by **~100 nodes**; neurons need **~1,500** for half | §3.2, Fig. 3 | read off the bold mean curve |
| Completeness: a few feature nodes remove the task; neurons need hundreds | §3.2, Fig. 3 | qualitative |
| Removing the resid errors caps faithfulness; removing the attn/mlp errors is mild | §3.2, Fig. 3 | qualitative |
| IG ≫ AtP for MLP0 and early resid; 30 across-RC inputs | App. H, Fig. 25 | qualitative scatter |
| SAE quality per site | App. B.1.3, Tables 3-6 | numeric, on Pile |
| BiB Pythia: Original 61.9 / 87.4 / 24.4; SHIFT 88.5 / 54.0 / 76.0; +retrain 93.1 / 52.0 / 89.0 (profession / gender / worst group) | §4 Table 2 | needs the unreleased Pythia feature list (§5.2) |

No per-task faithfulness table is published; Fig. 3 is a plot only [paper §3.2].

---

## 7. Rooting at something other than a task metric

- **The code supports any metric.** `patching_effect(clean, patch, model, submodules, dictionaries, metric_fn,
  method, steps, metric_kwargs)` takes any callable `metric_fn(model, **metric_kwargs)` that returns a per-example
  tensor; it is summed before `.backward()` [code attribution.py:224-243, 38-39, 135-136].
  - `get_circuit(..., metric_fn, metric_kwargs)` passes it through [code circuit.py:19-49].
  - **Precedent in the repo:** the BiB notebook roots at an internal readout, a linear probe on mean-pooled layer-22
    residual activations, with labels passed through `metric_kwargs` [code bib_shift.ipynb cells 6-7]. So rooting at an
    internal linear function of activations is something the authors did.
- **What rooting at one SAE latent would take:**
  - A metric such as `SAE_t.encode(t.get_activation())[:, pos, j]` for a target site t [inferred]. Restrict
    `submodules` to sites upstream of t; nodes downstream of t would get zero IE.
  - Nodes only, via `patching_effect`: no code change, just a different metric_fn and submodule list.
  - Full circuits via `get_circuit`: the edge loop hard-codes resid_5 → "y" as the root edge and loops over all six
    layers [code circuit.py:73-124]. A latent root would need a changed edge construction. That is **our adaptation**.
  - **Unverified nnsight risk:** reading an earlier module's output inside `metric_fn`, after later `set_activation`
    calls, under nnsight < 0.4 [inferred risk; test it].
- **Did anyone do it?**
  - The paper roots only at output metrics: logit/log-prob difference (SVA), the probe logit (BiB) and −log P (clusters)
    [paper §3.2, §4, §5].
  - Kharlapenko et al. 2025 (arXiv 2504.13756) also root at a task loss. Their nodes are (latent, token type), they
    use zero ablation, and they report IE approximations to be weak before layer 6 [their §4.1, App. E.2, read via
    summary].
  - 107's literature check found no SFC replication rooted at a latent [107 README "Prior replications"].
  - So a latent-rooted SFC row is **our adaptation** and must be labelled as such.

---

## 8. Exact commands (Pythia-70m, one SVA task)

**Setup** (environment choices, not algorithm changes):

```
git clone https://github.com/saprmarks/feature-circuits && cd feature-circuits
git checkout 7fbd82b895ae16294f4e6fc7bfc675d1d680d659
pip install "torch<2.6" "nnsight==0.3.7" "numpy<2"   # see §12.5 for why each pin
pip install -r requirements.txt                    # nnsight<0.4, numpy<2, ..., dictionary_learning (unpinned)
pip install "git+https://github.com/saprmarks/dictionary_learning@61ac634845bd76c839482f3b725ab3d898c8b277"
#   PyPI's dictionary-learning has only 0.1.0 (2025-02-12), which predates IdentityDict(device, dtype)
#   used by dictionary_loading_utils.py:47, 88-90 (the neuron path). Pin the companion commit.
#   Its pyproject requires nnsight >=0.3.0,<0.4.0, which is consistent with nnsight 0.3.7.
pip freeze > env_freeze.txt                        # record exact versions (§9.7 depends on nnsight)
mkdir -p dictionaries && <link or copy experiments/030-cross-sae/dictionaries/pythia-70m-deduped there>
mkdir -p circuits/figures experiments/faithfulness
# graphviz system binary needed for plotting (circuit_plotting.py); the circuit .pt is saved before plotting.
```

**Discovery: nodes only, per position.** This is the input to Fig. 3 and to our curve:

```
python circuit.py --model EleutherAI/pythia-70m-deduped --dataset rc_train \
    --num_examples 100 --batch_size 10 --aggregation none --nodes_only --device cuda:0
# -> circuits/pythia-70m-deduped_rc_train_n100_aggnone_nodeall.pt
#    {"examples": [...100 dicts], "nodes": {"embed","attn_i","mlp_i","resid_i": SparseAct(act=(pos,32768), resc=(pos,1))}, "edges": None}
```

- After saving, the per-position plotter is called with `edges=None` and will probably fail. That is harmless; the file
  already exists [inferred from circuit.py:629-661].
- Repeat for `nounpp_train`, `simple_train` and `within_rc_train`.

**Discovery: the full circuit with edges**, for display or the edge comparison. Paper settings: RC/PP 0.1 / 0.01,
simple/within-RC 0.2 / 0.02.

```
scripts/get_circuit.sh EleutherAI/pythia-70m-deduped rc_train 0.1 0.01 none
# -> circuits/pythia-70m-deduped_rc_train_n100_aggnone_node0.1.pt  (+ figure under circuits/figures/)
```

**Evaluation.** At this SHA, both official evaluation entry points fail before computing anything:

1. **ablation.py:** line 5 does `from loading_utils import load_examples`, but `load_examples` now lives in
   `data_loading_utils.py`. That file first appears in commit `cbb5692` ("refactoring stuff", 2025-01-21), so the move
   most likely happened there [inferred from the commit history]. `faithfulness.ipynb` cell 0 has the same
   import, so both raise an ImportError [code loading_utils.py has no `load_examples`; data_loading_utils.py:8].
2. **scripts/evaluate_circuit.sh** has three further defects:
   - it passes `--num_examples`, `--dict_id $DICTID` and `--batch_size`, which ablation.py's parser does not define
     [code evaluate_circuit.sh:15-16, 21 vs ablation.py:69-113];
   - `$DICTID` is never set;
   - it passes `--data ${EVAL_DATA}.json`, and ablation.py appends `.json` again [code evaluate_circuit.sh:14 vs
     ablation.py:144].
3. **ablation.py's `__main__` computes faithfulness only, not completeness** [code ablation.py:229-235].

4. **`run_with_ablations` raises a TypeError under nnsight 0.3.x** [issue #15, closed by the reporter with a
   user-supplied fix]. At ablation.py:47-49, `res.shape[-1]` is an nnsight proxy inside the trace, so `expand()`
   rejects it. The fix is to take the last dimension from the concrete `patch_states[submodule].res.shape[-1]`, which is
   the same value, d_model = 512.

**Minimal path:** run `experiments/faithfulness.ipynb` with three edits, which together are **our adaptation (an
import path, a shape read and a config value; no algorithm change)**:
- cell 0: `from data_loading_utils import load_examples`. This restores the paper-era import; at `64d1cb3`,
  `loading_utils.load_examples` is the identical function (§9.8).
- ablation.py:47-49 (imported by the notebook): the issue #15 shape fix above.
- cell 1: `model_name = "EleutherAI/pythia-70m-deduped"`. HEAD is set to Gemma.
- If torch ≥ 2.6 is installed despite the pin, `t.load` of the pickled circuit (circuit.py:510, notebook cell 4,
  ablation.py:137) fails, because `weights_only` now defaults to True. `cluster_rerun` passes `weights_only=False`
  [code main...cluster_rerun diff, circuit.py]. Prefer the torch pin, which needs no code change.

Then run all cells. Each task gives `out[threshold] = {n_nodes, fc, fccomp, faithfulness, completeness}` for the 15
thresholds, plus `fm` and `fempty`.

**Single circuit at T_N = 0.1** (for check F3): call `get_fcs(..., thresholds=[0.1])` in the same notebook. Or run
ablation.py with the import fixed: `python ablation.py --model EleutherAI/pythia-70m-deduped --circuit
circuits/pythia-70m-deduped_rc_train_n100_aggnone_nodeall.pt --data rc_test --examples 40 --threshold 0.1 --ablation
mean --handle_errors default --start_layer 2` (this gives faithfulness only).

**Runtimes and expected outputs:**
- Neither the paper nor the README states a runtime, and neither gives expected numerical outputs.
- The BiB notebook's saved outputs show about 11 s per example for IG over 23 Gemma-2-2B sites [code bib_shift.ipynb
  cells 20, 27]. That is not informative for Pythia.
- Pythia nodes-only IG: about 19 sites × 10 steps of forward+backward per batch on prompts of ≤ 10 tokens. I expect
  seconds to a few minutes on one GPU [inferred, not measured].
- The edge pass costs one backward per selected downstream (example, position, unit), so it scales with how many nodes
  pass T_N [inferred]. Measure it.
- **Memory:** see §12.3. Nodes-only with `--batch_size 10` should fit comfortably on a 16 GB GPU (about 4 GB
  estimated). The faithfulness curves need nothing more, because edges are never used in evaluation [issue #14].

---

## 9. Ambiguities and paper-vs-code differences

1. **IG endpoint** [paper Eq. 4 vs code attribution.py:126-128].
   - Paper: α ∈ {0, …, (N−1)/N} at α·clean + (1−α)·patch, which samples the patch end and never the clean end.
   - Code: the reverse; it samples clean (α = 0) and never patch.
   - Both are 10-point Riemann sums; the code's is the left sum from clean. Run the code as is; mention it in the write-up.
2. **Edge gradient factor** [paper App. A.1 Eq. 5 vs code circuit.py:86, attribution.py:138-146].
   - Paper: ∇_d m evaluated at d_clean.
   - Code: the IG-averaged gradient from the node pass.
   - The Jacobian ∂d/∂u is at clean in both.
3. **Edge computation is gated per example and position** [code circuit.py:51-53 vs paper §3.1].
   - Downstream units get edges only in the (example, position) cells where their own unaggregated |IE| > T_N.
   - The paper thresholds the example-mean.
   - So a node that passes on the mean has edges only from the examples where it passed individually. Those are summed
     and divided by all b, which can shrink the reported weights [inferred].
   - Upstream units are not gated in jvp, only at plot time.
4. **Mean-ablation reference set** [paper §3.2 fn. 2 vs code ablation.py:24-34].
   - Paper: the mean is "over data from D".
   - Code: the per-position mean over the **patch** prompts of the 40-example evaluation batch, recomputed for every
     batch and task. For SVA the patch prompts are the same templates with the subject number flipped, so this is close
     to a dataset mean, but it is in-sample to the evaluation batch [inferred].
5. **Evaluated layers.** v3 says "ignore the first 1/3"; v2 says "starting at layer 2"; the notebook uses
   `start_layer = 2` [paper §3.2; v2 §3.2; notebook cell 2].
   - Layers 0-1 and the embedding are left at clean values: they are not spliced, not ablated, and not counted in
     n_nodes [code ablation.py:129-134].
   - The 86-node figure for the RC circuit (Fig. 7) shows all layers. **It is unclear whether 86 counts all layers
     (embedding included) or only layers ≥ 2.** Count both.
6. **The metric in different places.**
   - §2 uses the log-probability difference, Table 1 writes p(·) − p(·), and the code uses the logit difference. Logit
     difference = log-prob difference exactly; Table 1's "p" is read as notation [inferred].
   - For unpaired data, §5 says m = −log P(y|x) but the code uses −logit(y) [code circuit.py:554-559]. These differ by
     log Σ exp(logits), which has its own gradient.
   - For BiB, §4 says m = −log C(y|x) but the notebook uses the signed probe logit [code bib_shift.ipynb cell 6].
   - None of these affect SVA.
7. **Gradient-hook semantics under nnsight < 0.4** [code attribution.py:33-37, 311-313].
   - In jvp the upstream error's gradient is zeroed (line 311) **before** `upstream_act.grad` is read (line 313). If
     nnsight returns the swapped (zero) gradient there, every edge from an upstream error node would be exactly 0.
   - In `_pe_attrib` the read (line 33) comes before the zeroing (line 34).
   - I could not settle this without running it; it is check F7.
   - **The order changed over the code's history** (§12.2): read-then-zero in `bce4ecd` (jvp-fix) and in `8eaa619`
     (iclr-updates, `jvp_new`); zero-then-read at `64d1cb3` (attribution.py:337-339) and at HEAD (311-313).
   - **Issue #16** (2025-06-28, no reply) asks whether lines 313-314 should be swapped, i.e. whether the gradient is
     read before the backward that produces it. Probably not: the same read-before-backward pattern is how
     `_pe_attrib` gets every node gradient (attribution.py:33 then :39) [inferred from the nnsight idiom used in the
     repo].
8. **Paper-era vs HEAD code.**
   - The same at `64d1cb3` (2024-10-02, the last commit to faithfulness.ipynb): the IG implementation (steps, α
     convention, joint path), `features_by_submod`, `load_examples` and the thresholds [code @64d1cb3 attribution.py:120-193;
     circuit.py:59, 71-72; loading_utils.py:8-67].
   - Changed in 2025: the jvp internals ("monkey-patching to get around an nnsight bug", sparse-tensor construction),
     `retain_graph` removed, and the submodule loading refactored.
   - Small numerical differences against the paper's figures are therefore possible. The algorithm is the same [inferred].
9. **Three node-threshold defaults** (0.2 CLI, 0.1 function, 0.1 evaluator), and the CLI aggregation default `sum` is
   not the SVA setting [§4 table]. Always pass both explicitly.
10. **"Held-out" test data overlap the training files** (§5.1). The paper's statement [paper §3.2] does not hold at the
    string level for three of the four tasks.
11. **`load_examples` returns None** if fewer than n records of minimum length exist [code data_loading_utils.py:66-67].
    `len(examples)` then crashes, so this would be loud, not silent.
12. **Edges vs v2.** v2 corrected for paths through intermediate nodes by subtracting Jacobian products [paper v2 App.].
    v3 and the code use stop-gradients on the intermediate sites [paper App. A.1; code circuit.py:98-115]. The two are
    the same estimand by different arithmetic [inferred].
13. **Annotations path.** The README names `annotations/10_32768.jsonl`, but the code reads
    `annotations/pythia-70m-deduped.jsonl`, which exists [code README.md:38 vs circuit.py:634-645]. Only plots use it.

---

## 10. Fidelity checks (to know our run of their code matches the paper)

Tolerances are **our choice**. The paper gives none.

| # | check | reference | pass if |
|---|---|---|---|
| F1 | Dictionaries load and behave: variance explained, L0 and % CE recovered per site on 128 Pile contexts × 128 tokens | App. B.1.3 Tables 3-6 | VE within ±3 pp, L0 within ±20%, % CE recovered within ±3 pp of each table entry (a different Pile sample, so loose) |
| F2 | Splice exactness: with every node kept (mask all True, errors kept), `run_with_ablations` = F(M) | follows from Eq. 1 / Eq. 6 | \|F(C_all) − F(M)\| / \|F(M) − F(∅)\| < 1e-3 |
| F3 | Across-RC at T_N = 0.1 (`aggnone`, 100 train, 40 test, start layer 2, mean ablation, default errors) | §3.3 / Fig. 7: 86 nodes, faithfulness 0.21 | faithfulness 0.21 ± 0.05. Report n_nodes both ways (layers ≥ 2; all sites); pass if one is within ±25% of 86 |
| F4 | Fig. 3 mean curves, `features` setting | §3.2, Fig. 3 | interpolated faithfulness ≥ 0.5 at ≤ 200 nodes (paper: about 100). For `neurons`, faithfulness 0.5 reached between about 750 and 3,000 nodes (paper: about 1,500). Completeness of feature circuits ≤ 0.2 within the first few hundred nodes |
| F5 | IG vs exact IE on 30 across-RC inputs (`method='exact'`) | App. H, Fig. 25 | IG tracks exact on MLP0 / early resid better than AtP (qualitative; report the correlation per site) |
| F6 | Determinism: two discovery runs | — | max \|ΔIE\| ≤ 1e-4 (float32 GPU; small precision noise is tolerated, judged by how widespread it is) |
| F7 | Error-node edges are not all zero (§9.7) | App. A.1 (error nodes have edges, e.g. the triangles in Fig. 7) | at least one non-zero upstream-error entry in RR/MR/AR edges of the RC circuit; if all zero, report it as a property of the code at this SHA |
| F8 | The discovery/evaluation overlap is measured, not assumed | §3.2 vs §5.1 | the count of the 40 test examples that equal a discovery example is reported per task |
| F9 | Memory and environment record | §12.3, §12.5 | `pip freeze` saved. Peak GPU memory (`torch.cuda.max_memory_allocated`) logged for the nodes-only pass and the evaluation; nodes-only at `--batch_size 10` stays under 8 GB. If it does not, investigate before raising the batch size |
| F10 | Node effects do not depend on batch size | §3.3 (per-example gradients do not interact) | nodes-only `rc_train` at `--batch_size 10` and `--batch_size 1`: max \|ΔIE\| ≤ 1e-4 on the saved means |
| F11 (optional) | Paper-era code agrees with HEAD on nodes | §12.4 | nodes-only `rc_train` at `64d1cb3` (dictionary_learning `sfc` branch, nnsight 0.3.x) vs HEAD: max \|ΔIE\| ≤ 1e-4; and the T_N = 0.1 node set is identical. Run only if F3 or F4 fails |

If F3 or F4 fails, stop and investigate before any comparison. These two are the published SVA numbers.

---

## 11. Mapping onto arena C

**What we take from them, unmodified:**
- the model, the 19 dictionaries and the error nodes (§2);
- the four SVA tasks and their files: the 100-example min-length discovery subset and the 40-example min-length
  evaluation subset (§3.1, §5.1);
- the metric (§3.2);
- IG node effects (§3.3) and per-position (`aggnone`) aggregation (§3.4);
- the T_N sweep `logspace(-4, 0, 15)`, giving their faithfulness/completeness-vs-n_nodes curves (§6);
- their evaluator (`get_fcs`), with the import fix in §8 as the only change.

**Gaps that would make a comparison unfair if left implicit.** Each needs a decision; anything we change is **our
adaptation**.

1. **Node units.**
   - SFC's size is the number of (position, feature) plus (position, error) pairs at layers 2-5. A latent at 3 positions
     counts 3.
   - If WCM selects position-free latents, the sizes are not comparable. Expanding them to every position inflates WCM's
     count.
   - Report WCM in SFC's units, as the position-expanded mask that SFC's evaluator actually ablates. Report latent counts
     beside it. **Our adaptation** for the WCM side.
2. **Error nodes.**
   - SFC treats errors as ablatable, countable nodes, selected by |IE|.
   - Our method keeps errors at clean values. Scored by `get_fcs`, a WCM circuit must state its error membership:
     - "keep all errors" is SFC's `random` convention, and gives WCM the resid-error advantage seen in Fig. 3;
     - "no errors" is the `features_wo_errs` convention.
   - Score WCM under both, and count kept errors in n_nodes. **Our adaptation.**
3. **Ablation reference.**
   - SFC mean-ablates to per-position means of the 40 evaluation patch prompts.
   - Our tests use our own A/C means and zero ablation.
   - Both methods are scored by both evaluators (rule 5 of the 107 README). Neither is converted into the other.
4. **Layers 0-1 and the embedding are free** in SFC's evaluation.
   - A WCM fitted over all 19 sites but scored by `get_fcs` is credited only for layers 2-5.
   - Whether WCM's node universe should exclude layers 0-1 and the embedding to match is a decision. Excluding them is
     **our adaptation**; keeping them is allowed but must be stated.
5. **Discovery data.**
   - SFC sees 100 training examples of one token length.
   - Fitting WCM on the full training split (081: simple 150 … rc 34,003 items under GPT-2) gives it far more data.
   - Fit WCM on the same 100 examples for the matched row; any larger-data WCM row is labelled as such.
6. **The test overlaps the training data** (§5.1). Held-out faithfulness on simple / within_rc / rc is close to
   in-sample for both methods. Report F8. Optionally add the `SFC_SPLIT=prefix` split as an extra evaluation, labelled
   **our adaptation** and never presented as SFC's protocol.
7. **No operating point.**
   - SFC's natural knob is T_N. Fig. 7's T_N = 0.1 is one point of the curve, not "SFC's circuit".
   - Compare whole curves, by reading the curves at matched n_nodes and by area under the curve on a fixed grid (107
     rule 3). The paper's circuits were tuned for annotation, to faithfulness > 0.2 [paper §3.3, App. C.1].
8. **Edges.** SFC's evaluation is node-only, and edges are display-only. Any edge comparison (for example against WCM
   edges) is outside their evaluation and is **our analysis**.
9. **Latent-rooted SFC** (DAN-48 / open decision 4). This is supported for nodes by a different `metric_fn`, but the
   edge root is hard-coded (§7). It is **our adaptation** and is shown as such.
10. **Software versions** (environment, not algorithm):
    - nnsight 0.3.7, torch < 2.6, numpy < 2, and `dictionary_learning` at `61ac634` (§12.5);
    - float32;
    - record the exact versions installed, because §9.7 depends on nnsight's gradient-hook behaviour.

---

## 12. Branches, bug fixes and memory (follow-up, 2026-11-01)

The GitHub REST API rate limit was hit partway through, so some ahead/behind counts come from the github.com compare
pages, which report "nothing to compare" (0 ahead) or "N commits" (N ahead) but not the behind count.

### 12.1 Branch map

| branch | tip | ahead of main | what it is |
|---|---|---|---|
| `main` | `7fbd82b` 2025-07-14 | — | HEAD; §0 |
| `jvp-fix` | `bce4ecd` 2024-08-02 | 0 ahead, 55 behind [API compare] | one commit, "hopeful bugfixes for jvp", on top of `cf08078`; §12.2 |
| `iclr-updates` | `8eaa619` 2024-09-15 | 0 (nothing to compare) | jvp-fix + "wip" + "new circuit discovery"; §12.2 |
| `gemma`, `gemma-2` | `cbb5692` 2025-01-21 | 0 [`cbb5692` is in main's history] | the Gemma line, merged; "refactoring stuff" |
| `cluster-fix` | `1d87366` 2025-03-01 | 0 | "fixing hacking", in main |
| `cluster_rerun` | `6f24d65` 2025-04-09 | 7 commits, 11 files | cluster pipeline (`circuit_clusters.py`, `convert_clusters.py`), memory hygiene in `get_circuit` (`.detach()`, `del`, `t.cuda.empty_cache()`), a fixed `get_circuit_cluster`, and `t.load(..., weights_only=False)`. Its sparse-storage jvp is already in main |
| `eric` | `0da9b74` 2025-10-14 | 7 commits, 55 files, additions only | E. Michaud's clustering scripts (Feb-Mar 2024) and "added circuit website files". Does not touch the SFC pipeline |
| `sam`, `triangles`, `nnsight_upgrade`, `no_pair` | — | 0 | merged |
| `can`, `4-gradient-clustering` | — | 44 and 30 commits | C. Rager's early clustering work (issues #2-#5, Jan 2024) [inferred from the issue list]; not read |

No release tags exist [github.com/.../tags].

### 12.2 What `jvp-fix` and `iclr-updates` change

**Background:** this is the edge code of the v2 era. Resid→resid edges were computed as the total effect minus
corrections through the MLP and attention: `RR − RMR − RAR` [issue #10, quoting circuit.py@c1a9b7b:157-208]. This is
the Jacobian-product correction of [paper v2 App.].

**Issue #10** (2024-08-01, J. Kaunismaa, open; 4 replies) reported:
1. RMR and RAR are always zero;
2. `downstream_feat` indexes the wrong tensor when `left_vec` is a dict;
3. `jv` is computed identically to `vjv`.

The owner confirmed points 1 and 3 ("completely bugged" correction terms) and pushed `jvp-fix`. In a later reply he
says he "never got around to merging" it and points to the gemma branch, which uses the stop-gradient formulation of
App. A.1 instead [issue #10, owner replies].

**`jvp-fix` (`bce4ecd`), the whole diff:**

| file:line (at the parent `cf08078`) | change |
|---|---|
| attribution.py:8 | `DEBUGGING = False → True` (turns on nnsight `validate` and `scan`) |
| attribution.py:~346-357 (jvp loop) | for dict-valued `left_vec`, backprop Σ_m (∇_d m ∇_m d)·m, summed over the intermediate features, instead of indexing the intermediate tensor with a downstream index. Fixes #10 points 1 and 2 |
| attribution.py:~355, 368-391 | the "without right" output becomes the plain upstream gradient `vj` instead of a copy of `vjv`. Fixes #10 point 3 |
| circuit.py:199 | blank line only |
| `dictionary_learning` | submodule replaced by a symlink `../dictionary_learning/`. Later removed in `6a1d1d1` ("Remove incorrect symlink", 2025-02-28) |

**`iclr-updates` (`8eaa619`), diff from `bce4ecd`:**

| file | change |
|---|---|
| attribution.py | `DEBUGGING` back to False. `profile()` caches each submodule's `is_tuple` from one scan trace instead of one per call |
| attribution.py | new `_pe_analytic`: gradients from `x.grad` projected through the decoder, with no splice. Not called by `get_circuit` |
| attribution.py | new `jvp_new`: splices both sites; loops over per-(example, position) downstream indices; reads `vjv`, **then** zeroes `x_res.grad`, then backprops; saves the **dense** `vjv` for every downstream index and `t.stack`s them (see §12.3) |
| circuit.py | MR, AR, RM and RA via `jvp_new`. RR computed analytically from the SAE weights: feature→feature, error→feature, feature→error and a dense error→error term of shape (b, s, b, s). This replaces `RR − RMR − RAR` |
| circuit.py | `features_by_submod_new = effects.abs() > T_N`: the per-example, per-position gating that HEAD still has (§9.3) |
| circuit.py | `DEBUGGING = True`; sparse utilities moved to `coo_utils.py` |
| activation_utils.py | `__eq__`, and SparseAct-vs-SparseAct `__gt__` / `__lt__` |

**Merge status:**
- Both branches are fully contained in main (0 commits ahead).
- Their content is **superseded**: by `64d1cb3` (2024-10-02) the analytic RR block is commented out. Every edge,
  including RR, uses one `jvp` with `intermediate_stopgrads` [code circuit.py@64d1cb3:96-141]. That is the v3 App. A.1
  formulation and what HEAD runs [code circuit.py:78-124].

**Do they resolve the §9 ambiguities?**

| §9 item | jvp-fix | iclr-updates | status at HEAD |
|---|---|---|---|
| 9.1 IG endpoint | not touched | `_pe_ig` not in the diff | unchanged since at least `64d1cb3`; code samples α = 0..0.9 from clean |
| 9.2 edge gradient factor (IG-mean vs clean) | n/a (v2 edges) | still `grads[downstream]` from the IG pass | unchanged |
| 9.3 per-example edge gating | n/a | **introduced here** | unchanged |
| 9.7 error-node edges zeroed | read-then-zero order | read-then-zero order (`jvp_new`) | zero-then-read since `64d1cb3`: still open, F7 |
| v2 RR-edge bug (#10) | fixed (correction terms non-zero) | made moot by the analytic RR | moot: stop-gradient edges |

**Consequences for the paper's numbers:**
- The #10 bug affects **edges only** (resid→resid weights in the v2-era plotted circuits).
- No evaluated number depends on edges. The owner says "only the nodes are ever actually used" and that no way of
  evaluating edges was proposed [issue #14, owner reply]. The faithfulness notebook loads `nodeall` files [code notebook
  cell 4].
- So Fig. 3, the 86-node / 0.21 figure, and Table 2 are unaffected by any edge bug [inferred from #14 + the code path].

### 12.3 Memory: where it blows up, and what controls it

**Where.** Through `64d1cb3` (main from 2024-10-02 until 2025-01-20, i.e. what "late 2024" ran), `jvp` keeps one
**dense** vector `vjv` of shape (b, s, F+1) for every selected downstream (example, position, unit):

```
vjv_values[downstream_feat_idx] = vjv.save()                       # attribution.py@64d1cb3:342
vjv_values = t.stack([v.value for v in vjv_values.values()])       # :378  -> (n_down, b, s, F+1), dense
return t.sparse_coo_tensor(vjv_indices, vjv_values, size=(b,s,F+1,b,s,F+1))   # :380, hybrid COO with dense blocks
```

- The same pattern is in `iclr-updates`' `jvp_new`.
- All ~30 edge tensors per batch stay on the GPU until `get_circuit` aggregates them (5 calls per layer × 6 layers)
  [code circuit.py@64d1cb3:110-141].
- **The fix:** `2b3ffae` ("more efficiently storing sparse tensors", 2025-03-24) and `5b7fa59` (same day) store only
  `vjv.nonzero()` and `vjv[vjv != 0]` [code attribution.py:316-317 at HEAD].
- **Other late-2024 problems:**
  - `requirements.txt@64d1cb3` had `nnsight>=0.2.9`, which resolved to nnsight 0.3.6 or 0.3.7 at the time (PyPI
    release dates: 0.3.7 on 2024-11-20).
  - 0.3.0 "had breaking changes" [PR #12, A. Karvonen, 2024-12-18, open].
  - The code needed the `sfc` branch of `dictionary_learning` for `hacked_forward_for_sfc`. Without it, `jvp` falls
    into a `try/except: breakpoint()`, which looks like a hang [issue #13, owner reply; code attribution.py@64d1cb3:317-345].
  - `circuit.py@64d1cb3:19` also sets `DEBUGGING = True`.
- **No issue in the repo (#1-#17) reports the memory blow-up itself.**
- `cluster_rerun` adds a "memory efficiency" commit (C. Rager, 2025-03-07): `.detach()`, `del` and `empty_cache` in
  `get_circuit`. It is not in main.

**Estimates from tensor shapes.** All [inferred], float32. b = batch size, s = prompt length in tokens, F = 32,768,
d = 512. One dense site state S = b·s·(F+d)·4 B = **8.0 MB** at b = 10, s = 6. Six tokens is my guess for the across-RC
minimum length; measure it.

| component | formula | b = 10, s = 6 |
|---|---|---|
| 19 dictionaries on GPU (discovery) | 19 × 134 MB (local `ae.pt` sizes) | 2.55 GB |
| clean + patch states, 19 sites | 2 × 19 × S | 0.30 GB |
| effects, deltas, IG grads, 19 sites | ≈ 3 × 19 × S | 0.46 GB |
| one site's IG trace (10 interpolants + their grads + 10·b forward passes) | ≈ 2 × 10 × S + small | ≈ 0.2 GB |
| **nodes-only peak** | sum | **≈ 3.5-4 GB**; grows about linearly in b (b = 32: ≈ 5-6 GB) |
| edges, pre-fix (`64d1cb3`, `iclr-updates`) | 4 B × b·s·(F+1) × Σ_calls n_down = **7.9 MB per downstream index-call** | 2,000 index-calls ≈ 16 GB, so OOM on a 16 GB card |
| edges, HEAD | Σ_calls n_down × nnz(vjv) × 52 B; nnz ≤ s·(L0_up + 1) for one example (no cross-example gradients); L0 54-215 [App. B.1.3] | ≈ 70 KB per index-call; 2,000 ≈ 0.14 GB. The cost becomes time: one backward per index-call |
| evaluation (`get_fcs`, 40 examples, 12 sites) | 18 dictionaries (2.4 GB) + 12 × S(b = 40) | < 3 GB |

**What sets n_down:** the number of (example, position, unit) cells with per-example |IE| > T_N at the downstream site
[code circuit.py:51-53]. Each resid site is downstream in 3 calls (MR, AR, RR), each MLP in 1 (RM) and each attention
in 1 (RA). n_down rises steeply as T_N falls and linearly with b. The paper's display thresholds (0.1 / 0.2) keep it
moderate, but at the bottom of the sweep (1e-4) almost every active unit qualifies. That is why the sweep is run
nodes-only.

**Memory-safe configuration for a 16 GB GPU:**
- HEAD `7fbd82b`, nnsight 0.3.7, `--aggregation none --nodes_only --batch_size 10`. Any batch size up to 32 should fit
  (estimate).
- `--nodes_only` skips every jvp: `get_circuit` returns at circuit.py:65-71, before any edge code.
- This loses nothing for the faithfulness and completeness curves, which read only the nodes [issue #14; notebook cell 4].
- **Full edges, only if needed for display:** HEAD (sparse storage), T_N at the paper's display values (≥ 0.1), batch
  10, with peak memory logged (F9).
- Never run edges at sweep thresholds, and never run edges on `64d1cb3` or `iclr-updates` on this card.

### 12.4 Which code produced the paper's figures and numbers

- **No tag or branch names the camera-ready.**
- **v2 (2024-03-31), Pythia only:** produced with late-March-2024 code ("evaluation script updates", "add agreement
  data", 2024-03-28/29). That code had the #10 RR-edge bug, which affects edges only (§12.2).
- **v3 (2025-03-27), adds Gemma:**
  - The Gemma pipeline lands on 2024-09-27 → 2024-10-02 ("add support for evaluating Gemma circuit faithfulness", "gemma
    rc circuit", "finalizing gemma stuff" = `64d1cb3`).
  - `experiments/faithfulness.ipynb` was last changed at `64d1cb3`, with `model_name` set to Gemma.
  - `dictionary_learning`'s `sfc` branch tip is 2024-09-30 ("hack for sfc", "IdentityDict import").
  - **Best candidate for the v3 Gemma figures: `64d1cb3` + `dictionary_learning@sfc`** [inferred from dates; not
    stated anywhere].
  - `iclr-updates` (2024-09-15) is a work-in-progress ancestor: analytic RR and DEBUGGING on, both removed by
    `64d1cb3`. It is not the final code [code diffs in §12.2].
- **The Pythia SVA numbers did not change:** 86 nodes and faithfulness 0.21 at T_N = 0.1 in both v2 and v3, with
  identical Fig. 7 thresholds [paper v2 §3.3; v3 §3.3, App. C.1].
  - They may have been carried over from the 2024 run rather than recomputed [inferred].
  - I did not read the March-2024 `_pe_ig`, so I cannot confirm that v2's node estimator equals HEAD's.
- **Nodes at `64d1cb3` and HEAD** (the only thing any number depends on): identical IG (steps, α convention, joint
  per-site path), aggregation, threshold rule, data loader, evaluator logic and edge formulation [code attribution.py,
  circuit.py, loading_utils.py and ablation.py @64d1cb3 vs HEAD, §9.8].
- **Changes since `64d1cb3`:**
  - jvp storage (memory) and the gradient-hook ordering are unchanged since `64d1cb3`;
  - the `retain_graph` and tracer flags;
  - the `parallel_attn` flag was inverted in name and condition, with the same effect: Pythia gets no attn→mlp edges at
    `64d1cb3` (`False` + `if parallel_attn`) or at HEAD (`True` + `if not parallel_attn`) [code circuit.py@64d1cb3:120,
    472-475 vs HEAD:103, 453-456];
  - `--nopair` and `--aggregation sum` raise `NotImplementedError` at `64d1cb3` [code circuit.py@64d1cb3:490-498];
    they work again at HEAD.

### 12.5 nnsight version by commit

| commit | `requirements.txt` | what pip resolves today | known status |
|---|---|---|---|
| `bce4ecd` (jvp-fix, 2024-08-02) | `nnsight>=0.2.9` | 0.5.x | written for nnsight 0.2.x: it predates 0.3.0 (2024-08-30) [PyPI dates] |
| `8eaa619` (iclr-updates, 2024-09-15) | `nnsight>=0.2.9` | 0.5.x | 0.2.x or 0.3.x; I cannot tell from the code [unknown] |
| `64d1cb3` (paper era) | `nnsight>=0.2.9` (+ `dictionary_learning@sfc`) | 0.5.x | in Dec 2024, bib_shift at this main ran on **0.2.21**; 0.2.9 failed, and 0.3.0 "had breaking changes" [PR #12]. The `sfc` branch is from late Sept 2024, when 0.3.x was current [inferred]. Which version the authors used is unknown; the evidence conflicts |
| `5b7fa59` … `7fbd82b` (HEAD) | `nnsight<0.4` (from `3b940e1`, "nnsight downgrade :(", 2025-02-26) + `dictionary_learning` | 0.3.7 | the maintained combination. dictionary-learning 0.1.0 and `61ac634` require `nnsight>=0.3.0,<0.4.0` [pyproject @61ac634; issue #17 resolver error] |

**Known breakage:**
- **nnsight 0.4** (2025-01-31) broke `jvp` (hang/loop, issue #13); the fix was the `<0.4` pin.
- **nnsight 0.5.x** (current): untested and expected to break the `<0.4` code [inferred].
- **ablation.py under 0.3.x:** the `expand()` TypeError of issue #15 (§8 fix).
- **bib_shift:** the notebook hits an nnsight graph IndexError on 0.3.x for one user. Pinning 0.2.21 conflicts with
  dictionary-learning's `>=0.3` [issue #17, open].
- **torch ≥ 2.6:** `t.load` of the pickled circuit dicts fails (`weights_only` default); pin torch < 2.6 (§8).

### 12.6 Recommendation for arena C

**Run main HEAD `7fbd82b895ae16294f4e6fc7bfc675d1d680d659`** with:
- nnsight 0.3.7, torch < 2.6, numpy < 2, and `dictionary_learning@61ac634`;
- the three evaluator edits of §8: the import path, the issue #15 shape read, and the Pythia `model_name`. Each is
  labelled **our adaptation**; none changes the algorithm;
- nodes-only discovery for the curves.

Why:
1. **Fidelity.** The node estimator, aggregation, threshold rule, data subset and evaluator are line-for-line the same
   as the paper-era `64d1cb3` (§12.4). Every published number depends only on nodes (§12.2).
2. **Bug fixes.** HEAD has the sparse jvp storage (no memory blow-up), the stop-gradient edges of App. A.1 (the #10 bug
   is moot), and no `breakpoint()` trap.
   - `jvp-fix` and `iclr-updates` are older, fully contained in main and superseded. Running either would reintroduce
     the v2 edge code or the dense-storage blow-up, and fixes nothing in §9.1-§9.3.
3. **Runnability.** HEAD is the only commit with a pinned, resolvable dependency set (`nnsight<0.4` + the PyPI or git
   `dictionary_learning`). `64d1cb3` needs the unpinned `dictionary_learning@sfc` and an nnsight version that cannot
   be determined.

**Use `64d1cb3` only as a diagnostic** (F11), and only if F3 or F4 fails. `cluster_rerun`'s extra memory hygiene is
not needed for Pythia nodes-only and is not used.
