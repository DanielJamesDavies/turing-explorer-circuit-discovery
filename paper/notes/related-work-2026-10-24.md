# Related work, DAN-31 pass (2026-10-24)

Scope: `paper/references.bib`, main-text Related Work, and the Extended Related Work appendix (`app:related`).
Linear was read only; nothing there was changed.

## Bib entries added (20), each checked against its source page

| Key | Paper | Checked against |
|---|---|---|
| `seth2026circuitkit` | CircuitKIT (2607.19317) | arXiv abs + HTML (Pillar 5: equal-size random subgraphs, z-score, no fitting) |
| `meloux2025dead` | The Dead Salmons of AI Interpretability (2512.18792) | arXiv abs (no venue) |
| `mahale2026explanation` | Explanation Multiplicity (2608.13754) | arXiv abs (single author) |
| `wu2026demystifying` | Demystifying Variance in Circuit Discovery of LLMs (2606.16920) | arXiv abs |
| `anani2026certified` | Certified Circuits (2602.22968) | arXiv abs, comments say ICML 2026 |
| `meloux2025everything` | Everything, Everywhere, All at Once (2502.20914) | arXiv abs + ICLR 2025 poster page (author order Meloux, Maniu, Portet, Peyrard; the v1 HTML title block orders Portet before Maniu, so I followed arXiv and ICLR) |
| `chen2026rome` | All Circuits Lead to Rome (2605.12671) | arXiv abs, comments say ICML 2026 |
| `mcgrath2023hydra` | The Hydra Effect (2307.15771) | arXiv abs (no venue listed) |
| `mcdougall2024copy` | Copy Suppression (2310.04625) | ACL Anthology 2024.blackboxnlp-1.22 (venue title differs from arXiv: "...a Motif in Language Model Attention Heads") |
| `rushing2024explorations` | Explorations of Self-Repair (2402.15390) | arXiv journal-ref: ICML 2024, PMLR 235:42836-42855 |
| `gong2026conditional` | Conditional Co-Ablation, CoAx (2607.01940) | arXiv abs (v3) |
| `chen2026pie` | PIE: Prune, Interpret, Evaluate (2604.16889) | arXiv abs |
| `kendiukhov2026causal` | Causal circuit tracing in single-cell FMs (2603.01752) | arXiv abs |
| `kendiukhov2026exhaustive` | Exhaustive circuit mapping of a single-cell FM (2603.11940) | arXiv abs |
| `turner2026interference` | Characterizing interference weights in a tiny LM | transformer-circuits.pub page (Turner, Wu, Batson; 21 Aug 2026) |
| `engels2025dark` | Decomposing the Dark Matter of SAEs (2410.14670) | arXiv (TMLR in comments) + mlanthology TMLR 2025 (second author listed as "Logan Riggs Smith"; arXiv shows "Logan Riggs") |
| `cui2026sae` | SAE Interventions are Unreliable (2606.18322) | arXiv abs |
| `cho2026single` | Are Single-Token SAE Features Causally Necessary? (2607.20596) | arXiv abs |
| `poduval2026resae` | ReSAE (2605.27819) | arXiv abs |
| `paulo2025sparse` | SAEs Trained on the Same Data Learn Different Features (2501.16615) | arXiv abs |

All other DAN-31 items were already in the bib (Caples, Lepori, He, Ge, Dunefsky, CircuitLens, Palit, Droste, Gao,
Stoehr, Li & Janson, Shi, Miller, MIB, EAP-IG, Heimersheim & Nanda, Makou, CircuitLasso = `yin2026scalable`).

## Existing entries fixed

- `chan2022causal`: the author list matches the post's byline (LawrenceC, Adria Garriga-alonso, Nicholas Goldowsky-Dill,
  ryan_greenblatt, jenny, Ansh Radhakrishnan, Buck, Nate Thomas; Redwood Research, 3 Dec 2022). I added the URL
  (alignmentforum.org/posts/JvZhhzycHu2Yd57RN) and removed the "to verify" comment.
- `shi2024hypothesis`: venue changed from arXiv to NeurIPS 2024 (nips.cc poster 96582, mlanthology). Authors re-checked
  against arXiv.

## Text added

Main text, Related Work:
- Circuit discovery: ACDC and Edge Pruning named as pruning against an output metric; AtP / AtP* / EAP-IG / RelP as
  first-order scoring; "our IG baseline adapts EAP-IG's estimator to nodes"; attribution graphs over transcoder
  features (now also citing Dunefsky) described as "a substrate we do not run". To save space, "at production scale"
  and "for the reconstruction residual" were cut.
- Masks and scalars: the second sentence is compressed (same citations), and its `\cref{app:related}` is dropped
  because the closing paragraph already points there.
- Evaluating circuits: optimal ablation as the dual of our coefficients, with our ablations fixed by data; Shi and
  CircuitKIT as size-matched random circuits that go unfitted, like the circuits they test, while ours get the same
  fitting; causal scrubbing as the stricter test (`app:resample`).

Appendix, `app:related`:
- New paragraph "Discovery methods and their settings": ACDC, Edge Pruning, AtP, AtP*, EAP-IG (edge-level, integrates
  along corrupted-to-clean input embeddings), how our path-attribution baseline differs (node-level, joint over SAE
  sites from their mean-ablation values), transcoders, attribution graphs, and PIE; why they are not run here.
- "Masks, scalars, and ablations": one sentence on how optimal ablation chooses its constant (the one that minimises
  expected loss), set against our fixed, data-defined ablations.
- New paragraph "Nulls, variance, and non-uniqueness": Shi, CircuitKIT, why our null fits coefficients, Dead Salmons,
  causal scrubbing, Wu et al., Explanation Multiplicity, Certified Circuits, Meloux (identifiability), All Circuits
  Lead to Rome, Makou, with a pointer to `app:underdetermination`.
- New paragraph "Self-repair and inhibition": hydra effect, self-repair, copy suppression, CoAx, and the point that our
  scores intervene on the circuit as one set.
- "The SAE basis": interference weights (Turner et al.), Kendiukhov x2, dark matter, Paulo & Belrose, SAE
  interventions unreliable, single-token features, and ReSAE.

## Caveats / not done

- **Reading depth.** I read the 2026 papers at abstract-and-method level (arXiv abstract pages, plus HTML for
  CircuitKIT, Shi, EAP-IG, Meloux, and optimal ablation), not in full. For Turner et al., only the introduction was
  read, so it is cited only for its definition of interference weights.
- **EAP-IG wording.** Daniel's brief says EAP-IG "is our IG baseline's estimator". EAP-IG interpolates the *input
  embeddings* and scores *edges*, while our `ig_mean` interpolates internal SAE sites from their mean-ablation values,
  which is closer to the Marks et al. IE_ig node estimator. The text therefore says "adapts", not "is".
- **Venues not found.** Hydra effect, Dead Salmons, CircuitKIT, PIE, Kendiukhov, and the other 2026 preprints list no
  venue, so they are cited as arXiv preprints.
- **Not in my regions.** The DAN-31 "Done when" item about updating the 2x2 "empty cell" framing in the introduction
  is outside the files and regions I own, so it is still open.
- **Page budget.** The body ends on page 13. A scratch build with the original Related Work text restored (everything
  else current) also ends on page 13, so the overflow comes from concurrent edits elsewhere. My main-text change
  adds about 230 rendered characters (about 2 lines).
- **Build references.** `app:nulls` is undefined (referenced at about L341 and L825, outside my regions). There are no
  undefined citations.
