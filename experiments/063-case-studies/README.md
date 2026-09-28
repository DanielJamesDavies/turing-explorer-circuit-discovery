# 063: case studies from the protocol-v1 stage-1 circuits

**Goal:** find readable, specific circuits for the paper's case studies (the running example was deferred until after
the production run; paper-updates live checklist).

**Tool:** `inspect_circuit.py` writes one markdown report per circuit, to `results/reports/<target>.md`.
- Target: DAN-8 scores and flags, peak tokens and consistency, marked windows, a contrast snippet, logit effect.
- Top 12 members by contribution at the target's anchor (α × activation at the anchor × decoder norm), each with:
  engagement on the target's contexts vs contrast, own peak tokens (LEXICAL if ≥ 80% on one token), windows, logits.
- Token-driven reads only, no auto-interp.
- Usage: `KEYS=a,b` or `SELECT=1`, about 3 s per circuit on the 5070 Ti.

```
KEYS=9.resid.7640 PYTHONPATH=src python experiments/063-case-studies/inspect_circuit.py
```

## Screen (2026-09-27, stage-1 download `062-.../out_stage1`)

385 candidates: passing the DAN-8 rule, not amplifier-flagged, not near-threshold. Seven read-only subagents screened
them in depth/kind batches and read ~120 reports in full. The top picks were checked by hand against the reports.

### Shortlist

| Role | Target | Story |
|---|---|---|
| deep, functional | **9.resid.7640** | the pivot in "not X, but rather Y" ("isn't about socializing [[,]] but rather"). Generic comma members fire equally on contrast (20.3 vs 19.3); negation-scope members are specific: 8.resid.12223 "more than just" (36.0 vs 9.6, logits but/nor/anymore), 8.resid.29816 "not just about X[,]" (17.1 vs 0.3), 4.resid.26922 "does not [contain]" (logits nor/instead). The contrast has "but" without a negation. |
| function-word gate | **7.resid.12579** | "the" only after "beyond" (logits borders/limits). Generic "the" members fire on contrast too (5.resid.29335: 20.5 vs 18.7); "beyond the" members are silent on it (6.resid.3715 15.1 vs 0.1, 4.resid.1883 8.2 vs 0.15, 5.resid.10642 9.3 vs 0.13, 6.resid.38790 → boundaries/limits). Same pattern: 5.resid.18028 "draws [[the]]", 1.mlp.34772 "response [[to]]", 7.resid.30866 "process [[of]]". |
| conceptual composition | **3.resid.18143 → 4.resid.37051** | philosophy of mind: "[[physical]]ism / [[functional]]ism", then "mental [[states]]". A shared topic feature (2.resid.20535, fires on contrast) combines with specific word detectors (0.mlp.25397 mental, 2.resid.35469 "mental [states]", 2.resid.40481 physical). The contrast is on-topic, so only the word detectors switch off. Alternative: 2.mlp.5566 kitchen knives (word sense: its 'kn' member 1.resid.7834 peaks on "knots"). |
| easy first example | **0.mlp.824** | the token after "frame": 28 members, layer-0 attention latents carry "frame" into an MLP phrase detector. Not 0.resid.36589 "fract[[ure]]": its biggest member is its own MLP twin (below). |

**Mentions:**
- 7.mlp.37588: a preposition cited as a word in grammar text ("prepositions like " [[from]]"").
- 11.resid.21315: the "which/that" after an equation is named.
- The list-continuation family: 8.resid.7581 emotions, 11.resid.33267 science/technology, 8.resid.24673 religious.
- 5.attn.3271: environmental sustainability, from sub-concept features.
- 2.resid.29625: the period after a middle initial.
- 1.resid.17403: flashback.

### Caveats (must be handled before any figure)

1. **Dense circuits.** The top 12 members carry about 15–40% of the contribution at depth (88% only for tiny layer-0
   circuits). A figure shows the readable core. The causal check to add: knock out just the named story members and
   show the target drops on its contexts, while knocking out the same number of random members does not.
2. **resid/mlp twins.** At layers 0–1, `L.resid.N` and `L.mlp.N` often behave identically (e.g. 36589, 17458, 5338,
   13370, 14621, 35546), though not always (0.mlp.37130 "spir"). Check how the SAEs were initialised or trained before
   counting members or claiming mixed-site composition.
3. **Shared generic core.** About ten members appear near the top of almost every circuit and fire equally on target
   and contrast: 0.mlp.1349, 0.resid.6299, 0.attn.5663, 0.attn.712, 1.attn.13899, 1.attn.34495, 0.mlp.26380,
   1.mlp.621, 1.attn.16130, 1.resid.39697. They act like a constant term. Name them once and set them aside; the
   LEXICAL tag is misleading for dense latents like 0.attn.5663.
4. **Positions.** "Fires at anchor" is measured at the target's anchor. Previous-word detectors firing there means that
   information has already been moved forward. A position-resolved view is needed for attention stories.
5. Necessity > 1 occurs (1.resid.22765 2.19, 0.mlp.5196 2.24). The ratio allows it when removing the circuit pushes
   the target below the empty-circuit value; explain it if quoted.

### Patterns

- About 40% of targets are string/word-piece detectors whose circuits relay the same string; they are not usable.
- The dominant readable mechanism is generic token features plus specific context features, where the specific
  features explain why the contrast is silent.
- In the good cases, members' logit effects agree with the target's, which is a cheap coherence check.
- Induce differs by kind: attention targets often overshoot (~2), MLP targets sit at 0.1–0.7.

## Deep concept-composition search (2026-09-27)

Daniel asked for a deep target that pieces concepts together into a new concept.
- Pool: `SELECT=deep`, all 204 passing, non-near-threshold stage-1 circuits at layers 6–11, amplifier-flagged
  INCLUDED. Top 25 members, in `results_deep/`.
- Pre-filter: target peak-token consistency ≤ 70% gives 46 circuits.
- Three subagents read all 46 in full. Each candidate is labelled composition vs union of synonyms vs relay.

**Report bug found and fixed:** headers printed "not amplifier" for every circuit, because the check was
`row.amp_any is True` and a numpy bool is never `True`. Selection was unaffected (vectorised). The deep reports are
regenerated with the fix.

| Rank | Target | Story | Caveats |
|---|---|---|---|
| **1** | **9.mlp.10663** (amplifier) | "a new offspring from two parents' genetic material", in biology ("formation of [[a]] zygote") and genetic algorithms ("two parents to generate [[new]] offspring"). Ingredients: meiosis 8.resid.38032 (14.7 vs 1.1, → daughter/gam/sexual); one chromosome from each pair 7.resid.102 (7.8 vs 0.9); mate choice 8.resid.37715 (9.5 vs 1.9, → mate/sex/parent); passes from each parent 8.resid.7047 (10.5 vs 2.9, → parent); gene transfer 8.mlp.5777 (7.9 vs 0.8); fertility 8.mlp.12873 (→ off-spring); genes/DNA 8.resid.29676 (20.3 vs 9.8). The genetic-drift contrast keeps gene/evolution members partly on and switches the reproduction ingredients off. | amplifier-flagged; top member 4%, 57% outside the top 25; induce 0.71; predator/species members ~1.6× |
| 2 | **8.attn.29991** | acid–base dissociation ("donates a [[pro]]ton", "the [[p]]H scale"). Ingredients: dissociation 5.resid.38102 / 6.resid.29630; bases and hydroxide 4.resid.24519, 7.resid.10184, 6.resid.34419 (Brønsted); pH 7.resid.35144 (11.2 vs 1.9, → buffer/acid). On the general-chemistry contrast, the generic chemistry member 7.resid.9342 stays on (15.4 vs 14.2) while the acid–base ingredients switch off. | biggest member is the generic one; ingredients 3–4×; induce 0.75 |
| 3 | 9.resid.16239 (amplifier) | cooking used as a metaphor. "is like…" frame members stay on for an iceberg-metaphor contrast; cooking members are silent there. The cleanest AND. | rhetorical, not knowledge; part relay |

Others: 9.mlp.24426 public/private key (word × crypto domain, amplifier; check sense on "private sector" text);
9.resid.10889 climate → harm causation (amplifier, function-word target); 7.resid.34018 extensions of classical logic
(partly a union); 8.attn.21835 printing press → dissemination (relay-heavy).

## Chosen and validated (2026-09-27): `validate.py` → `results_validate/<key>.md`

Daniel did not want the reproduction topic (9.mlp.10663) as the headline. He picked **8.attn.29991** (acid–base) and
**9.resid.7640** (the "not X, but rather Y" pivot).

**8.attn.29991, acid–base dissociation: HEADLINE.**
- Probes (clean model, % of its mean anchor activation 4.43):
  - acid–base sentences 46–87% (peaks on "partially" (dissociates) and on the "p" of pH);
  - other chemistry (redox, ionic bonds, alkanes, balancing, gas law) 0–16%;
  - the same words outside chemistry 0% ("base its strategy", "professional", "dissolution"), except "acid wit" at 38%.
- Knock-outs (run scorer; random = same n, contribution-rank matched ±15, 8 draws):
  - all 10 story ingredients (2% of 492 members) take Z/A/C from 0.96/1.06/0.81 to 0.25/0.11/0.16; random gives
    0.61/1.19/0.80.
  - The pH latent 7.resid.35144 alone takes A to 0.38. Bases/hydroxide (4) give Z 0.47, C 0.33. Dissociation (2) gives
    Z 0.63, C 0.53.
  - The equilibrium and lexical "acid" members are no better than random, so drop them from the story.
  - The generic chemistry group (the largest shares) is no better than random either, so the specific ingredients
    carry the concept, not the topic.
- Specificity: siblings 0.57–0.65 vs target 0.82–1.04; lifted latents 525 vs 371 for the empty circuit. Clean.

**9.resid.7640, the negation pivot: second, functional example, with caveats.**
- Probes (% of 27.9, at the comma or semicolon): negated clause then pivot 37–124%; the same comma without the
  negation 0–13% ("Leadership IS about…, but also…" 13% vs 124% negated); negation without a contrastive pivot 0–22%.
- Knock-outs: the 6 negation-scope members take Z/A/C from 0.91/1.37/0.99 to 0.00/0.03/0.00. The generic commas
  have no effect.
- Caveat 1: the circuit is FRAGILE. Rank-matched random removals also cost a lot (Z 0.29 ± 0.22, A 0.73 ± 0.23,
  C 0.44 ± 0.19).
- Caveat 2: it perturbs the whole site. Under A/C it lifts ~15–19k latents above the Top-K cut vs ~1 for the empty
  circuit, and control latents are over-driven 2–5×. The target itself is selective (probes), but the circuit is not
  a clean single-target mechanism.

## Full-run deep concept search (2026-09-27, DAN-132): `results_full_deep/`

- Pool: `SELECT=deepconcept` on the full run. 2,238 deep (L6–11) passing circuits; 595 remain after the target
  peak-token consistency ≤ 70% filter; 25 members each.
- Cheap member pre-filter: ≥ 5 specific non-lexical members. A member counts if its seq max is ≥ 3× contrast and
  > 2, and its own consistency is < 80%. This leaves 284 candidates (126 amplifier-flagged). The known cases were
  excluded.
- 8 subagents read all 284. The large majority are UNION (near-synonym sets) or RELAY (a word or word piece carried
  forward). About 20 were labelled COMPOSITION, none of them strong.

| Rank | Target | Story | Caveats |
|---|---|---|---|
| **1** | **10.resid.34899** (clean, not amp; Z/A/C 0.92/1.08/1.05, worst dev 0.08, 681 members) | Animation. Logits → animation/Disney/CG. Ingredients that switch off on the contrast: CG/3D rendering 8.resid.28274 (16.2 vs 2.0), GPU/real-time rendering 9.resid.9633 (17.0 vs 3.9), rendering algorithms 8.mlp.13614 (7.1 vs 1.0), theatre performance 9.resid.29339 (12.7 vs 1.0), motion/velocity 8.attn.7416 (9.3 vs 3.3), dynamism 9.resid.1275 (8.4 vs 2.9). The biggest members, narrative 8.resid.26816 / 9.resid.9522 and painting 8.resid.38219 / 9.resid.4310, stay ON for the perspective-drawing contrast: art + story is shared, and motion + CG + performance is what makes it animation. | target peaks on function words ("involves", "create"); the concept is read from its logits |
| 2 | 9.resid.12586 (not amp; 1.02/1.18/0.87) | Improving patient care: medical technology 8.resid.429 (29.1 vs 5.9), new therapies 7.resid.1956, healthcare access/finance 8.resid.26700 (9.6 vs 0.1), diagnosis/monitoring 8.mlp.16059, biomedical engineering 8.resid.28224. The "improve X" member stays partly on for the computer-vision-improvement contrast. | some members are lexical "patient"/"health care" |
| 3 | 10.mlp.32418 (not amp; 1.04/1.31/0.99) | Recursion / dynamic programming: base case 9.resid.18573 (23.9 vs 1.8), self-call 9.mlp.2120 (12.5 vs 0.7), induction 8.resid.2137, proof by contradiction 9.resid.39380. | top shares are generic algebra/software members that fire on the contrast too; the discriminative ones mostly restate recursion (relay-ish) |
| 4 | 10.resid.32101 (not amp, clean rank 1; 0.99/1.28/0.94) | Phonetics / nasal articulation: sound production 9.resid.38162 (28.0 vs 0.4), velum lowering 8.resid.10366, vowel quality 8.mlp.2233, syllable structure 9.resid.28548. | contrast lacks all of linguistics, so it does not isolate one ingredient |
| 5 | 10.resid.11617 (amp; 0.84/1.20/0.95) | Teleological argument: problem of evil, determinism/causation, free will, divine being, nature. The God members stay partly on for the Divine-Simplicity contrast; the causation/design members switch off. | amplifier; Z 0.84 |

Others (amplifier, 5–6/10): 11.mlp.39065 cat taxonomy (Panthera + classification + predation; its contrast is a
language family, which keeps "classification"); 10.resid.32152 chiaroscuro; 11.mlp.19493 speech recognition;
11.resid.7771 supply-side economics; 11.resid.26282 presupposition (not amp).

Verdict: nothing in the full run beats 8.attn.29991 on validation evidence. **10.resid.34899 (animation) is the one
full-run candidate worth validating.** It is clean (not an amplifier, worst dev 0.08), and its members split into
shared-with-contrast (art, story) and discriminative (motion, CG, performance) groups.

## Next

- Validate 10.resid.34899 with `validate.py` (story knockouts: CG/rendering, performance, motion; shared: narrative,
  painting) if it is wanted as a third case or a swap.
- Position-resolved member activations; 057 wiring on the chosen circuits.
