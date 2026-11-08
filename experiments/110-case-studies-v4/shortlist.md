# Part B: case-study candidates on physics, AI and computing (production circuits, 2026-11-08)

## How they were found

The search was token-driven, with no auto-interp labels.

1. **Topic scan.** `topic_scan.py` (CPU) read all 7,786 production targets. For each it decoded the 64 strongest and
   64 contrast fresh-record contexts and counted physics, AI and computing terms. It also recorded whether the anchor
   token is itself a domain word, and the peak-token consistency.
2. **Filter.** A target was kept if it passes, is at layer 5 or deeper, has at least 70–75% of its strongest contexts
   on-domain, and either has a non-domain anchor or a domain anchor with low consistency. Size had to be 150–2,000.
   About 75 targets survived, and their anchor windows were read by hand (`windows.py`).
3. **Probes.** For 11 candidates, `probes.py` ran hand-written sentences through the clean model:
   - in-domain sentences;
   - near-domain sentences without the concept;
   - the **same word in another sense** (the mandatory string-detector control).

   Values are the target's maximum over the sentence as a percentage of its held-out reference.
4. **Members.** `report.py` produced the top 25 members (60 for the worked cases) with their own top contexts.
5. **Specificity.** The production spec rows gave, under activating-mean (A) and contrast-mean (C) ablation, the
   latents lifted above Top-K and the sibling / control faithfulness medians. The run medians of latents lifted are
   497 (A) and 488 (C).

The relativity seeds of experiment 043 (9.resid.3736, 3.resid.18699, 6.resid.23753, 8.resid.455, 11.resid.13633) are
**not in the 8k list**, nor in the earlier 15k. They have no production circuit.

## Ranked shortlist

The table columns:
- **pass**: held-out strongest Z / A / C; necessity is 1.00 for all of them.
- **mid**: the mid-band verdict.
- **A-lift / C-lift**: latents lifted above Top-K (sibling median, control median).

| # | target | topic | n | pass Z/A/C | mid | A-lift (sib/ctl) | C-lift (sib/ctl) | probes (in-domain vs controls) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **8.resid.31646** | AI as an entity: fires on "intelligence" in "artificial intelligence" and on "I" in "AI" | 1046 | 0.87/0.96/0.93 | fail (A 2.11) | **281 (0.15/0.01)** | **197 (0.12/–)** | AI sentences 87–101%; operating system / database 0%; military intelligence 0%, emotional intelligence 0%, pronoun "I" 0%, dolphin intelligence 18% | the cleanest latent (string control perfect, spec rows the most specific), BUT the worked-up knock-outs do not validate a readable core (below) |
| 2 | **9.resid.13945** | magnetic-confinement fusion (tokamak / stellarator), at "uses **a** combination of toroidal…" | 1322 | 0.92/1.11/0.99 | vacuous | 12,080 (2.40/2.98) | 7,151 (1.44/1.80) | tokamak / stellarator 98–103%; MRI / motor / transformer 0–10%; same frame (bakery, architect) 0–6%; Sun's fusion 13%, inertial confinement 68% | **recommended (physics), with an amplifier caveat**: readable physics concept graph and decisive knock-outs (below) |
| 3 | 10.resid.15497 | the possessive after a name that carries a law (Ohm's, Snell's, Gauss') | 1086 | 0.91/1.02/0.95 | fail (A 1.95) | 1,773 (1.76/0.42) | 1,023 (0.95/0.30) | laws 50–98%; Darwin's, Rutherford's, Turing's 0%; Picasso's, Mozart's, Mary's 0%; invented "Zorbel's" 79% in electromagnetism vs 0% in painting | strong physics alternative, already in the appendix (old circuit validated). The members are possessive-after-name detectors plus electromagnetism (Maxwell's equations, magnetic fields, Ampère) |
| 4 | 10.resid.18743 | software code review / code quality | 1066 | 0.94/0.94/0.92 | vacuous | 8,389 (0.53/1.95) | 6,210 (0.54/1.25) | code-review sentences 80–103%; unit testing 38%, version control 19%; peer review of a paper 15%, editors 14% | best computing concept by the probes. The members read as software engineering (clean / maintainable code, compiled languages, static typing, modular OOP code, greedy algorithms), but it is a strong amplifier |
| 5 | 6.attn.9288 | overfitting / underfitting in ML | 539 | 0.96/0.87/1.01 | **pass** | 500 (–/0.13) | 314 (0.10/–) | overfitting sentences 82–93%; gradient descent / neural network 0%; tailor "poor fit" 0%; "fitting kitchen cabinets" 27% | the only candidate passing BOTH bands, and specific. But it is an attention target with a tiny reference (0.88), induce 0.30, and its top members are the generic core (regularisation and "training data" latents rank ~12–15) |
| 6 | 9.resid.28803 | CPU ↔ main-memory (memory hierarchy, cache) | 1380 | 0.93/1.04/1.05 | fail (A 2.18) | 1,421 (0.78/0.67) | 436 (0.78/0.23) | CPU sentences 99–112%; GPU 25%, hard drive 34%; human memory 0% | partly a "CPU" word detector. The members are CPU-token latents, cache and processor; weaker as a concept |
| 7 | 11.attn.16140 | superconductivity | 719 | 0.84/0.79/0.93 | vacuous | 5,516 | 5,299 | superconductors 47–97%; copper / semiconductors 0%; "supermarket", "super student" 0% | specific probes, but attention, reference 3.5, near the A floor, amplifier; failed the old run |
| 8 | 9.attn.29578 | expert systems (AI) | 789 | 0.98/0.98/0.97 | pass | 2,111 | 2,411 | expert-system sentences 10–76% (inconsistent); other AI 0%; "expert witnesses" 29% | weak; induce 0.00 |

**Rejected after probes** (fail the string or near-domain control):
- **7.resid.4780, "gravity".** A string detector: "the gravity of the situation" reaches 94%, against 118–126% for
  physics gravity.
- **9.mlp.37206, quantum key distribution.** Diffie-Hellman reaches 93% and a burglar alarm 48%; it is a "detected /
  eavesdropping" frame, not quantum.
- **8.mlp.2861, isotopes.** It fires on "protons" in any atomic-structure sentence (54–85%) and once misses an isotope
  sentence (0%).
- **9.resid.35746 "artificial", 10.resid.11335 "CPU", 9.resid.1423 "operating", 6.resid.9403 "angular".** 100%
  single-token anchors. Not probed; read as string detectors from their anchors.
- **Narrow topic latents seen in the screen but not probed:**
  - 8.attn.36027, Reynolds number;
  - 10.mlp.30456, renormalisation; it also fires on lubricants and proteins;
  - 5.resid.10280, dynamic systems;
  - 9.mlp.5397, CNN/RNN "NN";
  - 5.mlp.29255, "Neural Network(s)";
  - 7.resid.37823, datasets.

## Worked up

All three runs used `case_v4.py` with the groups in `groups_<key>.json`: knock-outs against 8 rank-matched random
draws, group edges, and probes in the clean model and with the circuit alone. The full circuit re-scored locally
reads:
- 9.resid.13945: 0.91 / 1.09 / 0.96;
- 8.resid.31646: 0.89 / 0.99 / 0.95;
- 10.resid.15497: 0.91 / 1.02 / 0.95.

### 9.resid.13945, magnetic-confinement fusion (physics): VALIDATED readable core, amplifier

Figure: `paper/figures/run2026_case_fusion.pdf`.

Members are named from their own top contexts.

| group | n | layers | share | knock-out Z / A / C | random |
|---|---|---|---|---|---|
| plasma confinement (tokamak / stellarator / magnetic confinement for fusion) | 10 | L5–9 | 9.1% | **0.17 / −0.01 / 0.12** | 0.79 / 0.99 / 0.89 |
| nuclear and power engineering (reactor types and coolant, nuclear reactions, steam turbine, Otto / Carnot cycles, power grid) | 6 | L7–8 | 5.1% | **0.69 / 0.71 / 0.71** | 0.82 / 1.02 / 0.90 |
| stars, cosmos, particles (stellar cores and hydrogen burning, dark matter, unification / top quark) | 3 | L7–8 | 4.0% | 0.85 / 0.98 / 0.90 | 0.88 / 1.06 / 0.94 |
| quantum and magnetism (wave function, NMR nuclear spin, qubit, permanent magnetism, AC coils) | 5 | L6–9 | 3.5% | 0.88 / 1.08 / 0.95 | 0.83 / 1.02 / 0.91 |
| off-topic (control: NoSQL, intercultural communication, traffic, oceanography, composites, CI/CD, Elam) | 7 | L7–8 | 6.3% | 0.86 / 1.07 / 0.93 | 0.84 / 1.02 / 0.92 |
| generic core (control) | 7 | L0–5 | 3.3% | 0.00 / 0.73 / 0.74 | 0.83 / 1.05 / 0.92 |
| all four physics groups | 24 | | 21.7% | **0.07 / −0.05 / 0.05** | 0.73 / 0.93 / 0.85 |

**Edges (A fill).**
- Plasma confinement → nuclear / power 0.28, and → stars 0.13.
- Direct drops in the target: plasma confinement 0.59, nuclear / power 0.21, others ≤ 0.06.

**Readings.**
- The target's computation runs through a 10-latent plasma-confinement chain, supported by reactor and heat-engine
  latents.
- The astrophysics and quantum latents are present but no more necessary than random members.
- The off-topic members are confirmed as passengers.

**Probes.**

| sentence | clean model | circuit alone (C) |
|---|---|---|
| tokamak / stellarator | 69–103% | 104–115% |
| inertial confinement fusion | 68% | 94% |
| the Sun's fusion | 13% | 66% |
| fission reactor | 22% | 45% |
| motor / MRI / transformer | 0–10% | 22–53% |
| bakery / architect ("uses a combination of", "employs a complex arrangement of") | 0–6% | 49–56% |

The empty circuit reads 0 under C.

The circuit alone keeps the ordering but not the clean model's sharpness. That matches the spec rows: 12,080 latents
lifted under A and 7,151 under C, with control faithfulness 2.98 / 1.80. **It is an amplifier.**

### 8.resid.31646, artificial intelligence (AI): the latent is clean, the readable story is NOT validated

Figure: `paper/figures/run2026_case_ai.pdf`.

| group | n | share | knock-out Z / A / C | random |
|---|---|---|---|---|
| AI word forms ("A-I", "artificial intelligence"; L3–7 relay) | 12 | 7.0% | 0.86 / 0.95 / 0.90 | 0.53 / 0.67 / 0.65 |
| machine learning and agents | 6 | 4.0% | 0.69 / 0.81 / 0.71 | 0.73 / 0.84 / 0.80 |
| technology applications | 6 | 4.6% | 0.74 / 0.79 / 0.76 | 0.68 / 0.76 / 0.72 |
| all story groups | 24 | 15.6% | 0.57 / 0.67 / 0.60 | 0.36 / 0.54 / 0.52 |

- **Knock-outs.** The named groups are NO more necessary than random members of similar rank. Removing the word-form
  relay hurts LESS than random.
- **Edges.** The machine-learning group feeds the word-form group (edge 0.74), but the word-form group's direct drop
  in the target is only 0.03.
- **Probes.** The circuit alone does not keep the string control: emotional intelligence 92%, "artificial flowers"
  89%, operating system 79% (clean model 0% on all three).

The target itself is an excellent AI-concept latent (clean model: AI 68–101%, every other sense 0–18%), and its
spec-row specificity is the best in the list. But its circuit has no readable, necessary core under our groups.
Three-quarters of the contribution sits in 1,008 unnamed members.

**Not recommended as a case study** until a regrouping finds a necessary core. This is a useful negative example of
"specific ≠ readable".

### 10.resid.15497, eponymous laws (physics): VALIDATED on the production circuit, weaker than on the old one

Figure: `paper/figures/run2026_case_laws.pdf`.

The groups are 063's, and all are present.

| group | knock-out Z / A / C | random | old (062 full run) |
|---|---|---|---|
| name → law (3) | 0.60 / 0.69 / 0.67 | 0.80 / 0.92 / 0.83 | |
| physics context (2) | 0.77 / 0.86 / 0.82 | 0.82 / 0.94 / 0.87 | |
| generic possessive (control, 4) | 0.90 / 1.02 / 0.95 | 0.77 / 0.89 / 0.83 | |
| story groups (5) | **0.46 / 0.55 / 0.54** | 0.72 / 0.85 / 0.76 | 0.00 / −0.12 / 0.03 |

- **Probes.**

  | probe | clean model |
  |---|---|
  | named laws | 50–98% |
  | non-law possessives | 0% |
  | "Zorbel's" in electromagnetism | 79% |
  | "Zorbel's" in painting | 0% |

  The circuit alone gives laws 97–125% against non-law possessives 41–63%.
- **Spec rows.** 1,773 lifted under A, with siblings 1.76 and controls 0.42; under C, controls 0.30.

## Recommendation

1. **Physics: 9.resid.13945 (magnetic-confinement fusion)** is the best new case.
   - The minimal pairs in the clean model are among the cleanest we have: tokamak 103% vs MRI 0% vs bakery 6%.
   - Its members read as a physics concept graph.
   - Its knock-outs are the most decisive of any case so far: removing 10 latents (0.8% of the circuit) takes Z / A / C
     to 0.17 / −0.01 / 0.12 against random 0.79 / 0.99 / 0.89, with controls at random level.
   - It must carry the amplifier caveat, as 9.resid.7640 did in the old appendix.
2. **Second physics option: 10.resid.15497 (eponymous laws).**
   - It is already in the paper, and the production circuit still validates it (0.46 / 0.55 / 0.54 vs 0.72 / 0.85 /
     0.76).
   - It has the best physics minimal pair ("Zorbel's").
   - It is less of an amplifier.
   - The appendix row's numbers must be updated if the circuit is switched to the production one.
3. **AI / computing.** No candidate is ready.
   - **8.resid.31646** has the cleanest latent, but its circuit has no validated readable core.
   - **10.resid.18743** (code review) and **6.attn.9288** (overfitting; the only candidate passing both bands) are the
     next to work up, at about 10 GPU-minutes each with `case_v4.py` and a groups file.

**Caveat on "circuit alone (C)" probes.** The fill is the target's own contrast means. On sentences far from the
target's domain, the rest of the stream is therefore replaced by in-domain background, so every circuit reads 40–60%
there. Compare in-domain against near-domain, not against 0.
