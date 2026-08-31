# Concept circuits: can a validated circuit be read as knowledge?

**Question (Daniel, 2026-08-26):** if a latent stands for a concept, and
our circuits are causally validated, then a circuit is a causal graph
over CONCEPTS -- "these concepts compose into that one". Can we read
knowledge out of a model that way? Test case: a WW2-like latent should
decompose into war / period / place concepts.

## Pipeline

1. `find_factual.py` / `find_concept.py` -- Neuronpedia explanation
   search for knowledge-bearing GemmaScope transcoder features
   (`OUT_FILE` names the candidate file; layers 6-18).
2. `screen_factual.py` -- run candidates over the SAME 20k wikitext
   windows and the SAME firing band (0.005-0.05) as the paper's Gemma
   arena, so a concept seed is not a different kind of object.
   `CANDS`/`OUT` select the pool. Writes pos/neg windows per seed.
3. `show_contexts.py` -- **THE GATE THAT MATTERS**: print the actual
   top-firing tokens in context. Auto-interp labels are NOT trusted.
4. `build_run_set.py` -- merge chosen seeds into `run_seeds.pt`
   (asserts pools share token tensors before merging).
5. Circuits: the unmodified arena harness via `SCAN_FILE` + `ROWS_TAG`
   (`run_circuits.sh`) -> `ours_gtc_fact_{rows,members}.jsonl`.
6. `fetch_member_labels.py` + `concept_graph.py` -- label members and
   render circuits as concept graphs, GATED on zero-fill AND mean-fill
   in [0.8, 1.25] plus necessity above the run's nulls.

## Result 1: labels are unreliable; activation screening is mandatory

Of 8 hand-picked candidates, **3 were fatally mislabelled** and 2 more
were narrower than advertised:

| latent | Neuronpedia label | what it ACTUALLY fires on | verdict |
|---|---|---|---|
| L8/2415 | "World War II" | 79% of ALL windows | dead |
| L12/13697 | "Heisenberg" | Sevket, "his", Internazionale, Kelley, Miami | dead |
| L12/404 | "dates ... political or natural events" | copulas in hurricane-season text | dead |
| L12/12770 | "historical political and military conflicts" | "break up", "rescue", "arrival" | mislabelled |
| L12/15052 | "WWII aircraft" | aircraft type names incl. civil 737s | narrower |

Had we trusted labels, the Heisenberg circuit would have been published
as a physics-association claim built on a feature that fires on football
clubs. **Any latents-as-concepts claim needs activation verification as
a gate.** (Neuronpedia's labels may reflect max-activating examples from
a different corpus; on wikitext they do not hold.)

## Result 2: validated circuits ARE compositionally readable

5 verified seeds run; 3 pass the two-criteria gate. Nulls dead (<=0.03).

| seed | concept | n | zero-fill | mean-fill | necessity | best null |
|---|---|---|---|---|---|---|
| L12/2064 | World War | 17 | **1.000** | 0.966 | 0.364 | 0.002 |
| L12/8460 | war discourse | 54 | 1.202 | 0.916 | 0.268 | 0.027 |
| L12/15052 | aircraft types | 60 | 0.945 | 0.965 | 0.549 | 0.005 |
| L12/2889 | military branch | 166 | 0.876 | *0.792* | 0.785 | 0.009 |
| L14/15757 | combatant forces | 197 | 0.362 | 0.466 | 0.262 | 0.012 |

L12/2889 misses mean-fill by 0.008 and is reported as NOT VALIDATED --
the gate is not nudged.

**L12/2064 "World War" (17 nodes, zero-fill 1.000)** decomposes as
Daniel predicted: wars incl. WWI/WWII (a 3.08), military rank + calendar
months (2.61), WWII/Nazi Germany (2.54), **the word "world"** (2.27),
war esp. WWI (2.02), military conscription and service (1.70), military
history esp. WWII (1.56) -- 7 of 17 directly war/military, composed
across layers 2->11.
A SECOND cluster is unexplained by the label and carries the LARGEST
weights: economic downturns and financial hardship (3.99), societal ills
(3.52), the COVID-19 pandemic and disruptions (3.35), politics/finance/
economics (2.16) -- a "large-scale societal disruption" direction.
Remaining ~5 nodes resist interpretation (anime terminology, code
snippets, suffix endings).

**L12/8460 "war discourse" (54 nodes)** is the most coherent: 10 of its
top 14 are conflict concepts -- wars/battles/military actions (3.03),
war and violence (2.73), war involving the US/Iraq (2.73), conflict and
competition (2.53), fighting or revenge (2.50), places with political or
religious conflict (2.33), politics/war/revolution (2.27), Israel and
Palestine (2.12).

**L12/15052 "aircraft types" (60 nodes)** shows cross-domain
composition: WWII planes from Japan or the USA (2.72), WWII Japanese
fighter planes (2.52), WWII airplanes and fighter pilots (2.31),
aircraft and flight (2.30) -- BUT ALSO automobile brands and models
(2.81), transport/vehicles/driving (2.48), and its single largest
weight on "sentences describing a car chase and crash" (3.60). The
aircraft detector is built partly from a general VEHICLE direction.

## Caveats that travel with every claim here

* Auto-interp labels can be wrong or vague (Result 1 quantifies it).
* A latent need not be exactly one concept (polysemanticity).
* A circuit is a SUFFICIENT set, not the model's unique route to the
  seed -- different sets can reconstruct ([[l2-crossover-universal-core]]).
  So an uninterpretable member is not evidence of a hidden concept; the
  fit may simply be using what is available.
* 3 validated circuits on one model and one dictionary. The societal-
  disruption and vehicle findings are single-seed observations, not
  replicated results. They are hypotheses worth a targeted run, not
  claims to publish as they stand.
* 16k-wide single-layer transcoders carry coarse concepts: there is NO
  "Einstein" latent, only "theoretical physics". Entity-level knowledge
  extraction is out of reach at this dictionary width.
