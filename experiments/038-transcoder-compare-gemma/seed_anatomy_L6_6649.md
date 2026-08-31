# Seed anatomy: L6/6649 (Gemma-tc arena)

Seed feature (Neuronpedia): seed: "each"/"other" (+place name "Los")

tri-amp circuit n=43, layer spread 11/6/4/8/10/4 over layers 0-5.
Cross-refs: rank in their direct-edge ranking (ct), SFC ranking, and
window-survival count in their as-published pruned circuits (of 48).
Labels fetched from Neuronpedia (gemmascope-transcoder-16k), 2026-08-26.

| lyr | feat | alpha | ct | sfc | wins/48 | Neuronpedia label |
|---|---|---|---|---|---|---|
| 3 | 15241 | 3.86 | 1155 | 32 | 8 | pairs of people or entities |
| 4 | 6128 | 3.40 | 55 | 122 | 6 | romantic relationships, affairs, family dynamics |
| 2 | 7602 | 3.36 | 14476 | 1517 | - | verbs ending in -ate/-ates |
| 0 | 8954 | 3.27 | 148 | 1117 | 8 | present tense verbs |
| 2 | 3978 | 3.18 | 1085 | 58 | 7 | the word "and" |
| 3 | 3205 | 2.86 | 408 | 401 | 12 | code snippets / documentation refs |
| 1 | 13370 | 2.84 | 134 | 480 | 3 |  |
| 4 | 2267 | 2.77 | 126 | 107 | 7 |  |
| 4 | 2670 | 2.51 | 51 | 47 | 6 |  |
| 1 | 7816 | 2.35 | 188 | 154 | 7 |  |
| 0 | 3498 | 2.33 | 146 | 42 | 5 |  |
| 2 | 5594 | 2.20 | 57 | 1799 | 4 |  |
| 0 | 2848 | 2.20 | 208 | 344 | 7 |  |
| 4 | 48 | 2.15 | 185 | 79 | 3 |  |
| 3 | 8510 | 2.13 | 161 | 147 | 3 |  |
| 4 | 7536 | 2.11 | 5 | 7 | 9 | the word "each" |
| 3 | 4583 | 2.09 | 1270 | 347 | 2 |  |
| 3 | 2045 | 2.08 | 171 | 31 | 2 |  |
| 1 | 3959 | 1.95 | 7 | 10 | 5 | "each" in mathematical/scientific writing |
| 4 | 14506 | 1.92 | 1631 | 421 | 1 |  |
| 5 | 14152 | 1.89 | 2306 | 29 | 2 |  |
| 0 | 7532 | 1.85 | 73 | 350 | 8 |  |
| 4 | 1630 | 1.78 | 113 | 749 | 3 |  |
| 3 | 3638 | 1.66 | 9605 | 246 | 1 |  |
| 0 | 9588 | 1.61 | 84 | 57 | 18 |  |
| 1 | 1820 | 1.59 | 1373 | 24 | 3 |  |
| 1 | 9148 | 1.57 | 1137 | 84 | 2 |  |
| 1 | 1286 | 1.54 | 13 | 14 | 10 |  |
| 4 | 13826 | 1.53 | 76 | 116 | 2 |  |
| 3 | 12967 | 1.48 | 22 | 8 | 4 |  |
| 0 | 3820 | 1.42 | 48 | 141 | 12 |  |
| 0 | 10109 | 1.41 | 107 | 671 | 3 |  |
| 5 | 14614 | 1.26 | 1 | 1 | 16 | the phrase "each other" |
| 0 | 3564 | 1.19 | 46 | 101 | 15 |  |
| 5 | 260 | 1.09 | 2 | 2 | 10 | people helping each other |
| 3 | 4558 | 0.87 | 38 | 9 | 6 |  |
| 0 | 8444 | 0.83 | 37 | 83 | 10 |  |
| 0 | 7424 | 0.73 | 35 | 1838 | 16 |  |
| 4 | 2803 | 0.72 | 8 | 4 | 7 |  |
| 0 | 15484 | 0.62 | 3 | 3 | 17 | "each" in formal writing |
| 2 | 8882 | 0.57 | 4 | 5 | 13 | the word "each" |
| 5 | 2267 | 0.40 | 310 | 19 | 2 |  |
| 4 | 14857 | 0.37 | 78 | 40 | 8 |  |

Reading: attribution's top-8 (ct-rank 1-8) are the token-copy chain --
"each"-detectors at L0-L5 feeding the seed -- and get our SMALLEST
alphas (0.4-1.3, near-redundant echoes). Our largest alphas (2.8-3.9)
go to CONTEXT nodes attribution barely credits (ct-rank 55-14476):
pairs-of-entities, reciprocal-relationship, plural/present-verb
features -- the semantic preconditions of 'each other'. All 43 nodes
appear somewhere in their 20k ranking (necessity agrees); the ordering
is what differs, and the amplification the fit assigns is roughly
inverse to their rank.