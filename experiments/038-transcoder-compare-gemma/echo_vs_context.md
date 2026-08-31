# The echo test: what attribution ranks vs what the fit amplifies

ECHO = member label shares a content word or a quoted token with
the SEED's label. CONTEXT = everything else. Label-lexical and
descriptive: it is a statistic over auto-interp text, not semantic
ground truth.

| seed | seed label | n | echo | context | med alpha echo | med alpha ctx | med ct-rank echo | med ct-rank ctx | MWU p (alpha) |
|---|---|---|---|---|---|---|---|---|---|
| L4/7115 | the word "respectively" | 10 | 3 | 7 | 1.75 | 2.02 | 8 | 13 | 0.909 |
| L4/10430 | relative clauses starting with "which." | 21 | 0 | 21 | - | 1.92 | - | 119 | - |
| L4/12424 | scientific names ending in "us", "a", or "ii | 21 | 8 | 13 | 2.76 | 1.86 | 5 | 122 | 0.014 |
| L6/16231 | scientific names of plants. | 140 | 28 | 112 | 1.52 | 1.43 | 55 | 218 | 0.864 |
| L6/2254 | references to academic degrees | 56 | 14 | 42 | 1.43 | 1.59 | 26 | 198 | 0.405 |
| L6/6649 | the words "each" and "other" and the place n | 43 | 6 | 37 | 1.54 | 1.89 | 7 | 134 | 0.080 |

## Control: echo share, ours vs their matched head

| seed | ours echo share | their top-n echo share | their 20k echo share |
|---|---|---|---|
| L4/7115 | 30% (n=10) | 50% (labelled 10/10) | 0% (labelled 6) |
| L4/10430 | 0% (n=21) | 5% (labelled 21/21) | 0% (labelled 1) |
| L4/12424 | 38% (n=21) | 57% (labelled 21/21) | 50% (labelled 2) |
| L6/16231 | 20% (n=140) | 30% (labelled 140/140) | 0% (labelled 4) |
| L6/2254 | 25% (n=56) | 43% (labelled 56/56) | 50% (labelled 2) |
| L6/6649 | 14% (n=43) | 28% (labelled 43/43) | 0% (labelled 4) |

Their matched head is fully labelled (fetch_labels.py fetches it),
so the middle column is a like-for-like base rate, not a subset.
The 20k column samples 400 features and counts only those already
cached, so it is indicative only.

OUR echo share is lower than their matched head on 6 of 6 seeds
(sign test p=0.031); ratio of shares, median 0.60 -- our circuits
carry roughly half the seed-vocabulary density of the equally
sized attribution head.


## Pooled over six seeds

| class | n | median alpha | median ct-rank |
|---|---|---|---|
| echo | 59 | 1.57 | 23 |
| context | 232 | 1.55 | 166 |

Mann-Whitney on alpha (context > echo): z=0.13, p=0.9
Mann-Whitney on attribution rank (echo better i.e. lower): z=-6.47, p=9.545e-11

Caveat: members are not independent (six circuits, shared
features across seeds), so p-values are indicative, not a
licence to claim significance at a seed level.