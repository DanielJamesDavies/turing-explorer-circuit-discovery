# 059 activating-context pool test (363 rows)

Primary config, stratified split. All arms scored on the same two held-out sets with one evaluation reference (A ablation value from the 48 strongest-train contexts; close contrast set). Activation read. Pass = all three faithfulness scores in [0.80, 1.25] (illustrative; the pass rule is DAN-8).

Targets scored by every arm: 15 (arms B/C need a mid-band pool).

## Held-out: strongest (16 of the 64 strongest)

| arm | trains on | nodes (median) | free0 | freeM_topk | freeN_topk | worst-of-3 dev | necessity | sufficiency to induce | pass | closer to 1 than A | closer to 1 than B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 32 strongest | 337 | 0.891 | 0.892 | 0.891 | 0.19 | 1.000 | 0.911 | 8/15 | 11/15 (-0.047) | 4/15 (+0.017) |
| A | 48 strongest | 397 | 0.922 | 0.859 | 0.870 | 0.30 | 1.000 | 0.925 | 5/15 | - | 1/15 (+0.072) |
| B | 32 strongest + 16 mid | 474 | 0.952 | 0.918 | 0.903 | 0.14 | 1.000 | 0.855 | 9/15 | 14/15 (-0.072) | - |
| C | 48 strongest + 48 mid | 536 | 0.896 | 0.902 | 0.867 | 0.32 | 1.000 | 0.955 | 5/15 | 8/15 (-0.019) | 4/15 (+0.053) |
| E | 48 strongest + 16 mid | 453 | 0.835 | 0.754 | 0.831 | 0.25 | 1.000 | 0.897 | 6/15 | 6/15 (+0.020) | 2/15 (+0.097) |
| B2 | 32 + 16, resampled | 462 | 0.946 | 0.859 | 0.854 | 0.23 | 1.000 | 0.875 | 6/15 | 11/15 (-0.022) | 5/15 (+0.040) |
| C800 | 48 + 48, 800 steps | 308 | 0.913 | 0.913 | 0.743 | 0.39 | 1.000 | 0.686 | 4/15 | 7/15 (+0.011) | 4/15 (+0.159) |
| Dm | 32 strongest, 32 contrast | 365 | 0.981 | 0.907 | 0.957 | 0.24 | 1.000 | 0.904 | 6/15 | 10/15 (-0.032) | 5/15 (+0.029) |
| Em | 48 + 16 mid, 64 contrast | 500 | 0.859 | 0.875 | 0.877 | 0.28 | 1.000 | 0.840 | 6/15 | 9/15 (-0.022) | 4/15 (+0.028) |
| Cm | 48 + 48 mid, 96 contrast | 570 | 0.933 | 0.954 | 0.816 | 0.29 | 1.000 | 0.724 | 4/15 | 7/15 (+0.057) | 5/15 (+0.094) |
| Bw | 32 + 16, gamma 1, lambda 2e-3 | 478 | 0.873 | 0.924 | 0.881 | 0.28 | 1.000 | 0.907 | 7/15 | 9/15 (-0.032) | 2/15 (+0.054) |
| B2w | 32 + 16 resampled, gamma 1, lambda 2e-3 | 476 | 0.871 | 0.860 | 0.876 | 0.33 | 1.000 | 0.830 | 5/15 | 9/15 (-0.028) | 5/15 (+0.029) |

## Held-out: mid-band (16 of the reservoir)

| arm | trains on | nodes (median) | free0 | freeM_topk | freeN_topk | worst-of-3 dev | necessity | sufficiency to induce | pass | closer to 1 than A | closer to 1 than B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 32 strongest | 337 | 1.012 | 1.050 | 1.068 | 0.74 | 1.000 | 1.731 | 1/15 | 6/15 (+0.033) | 4/15 (+0.131) |
| A | 48 strongest | 397 | 1.051 | 0.991 | 0.936 | 0.72 | 1.000 | 1.806 | 1/15 | - | 5/15 (+0.099) |
| B | 32 strongest + 16 mid | 474 | 0.999 | 0.981 | 0.940 | 0.41 | 1.000 | 1.749 | 2/15 | 10/15 (-0.099) | - |
| C | 48 strongest + 48 mid | 536 | 1.055 | 1.212 | 0.952 | 0.63 | 1.000 | 1.477 | 4/15 | 9/15 (-0.045) | 6/15 (+0.047) |
| E | 48 strongest + 16 mid | 453 | 0.851 | 0.995 | 0.906 | 0.59 | 1.000 | 1.642 | 2/15 | 9/15 (-0.080) | 8/15 (-0.003) |
| B2 | 32 + 16, resampled | 462 | 1.078 | 1.055 | 0.929 | 0.35 | 1.000 | 1.910 | 5/15 | 9/15 (-0.125) | 6/15 (+0.026) |
| C800 | 48 + 48, 800 steps | 308 | 1.034 | 1.024 | 0.878 | 0.57 | 1.000 | 1.495 | 2/15 | 7/15 (+0.003) | 6/15 (+0.027) |
| Dm | 32 strongest, 32 contrast | 365 | 1.081 | 1.017 | 1.060 | 0.72 | 1.000 | 1.716 | 1/15 | 7/15 (+0.017) | 5/15 (+0.055) |
| Em | 48 + 16 mid, 64 contrast | 500 | 0.780 | 0.905 | 0.940 | 0.60 | 1.000 | 1.531 | 2/15 | 8/15 (-0.026) | 6/15 (+0.018) |
| Cm | 48 + 48 mid, 96 contrast | 570 | 0.990 | 1.086 | 1.011 | 0.53 | 1.000 | 1.584 | 4/15 | 11/15 (-0.030) | 7/15 (+0.030) |
| Bw | 32 + 16, gamma 1, lambda 2e-3 | 478 | 0.957 | 0.937 | 0.985 | 0.75 | 1.000 | 1.683 | 2/15 | 8/15 (-0.013) | 5/15 (+0.046) |
| B2w | 32 + 16 resampled, gamma 1, lambda 2e-3 | 476 | 0.949 | 1.039 | 0.785 | 0.79 | 1.000 | 1.871 | 2/15 | 7/15 (+0.023) | 4/15 (+0.080) |

## Node overlap between arms (median Jaccard over common targets)

- D vs A: 0.59 (n=15)
- D vs B: 0.50 (n=15)
- A vs B: 0.49 (n=15)
- A vs C: 0.44 (n=15)
- B vs C: 0.41 (n=15)
- B vs B2: 0.52 (n=15)
- A vs E: 0.52 (n=15)
- B vs E: 0.52 (n=15)
- C vs C800: 0.44 (n=15)
- B vs Bw: 0.58 (n=15)
- B2 vs B2w: 0.58 (n=15)
- Bw vs B2w: 0.53 (n=15)