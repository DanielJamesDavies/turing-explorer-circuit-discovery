# Eval-substrate sensitivity: fixed circuits, cf across negative modes

## Median cf by set x substrate (store = frozen exam reference)

| set | store | close | random | distant | fused |
|---|---:|---:|---:|---:|---:|
| AR16 | 1.043 | 1.035 | 0.972 | 0.966 | 1.020 |
| R64 | 0.711 | 0.711 | 0.705 | 0.719 | 0.719 |
| R1024 | 1.108 | 1.103 | 1.103 | 1.101 | 1.116 |

## Per-seed max deviation from store (cf, across the 4 modes)

26/17432   R64    store 0.337, max |dev| 0.199
9/38734    R64    store 0.459, max |dev| 0.138
25/10628   R64    store 0.055, max |dev| 0.138
26/17432   AR16   store 1.063, max |dev| 0.125
17/38268   AR16   store 1.047, max |dev| 0.112
27/6859    R1024  store 0.202, max |dev| 0.111
8/20333    R64    store 0.711, max |dev| 0.080
8/20333    AR16   store 1.030, max |dev| 0.075

median of per-(seed,set) max deviations: 0.041
a_base across ALL AR16 substrate cells: max 0.0000 (all effectively 0 -> negatives are genuinely silent post-top-k on every mode)
