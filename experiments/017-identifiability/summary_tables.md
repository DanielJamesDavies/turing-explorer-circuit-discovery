# D4.2 summary tables (11 seeds; split-half + precision + K* + alpha)

## T1 — split-half Jaccard medians: driver heads vs full membership

| group | metric | R | A (abl-ig) | C (direct) |
|---|---|---:|---:|---:|
| non-attn (9) | head @16 | 0.78 | 0.68 | 0.68 |
| non-attn (9) | head @64 | 0.66 | 0.66 | 0.68 |
| non-attn (9) | FULL membership | 0.61 | 0.51 | — |
| attn (2) | head @16 | 0.36 | 0.52 | 0.50 |
| attn (2) | head @64 | 0.34 | 0.52 | 0.43 |
| attn (2) | FULL membership | 0.61 | 0.55 | — |

## T2 — precision (bf16 vs fp32 archived R): head Jaccards

@16: median 1.00
@64: median 1.00
@256: median 0.99
@1024: median 0.99
full membership: median 1.00
wall-clock R full-48: fp32 (D2.2) vs bf16: 2/19766 94->2s; 8/20333 8->8s; 9/38734 8->8s; 13/30053 11->12s; 17/38268 23->21s; 20/35678 30->29s
median speedup: 0.97x

## T3 — K* split-half spread (A-ranking, eval anchors)

2/19766    L0  resid K*    155 /    207 (ratio 1.34)
8/20333    L2  resid K*     70 /     98 (ratio 1.40)
9/38734    L3  attn  K*   1593 /    764 (ratio 2.09)
13/30053   L4  mlp   K*   3585 /   3339 (ratio 1.07)
17/38268   L5  resid K*   3529 /   4693 (ratio 1.33)
20/35678   L6  resid K*   9353 /   8273 (ratio 1.13)
25/10628   L8  mlp   K*   4694 /   5264 (ratio 1.12)
26/17432   L8  resid K*   6868 /   4067 (ratio 1.69)
27/6859    L9  attn  K* -1 / -1
29/2753    L9  resid K*   9622 /   6518 (ratio 1.48)
35/6599    L11 resid K*  26777 /  23117 (ratio 1.16)

## T4 — AMPC alpha stability and cf transfer (K=16)

2/19766    a* 2.854/3.096  cf 1.0435/1.029
8/20333    a* 2.006/2.006  cf 1.0896/1.0398
9/38734    a* 4.67/3.338  cf 1.0807/0.9839
13/30053   a* 8.0/8.0  cf 0.8354/0.7405
17/38268   a* 3.338/2.369  cf 1.0738/1.047
20/35678   a* 1.521/1.521  cf 1.0128/1.047
25/10628   a* 8.0/8.0  cf 0.9953/0.7062
26/17432   a* 2.49/2.611  cf 1.0084/1.0336
27/6859    a* 8.0/8.0  cf 0.0/0.0
29/2753    a* 1.885/1.764  cf 1.0909/1.1082
35/6599    a* 1.037/1.158  cf 0.9793/1.0138
