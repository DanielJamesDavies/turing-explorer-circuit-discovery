# 061 — WCM vs unweighted circuit masking across the sparsity price (pilot for DAN-76 / DAN-77)

2026-09-25, local, 16 pilot targets (15 with a mid-band pool). Both methods train on the protocol-v1 contexts of 059
(arm B's training set: 32 strongest + 16 mid-band, stratified split), with close contrast contexts, rank-keep 3e-3,
train-only ablation values and γ = 0.25. The only difference is whether the coefficients are fitted. Scored by the
eval pass on the held-out strongest contexts.

Sanity check: WCM at λ = 1e-3 reproduces 059 arm B exactly (474 nodes, identical scores).
Files: `sweep.py`, `results/summary.md`, `results/faithfulness_vs_size.png`, `results/eval_*.jsonl`.

| Method | λ | Nodes (median) | free0 | freeM | freeN | Worst-of-3 dev | In band | Suff. to induce |
|---|---|---|---|---|---|---|---|---|
| **WCM** | 2.5e-4 | 1,237 | 0.96 | 1.00 | 0.93 | 0.14 | 9/15 | 0.83 |
| | 5e-4 | 748 | 0.98 | 0.95 | 0.95 | 0.20 | 9/15 | 0.86 |
| | **1e-3** | **474** | 0.95 | 0.92 | 0.90 | 0.14 | 9/15 | 0.86 |
| | 2e-3 | 321 | 0.95 | 0.82 | 0.81 | 0.36 | 5/15 | 0.91 |
| | 4e-3 | 239 | 0.81 | 0.72 | 0.43 | 0.60 | 4/15 | 0.85 |
| **Unweighted** | 1e-5 | 14,606 | 0.92 | 0.97 | 1.01 | 0.13 | 11/15 | 0.92 |
| | 3e-5 | 6,741 | 0.90 | 0.96 | 0.98 | 0.16 | 8/15 | 0.86 |
| | 1e-4 | 3,737 | 0.87 | 0.93 | 0.85 | 0.25 | 7/15 | 0.71 |
| | 3e-4 | 2,174 | 0.83 | 0.82 | 0.81 | 0.41 | 5/15 | 0.70 |
| | 1e-3 | 820 | 0.70 | 0.35 | 0.40 | 0.69 | 3/15 | 0.54 |

Necessity is 1.00 everywhere.

**At matched size** (about 750–820 nodes), WCM holds 0.98 / 0.95 / 0.95 with 9/15 in band. Unweighted holds 0.70 /
0.35 / 0.40 with 3/15 in band.

## Natural-scale cost, per target (smallest circuit with worst-of-3 deviation ≤ 0.3)

| Depth | Targets where both reach it | WCM nodes | Unweighted nodes | Ratio |
|---|---|---|---|---|
| Layers 0–5 | 7 | 4–1,352 | 14–3,350 | median ~3.5× (1.3–11×) |
| Layers 6–7 | 1 (6.resid) | 1,345 | 2,720 | 2× |
| **Layers 8–11** | 2 (8.attn, 11.mlp) | **539–568** | **14,606–24,339** | **27–43×** |

At ≤ 0.2 the deep ratio is 19× (11.mlp.30743: 770 vs 14,606).

Deep targets that only one method reaches:
- unweighted only: 9.attn.21759 at 26,959 nodes and 10.attn.36603 (near-threshold) at 85,940;
- WCM only: 11.resid.8702 at 679.

## Reading

1. **The coefficients buy one to two orders of magnitude at depth.** At natural scale (α = 1), a trained circuit for
   a deep target needs about 15k–86k latents to be as faithful as a WCM circuit of about 500–800. Shallow targets
   cost only 2–5×. This is the first measurement under the current protocol that supports the abstract's claim
   ("for deep targets, 10^4–10^5 latents at natural scale"), which rested on 3–4-target July panels with the old
   evaluation. Keep it `\pending` until DAN-76 runs on about 100 targets per model.
2. **Coverage is similar; cost is not.** Each method reaches the ≤ 0.3 bar on 12/15 targets, not the same 12. With
   enough nodes, unweighted masking eventually reaches most targets. The difference is size.
3. **WCM's operating range is λ = 5e-4 to 1e-3** (about 500–750 nodes). Below it circuits grow with no gain. Above
   it (2e-3, 4e-3) faithfulness under mean ablation drops fast. λ = 1e-3 remains a sound default.
4. **Figure 4 draft** (`results/faithfulness_vs_size.png`): the two curves separate cleanly under mean ablation, and
   converge only past about 10^4 unweighted nodes. The Caples-style comparison figure (DAN-77) has its first two
   curves.

**Caveats:** 15 targets, one fit per point; medians over targets at each λ mix depths, so the per-target ratios
above are the fairer read. The random-circuit baseline curve and the attribution / external-method curves are not
in yet.
