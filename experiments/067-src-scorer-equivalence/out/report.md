# DAN-73 src scorer equivalence

Targets: 12 stratified (3.attn.37736, 2.mlp.29792, 4.resid.29644, 7.attn.6600, 7.mlp.15221, 6.resid.35258, 10.attn.14758, 10.mlp.14093, 11.resid.13385, 3.attn.17933, 0.mlp.826, 0.resid.21319), 4 padded mid-band (6.attn.21578, 9.attn.10280, 8.attn.26692, 9.attn.12940). Scoring errors: 0.

| comparison | rows | fields | exact | <=1e-6 | <=1e-3 | >1e-3 | None mismatch | other | missing | max abs diff (at) | pass flips |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (b) src vs experiment scorer, as_run, held strong (same GPU) | 12 | 1749 | 1749 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (c) exp vs stored H100 rows, as_run, held strong | 12 | 1749 | 1008 | 0 | 118 | 622 | 1 | 0 | 0 | 8.44e+04 (11.resid.13385 e0_pre) | 1 |
| (c) src vs stored H100 rows, as_run, held strong | 12 | 1749 | 1008 | 0 | 118 | 622 | 1 | 0 | 0 | 8.44e+04 (11.resid.13385 e0_pre) | 1 |
| (b) src vs experiment scorer, as_run, held mid (same GPU) | 15 | 2190 | 2190 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (c) exp vs stored H100 rows, as_run, held mid | 15 | 2190 | 1298 | 0 | 101 | 791 | 0 | 0 | 0 | 2.79e+04 (11.resid.13385 e0_tr_pre) | 0 |
| (c) src vs stored H100 rows, as_run, held mid | 15 | 2190 | 1298 | 0 | 101 | 791 | 0 | 0 | 0 | 2.79e+04 (11.resid.13385 e0_tr_pre) | 0 |
| (d) src vs experiment scorer, explicit, held mid (same GPU) | 4 | 584 | 584 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (d) exp explicit vs out_rescore_mid rows | 4 | 584 | 584 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (d) src explicit vs out_rescore_mid rows | 4 | 584 | 584 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (b) specificity: src vs 056 (same GPU) | 36 | 2625 | 2625 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (-) | 0 |
| (c) specificity: src vs stored H100 rows | 36 | 2625 | 2225 | 1 | 33 | 364 | 2 | 0 | 0 | 3.13e+04 (10.attn.14758 Z rank_empty) | 0 |

Fields with a None / type mismatch or |diff| > 1e-3 (count of rows), per comparison:

- (c) exp vs stored H100 rows, as_run, held strong: {'phi_pin_alpha_topk_raw_pre': 12, 'freeN_topk_pre': 12, 'freeM_topk_raw_pre': 12, 'free0_tr_raw_pre': 12, 'freeM_topk_pre': 12, 'free0_raw_pre': 12, 'phi_pin_alpha_zero_raw_pre': 12, 'freeN_topk_raw_pre': 12, 'free0_a1_raw_pre': 12, 'free0_tr_raw_tk': 11, 'freeM_dense_pre': 11, 'freeM_topk_tk': 11}
- (c) src vs stored H100 rows, as_run, held strong: {'phi_pin_alpha_topk_raw_pre': 12, 'freeN_topk_pre': 12, 'freeM_topk_raw_pre': 12, 'free0_tr_raw_pre': 12, 'freeM_topk_pre': 12, 'free0_raw_pre': 12, 'phi_pin_alpha_zero_raw_pre': 12, 'freeN_topk_raw_pre': 12, 'free0_a1_raw_pre': 12, 'free0_tr_raw_tk': 11, 'freeM_dense_pre': 11, 'freeM_topk_tk': 11}
- (c) exp vs stored H100 rows, as_run, held mid: {'phi_pin_alpha_topk_raw_pre': 15, 'freeN_topk_pre': 15, 'free0_perm_raw_pre': 15, 'free0_raw_pre': 15, 'freeN_dense_pre': 15, 'freeN_topk_raw_pre': 15, 'free0_a1_raw_pre': 15, 'freeM_topk_raw_pre': 14, 'free0_tr_raw_pre': 14, 'freeM_topk_pre': 14, 'phi_pin_alpha_topk_pre': 14, 'freeN_dense_raw_pre': 13}
- (c) src vs stored H100 rows, as_run, held mid: {'phi_pin_alpha_topk_raw_pre': 15, 'freeN_topk_pre': 15, 'free0_perm_raw_pre': 15, 'free0_raw_pre': 15, 'freeN_dense_pre': 15, 'freeN_topk_raw_pre': 15, 'free0_a1_raw_pre': 15, 'freeM_topk_raw_pre': 14, 'free0_tr_raw_pre': 14, 'freeM_topk_pre': 14, 'phi_pin_alpha_topk_pre': 14, 'freeN_dense_raw_pre': 13}
- (c) specificity: src vs stored H100 rows: {'rank_empty': 35, 'switched_on_circuit': 35, 'n_new_circuit': 34, 'control_faith_median': 33, 'target_faith_pre': 32, 'specificity_gap': 31, 'sibling_faith_median': 30, 'n_new_empty': 27, 'jaccard_circuit': 27, 'switched_on_empty': 22, 'n_control_scored': 22, 'jaccard_empty': 13}