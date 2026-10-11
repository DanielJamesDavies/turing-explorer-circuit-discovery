# DAN-64 smoke checks (experiments/071-contrast-sweep/out_smoke2)
sets built: {'close_silent': np.int64(8), 'distant_silent': np.int64(8), 'fused_silent': np.int64(8), 'random_silent': np.int64(8)}; unusable: {'close_silent': np.int64(0), 'distant_silent': np.int64(0), 'fused_silent': np.int64(0), 'random_silent': np.int64(0)}
- [ok] every set usable on every target 
  close_silent    n [64]  fire 0.000 (train 0.000, held 0.000)  pre_rel med 0.040 p90 0.094  sim med 0.880  act_overlap 0  reserve [32]  dropped [{'id': 64, 'content': 0, 'dup': 0}, {'id': 64, 'content': 0, 'dup': 0}, {'id': 64, 'content': 0, 'dup': 0}]
- [ok] close_silent is silent post-Top-K (max per-target fire rate 0.000; a context or two can flip with the forward batch size)
- [ok] close_silent excludes the activating contexts 
- [ok] close_silent has a reserve (min 32)
  random_silent   n [64]  fire 0.000 (train 0.000, held 0.000)  pre_rel med 0.015 p90 0.066  sim med 0.674  act_overlap 0  reserve [32]  dropped [{'id': 0, 'content': 0, 'dup': 0}, {'id': 0, 'content': 0, 'dup': 0}, {'id': 0, 'content': 0, 'dup': 0}]
- [ok] random_silent is silent post-Top-K (max per-target fire rate 0.000; a context or two can flip with the forward batch size)
- [ok] random_silent excludes the activating contexts 
- [ok] random_silent has a reserve (min 32)
  distant_silent  n [64]  fire 0.000 (train 0.000, held 0.000)  pre_rel med 0.014 p90 0.059  sim med 0.239  act_overlap 0  reserve [32]  dropped [{'id': 0, 'content': 0, 'dup': 0}, {'id': 0, 'content': 0, 'dup': 0}, {'id': 0, 'content': 0, 'dup': 1}]
- [ok] distant_silent is silent post-Top-K (max per-target fire rate 0.000; a context or two can flip with the forward batch size)
- [ok] distant_silent excludes the activating contexts 
- [ok] distant_silent has a reserve (min 32)
  fused_silent    n [64]  fire 0.000 (train 0.000, held 0.000)  pre_rel med 0.008 p90 0.068  sim med 0.632  act_overlap 0  reserve [32]  dropped [{'id': 288, 'content': 0, 'dup': 0}, {'id': 288, 'content': 0, 'dup': 0}, {'id': 289, 'content': 0, 'dup': 1}]
- [ok] fused_silent is silent post-Top-K (max per-target fire rate 0.000; a context or two can flip with the forward batch size)
- [ok] fused_silent excludes the activating contexts 
- [ok] fused_silent has a reserve (min 32)
- [ok] the 4 sets are pairwise disjoint (content and id, incl. reserves) 
- [ok] no held-out context of any set is a training context of any arm (sets) 
- [ok] similarity to the top contexts: close > random > distant (medians 0.880 > 0.674 > 0.239; ordered on 100% of targets)
- [ok] close_silent re-selection reproduces the production set (Jaccard mean 1.0000, min 1.0000)
- [ok] fused_silent is disjoint from the three eval sets (content and id, incl. reserves) 
- [ok] fused_silent trains on no held-out context of an eval set (ho_in_train_fused_silent logged on 24 of 24 eval-set rows)
  fused_silent parts (set / train / held / reserve, per target): close [22.0] / [16.0] / [6.0] / [12.0]; distant [21.0] / [16.0] / [5.0] / [10.0]; random [21.0] / [16.0] / [5.0] / [10.0]
- [ok] fused_silent has its sub-mode thirds (set 22/21/21, train 16/16/16, held-out 6/5/5, reserve 12/10/10) 
- [ok] fused_silent parts' similarity: close > random > distant (medians close 0.869, random 0.626, distant 0.267)
- [ok] no target errors (status) 
- [ok] no errors in eval rows 
eval rows per (arm, eval): {('close_silent', 'close_silent'): np.int64(8), ('close_silent', 'distant_silent'): np.int64(8), ('close_silent', 'random_silent'): np.int64(8), ('distant_silent', 'close_silent'): np.int64(8), ('distant_silent', 'distant_silent'): np.int64(8), ('distant_silent', 'random_silent'): np.int64(8), ('fused_silent', 'close_silent'): np.int64(8), ('fused_silent', 'distant_silent'): np.int64(8), ('fused_silent', 'random_silent'): np.int64(8), ('random_silent', 'close_silent'): np.int64(8), ('random_silent', 'distant_silent'): np.int64(8), ('random_silent', 'random_silent'): np.int64(8)}
- [ok] every arm scored on every eval set (8 targets x 1 lambdas)
- [ok] eval close_silent: the scorer read 48 + 16 activating contexts (n_train [48], n_ho [16])
- [ok] eval close_silent: the scorer read 48 + 16 contrast contexts (n_neg_train [48], n_neg_ho [16])
- [ok] eval random_silent: the scorer read 48 + 16 activating contexts (n_train [48], n_ho [16])
- [ok] eval random_silent: the scorer read 48 + 16 contrast contexts (n_neg_train [48], n_neg_ho [16])
- [ok] eval distant_silent: the scorer read 48 + 16 activating contexts (n_train [48], n_ho [16])
- [ok] eval distant_silent: the scorer read 48 + 16 contrast contexts (n_neg_train [48], n_neg_ho [16])
- [ok] n_ho_trained = 0 on every eval and random row (max 0)
- [ok] every upstream site ablated 
- [ok] cross-source rows share no training contexts 
- [ok] fused_silent scored on all three eval sets (rows per eval set {'close_silent': np.int64(8), 'distant_silent': np.int64(8), 'random_silent': np.int64(8)})
- [ok] fused_silent: n_ho_trained = 0 on its eval and random rows (27 rows, max 0)
close_silent at lambda 0.001 (production circuit, re-scored on close_silent) vs the production eval rows:
```
                       median_abs_diff  max_abs_diff
free0_tk                        0.0177        0.0426
freeM_topk_tk                   0.0069        0.0601
freeN_topk_tk                   0.0073        0.0408
phi_sup_blind_tk                0.0000        0.0120
phi_cf_alpha_blind_tk           0.0009        0.0027
```
- [miss, not binding here] [pod-smoke, H100] re-scoring the production circuit reproduces production (median |diff| <= 0.005, max <= 0.05): device ['NVIDIA GeForce RTX 5070 Ti']. RTX vs H100 float noise cannot meet it locally; it is checked in the pod smoke
- [ok] same circuit size as production 
timings: fit median 46.8 s (fresh fits: 24), scorer run median 3.05 s, set build median 0.8 s
pass (protocol v1) by arm x eval: {('close_silent', 'close_silent'): np.float64(0.5), ('close_silent', 'distant_silent'): np.float64(0.25), ('close_silent', 'random_silent'): np.float64(0.38), ('distant_silent', 'close_silent'): np.float64(0.5), ('distant_silent', 'distant_silent'): np.float64(0.38), ('distant_silent', 'random_silent'): np.float64(0.38), ('fused_silent', 'close_silent'): np.float64(0.38), ('fused_silent', 'distant_silent'): np.float64(0.12), ('fused_silent', 'random_silent'): np.float64(0.38), ('random_silent', 'close_silent'): np.float64(0.62), ('random_silent', 'distant_silent'): np.float64(0.62), ('random_silent', 'random_silent'): np.float64(0.62)}
- [ok] random baseline rows without errors 
- [ok] random baseline scored for every arm on every eval set 
random baseline: 12 rows over ['close_silent', 'distant_silent', 'fused_silent', 'random_silent']; members drawn / reference 1.00; pass by arm x eval {('close_silent', 'close_silent'): np.float64(0.0), ('close_silent', 'distant_silent'): np.float64(0.0), ('close_silent', 'random_silent'): np.float64(0.0), ('distant_silent', 'close_silent'): np.float64(0.0), ('distant_silent', 'distant_silent'): np.float64(0.0), ('distant_silent', 'random_silent'): np.float64(0.0), ('fused_silent', 'close_silent'): np.float64(0.0), ('fused_silent', 'distant_silent'): np.float64(0.0), ('fused_silent', 'random_silent'): np.float64(0.0), ('random_silent', 'close_silent'): np.float64(0.0), ('random_silent', 'distant_silent'): np.float64(0.0), ('random_silent', 'random_silent'): np.float64(0.0)}

ALL CHECKS OK
