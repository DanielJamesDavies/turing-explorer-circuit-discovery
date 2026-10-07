### Prediction-like targets by kind

| kind | n | writer | pred_like | focused | pred_like_pass | med_boost_peak | med_ctx_z | pct_pred_like |
|---|---|---|---|---|---|---|---|---|
| attn | 4094 | 3 | 1 | 1 | 0 | 0.0664 | 0.0462 | 0.0244 |
| mlp | 5457 | 249 | 100 | 20 | 23 | 0.288 | 0.391 | 1.83 |
| resid | 5453 | 2138 | 1192 | 119 | 652 | 0.761 | 0.894 | 21.9 |

### Prediction-like targets by layer (count / targets at that site)

| layer (1-based) | attn | mlp | resid | pred-like passing |
|---|---|---|---|---|
| 1 | - | 1/454 (0%) | 0/453 (0%) | 0 |
| 2 | 0/133 (0%) | 0/455 (0%) | 2/455 (0%) | 2 |
| 3 | 0/296 (0%) | 0/455 (0%) | 5/455 (1%) | 5 |
| 4 | 0/406 (0%) | 0/455 (0%) | 9/453 (2%) | 8 |
| 5 | 0/444 (0%) | 0/455 (0%) | 18/453 (4%) | 15 |
| 6 | 0/355 (0%) | 0/454 (0%) | 59/455 (13%) | 50 |
| 7 | 0/424 (0%) | 0/455 (0%) | 106/455 (23%) | 82 |
| 8 | 0/379 (0%) | 1/454 (0%) | 164/455 (36%) | 105 |
| 9 | 0/338 (0%) | 7/455 (2%) | 162/454 (36%) | 94 |
| 10 | 0/447 (0%) | 23/455 (5%) | 185/455 (41%) | 117 |
| 11 | 1/434 (0%) | 35/455 (8%) | 194/455 (43%) | 95 |
| 12 | 0/438 (0%) | 33/455 (7%) | 288/455 (63%) | 102 |

### Targets vs random non-target latents at the same sites (medians)

| kind | z1 targets | z1 null | gain targets | gain null | boost@mean targets | boost@mean null | ctx_z targets | ctx_z null | ctx_z shuffled |
|---|---|---|---|---|---|---|---|---|---|
| attn | 4.73 | 4.74 | 1.03 | 1.02 | 0.0213 | 0.019 | 0.0462 | -0.0138 | 0.0292 |
| mlp | 4.78 | 4.78 | 1.07 | 1.06 | 0.0629 | 0.063 | 0.391 | 0.368 | 0.0774 |
| resid | 5.22 | 5.23 | 1.05 | 1.05 | 0.151 | 0.153 | 0.894 | 0.927 | 0.289 |

### Top circuit per position: explorer score vs DLA (1,258 positions)

| ranking | mean layer (0-based) | resid/mlp/attn % | median DLA to top-1 (nats) | pass % |
|---|---|---|---|---|
| explorer score (top-1) | 5.02 | 56/30/13 | 1.07e-05 | 51.6 |
| DLA to top-1 (top-1) | 8.06 | 80/16/4 | 0.041 | 38.4 |

### Causal check: single-target ablations (144 per set, 48 positions) and set-joint ablations

| set | pred. DLA (single) | act | d logit top1 | d logp top1 | 95% CI | KL | top1 flips % | joint d logp | joint KL | joint flips % |
|---|---|---|---|---|---|---|---|---|---|---|
| dla | 0.0325 | 3.0752 | -0.0469 | -0.0194 | [-0.0301, -0.0136] | 0.0010 | 8.3333 | -0.1174 | 0.0161 | 22.9167 |
| score | 0.0004 | 2.6763 | -0.0081 | -0.0031 | [-0.0054, -0.0005] | 0.0016 | 4.8611 | -0.0206 | 0.0201 | 12.5000 |
| act | 0.0034 | 5.3459 | -0.0279 | -0.0047 | [-0.0182, -0.0000] | 0.0041 | 9.7222 | -0.0360 | 0.0272 | 12.5000 |
| random | 0.0002 | 0.6856 | -0.0013 | -0.0006 | [-0.0015, 0.0000] | 0.0001 | 2.0833 | -0.0003 | 0.0006 | 6.2500 |

### Direct-path fidelity: predicted (-DLA) vs measured change of log p(top-1), single ablations

| layers | n | pearson | spearman | slope measured/predicted |
|---|---|---|---|---|
| L1-4 | 96 | 0.742 | 0.351 | 3.78 |
| L5-8 | 162 | 0.575 | 0.47 | 2.24 |
| L9-12 | 318 | 0.719 | 0.71 | 0.613 |
| all | 576 | 0.5 | 0.6 | 1.1 |

### The 15 largest fired-circuit DLAs to the model's top-1 (all 1,258 positions)

| token | model top-1 | target | act | DLA | score | pass | target promotes |
|---|---|---|---|---|---|---|---|
| ' of' | ' ph' (0.29) | L11 · RESID · 32102  (10.resid.32101) | 16.770 | 1.002 | 0.114 | True | ' ph' ' sounds' ' pron' ' sound' ' sy' |
| ' the' | ' same' (0.03) | L4 · RESID · 14287  (3.resid.14286) | 11.606 | 0.791 | 0.492 | True | ' latter' ' same' ' entire' ' first' ' way' |
| 's' | ' integrity' (0.40) | L11 · RESID · 4113  (10.resid.4112) | 11.825 | 0.703 | 0.142 | True | ' integrity' ' equilibrium' ' balance' ' throughout' ' dign' |
| ' d' | 'ough' (0.74) | L12 · RESID · 10793  (11.resid.10792) | 20.979 | 0.682 | 0.163 | True | 'ough' 'ots' 'um' 'arts' 'anks' |
| ' add' | ' a' (0.19) | L10 · RESID · 7550  (9.resid.7549) | 21.405 | 0.545 | 0.312 | False | 'usk' ' amid' ' dawn' '<s>' ' Copy' |
| ' Swift' | ',' (0.21) | L8 · RESID · 39012  (7.resid.39011) | 12.225 | 0.508 | 0.371 | False | ' of' ' and' ' Of' ' is' ' levels' |
| ' I' | 'bn' (0.23) | L9 · RESID · 13343  (8.resid.13342) | 25.474 | 0.499 | 0.248 | False | 'onic' 'ne' 'tem' 'nex' 'so' |
| ' periods' | ' of' (0.32) | L9 · RESID · 6390  (8.resid.6389) | 4.806 | 0.494 | 0.050 | True | ' of' ' Of' 'of' ' OF' ' von' |
| ' consider' | 'ations' (0.49) | L9 · RESID · 2479  (8.resid.2478) | 22.572 | 0.489 | 0.572 | False | 'ARY' 'icon' ' necessary' 'isser' 'entials' |
| ' population' | ' parameter' (0.49) | L11 · RESID · 6199  (10.resid.6198) | 23.241 | 0.484 | 0.165 | True | ' growth' ' density' ' cent' ' sizes' ' movements' |
| ' When' | ' we' (0.12) | L11 · RESID · 13351  (10.resid.13350) | 25.455 | 0.473 | 0.262 | True | ' discuss' 'ever' ' considering' ' exam' ' studying' |
| ' particular' | ' attention' (0.39) | L8 · RESID · 19858  (7.resid.19857) | 20.847 | 0.469 | 0.427 | True | ' attention' 'ized' ' type' 'ization' ' interest' |
| ' allows' | ' for' (0.35) | L8 · RESID · 14684  (7.resid.14683) | 6.648 | 0.437 | 0.160 | False | ' for' 'for' ' für' ' profession' ' sch' |
| ' access' | ' to' (0.64) | L10 · RESID · 32204  (9.resid.32203) | 29.088 | 0.425 | 0.348 | True | ' into' ' to' 'ibility' ' thre' ' destin' |
| ' it' | "'" (0.26) | L7 · RESID · 36319  (6.resid.36318) | 17.155 | 0.423 | 0.395 | False | 'iner' "'" ' doesn' ' wasn' 'chy' |

### Most prediction-like targets (largest boost at peak among prediction-like)

| target | pass | boost@peak | z1 | ctx_z | promotes | logit_ctx next |
|---|---|---|---|---|---|---|
| L12 · RESID · 29306  (11.resid.29305) | False | 10.24 | 14.04 | 3.25 | 'ans' 'anned' 'anning' 'ann' 'ind' 'anse' | 'in' 'it' 'ed' 'ill' 'our' |
| L12 · RESID · 10793  (11.resid.10792) | True | 9.76 | 9.48 | 2.22 | 'ough' 'ots' 'um' 'arts' 'anks' 'ishes' | 'on' 'al' 'ent' 'end' 'os' |
| L10 · RESID · 5372  (9.resid.5371) | False | 9.06 | 13.17 | 1.69 | 'unci' 'ounced' 'oun' 'ounce' 'ounds' 'ound' | 'on' 'ing' 'ot' 'id' 'ig' |
| L12 · RESID · 25381  (11.resid.25380) | False | 8.33 | 10.91 | 1.60 | 'ity' 'ization' 'izing' ' culture' 'ized' 'ising' | 'es' 'is' 'ar' 'ed' 'ing' |
| L11 · RESID · 40724  (10.resid.40723) | True | 8.04 | 12.63 | 2.06 | 'elling' 'ellers' 'eller' 'aking' 'ension' 'ali' | 'on' 'al' 'ing' ' of' 'am' |
| L11 · RESID · 32980  (10.resid.32979) | True | 7.95 | 11.31 | 1.38 | 'ian' 'ily' 'itar' 'iness' 'iel' 'itan' | 'it' 'al' 'ion' 'ic' 'el' |
| L11 · RESID · 20082  (10.resid.20081) | True | 7.88 | 11.98 | 1.97 | 'raint' 'rained' 'ruct' 'rain' 'itution' 'ock' | 'es' 'at' 'or' 'ar' 'ed' |
| L12 · RESID · 22718  (11.resid.22717) | False | 7.83 | 9.36 | 2.90 | 'reading' ' assistant' 'reader' ' assist' 'read' 'writing' | 'on' 'or' 'le' 'ed' 'ing' |
| L12 · RESID · 27992  (11.resid.27991) | True | 7.44 | 11.28 | 1.76 | 'ge' ' Set' 'acity' 'port' ' net' 'arp' | 'er' 'en' 'ar' 'al' 'el' |
| L12 · RESID · 36165  (11.resid.36164) | True | 7.26 | 7.85 | 2.97 | 'or' 'cite' 'elf' 'ison' 'str' 'N' | 'er' 'en' 'on' 'es' 'or' |
| L11 · RESID · 32102  (10.resid.32101) | True | 7.12 | 10.50 | 1.08 | ' ph' ' sounds' ' pron' ' sound' ' sy' ' vocal' | 'in' 'on' 'es' 'an' 'ed' |
| L12 · RESID · 24567  (11.resid.24566) | False | 7.07 | 14.86 | 1.69 | 'D' 'd' 'DD' 'DS' 'DR' 'DA' | 'es' 'or' 'ou' 'ed' 'ing' |
| L11 · RESID · 2582  (10.resid.2581) | False | 7.03 | 14.88 | 7.35 | 'ins' 'in' 'inas' 'iny' 'ining' 'ints' | 'er' 'in' |
| L10 · RESID · 32707  (9.resid.32706) | True | 6.98 | 11.29 | 1.42 | 'ematic' 'ém' 'eman' 'em' 'ema' 'emat' | 'es' 'ar' 'al' 'ed' 'ing' |
| L12 · RESID · 4158  (11.resid.4157) | True | 6.93 | 10.30 | 1.74 | ' b' ' bond' ' Bond' ' B' ' links' ' character' | 'es' 'ar' 'ed' 'ing' 'ic' |
