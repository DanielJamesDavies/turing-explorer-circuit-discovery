# 055 contrast-source matrix (16 targets, activation read, held-out contexts)

rows per (arm, eval source):
ev        close  random  distant  store
arm                                    
old        16.0    16.0     16.0   16.0
store      16.0    16.0     16.0   16.0
close      16.0    16.0     16.0   16.0
random     16.0    16.0     16.0   16.0
distant    16.0    16.0     16.0   16.0
eq1        16.0    16.0     16.0   16.0
eq2        16.0    16.0     16.0   16.0
otall3     16.0     NaN      NaN    NaN
otcut3     16.0     NaN      NaN    NaN
otcut3e3   16.0     NaN      NaN    NaN
otcut2     16.0     NaN      NaN    NaN

## HEADLINE: faithfulness under each ablation method (freeN on close contrast contexts)
band = share of targets with ALL THREE in [0.80, 1.25] (illustrative; the pass rule is DAN-8)
old      nodes  354 | free0 0.965 | freeM_topk 0.829 | freeN_topk 0.804 | all three in band  5/16 | dense: freeM_dense 1.017 | freeN_dense 0.710 | vs close@0.25 -0.004 -0.023 -0.004
store    nodes  346 | free0 0.926 | freeM_topk 0.923 | freeN_topk 0.681 | all three in band  5/16 | dense: freeM_dense 1.050 | freeN_dense 0.673 | vs close@0.25 -0.004 +0.019 -0.001
close    nodes  398 | free0 0.968 | freeM_topk 0.918 | freeN_topk 0.787 | all three in band  6/16 | dense: freeM_dense 1.042 | freeN_dense 0.889 | vs close@0.25 (reference)
random   nodes  341 | free0 0.988 | freeM_topk 0.835 | freeN_topk 0.756 | all three in band  4/16 | dense: freeM_dense 1.056 | freeN_dense 0.843 | vs close@0.25 +0.022 +0.000 -0.035
distant  nodes  377 | free0 1.006 | freeM_topk 0.866 | freeN_topk 0.778 | all three in band  6/16 | dense: freeM_dense 1.014 | freeN_dense 0.822 | vs close@0.25 -0.006 -0.029 -0.019
eq1      nodes  497 | free0 0.912 | freeM_topk 0.901 | freeN_topk 0.911 | all three in band  7/16 | dense: freeM_dense 1.048 | freeN_dense 0.959 | vs close@0.25 -0.010 -0.074 -0.006
eq2      nodes  303 | free0 0.984 | freeM_topk 0.957 | freeN_topk 0.908 | all three in band  7/16 | dense: freeM_dense 1.064 | freeN_dense 0.974 | vs close@0.25 +0.002 -0.003 +0.005
otall3   nodes  520 | free0 0.963 | freeM_topk 0.971 | freeN_topk 0.897 | all three in band  6/16 | dense: freeM_dense 1.088 | freeN_dense 0.853 | vs close@0.25 -0.006 +0.048 +0.039
otcut3   nodes  401 | free0 0.963 | freeM_topk 0.942 | freeN_topk 0.853 | all three in band  7/16 | dense: freeM_dense 1.048 | freeN_dense 0.951 | vs close@0.25 +0.009 +0.005 -0.028
otcut3e3 nodes  438 | free0 0.996 | freeM_topk 0.906 | freeN_topk 0.830 | all three in band  6/16 | dense: freeM_dense 1.055 | freeN_dense 0.890 | vs close@0.25 +0.020 +0.008 +0.009
otcut2   nodes  439 | free0 0.966 | freeM_topk 0.850 | freeN_topk 0.805 | all three in band  5/16 | dense: freeM_dense 1.030 | freeN_dense 0.882 | vs close@0.25 -0.003 +0.025 -0.012

## Circuit size (nodes), by training source
          count     25%    50%     75%
arm                                   
old        16.0  170.75  354.0  487.25
store      16.0  150.50  346.0  511.00
close      16.0  164.50  398.5  486.00
random     16.0  164.50  341.0  512.50
distant    16.0  169.50  377.0  524.50
eq1        16.0  253.50  497.0  627.25
eq2        16.0  170.50  303.0  422.50
otall3     16.0  355.75  520.0  632.75
otcut3     16.0  209.00  401.0  494.25
otcut3e3   16.0  219.00  438.0  503.75
otcut2     16.0  273.25  439.0  540.75

## faith Z (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.965   0.965    0.965  0.965
store     0.926   0.926    0.926  0.926
close     0.968   0.968    0.968  0.968
random    0.988   0.988    0.988  0.988
distant   1.006   1.006    1.006  1.006
eq1       0.912   0.912    0.912  0.912
eq2       0.984   0.984    0.984  0.984
otall3    0.963     NaN      NaN    NaN
otcut3    0.963     NaN      NaN    NaN
otcut3e3  0.996     NaN      NaN    NaN
otcut2    0.966     NaN      NaN    NaN

## faith A (sp) (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.829   0.829    0.829  0.829
store     0.923   0.923    0.923  0.923
close     0.918   0.918    0.918  0.918
random    0.835   0.835    0.835  0.835
distant   0.866   0.866    0.866  0.866
eq1       0.901   0.901    0.901  0.901
eq2       0.957   0.957    0.957  0.957
otall3    0.971     NaN      NaN    NaN
otcut3    0.942     NaN      NaN    NaN
otcut3e3  0.906     NaN      NaN    NaN
otcut2    0.850     NaN      NaN    NaN

## faith C (sp) (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.804   0.825    0.653  0.869
store     0.681   0.787    0.630  0.896
close     0.787   0.823    0.687  0.861
random    0.756   0.790    0.694  0.812
distant   0.778   0.873    0.736  0.875
eq1       0.911   0.831    0.698  0.858
eq2       0.908   0.842    0.731  0.866
otall3    0.897     NaN      NaN    NaN
otcut3    0.853     NaN      NaN    NaN
otcut3e3  0.830     NaN      NaN    NaN
otcut2    0.805     NaN      NaN    NaN

## faith C (dense) (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.710   0.598    0.403  0.844
store     0.673   0.343    0.201  0.875
close     0.889   0.501    0.435  0.832
random    0.843   0.917    0.547  0.878
distant   0.822   0.923    0.923  0.918
eq1       0.959   0.672    0.486  0.820
eq2       0.974   0.794    0.338  0.849
otall3    0.853     NaN      NaN    NaN
otcut3    0.951     NaN      NaN    NaN
otcut3e3  0.890     NaN      NaN    NaN
otcut2    0.882     NaN      NaN    NaN

## necessity (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old         1.0     1.0      1.0    1.0
store       1.0     1.0      1.0    1.0
close       1.0     1.0      1.0    1.0
random      1.0     1.0      1.0    1.0
distant     1.0     1.0      1.0    1.0
eq1         1.0     1.0      1.0    1.0
eq2         1.0     1.0      1.0    1.0
otall3      1.0     NaN      NaN    NaN
otcut3      1.0     NaN      NaN    NaN
otcut3e3    1.0     NaN      NaN    NaN
otcut2      1.0     NaN      NaN    NaN

## suff. to induce (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.744   0.754    0.740  0.727
store     0.817   0.772    0.759  0.783
close     0.737   0.720    0.735  0.695
random    0.756   0.742    0.752  0.716
distant   0.685   0.654    0.648  0.608
eq1       0.685   0.664    0.686  0.645
eq2       0.643   0.631    0.661  0.641
otall3    0.730     NaN      NaN    NaN
otcut3    0.766     NaN      NaN    NaN
otcut3e3  0.809     NaN      NaN    NaN
otcut2    0.796     NaN      NaN    NaN

## clamped A (sp) (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old       0.431   0.431    0.431  0.431
store     0.319   0.319    0.319  0.319
close     0.383   0.383    0.383  0.383
random    0.371   0.371    0.371  0.371
distant   0.301   0.301    0.301  0.301
eq1       0.584   0.584    0.584  0.584
eq2       0.452   0.452    0.452  0.452
otall3    0.662     NaN      NaN    NaN
otcut3    0.607     NaN      NaN    NaN
otcut3e3  0.733     NaN      NaN    NaN
otcut2    0.743     NaN      NaN    NaN

## random baseline Z (median; rows = training source, columns = evaluation contrast source)
ev        close  random  distant  store
arm                                    
old         0.0     0.0      0.0    0.0
store       0.0     0.0      0.0    0.0
close       0.0     0.0      0.0    0.0
random      0.0     0.0      0.0    0.0
distant     0.0     0.0      0.0    0.0
eq1         0.0     0.0      0.0    0.0
eq2         0.0     0.0      0.0    0.0
otall3      0.0     NaN      NaN    NaN
otcut3      0.0     NaN      NaN    NaN
otcut3e3    0.0     NaN      NaN    NaN
otcut2      0.0     NaN      NaN    NaN

## Evaluated on close contrast contexts, split by whether the target was on the store's fallback list
faith Z            {'old': {'retrieved': 0.988, 'fallback': 0.924}, 'store': {'retrieved': 0.925, 'fallback': 0.963}, 'close': {'retrieved': 0.876, 'fallback': 0.997}, 'random': {'retrieved': 0.973, 'fallback': 1.046}, 'distant': {'retrieved': 0.973, 'fallback': 1.007}, 'eq1': {'retrieved': 0.867, 'fallback': 0.929}, 'eq2': {'retrieved': 0.958, 'fallback': 1.002}, 'otall3': {'retrieved': 0.963, 'fallback': 0.983}, 'otcut3': {'retrieved': 0.954, 'fallback': 1.002}, 'otcut3e3': {'retrieved': 0.953, 'fallback': 1.003}, 'otcut2': {'retrieved': 0.965, 'fallback': 1.019}}
faith A (sp)       {'old': {'retrieved': 0.979, 'fallback': 0.33}, 'store': {'retrieved': 1.075, 'fallback': 0.056}, 'close': {'retrieved': 0.985, 'fallback': 0.031}, 'random': {'retrieved': 0.934, 'fallback': 0.73}, 'distant': {'retrieved': 0.913, 'fallback': 0.59}, 'eq1': {'retrieved': 0.916, 'fallback': 0.795}, 'eq2': {'retrieved': 0.957, 'fallback': 1.009}, 'otall3': {'retrieved': 0.971, 'fallback': 0.897}, 'otcut3': {'retrieved': 0.923, 'fallback': 0.998}, 'otcut3e3': {'retrieved': 0.943, 'fallback': 0.637}, 'otcut2': {'retrieved': 0.902, 'fallback': 0.85}}
faith C (sp)       {'old': {'retrieved': 0.963, 'fallback': 0.509}, 'store': {'retrieved': 1.028, 'fallback': 0.367}, 'close': {'retrieved': 0.919, 'fallback': 0.425}, 'random': {'retrieved': 0.84, 'fallback': 0.657}, 'distant': {'retrieved': 0.894, 'fallback': 0.622}, 'eq1': {'retrieved': 0.911, 'fallback': 0.861}, 'eq2': {'retrieved': 0.908, 'fallback': 0.752}, 'otall3': {'retrieved': 0.897, 'fallback': 0.892}, 'otcut3': {'retrieved': 0.853, 'fallback': 0.706}, 'otcut3e3': {'retrieved': 0.884, 'fallback': 0.646}, 'otcut2': {'retrieved': 0.797, 'fallback': 0.828}}
faith C (dense)    {'old': {'retrieved': 0.738, 'fallback': 0.689}, 'store': {'retrieved': 0.779, 'fallback': 0.642}, 'close': {'retrieved': 0.873, 'fallback': 0.902}, 'random': {'retrieved': 0.843, 'fallback': 0.74}, 'distant': {'retrieved': 0.822, 'fallback': 0.761}, 'eq1': {'retrieved': 0.959, 'fallback': 0.964}, 'eq2': {'retrieved': 0.94, 'fallback': 0.984}, 'otall3': {'retrieved': 0.853, 'fallback': 0.853}, 'otcut3': {'retrieved': 0.884, 'fallback': 0.971}, 'otcut3e3': {'retrieved': 0.868, 'fallback': 0.91}, 'otcut2': {'retrieved': 0.825, 'fallback': 0.923}}
necessity          {'old': {'retrieved': 1.0, 'fallback': 1.0}, 'store': {'retrieved': 1.0, 'fallback': 1.0}, 'close': {'retrieved': 1.0, 'fallback': 1.0}, 'random': {'retrieved': 1.0, 'fallback': 1.0}, 'distant': {'retrieved': 1.0, 'fallback': 1.0}, 'eq1': {'retrieved': 1.0, 'fallback': 1.0}, 'eq2': {'retrieved': 1.0, 'fallback': 1.0}, 'otall3': {'retrieved': 1.0, 'fallback': 1.0}, 'otcut3': {'retrieved': 1.0, 'fallback': 1.0}, 'otcut3e3': {'retrieved': 1.0, 'fallback': 1.0}, 'otcut2': {'retrieved': 1.0, 'fallback': 1.0}}
suff. to induce    {'old': {'retrieved': 0.865, 'fallback': 0.643}, 'store': {'retrieved': 0.892, 'fallback': 0.733}, 'close': {'retrieved': 0.859, 'fallback': 0.645}, 'random': {'retrieved': 0.857, 'fallback': 0.677}, 'distant': {'retrieved': 0.769, 'fallback': 0.615}, 'eq1': {'retrieved': 0.815, 'fallback': 0.626}, 'eq2': {'retrieved': 0.781, 'fallback': 0.643}, 'otall3': {'retrieved': 0.878, 'fallback': 0.554}, 'otcut3': {'retrieved': 0.874, 'fallback': 0.662}, 'otcut3e3': {'retrieved': 0.899, 'fallback': 0.691}, 'otcut2': {'retrieved': 0.91, 'fallback': 0.729}}
clamped A (sp)     {'old': {'retrieved': 0.538, 'fallback': 0.331}, 'store': {'retrieved': 0.656, 'fallback': 0.1}, 'close': {'retrieved': 0.561, 'fallback': 0.187}, 'random': {'retrieved': 0.513, 'fallback': 0.274}, 'distant': {'retrieved': 0.511, 'fallback': 0.111}, 'eq1': {'retrieved': 0.623, 'fallback': 0.339}, 'eq2': {'retrieved': 0.452, 'fallback': 0.366}, 'otall3': {'retrieved': 0.69, 'fallback': 0.544}, 'otcut3': {'retrieved': 0.607, 'fallback': 0.54}, 'otcut3e3': {'retrieved': 0.733, 'fallback': 0.65}, 'otcut2': {'retrieved': 0.743, 'fallback': 0.548}}
random baseline Z  {'old': {'retrieved': 0.0, 'fallback': 0.0}, 'store': {'retrieved': 0.0, 'fallback': 0.0}, 'close': {'retrieved': 0.0, 'fallback': 0.0}, 'random': {'retrieved': 0.0, 'fallback': 0.0}, 'distant': {'retrieved': 0.0, 'fallback': 0.0}, 'eq1': {'retrieved': 0.0, 'fallback': 0.0}, 'eq2': {'retrieved': 0.0, 'fallback': 0.0}, 'otall3': {'retrieved': 0.0, 'fallback': 0.0}, 'otcut3': {'retrieved': 0.0, 'fallback': 0.0}, 'otcut3e3': {'retrieved': 0.0, 'fallback': 0.0}, 'otcut2': {'retrieved': 0.0, 'fallback': 0.0}}

## Paired change vs the old circuits (evaluated on close; per target new - old, median and share improved)
store    n=16 | faith Z +0.000 (47% up) | faith A (sp) +0.069 (62% up) | faith C (sp) +0.005 (56% up) | faith C (dense) +0.013 (50% up) | necessity +0.000 (0% up) | suff. to induce +0.002 (50% up) | clamped A (sp) -0.001 (31% up) | random baseline Z +0.000 (13% up)
close    n=16 | faith Z +0.004 (53% up) | faith A (sp) +0.023 (56% up) | faith C (sp) +0.004 (50% up) | faith C (dense) +0.066 (69% up) | necessity +0.000 (0% up) | suff. to induce -0.001 (31% up) | clamped A (sp) +0.009 (50% up) | random baseline Z +0.000 (13% up)
random   n=16 | faith Z +0.000 (47% up) | faith A (sp) +0.000 (44% up) | faith C (sp) +0.000 (38% up) | faith C (dense) +0.003 (50% up) | necessity +0.000 (7% up) | suff. to induce +0.000 (44% up) | clamped A (sp) +0.049 (56% up) | random baseline Z +0.000 (7% up)
distant  n=16 | faith Z -0.007 (40% up) | faith A (sp) +0.000 (44% up) | faith C (sp) +0.000 (44% up) | faith C (dense) +0.045 (50% up) | necessity +0.000 (13% up) | suff. to induce +0.004 (50% up) | clamped A (sp) +0.000 (38% up) | random baseline Z +0.000 (7% up)
eq1      n=16 | faith Z -0.003 (40% up) | faith A (sp) -0.024 (31% up) | faith C (sp) +0.044 (62% up) | faith C (dense) +0.141 (81% up) | necessity +0.000 (7% up) | suff. to induce -0.026 (12% up) | clamped A (sp) +0.000 (38% up) | random baseline Z +0.000 (7% up)
eq2      n=16 | faith Z +0.000 (47% up) | faith A (sp) +0.029 (50% up) | faith C (sp) +0.047 (62% up) | faith C (dense) +0.189 (81% up) | necessity +0.000 (7% up) | suff. to induce +0.000 (44% up) | clamped A (sp) -0.014 (31% up) | random baseline Z +0.000 (20% up)
otall3   n=16 | faith Z -0.002 (47% up) | faith A (sp) +0.127 (75% up) | faith C (sp) +0.117 (69% up) | faith C (dense) +0.095 (75% up) | necessity +0.000 (7% up) | suff. to induce +0.016 (56% up) | clamped A (sp) +0.070 (56% up) | random baseline Z +0.000 (7% up)
otcut3   n=16 | faith Z +0.059 (67% up) | faith A (sp) +0.037 (62% up) | faith C (sp) +0.000 (44% up) | faith C (dense) +0.197 (56% up) | necessity +0.000 (7% up) | suff. to induce +0.000 (50% up) | clamped A (sp) +0.043 (69% up) | random baseline Z +0.000 (0% up)
otcut3e3 n=16 | faith Z -0.009 (47% up) | faith A (sp) +0.068 (69% up) | faith C (sp) +0.058 (69% up) | faith C (dense) +0.148 (75% up) | necessity +0.000 (0% up) | suff. to induce +0.017 (62% up) | clamped A (sp) +0.059 (69% up) | random baseline Z +0.000 (7% up)
otcut2   n=16 | faith Z +0.011 (67% up) | faith A (sp) +0.075 (69% up) | faith C (sp) +0.008 (50% up) | faith C (dense) +0.127 (56% up) | necessity +0.000 (13% up) | suff. to induce +0.026 (56% up) | clamped A (sp) +0.009 (50% up) | random baseline Z +0.000 (7% up)

## Node overlap between training sources (Jaccard, median over targets)
old      old 1.00  store 0.66  close 0.59  random 0.58  distant 0.56  eq1 0.49  eq2 0.56  otall3 0.41  otcut3 0.57  otcut3e3 0.51  otcut2 0.46
store    old 0.66  store 1.00  close 0.58  random 0.58  distant 0.56  eq1 0.50  eq2 0.59  otall3 0.43  otcut3 0.55  otcut3e3 0.52  otcut2 0.48
close    old 0.59  store 0.58  close 1.00  random 0.60  distant 0.58  eq1 0.54  eq2 0.61  otall3 0.41  otcut3 0.59  otcut3e3 0.55  otcut2 0.46
random   old 0.58  store 0.58  close 0.60  random 1.00  distant 0.61  eq1 0.51  eq2 0.58  otall3 0.40  otcut3 0.56  otcut3e3 0.52  otcut2 0.45
distant  old 0.56  store 0.56  close 0.58  random 0.61  distant 1.00  eq1 0.51  eq2 0.56  otall3 0.37  otcut3 0.55  otcut3e3 0.48  otcut2 0.42
eq1      old 0.49  store 0.50  close 0.54  random 0.51  distant 0.51  eq1 1.00  eq2 0.58  otall3 0.43  otcut3 0.57  otcut3e3 0.53  otcut2 0.46
eq2      old 0.56  store 0.59  close 0.61  random 0.58  distant 0.56  eq1 0.58  eq2 1.00  otall3 0.40  otcut3 0.57  otcut3e3 0.53  otcut2 0.42
otall3   old 0.41  store 0.43  close 0.41  random 0.40  distant 0.37  eq1 0.43  eq2 0.40  otall3 1.00  otcut3 0.46  otcut3e3 0.47  otcut2 0.50
otcut3   old 0.57  store 0.55  close 0.59  random 0.56  distant 0.55  eq1 0.57  eq2 0.57  otall3 0.46  otcut3 1.00  otcut3e3 0.63  otcut2 0.53
otcut3e3 old 0.51  store 0.52  close 0.55  random 0.52  distant 0.48  eq1 0.53  eq2 0.53  otall3 0.47  otcut3 0.63  otcut3e3 1.00  otcut2 0.62
otcut2   old 0.46  store 0.48  close 0.46  random 0.45  distant 0.42  eq1 0.46  eq2 0.42  otall3 0.50  otcut3 0.53  otcut3e3 0.62  otcut2 1.00