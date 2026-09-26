# 061 lambda sweep: WCM vs unweighted circuit masking (held-out strongest, activation read)

gamma = 0.25, protocol-v1 contexts (059 arm B training set), 16 pilot targets (15 with a mid-band pool).

                  n_targets    nodes  free0  freeM  freeN  worst_dev  in_band  necessity  induce
weighted lam                                                                                    
False    0.00001         15  14606.0  0.921  0.967  1.010      0.134       11        1.0   0.920
         0.00003         15   6741.0  0.903  0.963  0.982      0.156        8        1.0   0.861
         0.00010         15   3737.0  0.868  0.925  0.853      0.251        7        1.0   0.706
         0.00030         15   2174.0  0.832  0.819  0.806      0.407        5        1.0   0.704
         0.00100         15    820.0  0.697  0.354  0.403      0.691        3        1.0   0.542
True     0.00025         15   1237.0  0.956  1.001  0.925      0.138        9        1.0   0.830
         0.00050         15    748.0  0.977  0.953  0.947      0.197        9        1.0   0.864
         0.00100         15    474.0  0.952  0.918  0.903      0.142        9        1.0   0.855
         0.00200         15    321.0  0.950  0.819  0.811      0.363        5        1.0   0.907
         0.00400         15    239.0  0.814  0.717  0.425      0.602        4        1.0   0.845

## Matched size (unweighted interpolated in log node count)

- WCM lam 0.004: 239 nodes, free0/M/N 0.81/0.72/0.42 | unweighted at the same size: 0.70/0.35/0.40 (extrapolated: outside the unweighted range)
- WCM lam 0.002: 321 nodes, free0/M/N 0.95/0.82/0.81 | unweighted at the same size: 0.70/0.35/0.40 (extrapolated: outside the unweighted range)
- WCM lam 0.001: 474 nodes, free0/M/N 0.95/0.92/0.90 | unweighted at the same size: 0.70/0.35/0.40 (extrapolated: outside the unweighted range)
- WCM lam 0.0005: 748 nodes, free0/M/N 0.98/0.95/0.95 | unweighted at the same size: 0.70/0.35/0.40 (extrapolated: outside the unweighted range)
- WCM lam 0.00025: 1237 nodes, free0/M/N 0.96/1.00/0.93 | unweighted at the same size: 0.75/0.55/0.57