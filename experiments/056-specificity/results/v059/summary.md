# 056 specificity diagnostic (180 rows)


## Z ablation: medians over targets
     target_faith_pre  sibling_faith_median  control_faith_median  specificity_gap  in_topk_circuit  rank_circuit  jaccard_circuit  jaccard_empty  n_new_circuit  switched_on_circuit  n_siblings
arm                                                                                                                                                                                              
B               0.955                 0.281                 0.253            0.391              1.0           5.0            0.101          0.014        104.562                631.0        59.0
B2              0.923                 0.413                 0.207            0.295              1.0           4.0            0.092          0.014        106.375                771.0        59.0
Bw              0.890                 0.400                 0.231            0.343              1.0           5.0            0.098          0.014        105.250                818.0        59.0
B2w             0.872                 0.204                 0.289            0.512              1.0           4.0            0.100          0.014        104.688               1514.0        59.0

## A ablation: medians over targets
     target_faith_pre  sibling_faith_median  control_faith_median  specificity_gap  in_topk_circuit  rank_circuit  jaccard_circuit  jaccard_empty  n_new_circuit  switched_on_circuit  n_siblings
arm                                                                                                                                                                                              
B               0.917                 0.400                 0.177            0.400              1.0           5.0            0.118          0.134        101.062                259.0        59.0
B2              0.859                 0.425                 0.186            0.434              1.0           5.0            0.106          0.134        103.625                371.0        59.0
Bw              0.937                 0.412                 0.338            0.440              1.0           6.0            0.117          0.134        101.312                474.0        59.0
B2w             0.858                 0.350                 0.194            0.388              1.0           7.0            0.125          0.134         99.688                534.0        59.0

## C ablation: medians over targets
     target_faith_pre  sibling_faith_median  control_faith_median  specificity_gap  in_topk_circuit  rank_circuit  jaccard_circuit  jaccard_empty  n_new_circuit  switched_on_circuit  n_siblings
arm                                                                                                                                                                                              
B               0.903                 0.356                 0.192            0.541              1.0           5.0            0.101          0.099        104.625                235.0        59.0
B2              0.857                 0.393                 0.105            0.459              1.0           5.0            0.109          0.099        102.938                342.0        59.0
Bw              0.881                 0.381                 0.322            0.426              1.0          10.0            0.102          0.099        104.250                267.0        59.0
B2w             0.897                 0.356                 0.174            0.556              1.0          13.0            0.097          0.099        105.438                453.0        59.0

## Concept-amplifier flags (sibling faithfulness >= 0.8 x target faithfulness, target faithfulness >= 0.5)
arm          seed pi  target_faith_pre  sibling_faith_median  control_faith_median  n_siblings  in_topk_circuit  switched_on_circuit
  B   3.mlp.23075  C             0.609                 0.634                 0.000           5            0.938                 36.0
  B  5.attn.34661  A             0.758                 1.236                 1.066          63            0.500               6853.0
  B  5.attn.34661  C             0.735                 1.319                 0.806          63            0.625               5071.0
  B   7.mlp.28744  Z             0.959                 1.540                -0.042           3            0.750                535.0
  B 10.attn.36603  Z             0.944                 0.935                 0.277          17            0.938                143.0
  B  11.mlp.30743  A             1.069                 0.889                 0.672          46            1.000               1037.0
 B2   3.mlp.23075  Z             1.079                 2.817                -0.048           5            1.000                 48.0
 B2  5.attn.34661  C             0.683                 0.683                 0.914          63            0.625               4946.0
 B2   7.mlp.28744  Z             0.843                 1.232                -0.141           3            0.938                139.0
 B2   7.mlp.28744  A             0.774                 0.754                 0.000           3            1.000                 54.0
 B2   7.mlp.28744  C             0.787                 2.375                 0.000           3            0.938                 52.0
 B2  9.attn.21759  Z             0.980                 0.798                 0.681          68            1.000                600.0
 B2 10.attn.36603  Z             0.834                 1.108                 0.000          17            0.000                962.0
 B2  11.mlp.30743  A             1.008                 0.933                 1.134          46            1.000               1877.0
 Bw   3.mlp.23075  Z             0.994                 1.035                 0.000           5            1.000                395.0
 Bw   7.mlp.28744  A             0.513                 0.742                 0.338           3            0.188                203.0
 Bw   7.mlp.28744  C             0.665                 0.688                 0.322           3            0.375                258.0
 Bw 10.attn.36603  Z             0.837                 0.891                 0.231          17            0.750                238.0
 Bw  11.mlp.30743  A             0.962                 0.849                 0.474          46            1.000                547.0
B2w  5.attn.34661  A             0.700                 1.054                 0.972          63            0.500               4735.0
B2w  5.attn.34661  C             0.638                 1.100                 0.898          63            0.500               4184.0
B2w   7.mlp.28744  C             0.897                 3.804                 0.003           3            0.625                453.0
B2w    0.mlp.5196  Z             0.558                 0.450                 0.374          87            1.000                 43.0
B2w 10.attn.36603  Z             0.869                 1.259                 0.079          17            0.750                211.0
B2w  11.mlp.30743  Z             0.889                 0.730                 0.292          46            1.000                774.0
B2w  11.mlp.30743  A             1.030                 0.906                 0.814          46            1.000               1060.0

## Per target (C ablation, first arm)
         seed  n_nodes  target_faith_pre  sibling_faith_median  control_faith_median  rank_clean  rank_circuit  in_topk_clean  in_topk_circuit  jaccard_circuit  jaccard_empty  switched_on_circuit
  0.mlp.16000       38             0.904                 0.304                 0.386        16.0          12.0            1.0            1.000            0.271          0.192                 32.0
 2.attn.33479      309             0.896                 0.529                 0.000        32.0          39.0            1.0            1.000            0.177          0.167                173.0
  3.mlp.23075      469             0.609                 0.634                 0.000        13.0           4.0            1.0            0.938            0.029          0.024                 36.0
 5.attn.34661      474             0.735                 1.319                 0.806        11.0         104.0            1.0            0.625            0.079          0.138               5071.0
  7.mlp.28744      803             0.494                 1.315                 0.021        25.0         357.0            1.0            0.188            0.012          0.010                 95.0
 9.attn.21759      739             0.694                 0.092                 0.228         2.0           6.0            1.0            1.000            0.153          0.117                431.0
11.resid.8702      679             0.991                 0.000                 0.134        10.0           5.0            1.0            1.000            0.024          0.010                235.0
   0.mlp.5196       12             1.028                 0.576                 0.560         1.0           1.0            1.0            1.000            0.305          0.210                 51.0
1.resid.12137      158             0.996                 0.042                 0.000        17.0           2.0            1.0            1.000            0.092          0.056                131.0
2.resid.18148      288             1.009                 0.356                 0.192         3.0           1.0            1.0            1.000            0.198          0.099                281.0
  5.mlp.11680      606             0.928                 0.273                 0.531         5.0           5.0            1.0            0.938            0.058          0.006                338.0
6.resid.18234      668             1.197                 0.336                 0.519         5.0           2.0            1.0            1.000            0.076          0.040                687.0
 8.attn.37097      568             0.871                 0.330                 0.129        12.0          23.0            1.0            1.000            0.223          0.191                375.0
10.attn.36603      474             0.402                 0.437                 0.065        65.0         688.0            1.0            0.000            0.101          0.110                148.0
 11.mlp.30743      770             0.903                 0.465                 0.363         1.0           1.0            1.0            1.000            0.113          0.012                799.0